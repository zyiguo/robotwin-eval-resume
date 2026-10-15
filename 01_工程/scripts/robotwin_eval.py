"""Persistent, audited 50 x 10 RoboTwin evaluation. No GPU imports for status/report."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import uuid

from full_training_common import atomic_json, exclusive_lock, read_json
from robotwin_eval_guard import stop_group
from robotwin_eval_worker import instrument

HERE = Path(__file__).resolve().parent
DEFAULT_PARENT = "/mnt/workspace/runtime/formal_muon_ga50_150_20261015_085411"
PRIORITY = ["adjust_bottle", "click_bell", "handover_block", "lift_pot", "place_empty_cup"]
EXPECTED_TASKS = read_json(HERE.parent / "configs/robotwin_eval_tasks.json")["tasks"]
SETTINGS = {"task_config": "demo_clean", "instruction_type": "seen", "episodes_per_task": 10,
            "seed_start": 100000, "use_fp32": True, "use_bf16": False, "use_length": 50,
            "use_compile": False, "expert_check": True, "accept_expert_info_on_failure": True,
            "eval_batch": False, "video": False, "planner": "mplib", "gpu": 0}
PACKAGE_NAMES = ("torch", "torchvision", "transformers", "numpy", "sapien", "mplib", "open3d",
                 "flash-attn", "triton", "pytorch-triton-rocm", "aiter", "websockets", "PyYAML",
                 "safetensors", "tokenizers", "huggingface-hub", "xpolicylab")
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".sh", ".txt", ".xml", ".urdf", ".srdf"}
IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".cache", "eval_result", "eval_results",
                "data", "logs", "wandb", "outputs", "checkpoints", "node_modules"}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed during identity check: {path}")
    return h.hexdigest()


def fsync_directory(path):
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def publish(path, value):
    """Atomic, durable, no-overwrite commit. Interrupted pending files are evidence only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # link(2) publishes only if the destination does not exist, unlike replace().
    os.link(pending, path)
    fsync_directory(path.parent)
    pending.unlink()
    fsync_directory(path.parent)


def safe_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe relative evidence path: {relative}")
    path = root / relative
    if root not in path.resolve().parents:
        raise ValueError(f"Evidence path escapes batch: {path}")
    return path


def checkpoint_files(root):
    """Inspect safetensors framing and index without loading tensors or importing torch."""
    root = Path(root)
    required = {"config.json", "lingbotvla_cli.yaml", "norm_stats.json", "export_metadata.json"}
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    for name in ("config.json", "norm_stats.json"):
        if not isinstance(read_json(root / name), dict) or not read_json(root / name):
            raise ValueError(f"Empty/invalid {name}")
    metadata = read_json(root / "export_metadata.json")
    if metadata.get("save_kind") != "model_weights_only" or metadata.get("optimizer_state_saved") is not False:
        raise ValueError("Expected a completed self-trained weights-only export")
    if metadata.get("kind") == "formal_weights":
        manifest = read_json(root / "run_manifest.json")
        if metadata.get("run_id") != manifest.get("run_id"):
            raise ValueError("Export provenance mismatch")
    index_file = root / "model.safetensors.index.json"
    index = read_json(index_file)["weight_map"] if index_file.exists() else None
    names = set(index.values()) if index else {"model.safetensors"}
    if set(p.name for p in root.glob("*.safetensors")) != names:
        raise ValueError("Missing or unexpected weight shards")
    tensors, tensor_bytes = {}, 0
    for name in sorted(names):
        if Path(name).name != name:
            raise ValueError("Weight index contains a nonlocal shard")
        path = root / name
        with path.open("rb") as stream:
            framing = stream.read(8)
            if len(framing) != 8:
                raise ValueError(f"Truncated safetensors: {path}")
            length = struct.unpack("<Q", framing)[0]
            if not 0 < length <= min(100_000_000, path.stat().st_size - 8):
                raise ValueError(f"Invalid safetensors header: {path}")
            header = json.loads(stream.read(length))
        ranges = []
        for key, row in header.items():
            if key == "__metadata__":
                continue
            if key in tensors:
                raise ValueError(f"Duplicate tensor {key}")
            start, end = row["data_offsets"]
            if not (type(start) is int and type(end) is int and 0 <= start <= end):
                raise ValueError("Invalid tensor offsets")
            ranges.append((start, end))
            tensors[key] = name
        last = 0
        for start, end in sorted(ranges):
            if start != last:
                raise ValueError("Safetensors has gaps/overlap")
            last = end
        if not ranges or 8 + length + last != path.stat().st_size:
            raise ValueError(f"Truncated or trailing safetensors data: {path}")
        tensor_bytes += last
    if index is not None and tensors != index:
        raise ValueError("Weight index does not match shard contents")
    if metadata.get("total_tensor_bytes") != tensor_bytes:
        raise ValueError("Export byte count does not match weights")
    return sorted(p for p in root.iterdir() if p.is_file() and not p.name.endswith(".pending"))


def snapshot_model(source, destination):
    files = checkpoint_files(source)
    destination.mkdir()
    for path in files:
        target = destination / path.name
        # Completed training exports are immutable. A hardlink protects against
        # rolling checkpoint deletion without duplicating 12+ GB on the same FS.
        if path.suffix == ".safetensors":
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)
        else:
            shutil.copy2(path, target)
    checkpoint_files(destination)


def tasks_from_launcher(path):
    match = re.search(r"^task_list_all=\(([^\n]+)\)", path.read_text(encoding="utf-8"), re.M)
    if not match:
        raise ValueError("Cannot read pinned launcher's task_list_all")
    import shlex
    tasks = shlex.split(match.group(1))
    if len(tasks) != 50 or set(tasks) != set(EXPECTED_TASKS):
        raise ValueError("Expected 50 distinct RoboTwin tasks")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]+", t) for t in tasks):
        raise ValueError("Invalid task identifier")
    return PRIORITY + [t for t in tasks if t not in PRIORITY]


def runtime_paths(args, batch, manifest=None):
    defaults = manifest["paths"] if manifest else {
        "lingbot_root": "/mnt/workspace/src/lingbot-vla-v2",
        "robotwin_root": "/mnt/workspace/src/RoboTwin",
        "qwen_path": "/mnt/workspace/models/Qwen3-VL-4B-Instruct-config-tokenizer",
        "python": sys.executable,
        "vulkan_icd": "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"}
    paths = {}
    for key, value in defaults.items():
        path = Path(getattr(args, key, None) or value)
        # Resolving a venv's bin/python symlink selects its base interpreter and
        # silently loses the venv. Preserve that invocation path across resumes.
        paths[key] = str(path.absolute() if key == "python" else path.resolve())
    paths["model_path"] = str((batch / "model").resolve())
    return paths


def package_identity(python):
    code = ("import importlib.metadata as m,json,sys,platform; "
            f"names={PACKAGE_NAMES!r}; "
            "d={}; "
            "exec('for n in names:\\n try: d[n]=m.version(n)\\n except m.PackageNotFoundError: d[n]=None'); "
            "print(json.dumps({'python':list(sys.version_info[:2]),'system':platform.system(),"
            "'machine':platform.machine(),'packages':d}))")
    result = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=60, check=True)
    return json.loads(result.stdout)


def normalized_hash(path, replacements):
    if path.suffix.lower() in TEXT_SUFFIXES:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file_hash(path)
        for original, token in sorted(replacements.items(), key=lambda kv: -len(kv[0])):
            text = text.replace(original, token)
        return hashlib.sha256(text.encode()).hexdigest()
    return file_hash(path)


def tree_inventory(root, replacements, *, source_only=False):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = {}
    # Follow source/asset symlinks, but reject cycles. Linked dependencies must
    # be migrated too; hashing their targets makes a broken link visible.
    def walk(directory, relative, ancestors):
        resolved = directory.resolve()
        if resolved in ancestors:
            raise ValueError(f"Cyclic dependency symlink: {directory}")
        for path in sorted(directory.iterdir()):
            rel = relative / path.name
            if path.is_dir():
                # Root-level training data is not a simulator dependency;
                # lingbotvla/data is model input/normalization *source code*.
                ignored = path.name in (IGNORED_DIRS - {"data"}) or (relative == Path() and path.name == "data")
                if not ignored and not path.name.endswith(".egg-info"):
                    walk(path, rel, ancestors | {resolved})
            elif path.is_file():
                if not source_only or path.suffix.lower() in TEXT_SUFFIXES:
                    result[rel.as_posix()] = normalized_hash(path, replacements)
            elif path.is_symlink():
                raise FileNotFoundError(f"Broken dependency link: {path}")
    walk(root, Path(), set())
    if not result:
        raise ValueError(f"Empty identity tree: {root}")
    return result


def identity(paths, original_paths=None):
    replacements = {}
    for mapping in (original_paths or {}, paths):
        for key in ("lingbot_root", "robotwin_root", "qwen_path", "model_path"):
            if key in mapping:
                replacements[mapping[key]] = "${" + key + "}"
    return {
        "model": {p.name: file_hash(p) for p in checkpoint_files(Path(paths["model_path"]))},
        "qwen": tree_inventory(paths["qwen_path"], replacements),
        "lingbot": tree_inventory(paths["lingbot_root"], replacements, source_only=True),
        # Includes configs, instructions, XPolicyLab, assets and linked assets.
        "robotwin": tree_inventory(paths["robotwin_root"], replacements),
        "runner": {name: file_hash(HERE / name) for name in (
            "robotwin_eval.py", "robotwin_eval_guard.py", "robotwin_eval_worker.py",
            "robotwin_eval_server.py", "fix_moe_eval_dtype.py", "full_training_common.py")},
        "environment": package_identity(paths["python"]),
        "vulkan_icd": file_hash(Path(paths["vulkan_icd"]))}


def source_preflight(paths):
    lingbot, robotwin = Path(paths["lingbot_root"]), Path(paths["robotwin_root"])
    server = (lingbot / "deploy/lingbot_vla_v2_policy.py").read_text(encoding="utf-8")
    evaluator = robotwin / "scripts/eval_policy_xpolicylab.py"
    source = evaluator.read_text(encoding="utf-8")
    for token in ("--seed_start", "lingbot_vla_v2", "--accept_expert_info_on_failure"):
        if token not in source:
            raise ValueError(f"Missing AMD evaluator support: {token}")
    if "LINGBOTVLA_TRAINING_CONFIG" not in server:
        raise ValueError("Missing inference training-config override patch")
    planner = (robotwin / "envs/robot/planner.py").read_text(encoding="utf-8")
    robot = (robotwin / "envs/robot/robot.py").read_text(encoding="utf-8")
    if "ROBOTWIN_DISABLE_CUROBO" not in planner or "ROBOTWIN_EE_PLANNER" not in robot:
        raise ValueError("Missing ROCm/MPLib simulator patches")
    instrument(source, str(evaluator))
    for directory in ("objects", "embodiments", "background_texture"):
        path = robotwin / "assets" / directory
        if not path.is_dir() or not any(path.iterdir()):
            raise FileNotFoundError(f"Missing simulation assets: {path}")
    for path in (robotwin / "XPolicyLab", Path(paths["qwen_path"]), Path(paths["vulkan_icd"])):
        if not path.exists():
            raise FileNotFoundError(path)


def prepare(args):
    import yaml
    batch = args.batch_dir.resolve()
    if batch.exists():
        raise FileExistsError(f"Preserving existing batch: {batch}")
    paths = runtime_paths(args, batch)
    source_preflight(paths)
    parent = args.parent_run.resolve()
    best = read_json(parent / "best.json")
    source = Path(args.model_source or best["path"])
    if not source.is_absolute():
        source = parent / source
    source = source.resolve()
    checkpoint_files(source)
    metadata = read_json(source / "export_metadata.json")
    if type(best.get("global_step")) is not int or best["global_step"] < 0:
        raise ValueError("Invalid best.json step")
    if best["global_step"] > 0 and (metadata.get("global_step") != best["global_step"]
                                     or metadata.get("run_id") != best.get("run_id")):
        raise ValueError("Selected export does not match the frozen best pointer")
    selected_step = re.search(r"(?:^|/)global_step_(\d+)/hf_ckpt/?$", best["path"])
    if selected_step and metadata.get("global_step") != int(selected_step.group(1)):
        raise ValueError("Relocated export does not match the checkpoint named by best.json")
    config = yaml.safe_load((source / "lingbotvla_cli.yaml").read_text(encoding="utf-8"))
    if not all(isinstance(config.get(key), dict) for key in ("model", "data", "train")):
        raise ValueError("Training config lacks model/data/train")
    tasks = tasks_from_launcher(Path(paths["lingbot_root"]) / "experiment/robotwin/start_robotwin_infer_and_eval.sh")
    batch.mkdir(parents=True)
    # Incomplete preparation has no manifest, so it can never be run by accident.
    snapshot_model(source, batch / "model")
    atomic_json(batch / "selection.json", {"parent_run": str(parent), "best": best,
                                            "copied_source": str(source), "frozen_at": now()})
    frozen = identity(paths)
    spec = {"schema": 1, "batch_id": uuid.uuid4().hex, "settings": SETTINGS, "tasks": tasks,
            "identity": frozen}
    manifest = {**spec, "fingerprint": digest(spec), "created_at": now(), "paths": paths}
    publish(batch / "manifest.json", manifest)
    print(f"PREPARED: {batch}\nFINGERPRINT: {manifest['fingerprint']}\nEPISODES: 500", flush=True)
    return manifest


def load_manifest(batch):
    manifest = read_json(batch / "manifest.json")
    spec = {key: manifest[key] for key in ("schema", "batch_id", "settings", "tasks", "identity")}
    if manifest["schema"] != 1 or manifest["settings"] != SETTINGS or digest(spec) != manifest["fingerprint"]:
        raise ValueError("Manifest is corrupt or its evaluation settings changed")
    tasks = manifest["tasks"]
    if (len(tasks) != 50 or set(tasks) != set(EXPECTED_TASKS) or tasks[:5] != PRIORITY
            or any(not re.fullmatch(r"[a-z][a-z0-9_]+", t) for t in tasks)):
        raise ValueError("Invalid frozen task list")
    return manifest


def episode_key(task, episode):
    return f"{task}/{episode:02d}"


def validate_attempt(attempt, task, seed):
    guard = read_json(attempt / "guard.json")
    if (guard.get("state") != "finished" or guard.get("reason") != "exited"
            or guard.get("returncode") != 0):
        raise ValueError("Worker did not exit normally")
    audit = read_json(attempt / "audit.json")
    episodes = audit.get("episodes", [])
    if (audit.get("valid") is not True or audit.get("main_returned") is not True
            or audit.get("task") != task or audit.get("seed") != seed or audit.get("caught_errors")
            or len(episodes) != 1 or episodes[0].get("seed") != seed
            or type(episodes[0].get("success")) is not bool
            or episodes[0].get("check_success") is not episodes[0]["success"]):
        raise ValueError("Audited success/seed/error evidence is invalid")
    text = (attempt / "worker.log").read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    finals = re.findall(r"Final success rate:\s*(\d+)/(\d+)\s*=\s*([0-9.]+)%", text)
    if len(finals) != 1:
        raise ValueError("Expected exactly one final episode statistic")
    numerator, denominator, pct = finals[0]
    success = episodes[0]["success"]
    if (int(denominator) != 1 or int(numerator) != int(success)
            or float(pct) != (100.0 if success else 0.0)):
        raise ValueError("Final statistic contradicts audited episode")
    if "Policy rollout error:" in text or "Traceback (most recent call last)" in text:
        raise ValueError("Caught rollout error must not be counted as model failure")
    return success


def evidence_inventory(attempt):
    required = ("request.json", "guard.json", "audit.json", "worker.log")
    result = {name: file_hash(attempt / name) for name in required}
    if not (attempt / "raw/_result.txt").is_file():
        raise FileNotFoundError(f"Missing upstream result artifact: {attempt / 'raw/_result.txt'}")
    raw = attempt / "raw"
    if raw.exists():
        for path in sorted(raw.rglob("*")):
            if path.is_file():
                result[path.relative_to(attempt).as_posix()] = file_hash(path)
    return result


def completed_results(batch, manifest):
    result = {}
    root = batch / "results"
    if not root.exists():
        return result
    expected = {episode_key(t, e) for t in manifest["tasks"] for e in range(10)}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Linked result is not allowed: {path}")
        if path.is_dir() or path.name.endswith(".pending"):
            continue
        if path.suffix != ".json":
            raise ValueError(f"Unexpected result file: {path}")
        try:
            row = read_json(path)
            if (not isinstance(row, dict) or type(row.get("episode")) is not int
                    or not 0 <= row["episode"] < 10 or not isinstance(row.get("task"), str)
                    or type(row.get("elapsed_s")) not in (int, float)
                    or not math.isfinite(row["elapsed_s"]) or row["elapsed_s"] < 0):
                raise ValueError("Invalid result fields")
        except (ValueError, TypeError) as error:
            raise ValueError(f"Corrupt result {path}: {error}") from error
        key = episode_key(row["task"], row["episode"])
        expected_path = root / (key + ".json")
        if (key not in expected or key in result or path != expected_path
                or row.get("fingerprint") != manifest["fingerprint"]
                or row.get("seed") != 100000 + row["episode"]
                or row.get("outcome") not in ("success", "failure")):
            raise ValueError(f"Invalid, duplicate, or foreign result: {path}")
        attempt = safe_path(batch, row["attempt"])
        request = read_json(attempt / "request.json")
        if any(request.get(k) != row.get(k) for k in ("fingerprint", "task", "episode", "seed")):
            raise ValueError(f"Attempt identity mismatch: {path}")
        if row.get("evidence") != evidence_inventory(attempt):
            raise ValueError(f"Missing or changed evidence: {path}")
        success = validate_attempt(attempt, row["task"], row["seed"])
        if success != (row["outcome"] == "success"):
            raise ValueError(f"Corrupt success classification: {path}")
        result[key] = row
    return result


def report(batch, *, save=False):
    manifest = load_manifest(batch)
    rows = completed_results(batch, manifest)
    per_task = {}
    pending = []
    for task in manifest["tasks"]:
        selected = [rows[episode_key(task, e)] for e in range(10) if episode_key(task, e) in rows]
        successes = sum(r["outcome"] == "success" for r in selected)
        per_task[task] = {"completed": len(selected), "target": 10, "success": successes,
                          "failure": len(selected) - successes, "complete": len(selected) == 10,
                          "success_rate": successes / len(selected) if selected else None}
        pending += [{"task": task, "episode": e, "seed": 100000 + e} for e in range(10)
                    if episode_key(task, e) not in rows]
    attempts, sessions = [], []
    for path in sorted((batch / "sessions").glob("*/session.json")):
        session = read_json(path)
        # A snapshot taken while running deliberately reports only a lower
        # bound for elapsed time. No claim that a prior running session ended.
        sessions.append(session)
    committed_attempts = {row["attempt"] for row in rows.values()}
    for path in sorted((batch / "sessions").glob("*/attempts/*/request.json")):
        attempt = path.parent
        state_file = attempt / "attempt.json"
        state = read_json(state_file).get("state", "unknown") if state_file.exists() else "interrupted_or_running"
        if state == "interrupted_or_running" and (attempt / "guard.json").is_file():
            guard_state = read_json(attempt / "guard.json")
            if guard_state.get("reason") in ("parent_lost", "interrupted", "timeout"):
                state = "timeout" if guard_state["reason"] == "timeout" else "interrupted"
        if attempt.relative_to(batch).as_posix() in committed_attempts:
            state = "committed"
        attempts.append({"path": str(attempt.relative_to(batch)), "state": state})
    complete = len(rows) == 500
    successes = sum(r["outcome"] == "success" for r in rows.values())
    durations = [r["elapsed_s"] for r in rows.values()]
    summary = {"schema": 1, "batch_id": manifest["batch_id"], "fingerprint": manifest["fingerprint"],
               "generated_at": now(), "complete": complete, "completed": len(rows), "target": 500,
               "success": successes, "failure": len(rows) - successes,
               "tasks_covered": sum(t["completed"] > 0 for t in per_task.values()),
               "tasks_complete": sum(t["complete"] for t in per_task.values()),
               "partial_success_rate": successes / len(rows) if rows else None,
               "final_macro_success_rate": sum(t["success_rate"] for t in per_task.values()) / 50 if complete else None,
               "final_overall_success_rate": successes / 500 if complete else None,
               "session_elapsed_s": sum(s.get("elapsed_s", 0) for s in sessions),
               "remaining_estimate_s": sum(durations) / len(durations) * len(pending) if durations else None,
               "estimate_note": "Estimate excludes future startup/retries; task durations differ.",
               "attempt_states": dict(Counter(a["state"] for a in attempts)), "sessions": sessions,
               "per_task": per_task, "pending": pending}
    if save:
        atomic_json(batch / "report.json", summary)
        lines = ["# RoboTwin 累计评测报告", "", f"- 正常完成：{len(rows)}/500",
                 f"- 成功：{successes}；正常失败：{len(rows)-successes}",
                 f"- 完整任务：{summary['tasks_complete']}/50",
                 f"- 状态：{'完整评测完成' if complete else '部分完成，等待续跑'}", "",
                 "任务 | 完成数 | 成功数 | 正常失败 | 已完成部分成功率", "---|---:|---:|---:|---:"]
        for task, row in per_task.items():
            rate = f"{row['success_rate']:.1%}" if row["success_rate"] is not None else "—"
            lines.append(f"{task} | {row['completed']}/10 | {row['success']} | {row['failure']} | {rate}")
        if complete:
            lines += ["", f"最终宏平均：{summary['final_macro_success_rate']:.2%}",
                      f"最终总体成功率：{summary['final_overall_success_rate']:.2%}"]
        lines += ["", "异常尝试：`" + json.dumps(summary["attempt_states"], ensure_ascii=False) + "`",
                  "", "原始证据与逐 episode 正式结果为准；未完成项不是模型失败。"]
        temporary = batch / ("report." + uuid.uuid4().hex + ".pending")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(batch / "report.md")
    return summary


def runtime_environment(paths, session):
    env = dict(os.environ)
    for key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        env.pop(key, None)
    env.update(HIP_VISIBLE_DEVICES="0", AITER_TRITON_ONLY="1",
               FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE", PYOPENGL_PLATFORM="egl",
               ROBOTWIN_DISABLE_CUROBO="1", ROBOTWIN_EE_PLANNER="mplib",
               SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0", PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1",
               QWEN3VL_PATH=paths["qwen_path"], ROBOTWIN_ROOT=paths["robotwin_root"],
               VK_ICD_FILENAMES=paths["vulkan_icd"],
               LINGBOTVLA_TRAINING_CONFIG=str(session / "inference.yaml"))
    xdg = session / "xdg"
    xdg.mkdir(mode=0o700)
    if os.name == "posix":
        xdg.chmod(0o700)
    env["XDG_RUNTIME_DIR"] = str(xdg)
    prefixes = [paths["lingbot_root"], paths["robotwin_root"], str(HERE)]
    if Path("/opt/flash-attention-source").is_dir():
        prefixes.append("/opt/flash-attention-source")
    if env.get("PYTHONPATH"):
        prefixes.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(prefixes)
    return env


def guarded(command, folder, seconds, cwd, env):
    folder.mkdir(parents=True, exist_ok=True)
    log = (folder / "worker.log").open("xb")
    try:
        process = subprocess.Popen([sys.executable, str(HERE / "robotwin_eval_guard.py"),
                                    "--status", str(folder / "guard.json"), "--seconds", str(seconds),
                                    "--parent", str(os.getpid()), "--", *map(str, command)],
                                   cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    except BaseException:
        log.close()
        raise
    return process, log


def finish_guard(process, stream):
    # Ask the guard to clean up its own child group; do not kill arbitrary PIDs.
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stop_group(process)
    stream.close()


def rendering_preflight(paths, session, env):
    folder = session / "preflight"
    folder.mkdir()
    checks = [
        [paths["python"], "-c", "import torch,numpy,transformers,sapien,mplib,open3d; "
         "print(torch.__version__,torch.version.hip,numpy.__version__,transformers.__version__); "
         "assert torch.cuda.is_available() and torch.version.hip; "
         "assert numpy.__version__=='1.26.4' and transformers.__version__=='4.57.3'"],
        ["vulkaninfo", "--summary"],
        [paths["python"], "-m", "sapien.example.offscreen"],
        [paths["python"], "-c", "from test_render import Sapien_TEST; "
         "Sapien_TEST(); print('ROBOTWIN_RENDER_READY')"]]
    for index, command in enumerate(checks):
        target = folder / str(index)
        process, stream = guarded(command, target, 60, folder, env)
        try:
            process.wait()
        finally:
            finish_guard(process, stream)
        state = read_json(target / "guard.json")
        if state.get("reason") != "exited" or state.get("returncode") != 0:
            raise RuntimeError(f"GPU/render preflight failed: {target / 'worker.log'}")
        if index == 3 and "ROBOTWIN_RENDER_READY" not in (target / "worker.log").read_text(encoding="utf-8", errors="replace"):
            raise RuntimeError("RoboTwin render test exited without confirming initialization")
    if not list(folder.glob("*.png")):
        raise RuntimeError("SAPIEN offscreen command returned without producing an image")


def port_free(port):
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"Port {port} is in use; no existing process was terminated") from error


def server_ready(port):
    # Use the policy server's documented HTTP health endpoint, not just a TCP
    # listener (which can become available before the model finishes loading).
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
            return response.status == 200
    except (OSError, ValueError):
        return False


def rendering_fault(attempt):
    """Recognize shared renderer failures, not ordinary invalid task seeds."""
    log = attempt / "worker.log"
    if not log.is_file():
        return False
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 128000))
        tail = stream.read().decode("utf-8", errors="replace").lower()
    return any(token in tail for token in ("vk_error", "error_incompatible_driver", "render error",
                                            "failed to create vulkan", "vulkan device lost"))


def execute_session(batch, manifest, paths, *, budget_s=28800, grace_s=900, episode_timeout_s=3600,
                    startup_timeout_s=1800, port=13400, runtime=None):
    """runtime is an internal test seam; the public CLI only uses RealRuntime."""
    if not all(math.isfinite(v) and v > 0 for v in (budget_s, episode_timeout_s, startup_timeout_s)):
        raise ValueError("Deadlines must be finite and positive")
    if not 0 <= grace_s < budget_s or budget_s > 28800:
        raise ValueError("Grace must be shorter than the session; maximum budget is 8 hours")
    if not 1 <= port <= 65535:
        raise ValueError("Invalid model service port")
    runtime = runtime or RealRuntime()
    # OS-held lock survives a stale lock *file* and is released on process death.
    with exclusive_lock(batch / ".run.lock"):
        load_manifest(batch)
        rows = completed_results(batch, manifest)
        if len(rows) == 500:
            return report(batch, save=True)
        for previous in (batch / "sessions").glob("*/session.json"):
            old = read_json(previous)
            if old.get("state") in ("preflight", "running"):
                # Owning the OS lock proves the old scheduler no longer holds it.
                # A guard may have finished after SIGKILL; recover its measured
                # service lifetime instead of treating the session as zero time.
                guard_file = previous.parent / "server/guard.json"
                if guard_file.is_file():
                    old["elapsed_s"] = max(old.get("elapsed_s", 0),
                                           read_json(guard_file).get("elapsed_s", 0))
                old.update(state="interrupted", recovered_at=now())
                atomic_json(previous, old)
        session = batch / "sessions" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
        session.mkdir(parents=True)
        state = {"schema": 1, "session_id": session.name, "started_at": now(), "state": "preflight",
                 "fingerprint": manifest["fingerprint"], "budget_s": budget_s, "paths": paths,
                 "elapsed_s": 0, "started_completed": len(rows)}
        atomic_json(session / "session.json", state)
        stop = []
        handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda number, frame: stop.append(number))
        started = None
        service = None
        try:
            runtime.preflight(batch, manifest, paths, session)
            if stop:
                state["state"] = "interrupted"
                return report(batch, save=True)
            started = time.monotonic()
            deadline = started + budget_s
            state.update(state="running", budget_started_at=now())
            atomic_json(session / "session.json", state)
            service = runtime.start_server(paths, session, budget_s, port)
            ready_deadline = min(deadline, started + startup_timeout_s)
            while not runtime.ready(service, port):
                if stop or time.monotonic() >= ready_deadline or not runtime.alive(service):
                    raise RuntimeError("Model startup interrupted, timed out, or service exited")
                time.sleep(0.05)
            fatal_service = False
            infrastructure_failures = 0
            for task in manifest["tasks"]:
                for episode in range(10):
                    key = episode_key(task, episode)
                    if key in rows:
                        continue
                    if stop or time.monotonic() >= deadline - grace_s or fatal_service:
                        break
                    seed = 100000 + episode
                    committed = False
                    common_render_error = False
                    for retry in range(3):
                        if stop or time.monotonic() >= deadline - grace_s:
                            break
                        if not runtime.alive(service):
                            fatal_service = True
                            break
                        attempt = session / "attempts" / (f"{task}_{episode:02d}_" + uuid.uuid4().hex[:10])
                        attempt.mkdir(parents=True)
                        request = {"fingerprint": manifest["fingerprint"], "task": task, "episode": episode,
                                   "seed": seed, "retry": retry, "session_id": session.name}
                        publish(attempt / "request.json", request)
                        begin = time.monotonic()
                        try:
                            runtime.episode(paths, attempt, task, seed, port,
                                            min(episode_timeout_s, deadline - time.monotonic()), stop)
                            success = validate_attempt(attempt, task, seed)
                            row = {**request, "outcome": "success" if success else "failure",
                                   "attempt": attempt.relative_to(batch).as_posix(), "finished_at": now(),
                                   "elapsed_s": time.monotonic() - begin, "evidence": evidence_inventory(attempt)}
                            publish(batch / "results" / (key + ".json"), row)
                            rows[key] = row
                            committed = True
                            atomic_json(attempt / "attempt.json", {"state": "committed", "outcome": row["outcome"]})
                            print(f"COMMITTED: {key} {row['outcome']} total={len(rows)}/500", flush=True)
                            break
                        except (OSError, ValueError, RuntimeError, KeyError) as error:
                            # Never retry a commit/storage error as if it were an
                            # ordinary policy failure; an already published record
                            # will be independently validated on the next run.
                            if (batch / "results" / (key + ".json")).exists():
                                raise
                            if isinstance(error, OSError) and error.errno in (28, 122, 5, 30):
                                raise
                            timed_out = False
                            if (attempt / "guard.json").is_file():
                                timed_out = read_json(attempt / "guard.json").get("reason") == "timeout"
                            atomic_json(attempt / "attempt.json", {"state": "timeout" if timed_out else "error",
                                                                     "error": str(error), "type": type(error).__name__})
                            print(f"RETRYABLE: {key}: {error}", flush=True)
                            common_render_error = common_render_error or rendering_fault(attempt)
                            if not runtime.alive(service) or not runtime.ready(service, port):
                                fatal_service = True
                                break
                    infrastructure_failures = infrastructure_failures + 1 if not committed and common_render_error else 0
                    if infrastructure_failures >= 3:
                        # Repeated shared renderer faults can leave model health
                        # endpoint healthy. Stop instead of trying all 500 seeds.
                        fatal_service = True
                    state.update(elapsed_s=time.monotonic() - started, completed=len(rows))
                    atomic_json(session / "session.json", state)
                    if fatal_service:
                        break
                if stop or time.monotonic() >= deadline - grace_s or fatal_service:
                    break
            state["state"] = ("complete" if len(rows) == 500 else "interrupted" if stop else
                              "service_failed" if fatal_service else "budget_or_pending")
        except BaseException as error:
            state.update(state="error", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            if service is not None:
                runtime.stop_server(service)
            state.update(ended_at=now(), elapsed_s=time.monotonic() - started if started else 0,
                         completed=len(rows))
            atomic_json(session / "session.json", state)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            report(batch, save=True)
        return report(batch)


def verify_identity(paths, manifest):
    source_preflight(paths)
    print("VERIFYING_IDENTITY: model, assets, code, packages (before 8h budget)", flush=True)
    actual = identity(paths, manifest["paths"])
    if actual != manifest["identity"]:
        changed = [key for key in actual if actual[key] != manifest["identity"].get(key)]
        raise ValueError("Evaluation identity changed: " + ", ".join(changed))


class RealRuntime:
    def preflight(self, batch, manifest, paths, session):
        if os.name != "posix":
            raise RuntimeError("Real evaluation is supported on the Linux ModelScope instance only")
        verify_identity(paths, manifest)
        import yaml
        config = yaml.safe_load((batch / "model/lingbotvla_cli.yaml").read_text(encoding="utf-8"))
        config["data"]["norm_stats_file"] = str(batch / "model/norm_stats.json")
        config["data"]["robot_config_root"] = str(Path(paths["lingbot_root"]) / "configs/robot_configs")
        config["model"]["tokenizer_path"] = paths["qwen_path"]
        # The official server merges model and train into its runtime config.
        # Keep its config_path valid after relocating the frozen model.
        for key in ("model_path", "config_path"):
            config["model"][key] = str(batch / "model")
        (session / "inference.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        self.env = runtime_environment(paths, session)
        rendering_preflight(paths, session, self.env)
        atomic_json(session / "effective_environment.json", {k: self.env.get(k) for k in (
            "HIP_VISIBLE_DEVICES", "VK_ICD_FILENAMES", "PYTHONPATH", "PYOPENGL_PLATFORM",
            "AMD_SERIALIZE_KERNEL", "AMD_SERIALIZE_COPY", "ROBOTWIN_EE_PLANNER")})

    def start_server(self, paths, session, budget_s, port):
        port_free(port)
        command = [paths["python"], str(HERE / "robotwin_eval_server.py"), "--lingbot-root",
                   paths["lingbot_root"], "--model-path", paths["model_path"], "--port", str(port)]
        self.server = guarded(command, session / "server", budget_s,
                              paths["lingbot_root"], self.env)
        return self.server

    def ready(self, service, port):
        return server_ready(port)

    def alive(self, service):
        return service[0].poll() is None

    def stop_server(self, service):
        finish_guard(*service)

    def episode(self, paths, attempt, task, seed, port, seconds, stop):
        command = [paths["python"], str(HERE / "robotwin_eval_worker.py"), "--robotwin-root",
                   paths["robotwin_root"], "--task", task, "--seed", str(seed), "--port", str(port),
                   "--output", str(attempt / "audit.json"), "--raw-dir", str(attempt / "raw")]
        process, stream = guarded(command, attempt, seconds, paths["robotwin_root"], self.env)
        try:
            while process.poll() is None:
                if stop:
                    break
                time.sleep(0.05)
        finally:
            finish_guard(process, stream)


def add_paths(parser):
    for key in ("lingbot_root", "robotwin_root", "qwen_path", "python", "vulkan_icd"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Freeze best weights, 500 episodes, and source/environment identity")
    prep.add_argument("--batch-dir", type=Path, required=True)
    prep.add_argument("--parent-run", type=Path, default=Path(DEFAULT_PARENT))
    prep.add_argument("--model-source", type=Path, help="Explicit relocated copy of best weights; recorded in selection.json")
    add_paths(prep)
    run = sub.add_parser("run", help="Continue only uncommitted episodes")
    run.add_argument("--batch-dir", type=Path, required=True)
    run.add_argument("--budget-seconds", type=float, default=28800)
    run.add_argument("--grace-seconds", type=float, default=900)
    run.add_argument("--episode-timeout-seconds", type=float, default=3600)
    run.add_argument("--startup-timeout-seconds", type=float, default=1800)
    run.add_argument("--port", type=int, default=13400)
    add_paths(run)
    for command in ("status", "report"):
        child = sub.add_parser(command)
        child.add_argument("--batch-dir", type=Path, required=True)
    args = parser.parse_args()
    batch = args.batch_dir.resolve()
    if args.command == "prepare":
        prepare(args)
        return
    if args.command == "run":
        manifest = load_manifest(batch)
        result = execute_session(batch, manifest, runtime_paths(args, batch, manifest),
                                 budget_s=args.budget_seconds, grace_s=args.grace_seconds,
                                 episode_timeout_s=args.episode_timeout_seconds,
                                 startup_timeout_s=args.startup_timeout_seconds, port=args.port)
    else:
        # status is a read-only snapshot of atomic records and can run while
        # the scheduler is active; report writes require exclusive ownership.
        if args.command == "status":
            result = report(batch)
        else:
            with exclusive_lock(batch / ".run.lock"):
                result = report(batch, save=True)
    print(json.dumps({k: result[k] for k in ("batch_id", "complete", "completed", "target", "success",
                                            "failure", "tasks_covered", "tasks_complete", "attempt_states")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
