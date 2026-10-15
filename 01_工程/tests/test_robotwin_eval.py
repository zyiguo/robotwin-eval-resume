"""Real subprocess/state-machine tests; no claim of RoboTwin/MI300X execution."""
import argparse
import ast
import contextlib
import io
import json
import os
import re
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import robotwin_eval as mod
import robotwin_eval_worker as worker
from full_training_common import atomic_json, exclusive_lock


def patched_evaluator():
    """Apply the actual pinned AMD file hunks in memory, checking every context line."""
    archives = Path(__file__).resolve().parents[2] / "02_运行归档/源码备份"
    with ZipFile(archives / "RoboTwin.zip") as archive:
        name = next(n for n in archive.namelist() if n.endswith("/scripts/eval_policy_xpolicylab.py"))
        source = archive.read(name).decode()
    with ZipFile(archives / "Robotwin-radeon-cloud.zip") as archive:
        name = next(n for n in archive.namelist() if n.endswith("/docker/patches/robotwin-rocm-reproduction.patch"))
        patch_text = archive.read(name).decode()
    section = patch_text.split("diff --git a/scripts/eval_policy_xpolicylab.py", 1)[1]
    section = section.split("\ndiff --git", 1)[0]
    original, result, cursor = source.splitlines(keepends=True), [], 0
    pieces = re.split(r"(?m)^@@ -([0-9]+)(?:,[0-9]+)? \+[0-9]+(?:,[0-9]+)? @@[^\n]*\n", section)
    for index in range(1, len(pieces), 2):
        start = int(pieces[index]) - 1
        result.extend(original[cursor:start])
        cursor = start
        for line in pieces[index+1].splitlines(keepends=True):
            if line.startswith((" ", "-")):
                if original[cursor] != line[1:]:
                    raise AssertionError(f"Pinned patch context mismatch: {cursor+1}")
                cursor += 1
            if line.startswith((" ", "+")):
                result.append(line[1:])
    result.extend(original[cursor:])
    return "".join(result)


def fixture_manifest(batch):
    tasks = mod.PRIORITY + [t for t in mod.EXPECTED_TASKS if t not in mod.PRIORITY]
    spec = {"schema": 1, "batch_id": "fixture", "settings": mod.SETTINGS,
            "tasks": tasks, "identity": {"fixture": "not-a-real-GPU-run"}}
    manifest = {**spec, "fingerprint": mod.digest(spec), "created_at": mod.now(), "paths": {}}
    mod.publish(batch / "manifest.json", manifest)
    return manifest


def write_attempt(attempt, task, seed, success, *, error=False):
    atomic_json(attempt / "guard.json", {"state": "finished", "reason": "exited", "returncode": 0})
    atomic_json(attempt / "audit.json", {"schema": 1, "task": task, "seed": seed,
                "valid": True, "main_returned": True, "caught_errors": [],
                "episodes": [{"seed": seed, "success": success, "check_success": success}]})
    text = f"Final success rate: {int(success)}/1 = {100.0 if success else 0.0}%\n"
    if error:
        text = "Policy rollout error:\nTraceback (most recent call last)\n" + text
    (attempt / "worker.log").write_text(text, encoding="utf-8")
    (attempt / "raw").mkdir(exist_ok=True)
    (attempt / "raw/_result.txt").write_text(str(float(success)), encoding="utf-8")


class FixtureRuntime:
    def __init__(self, max_calls=1, *, errors=False, interrupt_at=None):
        self.calls = []
        self.max_calls = max_calls
        self.errors = errors
        self.interrupt_at = interrupt_at
        self.stopped = False

    def preflight(self, *args):
        pass

    def start_server(self, *args):
        return object()

    def ready(self, *args):
        return True

    def alive(self, *args):
        return len(self.calls) < self.max_calls

    def stop_server(self, *args):
        self.stopped = True

    def episode(self, paths, attempt, task, seed, port, seconds, stop):
        self.calls.append((task, seed))
        if self.interrupt_at == len(self.calls):
            raise KeyboardInterrupt()
        write_attempt(attempt, task, seed, seed % 2 == 1, error=self.errors)


class SubprocessRuntime(FixtureRuntime):
    def __init__(self, max_calls=1, sleep_s=0):
        super().__init__(max_calls)
        self.sleep_s = sleep_s

    def episode(self, paths, attempt, task, seed, port, seconds, stop):
        self.calls.append((task, seed))
        code = '''import sys,time,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from full_training_common import atomic_json
root=Path(sys.argv[2]); task=sys.argv[3]; seed=int(sys.argv[4])
time.sleep(float(sys.argv[5]))
success=bool(seed%2)
atomic_json(root/'audit.json',dict(schema=1,task=task,seed=seed,valid=True,main_returned=True,
    caught_errors=[],episodes=[dict(seed=seed,success=success,check_success=success)]))
(root/'raw').mkdir()
(root/'raw/_result.txt').write_text(str(float(success)))
print(f'Final success rate: {int(success)}/1 = {100.0 if success else 0.0}%',flush=True)
'''
        process, stream = mod.guarded([sys.executable, "-c", code, str(SCRIPTS), str(attempt),
                                        task, str(seed), str(self.sleep_s)],
                                       attempt, seconds, attempt, os.environ.copy())
        try:
            process.wait(timeout=10)
        finally:
            mod.finish_guard(process, stream)


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup)
        self.batch = Path(self.tmp.name) / "batch"
        self.batch.mkdir()
        self.manifest = fixture_manifest(self.batch)
        self.quiet = self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def run_session(self, runtime, **kwargs):
        return mod.execute_session(self.batch, self.manifest, {}, runtime=runtime,
                                   budget_s=kwargs.pop("budget_s", 10),
                                   grace_s=kwargs.pop("grace_s", 0), **kwargs)

    def test_resume_skips_success_and_normal_failure(self):
        first = FixtureRuntime(max_calls=2)
        self.run_session(first)
        second = FixtureRuntime(max_calls=1)
        summary = self.run_session(second)
        self.assertEqual(first.calls, [("adjust_bottle", 100000), ("adjust_bottle", 100001)])
        self.assertEqual(second.calls, [("adjust_bottle", 100002)])
        self.assertEqual((summary["completed"], summary["success"], summary["failure"]), (3, 1, 2))
        self.assertIsNone(summary["final_overall_success_rate"])
        self.assertTrue(first.stopped and second.stopped)

    def test_real_subprocess_results_resume_without_repeating(self):
        first = SubprocessRuntime(2)
        self.run_session(first)
        second = SubprocessRuntime(1)
        result = self.run_session(second)
        self.assertEqual(second.calls, [("adjust_bottle", 100002)])
        self.assertEqual(result["completed"], 3)

    def test_hard_deadline_kills_real_episode_and_leaves_it_pending(self):
        runtime = SubprocessRuntime(1, sleep_s=20)
        summary = self.run_session(runtime, budget_s=0.4)
        self.assertEqual(summary["completed"], 0)
        self.assertEqual(summary["attempt_states"].get("timeout"), 1)
        second = SubprocessRuntime(1)
        self.run_session(second)
        self.assertEqual(second.calls, [("adjust_bottle", 100000)])

    def test_changed_runtime_identity_stops_before_server_start(self):
        with patch.object(mod, "source_preflight"), patch.object(mod, "identity", return_value={"fixture": "changed"}):
            with self.assertRaisesRegex(ValueError, "identity changed"):
                mod.verify_identity({}, self.manifest)

    def test_source_data_code_is_hashed_but_training_data_is_excluded(self):
        root = self.batch / "source"
        (root / "lingbotvla/data").mkdir(parents=True)
        (root / "data").mkdir()
        (root / "lingbotvla/data/normalize.py").write_text("x=1", encoding="utf-8")
        (root / "data/raw.bin").write_bytes(b"training data")
        inventory = mod.tree_inventory(root, {}, source_only=True)
        self.assertIn("lingbotvla/data/normalize.py", inventory)
        self.assertNotIn("data/raw.bin", inventory)

    def test_packaged_cli_help_and_status_work_without_model_imports(self):
        import build_robotwin_eval_package as builder
        package = builder.build(self.batch.parent / "upload.zip")
        target = self.batch.parent / "unpacked"
        with ZipFile(package) as archive:
            names = archive.namelist()
            self.assertEqual(len(names), 11)
            self.assertFalse(any(n.endswith((".safetensors", ".cookie", ".enw")) for n in names))
            archive.extractall(target)
        entry = target / "robotwin_eval/scripts/robotwin_eval.py"
        help_result = subprocess.run([sys.executable, str(entry), "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        with exclusive_lock(self.batch / ".run.lock"):
            result = subprocess.run([sys.executable, str(entry), "status", "--batch-dir", str(self.batch)],
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["completed"], 0)

    def test_keyboard_interrupt_preserves_first_and_retries_current(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_session(FixtureRuntime(100, interrupt_at=2))
        runtime = FixtureRuntime(1)
        self.run_session(runtime)
        self.assertEqual(runtime.calls, [("adjust_bottle", 100001)])
        self.assertEqual(mod.report(self.batch)["completed"], 2)

    def test_rollout_error_with_final_zero_is_not_model_failure(self):
        self.run_session(FixtureRuntime(3, errors=True))
        summary = mod.report(self.batch)
        self.assertEqual(summary["completed"], 0)
        self.assertEqual(summary["failure"], 0)
        self.assertEqual(summary["attempt_states"]["error"], 3)

    def test_inconsistent_seed_or_check_success_rejected(self):
        attempt = self.batch / "attempt"
        attempt.mkdir()
        write_attempt(attempt, "adjust_bottle", 100000, False)
        with self.assertRaises(ValueError):
            mod.validate_attempt(attempt, "adjust_bottle", 100001)
        audit = mod.read_json(attempt / "audit.json")
        audit["episodes"][0]["check_success"] = True
        atomic_json(attempt / "audit.json", audit)
        with self.assertRaises(ValueError):
            mod.validate_attempt(attempt, "adjust_bottle", 100000)

    def test_corrupt_record_or_missing_evidence_refuses_resume(self):
        self.run_session(FixtureRuntime(1))
        row = mod.read_json(self.batch / "results/adjust_bottle/00.json")
        (self.batch / row["attempt"] / "worker.log").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "evidence"):
            mod.completed_results(self.batch, self.manifest)

    def test_partial_and_duplicate_records_not_silently_accepted(self):
        result_dir = self.batch / "results/adjust_bottle"
        result_dir.mkdir(parents=True)
        (result_dir / "00.json.pending").write_text("{", encoding="utf-8")
        self.assertEqual(mod.report(self.batch)["completed"], 0)
        self.run_session(FixtureRuntime(1))
        shutil.copy2(result_dir / "00.json", result_dir / "duplicate.json")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            mod.report(self.batch)

    def test_atomic_publish_never_overwrites_or_commits_partial_write(self):
        path = self.batch / "one.json"
        mod.publish(path, {"ok": 1})
        with self.assertRaises(FileExistsError):
            mod.publish(path, {"ok": 2})
        self.assertEqual(mod.read_json(path), {"ok": 1})
        with patch.object(mod.os, "fsync", side_effect=OSError(28, "disk full")):
            with self.assertRaises(OSError):
                mod.publish(self.batch / "missing.json", {"ok": 3})
        self.assertFalse((self.batch / "missing.json").exists())

    def test_disk_full_during_commit_stops_without_false_progress(self):
        original = mod.publish
        def publish(path, value):
            if Path(path).parent.name == "adjust_bottle":
                raise OSError(28, "disk full")
            return original(path, value)
        with patch.object(mod, "publish", side_effect=publish):
            with self.assertRaises(OSError):
                self.run_session(FixtureRuntime(5))
        self.assertEqual(mod.report(self.batch)["completed"], 0)

    def test_identity_change_rejected(self):
        changed = dict(self.manifest)
        changed["settings"] = {**mod.SETTINGS, "use_fp32": False}
        atomic_json(self.batch / "manifest.json", changed)
        with self.assertRaises(ValueError):
            mod.load_manifest(self.batch)

    def test_cross_directory_restore_keeps_progress_and_resumes(self):
        self.run_session(FixtureRuntime(2))
        restored = self.batch.parent / "restored"
        shutil.copytree(self.batch, restored)
        original_summary = mod.report(self.batch)
        self.batch = restored
        summary = mod.report(self.batch)
        self.assertEqual(summary["completed"], original_summary["completed"])
        runtime = FixtureRuntime(1)
        self.run_session(runtime)
        self.assertEqual(runtime.calls, [("adjust_bottle", 100002)])

    def test_lock_denies_second_scheduler(self):
        with exclusive_lock(self.batch / ".run.lock"):
            process = subprocess.run([sys.executable, "-c",
                "import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
                "from full_training_common import exclusive_lock; "
                "exec('with exclusive_lock(Path(sys.argv[2])): pass')", str(SCRIPTS), str(self.batch / ".run.lock")],
                capture_output=True, text=True)
        self.assertNotEqual(process.returncode, 0)
        with exclusive_lock(self.batch / ".run.lock"):
            pass

    def test_all_500_include_failures_and_no_501st_execution(self):
        runtime = FixtureRuntime(501)
        summary = self.run_session(runtime, budget_s=600)
        self.assertTrue(summary["complete"])
        self.assertEqual((summary["completed"], summary["tasks_complete"]), (500, 50))
        self.assertEqual(summary["final_macro_success_rate"], 0.5)
        later = FixtureRuntime(1)
        self.run_session(later)
        self.assertEqual(later.calls, [])

    def test_time_budget_and_grace_stop_dispatch(self):
        class SlowRuntime(FixtureRuntime):
            def episode(self, *args):
                super().episode(*args)
                time.sleep(0.06)
        runtime = SlowRuntime(500)
        summary = self.run_session(runtime, budget_s=0.12, grace_s=0.07)
        self.assertLess(len(runtime.calls), 3)
        self.assertGreaterEqual(summary["completed"], 1)
        self.assertTrue(runtime.stopped)

    def test_package_identity_probe_is_executable(self):
        identity = mod.package_identity(sys.executable)
        self.assertEqual(identity["python"], list(sys.version_info[:2]))

    def test_venv_python_invocation_is_not_resolved_to_base_interpreter(self):
        executable = self.batch / "venv/bin/python"
        with patch.object(Path, "resolve", autospec=True, side_effect=lambda p: p.absolute()) as resolve:
            paths = mod.runtime_paths(argparse.Namespace(), self.batch, {"paths": {"python": str(executable)}})
        self.assertEqual(paths["python"], str(executable.absolute()))
        self.assertNotIn(executable, [call.args[0] for call in resolve.call_args_list])

    def test_pinned_launcher_and_both_evaluator_shapes(self):
        archives = Path(__file__).resolve().parents[2] / "02_运行归档/源码备份"
        with ZipFile(archives / "lingbot-vla-v2.zip") as archive:
            name = next(n for n in archive.namelist() if n.endswith("/experiment/robotwin/start_robotwin_infer_and_eval.sh"))
            launcher = self.batch / "launcher.sh"
            launcher.write_bytes(archive.read(name))
            tasks = mod.tasks_from_launcher(launcher)
            self.assertEqual(len(tasks), 50)
            self.assertEqual(tasks[:5], mod.PRIORITY)
        with ZipFile(archives / "RoboTwin.zip") as archive:
            name = next(n for n in archive.namelist() if n.endswith("/scripts/eval_policy_xpolicylab.py"))
            source = archive.read(name).decode()
        worker.instrument(source, "pinned-upstream")
        amd_shape = patched_evaluator()
        self.assertIn("--seed_start", amd_shape)
        worker.instrument(amd_shape, "pinned-amd-evaluator")
        with self.assertRaises(ValueError):
            worker.instrument(source.replace("now_id += 1", "now_id = now_id + 1"), "unsupported-drift")


class GuardTests(unittest.TestCase):
    def test_guard_cleans_owned_child_even_if_first_status_write_fails(self):
        import robotwin_eval_guard as guard
        child = Mock(pid=12345, returncode=None)
        with patch.object(guard.subprocess, "Popen", return_value=child), \
                patch.object(guard, "atomic_json", side_effect=OSError(28, "disk full")), \
                patch.object(guard, "stop_group") as cleanup, patch.object(guard.signal, "signal"):
            with self.assertRaises(OSError):
                guard.supervise(["fixture"], Path("unused"), 5, os.getppid())
            cleanup.assert_called_once_with(child)

    def test_real_child_deadline_and_cleanup(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder = Path(tmp)
            process, stream = mod.guarded([sys.executable, "-c", "import time; time.sleep(20)"],
                                          folder, 0.15, folder, os.environ.copy())
            try:
                self.assertNotEqual(process.wait(timeout=10), 0)
            finally:
                mod.finish_guard(process, stream)
            status = mod.read_json(folder / "guard.json")
            self.assertEqual(status["reason"], "timeout")

    @unittest.skipUnless(os.name == "posix", "Parent reparenting is the Linux production contract")
    def test_sigkill_scheduler_guard_reaps_child(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder = Path(tmp)
            parent = subprocess.Popen([sys.executable, "-c",
                "import sys,time; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
                "import robotwin_eval as m,os; "
                "p,s=m.guarded([sys.executable,'-c','import time; time.sleep(60)'], "
                "Path(sys.argv[2]), 60,Path(sys.argv[2]),os.environ.copy()); time.sleep(60)",
                str(SCRIPTS), str(folder)])
            try:
                end = time.monotonic() + 10
                while not (folder / "guard.json").is_file() and time.monotonic() < end:
                    time.sleep(0.02)
                self.assertTrue((folder / "guard.json").is_file())
                parent.kill()
                parent.wait(timeout=5)
                end = time.monotonic() + 10
                while time.monotonic() < end:
                    status = mod.read_json(folder / "guard.json")
                    if status.get("reason") == "parent_lost":
                        break
                    time.sleep(0.02)
                self.assertEqual(status.get("reason"), "parent_lost")
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait()


class AuditedWorkerTests(unittest.TestCase):
    def exercise(self, *, success=False, policy_error=False):
        # Execute the real patched policy loop with lightweight simulator/model
        # doubles. This checks caught-error handling and audit placement inside
        # upstream control flow; it is still not a real physical simulation.
        tree = ast.parse(patched_evaluator())
        loop = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "eval_remote_policy")
        prelude = '''from __future__ import annotations
import argparse, time, traceback
from pathlib import Path
import numpy as np
class UnStableError(Exception): pass
class Env:
    def __init__(self):
        self.plan_success=True; self.eval_success=False; self.ticks=0
        self.render_freq=0; self.eval_video_path=None
    def setup_demo(self, **kwargs): self.ticks=0; self.eval_success=False
    def play_once(self): return {"info": {}}
    def close_env(self, **kwargs): pass
    def check_success(self): return self.eval_success
    def get_instruction(self): return "test seen instruction"
    def set_instruction(self, instruction): pass
    def get_obs(self):
        camera={"rgb": np.zeros((2,2,3),dtype=np.uint8)}
        return {"observation": {"head_camera": camera,"left_camera": camera,"right_camera": camera},
                "joint_action": {"vector": np.zeros(14)}}
    def take_action(self, action):
        self.ticks+=1; self.eval_success=SUCCESS
class Client:
    _robotwin_protocol="lingbot_vla_v2"
    def infer(self, request):
        if request.get("reset"): return {}
        if POLICY_ERROR: raise ConnectionError("fixture model disconnected")
        return {"action": [[0.0]*14]}
def class_decorator(name): return Env()
def is_episode_end(env): return env.ticks>=1
def normalize_action_chunk(value): return value
def build_instruction(*args): return "test seen instruction"
def notify_trial_end(*args): pass
def parse_bool(value): return bool(value)
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--task_name"); parser.add_argument("--seed_start",type=int)
    args,unused=parser.parse_known_args()
    return {"task_name": args.task_name,"seed_start": args.seed_start}
def main(args):
    env=class_decorator(args["task_name"])
    cfg={"task_name":args["task_name"],"policy_name":"LingBot-VLA-v2", "task_config":"demo_clean",
         "ckpt_setting":"fixture", "clear_cache_freq":1,"render_freq":0}
    unused,successes=eval_remote_policy(args["task_name"],env,cfg,Client(),args["seed_start"],
        usr_args={"expert_check":True,"accept_expert_info_on_failure":True},test_num=1,instruction_type="seen")
    print(f"Final success rate: {successes}/1 = {successes*100.0}%")
'''
        source = (prelude + f"\nSUCCESS={success!r}\nPOLICY_ERROR={policy_error!r}\n" + ast.unparse(loop))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            (root / "test_render.py").write_text("def Sapien_TEST(): pass\n", encoding="utf-8")
            evaluator = root / "eval_policy.py"
            evaluator.write_text(source, encoding="utf-8")
            sys.path.insert(0, str(root))
            old_argv = sys.argv[:]
            try:
                sys.modules.pop("test_render", None)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    rc = worker.evaluate(evaluator, "adjust_bottle", 100000, 13400,
                                         root / "audit.json", root / "raw")
                audit = mod.read_json(root / "audit.json")
            finally:
                sys.argv = old_argv
                sys.path.remove(str(root))
                sys.modules.pop("test_render", None)
        return rc, audit

    def test_actual_amd_loop_success_is_audited(self):
        rc, audit = self.exercise(success=True)
        self.assertEqual(rc, 0)
        self.assertTrue(audit["episodes"][0]["check_success"])

    def test_actual_amd_loop_normal_failure_is_audited(self):
        rc, audit = self.exercise(success=False)
        self.assertEqual(rc, 0)
        self.assertFalse(audit["episodes"][0]["success"])

    def test_actual_amd_loop_swallowed_disconnect_is_rejected(self):
        rc, audit = self.exercise(policy_error=True)
        self.assertEqual(rc, 1)
        self.assertFalse(audit["valid"])
        self.assertEqual(audit["caught_errors"][0]["type"], "ConnectionError")


class CheckpointTests(unittest.TestCase):
    def test_truncated_shard_rejected_and_snapshot_survives_source_unlink(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
            shard = source / "model.safetensors"
            shard.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
            for name in ("config.json", "norm_stats.json"):
                atomic_json(source / name, {"fixture": True})
            (source / "lingbotvla_cli.yaml").write_text("model: {}\ndata: {}\ntrain: {}\n", encoding="utf-8")
            atomic_json(source / "export_metadata.json", {"save_kind": "model_weights_only",
                                                          "optimizer_state_saved": False, "total_tensor_bytes": 4})
            mod.checkpoint_files(source)
            destination = Path(tmp) / "snapshot"
            mod.snapshot_model(source, destination)
            shard.unlink()
            mod.checkpoint_files(destination)
            frozen_shard = destination / "model.safetensors"
            frozen_shard.write_bytes(frozen_shard.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                mod.checkpoint_files(destination)

    def test_prepare_freezes_best_and_real_identity_detects_model_change(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            source = root / "parent/checkpoints/global_step_5/hf_ckpt"
            source.mkdir(parents=True)
            header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
            (source / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
            for name in ("config.json", "norm_stats.json"):
                atomic_json(source / name, {"fixture": True})
            (source / "lingbotvla_cli.yaml").write_text("model: {}\ndata: {}\ntrain: {}\n", encoding="utf-8")
            atomic_json(source / "export_metadata.json", {"save_kind": "model_weights_only", "kind": "formal_weights",
                "optimizer_state_saved": False, "total_tensor_bytes": 4, "global_step": 5, "run_id": "fixture-run"})
            atomic_json(source / "run_manifest.json", {"run_id": "fixture-run"})
            parent = root / "parent"
            atomic_json(parent / "best.json", {"global_step": 5, "run_id": "fixture-run", "path": str(source)})
            lingbot, robotwin, qwen = root / "lingbot", root / "RoboTwin", root / "qwen"
            files = {
                lingbot / "deploy/lingbot_vla_v2_policy.py": "# LINGBOTVLA_TRAINING_CONFIG\n",
                lingbot / "experiment/robotwin/start_robotwin_infer_and_eval.sh":
                    "task_list_all=(" + " ".join(json.dumps(t) for t in mod.EXPECTED_TASKS) + ")\n",
                robotwin / "scripts/eval_policy_xpolicylab.py": patched_evaluator(),
                robotwin / "envs/robot/planner.py": "# ROBOTWIN_DISABLE_CUROBO\n",
                robotwin / "envs/robot/robot.py": "# ROBOTWIN_EE_PLANNER\n",
                robotwin / "XPolicyLab/__init__.py": "# fixture\n",
                qwen / "config.json": "{}\n",
                root / "icd.json": "{}\n"}
            for directory in ("objects", "embodiments", "background_texture"):
                files[robotwin / "assets" / directory / "fixture.bin"] = "asset"
            for path, contents in files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            batch = root / "frozen"
            args = argparse.Namespace(batch_dir=batch, parent_run=parent, model_source=None,
                lingbot_root=lingbot, robotwin_root=robotwin, qwen_path=qwen, python=Path(sys.executable),
                vulkan_icd=root / "icd.json")
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = mod.prepare(args)
                mod.verify_identity(manifest["paths"], manifest)
                # Changes to the original best pointer cannot switch the model.
                atomic_json(parent / "best.json", {"path": "/different-model", "global_step": 10})
                self.assertEqual(mod.read_json(batch / "selection.json")["best"]["global_step"], 5)
                (source / "model.safetensors").unlink()
                mod.verify_identity(manifest["paths"], manifest)
                atomic_json(batch / "model/norm_stats.json", {"fixture": "different"})
                with self.assertRaisesRegex(ValueError, "model"):
                    mod.verify_identity(manifest["paths"], manifest)


if __name__ == "__main__":
    unittest.main()
