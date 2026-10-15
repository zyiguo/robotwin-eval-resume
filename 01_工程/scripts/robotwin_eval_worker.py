"""Audit one real RoboTwin episode, without changing its success criterion or source."""
import argparse
import ast
import os
from pathlib import Path
import sys
import traceback
import types

from full_training_common import atomic_json


class FixedSeedError(BaseException):
    pass


class EpisodeAudit(ast.NodeTransformer):
    """Instrument only the pinned non-batch policy loop; fail closed on shape drift."""
    def __init__(self):
        self.inside = False
        self.functions = 0
        self.completions = 0

    def visit_FunctionDef(self, node):
        if node.name != "eval_remote_policy":
            return node
        self.functions += 1
        self.inside = True
        node = self.generic_visit(node)
        self.inside = False
        return node

    def visit_ExceptHandler(self, node):
        node = self.generic_visit(node)
        if self.inside:
            call = ast.parse("_audit_caught_exception()\n").body[0]
            node.body.insert(0, ast.copy_location(call, node))
        return node

    def visit_AugAssign(self, node):
        if (self.inside and isinstance(node.target, ast.Name) and node.target.id == "now_id"
                and isinstance(node.op, ast.Add) and isinstance(node.value, ast.Constant)
                and node.value.value == 1):
            self.completions += 1
            call = ast.parse("_audit_finish(task_env, now_seed, succ)\n").body[0]
            return [ast.copy_location(call, node), node]
        return node


def instrument(source, filename):
    tree = ast.parse(source, filename=filename)
    audit = EpisodeAudit()
    tree = audit.visit(tree)
    if (audit.functions, audit.completions) != (1, 1):
        raise ValueError("Unsupported RoboTwin policy loop: expected one episode completion point")
    ast.fix_missing_locations(tree)
    return compile(tree, filename, "exec")


def evaluate(evaluator, task, seed, port, output, raw_dir):
    record = {"schema": 1, "task": task, "seed": seed, "caught_errors": [], "episodes": []}

    def caught():
        typ, exc, unused = sys.exc_info()
        record["caught_errors"].append({"type": typ.__name__ if typ else "unknown", "message": str(exc)})

    def finish(env, actual_seed, success):
        checked = bool(env.check_success())
        record["episodes"].append({"seed": actual_seed, "success": bool(success),
                                    "check_success": checked})
        if len(record["episodes"]) != 1 or actual_seed != seed or checked != bool(success):
            raise FixedSeedError("Unexpected seed, repeated completion, or inconsistent check_success()")

    module_name = "_robotwin_audited_episode"
    module = types.ModuleType(module_name)
    module.__file__ = str(evaluator)
    module.__dict__.update(_audit_caught_exception=caught, _audit_finish=finish)
    sys.modules[module_name] = module
    try:
        # __name__ is deliberately not __main__: the upstream CLI entry point is
        # invoked below, after installing fixed-seed and raw-output guards.
        exec(instrument(evaluator.read_text(encoding="utf-8"), str(evaluator)), module.__dict__)
        original_factory = module.class_decorator

        def factory(name):
            env = original_factory(name)
            original_setup = env.setup_demo

            def setup(*args, **kwargs):
                if kwargs.get("seed") != seed:
                    raise FixedSeedError("RoboTwin attempted to skip the prescribed seed")
                return original_setup(*args, **kwargs)

            env.setup_demo = setup
            return env

        module.class_decorator = factory
        module.build_eval_save_dir = lambda *args, **kwargs: raw_dir
        sys.argv = [str(evaluator), "--task_name", task, "--task_config", "demo_clean",
                    "--policy_name", "LingBot-VLA-v2", "--protocol", "lingbot_vla_v2",
                    "--host", "127.0.0.1", "--port", str(port), "--device_id", "0",
                    "--seed", "0", "--seed_start", str(seed), "--test_num", "1",
                    "--instruction_type", "seen", "--expert_check", "true",
                    "--accept_expert_info_on_failure", "true", "--eval_batch", "false",
                    "--additional_info", "eval_video_log=false"]
        from test_render import Sapien_TEST
        Sapien_TEST()
        module.main(module.parse_args())
        record["main_returned"] = True
        if record["caught_errors"] or len(record["episodes"]) != 1:
            raise RuntimeError("Episode had caught infrastructure errors or no audited completion")
        record["valid"] = True
    except BaseException as exc:
        record["valid"] = False
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    finally:
        atomic_json(output, record)
    return 0 if record.get("valid") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.robotwin_root.resolve()
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "scripts"), str(root / "description/utils")]
    raise SystemExit(evaluate(root / "scripts/eval_policy_xpolicylab.py", args.task,
                              args.seed, args.port, args.output, args.raw_dir))


if __name__ == "__main__":
    main()
