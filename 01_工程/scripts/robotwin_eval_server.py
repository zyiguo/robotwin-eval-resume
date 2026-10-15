"""Launch the official policy with the training-time MoE guard applied in memory."""
import argparse
import ast
from pathlib import Path
import runpy
import sys

from fix_moe_eval_dtype import patched_source


def install_moe_guard():
    import lingbotvla.models.vla.lingbot_vla.qwen2_action_expert as module
    from transformers.utils import logging
    source = patched_source(Path(module.__file__).read_text(encoding="utf-8"))
    module.logger = logging.get_logger(module.__name__)
    count = 0
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if (isinstance(method, ast.FunctionDef) and method.name == "forward"
                        and "use_robby_moe" in ast.unparse(method)):
                    scope = dict(module.__dict__)
                    exec(compile(ast.Module(body=[method], type_ignores=[]), module.__file__, "exec"), scope)
                    getattr(module, node.name).forward = scope["forward"]
                    count += 1
    if count != 1:
        raise ValueError("Unsupported upstream MoE eval method; no shared source was modified")
    print("MOE_EVAL_DTYPE_FIXED: in-memory", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.lingbot_root.resolve()))
    install_moe_guard()
    sys.argv = ["lingbot_vla_v2_policy", "--model_path", str(args.model_path),
                "--use_length", "50", "--use_bf16", "False", "--use_fp32", "True",
                "--use_compile", "False", "--port", str(args.port)]
    runpy.run_module("deploy.lingbot_vla_v2_policy", run_name="__main__")


if __name__ == "__main__":
    main()
