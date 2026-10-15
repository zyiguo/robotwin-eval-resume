"""Patch the official eval-only MoE fast path without changing parameter dtype."""
import ast
from pathlib import Path
import sys


def patched_source(source):
    anchor = "                and hidden_flat.is_cuda\n"
    guard = ("                and hidden_flat.dtype == self.experts.gate_proj.dtype\n"
             "                and hidden_flat.dtype == self.experts.up_proj.dtype\n"
             "                and hidden_flat.dtype == self.experts.down_proj.dtype\n")
    if guard not in source:
        if source.count(anchor) != 1:
            raise ValueError("Unexpected upstream MoE condition; stopped without writing")
        source = source.replace(anchor, anchor + guard)
    logger_anchor = "from transformers.processing_utils import Unpack\n"
    if "logger = logging.get_logger(__name__)" not in source:
        if source.count(logger_anchor) != 1:
            raise ValueError("Unexpected upstream logging imports; stopped without writing")
        source = source.replace(logger_anchor, logger_anchor + "\nlogger = logging.get_logger(__name__)\n")
    ast.parse(source)
    return source


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/workspace/src/lingbot-vla-v2")
    path = root / "lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py"
    source = path.read_text(encoding="utf-8")
    updated = patched_source(source)
    if updated != source:
        backup = path.with_suffix(".py.before_eval_dtype_fix")
        if not backup.exists():
            backup.write_text(source, encoding="utf-8")
        path.write_text(updated, encoding="utf-8")
    print("MOE_EVAL_DTYPE_FIXED", path)


if __name__ == "__main__":
    main()
