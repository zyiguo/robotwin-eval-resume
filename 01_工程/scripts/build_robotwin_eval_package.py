"""Build a small standalone upload, excluding models, credentials, and datasets."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


SCRIPTS = ("robotwin_eval.py", "robotwin_eval_guard.py", "robotwin_eval_worker.py",
           "robotwin_eval_server.py", "full_training_common.py", "fix_moe_eval_dtype.py")


def build(output=None):
    engineering = Path(__file__).resolve().parents[1]
    workspace = engineering.parent
    output = Path(output) if output else workspace / "02_运行归档/闭环评测/robotwin_eval_upload.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(".pending.zip")
    with ZipFile(pending, "w", ZIP_DEFLATED) as package:
        for name in SCRIPTS:
            package.write(engineering / "scripts" / name, "robotwin_eval/scripts/" + name)
        package.write(engineering / "configs/robotwin_eval_tasks.json", "robotwin_eval/configs/robotwin_eval_tasks.json")
        package.write(workspace / "docs/plans/2026-10-15-robotwin-closed-loop-evaluation.md",
                      "robotwin_eval/README.md")
        package.writestr("robotwin_eval/requirements.txt", "PyYAML>=6,<7\n")
        # Reference patches are packaged as data. They are never automatically
        # applied to an already modified or running cloud checkout.
        source = workspace / "02_运行归档/源码备份/Robotwin-radeon-cloud.zip"
        with ZipFile(source) as archive:
            for name in ("lingbot-vla-v2-rocm.patch", "robotwin-rocm-reproduction.patch"):
                entry = next(n for n in archive.namelist() if n.endswith("/docker/patches/" + name))
                package.writestr("robotwin_eval/reference_patches/" + name, archive.read(entry))
    with ZipFile(pending) as package:
        error = package.testzip()
        if error:
            raise ValueError(f"Upload archive failed CRC: {error}")
    pending.replace(output)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(checksum + "  " + output.name + "\n", encoding="ascii")
    print(f"PACKAGE_READY: {output} ({output.stat().st_size} bytes)\nSHA256: {checksum}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    build(parser.parse_args().output)
