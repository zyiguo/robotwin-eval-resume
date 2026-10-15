"""Own a subprocess group; stop it on deadline, signal, or scheduler death (Linux)."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from full_training_common import atomic_json


def stop_group(process, grace=1.0):
    # Only used for a process we created with start_new_session=True.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        end = time.monotonic() + grace
        while process.poll() is None and time.monotonic() < end:
            time.sleep(0.02)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait()


def supervise(command, status, seconds, parent):
    if os.getppid() != parent:
        return 125
    interrupted = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    reason = "exited"
    try:
        atomic_json(status, {"state": "running", "pid": process.pid, "parent": parent})
        while process.poll() is None:
            if interrupted:
                reason = "interrupted"
                break
            if os.getppid() != parent:
                reason = "parent_lost"
                break
            if time.monotonic() - started >= seconds:
                reason = "timeout"
                break
            time.sleep(0.05)
    finally:
        stop_group(process)
        atomic_json(status, {"state": "finished", "pid": process.pid, "reason": reason,
                             "returncode": process.returncode,
                             "elapsed_s": time.monotonic() - started})
    return process.returncode if reason == "exited" and process.returncode >= 0 else 124


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.seconds <= 0 or not command:
        parser.error("A positive deadline and command are required")
    raise SystemExit(supervise(command, args.status, args.seconds, args.parent))


if __name__ == "__main__":
    main()
