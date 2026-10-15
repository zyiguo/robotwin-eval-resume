"""Small, dependency-free filesystem and provenance helpers for formal runs."""
import contextlib
import json
import os
from pathlib import Path
import shutil
import uuid


def is_link(path):
    path = Path(path)
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def contained(path, root):
    """Check lexical and resolved confinement, including linked ancestors."""
    path, root = Path(path).absolute(), Path(root).absolute()
    if path == root or root not in path.parents:
        raise ValueError(f"Expected a child of {root}: {path}")
    for candidate in (path, *path.parents):
        if is_link(candidate):
            raise ValueError(f"Linked path is not managed: {candidate}")
        if candidate == root:
            break
    if root.resolve() not in path.resolve().parents:
        raise ValueError(f"Resolved path escapes {root}: {path}")
    return path


def reject_links(root):
    root = Path(root)
    if is_link(root):
        raise ValueError(f"Linked directory: {root}")
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            if is_link(Path(parent) / name):
                raise ValueError(f"Linked content: {Path(parent) / name}")


def owned_directory(path, identity):
    path = Path(path)
    marker = path / ".managed.json"
    if path.exists():
        if path.is_symlink() or not marker.is_file() or read_json(marker) != identity:
            raise FileExistsError(f"Preserving directory without matching ownership: {path}")
        reject_links(path)
    else:
        path.mkdir(parents=True)
        atomic_json(marker, identity)
    return path


def remove_owned(path, root, identity):
    path = contained(path, root)
    marker = path / ".managed.json"
    if read_json(marker) != identity:
        raise ValueError(f"Ownership mismatch: {path}")
    reject_links(path)
    # Keep the outer owner record during lengthy recursive work. If interrupted,
    # the next invocation can still authenticate and finish this same cleanup.
    for child in path.iterdir():
        if child == marker:
            continue
        contained(child, path)
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    marker.unlink()
    try:
        path.rmdir()
    except BaseException:
        if path.is_dir() and not marker.exists():
            atomic_json(marker, identity)
        raise


@contextlib.contextmanager
def exclusive_lock(path):
    """OS locks release on process exit, so crashes do not leave stale locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Linked lock file: {path}")
    stream = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError(f"Another process is using {path}") from None
    try:
        yield
    finally:
        if os.name == "nt":
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def tree_bytes(root):
    # Quota inventory is read-only: count links themselves, never traverse them.
    total = 0
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in files:
            try:
                total += (Path(parent) / name).lstat().st_size
            except FileNotFoundError:
                # Other supervised tasks atomically replace tiny control records.
                # This is a quota estimate with a margin, not a filesystem snapshot.
                continue
        for name in dirs:
            path = Path(parent) / name
            if path.is_symlink():
                total += path.lstat().st_size
    return total


def require_space(target, needed_bytes, quota_root=None, quota_bytes=None):
    target = Path(target)
    existing = target
    while not existing.exists():
        existing = existing.parent
    available = shutil.disk_usage(existing).free
    used = None
    if quota_bytes is not None:
        if quota_root is None or not Path(quota_root).is_dir():
            raise ValueError("quota_root must exist when a personal quota is supplied")
        used = tree_bytes(quota_root)
        available = min(available, int(quota_bytes) - used)
    report = {"available_bytes": available, "needed_bytes": int(needed_bytes),
              "managed_quota_usage_bytes": used,
              "quota_note": "Local tree size is an estimate; provider quota may include other files/snapshots."}
    if available < needed_bytes:
        raise RuntimeError(f"Insufficient storage budget: {report}")
    return report
