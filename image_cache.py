"""Plugin-data image cache. Original bytes are retained; cleanup never follows links."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
from collections.abc import Iterable
from pathlib import Path

_LOCK = threading.Lock()
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".avif"}
SEND_GRACE_SECONDS = 300


def save_image(root: Path, filename: str, content: bytes) -> Path:
    """Atomically cache bytes by content hash, refreshing retention on reuse."""
    suffix = Path(str(filename).replace("\\", "/")).suffix.lower()
    if suffix not in _IMAGE_SUFFIXES:
        suffix = ".png"
    with _LOCK:
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(content).hexdigest()
        target = root / (digest + suffix)
        if target.is_symlink():
            raise ValueError("缓存目标不能为符号链接")
        if (
            target.is_file()
            and hashlib.sha256(target.read_bytes()).hexdigest() == digest
        ):
            target.touch()
            return target
        fd, name = tempfile.mkstemp(dir=root, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, target)
        finally:
            Path(name).unlink(missing_ok=True)
        return target


def manage_cache(
    root: Path,
    *,
    action: str = "status",
    days: int = 7,
    max_mb: int = 1024,
    now: float | None = None,
    protected_paths: Iterable[str | Path] = (),
) -> dict:
    """Inspect or prune image files only; protect recently saved images during sending."""
    if action not in {"status", "clear", "expired"}:
        raise ValueError("未知缓存操作")
    now = time.time() if now is None else now
    removed = freed = protected = errors = 0
    with _LOCK:
        root = root.resolve()
        pinned = {Path(path).resolve() for path in protected_paths if path}
        rows = []
        for path in root.iterdir() if root.exists() else []:
            try:
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.suffix.lower() not in _IMAGE_SUFFIXES
                ):
                    continue
                stat = path.stat()
                rows.append((stat.st_mtime, path, stat.st_size))
            except OSError:
                errors += 1
        total = sum(row[2] for row in rows)
        for modified, path, size in sorted(rows):
            if path in pinned or now - modified < SEND_GRACE_SECONDS:
                protected += 1
                continue
            if action == "status":
                continue
            expired = days > 0 and now - modified >= days * 86400
            over_limit = max_mb > 0 and total > max_mb * 1024 * 1024
            if action == "clear" or expired or over_limit:
                try:
                    path.unlink()
                except OSError:
                    errors += 1
                    continue
                removed += 1
                freed += size
                total -= size
        return {
            "files": len(rows) - removed,
            "bytes": total,
            "removed": removed,
            "freed": freed,
            "protected": protected,
            "errors": errors,
        }
