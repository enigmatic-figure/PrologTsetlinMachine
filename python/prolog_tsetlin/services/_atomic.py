"""Atomic publication helpers for fully materialized service artifacts."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def _sync_parent_directory(target: Path) -> None:
    """Persist a directory entry update on platforms that expose directory fsync."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(target.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_bytes(
    destination: str | Path,
    data: bytes,
    *,
    overwrite: bool,
) -> Path:
    """Publish complete bytes atomically, optionally replacing the destination."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.tmp."
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
            temporary.unlink()
        _sync_parent_directory(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
