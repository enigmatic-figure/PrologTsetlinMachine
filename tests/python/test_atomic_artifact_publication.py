from __future__ import annotations

import os
from pathlib import Path

import pytest

from prolog_tsetlin.services import _atomic


def _temporary_files(target: Path) -> list[Path]:
    return list(target.parent.glob(f".{target.name}.tmp.*"))


def test_atomic_no_overwrite_publication_succeeds_without_residue(
    tmp_path: Path,
) -> None:
    target = tmp_path / "model.ptm"

    _atomic.publish_bytes(target, b"complete artifact", overwrite=False)

    assert target.read_bytes() == b"complete artifact"
    assert _temporary_files(target) == []


def test_atomic_overwrite_publication_replaces_complete_bytes(tmp_path: Path) -> None:
    target = tmp_path / "model.ptm"
    target.write_bytes(b"original")

    _atomic.publish_bytes(target, b"complete replacement", overwrite=True)

    assert target.read_bytes() == b"complete replacement"
    assert _temporary_files(target) == []


def test_atomic_publication_preserves_no_overwrite_semantics(tmp_path: Path) -> None:
    target = tmp_path / "model.ptm"
    target.write_bytes(b"original")

    with pytest.raises(FileExistsError):
        _atomic.publish_bytes(target, b"replacement", overwrite=False)

    assert target.read_bytes() == b"original"
    assert _temporary_files(target) == []


def test_failed_atomic_overwrite_preserves_existing_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "model.ptm"
    target.write_bytes(b"original")

    def fail_replace(source, destination):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated publication failure"):
        _atomic.publish_bytes(target, b"replacement", overwrite=True)

    assert target.read_bytes() == b"original"
    assert _temporary_files(target) == []


def test_failed_no_overwrite_publication_leaves_no_partial_destination(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "model.ptm"

    def fail_link(source, destination):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="simulated publication failure"):
        _atomic.publish_bytes(target, b"complete artifact", overwrite=False)

    assert not target.exists()
    assert _temporary_files(target) == []


def test_atomic_publication_synchronizes_parent_directory(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "model.ptm"
    synchronized: list[Path] = []
    monkeypatch.setattr(
        _atomic,
        "_sync_parent_directory",
        lambda published: synchronized.append(published),
    )

    _atomic.publish_bytes(target, b"durable artifact", overwrite=False)

    assert synchronized == [target]
