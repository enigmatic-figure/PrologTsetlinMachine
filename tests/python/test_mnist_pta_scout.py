from __future__ import annotations

import json
from pathlib import Path
import runpy

import pytest

np = pytest.importorskip("numpy")


def _namespace() -> dict[str, object]:
    project = Path(__file__).resolve().parents[2]
    return runpy.run_path(
        str(
            project
            / "benchmarks"
            / "initial_capacity"
            / "run_mnist_pta_scout.py"
        ),
        run_name="mnist_pta_scout_test",
    )


def test_only_json_finds_one_content_addressed_object(tmp_path: Path) -> None:
    only_json = _namespace()["_only_json"]
    object_path = tmp_path / "ab" / "abcdef.json"
    object_path.parent.mkdir()
    object_path.write_text('{"observations": 7}\n', encoding="utf-8")

    assert only_json(tmp_path) == {"observations": 7}

    second = tmp_path / "cd" / "cdef.json"
    second.parent.mkdir()
    second.write_text("{}\n", encoding="utf-8")
    assert only_json(tmp_path) is None


def test_take_unique_skips_repeated_projected_records() -> None:
    take_unique = _namespace()["_take_unique"]
    values = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    labels = np.asarray([8, 8, 8, 3])
    used: set[tuple[int, tuple[float, ...]]] = set()

    selected, cursor = take_unique(
        np.asarray([0, 1, 2]),
        values,
        labels,
        (0, 1),
        target=8,
        count=2,
        used=used,
        start=0,
    )

    assert selected.tolist() == [0, 2]
    assert cursor == 3
    assert len(used) == 2


def test_only_json_requires_an_object_payload(tmp_path: Path) -> None:
    only_json = _namespace()["_only_json"]
    path = tmp_path / "aa" / "array.json"
    path.parent.mkdir()
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    assert only_json(tmp_path) is None
