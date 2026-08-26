"""Adversarial contracts for PTM's shared subprocess containment boundary."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pytest

from prolog_tsetlin._bounded_process import (
    BoundedProcessOutputLimit,
    BoundedProcessTimeout,
    run_bounded_process,
)


def test_non_finite_deadline_is_rejected_before_launch() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        run_bounded_process(
            [sys.executable, "-c", "pass"],
            timeout_seconds=math.nan,
            max_output_bytes=1_024,
        )


def _descendant_parent(
    *,
    parent_sleep: float,
    survivor_marker: Path | None = None,
) -> str:
    if survivor_marker is None:
        descendant = "import time; time.sleep(5)"
    else:
        descendant = (
            "import time; from pathlib import Path; time.sleep(0.3); "
            f"Path({str(survivor_marker)!r}).write_text('survived', encoding='utf-8')"
        )
    return (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}], "
        "stdout=sys.stdout, stderr=sys.stderr); "
        "print('parent-ready', flush=True); "
        f"time.sleep({parent_sleep!r})"
    )


def test_completed_parent_cannot_leave_a_descendant_holding_pipes(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "escaped-descendant.txt"
    started = time.monotonic()

    completed = run_bounded_process(
        [
            sys.executable,
            "-c",
            _descendant_parent(
                parent_sleep=0.05,
                survivor_marker=marker,
            ),
        ],
        timeout_seconds=1.0,
        max_output_bytes=1_024,
    )

    assert time.monotonic() - started < 1.0
    assert completed.returncode == 0
    assert b"parent-ready" in completed.stdout
    time.sleep(0.5)
    assert not marker.exists()


def test_timeout_terminates_the_contained_process_tree() -> None:
    started = time.monotonic()

    with pytest.raises(BoundedProcessTimeout):
        run_bounded_process(
            [sys.executable, "-c", _descendant_parent(parent_sleep=5.0)],
            timeout_seconds=0.5,
            max_output_bytes=1_024,
        )

    assert time.monotonic() - started < 1.0


def test_nonisolated_leaf_remains_inside_outer_process_tree(tmp_path: Path) -> None:
    started_marker = tmp_path / "started-leaf.txt"
    marker = tmp_path / "escaped-leaf.txt"
    leaf = (
        "import time; from pathlib import Path; "
        f"Path({str(started_marker)!r}).write_text('started', encoding='utf-8'); "
        "time.sleep(2); "
        f"Path({str(marker)!r}).write_text('escaped', encoding='utf-8')"
    )
    parent = (
        "import sys; "
        "from prolog_tsetlin._bounded_process import run_bounded_process; "
        "run_bounded_process([sys.executable, '-c', "
        f"{leaf!r}], timeout_seconds=2, max_output_bytes=1024, "
        "isolate_process_tree=False)"
    )

    with pytest.raises(BoundedProcessTimeout):
        run_bounded_process(
            [sys.executable, "-c", parent],
            timeout_seconds=1.0,
            max_output_bytes=1_024,
        )

    assert started_marker.is_file()
    time.sleep(1.2)
    assert not marker.exists()


def test_output_flood_is_capped_before_the_child_can_exhaust_memory() -> None:
    with pytest.raises(BoundedProcessOutputLimit) as captured:
        run_bounded_process(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'x' * 1000000); "
                "sys.stdout.buffer.flush()",
            ],
            timeout_seconds=1.0,
            max_output_bytes=4_096,
        )

    assert len(captured.value.stdout) + len(captured.value.stderr) == 4_096
