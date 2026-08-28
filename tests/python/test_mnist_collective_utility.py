from __future__ import annotations

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
            / "analyze_mnist_collective_utility.py"
        ),
        run_name="mnist_collective_utility_test",
    )


def test_collective_search_selects_mask_then_confirms_it() -> None:
    analyze = _namespace()["_analyze_epoch"]
    truth = np.tile(np.arange(10, dtype=np.int64), 2)
    baseline = np.zeros((20, 10), dtype=np.int64)
    baseline[np.arange(20), truth] = 10
    baseline[[0, 10], 0] = 5
    baseline[[0, 10], 1] = 11
    candidate = baseline.copy()
    candidate[[0, 10], 0] = 12
    cells = []
    for digit in range(10):
        cells.append(
            {
                "score_vectors": {
                    "semantics": "unclipped signed clause votes",
                    "example_ids": list(range(20)),
                    "multiclass_truth": truth.tolist(),
                    "binary_truth": (truth == digit).astype(np.int64).tolist(),
                    "baseline": baseline[:, digit].tolist(),
                    "policy_governed": baseline[:, digit].tolist(),
                    "selected_child_counterfactual": candidate[:, digit].tolist(),
                }
            }
        )

    result = analyze(5, cells)

    assert result["search"]["configuration_count"] == 1024
    assert result["search"]["selected_mask"] == 1
    assert result["search"]["selected_digits"] == [0]
    assert result["search"]["selection_delta_utility"] == 1
    confirmation = result["partitions"]["confirmation"]
    assert confirmation["baseline"]["correct"] == 9
    assert confirmation["collective_utility_selected"]["correct"] == 10
    assert confirmation["collective_utility_selected"][
        "paired_against_baseline"
    ]["delta_utility"] == 1
