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
            / "run_mnist_pta_convergence.py"
        ),
        run_name="mnist_pta_convergence_test",
    )


def test_aggregate_builds_multiclass_scores_and_timestep_accounting() -> None:
    aggregate = _namespace()["_aggregate"]
    truth = list(range(10))
    example_ids = list(range(70_000, 70_010))
    baseline = np.zeros((10, 10), dtype=np.int64)
    baseline[np.arange(10), np.arange(10)] = 10
    baseline[0, 0] = 5
    baseline[0, 1] = 11
    governed = baseline.copy()
    governed[0, 0] = 12
    cells = []
    for digit in range(10):
        cells.append(
            {
                "config": {
                    "parent_rows": 400,
                    "adaptation_rows": 160,
                    "adaptation_epochs": 5,
                },
                "score_vectors": {
                    "semantics": "unclipped signed clause votes",
                    "example_ids": example_ids,
                    "multiclass_truth": truth,
                    "binary_truth": [int(value == digit) for value in truth],
                    "baseline": baseline[:, digit].tolist(),
                    "policy_governed": governed[:, digit].tolist(),
                    "selected_child_counterfactual": governed[:, digit].tolist(),
                    "deescalated": baseline[:, digit].tolist(),
                },
                "pta_usage": {
                    "input": {"candidate_count": 2},
                    "escalation": {
                        "candidate_evaluations": 2,
                        "activated": digit == 0,
                        "error": None,
                    },
                    "deescalation": {"activated": True},
                },
                "timing_seconds": {"parent_training": 1.0},
            }
        )

    result = aggregate(5, cells)

    assert result["metrics"]["baseline"]["accuracy"] == 0.9
    assert result["metrics"]["policy_governed"]["accuracy"] == 1.0
    assert result["paired_against_baseline"]["policy_governed"] == {
        "improvements": 1,
        "regressions": 0,
        "prediction_disagreements": 1,
    }
    accounting = result["matched_timestep_accounting"]
    assert accounting["parent_training_observations_all_banks"] == 20_000
    assert accounting["candidate_models_evaluated"] == 20
    assert accounting["candidate_adaptation_observations"] == 16_000
    assert result["pta_usage"]["escalation_activated_digits"] == [0]
    assert result["pta_usage"]["deescalation_activated_digits"] == list(range(10))


def test_load_cell_rejects_legacy_or_unmarked_score_semantics(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    load_cell = namespace["_load_cell"]
    value = {
        "schema": "ptm.mnist-pta-scout.v2",
        "config": {"target_digit": 3, "parent_epochs": 5, "seed": 11},
        "score_vectors": {
            "semantics": "unclipped signed clause votes",
            "multiclass_truth": [3, 8],
        },
    }
    path = tmp_path / "result.json"

    path.write_text(json.dumps(value), encoding="utf-8")
    assert load_cell(path, digit=3, epoch=5, seed=11, audit_rows=2) == value

    value["score_vectors"].pop("semantics")
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="campaign slot"):
        load_cell(path, digit=3, epoch=5, seed=11, audit_rows=2)

    value["schema"] = "ptm.mnist-pta-scout.v1"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsupported schema"):
        load_cell(path, digit=3, epoch=5, seed=11, audit_rows=2)
