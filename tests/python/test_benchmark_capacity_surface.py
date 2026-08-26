from __future__ import annotations

from pathlib import Path
import runpy


def _namespace() -> dict[str, object]:
    project = Path(__file__).resolve().parents[2]
    return runpy.run_path(
        str(
            project
            / "benchmarks"
            / "initial_capacity"
            / "summarize_capacity_surface.py"
        ),
        run_name="benchmark_capacity_surface_test",
    )


def test_xor_state_margins_are_oriented_toward_clean_truth() -> None:
    summarize = _namespace()["_state_margin_metrics"]

    metrics = summarize(
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        (-2, 3, 0, 1),
    )

    assert metrics["state_00_mean_correct_margin"] == 2
    assert metrics["state_01_mean_correct_margin"] == 3
    assert metrics["state_10_mean_correct_margin"] == 0
    assert metrics["state_11_mean_correct_margin"] == -1
    assert metrics["state_10_tie_fraction"] == 1
    assert metrics["state_11_error_fraction"] == 1


def test_one_standard_error_rule_prefers_smallest_eligible_capacity() -> None:
    select = _namespace()["_one_standard_error_selection"]
    cells = (
        {
            "clauses": 10,
            "metrics": {
                "clean_validation_accuracy": {
                    "mean": 0.86,
                    "stdev": 0.08,
                    "observations": 4,
                }
            },
        },
        {
            "clauses": 20,
            "metrics": {
                "clean_validation_accuracy": {
                    "mean": 0.90,
                    "stdev": 0.10,
                    "observations": 4,
                }
            },
        },
    )

    selection = select(cells, "clean_validation_accuracy")

    assert selection["best_mean_clauses"] == 20
    assert selection["acceptance_cutoff"] == 0.85
    assert selection["smallest_within_one_best_standard_error"] == 10
