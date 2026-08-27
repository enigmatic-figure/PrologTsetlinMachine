from __future__ import annotations

from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")


def _namespace() -> dict[str, object]:
    project = Path(__file__).resolve().parents[2]
    return runpy.run_path(
        str(
            project
            / "benchmarks"
            / "initial_capacity"
            / "run_mnist_jit_distillation.py"
        ),
        run_name="mnist_jit_distillation_test",
    )


def test_softmax_temperature_produces_normalized_probabilities() -> None:
    softmax = _namespace()["_softmax"]
    probabilities = softmax(
        np.asarray([[1000.0, 999.0], [-1000.0, -999.0]]),
        2.0,
    )

    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(np.isfinite(probabilities))
    assert probabilities[0, 0] > probabilities[0, 1]
    assert probabilities[1, 1] > probabilities[1, 0]


def test_paired_counts_collective_fixes_and_regressions() -> None:
    paired = _namespace()["_paired"]
    result = paired(
        np.asarray([0, 1, 2, 3]),
        np.asarray([0, 0, 2, 2]),
        np.asarray([1, 1, 2, 2]),
    )

    assert result == {"fixes": 1, "regressions": 1, "disagreements": 2}


def test_classification_uses_collective_argmax() -> None:
    classification = _namespace()["_classification"]
    truth = np.asarray([0, 1, 2])
    scores = np.full((3, 10), -4, dtype=np.int64)
    scores[0, 0] = 3
    scores[1, 1] = 2
    scores[2, 4] = 5

    result = classification(truth, scores)

    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["predictions"].tolist() == [0, 1, 4]


def test_resume_plan_fails_closed_on_configuration_change(tmp_path: Path) -> None:
    initialize = _namespace()["_initialize_or_validate_plan"]
    first = {"schema": "test", "configuration": {"seed": 1}}
    changed = {"schema": "test", "configuration": {"seed": 2}}

    initialize(tmp_path / "run", first, resume=False)
    initialize(tmp_path / "run", first, resume=True)

    with pytest.raises(ValueError, match="does not match"):
        initialize(tmp_path / "run", changed, resume=True)


def test_cached_pta_cell_is_bound_to_parent_snapshot() -> None:
    namespace = _namespace()
    validate = namespace["_validate_cached_cell"]
    machine_class = namespace["ScalarBinaryTsetlinMachine"]
    envelope_class = namespace["AdaptiveSnapshotEnvelope"]
    snapshot = machine_class(4, 2, seed=7).snapshot()
    value = {
        "schema": "ptm.mnist-pta-scout.v1",
        "config": {
            "target_digit": 3,
            "seed": 11,
            "features": 2,
            "clauses": 4,
            "parent_epochs": 5,
            "selected_pixels": [0, 1],
            "test_rows": 2,
            "parent_snapshot_id": envelope_class(snapshot).snapshot_id,
        },
        "score_vectors": {"multiclass_truth": [3, 8]},
    }
    arguments = {
        "digit": 3,
        "seed": 11,
        "features": 2,
        "clauses": 4,
        "selected_epoch": 5,
        "pixels": (0, 1),
        "audit_rows": 2,
        "snapshot": snapshot,
    }

    assert validate(value, **arguments) is value
    with pytest.raises(RuntimeError, match="campaign slot"):
        validate(value, **{**arguments, "seed": 12})

    different_snapshot = machine_class(4, 2, seed=8).snapshot()
    with pytest.raises(RuntimeError, match="snapshot identity"):
        validate(value, **{**arguments, "snapshot": different_snapshot})


def test_cli_preflight_rejects_nonfinite_and_empty_budgets() -> None:
    validate = _namespace()["_validate_cli_parameters"]
    valid = {
        "epochs": 2,
        "teacher_epochs": 1,
        "features": 12,
        "clauses": 40,
        "validation_rows": 100,
        "pta_audit_rows": 100,
        "teacher_temperature": 2.0,
        "student_temperature": 3.0,
        "residual_learning_rate": 2.0,
    }
    validate(SimpleNamespace(**valid))

    for key, value in (
        ("teacher_epochs", 0),
        ("clauses", 0),
        ("validation_rows", 0),
        ("pta_audit_rows", 0),
        ("teacher_temperature", float("nan")),
        ("student_temperature", 0.0),
        ("residual_learning_rate", -1.0),
    ):
        with pytest.raises(ValueError):
            validate(SimpleNamespace(**{**valid, key: value}))


def test_json_artifacts_reject_nonfinite_values(tmp_path: Path) -> None:
    write_json = _namespace()["_write_json"]
    path = tmp_path / "result.json"

    with pytest.raises(ValueError):
        write_json(path, {"accuracy": float("nan")})

    assert not path.exists()
