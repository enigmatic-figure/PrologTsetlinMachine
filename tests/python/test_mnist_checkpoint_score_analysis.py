from __future__ import annotations

from pathlib import Path
import runpy
import sys

import pytest


def _namespace(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    project = Path(__file__).resolve().parents[2]
    benchmark = project / "benchmarks" / "initial_capacity"
    monkeypatch.setattr(sys, "path", [str(benchmark), *sys.path])
    return runpy.run_path(
        str(benchmark / "analyze_mnist_checkpoint_scores.py"),
        run_name="mnist_checkpoint_score_analysis_test",
    )


def _reconstructed(*, raw: float, clipped: float) -> dict[str, object]:
    return {
        "raw_vote": {"accuracy": raw},
        "clipped_vote": {"accuracy": clipped},
    }


def test_legacy_schema_reproduces_reported_clipped_accuracy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate = _namespace(monkeypatch)["_validate_reported_metrics"]
    result = validate(
        "ptm.mnist-jit-distillation.v1",
        "A",
        {"accuracy": 0.6},
        _reconstructed(raw=0.7, clipped=0.6),
    )

    assert result == {
        "reported_accuracy_semantics": "margin-clipped signed clause votes",
        "reported_accuracy_reproduced": True,
    }


def test_raw_vote_schema_reproduces_primary_and_clipped_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate = _namespace(monkeypatch)["_validate_reported_metrics"]
    result = validate(
        "ptm.mnist-jit-distillation.v2",
        "B",
        {"accuracy": 0.7, "clipped_vote_comparison": {"accuracy": 0.6}},
        _reconstructed(raw=0.7, clipped=0.6),
    )

    assert result["reported_accuracy_semantics"] == "unclipped signed clause votes"
    assert result["reported_accuracy_reproduced"] is True
    assert result["stored_clipped_comparison_reproduced"] is True


@pytest.mark.parametrize(
    "schema",
    (
        "ptm.mnist-jit-distillation.v3",
        "ptm.mnist-jit-distillation.v4",
        "ptm.mnist-jit-distillation.v5",
    ),
)
def test_teacher_policy_schemas_use_raw_vote_reproduction_contract(
    monkeypatch: pytest.MonkeyPatch,
    schema: str,
) -> None:
    validate = _namespace(monkeypatch)["_validate_reported_metrics"]

    result = validate(
        schema,
        "D",
        {"accuracy": 0.7, "clipped_vote_comparison": {"accuracy": 0.6}},
        _reconstructed(raw=0.7, clipped=0.6),
    )

    assert result["reported_accuracy_semantics"] == "unclipped signed clause votes"
    assert result["stored_clipped_comparison_reproduced"] is True


@pytest.mark.parametrize(
    ("source_schema", "stored", "message"),
    (
        (
            "ptm.mnist-jit-distillation.v1",
            {"accuracy": 0.7},
            "legacy clipped-vote accuracy",
        ),
        (
            "ptm.mnist-jit-distillation.v2",
            {"accuracy": 0.6, "clipped_vote_comparison": {"accuracy": 0.6}},
            "reported raw-vote accuracy",
        ),
        (
            "ptm.mnist-jit-distillation.v2",
            {"accuracy": 0.7, "clipped_vote_comparison": {"accuracy": 0.5}},
            "stored clipped-vote comparison",
        ),
    ),
)
def test_schema_specific_reproduction_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    source_schema: str,
    stored: dict[str, object],
    message: str,
) -> None:
    validate = _namespace(monkeypatch)["_validate_reported_metrics"]

    with pytest.raises(RuntimeError, match=message):
        validate(
            source_schema,
            "C",
            stored,
            _reconstructed(raw=0.7, clipped=0.6),
        )


def test_unknown_source_schema_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate = _namespace(monkeypatch)["_validate_reported_metrics"]

    with pytest.raises(RuntimeError, match="unsupported source experiment schema"):
        validate(
            "ptm.mnist-jit-distillation.v99",
            "A",
            {"accuracy": 0.7},
            _reconstructed(raw=0.7, clipped=0.6),
        )


def test_exact_retained_vector_validation_rejects_same_accuracy_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate = _namespace(monkeypatch)["_validate_retained_array"]
    retained = {"arm_predictions": pytest.importorskip("numpy").asarray([0, 1, 0, 1])}

    validate(retained, "arm_predictions", pytest.importorskip("numpy").asarray([0, 1, 0, 1]))
    with pytest.raises(RuntimeError, match="retained vector mismatch"):
        validate(
            retained,
            "arm_predictions",
            pytest.importorskip("numpy").asarray([1, 0, 1, 0]),
        )
