from __future__ import annotations

import json
from pathlib import Path
import runpy
import struct

import pytest

np = pytest.importorskip("numpy")


def _namespace() -> dict[str, object]:
    project = Path(__file__).resolve().parents[2]
    return runpy.run_path(
        str(
            project
            / "benchmarks"
            / "initial_capacity"
            / "mnist_checkpoint_search.py"
        ),
        run_name="mnist_checkpoint_search_test",
    )


def _write_scores(
    path: Path, epoch: int, labels: np.ndarray, scores: np.ndarray
) -> None:
    payload = bytearray(b"PTMSCORE")
    payload.extend(struct.pack("<IIII", 1, epoch, len(labels), 10))
    payload.extend(np.asarray(labels, dtype=np.uint8).tobytes())
    payload.extend(np.asarray(scores, dtype="<i4").tobytes())
    path.write_bytes(payload)


def _predict(scores: np.ndarray) -> np.ndarray:
    return np.argmax(scores, axis=1).astype(np.uint8)


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> list[list[int]]:
    result = np.zeros((10, 10), dtype=np.int64)
    np.add.at(result, (labels, predictions), 1)
    return result.tolist()


def _base_scores(labels: np.ndarray) -> np.ndarray:
    result = np.zeros((len(labels), 10), dtype=np.int32)
    result[np.arange(len(labels)), labels] = 10
    return result


def test_search_selects_classifier_specific_checkpoint(tmp_path: Path) -> None:
    namespace = _namespace()
    load = namespace["load_score_checkpoint"]
    search = namespace["search_schedules"]
    labels = np.arange(10, dtype=np.uint8)
    final = _base_scores(labels)
    final[0, 1] = 11
    early = final.copy()
    early[0, 0] = 12
    for classifier in range(1, 10):
        early[classifier, classifier] = -1

    early_path = tmp_path / "early.ptms"
    final_path = tmp_path / "final.ptms"
    _write_scores(early_path, 10, labels, early)
    _write_scores(final_path, 30, labels, final)
    checkpoints = {10: load(early_path), 30: load(final_path)}

    result = search(checkpoints, (10, 30), batch_size=7)

    assert result["configuration_count"] == 1024
    assert result["baseline"]["correct"] == 9
    assert result["best"]["correct"] == 10
    assert result["best"]["schedule"] == [10] + [30] * 9
    assert result["classifier_epoch_savings"] == 20
    assert result["best_vs_baseline"]["improvements"] == 1
    assert result["best_vs_baseline"]["regressions"] == 0


def test_analysis_selects_on_validation_then_opens_audit(tmp_path: Path) -> None:
    namespace = _namespace()
    analyze = namespace["analyze_capture"]
    labels = np.arange(10, dtype=np.uint8)
    validation_final = _base_scores(labels)
    validation_final[0, 1] = 11
    validation_early = validation_final.copy()
    validation_early[0, 0] = 12
    for classifier in range(1, 10):
        validation_early[classifier, classifier] = -1
    validation_middle = validation_final.copy()

    audit_final = _base_scores(labels)
    audit_early = audit_final.copy()
    audit_early[1, 0] = 12
    audit_middle = audit_final.copy()
    scores = tmp_path / "scores"
    scores.mkdir()
    for prefix, family in (
        (
            "validation",
            {10: validation_early, 20: validation_middle, 30: validation_final},
        ),
        ("audit", {10: audit_early, 20: audit_middle, 30: audit_final}),
    ):
        for epoch, values in family.items():
            _write_scores(scores / f"{prefix}-epoch-{epoch}.ptms", epoch, labels, values)

    log = tmp_path / "training.jsonl"
    with log.open("w", encoding="utf-8") as stream:
        for epoch, values in (
            (10, validation_early),
            (20, validation_middle),
            (30, validation_final),
        ):
            stream.write(
                json.dumps(
                    {
                        "schema": "ptm.mnist-ovr-epoch.v1",
                        "epoch": epoch,
                        "confusion_matrix": _confusion(labels, _predict(values)),
                    }
                )
                + "\n"
            )

    result = analyze(scores, log, (10, 20, 30))

    assert result["selection"]["basis"] == "validation only"
    assert result["selection"]["schedule"] == [10] + [20] * 9
    assert result["validation_searches"][-1]["best"]["correct"] == 10
    assert result["audit"]["uniform_final"]["correct"] == 10
    assert result["audit"]["selected"]["correct"] == 9
    assert result["audit"]["correct_gain"] == -1
    assert result["audit"]["paired_against_uniform_final"]["improvements"] == 0
    assert result["audit"]["paired_against_uniform_final"]["regressions"] == 1
    assert len(result["audit"]["validation_selected_tracks"]) == 3


def test_analysis_rejects_cached_scores_that_disagree_with_direct_eval(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    analyze = namespace["analyze_capture"]
    labels = np.arange(10, dtype=np.uint8)
    values = _base_scores(labels)
    scores = tmp_path / "scores"
    scores.mkdir()
    for prefix in ("validation", "audit"):
        for epoch in (10, 30):
            _write_scores(scores / f"{prefix}-epoch-{epoch}.ptms", epoch, labels, values)
    log = tmp_path / "training.jsonl"
    records = [
        {"epoch": 10, "confusion_matrix": _confusion(labels, _predict(values))},
        {"epoch": 30, "confusion_matrix": [[0] * 10 for _ in range(10)]},
    ]
    log.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cached scores disagree"):
        analyze(scores, log, (10, 30))


def test_analysis_accepts_one_fixed_final_checkpoint(tmp_path: Path) -> None:
    namespace = _namespace()
    analyze = namespace["analyze_capture"]
    labels = np.arange(10, dtype=np.uint8)
    values = _base_scores(labels)
    scores = tmp_path / "scores"
    scores.mkdir()
    for prefix in ("validation", "audit"):
        _write_scores(scores / f"{prefix}-epoch-30.ptms", 30, labels, values)
    log = tmp_path / "training.jsonl"
    log.write_text(
        json.dumps(
            {
                "epoch": 30,
                "confusion_matrix": _confusion(labels, _predict(values)),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = analyze(scores, log, (30,))

    assert result["selection"] == {
        "basis": "fixed final checkpoint",
        "schedule": [30] * 10,
        "classifier_epoch_sum": 300,
    }
    assert result["validation_searches"] == []
    assert result["audit"]["correct_gain"] == 0
