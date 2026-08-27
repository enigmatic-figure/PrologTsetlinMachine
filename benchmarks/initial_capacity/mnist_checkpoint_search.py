#!/usr/bin/env python3
"""Search independently checkpointed MNIST one-vs-rest classifier banks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
import struct
from typing import Iterator, Sequence

import numpy as np


SCORE_MAGIC = b"PTMSCORE"
SCORE_VERSION = 1
CLASS_COUNT = 10
RESULT_SCHEMA = "ptm.mnist-checkpoint-search.v1"


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ScoreCheckpoint:
    epoch: int
    labels: np.ndarray
    scores: np.ndarray
    path: Path
    digest: str


def load_score_checkpoint(path: str | Path) -> ScoreCheckpoint:
    resolved = Path(path).expanduser().resolve()
    payload = resolved.read_bytes()
    if len(payload) < 24 or payload[:8] != SCORE_MAGIC:
        raise ValueError(f"invalid checkpoint score header: {resolved}")
    version, epoch, rows, classes = struct.unpack_from("<IIII", payload, 8)
    if version != SCORE_VERSION or epoch <= 0 or rows <= 0:
        raise ValueError(f"unsupported checkpoint score metadata: {resolved}")
    if classes != CLASS_COUNT:
        raise ValueError(f"checkpoint must contain {CLASS_COUNT} classes: {resolved}")
    expected = 24 + rows + rows * classes * 4
    if len(payload) != expected:
        raise ValueError(f"checkpoint score payload has the wrong size: {resolved}")
    labels = np.frombuffer(payload, dtype=np.uint8, count=rows, offset=24).copy()
    if np.any(labels >= classes):
        raise ValueError(f"checkpoint labels are outside the class range: {resolved}")
    scores = np.frombuffer(
        payload,
        dtype="<i4",
        count=rows * classes,
        offset=24 + rows,
    ).reshape(rows, classes).copy()
    labels.setflags(write=False)
    scores.setflags(write=False)
    return ScoreCheckpoint(epoch, labels, scores, resolved, _digest_file(resolved))


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    matrix = _confusion(labels, predictions)
    correct = int(np.trace(matrix))
    rows = int(labels.size)
    per_class = []
    for label in range(CLASS_COUNT):
        support = int(matrix[label].sum())
        predicted = int(matrix[:, label].sum())
        true_positive = int(matrix[label, label])
        alternatives = matrix[label].copy()
        alternatives[label] = -1
        confused_as = int(np.argmax(alternatives))
        confused_count = max(0, int(alternatives[confused_as]))
        per_class.append(
            {
                "class": label,
                "support": support,
                "correct": true_positive,
                "recall": true_positive / support if support else None,
                "precision": true_positive / predicted if predicted else None,
                "most_confused_as": confused_as if confused_count else None,
                "most_confused_count": confused_count,
            }
        )
    return {
        "rows": rows,
        "correct": correct,
        "accuracy": correct / rows,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def evaluate_schedule(
    checkpoints: dict[int, ScoreCheckpoint], schedule: Sequence[int]
) -> dict[str, object]:
    labels, predictions = _schedule_predictions(checkpoints, schedule)
    result = _metrics(labels, predictions)
    result["schedule"] = list(schedule)
    result["classifier_epoch_sum"] = sum(schedule)
    return result


def _schedule_predictions(
    checkpoints: dict[int, ScoreCheckpoint], schedule: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    if len(schedule) != CLASS_COUNT:
        raise ValueError(f"schedule must contain {CLASS_COUNT} checkpoint epochs")
    labels = next(iter(checkpoints.values())).labels
    selected = np.empty((labels.size, CLASS_COUNT), dtype=np.int32)
    for classifier, epoch in enumerate(schedule):
        try:
            checkpoint = checkpoints[epoch]
        except KeyError as error:
            raise ValueError(f"schedule uses unavailable epoch {epoch}") from error
        if not np.array_equal(checkpoint.labels, labels):
            raise ValueError("checkpoint label vectors differ")
        selected[:, classifier] = checkpoint.scores[:, classifier]
    predictions = np.argmax(selected, axis=1).astype(np.uint8)
    return labels, predictions


def _paired_comparison(
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> dict[str, object]:
    baseline_correct = baseline_predictions == labels
    candidate_correct = candidate_predictions == labels
    improvements = np.logical_and(~baseline_correct, candidate_correct)
    regressions = np.logical_and(baseline_correct, ~candidate_correct)
    return {
        "both_correct": int(np.count_nonzero(baseline_correct & candidate_correct)),
        "both_wrong": int(np.count_nonzero(~baseline_correct & ~candidate_correct)),
        "improvements": int(np.count_nonzero(improvements)),
        "regressions": int(np.count_nonzero(regressions)),
        "prediction_disagreements": int(
            np.count_nonzero(baseline_predictions != candidate_predictions)
        ),
        "improvements_by_true_class": [
            int(np.count_nonzero(improvements & (labels == label)))
            for label in range(CLASS_COUNT)
        ],
        "regressions_by_true_class": [
            int(np.count_nonzero(regressions & (labels == label)))
            for label in range(CLASS_COUNT)
        ],
    }


def search_schedules(
    checkpoints: dict[int, ScoreCheckpoint],
    allowed_epochs: Sequence[int],
    *,
    batch_size: int = 128,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    epochs = tuple(sorted(set(allowed_epochs)))
    if len(epochs) < 2 or any(epoch not in checkpoints for epoch in epochs):
        raise ValueError("schedule search requires at least two available checkpoints")
    reference_labels = checkpoints[epochs[0]].labels
    for epoch in epochs[1:]:
        if not np.array_equal(checkpoints[epoch].labels, reference_labels):
            raise ValueError("checkpoint label vectors differ")
    score_tensor = np.stack([checkpoints[epoch].scores for epoch in epochs])
    epoch_values = np.asarray(epochs, dtype=np.int64)
    baseline_schedule = (epochs[-1],) * CLASS_COUNT
    baseline = evaluate_schedule(checkpoints, baseline_schedule)
    baseline_correct = int(baseline["correct"])
    best_schedule: tuple[int, ...] | None = None
    best_correct = -1
    best_epoch_sum = 0
    improving = 0
    equal_or_better = 0
    configuration_count = len(epochs) ** CLASS_COUNT
    iterator = product(range(len(epochs)), repeat=CLASS_COUNT)
    while True:
        batch = list(_take(iterator, batch_size))
        if not batch:
            break
        choices = np.asarray(batch, dtype=np.intp)
        selected = np.empty(
            (len(batch), reference_labels.size, CLASS_COUNT), dtype=np.int32
        )
        for classifier in range(CLASS_COUNT):
            selected[:, :, classifier] = score_tensor[
                choices[:, classifier], :, classifier
            ]
        predictions = np.argmax(selected, axis=2)
        correct = np.count_nonzero(
            predictions == reference_labels[np.newaxis, :], axis=1
        )
        epoch_sums = epoch_values[choices].sum(axis=1)
        improving += int(np.count_nonzero(correct > baseline_correct))
        equal_or_better += int(np.count_nonzero(correct >= baseline_correct))
        for index, choice in enumerate(batch):
            schedule = tuple(epochs[value] for value in choice)
            candidate_correct = int(correct[index])
            candidate_sum = int(epoch_sums[index])
            if (
                candidate_correct > best_correct
                or (
                    candidate_correct == best_correct
                    and (
                        best_schedule is None
                        or candidate_sum < best_epoch_sum
                        or (
                            candidate_sum == best_epoch_sum
                            and schedule < best_schedule
                        )
                    )
                )
            ):
                best_schedule = schedule
                best_correct = candidate_correct
                best_epoch_sum = candidate_sum
    assert best_schedule is not None
    best = evaluate_schedule(checkpoints, best_schedule)
    labels, baseline_predictions = _schedule_predictions(
        checkpoints, baseline_schedule
    )
    _, best_predictions = _schedule_predictions(checkpoints, best_schedule)
    return {
        "allowed_epochs": list(epochs),
        "configuration_count": configuration_count,
        "baseline": baseline,
        "best": best,
        "best_vs_baseline": _paired_comparison(
            labels, baseline_predictions, best_predictions
        ),
        "validation_correct_gain": int(best["correct"]) - baseline_correct,
        "classifier_epoch_savings": sum(baseline_schedule) - best_epoch_sum,
        "configurations_improving_baseline": improving,
        "configurations_equal_or_better_than_baseline": equal_or_better,
    }


def _take(
    values: Iterator[tuple[int, ...]], count: int
) -> Iterator[tuple[int, ...]]:
    for _ in range(count):
        try:
            yield next(values)
        except StopIteration:
            return


def _load_family(
    capture_directory: Path, prefix: str, epochs: Sequence[int]
) -> dict[int, ScoreCheckpoint]:
    result = {
        epoch: load_score_checkpoint(
            capture_directory / f"{prefix}-epoch-{epoch}.ptms"
        )
        for epoch in epochs
    }
    labels = next(iter(result.values())).labels
    if any(not np.array_equal(item.labels, labels) for item in result.values()):
        raise ValueError(f"{prefix} checkpoint label vectors differ")
    return result


def _training_records(path: Path) -> dict[int, dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or type(value.get("epoch")) is not int:
            raise ValueError("training log contains an invalid epoch record")
        epoch = int(value["epoch"])
        if epoch in records:
            raise ValueError("training log contains a duplicate epoch")
        records[epoch] = value
    return records


def analyze_capture(
    capture_directory: str | Path,
    training_log: str | Path,
    checkpoints: Sequence[int],
) -> dict[str, object]:
    directory = Path(capture_directory).expanduser().resolve()
    log_path = Path(training_log).expanduser().resolve()
    epochs = tuple(sorted(set(checkpoints)))
    if not epochs:
        raise ValueError("at least one checkpoint epoch is required")
    validation = _load_family(directory, "validation", epochs)
    records = _training_records(log_path)
    conformance = []
    for epoch in epochs:
        direct = records.get(epoch)
        if direct is None:
            raise ValueError(f"training log omits checkpoint epoch {epoch}")
        cached = evaluate_schedule(validation, (epoch,) * CLASS_COUNT)
        if direct.get("confusion_matrix") != cached["confusion_matrix"]:
            raise ValueError(
                f"cached scores disagree with direct evaluation at epoch {epoch}"
            )
        conformance.append(
            {
                "epoch": epoch,
                "correct": cached["correct"],
                "accuracy": cached["accuracy"],
                "mismatches": 0,
            }
        )
    final_epoch = epochs[-1]
    single_substitutions = []
    final_baseline = evaluate_schedule(
        validation, (final_epoch,) * CLASS_COUNT
    )
    for earlier in epochs[:-1]:
        for classifier in range(CLASS_COUNT):
            schedule = [final_epoch] * CLASS_COUNT
            schedule[classifier] = earlier
            result = evaluate_schedule(validation, schedule)
            single_substitutions.append(
                {
                    "classifier": classifier,
                    "substitute_epoch": earlier,
                    "correct": result["correct"],
                    "accuracy": result["accuracy"],
                    "correct_delta": int(result["correct"])
                    - int(final_baseline["correct"]),
                }
            )
    tracks = []
    if len(epochs) >= 2:
        tracks.append(search_schedules(validation, (epochs[0], final_epoch)))
    if len(epochs) > 2:
        tracks.append(search_schedules(validation, (epochs[-2], final_epoch)))
        tracks.append(search_schedules(validation, epochs))
    selected_schedule = (
        tuple(tracks[-1]["best"]["schedule"])
        if tracks
        else (final_epoch,) * CLASS_COUNT
    )

    # Audit tensors are not loaded until validation has fixed the schedule.
    audit = _load_family(directory, "audit", epochs)
    audit_baseline = evaluate_schedule(audit, (final_epoch,) * CLASS_COUNT)
    audit_selected = evaluate_schedule(audit, selected_schedule)
    audit_labels, audit_baseline_predictions = _schedule_predictions(
        audit, (final_epoch,) * CLASS_COUNT
    )
    _, audit_selected_predictions = _schedule_predictions(audit, selected_schedule)
    audit_tracks = []
    for track in tracks:
        schedule = tuple(track["best"]["schedule"])
        evaluated = evaluate_schedule(audit, schedule)
        _, predictions = _schedule_predictions(audit, schedule)
        audit_tracks.append(
            {
                "allowed_epochs": track["allowed_epochs"],
                "validation_selected_schedule": list(schedule),
                "evaluation": evaluated,
                "correct_gain": int(evaluated["correct"])
                - int(audit_baseline["correct"]),
                "paired_against_uniform_final": _paired_comparison(
                    audit_labels, audit_baseline_predictions, predictions
                ),
            }
        )
    return {
        "schema": RESULT_SCHEMA,
        "checkpoint_epochs": list(epochs),
        "score_files": {
            "validation": [
                {
                    "epoch": epoch,
                    "path": str(validation[epoch].path),
                    "digest": validation[epoch].digest,
                }
                for epoch in epochs
            ],
            "audit": [
                {
                    "epoch": epoch,
                    "path": str(audit[epoch].path),
                    "digest": audit[epoch].digest,
                }
                for epoch in epochs
            ],
            "training_log": {
                "path": str(log_path),
                "digest": _digest_file(log_path),
            },
        },
        "cached_score_conformance": conformance,
        "validation_uniform_final": final_baseline,
        "single_classifier_substitutions": single_substitutions,
        "validation_searches": tracks,
        "selection": {
            "basis": (
                "validation only" if tracks else "fixed final checkpoint"
            ),
            "schedule": list(selected_schedule),
            "classifier_epoch_sum": sum(selected_schedule),
        },
        "audit": {
            "opened_after_selection": True,
            "uniform_final": audit_baseline,
            "selected": audit_selected,
            "correct_gain": int(audit_selected["correct"])
            - int(audit_baseline["correct"]),
            "classifier_epoch_savings": CLASS_COUNT * final_epoch
            - sum(selected_schedule),
            "paired_against_uniform_final": _paired_comparison(
                audit_labels,
                audit_baseline_predictions,
                audit_selected_predictions,
            ),
            "validation_selected_tracks": audit_tracks,
        },
        "claim_boundary": (
            "Checkpoint selection measures classifier-specific adaptation maturity; "
            "it does not establish reduced clause capacity."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("training_log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoints", nargs="+", type=int, required=True)
    args = parser.parse_args(argv)
    result = analyze_capture(
        args.capture_directory, args.training_log, args.checkpoints
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
