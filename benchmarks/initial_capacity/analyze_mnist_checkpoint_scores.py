"""Compare raw-vote and clipped-vote ranking from retained MNIST checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Mapping, Sequence

import numpy as np

from prolog_tsetlin.services._atomic import publish_bytes

from run_mnist_jit_distillation import (
    CLASS_COUNT,
    _canonical_json,
    _classification,
    _clip_collective_scores,
    _machine_from_snapshot,
    _paired,
    _prepare_bank,
    _score_collective,
)


SCHEMA = "ptm.mnist-checkpoint-score-analysis.v1"


def _metrics(
    truth: np.ndarray,
    raw_scores: np.ndarray,
    clipped_scores: np.ndarray,
) -> dict[str, object]:
    raw = _classification(truth, raw_scores)
    clipped = _classification(truth, clipped_scores)
    raw_predictions = raw.pop("predictions")
    clipped_predictions = clipped.pop("predictions")
    return {
        "raw_vote": raw,
        "clipped_vote": clipped,
        "clipped_paired_against_raw": _paired(
            truth, raw_predictions, clipped_predictions
        ),
        "prediction_disagreements": int(
            np.count_nonzero(raw_predictions != clipped_predictions)
        ),
        "raw_top_ties": int(
            np.count_nonzero(
                np.sum(raw_scores == raw_scores.max(axis=1, keepdims=True), axis=1)
                > 1
            )
        ),
        "clipped_top_ties": int(
            np.count_nonzero(
                np.sum(
                    clipped_scores == clipped_scores.max(axis=1, keepdims=True),
                    axis=1,
                )
                > 1
            )
        ),
        "clipped_cells": int(np.count_nonzero(raw_scores != clipped_scores)),
        "score_cells": int(raw_scores.size),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/mnist.pkl"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    source = args.source.expanduser().resolve()
    experiment = args.experiment.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        parser.error("MNIST source does not exist")
    result_path = experiment / "result.json"
    if not result_path.is_file():
        parser.error("experiment has no top-level result.json")
    if output.exists():
        parser.error("output already exists")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = result.get("config")
    arms = result.get("arms")
    if not isinstance(config, Mapping) or not isinstance(arms, Mapping):
        parser.error("experiment result has an unsupported shape")
    seed = int(config["seed"])
    features = int(config["features_per_bank"])
    validation_rows = int(config["validation_rows"])

    train, validation, test = pickle.loads(source.read_bytes(), encoding="latin1")
    train_x, train_y = np.asarray(train[0]), np.asarray(train[1], dtype=np.int64)
    validation_x = np.asarray(validation[0])
    validation_y = np.asarray(validation[1], dtype=np.int64)
    test_x, test_y = np.asarray(test[0]), np.asarray(test[1], dtype=np.int64)
    banks = [
        _prepare_bank(
            train_x,
            train_y,
            digit=digit,
            seed=seed,
            features=features,
        )
        for digit in range(CLASS_COUNT)
    ]

    report_arms: dict[str, object] = {}
    for arm_name, stored_metrics in arms.items():
        if not isinstance(stored_metrics, Mapping):
            parser.error(f"arm {arm_name} metrics are malformed")
        snapshots = []
        for digit in range(CLASS_COUNT):
            checkpoint = experiment / "checkpoints" / arm_name / f"digit-{digit}.pkl"
            if not checkpoint.is_file():
                parser.error(f"missing checkpoint: {checkpoint}")
            snapshots.append(pickle.loads(checkpoint.read_bytes()))
        machines = [_machine_from_snapshot(snapshot) for snapshot in snapshots]
        split_metrics: dict[str, object] = {}
        for split_name, values, truth in (
            (
                "validation",
                validation_x[:validation_rows],
                validation_y[:validation_rows],
            ),
            ("test", test_x, test_y),
        ):
            raw_scores, _ = _score_collective(machines, banks, values)
            clipped_scores = _clip_collective_scores(raw_scores, machines)
            split_metrics[split_name] = _metrics(
                truth, raw_scores, clipped_scores
            )
        reported_accuracy = float(stored_metrics["accuracy"])
        reconstructed_clipped_accuracy = float(
            split_metrics["test"]["clipped_vote"]["accuracy"]
        )
        if reported_accuracy != reconstructed_clipped_accuracy:
            raise RuntimeError(
                f"arm {arm_name} retained checkpoint does not reproduce its "
                "reported clipped-vote accuracy"
            )
        report_arms[arm_name] = {
            "selected_epoch": int(stored_metrics["selected_epoch"]),
            "reported_clipped_accuracy_reproduced": True,
            **split_metrics,
        }

    report = {
        "schema": SCHEMA,
        "source_experiment": str(experiment),
        "analysis": (
            "No retraining. Reconstructed the deterministic bank projections and "
            "evaluated the retained selected checkpoints with raw and margin-clipped "
            "signed clause votes."
        ),
        "arms": report_arms,
    }
    publish_bytes(output, _canonical_json(report), overwrite=False)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
