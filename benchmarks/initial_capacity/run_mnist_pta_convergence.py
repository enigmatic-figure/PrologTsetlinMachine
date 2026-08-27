#!/usr/bin/env python3
"""Run resumable ten-bank MNIST PTA checkpoint interventions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np


SCHEMA = "ptm.mnist-pta-convergence.v1"
CELL_SCHEMA = "ptm.mnist-pta-scout.v1"
CLASS_COUNT = 10


def _load_cell(
    path: Path,
    *,
    digit: int,
    epoch: int,
    seed: int,
    audit_rows: int,
) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != CELL_SCHEMA:
        raise RuntimeError(f"PTA cell has an unsupported schema: {path}")
    config = value.get("config")
    vectors = value.get("score_vectors")
    if (
        not isinstance(config, dict)
        or config.get("target_digit") != digit
        or config.get("parent_epochs") != epoch
        or config.get("seed") != seed
        or not isinstance(vectors, dict)
        or len(vectors.get("multiclass_truth", [])) != audit_rows
    ):
        raise RuntimeError(f"PTA cell does not match its campaign slot: {path}")
    return value


def _classification(truth: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    predictions = np.argmax(scores, axis=1)
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    np.add.at(confusion, (truth, predictions), 1)
    correct = int(np.count_nonzero(predictions == truth))
    return {
        "observations": len(truth),
        "correct": correct,
        "accuracy": correct / len(truth),
        "confusion_matrix": confusion.tolist(),
        "per_digit_recall": [
            (
                float(confusion[digit, digit] / confusion[digit].sum())
                if confusion[digit].sum()
                else None
            )
            for digit in range(CLASS_COUNT)
        ],
        "predictions": predictions,
    }


def _paired(
    truth: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, int]:
    baseline_correct = baseline == truth
    candidate_correct = candidate == truth
    return {
        "improvements": int(np.count_nonzero(~baseline_correct & candidate_correct)),
        "regressions": int(np.count_nonzero(baseline_correct & ~candidate_correct)),
        "prediction_disagreements": int(np.count_nonzero(baseline != candidate)),
    }


def _aggregate(epoch: int, cells: Sequence[Mapping[str, object]]) -> dict[str, object]:
    first_vectors = cells[0]["score_vectors"]
    assert isinstance(first_vectors, Mapping)
    truth = np.asarray(first_vectors["multiclass_truth"], dtype=np.int64)
    example_ids = first_vectors["example_ids"]
    score_names = (
        "baseline",
        "policy_governed",
        "selected_child_counterfactual",
        "deescalated",
    )
    matrices: dict[str, np.ndarray] = {}
    for name in score_names:
        columns: list[np.ndarray] = []
        for digit, cell in enumerate(cells):
            vectors = cell["score_vectors"]
            if not isinstance(vectors, Mapping):
                raise RuntimeError("PTA cell score vectors are malformed")
            if (
                vectors["example_ids"] != example_ids
                or vectors["multiclass_truth"] != truth.tolist()
            ):
                raise RuntimeError("PTA cells do not share one audit corpus")
            binary_truth = np.asarray(vectors["binary_truth"], dtype=np.int64)
            if not np.array_equal(binary_truth, (truth == digit).astype(np.int64)):
                raise RuntimeError("PTA cell binary labels disagree with multiclass truth")
            column = np.asarray(vectors[name], dtype=np.int64)
            if column.shape != truth.shape:
                raise RuntimeError("PTA cell score vector has the wrong shape")
            columns.append(column)
        matrices[name] = np.column_stack(columns)

    metrics = {name: _classification(truth, matrix) for name, matrix in matrices.items()}
    baseline_predictions = metrics["baseline"]["predictions"]
    paired = {
        name: _paired(truth, baseline_predictions, values["predictions"])
        for name, values in metrics.items()
        if name != "baseline"
    }
    for values in metrics.values():
        values.pop("predictions")

    candidate_counts = []
    candidate_evaluations = []
    activated_digits = []
    deescalated_digits = []
    errors: dict[str, object] = {}
    observed_wall = 0.0
    for digit, cell in enumerate(cells):
        usage = cell["pta_usage"]
        timing = cell["timing_seconds"]
        assert isinstance(usage, Mapping) and isinstance(timing, Mapping)
        input_usage = usage["input"]
        escalation = usage["escalation"]
        deescalation = usage["deescalation"]
        assert all(
            isinstance(item, Mapping)
            for item in (input_usage, escalation, deescalation)
        )
        candidate_counts.append(int(input_usage.get("candidate_count") or 0))
        candidate_evaluations.append(int(escalation["candidate_evaluations"]))
        if escalation["activated"]:
            activated_digits.append(digit)
        if deescalation["activated"]:
            deescalated_digits.append(digit)
        if escalation.get("error") is not None:
            errors[str(digit)] = escalation["error"]
        observed_wall += sum(float(value) for value in timing.values())

    config = cells[0]["config"]
    assert isinstance(config, Mapping)
    parent_rows = int(config["parent_rows"])
    adaptation_rows = int(config["adaptation_rows"])
    adaptation_epochs = int(config["adaptation_epochs"])
    return {
        "epoch": epoch,
        "matched_timestep_accounting": {
            "parent_epochs_per_bank": epoch,
            "parent_training_observations_per_bank": parent_rows * epoch,
            "parent_training_observations_all_banks": (
                parent_rows * epoch * CLASS_COUNT
            ),
            "candidate_models_evaluated": sum(candidate_evaluations),
            "candidate_adaptation_observations": (
                sum(candidate_evaluations) * adaptation_rows * adaptation_epochs
            ),
            "deployed_child_adaptation_observations_per_activated_bank": (
                adaptation_rows * adaptation_epochs
            ),
        },
        "metrics": metrics,
        "paired_against_baseline": paired,
        "pta_usage": {
            "input_threshold_candidates": candidate_counts,
            "input_threshold_candidates_total": sum(candidate_counts),
            "escalation_candidate_evaluations": candidate_evaluations,
            "escalation_activated_digits": activated_digits,
            "escalation_errors_by_digit": errors,
            "deescalation_activated_digits": deescalated_digits,
        },
        "observed_wall_seconds": observed_wall,
    }


def _write_result(
    path: Path, config: Mapping[str, object], checkpoints: Sequence[object]
) -> None:
    result = {
        "schema": SCHEMA,
        "claim_boundary": (
            "Each checkpoint is an independently reconstructed deterministic "
            "parent trajectory. Utility is compared by epochs and labeled "
            "update observations; wall time is observational only."
        ),
        "config": dict(config),
        "checkpoints": list(checkpoints),
    }
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("data/mnist.pkl"))
    parser.add_argument(
        "--ptmrt", type=Path, default=Path("out/build/Release/ptmrt.exe")
    )
    parser.add_argument("--checkpoints", nargs="+", type=int, default=(1, 5, 10, 20, 30))
    parser.add_argument("--digits", nargs="+", type=int, default=tuple(range(10)))
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--clauses", type=int, default=40)
    parser.add_argument("--adaptation-epochs", type=int, default=5)
    parser.add_argument("--audit-rows", type=int, default=2_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    checkpoints = tuple(sorted(set(args.checkpoints)))
    digits = tuple(sorted(set(args.digits)))
    if not checkpoints or any(value <= 0 for value in checkpoints):
        parser.error("checkpoints must be positive")
    if digits != tuple(range(CLASS_COUNT)):
        parser.error("a full run requires each digit 0..9 exactly once")
    if args.audit_rows <= 0 or args.audit_rows > 10_000:
        parser.error("--audit-rows must lie in 1..10000")
    output = args.output.expanduser().resolve()
    if output.exists() and not args.resume:
        parser.error("output already exists; pass --resume to continue it")
    output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[2]
    source = args.source.expanduser().resolve()
    ptmrt = args.ptmrt.expanduser().resolve()
    scout = Path(__file__).with_name("run_mnist_pta_scout.py")
    if not source.is_file() or not ptmrt.is_file():
        parser.error("MNIST source and ptmrt must exist")
    config = {
        "source": str(source),
        "ptmrt": str(ptmrt),
        "checkpoints": list(checkpoints),
        "digits": list(digits),
        "seed": args.seed,
        "features_per_bank": args.features,
        "clauses_per_bank": args.clauses,
        "adaptation_epochs": args.adaptation_epochs,
        "shared_audit_rows": args.audit_rows,
        "selection_basis": "independent per-bank promotion corpora",
        "audit_basis": "fixed shared leading MNIST test rows",
    }
    result_path = output / "result.json"
    aggregates: list[object] = []
    campaign_started = perf_counter()
    for epoch in checkpoints:
        cells: list[Mapping[str, object]] = []
        for digit in digits:
            cell = output / "cells" / f"epoch-{epoch}" / f"digit-{digit}"
            cell_result = cell / "result.json"
            if not cell_result.is_file():
                cell.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(scout),
                    "--source",
                    str(source),
                    "--output",
                    str(cell),
                    "--ptmrt",
                    str(ptmrt),
                    "--seed",
                    str(args.seed),
                    "--target-digit",
                    str(digit),
                    "--features",
                    str(args.features),
                    "--clauses",
                    str(args.clauses),
                    "--parent-epochs",
                    str(epoch),
                    "--adaptation-epochs",
                    str(args.adaptation_epochs),
                    "--shared-audit-rows",
                    str(args.audit_rows),
                    "--emit-score-vectors",
                ]
                environment = os.environ.copy()
                python_path = str(project / "python")
                environment["PYTHONPATH"] = os.pathsep.join(
                    item
                    for item in (python_path, environment.get("PYTHONPATH", ""))
                    if item
                )
                console = output / "cells" / f"epoch-{epoch}" / f"digit-{digit}.log"
                with console.open("w", encoding="utf-8") as stream:
                    completed = subprocess.run(
                        command,
                        cwd=project,
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"PTA cell epoch={epoch} digit={digit} failed; see {console}"
                    )
            cells.append(
                _load_cell(
                    cell_result,
                    digit=digit,
                    epoch=epoch,
                    seed=args.seed,
                    audit_rows=args.audit_rows,
                )
            )
            print(f"completed epoch={epoch} digit={digit}", flush=True)
        aggregates.append(_aggregate(epoch, cells))
        _write_result(result_path, config, aggregates)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "checkpoint_count": len(aggregates),
                "observed_campaign_wall_seconds": perf_counter() - campaign_started,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
