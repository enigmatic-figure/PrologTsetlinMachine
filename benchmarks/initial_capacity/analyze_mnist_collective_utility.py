#!/usr/bin/env python3
"""Evaluate bounded collective bank promotion over completed MNIST PTA cells."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCHEMA = "ptm.mnist-collective-utility-analysis.v2"
CELL_SCHEMA = "ptm.mnist-pta-scout.v2"
RAW_VOTE_SEMANTICS = "unclipped signed clause votes"
CLASS_COUNT = 10


def _stratified_alternating_split(truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seen = np.zeros(CLASS_COUNT, dtype=np.int64)
    selection: list[int] = []
    confirmation: list[int] = []
    for index, label in enumerate(truth):
        target = selection if seen[label] % 2 == 0 else confirmation
        target.append(index)
        seen[label] += 1
    if not selection or not confirmation:
        raise ValueError("collective utility split requires both partitions")
    return np.asarray(selection), np.asarray(confirmation)


def _metrics(truth: np.ndarray, scores: np.ndarray) -> dict[str, object]:
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
    truth: np.ndarray, parent: np.ndarray, child: np.ndarray
) -> dict[str, object]:
    parent_correct = parent == truth
    child_correct = child == truth
    improvements = int(np.count_nonzero(~parent_correct & child_correct))
    regressions = int(np.count_nonzero(parent_correct & ~child_correct))
    return {
        "corrections": improvements,
        "regressions": regressions,
        "delta_utility": improvements - regressions,
        "prediction_disagreements": int(np.count_nonzero(parent != child)),
        "per_truth_class": [
            {
                "digit": digit,
                "corrections": int(
                    np.count_nonzero(
                        (truth == digit) & ~parent_correct & child_correct
                    )
                ),
                "regressions": int(
                    np.count_nonzero(
                        (truth == digit) & parent_correct & ~child_correct
                    )
                ),
            }
            for digit in range(CLASS_COUNT)
        ],
    }


def _load_epoch_cells(campaign: Path, epoch: int) -> tuple[Mapping[str, object], ...]:
    cells = []
    for digit in range(CLASS_COUNT):
        path = campaign / "cells" / f"epoch-{epoch}" / f"digit-{digit}" / "result.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema") != CELL_SCHEMA:
            raise ValueError(f"unsupported PTA cell: {path}")
        vectors = value.get("score_vectors")
        if (
            not isinstance(vectors, Mapping)
            or vectors.get("semantics") != RAW_VOTE_SEMANTICS
        ):
            raise ValueError(f"PTA cell does not contain raw signed votes: {path}")
        cells.append(value)
    return tuple(cells)


def _score_matrices(
    cells: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    first = cells[0]["score_vectors"]
    if not isinstance(first, Mapping):
        raise ValueError("PTA cell omits score vectors")
    if first.get("semantics") != RAW_VOTE_SEMANTICS:
        raise ValueError("PTA cell score vectors are not raw signed votes")
    truth = np.asarray(first["multiclass_truth"], dtype=np.int64)
    example_ids = first["example_ids"]
    names = (
        "baseline",
        "policy_governed",
        "selected_child_counterfactual",
    )
    matrices: dict[str, np.ndarray] = {}
    for name in names:
        columns = []
        for digit, cell in enumerate(cells):
            vectors = cell["score_vectors"]
            if not isinstance(vectors, Mapping):
                raise ValueError("PTA cell score vectors are malformed")
            if vectors.get("semantics") != RAW_VOTE_SEMANTICS:
                raise ValueError("PTA cell score vectors are not raw signed votes")
            if (
                vectors["example_ids"] != example_ids
                or vectors["multiclass_truth"] != truth.tolist()
            ):
                raise ValueError("PTA cells use different collective corpora")
            binary_truth = np.asarray(vectors["binary_truth"], dtype=np.int64)
            if not np.array_equal(binary_truth, (truth == digit).astype(np.int64)):
                raise ValueError("PTA cell binary truth conflicts with its bank")
            columns.append(np.asarray(vectors[name], dtype=np.int64))
        matrices[name] = np.column_stack(columns)
    return truth, matrices


def _masked_scores(
    baseline: np.ndarray, candidates: np.ndarray, mask: int
) -> np.ndarray:
    result = baseline.copy()
    for digit in range(CLASS_COUNT):
        if mask & (1 << digit):
            result[:, digit] = candidates[:, digit]
    return result


def _analyze_epoch(
    epoch: int, cells: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    truth, matrices = _score_matrices(cells)
    selection_indices, confirmation_indices = _stratified_alternating_split(truth)
    baseline = matrices["baseline"]
    candidates = matrices["selected_child_counterfactual"]
    parent_selection = _metrics(truth[selection_indices], baseline[selection_indices])
    parent_predictions = parent_selection["predictions"]

    best_mask = 0
    best_key: tuple[int, int, int] | None = None
    selection_table = []
    for choices in product((0, 1), repeat=CLASS_COUNT):
        mask = sum(value << digit for digit, value in enumerate(choices))
        scores = _masked_scores(
            baseline[selection_indices], candidates[selection_indices], mask
        )
        values = _metrics(truth[selection_indices], scores)
        replacements = mask.bit_count()
        key = (int(values["correct"]), -replacements, -mask)
        if best_key is None or key > best_key:
            best_key = key
            best_mask = mask
        selection_table.append(
            {
                "mask": mask,
                "replaced_digits": [
                    digit for digit in range(CLASS_COUNT) if mask & (1 << digit)
                ],
                "correct": values["correct"],
                "delta_utility": int(values["correct"]) - int(parent_selection["correct"]),
            }
        )

    variants = {
        "baseline": baseline,
        "existing_policy": matrices["policy_governed"],
        "all_selected_children": candidates,
        "collective_utility_selected": _masked_scores(baseline, candidates, best_mask),
    }
    partitions = {
        "selection": selection_indices,
        "confirmation": confirmation_indices,
    }
    results: dict[str, object] = {}
    for partition, indices in partitions.items():
        baseline_metrics = _metrics(truth[indices], baseline[indices])
        baseline_predictions = baseline_metrics["predictions"]
        variant_results: dict[str, object] = {}
        for name, scores in variants.items():
            values = _metrics(truth[indices], scores[indices])
            predictions = values.pop("predictions")
            variant_results[name] = {
                **values,
                "paired_against_baseline": _paired(
                    truth[indices], baseline_predictions, predictions
                ),
            }
        results[partition] = variant_results
    parent_selection.pop("predictions")
    winner = next(item for item in selection_table if item["mask"] == best_mask)
    return {
        "epoch": epoch,
        "split": {
            "method": "per-class alternating occurrence, even to selection",
            "selection_observations": len(selection_indices),
            "confirmation_observations": len(confirmation_indices),
        },
        "search": {
            "configuration_count": len(selection_table),
            "objective": "maximize equally weighted correct classifications",
            "tie_break": "fewest replaced banks, then smallest numeric mask",
            "selected_mask": best_mask,
            "selected_digits": winner["replaced_digits"],
            "selection_delta_utility": winner["delta_utility"],
        },
        "partitions": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    campaign = args.campaign.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    campaign_result = json.loads((campaign / "result.json").read_text(encoding="utf-8"))
    checkpoints = campaign_result.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        parser.error("campaign result omits checkpoints")
    epochs = tuple(int(item["epoch"]) for item in checkpoints)
    result = {
        "schema": SCHEMA,
        "claim_boundary": (
            "Retrospective exploratory split of a corpus whose aggregate results "
            "were previously inspected. Confirmation is useful diagnostic evidence, "
            "not deployment-grade independent promotion evidence."
        ),
        "source_campaign": str(campaign),
        "epochs": [
            _analyze_epoch(epoch, _load_epoch_cells(campaign, epoch))
            for epoch in epochs
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "epochs": list(epochs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
