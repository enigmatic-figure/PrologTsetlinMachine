#!/usr/bin/env python3
"""Summarize a completed native PTM clause-capacity campaign."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Mapping, Sequence

from prolog_tsetlin.benchmark_campaign import (
    CampaignDatasetManifest,
    load_dense_bit_split,
)
from prolog_tsetlin.model_generation import canonical_json_bytes, content_digest
from prolog_tsetlin.services._atomic import publish_bytes


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    values = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        values.append(value)
    return tuple(values)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{label} is not a string-keyed mapping")
    return dict(value)


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} is not an integer")
    return value


def _number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} is not a finite number")
    return float(value)


def _read_scores(path: Path, expected_digest: str) -> tuple[int, ...]:
    data = path.read_bytes()
    if _digest(data) != expected_digest:
        raise ValueError(f"vote-score digest mismatch: {path}")
    try:
        scores = tuple(int(line) for line in data.decode("ascii").splitlines())
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"vote-score file is malformed: {path}") from error
    if not scores:
        raise ValueError(f"vote-score file is empty: {path}")
    return scores


def _accuracy(expected: Sequence[int], actual: Sequence[int]) -> float:
    if not expected or len(expected) != len(actual):
        raise ValueError("accuracy inputs are empty or misaligned")
    return sum(left == right for left, right in zip(expected, actual)) / len(expected)


def _subset_accuracy(
    expected: Sequence[int], actual: Sequence[int], indices: Sequence[int]
) -> float | None:
    if not indices:
        return None
    return sum(expected[index] == actual[index] for index in indices) / len(indices)


def _state_margin_metrics(
    rows: Sequence[Sequence[int]], scores: Sequence[int]
) -> dict[str, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    errors: dict[str, list[bool]] = defaultdict(list)
    for row, score in zip(rows, scores):
        state = f"{row[0]}{row[1]}"
        clean_label = row[0] ^ row[1]
        grouped[state].append(score if clean_label else -score)
        errors[state].append(int(score > 0) != clean_label)
    if set(grouped) != {"00", "01", "10", "11"}:
        raise ValueError("XOR material does not cover all four semantic states")
    result: dict[str, float] = {}
    for state in ("00", "01", "10", "11"):
        margins = grouped[state]
        result[f"state_{state}_mean_correct_margin"] = statistics.fmean(margins)
        result[f"state_{state}_median_correct_margin"] = float(
            statistics.median(margins)
        )
        result[f"state_{state}_error_fraction"] = sum(errors[state]) / len(margins)
        result[f"state_{state}_tie_fraction"] = sum(
            margin == 0 for margin in margins
        ) / len(margins)
    return result


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "observations": len(values),
    }


def _one_standard_error_selection(
    cells: Sequence[Mapping[str, object]], metric: str
) -> dict[str, object]:
    if not cells:
        raise ValueError("cannot select from no capacity cells")
    observations_by_cell = [
        _integer(
            _mapping(
                _mapping(cell["metrics"], "cell metrics")[metric],
                f"{metric} distribution",
            )["observations"],
            "observations",
        )
        for cell in cells
    ]
    if min(observations_by_cell) < 2:
        return {
            "metric": metric,
            "status": "insufficient_replicates",
            "minimum_required_per_cell": 2,
            "minimum_observed_per_cell": min(observations_by_cell),
        }
    best = max(cells, key=lambda cell: _number(
        _mapping(cell["metrics"], "cell metrics")[metric]["mean"],
        f"{metric} mean",
    ))
    best_distribution = _mapping(
        _mapping(best["metrics"], "best cell metrics")[metric],
        f"{metric} distribution",
    )
    best_mean = _number(best_distribution["mean"], "best metric mean")
    observations = _integer(best_distribution["observations"], "observations")
    standard_error = _number(best_distribution["stdev"], "metric stdev") / math.sqrt(
        observations
    )
    cutoff = best_mean - standard_error
    eligible = [
        cell
        for cell in cells
        if _number(
            _mapping(
                _mapping(cell["metrics"], "cell metrics")[metric],
                f"{metric} distribution",
            )["mean"],
            f"{metric} mean",
        )
        >= cutoff
    ]
    selected = min(eligible, key=lambda cell: _integer(cell["clauses"], "clauses"))
    return {
        "metric": metric,
        "status": "selected",
        "best_mean_clauses": best["clauses"],
        "best_mean": best_mean,
        "best_mean_standard_error": standard_error,
        "acceptance_cutoff": cutoff,
        "smallest_within_one_best_standard_error": selected["clauses"],
    }


def _flatten_cell(cell: Mapping[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "noise_basis_points": cell["noise_basis_points"],
        "clauses": cell["clauses"],
        "threshold": cell["threshold"],
        "seeds": cell["seeds"],
    }
    for name, distribution in sorted(
        _mapping(cell["metrics"], "cell metrics").items()
    ):
        values = _mapping(distribution, f"{name} distribution")
        for statistic_name in ("mean", "stdev", "minimum", "maximum"):
            row[f"{name}_{statistic_name}"] = values[statistic_name]
    return row


def summarize(campaign_root: Path, material_root: Path) -> dict[str, object]:
    campaign_root = campaign_root.resolve()
    material_root = material_root.resolve()
    plan = _read_json(campaign_root / "plan.json")
    plan_digest = plan.pop("plan_digest", None)
    if plan_digest != content_digest(plan):
        raise ValueError("campaign plan digest is invalid")
    attempts = {
        str(_mapping(value, "campaign attempt")["run_id"]):
        _mapping(value, "campaign attempt")
        for value in plan["attempts"]  # type: ignore[union-attr]
    }
    records = _read_jsonl(campaign_root / "raw.jsonl")
    if len(records) != len(attempts):
        raise ValueError("campaign is incomplete")
    provenance = _mapping(plan.get("route_provenance"), "route provenance")
    native_provenance = _mapping(
        provenance.get("ptm-native"), "PTM native route provenance"
    )
    planned_executable_digest = str(
        native_provenance.get("native_executable_digest")
    )
    if not planned_executable_digest.startswith("sha256:"):
        raise ValueError("campaign plan lacks a native executable digest")

    per_run = []
    for record in records:
        run_id = str(record.get("run_id"))
        if run_id not in attempts or record.get("status") != "ok":
            raise ValueError(f"campaign record is unplanned or unsuccessful: {run_id}")
        attempt = attempts[run_id]
        if record.get("model") != attempt.get("model"):
            raise ValueError(f"campaign record model disagrees with plan: {run_id}")
        record_dataset = _mapping(record["dataset"], "record dataset")
        if record_dataset.get("manifest_digest") != attempt.get("manifest_digest"):
            raise ValueError(
                f"campaign record manifest disagrees with plan: {run_id}"
            )
        manifest_path = material_root / str(attempt["manifest"])
        manifest = CampaignDatasetManifest.load(manifest_path)
        if manifest.dataset_id != "synthetic.xor20-noise.v1":
            raise ValueError("capacity summarizer requires generated XOR material")
        source = _mapping(manifest.source, "manifest source")
        noise = _integer(source["noise_basis_points"], "noise basis points")
        model = _mapping(record["model"], "record model")
        config = _mapping(model["config"], "model config")
        clauses = _integer(config["clauses"], "clause count")
        threshold = _integer(config["threshold"], "threshold")
        seed = _integer(config["seed"], "seed")
        diagnostics = _mapping(record["diagnostics"], "diagnostics")
        artifacts = _mapping(record["artifacts"], "artifacts")
        wrapper = _mapping(artifacts["wrapper"], "wrapper artifacts")
        if wrapper.get("native_executable_digest") != planned_executable_digest:
            raise ValueError(
                f"campaign record executable disagrees with plan: {run_id}"
            )
        vote_files = _mapping(wrapper["vote_score_files"], "vote-score artifacts")
        metrics = _mapping(record["metrics"], "record metrics")
        timing = _mapping(record["timing"], "record timing")
        run_metrics: dict[str, float] = {
            "noisy_train_accuracy": _number(
                _mapping(metrics["train"], "train metrics")["accuracy"],
                "train accuracy",
            ),
            "clean_validation_accuracy": _number(
                _mapping(metrics["validation"], "validation metrics")["accuracy"],
                "validation accuracy",
            ),
            "adaptive_training_s": _number(
                timing["adaptive_training_s"], "adaptive training time"
            ),
            "diagnostic_collection_s": _number(
                timing["diagnostic_collection_s"], "diagnostic collection time"
            ),
        }
        for name, value in diagnostics.items():
            if name == "vote_score_summaries":
                continue
            if type(value) in (int, float):
                run_metrics[name] = _number(value, f"diagnostic {name}")
        for name in (
            "dead_clauses",
            "low_support_clauses_below_1pct",
            "singleton_support_clauses",
            "duplicate_clause_behaviors_within_polarity",
            "opposite_polarity_shared_behaviors",
            "unique_signed_clause_behaviors",
        ):
            run_metrics[f"{name}_fraction"] = run_metrics[name] / clauses

        for split_name in ("train", "validation"):
            rows, observed_labels = load_dense_bit_split(
                manifest_path, manifest, split_name
            )
            vote_artifact = _mapping(vote_files[split_name], "vote-score artifact")
            score_path = campaign_root / "runs" / run_id / str(vote_artifact["path"])
            scores = _read_scores(score_path, str(vote_artifact["digest"]))
            if len(scores) != len(rows):
                raise ValueError(f"vote-score row count mismatch: {run_id} {split_name}")
            predictions = tuple(int(score > 0) for score in scores)
            clean_labels = tuple(row[0] ^ row[1] for row in rows)
            observed_accuracy = _accuracy(observed_labels, predictions)
            recorded_accuracy = _number(
                _mapping(metrics[split_name], f"{split_name} metrics")["accuracy"],
                f"{split_name} accuracy",
            )
            if not math.isclose(observed_accuracy, recorded_accuracy, abs_tol=1e-15):
                raise ValueError(f"retained scores disagree with metrics: {run_id}")
            clean_accuracy = _accuracy(clean_labels, predictions)
            run_metrics[f"clean_{split_name}_accuracy"] = clean_accuracy
            flipped = [
                index
                for index, (observed, clean) in enumerate(
                    zip(observed_labels, clean_labels)
                )
                if observed != clean
            ]
            flipped_set = set(flipped)
            unflipped = [
                index for index in range(len(rows)) if index not in flipped_set
            ]
            flipped_observed = _subset_accuracy(observed_labels, predictions, flipped)
            flipped_clean = _subset_accuracy(clean_labels, predictions, flipped)
            unflipped_clean = _subset_accuracy(clean_labels, predictions, unflipped)
            if flipped_observed is not None:
                run_metrics[f"{split_name}_flipped_observed_accuracy"] = flipped_observed
                run_metrics[f"{split_name}_flipped_clean_accuracy"] = float(flipped_clean)
            if unflipped_clean is not None:
                run_metrics[f"{split_name}_unflipped_clean_accuracy"] = unflipped_clean
            for name, value in _state_margin_metrics(rows, scores).items():
                run_metrics[f"{split_name}_{name}"] = value
        run_metrics["clean_generalization_gap"] = (
            run_metrics["clean_train_accuracy"]
            - run_metrics["clean_validation_accuracy"]
        )
        per_run.append(
            {
                "run_id": run_id,
                "noise_basis_points": noise,
                "clauses": clauses,
                "threshold": threshold,
                "seed": seed,
                "metrics": run_metrics,
            }
        )

    grouped: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    for run in per_run:
        grouped[(run["noise_basis_points"], run["clauses"], run["threshold"])].append(run)  # type: ignore[index]
    cells = []
    for (noise, clauses, threshold), runs in sorted(grouped.items()):
        metric_names = set(_mapping(runs[0]["metrics"], "run metrics"))
        if any(set(_mapping(run["metrics"], "run metrics")) != metric_names for run in runs):
            raise ValueError("run metrics are inconsistent within a surface cell")
        cells.append(
            {
                "noise_basis_points": noise,
                "clauses": clauses,
                "threshold": threshold,
                "seeds": len(runs),
                "metrics": {
                    name: _distribution(
                        [
                            _number(_mapping(run["metrics"], "run metrics")[name], name)
                            for run in runs
                        ]
                    )
                    for name in sorted(metric_names)
                },
            }
        )
    selections = []
    for noise in sorted({int(cell["noise_basis_points"]) for cell in cells}):
        noise_cells = [cell for cell in cells if cell["noise_basis_points"] == noise]
        selections.append(
            {"noise_basis_points": noise}
            | _one_standard_error_selection(
                noise_cells, "clean_validation_accuracy"
            )
        )
    return {
        "schema": "ptm.xor-capacity-surface-summary.v1",
        "source": {
            "plan_digest": plan_digest,
            "attempts": len(records),
            "commit": records[0]["environment"]["source_commit"],  # type: ignore[index]
        },
        "selection_rule": {
            "name": "smallest-within-one-best-standard-error",
            "purpose": "exploratory deterministic capacity reference, not a deployed governor",
            "metric": "clean_validation_accuracy",
        },
        "selections": selections,
        "cells": cells,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("material_root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    output = (arguments.output or arguments.campaign_root / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(arguments.campaign_root, arguments.material_root)
    publish_bytes(
        output / "surface.json",
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n",
        overwrite=True,
    )
    flattened = [_flatten_cell(cell) for cell in summary["cells"]]  # type: ignore[index]
    stream = io.StringIO(newline="")
    identity_fields = ("noise_basis_points", "clauses", "threshold", "seeds")
    metric_fields = sorted(
        set().union(*(set(row) for row in flattened)) - set(identity_fields)
    )
    writer = csv.DictWriter(
        stream, fieldnames=[*identity_fields, *metric_fields], restval=""
    )
    writer.writeheader()
    writer.writerows(flattened)
    publish_bytes(output / "surface.csv", stream.getvalue().encode("utf-8"), overwrite=True)
    print(canonical_json_bytes({
        "output": str(output),
        "source": summary["source"],
        "selections": summary["selections"],
    }).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
