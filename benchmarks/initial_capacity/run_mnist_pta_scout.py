#!/usr/bin/env python3
"""Run one bounded, auditable PTA lifecycle scout on MNIST digit 8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from prolog_tsetlin.model_generation import (
    CorpusExample,
    CorpusRole,
    DeescalationCorpora,
    LabeledCorpus,
    LifecycleCorpora,
    GenerationKind,
    ModelGeneration,
    ModelGenerationError,
    OrderedLiteralManifest,
    PromotionAuditPolicy,
    ThresholdCandidateBudget,
    ThresholdCandidateSelectionPolicy,
)
from prolog_tsetlin.reference import ScalarBinaryTsetlinMachine, TMSnapshot
from prolog_tsetlin.representation import FeatureSchema, FieldKind, LiteralCatalog
from prolog_tsetlin.services.model_generation import (
    ModelGenerationStore,
    execute_literal_deescalation_lifecycle,
    execute_trained_parent_lifecycle_with_candidates,
)
from prolog_tsetlin.services.telemetry import TelemetryEvent, TelemetrySession


SCHEMA = "ptm.mnist-pta-scout.v1"
PIXEL_THRESHOLD = 0.3


def _balanced(
    labels: np.ndarray,
    *,
    target: int,
    positive: np.ndarray,
    negative: np.ndarray,
) -> np.ndarray:
    indices = np.concatenate((positive, negative))
    if np.count_nonzero(labels[indices] == target) != len(positive):
        raise RuntimeError("positive MNIST partition is invalid")
    if np.count_nonzero(labels[indices] != target) != len(negative):
        raise RuntimeError("negative MNIST partition is invalid")
    return indices


def _records(
    values: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    pixels: Sequence[int],
    *,
    target: int,
    role: CorpusRole,
    dataset_id: str,
    first_id: int,
) -> LabeledCorpus:
    fields = tuple(f"p{pixel:03d}" for pixel in pixels)
    examples = tuple(
        CorpusExample(
            first_id + position,
            {
                field: float(values[int(index), pixel])
                for field, pixel in zip(fields, pixels)
            },
            int(labels[int(index)] == target),
        )
        for position, index in enumerate(indices)
    )
    return LabeledCorpus(dataset_id, role, examples)


def _take_unique(
    pool: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    pixels: Sequence[int],
    *,
    target: int,
    count: int,
    used: set[tuple[int, tuple[float, ...]]],
    start: int,
) -> tuple[np.ndarray, int]:
    selected: list[int] = []
    position = start
    while position < len(pool) and len(selected) < count:
        index = int(pool[position])
        position += 1
        fingerprint = (
            int(labels[index] == target),
            tuple(float(values[index, pixel]) for pixel in pixels),
        )
        if fingerprint in used:
            continue
        used.add(fingerprint)
        selected.append(index)
    if len(selected) != count:
        raise RuntimeError("MNIST partition lacks enough unique projected records")
    return np.asarray(selected, dtype=np.int64), position


def _metrics(
    snapshot: TMSnapshot,
    manifest: OrderedLiteralManifest,
    corpus: LabeledCorpus,
) -> dict[str, object]:
    batch = manifest.build_catalog().encode(corpus.records).ta
    machine = ScalarBinaryTsetlinMachine(
        snapshot.number_of_clauses,
        snapshot.number_of_features,
        states_per_action=snapshot.states_per_action,
        specificity=snapshot.specificity,
        threshold=snapshot.threshold,
        seed=0,
    )
    machine.restore(snapshot)
    rows = tuple(batch.row_values(index) for index in range(batch.row_count))
    predictions = tuple(machine.predict(rows))
    scores = tuple(machine.score(row) for row in rows)
    truth = corpus.labels
    true_positive = sum(a == 1 and b == 1 for a, b in zip(truth, predictions))
    true_negative = sum(a == 0 and b == 0 for a, b in zip(truth, predictions))
    false_positive = sum(a == 0 and b == 1 for a, b in zip(truth, predictions))
    false_negative = sum(a == 1 and b == 0 for a, b in zip(truth, predictions))
    return {
        "observations": len(truth),
        "correct": true_positive + true_negative,
        "accuracy": (true_positive + true_negative) / len(truth),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "predictions": list(predictions),
        "scores": list(scores),
    }


def _paired(
    truth: Sequence[int], baseline: Sequence[int], candidate: Sequence[int]
) -> dict[str, int]:
    improvements = regressions = disagreements = 0
    for expected, before, after in zip(truth, baseline, candidate):
        improvements += before != expected and after == expected
        regressions += before == expected and after != expected
        disagreements += before != after
    return {
        "improvements": improvements,
        "regressions": regressions,
        "prediction_disagreements": disagreements,
    }


def _public_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"predictions", "scores"}
    }


def _only_json(directory: Path) -> Mapping[str, object] | None:
    paths = sorted(directory.rglob("*.json")) if directory.is_dir() else []
    if len(paths) != 1:
        return None
    value = json.loads(paths[0].read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _generation_ids(
    store: ModelGenerationStore, kind: GenerationKind
) -> tuple[str, ...]:
    directory = store.root / "objects" / "generations"
    paths = sorted(directory.rglob("*.json")) if directory.is_dir() else []
    identifiers: list[str] = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        generation = ModelGeneration.from_dict(value)
        if generation.kind is kind:
            identifiers.append(generation.generation_id)
    return tuple(identifiers)


def _telemetry(event: TelemetryEvent) -> dict[str, object]:
    return {
        "sequence": event.sequence,
        "source": event.source,
        "kind": event.kind,
        "level": event.level,
        "payload": dict(event.payload),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/mnist.pkl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ptmrt", type=Path, default=Path("out/build/Release/ptmrt.exe")
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--target-digit", type=int, default=8)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--clauses", type=int, default=40)
    parser.add_argument("--parent-epochs", type=int, default=10)
    parser.add_argument("--adaptation-epochs", type=int, default=5)
    parser.add_argument(
        "--parent-snapshot",
        type=Path,
        help="trusted local TMSnapshot checkpoint to audit instead of retraining",
    )
    parser.add_argument("--selected-pixels", nargs="+", type=int)
    parser.add_argument("--shared-audit-rows", type=int, default=0)
    parser.add_argument("--emit-score-vectors", action="store_true")
    args = parser.parse_args(argv)
    if not 4 <= args.features <= 12:
        parser.error(
            "--features must lie in 4..12 for the bounded PTA session"
        )
    if not 0 <= args.target_digit <= 9:
        parser.error("--target-digit must lie in 0..9")
    if args.shared_audit_rows < 0:
        parser.error("--shared-audit-rows must be nonnegative")
    if args.selected_pixels is not None and len(args.selected_pixels) != args.features:
        parser.error("--selected-pixels must contain exactly --features positions")
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    source = args.source.expanduser().resolve()
    ptmrt = args.ptmrt.expanduser().resolve()
    if not source.is_file() or not ptmrt.is_file():
        parser.error("MNIST source and ptmrt must exist")
    output.mkdir(parents=True)

    preparation_started = perf_counter()
    train, validation, test = pickle.loads(source.read_bytes(), encoding="latin1")
    train_x, train_y = np.asarray(train[0]), np.asarray(train[1])
    validation_x, validation_y = np.asarray(validation[0]), np.asarray(validation[1])
    test_x, test_y = np.asarray(test[0]), np.asarray(test[1])
    rng = np.random.default_rng(args.seed)
    target_digit = args.target_digit
    train_pos = rng.permutation(np.flatnonzero(train_y == target_digit))
    train_neg = rng.permutation(np.flatnonzero(train_y != target_digit))
    validation_pos = rng.permutation(np.flatnonzero(validation_y == target_digit))
    validation_neg = rng.permutation(np.flatnonzero(validation_y != target_digit))
    test_pos = rng.permutation(np.flatnonzero(test_y == target_digit))
    test_neg = rng.permutation(np.flatnonzero(test_y != target_digit))

    feature_selection_indices = _balanced(
        train_y,
        target=target_digit,
        positive=train_pos[:200],
        negative=train_neg[:200],
    )
    difference = np.abs(
        train_x[feature_selection_indices][
            train_y[feature_selection_indices] == target_digit
        ].mean(axis=0)
        - train_x[feature_selection_indices][
            train_y[feature_selection_indices] != target_digit
        ].mean(axis=0)
    )
    if args.selected_pixels is None:
        informative = [
            int(pixel)
            for pixel in np.argsort(difference)[::-1]
            if int(pixel) not in (0, 1)
        ][: args.features - 2]
        pixels = (0, 1, *informative)
    else:
        pixels = tuple(args.selected_pixels)
        informative = list(pixels[2:])
    if len(set(pixels)) != len(pixels) or any(not 0 <= pixel < 784 for pixel in pixels):
        parser.error("--selected-pixels must be unique positions in 0..783")
    fields = tuple(f"p{pixel:03d}" for pixel in pixels)
    dataset_id = f"mnist-digit-{target_digit}-pta-scout-seed-{args.seed}"
    parent_fingerprints: set[tuple[int, tuple[float, ...]]] = set()
    parent_pos, parent_pos_cursor = _take_unique(
        train_pos, train_x, train_y, pixels,
        target=target_digit, count=200, used=parent_fingerprints, start=0,
    )
    parent_neg, parent_neg_cursor = _take_unique(
        train_neg, train_x, train_y, pixels,
        target=target_digit, count=200, used=parent_fingerprints, start=0,
    )
    parent_indices = _balanced(
        train_y,
        target=target_digit,
        positive=parent_pos,
        negative=parent_neg,
    )
    input_used = set(parent_fingerprints)
    invention_pos, train_pos_cursor = _take_unique(
        train_pos, train_x, train_y, pixels,
        target=target_digit, count=40, used=input_used, start=parent_pos_cursor,
    )
    invention_neg, train_neg_cursor = _take_unique(
        train_neg, train_x, train_y, pixels,
        target=target_digit, count=40, used=input_used, start=parent_neg_cursor,
    )
    adaptation_pos, train_pos_cursor = _take_unique(
        train_pos, train_x, train_y, pixels,
        target=target_digit, count=80, used=input_used, start=train_pos_cursor,
    )
    adaptation_neg, train_neg_cursor = _take_unique(
        train_neg, train_x, train_y, pixels,
        target=target_digit, count=80, used=input_used, start=train_neg_cursor,
    )
    promotion_pos, _ = _take_unique(
        validation_pos, validation_x, validation_y, pixels,
        target=target_digit, count=100, used=input_used, start=0,
    )
    promotion_neg, _ = _take_unique(
        validation_neg, validation_x, validation_y, pixels,
        target=target_digit, count=100, used=input_used, start=0,
    )
    deescalation_used = set(parent_fingerprints)
    proof_pos, deescalation_pos_cursor = _take_unique(
        train_pos, train_x, train_y, pixels,
        target=target_digit, count=40, used=deescalation_used, start=1_000,
    )
    proof_neg, deescalation_neg_cursor = _take_unique(
        train_neg, train_x, train_y, pixels,
        target=target_digit, count=40, used=deescalation_used, start=1_000,
    )
    confirmation_pos, _ = _take_unique(
        train_pos, train_x, train_y, pixels,
        target=target_digit, count=40, used=deescalation_used,
        start=deescalation_pos_cursor,
    )
    confirmation_neg, _ = _take_unique(
        train_neg, train_x, train_y, pixels,
        target=target_digit, count=40, used=deescalation_used,
        start=deescalation_neg_cursor,
    )
    deescalation_promotion_pos, _ = _take_unique(
        validation_pos, validation_x, validation_y, pixels,
        target=target_digit, count=100, used=deescalation_used, start=300,
    )
    deescalation_promotion_neg, _ = _take_unique(
        validation_neg, validation_x, validation_y, pixels,
        target=target_digit, count=100, used=deescalation_used, start=300,
    )
    parent_corpus = _records(
        train_x,
        train_y,
        parent_indices,
        pixels,
        target=target_digit,
        role=CorpusRole.PARENT_TRAINING,
        dataset_id=dataset_id,
        first_id=0,
    )
    invention = _records(
        train_x,
        train_y,
        _balanced(
            train_y,
            target=target_digit,
            positive=invention_pos,
            negative=invention_neg,
        ),
        pixels,
        target=target_digit,
        role=CorpusRole.INVENTION,
        dataset_id=dataset_id,
        first_id=10_000,
    )
    adaptation = _records(
        train_x,
        train_y,
        _balanced(
            train_y,
            target=target_digit,
            positive=adaptation_pos,
            negative=adaptation_neg,
        ),
        pixels,
        target=target_digit,
        role=CorpusRole.ADAPTATION,
        dataset_id=dataset_id,
        first_id=20_000,
    )
    promotion = _records(
        validation_x,
        validation_y,
        _balanced(
            validation_y,
            target=target_digit,
            positive=promotion_pos,
            negative=promotion_neg,
        ),
        pixels,
        target=target_digit,
        role=CorpusRole.PROMOTION,
        dataset_id=dataset_id,
        first_id=30_000,
    )
    deescalation_proof = _records(
        train_x,
        train_y,
        _balanced(
            train_y,
            target=target_digit,
            positive=proof_pos,
            negative=proof_neg,
        ),
        pixels,
        target=target_digit,
        role=CorpusRole.DEESCALATION_PROOF,
        dataset_id=dataset_id,
        first_id=40_000,
    )
    deescalation_confirmation = _records(
        train_x,
        train_y,
        _balanced(
            train_y,
            target=target_digit,
            positive=confirmation_pos,
            negative=confirmation_neg,
        ),
        pixels,
        target=target_digit,
        role=CorpusRole.DEESCALATION_CONFIRMATION,
        dataset_id=dataset_id,
        first_id=50_000,
    )
    deescalation_promotion = _records(
        validation_x,
        validation_y,
        _balanced(
            validation_y,
            target=target_digit,
            positive=deescalation_promotion_pos,
            negative=deescalation_promotion_neg,
        ),
        pixels,
        target=target_digit,
        role=CorpusRole.PROMOTION,
        dataset_id=dataset_id,
        first_id=60_000,
    )
    evaluation_indices = (
        np.arange(min(args.shared_audit_rows, len(test_y)), dtype=np.int64)
        if args.shared_audit_rows
        else _balanced(
            test_y,
            target=target_digit,
            positive=test_pos[:500],
            negative=test_neg[:500],
        )
    )
    evaluation = _records(
        test_x,
        test_y,
        evaluation_indices,
        pixels,
        target=target_digit,
        role=CorpusRole.LIVE,
        dataset_id=dataset_id,
        first_id=70_000,
    )
    schema = FeatureSchema.from_fields(
        **{field: FieldKind.NUMBER for field in fields}
    )
    catalog = LiteralCatalog(schema)
    for field in fields:
        catalog.numeric_ge(field, PIXEL_THRESHOLD)
    manifest = OrderedLiteralManifest.from_catalog(catalog)
    preparation_seconds = perf_counter() - preparation_started

    parent = ScalarBinaryTsetlinMachine(
        args.clauses,
        args.features,
        states_per_action=128,
        specificity=8.0,
        threshold=10,
        seed=args.seed,
    )
    training_started = perf_counter()
    if args.parent_snapshot is None:
        parent.fit_literal_batch(
            catalog.encode(parent_corpus.records).ta,
            parent_corpus.labels,
            epochs=args.parent_epochs,
        )
    else:
        checkpoint = args.parent_snapshot.expanduser().resolve()
        if not checkpoint.is_file():
            parser.error(f"parent snapshot does not exist: {checkpoint}")
        loaded_snapshot = pickle.loads(checkpoint.read_bytes())
        if not isinstance(loaded_snapshot, TMSnapshot):
            parser.error("--parent-snapshot must contain a TMSnapshot")
        parent.restore(loaded_snapshot)
    parent_training_seconds = perf_counter() - training_started
    parent_snapshot = parent.snapshot()
    baseline_test = _metrics(parent_snapshot, manifest, evaluation)

    events: list[TelemetryEvent] = []
    input_store = ModelGenerationStore(output / "input-escalation-store")
    input_started = perf_counter()
    input_result = None
    input_error = None
    try:
        input_result = execute_trained_parent_lifecycle_with_candidates(
            parent_snapshot=parent_snapshot,
            parent_manifest=manifest,
            parent_training_corpus=parent_corpus,
            corpora=LifecycleCorpora(invention, adaptation, promotion),
            numeric_fields=(f"p{informative[0]:03d}",),
            candidate_budget=ThresholdCandidateBudget(1, 64),
            selection_policy=ThresholdCandidateSelectionPolicy(
                len(adaptation.examples), True
            ),
            adaptation_epochs=args.adaptation_epochs,
            promotion_policy=PromotionAuditPolicy(len(promotion.examples)),
            store=input_store,
            ptmrt_executable=ptmrt,
            telemetry=TelemetrySession(),
            event_sink=events.append,
        )
    except ModelGenerationError as error:
        input_error = str(error)
    input_seconds = perf_counter() - input_started
    candidate_set = _only_json(
        input_store.root / "objects" / "threshold-candidate-sets"
    )
    candidate_selection = _only_json(
        input_store.root / "objects" / "threshold-candidate-selections"
    )
    input_audit = _only_json(input_store.root / "objects" / "audits")
    rejected_child_ids = _generation_ids(
        input_store, GenerationKind.ADAPTED_CHILD
    )

    deescalation_events: list[TelemetryEvent] = []
    deescalation_store = ModelGenerationStore(output / "deescalation-store")
    deescalation_started = perf_counter()
    deescalation_result = None
    deescalation_error = None
    try:
        deescalation_result = execute_literal_deescalation_lifecycle(
            parent_snapshot=parent_snapshot,
            parent_manifest=manifest,
            parent_training_corpus=parent_corpus,
            corpora=DeescalationCorpora(
                deescalation_proof,
                deescalation_confirmation,
                deescalation_promotion,
            ),
            promotion_policy=PromotionAuditPolicy(
                len(deescalation_promotion.examples),
                require_strict_improvement=False,
                maximum_regressions=0,
            ),
            store=deescalation_store,
            ptmrt_executable=ptmrt,
            maximum_candidates=64,
            telemetry=TelemetrySession(),
            event_sink=deescalation_events.append,
        )
    except ModelGenerationError as error:
        deescalation_error = str(error)
    deescalation_seconds = perf_counter() - deescalation_started
    deescalation_evidence = _only_json(
        deescalation_store.root / "objects" / "deescalation-evidence"
    )
    deescalation_audit = _only_json(
        deescalation_store.root / "objects" / "audits"
    )

    result: dict[str, object] = {
        "schema": SCHEMA,
        "scope": "bounded digit-8 one-vs-rest PTA integration scout",
        "claim_boundary": (
            "This exercises executable PTA lifecycles on MNIST but is not a "
            "ten-bank or full-dataset accuracy benchmark."
        ),
        "config": {
            "seed": args.seed,
            "target_digit": target_digit,
            "features": args.features,
            "selected_pixels": list(pixels),
            "feature_selection_source_rows": len(feature_selection_indices),
            "clauses": args.clauses,
            "parent_epochs": args.parent_epochs,
            "parent_snapshot": (
                str(args.parent_snapshot.expanduser().resolve())
                if args.parent_snapshot is not None
                else None
            ),
            "adaptation_epochs": args.adaptation_epochs,
            "pixel_threshold": PIXEL_THRESHOLD,
            "parent_rows": len(parent_corpus.examples),
            "invention_rows": len(invention.examples),
            "adaptation_rows": len(adaptation.examples),
            "promotion_rows": len(promotion.examples),
            "test_rows": len(evaluation.examples),
        },
        "timing_seconds": {
            "preparation": preparation_seconds,
            "parent_training": parent_training_seconds,
            "input_escalation_lifecycle": input_seconds,
            "deescalation_lifecycle": deescalation_seconds,
        },
        "baseline_test": {
            **_public_metrics(baseline_test)
        },
        "pta_usage": {
            "input": {
                "attempted": True,
                "durable_collective_episodes": int(candidate_set is not None),
                "candidate_count": (
                    candidate_set.get("available_candidates")
                    if candidate_set is not None
                    else None
                ),
                "evidence": candidate_set,
            },
            "escalation": {
                "attempted": True,
                "candidate_evaluations": (
                    len(candidate_selection.get("outcomes", []))
                    if candidate_selection is not None
                    else (
                        int(candidate_set.get("available_candidates", 0))
                        if candidate_set is not None
                        and input_error
                        == "no threshold candidate satisfies selection policy"
                        else 0
                    )
                ),
                "selection": candidate_selection,
                "promotion_audit": input_audit,
                "activated": input_result is not None,
                "error": input_error,
                "telemetry": [_telemetry(event) for event in events],
            },
            "deescalation": {
                "attempted": True,
                "durable_collective_episodes": int(
                    deescalation_evidence is not None
                ),
                "evidence": deescalation_evidence,
                "promotion_audit": deescalation_audit,
                "activated": deescalation_result is not None,
                "error": deescalation_error,
                "telemetry": [
                    _telemetry(event) for event in deescalation_events
                ],
            },
        },
    }
    selected_input_test: Mapping[str, object] | None = None
    deescalation_child_test: Mapping[str, object] | None = None
    if input_result is not None:
        child_test = _metrics(
            input_result.child.snapshot.snapshot,
            input_result.child.manifest,
            evaluation,
        )
        selected_input_test = child_test
        result["input_escalation_test"] = {
            **_public_metrics(child_test)
        }
        result["input_escalation_test_paired"] = _paired(
            evaluation.labels,
            baseline_test["predictions"],
            child_test["predictions"],
        )
    elif len(rejected_child_ids) == 1:
        rejected_generation = input_store.load_generation(rejected_child_ids[0])
        rejected_snapshot = input_store.load_snapshot(
            rejected_generation.snapshot_id
        ).snapshot
        rejected_manifest = input_store.load_manifest(
            rejected_generation.literal_manifest_id
        )
        child_test = _metrics(
            rejected_snapshot,
            rejected_manifest,
            evaluation,
        )
        selected_input_test = child_test
        result["input_escalation_rejected_child_test"] = {
            **_public_metrics(child_test)
        }
        result["input_escalation_rejected_child_test_paired"] = _paired(
            evaluation.labels,
            baseline_test["predictions"],
            child_test["predictions"],
        )
        result["input_escalation_rejected_child_generation_id"] = (
            rejected_generation.generation_id
        )
    if deescalation_result is not None:
        child_test = _metrics(
            deescalation_result.contracted_parent.snapshot.snapshot,
            deescalation_result.contracted_parent.manifest,
            evaluation,
        )
        deescalation_child_test = child_test
        result["deescalation_test"] = {
            **_public_metrics(child_test)
        }
        result["deescalation_test_paired"] = _paired(
            evaluation.labels,
            baseline_test["predictions"],
            child_test["predictions"],
        )
        result["deescalation_feature_delta"] = (
            deescalation_result.contracted_parent.snapshot.snapshot.number_of_features
            - parent_snapshot.number_of_features
        )
    if args.emit_score_vectors:
        counterfactual = selected_input_test or baseline_test
        governed = counterfactual if input_result is not None else baseline_test
        deescalated = deescalation_child_test or baseline_test
        result["score_vectors"] = {
            "example_ids": [item.example_id for item in evaluation.examples],
            "multiclass_truth": [
                int(test_y[int(index)]) for index in evaluation_indices
            ],
            "binary_truth": list(evaluation.labels),
            "baseline": baseline_test["scores"],
            "policy_governed": governed["scores"],
            "selected_child_counterfactual": counterfactual["scores"],
            "deescalated": deescalated["scores"],
        }
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
