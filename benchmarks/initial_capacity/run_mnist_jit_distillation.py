#!/usr/bin/env python3
"""Bounded MNIST neural-teacher / exact-PTM residual-feedback experiment."""

from __future__ import annotations

import argparse
import copy
from hashlib import sha256
import json
from math import exp, isfinite, log, log1p
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from prolog_tsetlin.model_generation import AdaptiveSnapshotEnvelope
from prolog_tsetlin.reference import ScalarBinaryTsetlinMachine, TMSnapshot
from prolog_tsetlin.services._atomic import publish_bytes


SCHEMA = "ptm.mnist-jit-distillation.v4"
PLAN_SCHEMA = "ptm.mnist-jit-distillation-plan.v4"
CLASS_COUNT = 10
PIXEL_THRESHOLD = 0.3
TEACHER_GATED_ARM = "D_teacher_gated_hard_residual"
ARM_NAMES = (
    "A_hard_margin",
    "B_hard_residual",
    "C_teacher_residual",
    TEACHER_GATED_ARM,
    "E_uniform_slowdown",
    "F_normalized_linear",
    "G_normalized_logarithmic",
    "H_normalized_squared",
    "I_protected_baseline_blend",
    "J_student_only_priorities",
    "K_shuffled_teacher_control",
    "L_delayed_gradual_soft_targets",
    "M_normalized_agreement",
    "N_normalized_teacher_advantage",
)
NORMALIZED_MULTIPLICATIVE_ARMS = {
    "F_normalized_linear",
    "G_normalized_logarithmic",
    "H_normalized_squared",
    "J_student_only_priorities",
    "K_shuffled_teacher_control",
    "M_normalized_agreement",
    "N_normalized_teacher_advantage",
}
NORMALIZED_ALLOCATION_ARMS = {"I_protected_baseline_blend"}
NORMALIZED_ARMS = NORMALIZED_MULTIPLICATIVE_ARMS | NORMALIZED_ALLOCATION_ARMS
TEACHER_DEPENDENT_ARMS = {
    "C_teacher_residual",
    TEACHER_GATED_ARM,
    "F_normalized_linear",
    "G_normalized_logarithmic",
    "H_normalized_squared",
    "I_protected_baseline_blend",
    "K_shuffled_teacher_control",
    "L_delayed_gradual_soft_targets",
    "M_normalized_agreement",
    "N_normalized_teacher_advantage",
}


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _git_identity(project: Path) -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": revision, "tracked_worktree_dirty": bool(status.strip())}


def _build_plan(
    args: argparse.Namespace,
    *,
    source: Path,
    ptmrt: Path,
    torch_version: str,
) -> dict[str, object]:
    project = Path(__file__).resolve().parents[2]
    return {
        "schema": PLAN_SCHEMA,
        "configuration": {
            "seed": args.seed,
            "features": args.features,
            "clauses": args.clauses,
            "epochs": args.epochs,
            "teacher_epochs": args.teacher_epochs,
            "teacher_temperature": args.teacher_temperature,
            "student_temperature": args.student_temperature,
            "residual_learning_rate": args.residual_learning_rate,
            "teacher_gate_floor": args.teacher_gate_floor,
            "validation_rows": args.validation_rows,
            "pta_audit_rows": args.pta_audit_rows,
            "skip_pta": args.skip_pta,
            "pixel_threshold": PIXEL_THRESHOLD,
            "arms": list(ARM_NAMES),
            "teacher_permutation_seed": args.seed ^ 0x5EED5EED,
            "normalization_protocol": "pre_epoch_fixed_scores_per_bank",
        },
        "inputs": {
            "source": str(source),
            "source_digest": _file_digest(source),
            "ptmrt": None if args.skip_pta else str(ptmrt),
            "ptmrt_digest": None if args.skip_pta else _file_digest(ptmrt),
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch_version,
        },
        "code": {
            "git": _git_identity(project),
            "runner_digest": _file_digest(Path(__file__).resolve()),
            "scout_digest": _file_digest(
                Path(__file__).with_name("run_mnist_pta_scout.py")
            ),
            "reference_digest": _file_digest(
                project / "python" / "prolog_tsetlin" / "reference.py"
            ),
        },
    }


def _initialize_or_validate_plan(
    output: Path,
    plan: Mapping[str, object],
    *,
    resume: bool,
) -> None:
    plan_path = output / "plan.json"
    if output.exists():
        if not resume:
            raise ValueError(f"output already exists: {output}; pass --resume to continue")
        if not plan_path.is_file():
            raise ValueError("resume output has no immutable plan.json")
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise ValueError("resume configuration or input identity does not match plan.json")
        return
    if resume:
        raise ValueError("--resume requires an existing output directory")
    output.mkdir(parents=True)
    publish_bytes(plan_path, _canonical_json(plan), overwrite=False)


def _validate_cli_parameters(args: argparse.Namespace) -> None:
    if args.epochs < 2:
        raise ValueError("--epochs must be at least 2")
    if args.teacher_epochs < 1:
        raise ValueError("--teacher-epochs must be positive")
    if not 4 <= args.features <= 12:
        raise ValueError("--features must lie in 4..12")
    if args.clauses <= 0:
        raise ValueError("--clauses must be positive")
    if args.validation_rows <= 0:
        raise ValueError("--validation-rows must be positive")
    if args.pta_audit_rows <= 0:
        raise ValueError("--pta-audit-rows must be positive")
    for option, value in (
        ("--teacher-temperature", args.teacher_temperature),
        ("--student-temperature", args.student_temperature),
        ("--residual-learning-rate", args.residual_learning_rate),
    ):
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"{option} must be finite and positive")
    if (
        not isfinite(args.teacher_gate_floor)
        or not 0.0 <= args.teacher_gate_floor <= 1.0
    ):
        raise ValueError("--teacher-gate-floor must be finite and in [0, 1]")


def _load_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for the temporary teacher") from error
    return torch, nn, DataLoader, TensorDataset


def _teacher_model(nn):
    return nn.Sequential(
        nn.Unflatten(1, (1, 28, 28)),
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 64),
        nn.ReLU(),
        nn.Linear(64, CLASS_COUNT),
    )


def _teacher_logits(model, values: np.ndarray, *, batch_size: int, torch) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for first in range(0, len(values), batch_size):
            batch = torch.from_numpy(
                np.asarray(values[first : first + batch_size], dtype=np.float32)
            )
            chunks.append(model(batch).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    values = np.exp(scaled)
    return values / values.sum(axis=1, keepdims=True)


def _train_teacher(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[object, dict[str, object]]:
    torch, nn, DataLoader, TensorDataset = _load_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = _teacher_model(nn)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(train_x, dtype=np.float32)),
            torch.from_numpy(np.asarray(train_y, dtype=np.int64)),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    history: list[dict[str, object]] = []
    selected_state = None
    selected_accuracy = -1.0
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = perf_counter()
        model.train()
        total_loss = 0.0
        observations = 0
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            observations += len(targets)
        validation_logits = _teacher_logits(
            model, validation_x, batch_size=batch_size, torch=torch
        )
        validation_accuracy = float(
            np.mean(np.argmax(validation_logits, axis=1) == validation_y)
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / observations,
                "validation_accuracy": validation_accuracy,
                "elapsed_seconds": perf_counter() - epoch_started,
            }
        )
        if validation_accuracy > selected_accuracy:
            selected_accuracy = validation_accuracy
            selected_state = copy.deepcopy(model.state_dict())
        print(
            f"teacher epoch={epoch} validation_accuracy={validation_accuracy:.6f}",
            flush=True,
        )
    assert selected_state is not None
    model.load_state_dict(selected_state)
    return model, {
        "training_seconds": perf_counter() - started,
        "history": history,
        "selected_validation_accuracy": selected_accuracy,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        ),
        "torch_version": torch.__version__,
        "device": "cpu",
    }


def _unique_projected_indices(
    pool: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    pixels: Sequence[int],
    *,
    target: int,
    count: int,
) -> np.ndarray:
    selected: list[int] = []
    used: set[tuple[int, tuple[float, ...]]] = set()
    for raw_index in pool:
        index = int(raw_index)
        fingerprint = (
            int(labels[index] == target),
            tuple(float(values[index, pixel]) for pixel in pixels),
        )
        if fingerprint in used:
            continue
        used.add(fingerprint)
        selected.append(index)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError("MNIST partition lacks enough unique projected records")
    return np.asarray(selected, dtype=np.int64)


def _prepare_bank(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    digit: int,
    seed: int,
    features: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    positive = rng.permutation(np.flatnonzero(train_y == digit))
    negative = rng.permutation(np.flatnonzero(train_y != digit))
    selection = np.concatenate((positive[:200], negative[:200]))
    difference = np.abs(
        train_x[selection][train_y[selection] == digit].mean(axis=0)
        - train_x[selection][train_y[selection] != digit].mean(axis=0)
    )
    informative = [
        int(pixel)
        for pixel in np.argsort(difference)[::-1]
        if int(pixel) not in (0, 1)
    ][: features - 2]
    pixels = (0, 1, *informative)
    positive_rows = _unique_projected_indices(
        positive, train_x, train_y, pixels, target=digit, count=200
    )
    negative_rows = _unique_projected_indices(
        negative, train_x, train_y, pixels, target=digit, count=200
    )
    indices = np.concatenate((positive_rows, negative_rows))
    encoded = np.asarray(train_x[indices][:, pixels] >= PIXEL_THRESHOLD, dtype=np.uint8)
    rows = tuple(tuple(int(value) for value in row) for row in encoded)
    targets = tuple(int(value) for value in (train_y[indices] == digit))
    positive_binary_rows = len(set(rows[: len(positive_rows)]))
    negative_binary_rows = len(set(rows[len(positive_rows) :]))
    return {
        "digit": digit,
        "pixels": pixels,
        "indices": indices,
        "rows": rows,
        "targets": targets,
        "unique_binary_rows": {
            "positive": positive_binary_rows,
            "negative": negative_binary_rows,
            "total": len(set(zip(targets, rows))),
        },
    }


def _machine_from_snapshot(snapshot: TMSnapshot) -> ScalarBinaryTsetlinMachine:
    machine = ScalarBinaryTsetlinMachine(
        snapshot.number_of_clauses,
        snapshot.number_of_features,
        states_per_action=snapshot.states_per_action,
        specificity=snapshot.specificity,
        threshold=snapshot.threshold,
        seed=0,
    )
    machine.restore(snapshot)
    return machine


def _score_collective(
    machines: Sequence[ScalarBinaryTsetlinMachine],
    banks: Sequence[Mapping[str, object]],
    values: np.ndarray,
) -> tuple[np.ndarray, float]:
    started = perf_counter()
    columns: list[np.ndarray] = []
    for machine, bank in zip(machines, banks):
        pixels = bank["pixels"]
        assert isinstance(pixels, tuple)
        rows = np.asarray(values[:, pixels] >= PIXEL_THRESHOLD, dtype=np.uint8)
        columns.append(
            np.fromiter(
                (machine.raw_vote(row.tolist()) for row in rows),
                dtype=np.int16,
                count=len(rows),
            )
        )
    return np.column_stack(columns), perf_counter() - started


def _clip_collective_scores(
    raw_scores: np.ndarray,
    machines: Sequence[ScalarBinaryTsetlinMachine],
) -> np.ndarray:
    """Apply each bank's training margin without re-evaluating its clauses."""

    if raw_scores.ndim != 2 or raw_scores.shape[1] != len(machines):
        raise ValueError("raw score matrix does not match the collective")
    clipped = raw_scores.copy()
    for digit, machine in enumerate(machines):
        clipped[:, digit] = np.clip(
            clipped[:, digit], -machine.threshold, machine.threshold
        )
    return clipped


def _classification(truth: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    predictions = np.argmax(scores, axis=1)
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    np.add.at(confusion, (truth, predictions), 1)
    correct = predictions == truth
    return {
        "observations": len(truth),
        "accuracy": float(np.mean(correct)),
        "correct": int(np.count_nonzero(correct)),
        "per_digit_recall": [
            (
                float(confusion[digit, digit] / confusion[digit].sum())
                if confusion[digit].sum()
                else None
            )
            for digit in range(CLASS_COUNT)
        ],
        "confusion_matrix": confusion.tolist(),
        "predictions": predictions,
    }


def _paired(
    truth: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, int]:
    before = baseline == truth
    after = candidate == truth
    return {
        "fixes": int(np.count_nonzero(~before & after)),
        "regressions": int(np.count_nonzero(before & ~after)),
        "disagreements": int(np.count_nonzero(baseline != candidate)),
    }


def _sigmoid_scores(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = scores.astype(np.float64) / temperature
    return np.where(
        scaled >= 0,
        1.0 / (1.0 + np.exp(-scaled)),
        np.exp(scaled) / (1.0 + np.exp(scaled)),
    )


def _sigmoid_vote(raw_vote: int, temperature: float) -> float:
    scaled = raw_vote / temperature
    if scaled >= 0.0:
        return 1.0 / (1.0 + exp(-scaled))
    scaled_exp = exp(scaled)
    return scaled_exp / (1.0 + scaled_exp)


def _teacher_disagreement_gate(
    teacher_probability: float,
    student_probability: float,
    *,
    floor: float,
) -> tuple[float, float]:
    for name, value in (
        ("teacher_probability", teacher_probability),
        ("student_probability", student_probability),
        ("floor", floor),
    ):
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    disagreement = abs(teacher_probability - student_probability)
    return disagreement, floor + (1.0 - floor) * disagreement


def _priority_weight(
    arm: str,
    *,
    hard_residual: float,
    disagreement: float,
    hard_target: float,
    teacher_probability: float,
) -> float:
    if arm == "F_normalized_linear":
        priority = disagreement
        return 0.1 + 0.9 * priority
    if arm in {
        "G_normalized_logarithmic",
        "I_protected_baseline_blend",
        "K_shuffled_teacher_control",
    }:
        priority = disagreement
    elif arm == "H_normalized_squared":
        priority = disagreement * disagreement
        return 0.1 + 0.9 * priority
    elif arm == "J_student_only_priorities":
        priority = hard_residual
    elif arm == "M_normalized_agreement":
        priority = 1.0 - disagreement
    elif arm == "N_normalized_teacher_advantage":
        teacher_error = abs(hard_target - teacher_probability)
        priority = max(0.0, hard_residual - teacher_error)
    else:
        raise ValueError(f"arm has no normalized priority: {arm}")
    return 0.1 + 0.9 * log1p(9.0 * priority) / log(10.0)


def _solve_clipped_scale(values: np.ndarray, target_sum: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("normalization values must be one finite nonnegative vector")
    if not isfinite(target_sum) or target_sum < 0.0:
        raise ValueError("normalization target must be finite and nonnegative")
    if target_sum == 0.0:
        return 0.0
    capacity = int(np.count_nonzero(values > 0.0))
    if target_sum > capacity + 1e-12:
        raise ValueError("normalization target exceeds clipped probability capacity")

    lower = 0.0
    upper = 1.0
    while float(np.minimum(1.0, upper * values).sum()) < target_sum:
        upper *= 2.0
    for _ in range(80):
        middle = (lower + upper) / 2.0
        if float(np.minimum(1.0, middle * values).sum()) < target_sum:
            lower = middle
        else:
            upper = middle
    return upper


def _within_class_permutation(labels: np.ndarray, *, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(labels), dtype=np.int64)
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        permutation[positions] = rng.permutation(positions)
    if not np.array_equal(labels, labels[permutation]):
        raise RuntimeError("teacher permutation crossed ground-truth classes")
    return permutation


def _teacher_probability_for_arm(
    arm: str,
    teacher_probabilities: np.ndarray,
    teacher_permutation: np.ndarray,
    *,
    source_index: int,
    digit: int,
) -> float:
    teacher_index = (
        int(teacher_permutation[source_index])
        if arm == "K_shuffled_teacher_control"
        else source_index
    )
    return float(teacher_probabilities[teacher_index, digit])


def _delayed_teacher_alpha(epoch: int) -> float:
    return min(0.5, 0.1 * max(0, epoch - 4))


def _calibrate_policy(
    arm: str,
    machine: ScalarBinaryTsetlinMachine,
    bank: Mapping[str, object],
    *,
    digit: int,
    teacher_probabilities: np.ndarray,
    teacher_permutation: np.ndarray,
    temperature: float,
    learning_rate: float,
) -> dict[str, float]:
    started = perf_counter()
    rows = bank["rows"]
    targets = bank["targets"]
    indices = bank["indices"]
    assert isinstance(rows, tuple) and isinstance(targets, tuple)
    assert isinstance(indices, np.ndarray)
    bases: list[float] = []
    weights: list[float] = []
    for row, hard_target, source_index in zip(rows, targets, indices):
        hard_target_value = float(hard_target)
        student_probability = _sigmoid_vote(
            machine.raw_vote(row), temperature
        )
        hard_residual = abs(hard_target_value - student_probability)
        base_probability = min(1.0, learning_rate * hard_residual)
        teacher_probability = (
            _teacher_probability_for_arm(
                arm,
                teacher_probabilities,
                teacher_permutation,
                source_index=int(source_index),
                digit=digit,
            )
            if arm in TEACHER_DEPENDENT_ARMS
            else 0.0
        )
        disagreement = abs(teacher_probability - student_probability)
        bases.append(base_probability)
        weights.append(
            _priority_weight(
                arm,
                hard_residual=hard_residual,
                disagreement=disagreement,
                hard_target=hard_target_value,
                teacher_probability=teacher_probability,
            )
        )
    base_values = np.asarray(bases, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    target_sum = float(base_values.sum())
    if arm in NORMALIZED_MULTIPLICATIVE_ARMS:
        scale = _solve_clipped_scale(base_values * weight_values, target_sum)
        calibrated = np.minimum(1.0, scale * base_values * weight_values)
    elif arm in NORMALIZED_ALLOCATION_ARMS:
        scale = _solve_clipped_scale(weight_values, target_sum)
        allocation = np.minimum(1.0, scale * weight_values)
        calibrated = 0.75 * base_values + 0.25 * allocation
    else:
        raise ValueError(f"arm is not normalized: {arm}")
    probability_sum = float(calibrated.sum())
    return {
        "scale": scale,
        "target_base_probability_sum": target_sum,
        "calibrated_probability_sum": probability_sum,
        "calibrated_budget_ratio": (
            probability_sum / target_sum if target_sum > 0.0 else 1.0
        ),
        "seconds": perf_counter() - started,
    }


def _policy_feedback_probability(
    arm: str,
    *,
    hard_base_probability: float,
    hard_residual: float,
    disagreement: float,
    hard_target: float,
    teacher_probability: float,
    normalization_scale: float | None,
    teacher_gate_floor: float = 0.1,
) -> float | None:
    if arm == TEACHER_GATED_ARM:
        return hard_base_probability * (
            teacher_gate_floor + (1.0 - teacher_gate_floor) * disagreement
        )
    if arm == "E_uniform_slowdown":
        return 0.36 * hard_base_probability
    if arm in NORMALIZED_ARMS:
        if normalization_scale is None:
            raise ValueError("normalized policy requires a calibration scale")
        weight = _priority_weight(
            arm,
            hard_residual=hard_residual,
            disagreement=disagreement,
            hard_target=hard_target,
            teacher_probability=teacher_probability,
        )
        if arm in NORMALIZED_MULTIPLICATIVE_ARMS:
            return min(
                1.0,
                normalization_scale * hard_base_probability * weight,
            )
        allocation = min(1.0, normalization_scale * weight)
        return 0.75 * hard_base_probability + 0.25 * allocation
    return None


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    publish_bytes(
        path,
        (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
        overwrite=True,
    )


def _checkpoint_snapshot_id(snapshot: TMSnapshot) -> str:
    return AdaptiveSnapshotEnvelope(snapshot).snapshot_id


def _publish_or_validate_checkpoint(path: Path, snapshot: TMSnapshot) -> None:
    expected_id = _checkpoint_snapshot_id(snapshot)
    if path.is_file():
        existing = pickle.loads(path.read_bytes())
        if not isinstance(existing, TMSnapshot):
            raise RuntimeError(f"cached checkpoint is not a TMSnapshot: {path}")
        if _checkpoint_snapshot_id(existing) != expected_id:
            raise RuntimeError(f"cached checkpoint identity mismatch: {path}")
        return
    publish_bytes(path, pickle.dumps(snapshot, protocol=5), overwrite=False)


def _validate_cached_cell(
    value: object,
    *,
    digit: int,
    seed: int,
    features: int,
    clauses: int,
    selected_epoch: int,
    pixels: Sequence[int],
    audit_rows: int,
    snapshot: TMSnapshot,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or value.get("schema") != "ptm.mnist-pta-scout.v2":
        raise RuntimeError("cached PTA cell has an unsupported schema")
    config = value.get("config")
    vectors = value.get("score_vectors")
    if not isinstance(config, Mapping) or not isinstance(vectors, Mapping):
        raise RuntimeError("cached PTA cell is missing configuration or score vectors")
    if vectors.get("semantics") != "unclipped signed clause votes":
        raise RuntimeError("cached PTA cell has incompatible score semantics")
    expected_snapshot_id = _checkpoint_snapshot_id(snapshot)
    recorded_snapshot_id = config.get("parent_snapshot_id")
    if recorded_snapshot_id is None:
        usage = value.get("pta_usage")
        if isinstance(usage, Mapping):
            deescalation = usage.get("deescalation")
            if isinstance(deescalation, Mapping):
                evidence = deescalation.get("evidence")
                if isinstance(evidence, Mapping):
                    recorded_snapshot_id = evidence.get("parent_snapshot_id")
    expected_config = {
        "target_digit": digit,
        "seed": seed,
        "features": features,
        "clauses": clauses,
        "parent_epochs": selected_epoch,
        "selected_pixels": list(pixels),
        "test_rows": audit_rows,
    }
    if any(config.get(key) != expected for key, expected in expected_config.items()):
        raise RuntimeError("cached PTA cell does not match its campaign slot")
    if recorded_snapshot_id != expected_snapshot_id:
        raise RuntimeError("cached PTA cell parent snapshot identity mismatch")
    if len(vectors.get("multiclass_truth", [])) != audit_rows:
        raise RuntimeError("cached PTA cell audit corpus width is incorrect")
    return value


def _run_pta_cells(
    *,
    project: Path,
    output: Path,
    source: Path,
    ptmrt: Path,
    seed: int,
    features: int,
    clauses: int,
    selected_epochs: Mapping[str, int],
    banks: Sequence[Mapping[str, object]],
    arm_snapshots: Mapping[str, Sequence[TMSnapshot]],
    audit_rows: int,
) -> dict[str, object]:
    scout = Path(__file__).with_name("run_mnist_pta_scout.py")
    convergence_module = Path(__file__).with_name("run_mnist_pta_convergence.py")
    sys.path.insert(0, str(convergence_module.parent))
    from run_mnist_pta_convergence import _aggregate

    summaries: dict[str, object] = {}
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (str(project / "python"), environment.get("PYTHONPATH", ""))
        if item
    )
    for arm, snapshots in arm_snapshots.items():
        cells: list[Mapping[str, object]] = []
        for digit, (snapshot, bank) in enumerate(zip(snapshots, banks)):
            cell = output / "pta" / arm / f"digit-{digit}-cell"
            cell.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path = output / "checkpoints" / arm / f"digit-{digit}.pkl"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            _publish_or_validate_checkpoint(snapshot_path, snapshot)
            result_path = cell / "result.json"
            if not result_path.is_file():
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
                    str(seed),
                    "--target-digit",
                    str(digit),
                    "--features",
                    str(features),
                    "--clauses",
                    str(clauses),
                    "--parent-epochs",
                    str(selected_epochs[arm]),
                    "--parent-snapshot",
                    str(snapshot_path),
                    "--selected-pixels",
                    *(str(value) for value in bank["pixels"]),
                    "--shared-audit-rows",
                    str(audit_rows),
                    "--emit-score-vectors",
                ]
                log_path = cell.parent / f"digit-{digit}.log"
                with log_path.open("w", encoding="utf-8") as stream:
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
                    raise RuntimeError(f"PTA cell failed; see {log_path}")
            cells.append(
                _validate_cached_cell(
                    json.loads(result_path.read_text(encoding="utf-8")),
                    digit=digit,
                    seed=seed,
                    features=features,
                    clauses=clauses,
                    selected_epoch=selected_epochs[arm],
                    pixels=bank["pixels"],
                    audit_rows=audit_rows,
                    snapshot=snapshot,
                )
            )
            print(f"pta arm={arm} digit={digit} complete", flush=True)
        summaries[arm] = _aggregate(selected_epochs[arm], cells)
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/mnist.pkl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ptmrt", type=Path, default=Path("out/build/Release/ptmrt.exe"))
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--clauses", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--teacher-epochs", type=int, default=3)
    parser.add_argument("--teacher-temperature", type=float, default=2.0)
    parser.add_argument("--student-temperature", type=float, default=3.0)
    parser.add_argument("--residual-learning-rate", type=float, default=2.0)
    parser.add_argument("--teacher-gate-floor", type=float, default=0.1)
    parser.add_argument("--validation-rows", type=int, default=2_000)
    parser.add_argument("--pta-audit-rows", type=int, default=2_000)
    parser.add_argument("--skip-pta", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        _validate_cli_parameters(args)
    except ValueError as error:
        parser.error(str(error))
    output = args.output.expanduser().resolve()
    source = args.source.expanduser().resolve()
    ptmrt = args.ptmrt.expanduser().resolve()
    if not source.is_file() or (not args.skip_pta and not ptmrt.is_file()):
        parser.error("MNIST source and (unless skipped) ptmrt must exist")
    campaign_started = perf_counter()
    train, validation, test = pickle.loads(source.read_bytes(), encoding="latin1")
    train_x, train_y = np.asarray(train[0]), np.asarray(train[1], dtype=np.int64)
    validation_x = np.asarray(validation[0])
    validation_y = np.asarray(validation[1], dtype=np.int64)
    test_x, test_y = np.asarray(test[0]), np.asarray(test[1], dtype=np.int64)
    if args.validation_rows > len(validation_y):
        parser.error("--validation-rows exceeds the validation split")
    if args.pta_audit_rows > len(test_y):
        parser.error("--pta-audit-rows exceeds the test split")
    torch, _, _, _ = _load_torch()
    plan = _build_plan(
        args,
        source=source,
        ptmrt=ptmrt,
        torch_version=torch.__version__,
    )
    try:
        _initialize_or_validate_plan(output, plan, resume=args.resume)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))

    teacher, teacher_report = _train_teacher(
        train_x,
        train_y,
        validation_x,
        validation_y,
        seed=args.seed,
        epochs=args.teacher_epochs,
        batch_size=256,
    )
    annotation_started = perf_counter()
    train_logits = _teacher_logits(teacher, train_x, batch_size=512, torch=torch)
    validation_logits = _teacher_logits(
        teacher, validation_x, batch_size=512, torch=torch
    )
    test_logits = _teacher_logits(teacher, test_x, batch_size=512, torch=torch)
    annotation_seconds = perf_counter() - annotation_started
    np.savez_compressed(
        output / "teacher_logits.npz",
        train=train_logits,
        validation=validation_logits,
        test=test_logits,
    )
    teacher_probabilities = _softmax(train_logits, args.teacher_temperature)
    teacher_test_probabilities = _softmax(test_logits, args.teacher_temperature)
    teacher_test_predictions = np.argmax(test_logits, axis=1)
    teacher_report["annotation_seconds"] = annotation_seconds
    teacher_report["test_accuracy"] = float(np.mean(teacher_test_predictions == test_y))
    teacher_permutation_seed = args.seed ^ 0x5EED5EED
    teacher_permutation = _within_class_permutation(
        train_y, seed=teacher_permutation_seed
    )
    teacher_permutation_digest = "sha256:" + sha256(
        np.asarray(teacher_permutation, dtype="<i8").tobytes()
    ).hexdigest()

    banks = [
        _prepare_bank(
            train_x,
            train_y,
            digit=digit,
            seed=args.seed,
            features=args.features,
        )
        for digit in range(CLASS_COUNT)
    ]
    common_snapshots: list[TMSnapshot] = []
    initial_training_started = perf_counter()
    for digit, bank in enumerate(banks):
        machine = ScalarBinaryTsetlinMachine(
            args.clauses,
            args.features,
            states_per_action=128,
            specificity=8.0,
            threshold=10,
            seed=args.seed,
        )
        machine.fit(bank["rows"], bank["targets"], epochs=1)
        common_snapshots.append(machine.snapshot())
    common_training_seconds = perf_counter() - initial_training_started
    arms = {
        name: [_machine_from_snapshot(snapshot) for snapshot in common_snapshots]
        for name in ARM_NAMES
    }
    fork_is_bit_exact = all(
        machine.snapshot() == common_snapshot
        for machines in arms.values()
        for machine, common_snapshot in zip(machines, common_snapshots)
    )
    validation_view_x = validation_x[: args.validation_rows]
    validation_view_y = validation_y[: args.validation_rows]
    histories: dict[str, list[dict[str, object]]] = {name: [] for name in arms}
    selected: dict[str, dict[str, object]] = {}

    for name, machines in arms.items():
        scores, evaluation_seconds = _score_collective(
            machines, banks, validation_view_x
        )
        metrics = _classification(validation_view_y, scores)
        histories[name].append(
            {
                "epoch": 1,
                "validation_accuracy": metrics["accuracy"],
                "training_seconds": common_training_seconds,
                "evaluation_seconds": evaluation_seconds,
            }
        )
        selected[name] = {
            "epoch": 1,
            "accuracy": metrics["accuracy"],
            "snapshots": [machine.snapshot() for machine in machines],
        }

    for epoch in range(2, args.epochs + 1):
        b_probability_reference: dict[tuple[int, int], float] = {}
        for name, machines in arms.items():
            training_started = perf_counter()
            telemetry = {
                "observations": 0,
                "controller_base_probability_sum": 0.0,
                "hard_base_probability_sum": 0.0,
                "applied_probability_sum": 0.0,
                "independent_b_probability_sum": 0.0,
                "disagreement_sum": 0.0,
                "priority_weight_sum": 0.0,
                "realized_clause_feedback_count": 0,
                "policy_seconds": 0.0,
                "update_seconds": 0.0,
                "calibration_seconds": 0.0,
                "calibration_target_sum": 0.0,
                "calibration_probability_sum": 0.0,
                "calibration_scales": [],
            }
            for digit, (machine, bank) in enumerate(zip(machines, banks)):
                rows = bank["rows"]
                targets = bank["targets"]
                indices = bank["indices"]
                assert isinstance(rows, tuple) and isinstance(targets, tuple)
                assert isinstance(indices, np.ndarray)
                if name == "A_hard_margin":
                    machine.fit(rows, targets, epochs=1)
                    continue

                normalization_scale = None
                if name in NORMALIZED_ARMS:
                    calibration = _calibrate_policy(
                        name,
                        machine,
                        bank,
                        digit=digit,
                        teacher_probabilities=teacher_probabilities,
                        teacher_permutation=teacher_permutation,
                        temperature=args.student_temperature,
                        learning_rate=args.residual_learning_rate,
                    )
                    normalization_scale = calibration["scale"]
                    telemetry["calibration_seconds"] += calibration["seconds"]
                    telemetry["calibration_target_sum"] += calibration[
                        "target_base_probability_sum"
                    ]
                    telemetry["calibration_probability_sum"] += calibration[
                        "calibrated_probability_sum"
                    ]
                    telemetry["calibration_scales"].append(normalization_scale)

                for row_position, (row, hard_target, source_index) in enumerate(
                    zip(rows, targets, indices)
                ):
                    policy_started = perf_counter()
                    hard_target_value = float(hard_target)
                    student_probability = _sigmoid_vote(
                        machine.raw_vote(row), args.student_temperature
                    )
                    hard_residual = abs(hard_target_value - student_probability)
                    hard_base_probability = min(
                        1.0, args.residual_learning_rate * hard_residual
                    )
                    teacher_probability = (
                        _teacher_probability_for_arm(
                            name,
                            teacher_probabilities,
                            teacher_permutation,
                            source_index=int(source_index),
                            digit=digit,
                        )
                        if name in TEACHER_DEPENDENT_ARMS
                        else 0.0
                    )
                    disagreement = abs(teacher_probability - student_probability)
                    target_probability = hard_target_value
                    if name == "C_teacher_residual":
                        target_probability = (
                            0.5 * hard_target_value + 0.5 * teacher_probability
                        )
                    elif name == "L_delayed_gradual_soft_targets":
                        alpha = _delayed_teacher_alpha(epoch)
                        target_probability = (
                            (1.0 - alpha) * hard_target_value
                            + alpha * teacher_probability
                        )
                    feedback_probability = _policy_feedback_probability(
                        name,
                        hard_base_probability=hard_base_probability,
                        hard_residual=hard_residual,
                        disagreement=disagreement,
                        hard_target=hard_target_value,
                        teacher_probability=teacher_probability,
                        normalization_scale=normalization_scale,
                        teacher_gate_floor=args.teacher_gate_floor,
                    )
                    priority_weight = (
                        _priority_weight(
                            name,
                            hard_residual=hard_residual,
                            disagreement=disagreement,
                            hard_target=hard_target_value,
                            teacher_probability=teacher_probability,
                        )
                        if name in NORMALIZED_ARMS
                        else 0.0
                    )
                    telemetry["policy_seconds"] += perf_counter() - policy_started

                    update_started = perf_counter()
                    update = machine.update_residual(
                        row,
                        target_probability,
                        temperature=args.student_temperature,
                        learning_rate=args.residual_learning_rate,
                        feedback_probability=feedback_probability,
                    )
                    telemetry["update_seconds"] += perf_counter() - update_started
                    reference_key = (digit, row_position)
                    if name == "B_hard_residual":
                        b_probability_reference[reference_key] = hard_base_probability
                    elif reference_key not in b_probability_reference:
                        raise RuntimeError("independent B budget was not recorded first")
                    independent_b_probability = b_probability_reference[reference_key]

                    telemetry["observations"] += 1
                    telemetry["controller_base_probability_sum"] += (
                        update.base_feedback_probability
                    )
                    telemetry["hard_base_probability_sum"] += hard_base_probability
                    telemetry["applied_probability_sum"] += update.feedback_probability
                    telemetry["independent_b_probability_sum"] += (
                        independent_b_probability
                    )
                    telemetry["disagreement_sum"] += disagreement
                    telemetry["priority_weight_sum"] += priority_weight
                    telemetry["realized_clause_feedback_count"] += (
                        update.clauses_feedback_applied
                    )
            training_seconds = perf_counter() - training_started
            scores, evaluation_seconds = _score_collective(
                machines, banks, validation_view_x
            )
            metrics = _classification(validation_view_y, scores)
            history_item = {
                "epoch": epoch,
                "validation_accuracy": metrics["accuracy"],
                "training_seconds": training_seconds,
                "evaluation_seconds": evaluation_seconds,
            }
            if name != "A_hard_margin":
                observations = int(telemetry["observations"])
                controller_base_sum = float(
                    telemetry["controller_base_probability_sum"]
                )
                hard_base_sum = float(telemetry["hard_base_probability_sum"])
                applied_sum = float(telemetry["applied_probability_sum"])
                independent_b_sum = float(
                    telemetry["independent_b_probability_sum"]
                )
                expected_clause_feedback = applied_sum * args.clauses
                calibration_target = float(telemetry["calibration_target_sum"])
                calibration_applied = float(
                    telemetry["calibration_probability_sum"]
                )
                feedback = {
                    "observations": observations,
                    "controller_base_probability_sum": controller_base_sum,
                    "hard_base_probability_sum": hard_base_sum,
                    "applied_probability_sum": applied_sum,
                    "independent_b_probability_sum": independent_b_sum,
                    "mean_controller_base_probability": controller_base_sum
                    / observations,
                    "mean_hard_base_probability": hard_base_sum / observations,
                    "mean_applied_probability": applied_sum / observations,
                    "mean_independent_b_probability": independent_b_sum
                    / observations,
                    "mean_teacher_student_disagreement": (
                        float(telemetry["disagreement_sum"]) / observations
                        if name in TEACHER_DEPENDENT_ARMS
                        else None
                    ),
                    "mean_priority_weight": float(
                        telemetry["priority_weight_sum"]
                    )
                    / observations,
                    "applied_to_controller_base_ratio": (
                        applied_sum / controller_base_sum
                        if controller_base_sum > 0.0
                        else 1.0
                    ),
                    "applied_to_current_hard_base_ratio": (
                        applied_sum / hard_base_sum if hard_base_sum > 0.0 else 1.0
                    ),
                    "applied_to_independent_b_ratio": (
                        applied_sum / independent_b_sum
                        if independent_b_sum > 0.0
                        else 1.0
                    ),
                    "realized_clause_feedback_count": int(
                        telemetry["realized_clause_feedback_count"]
                    ),
                    "expected_clause_feedback_count": expected_clause_feedback,
                    "realized_to_expected_clause_feedback_ratio": (
                        int(telemetry["realized_clause_feedback_count"])
                        / expected_clause_feedback
                        if expected_clause_feedback > 0.0
                        else 1.0
                    ),
                    "policy_seconds": float(telemetry["policy_seconds"]),
                    "update_seconds": float(telemetry["update_seconds"]),
                    "calibration_seconds": float(
                        telemetry["calibration_seconds"]
                    ),
                }
                if name in NORMALIZED_ARMS:
                    online_ratio = (
                        applied_sum / hard_base_sum
                        if hard_base_sum > 0.0
                        else 1.0
                    )
                    scales = list(telemetry["calibration_scales"])
                    feedback["normalization"] = {
                        "protocol": "pre_epoch_fixed_scores_per_bank",
                        "calibration_target_probability_sum": calibration_target,
                        "calibration_applied_probability_sum": calibration_applied,
                        "calibration_budget_ratio": (
                            calibration_applied / calibration_target
                            if calibration_target > 0.0
                            else 1.0
                        ),
                        "minimum_scale": min(scales),
                        "maximum_scale": max(scales),
                        "mean_scale": sum(scales) / len(scales),
                        "online_budget_ratio": online_ratio,
                        "online_budget_within_5_percent": 0.95
                        <= online_ratio
                        <= 1.05,
                    }
                if name == "L_delayed_gradual_soft_targets":
                    feedback["teacher_target_alpha"] = _delayed_teacher_alpha(epoch)
                history_item["feedback"] = feedback
            histories[name].append(history_item)
            if metrics["accuracy"] > selected[name]["accuracy"]:
                selected[name] = {
                    "epoch": epoch,
                    "accuracy": metrics["accuracy"],
                    "snapshots": [machine.snapshot() for machine in machines],
                }
            print(
                f"arm={name} epoch={epoch} validation_accuracy={metrics['accuracy']:.6f}",
                flush=True,
            )

    learning_progress: dict[str, dict[str, object]] = {}
    for name, history in histories.items():
        cumulative_seconds = 0.0
        first_sixty = None
        for item in history:
            cumulative_seconds += float(item["training_seconds"]) + float(
                item["evaluation_seconds"]
            )
            item["cumulative_student_plus_validation_seconds"] = cumulative_seconds
            if first_sixty is None and float(item["validation_accuracy"]) >= 0.60:
                first_sixty = {
                    "epoch": int(item["epoch"]),
                    "cumulative_student_plus_validation_seconds": cumulative_seconds,
                }
        learning_progress[name] = {
            "first_validation_accuracy_at_least_0_60": first_sixty,
            "final_epoch_validation_accuracy": history[-1]["validation_accuracy"],
        }

    arm_metrics: dict[str, dict[str, object]] = {}
    vector_payload: dict[str, np.ndarray] = {"truth": test_y}
    selected_snapshots: dict[str, Sequence[TMSnapshot]] = {}
    for name, choice in selected.items():
        snapshots = choice["snapshots"]
        assert isinstance(snapshots, list)
        selected_snapshots[name] = snapshots
        checkpoint_directory = output / "checkpoints" / name
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        for digit, snapshot in enumerate(snapshots):
            (checkpoint_directory / f"digit-{digit}.pkl").write_bytes(
                pickle.dumps(snapshot, protocol=5)
            )
        machines = [_machine_from_snapshot(snapshot) for snapshot in snapshots]
        scores, inference_seconds = _score_collective(machines, banks, test_x)
        clipped_scores = _clip_collective_scores(scores, machines)
        metrics = _classification(test_y, scores)
        predictions = metrics.pop("predictions")
        clipped_metrics = _classification(test_y, clipped_scores)
        clipped_predictions = clipped_metrics.pop("predictions")
        student_probabilities = _sigmoid_scores(scores, args.student_temperature)
        clipped_probabilities = _sigmoid_scores(
            clipped_scores, args.student_temperature
        )
        feedback_items = [
            item["feedback"] for item in histories[name] if "feedback" in item
        ]
        feedback_summary = None
        if feedback_items:
            hard_base_sum = sum(
                float(item["hard_base_probability_sum"])
                for item in feedback_items
            )
            applied_sum = sum(
                float(item["applied_probability_sum"]) for item in feedback_items
            )
            independent_b_sum = sum(
                float(item["independent_b_probability_sum"])
                for item in feedback_items
            )
            feedback_summary = {
                "observations": sum(int(item["observations"]) for item in feedback_items),
                "mean_hard_base_probability": hard_base_sum
                / sum(int(item["observations"]) for item in feedback_items),
                "mean_applied_probability": applied_sum
                / sum(int(item["observations"]) for item in feedback_items),
                "applied_to_current_hard_base_ratio": (
                    applied_sum / hard_base_sum if hard_base_sum > 0.0 else 1.0
                ),
                "applied_to_independent_b_ratio": (
                    applied_sum / independent_b_sum
                    if independent_b_sum > 0.0
                    else 1.0
                ),
                "realized_clause_feedback_count": sum(
                    int(item["realized_clause_feedback_count"])
                    for item in feedback_items
                ),
                "policy_seconds": sum(
                    float(item["policy_seconds"]) for item in feedback_items
                ),
                "update_seconds": sum(
                    float(item["update_seconds"]) for item in feedback_items
                ),
                "calibration_seconds": sum(
                    float(item["calibration_seconds"]) for item in feedback_items
                ),
            }
            if name in NORMALIZED_ARMS:
                online_ratio = feedback_summary[
                    "applied_to_current_hard_base_ratio"
                ]
                feedback_summary["online_budget_within_5_percent"] = (
                    0.95 <= online_ratio <= 1.05
                )
                feedback_summary["budget_classification"] = (
                    "budget-matched"
                    if feedback_summary["online_budget_within_5_percent"]
                    else "approximate/budget-mismatched"
                )
        arm_metrics[name] = {
            **metrics,
            "selected_epoch": choice["epoch"],
            "selected_validation_accuracy": choice["accuracy"],
            "shared_epoch_one_training_seconds": common_training_seconds,
            "continuation_training_seconds": sum(
                item["training_seconds"] for item in histories[name][1:]
            ),
            "validation_evaluation_seconds": sum(
                item["evaluation_seconds"] for item in histories[name]
            ),
            "test_inference_seconds": inference_seconds,
            "test_examples_per_second": len(test_y) / inference_seconds,
            "learning_progress": learning_progress[name],
            "feedback_summary": feedback_summary,
            "teacher_top_one_agreement": float(
                np.mean(predictions == teacher_test_predictions)
            ),
            "teacher_probability_mae": float(
                np.mean(np.abs(student_probabilities - teacher_test_probabilities))
            ),
            "clipped_vote_comparison": {
                **clipped_metrics,
                "paired_against_raw": _paired(
                    test_y, predictions, clipped_predictions
                ),
                "prediction_disagreements": int(
                    np.count_nonzero(predictions != clipped_predictions)
                ),
                "teacher_top_one_agreement": float(
                    np.mean(clipped_predictions == teacher_test_predictions)
                ),
                "teacher_probability_mae": float(
                    np.mean(
                        np.abs(
                            clipped_probabilities - teacher_test_probabilities
                        )
                    )
                ),
            },
            "student_snapshot_pickle_bytes": sum(
                len(pickle.dumps(snapshot, protocol=5)) for snapshot in snapshots
            ),
            "ta_state_cells": sum(
                snapshot.number_of_clauses * snapshot.number_of_features * 2
                for snapshot in snapshots
            ),
        }
        vector_payload[f"{name}_raw_votes"] = scores
        vector_payload[f"{name}_clipped_scores"] = clipped_scores
        vector_payload[f"{name}_predictions"] = predictions
        vector_payload[f"{name}_clipped_predictions"] = clipped_predictions
    baseline_predictions = vector_payload["A_hard_margin_predictions"]
    for name in ARM_NAMES[1:]:
        arm_metrics[name]["paired_against_A"] = _paired(
            test_y, baseline_predictions, vector_payload[f"{name}_predictions"]
        )
    hard_residual_predictions = vector_payload["B_hard_residual_predictions"]
    for name in ARM_NAMES:
        if name == "B_hard_residual":
            continue
        arm_metrics[name]["paired_against_B"] = _paired(
            test_y,
            hard_residual_predictions,
            vector_payload[f"{name}_predictions"],
        )
        arm_metrics[name]["test_accuracy_delta_vs_B"] = float(
            arm_metrics[name]["accuracy"]
        ) - float(arm_metrics["B_hard_residual"]["accuracy"])
    np.savez_compressed(output / "student_test_vectors.npz", **vector_payload)

    pta_summary = None
    if not args.skip_pta:
        pta_summary = _run_pta_cells(
            project=Path(__file__).resolve().parents[2],
            output=output,
            source=source,
            ptmrt=ptmrt,
            seed=args.seed,
            features=args.features,
            clauses=args.clauses,
            selected_epochs={name: int(choice["epoch"]) for name, choice in selected.items()},
            banks=banks,
            arm_snapshots=selected_snapshots,
            audit_rows=args.pta_audit_rows,
        )

    invocation_seconds = perf_counter() - campaign_started
    teacher_preparation_seconds = (
        teacher_report["training_seconds"] + annotation_seconds
    )
    for name, metrics in arm_metrics.items():
        standalone_student_seconds = (
            common_training_seconds
            + float(metrics["continuation_training_seconds"])
            + float(metrics["validation_evaluation_seconds"])
            + float(metrics["test_inference_seconds"])
        )
        teacher_cost = (
            teacher_preparation_seconds if name in TEACHER_DEPENDENT_ARMS else 0.0
        )
        metrics["standalone_cost"] = {
            "teacher_dependent": name in TEACHER_DEPENDENT_ARMS,
            "teacher_preparation_seconds": teacher_cost,
            "student_training_validation_and_test_seconds": standalone_student_seconds,
            "end_to_end_seconds": standalone_student_seconds + teacher_cost,
            "shared_campaign_teacher_cost_is_charged_once": True,
        }
    student_component_seconds = common_training_seconds + sum(
        float(item["continuation_training_seconds"])
        + float(item["validation_evaluation_seconds"])
        + float(item["test_inference_seconds"])
        for item in arm_metrics.values()
    )
    pta_component_seconds = (
        sum(float(item["observed_wall_seconds"]) for item in pta_summary.values())
        if pta_summary is not None
        else 0.0
    )
    report = {
        "schema": SCHEMA,
        "claim_boundary": (
            "Internal bounded discovery run. Student validation uses the leading "
            "validation subset; arm metrics are pre-lifecycle and use raw-vote "
            "multiclass ranking over all test rows. PTA outcomes are reported "
            "separately under pta when enabled; config.pta_skipped records when "
            "the Prolog/PTA lifecycle was not invoked."
        ),
        "config": {
            "seed": args.seed,
            "features_per_bank": args.features,
            "clauses_per_bank": args.clauses,
            "total_epochs": args.epochs,
            "common_hard_label_epochs": 1,
            "student_training_rows_per_bank_per_epoch": 400,
            "continuation_examples_per_arm": (
                (args.epochs - 1) * 400 * CLASS_COUNT
            ),
            "continuation_clause_gate_trials_per_arm": (
                (args.epochs - 1) * 400 * CLASS_COUNT * args.clauses
            ),
            "pixel_threshold": PIXEL_THRESHOLD,
            "teacher_input": "784 grayscale pixels",
            "student_input": (
                f"{args.features} bank-specific pixels binarized at 0.3"
            ),
            "teacher_temperature": args.teacher_temperature,
            "student_sigmoid_temperature": args.student_temperature,
            "residual_learning_rate": args.residual_learning_rate,
            "teacher_target_mix": "0.5 one_hot + 0.5 teacher_softmax",
            "arms": list(ARM_NAMES),
            "policy_matrix": {
                "A_hard_margin": "canonical hard-margin feedback",
                "B_hard_residual": "hard targets; b=min(1,2*abs(y-p))",
                "C_teacher_residual": "q=0.5*y+0.5*t; residual feedback",
                TEACHER_GATED_ARM: "hard targets; b*w_linear(d)",
                "E_uniform_slowdown": "hard targets; 0.36*b",
                "F_normalized_linear": "hard targets; N(w_linear(d))",
                "G_normalized_logarithmic": "hard targets; N(w_log(d))",
                "H_normalized_squared": "hard targets; N(w_square(d))",
                "I_protected_baseline_blend": (
                    "hard targets; 0.75*b+0.25*R(w_log(d))"
                ),
                "J_student_only_priorities": "hard targets; N(w_log(e))",
                "K_shuffled_teacher_control": (
                    "G with fixed within-ground-truth-class teacher permutation"
                ),
                "L_delayed_gradual_soft_targets": (
                    "B through epoch 4; residual q=(1-alpha)*y+alpha*t thereafter"
                ),
                "M_normalized_agreement": "hard targets; N(w_log(1-d))",
                "N_normalized_teacher_advantage": (
                    "hard targets; N(w_log(max(0,abs(y-p)-abs(y-t))))"
                ),
            },
            "normalization": {
                "calibration": (
                    "once per bank at each epoch start from fixed pre-update scores"
                ),
                "online": (
                    "hold scale fixed while recomputing current b and priorities"
                ),
                "clipping": "scalar solve accounts for min(1, scaled probability)",
                "budget_flag": "online applied/current-hard ratio outside 0.95..1.05",
            },
            "shuffled_teacher_control": {
                "seed": teacher_permutation_seed,
                "permutation_digest": teacher_permutation_digest,
                "scope": "fixed permutation within ground-truth training class",
                "student_feedback_rng_is_separate": True,
            },
            "teacher_feedback_gate": {
                "arm": TEACHER_GATED_ARM,
                "direction_and_target": "hard-label residual, identical to B",
                "disagreement": "abs(teacher_probability - student_probability)",
                "multiplier": (
                    f"{args.teacher_gate_floor} + "
                    f"{1.0 - args.teacher_gate_floor} * disagreement"
                ),
                "floor": args.teacher_gate_floor,
                "ceiling": 1.0,
                "student_probability_source": (
                    "sigmoid(unclipped_raw_vote / temperature)"
                ),
            },
            "multiclass_score_semantics": "unclipped signed clause votes",
            "binary_score_semantics": "margin-clipped signed clause votes",
            "validation_rows": args.validation_rows,
            "test_rows": len(test_y),
            "pta_audit_rows": args.pta_audit_rows,
            "pta_skipped": args.skip_pta,
            "execution": {
                "arm_concurrency": 1,
                "student_execution": "sequential in ARM_NAMES order",
                "torch_threads": int(torch.get_num_threads()),
            },
        },
        "teacher": teacher_report,
        "common_checkpoint": {
            "training_seconds": common_training_seconds,
            "bit_exact_fork": fork_is_bit_exact,
            "selected_pixels_by_digit": {
                str(digit): list(bank["pixels"]) for digit, bank in enumerate(banks)
            },
            "unique_binary_training_rows_by_digit": {
                str(digit): bank["unique_binary_rows"]
                for digit, bank in enumerate(banks)
            },
        },
        "history": histories,
        "arm_metric_phase": "pre_lifecycle",
        "arms": arm_metrics,
        "pta": pta_summary,
        "timing": {
            "invocation_seconds": invocation_seconds,
            "teacher_preparation_seconds": teacher_preparation_seconds,
            "student_measured_component_seconds": student_component_seconds,
            "pta_measured_component_seconds": pta_component_seconds,
            "measured_component_total_seconds": (
                teacher_preparation_seconds
                + student_component_seconds
                + pta_component_seconds
            ),
            "timing_note": (
                "Invocation time may describe a resume that reused durable PTA "
                "cells; measured component totals retain the cell-local timings."
            ),
            "measured_teacher_student_overlap_seconds": 0.0,
        },
        "artifacts": {
            "teacher_logits": "teacher_logits.npz",
            "student_test_vectors": "student_test_vectors.npz",
            "selected_checkpoints": "checkpoints/<arm>/digit-<n>.pkl",
        },
    }
    _write_json(output / "result.json", report)
    print(json.dumps({"result": str(output / "result.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
