"""Deterministic training workflows shared by interactive frontends."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Callable

from ..reference import ScalarBinaryTsetlinMachine, TMSnapshot
from ..representation import FeatureSchema, FieldKind, LiteralCatalog


class TrainingCancelled(RuntimeError):
    """Raised at an epoch boundary when a caller requests cancellation."""


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    number_of_clauses: int = 20
    states_per_action: int = 100
    specificity: float = 3.0
    threshold: int = 10
    seed: int = 7
    epochs: int = 150

    def validate(self) -> None:
        if self.number_of_clauses <= 0:
            raise ValueError("number_of_clauses must be positive")
        if self.states_per_action <= 0:
            raise ValueError("states_per_action must be positive")
        if self.specificity <= 1:
            raise ValueError("specificity must be greater than one")
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    epoch: int
    epochs: int
    accuracy: float


@dataclass(frozen=True, slots=True)
class TrainingRun:
    request: TrainingRequest
    rows: tuple[tuple[bool, bool], ...]
    targets: tuple[int, ...]
    predictions: tuple[int, ...]
    accuracy: float
    snapshot: TMSnapshot


ProgressCallback = Callable[[TrainingProgress], None]


def train_xor(
    request: TrainingRequest,
    *,
    progress: ProgressCallback | None = None,
    cancel: Event | None = None,
) -> TrainingRun:
    """Train the scalar semantic oracle on the built-in XOR dataset."""
    request.validate()
    records = (
        {"x0": False, "x1": False},
        {"x0": False, "x1": True},
        {"x0": True, "x1": False},
        {"x0": True, "x1": True},
    )
    rows = tuple((row["x0"], row["x1"]) for row in records)
    targets = (0, 1, 1, 0)
    schema = FeatureSchema.from_fields(x0=FieldKind.BOOLEAN, x1=FieldKind.BOOLEAN)
    catalog = LiteralCatalog(schema)
    catalog.category_eq("x0", True)
    catalog.category_eq("x1", True)
    batch = catalog.encode(records, row_ids=tuple(f"xor-{i}" for i in range(4)))
    machine = ScalarBinaryTsetlinMachine(
        number_of_clauses=request.number_of_clauses,
        number_of_features=batch.ta.literal_count,
        states_per_action=request.states_per_action,
        specificity=request.specificity,
        threshold=request.threshold,
        seed=request.seed,
    )
    encoded_rows = tuple(batch.ta.row_values(i) for i in range(batch.ta.row_count))
    for epoch in range(1, request.epochs + 1):
        if cancel is not None and cancel.is_set():
            raise TrainingCancelled(f"training cancelled before epoch {epoch}")
        machine.fit_literal_batch(batch.ta, targets, epochs=1)
        if progress is not None:
            predictions = machine.predict(encoded_rows)
            accuracy = sum(a == b for a, b in zip(predictions, targets)) / 4
            progress(TrainingProgress(epoch, request.epochs, accuracy))
    predictions = tuple(machine.predict(encoded_rows))
    accuracy = sum(a == b for a, b in zip(predictions, targets)) / 4
    return TrainingRun(request, rows, targets, predictions, accuracy, machine.snapshot())
