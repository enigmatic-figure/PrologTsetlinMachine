"""Deterministic training workflows shared by interactive frontends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil, isfinite
from pathlib import Path
from threading import Event
from typing import Callable

from ..reference import ScalarBinaryTsetlinMachine, TMSnapshot
from ..representation import FeatureSchema, FieldKind, LiteralCatalog


class TrainingCancelled(RuntimeError):
    """Raised at an epoch boundary when a caller requests cancellation."""


class TrainingWorkload(str, Enum):
    """Built-in workloads exposed through the interactive training service."""

    XOR = "xor"
    MNIST = "mnist"


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    number_of_clauses: int = 20
    states_per_action: int = 100
    specificity: float = 3.0
    threshold: int = 10
    seed: int = 7
    epochs: int = 150
    workload: TrainingWorkload = TrainingWorkload.XOR
    boost_true_positive_feedback: bool = False

    def validate(self) -> None:
        if not isinstance(self.workload, TrainingWorkload):
            raise ValueError("workload is unsupported")
        if type(self.boost_true_positive_feedback) is not bool:
            raise ValueError("boost_true_positive_feedback must be boolean")
        if self.number_of_clauses <= 0:
            raise ValueError("number_of_clauses must be positive")
        if self.states_per_action <= 0:
            raise ValueError("states_per_action must be positive")
        if not isfinite(self.specificity) or self.specificity <= 1:
            raise ValueError("specificity must be finite and greater than one")
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    epoch: int
    epochs: int
    accuracy: float
    training_seconds: float | None = None
    cumulative_training_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class TrainingDiagnosticSampling:
    """Explicit cadence for immutable in-training diagnostic samples."""

    every_epochs: int
    include_first: bool = True
    include_final: bool = True

    def validate(self) -> None:
        if self.every_epochs <= 0:
            raise ValueError("diagnostic sampling cadence must be positive")

    def includes(self, epoch: int, epochs: int) -> bool:
        self.validate()
        if epochs <= 0:
            raise ValueError("training epochs must be positive")
        if not 1 <= epoch <= epochs:
            raise ValueError("diagnostic sample epoch lies outside the training run")
        return (
            (self.include_first and epoch == 1)
            or epoch % self.every_epochs == 0
            or (self.include_final and epoch == epochs)
        )

    def selected_epochs(self, epochs: int) -> tuple[int, ...]:
        """Return the exact ordered epochs selected by this policy."""

        self.validate()
        if epochs <= 0:
            raise ValueError("training epochs must be positive")
        selected = set(range(self.every_epochs, epochs + 1, self.every_epochs))
        if self.include_first:
            selected.add(1)
        if self.include_final:
            selected.add(epochs)
        return tuple(sorted(selected))

    @classmethod
    def bounded(
        cls,
        epochs: int,
        *,
        maximum_samples: int = 25,
    ) -> "TrainingDiagnosticSampling":
        """Choose a cadence whose first/final-inclusive count stays bounded."""

        if epochs <= 0:
            raise ValueError("training epochs must be positive")
        if maximum_samples < 2:
            raise ValueError("maximum diagnostic samples must be at least two")
        if maximum_samples == 2:
            every_epochs = epochs + 1
        else:
            every_epochs = max(1, ceil(epochs / (maximum_samples - 2)))
        return cls(every_epochs=every_epochs)


@dataclass(frozen=True, slots=True)
class TrainingRun:
    request: TrainingRequest
    rows: tuple[tuple[bool, bool], ...]
    targets: tuple[int, ...]
    predictions: tuple[int, ...]
    accuracy: float
    snapshot: TMSnapshot


@dataclass(frozen=True, slots=True)
class MulticlassTrainingRun:
    """Completed native multiclass run without a portable snapshot contract."""

    request: TrainingRequest
    class_labels: tuple[int, ...]
    validation_rows: int
    confusion_matrix: tuple[tuple[int, ...], ...]
    accuracy: float
    training_seconds: float
    backend: str
    material_manifest: str

    def validate(self) -> None:
        self.request.validate()
        if self.request.workload is not TrainingWorkload.MNIST:
            raise ValueError("multiclass run must identify the MNIST workload")
        class_count = len(self.class_labels)
        if class_count < 2 or self.class_labels != tuple(range(class_count)):
            raise ValueError(
                "multiclass labels must be canonical zero-based classes"
            )
        if self.validation_rows <= 0:
            raise ValueError("multiclass validation rows must be positive")
        if len(self.confusion_matrix) != class_count or any(
            len(row) != class_count for row in self.confusion_matrix
        ):
            raise ValueError("multiclass confusion matrix has the wrong shape")
        if any(
            type(value) is not int or value < 0
            for row in self.confusion_matrix
            for value in row
        ):
            raise ValueError(
                "multiclass confusion counts must be nonnegative integers"
            )
        if sum(sum(row) for row in self.confusion_matrix) != self.validation_rows:
            raise ValueError("multiclass confusion counts do not cover validation")
        correct = sum(
            self.confusion_matrix[index][index]
            for index in range(class_count)
        )
        if abs(correct / self.validation_rows - self.accuracy) > 1e-12:
            raise ValueError("multiclass accuracy disagrees with confusion counts")
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError("multiclass accuracy is outside zero to one")
        if not isfinite(self.training_seconds) or self.training_seconds < 0.0:
            raise ValueError("multiclass training time is invalid")
        if not self.backend or not self.material_manifest:
            raise ValueError("multiclass provenance is incomplete")


@dataclass(frozen=True, slots=True)
class TrainingDiagnosticSample:
    """One evaluated, immutable model snapshot captured during training."""

    request: TrainingRequest
    epoch: int
    rows: tuple[tuple[bool, ...], ...]
    targets: tuple[int, ...]
    predictions: tuple[int, ...]
    accuracy: float
    snapshot: TMSnapshot


ProgressCallback = Callable[[TrainingProgress], None]
DiagnosticCallback = Callable[[TrainingDiagnosticSample], None]


def train_workload(
    request: TrainingRequest,
    *,
    workspace: str | Path,
    progress: ProgressCallback | None = None,
    diagnostic: DiagnosticCallback | None = None,
    diagnostic_sampling: TrainingDiagnosticSampling | None = None,
    cancel: Event | None = None,
) -> TrainingRun | MulticlassTrainingRun:
    """Dispatch one validated built-in workload without weakening its contract."""

    request.validate()
    if request.workload is TrainingWorkload.XOR:
        return train_xor(
            request,
            progress=progress,
            diagnostic=diagnostic,
            diagnostic_sampling=diagnostic_sampling,
            cancel=cancel,
        )
    if diagnostic is not None or diagnostic_sampling is not None:
        raise ValueError(
            "MNIST does not yet expose snapshot-derived diagnostic samples"
        )
    from .mnist_training import train_mnist_native

    return train_mnist_native(
        request, workspace=workspace, progress=progress, cancel=cancel
    )


def train_xor(
    request: TrainingRequest,
    *,
    progress: ProgressCallback | None = None,
    diagnostic: DiagnosticCallback | None = None,
    diagnostic_sampling: TrainingDiagnosticSampling | None = None,
    cancel: Event | None = None,
) -> TrainingRun:
    """Train the scalar semantic oracle on the built-in XOR dataset."""
    request.validate()
    if request.workload is not TrainingWorkload.XOR:
        raise ValueError("XOR trainer requires the XOR workload")
    if request.boost_true_positive_feedback:
        raise ValueError("Python XOR oracle does not implement boosted feedback")
    if (diagnostic is None) != (diagnostic_sampling is None):
        raise ValueError(
            "diagnostic callback and sampling policy must be supplied together"
        )
    if diagnostic_sampling is not None:
        diagnostic_sampling.validate()
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
    final_predictions: tuple[int, ...] | None = None
    final_accuracy: float | None = None
    final_snapshot: TMSnapshot | None = None
    for epoch in range(1, request.epochs + 1):
        if cancel is not None and cancel.is_set():
            raise TrainingCancelled(f"training cancelled before epoch {epoch}")
        machine.fit_literal_batch(batch.ta, targets, epochs=1)
        capture_diagnostic = (
            diagnostic_sampling is not None
            and diagnostic_sampling.includes(epoch, request.epochs)
        )
        needs_evaluation = progress is not None or capture_diagnostic
        predictions: tuple[int, ...] | None = None
        accuracy: float | None = None
        if needs_evaluation:
            predictions = tuple(machine.predict(encoded_rows))
            accuracy = sum(a == b for a, b in zip(predictions, targets)) / 4
        if progress is not None:
            assert accuracy is not None
            progress(TrainingProgress(epoch, request.epochs, accuracy))
        if capture_diagnostic:
            assert diagnostic is not None
            assert predictions is not None
            assert accuracy is not None
            sample_snapshot = machine.snapshot()
            diagnostic(
                TrainingDiagnosticSample(
                    request=request,
                    epoch=epoch,
                    rows=rows,
                    targets=targets,
                    predictions=predictions,
                    accuracy=accuracy,
                    snapshot=sample_snapshot,
                )
            )
            if epoch == request.epochs:
                final_snapshot = sample_snapshot
        if epoch == request.epochs and predictions is not None:
            final_predictions = predictions
            final_accuracy = accuracy
    if final_predictions is None:
        final_predictions = tuple(machine.predict(encoded_rows))
        final_accuracy = sum(
            a == b for a, b in zip(final_predictions, targets)
        ) / len(targets)
    assert final_accuracy is not None
    return TrainingRun(
        request,
        rows,
        targets,
        final_predictions,
        final_accuracy,
        final_snapshot or machine.snapshot(),
    )
