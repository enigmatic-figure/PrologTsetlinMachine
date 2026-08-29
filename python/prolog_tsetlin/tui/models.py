"""State models that do not depend on Textual widgets."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..services.artifacts import ArtifactSummary
from ..services.diagnostics import SampledTrainingDiagnostics
from ..services.inference import ArtifactInputField
from ..services.search import BoundedSearchRequest, BoundedSearchResult
from ..services.telemetry import TelemetryEvent
from ..services.training import (
    MulticlassTrainingRun,
    TrainingDiagnosticSampling,
    TrainingRequest,
    TrainingRun,
)


class JobState(str, Enum):
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class SessionState:
    """UI-neutral workbench state with explicit training provenance.

    ``configured_request`` is the currently visible parsed configuration, or
    ``None`` when the form is invalid. ``active_request`` is the immutable
    request owned by an executing training job. ``last_completed_run`` is
    retained until a newer run completes successfully. Sampled diagnostics are
    separated the same way: active samples never overwrite the last completed
    run's temporal history until completion is accepted. ``accuracy_history``
    belongs to the current or most recent attempt; its completed counterpart is
    retained separately for coherent last-completed projections.
    ``inspected_sample_epoch`` optionally selects one immutable historical
    sample from that completed history; ``None`` always means the final export
    snapshot.
    """

    name: str = "xor-demo"
    configured_request: TrainingRequest | None = field(
        default_factory=TrainingRequest
    )
    active_request: TrainingRequest | None = None
    job_state: JobState = JobState.IDLE
    last_completed_run: TrainingRun | MulticlassTrainingRun | None = None
    error: str | None = None
    configuration_dirty: bool = False
    progress_epoch: int = 0
    progress_accuracy: float = 0.0
    accuracy_history: list[float] = field(default_factory=list)
    last_completed_accuracy_history: tuple[float, ...] = ()
    active_diagnostic_sampling: TrainingDiagnosticSampling | None = None
    active_diagnostics: list[SampledTrainingDiagnostics] = field(
        default_factory=list
    )
    last_completed_diagnostics: tuple[SampledTrainingDiagnostics, ...] = ()
    inspected_sample_epoch: int | None = None
    artifact: ArtifactSummary | None = None
    loaded_artifact_path: Path | None = None
    artifact_inspection: dict[str, Any] | None = None
    artifact_fields: tuple[ArtifactInputField, ...] = ()
    search_state: JobState = JobState.IDLE
    search_request: BoundedSearchRequest | None = None
    search_result: BoundedSearchResult | None = None
    search_error: str | None = None
    events: list[TelemetryEvent] = field(default_factory=list)
