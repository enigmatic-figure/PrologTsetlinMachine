"""State models that do not depend on Textual widgets."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..services.artifacts import ArtifactSummary
from ..services.inference import ArtifactInputField
from ..services.search import BoundedSearchRequest, BoundedSearchResult
from ..services.telemetry import TelemetryEvent
from ..services.training import TrainingRequest, TrainingRun


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
    name: str = "xor-demo"
    request: TrainingRequest = field(default_factory=TrainingRequest)
    job_state: JobState = JobState.IDLE
    run: TrainingRun | None = None
    error: str | None = None
    configuration_dirty: bool = False
    progress_epoch: int = 0
    progress_accuracy: float = 0.0
    accuracy_history: list[float] = field(default_factory=list)
    artifact: ArtifactSummary | None = None
    loaded_artifact_path: Path | None = None
    artifact_inspection: dict[str, Any] | None = None
    artifact_fields: tuple[ArtifactInputField, ...] = ()
    search_state: JobState = JobState.IDLE
    search_request: BoundedSearchRequest | None = None
    search_result: BoundedSearchResult | None = None
    search_error: str | None = None
    events: list[TelemetryEvent] = field(default_factory=list)
