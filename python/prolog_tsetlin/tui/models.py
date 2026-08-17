"""State models that do not depend on Textual widgets."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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
