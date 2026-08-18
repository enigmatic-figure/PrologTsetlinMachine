"""Small presentation-neutral telemetry envelope for interactive sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic_ns
from typing import Any, Mapping
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    sequence: int
    timestamp_utc: str
    monotonic_ns: int
    session_id: str
    run_id: str | None
    source: str
    kind: str
    level: str
    payload: Mapping[str, Any]

    @property
    def display_line(self) -> str:
        stamp = self.timestamp_utc[11:19]
        message = self.payload.get("message")
        if message is None:
            message = " ".join(f"{key}={value}" for key, value in self.payload.items())
        return f"{stamp} {self.level.upper():<5} {self.source:<10} {message}"


class TelemetrySession:
    """Sequence and identify bounded in-memory UI events."""

    def __init__(self) -> None:
        self.session_id = str(uuid4())
        self.run_id: str | None = None
        self._sequence = 0

    def begin_run(self) -> str:
        self.run_id = str(uuid4())
        return self.run_id

    def emit(
        self,
        source: str,
        kind: str,
        level: str = "info",
        **payload: Any,
    ) -> TelemetryEvent:
        self._sequence += 1
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        return TelemetryEvent(
            sequence=self._sequence,
            timestamp_utc=timestamp.replace("+00:00", "Z"),
            monotonic_ns=monotonic_ns(),
            session_id=self.session_id,
            run_id=self.run_id,
            source=source,
            kind=kind,
            level=level,
            payload=payload,
        )
