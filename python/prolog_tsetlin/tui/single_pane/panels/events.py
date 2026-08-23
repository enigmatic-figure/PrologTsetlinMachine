from __future__ import annotations
from textual.containers import Vertical
from textual.widgets import Static, DataTable
from textual.app import ComposeResult
from ....services.telemetry import TelemetryEvent

class EventsPanel(Vertical):
    """Persistent telemetry log - the classic had a Log at the bottom, single-pane keeps it as a task view."""

    def compose(self) -> ComposeResult:
        yield Static("EVENTS  TelemetrySession  session/run/sequence  level/source/message", classes="card_title")
        yield DataTable(id="events-table", zebra_stripes=True)

    def on_mount(self) -> None:
        t = self.query_one("#events-table", DataTable)
        t.add_columns("SEQ","TIME","LEVEL","SOURCE","MESSAGE")

    def set_events(self, events: list[TelemetryEvent]) -> None:
        t = self.query_one("#events-table", DataTable)
        t.clear()
        for ev in events[-50:]:  # last 50
            t.add_row(str(ev.sequence), ev.timestamp_utc[11:19], ev.level.upper(), ev.source, str(ev.payload.get("message",""))[:60])
