"""Initial keyboard-first PTM workbench vertical slice."""

from __future__ import annotations

from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.events import Resize
from textual.widgets import DataTable, Footer, Header, Label, Log, Static

from ..services.training import TrainingProgress, train_xor
from .models import JobState, SessionState


class PTMApp(App[None]):
    TITLE = "PTM Workbench"
    SUB_TITLE = "scalar/oracle"
    CSS = """
    Screen { layout: vertical; }
    #workspace-bar { height: 3; padding: 1 2; background: $panel; }
    #content { height: 1fr; padding: 1 2; }
    #summary { width: 1fr; border: solid $primary; padding: 1 2; }
    #predictions { width: 2fr; border: solid $primary; }
    #events { height: 7; border-top: solid $primary; }
    .status { text-style: bold; }
    .hidden { display: none; }
    Screen.compact #content { layout: vertical; }
    Screen.compact #summary, Screen.compact #predictions { width: 1fr; height: 1fr; }
    Screen.compact #events { height: 5; }
    """
    BINDINGS = [
        ("t", "train", "Train XOR"),
        ("q", "quit", "Quit"),
        ("ctrl+l", "events", "Events"),
    ]

    def __init__(self, *, workspace: Path | None = None, demo: str = "xor") -> None:
        super().__init__()
        self.workspace = workspace
        self.demo = demo
        self.session = SessionState()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Overview  Data [planned]  Train  Clauses  Artifacts", id="workspace-bar")
        with Horizontal(id="content"):
            with VerticalScroll(id="summary"):
                yield Static("READY", id="job", classes="status")
                yield Static("Built-in XOR · seed 7 · 20 clauses · 150 epochs", id="config")
                yield Static("Press t to train the deterministic scalar oracle.", id="next-action")
            yield DataTable(id="predictions", zebra_stripes=True)
        yield Log(id="events", max_lines=200, auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#predictions", DataTable)
        table.add_columns("x0", "x1", "target", "prediction", "status")
        self.query_one("#events", Log).write_line("INFO session ready; Try XOR is selected")

    def on_resize(self, event: Resize) -> None:
        self.screen.set_class(event.size.width < 80 or event.size.height < 24, "compact")

    def action_events(self) -> None:
        self.query_one("#events").toggle_class("hidden")

    def action_train(self) -> None:
        if self.session.job_state is JobState.RUNNING:
            return
        self.session.job_state = JobState.RUNNING
        self.query_one("#job", Static).update("TRAINING")
        self.query_one("#events", Log).write_line("INFO training started")
        self._train_worker()

    @work(thread=True, exclusive=True)
    def _train_worker(self) -> None:
        def report(value: TrainingProgress) -> None:
            if value.epoch == 1 or value.epoch == value.epochs or value.epoch % 10 == 0:
                self.call_from_thread(self._show_progress, value)

        try:
            result = train_xor(self.session.request, progress=report)
        except (ValueError, RuntimeError) as error:
            self.call_from_thread(self._show_failure, str(error))
        else:
            self.call_from_thread(self._show_result, result)

    def _show_progress(self, value: TrainingProgress) -> None:
        self.query_one("#events", Log).write_line(
            f"INFO epoch {value.epoch}/{value.epochs} accuracy={value.accuracy:.0%}"
        )

    def _show_failure(self, message: str) -> None:
        self.session.job_state = JobState.FAILED
        self.session.error = message
        self.query_one("#job", Static).update("FAILED")
        self.query_one("#events", Log).write_line(f"ERROR {message}")

    def _show_result(self, result) -> None:
        self.session.run = result
        self.session.job_state = JobState.SUCCEEDED
        self.query_one("#job", Static).update(f"SUCCEEDED · accuracy {result.accuracy:.0%}")
        table = self.query_one("#predictions", DataTable)
        table.clear()
        for row, target, prediction in zip(result.rows, result.targets, result.predictions):
            table.add_row(int(row[0]), int(row[1]), target, prediction, "MATCH" if target == prediction else "MISS")
        self.query_one("#events", Log).write_line("INFO training completed")
