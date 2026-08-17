"""Initial keyboard-first PTM workbench vertical slice."""

from __future__ import annotations

from pathlib import Path
from threading import Event

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.events import Resize
from textual.widgets import DataTable, Footer, Header, Label, Log, Static

from ..services.training import TrainingCancelled, TrainingProgress, train_xor
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
    #clauses { width: 2fr; border: solid $primary; }
    #events { height: 7; border-top: solid $primary; }
    .status { text-style: bold; }
    .hidden { display: none; }
    Screen.compact #content { layout: vertical; }
    Screen.compact #summary, Screen.compact #predictions { width: 1fr; height: 1fr; }
    Screen.compact #events { height: 5; }
    """
    BINDINGS = [
        ("t", "train", "Train XOR"),
        ("x", "cancel", "Cancel"),
        ("o", "show_overview", "Overview"),
        ("c", "show_clauses", "Clauses"),
        ("q", "quit", "Quit"),
        ("ctrl+l", "events", "Events"),
    ]

    def __init__(self, *, workspace: Path | None = None, demo: str = "xor") -> None:
        super().__init__()
        self.workspace = workspace
        self.demo = demo
        self.session = SessionState()
        self._cancel = Event()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Overview  Data [planned]  Train  Clauses  Artifacts", id="workspace-bar")
        with Horizontal(id="content"):
            with VerticalScroll(id="summary"):
                yield Static("READY", id="job", classes="status")
                yield Static("Built-in XOR · seed 7 · 20 clauses · 150 epochs", id="config")
                yield Static("Press t to train the deterministic scalar oracle.", id="next-action")
            yield DataTable(id="predictions", zebra_stripes=True)
            yield DataTable(id="clauses", zebra_stripes=True, classes="hidden")
        yield Log(id="events", max_lines=200, auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#predictions", DataTable)
        table.add_columns("x0", "x1", "target", "prediction", "status")
        clauses = self.query_one("#clauses", DataTable)
        clauses.add_columns("clause", "polarity", "included", "literals")
        self.query_one("#events", Log).write_line("INFO session ready; Try XOR is selected")

    def on_resize(self, event: Resize) -> None:
        self.screen.set_class(event.size.width < 80 or event.size.height < 24, "compact")

    def action_events(self) -> None:
        self.query_one("#events").toggle_class("hidden")

    def action_train(self) -> None:
        if self.session.job_state in (
            JobState.QUEUED,
            JobState.RUNNING,
            JobState.CANCELLING,
        ):
            return
        self._cancel = Event()
        self.session.job_state = JobState.QUEUED
        self.query_one("#job", Static).update("QUEUED")
        self.query_one("#events", Log).write_line("INFO training queued")
        self._train_worker()

    def action_cancel(self) -> None:
        if self.session.job_state not in (JobState.QUEUED, JobState.RUNNING):
            return
        self.session.job_state = JobState.CANCELLING
        self._cancel.set()
        self.query_one("#job", Static).update("CANCELLING")
        self.query_one("#events", Log).write_line("INFO cancellation requested")

    def action_show_overview(self) -> None:
        self.query_one("#predictions").remove_class("hidden")
        self.query_one("#clauses").add_class("hidden")

    def action_show_clauses(self) -> None:
        self.query_one("#predictions").add_class("hidden")
        self.query_one("#clauses").remove_class("hidden")

    @work(thread=True, exclusive=True)
    def _train_worker(self) -> None:
        self.call_from_thread(self._show_running)

        def report(value: TrainingProgress) -> None:
            if value.epoch == 1 or value.epoch == value.epochs or value.epoch % 10 == 0:
                self.call_from_thread(self._show_progress, value)

        try:
            result = train_xor(self.session.request, progress=report, cancel=self._cancel)
        except TrainingCancelled as error:
            self.call_from_thread(self._show_cancelled, str(error))
        except (ValueError, RuntimeError) as error:
            self.call_from_thread(self._show_failure, str(error))
        else:
            self.call_from_thread(self._show_result, result)

    def _show_running(self) -> None:
        if self.session.job_state is JobState.QUEUED:
            self.session.job_state = JobState.RUNNING
            self.query_one("#job", Static).update("TRAINING")
            self.query_one("#events", Log).write_line("INFO training started")

    def _show_progress(self, value: TrainingProgress) -> None:
        self.query_one("#events", Log).write_line(
            f"INFO epoch {value.epoch}/{value.epochs} accuracy={value.accuracy:.0%}"
        )

    def _show_failure(self, message: str) -> None:
        self.session.job_state = JobState.FAILED
        self.session.error = message
        self.query_one("#job", Static).update("FAILED")
        self.query_one("#events", Log).write_line(f"ERROR {message}")

    def _show_cancelled(self, message: str) -> None:
        self.session.job_state = JobState.CANCELLED
        self.query_one("#job", Static).update("CANCELLED")
        self.query_one("#events", Log).write_line(f"INFO {message}")

    def _show_result(self, result) -> None:
        self.session.run = result
        self.session.job_state = JobState.SUCCEEDED
        self.query_one("#job", Static).update(f"SUCCEEDED · accuracy {result.accuracy:.0%}")
        table = self.query_one("#predictions", DataTable)
        table.clear()
        for row, target, prediction in zip(result.rows, result.targets, result.predictions):
            table.add_row(int(row[0]), int(row[1]), target, prediction, "MATCH" if target == prediction else "MISS")
        clauses = self.query_one("#clauses", DataTable)
        clauses.clear()
        literal_names = ("x0", "not x0", "x1", "not x1")
        boundary = result.snapshot.states_per_action
        for index, states in enumerate(result.snapshot.states):
            included = tuple(
                literal_names[literal] for literal, state in enumerate(states) if state > boundary
            )
            clauses.add_row(
                index,
                "positive" if index % 2 == 0 else "negative",
                len(included),
                ", ".join(included) or "(empty)",
            )
        self.query_one("#events", Log).write_line("INFO training completed")
