"""Keyboard-first PTM terminal workbench."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Log,
    ProgressBar,
    Select,
    Sparkline,
    Static,
    TextArea,
)

from ..help_topics import TUI_BINDINGS, binding_for_action, render_tui_help
from ..prolog_bridge import (
    NoDecisionTreeSolution,
    NoFeatureTemplateSolution,
    NoTAClauseSolution,
    NoThresholdSolution,
    PrologBridgeError,
    PrologSearchCancelled,
)
from ..services.artifacts import ArtifactExportRequest
from ..services.environment import inspect_environment
from ..services.inference import ArtifactInputField
from ..services.search import (
    BoundedSearchResult,
    SearchKind,
    demo_search_document,
)
from ..services.telemetry import TelemetrySession
from ..services.training import (
    TrainingCancelled,
    TrainingProgress,
    TrainingRequest,
    TrainingRun,
)
from .controllers import (
    ArtifactSessionController,
    SearchSessionController,
    TrainingSessionController,
)
from .models import JobState, SessionState


def _navigation_label(action: str) -> str:
    binding = binding_for_action(action)
    return f"{binding.display_key}  {binding.label}"


def _button_label(action: str) -> str:
    binding = binding_for_action(action)
    return f"{binding.label} [{binding.display_key}]"


def _binding_key(action: str) -> str:
    return binding_for_action(action).display_key


class HelpScreen(ModalScreen[None]):
    """Compact contextual help that works at terminal sizes."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding(binding_for_action("help").key, "dismiss", "Close", show=False),
    ]

    CSS = """
    HelpScreen { align: center middle; background: $background 70%; }
    #help-dialog {
        width: 72; max-width: 92%; height: auto; max-height: 90%;
        border: round $accent; background: $surface; padding: 1 2;
    }
    #help-title { text-style: bold; color: $accent; margin-bottom: 1; }
    #help-copy { height: auto; }
    #help-close { margin-top: 1; width: 100%; }
    """

    def __init__(self, view: str) -> None:
        super().__init__()
        self.view = view

    def compose(self) -> ComposeResult:
        title, copy = render_tui_help(self.view)
        with Vertical(id="help-dialog"):
            yield Static(title, id="help-title")
            yield Static(copy, id="help-copy")
            yield Button("Close", id="help-close", variant="primary")

    def action_dismiss(self) -> None:
        self.dismiss()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help-close":
            self.dismiss()


class PTMApp(App[None]):
    """Professional, log-centric explorer for the executable PTM foundation."""

    TITLE = "PTM Workbench"
    SUB_TITLE = "scalar oracle / XOR explorer"
    ENABLE_COMMAND_PALETTE = True
    CSS = """
    Screen { layout: vertical; background: $background; }
    Header { background: $primary-background; color: $text; }
    #workspace-bar {
        height: 3; padding: 1 2; background: $panel;
        border-bottom: solid $primary; color: $text-muted;
    }
    #shell { height: 1fr; }
    #rail {
        width: 20; height: 100%; padding: 1;
        background: $surface; border-right: solid $primary-background;
    }
    #brand { height: 4; padding: 0 1; color: $accent; text-style: bold; }
    #rail Button { width: 100%; margin-bottom: 1; text-align: left; }
    #rail Button.active { border-left: thick $accent; background: $primary-background; }
    #rail-note { height: auto; margin-top: 1; padding: 1; color: $text-muted; }
    #workspace { width: 1fr; height: 100%; }
    .view { width: 100%; height: 100%; }
    #view-overview { padding: 1 2; }
    #intro { height: 3; color: $text-muted; }
    #metrics { height: 6; }
    .metric {
        width: 1fr; height: 5; margin-right: 1; padding: 1 2;
        border: round $primary-background; background: $surface;
    }
    #overview-grid { height: 1fr; }
    .card {
        width: 1fr; height: 100%; margin-right: 1;
        border: round $primary-background; background: $surface;
    }
    .card-title { height: 3; padding: 1 2; color: $accent; text-style: bold; }
    .card DataTable { height: 1fr; }
    #overview-next {
        height: 4; margin-top: 1; padding: 1 2;
        border-left: thick $accent; background: $surface;
    }
    #view-train { layout: horizontal; }
    #summary {
        width: 38; height: 100%; padding: 1 2;
        border-right: solid $primary-background; background: $surface;
    }
    .panel-heading { height: 2; color: $accent; text-style: bold; }
    .field-label { margin-top: 1; color: $text-muted; }
    Input { height: 3; }
    Input:focus { border: tall $accent; }
    #train-actions { height: 4; margin-top: 1; }
    #train-actions Button { width: 1fr; margin-right: 1; }
    #validation { color: $error; min-height: 1; height: auto; }
    #results { width: 1fr; height: 100%; padding: 1 2; }
    #job { height: 2; text-style: bold; }
    #progress { height: 1; margin-bottom: 1; }
    #accuracy-line { height: 2; color: $text-muted; }
    #accuracy-spark { height: 3; margin-bottom: 1; color: $accent; }
    #predictions { height: 1fr; border: round $primary-background; }
    #next-action { height: 3; padding: 1; color: $text-muted; }
    #view-clauses { padding: 1 2; }
    #clauses-copy { height: 3; color: $text-muted; }
    #clauses { height: 1fr; border: round $primary-background; }
    #clause-detail {
        height: 8; margin-top: 1; padding: 1 2;
        border: round $primary-background; background: $surface;
    }
    #view-artifacts { layout: horizontal; }
    #export-form {
        width: 40; height: 100%; padding: 1 2;
        border-right: solid $primary-background; background: $surface;
    }
    #export-button { width: 100%; margin-top: 1; }
    #artifact-result { width: 1fr; height: 100%; padding: 1 2; }
    #artifact-status { height: auto; min-height: 3; margin-bottom: 1; }
    #artifact-open-actions { height: 4; margin-top: 1; }
    #artifact-open-actions Button { width: 1fr; margin-right: 1; }
    #artifact-open-status { height: auto; min-height: 2; margin-bottom: 1; }
    #artifact-details {
        height: auto; min-height: 8; padding: 1 2;
        border: round $primary-background; background: $surface;
    }
    #record-heading { height: auto; margin-top: 1; color: $accent; text-style: bold; }
    #record-fields { height: auto; }
    .record-label { height: auto; margin-top: 1; color: $text-muted; }
    #run-record-button { width: 100%; margin-top: 1; }
    #artifact-inference {
        height: auto; min-height: 3; margin-top: 1; padding: 1;
        border-left: thick $accent; background: $surface;
    }
    #feature-trace { height: 10; margin-top: 1; border: round $primary-background; }
    #view-search { layout: horizontal; }
    #search-form {
        width: 48; height: 100%; padding: 1 2;
        border-right: solid $primary-background; background: $surface;
    }
    #search-json { height: 1fr; min-height: 10; margin-top: 1; }
    #search-actions { height: 4; margin-top: 1; }
    #search-actions Button { width: 1fr; margin-right: 1; }
    #search-result-panel { width: 1fr; height: 100%; padding: 1 2; }
    #search-status { height: auto; min-height: 2; text-style: bold; }
    #search-result { height: 1fr; min-height: 8; padding: 1; border: round $primary-background; }
    #counterexamples { height: 9; margin-top: 1; border: round $primary-background; }
    #search-export-path { margin-top: 1; }
    #search-export-button { width: 100%; margin-top: 1; }
    #events { height: 8; border-top: solid $primary; background: $surface; }
    .hidden { display: none; }
    Screen.wide #rail { width: 23; }
    Screen.compact #rail { display: none; }
    Screen.compact #view-train, Screen.compact #view-artifacts { layout: vertical; }
    Screen.compact #summary, Screen.compact #results,
    Screen.compact #export-form, Screen.compact #artifact-result,
    Screen.compact #search-form, Screen.compact #search-result-panel {
        width: 100%; height: 1fr; border: none;
    }
    Screen.compact #view-search { layout: vertical; }
    Screen.compact #overview-grid { layout: vertical; }
    Screen.compact .card { width: 100%; height: 1fr; }
    Screen.compact #events { height: 5; }
    """
    BINDINGS = [
        Binding(binding.key, binding.action, binding.label, show=binding.show)
        for binding in TUI_BINDINGS
    ]

    def __init__(self, *, workspace: Path | None = None, demo: str = "xor") -> None:
        super().__init__()
        self.workspace = (workspace or Path.cwd()).expanduser().resolve()
        self.demo = demo
        self.session = SessionState()
        self.training = TrainingSessionController(self.session)
        self.artifacts = ArtifactSessionController(self.session)
        self.search = SearchSessionController(self.session)
        self.telemetry = TelemetrySession()
        self._cancel = Event()
        self._search_cancel = Event()
        self._search_generation = 0
        self._hydrating = True
        self._run_started = 0.0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("OVERVIEW  /  XOR-DEMO  /  IDLE", id="workspace-bar")
        with Horizontal(id="shell"):
            with Vertical(id="rail"):
                yield Static("PTM\nWORKBENCH", id="brand")
                yield Button(
                    _navigation_label("show_overview"),
                    id="nav-overview",
                    classes="active",
                )
                yield Button(_navigation_label("show_train"), id="nav-train")
                yield Button(_navigation_label("show_clauses"), id="nav-clauses")
                yield Button(_navigation_label("show_artifacts"), id="nav-artifacts")
                yield Button(_navigation_label("show_search"), id="nav-search")
                yield Static(
                    "LOG-CENTRIC MODE\n\nBounded Prolog search is available when GNU "
                    "Prolog passes environment preflight.",
                    id="rail-note",
                )
            with ContentSwitcher(initial="view-overview", id="workspace"):
                with Vertical(id="view-overview", classes="view"):
                    yield Static(
                        "Explore a complete deterministic path: understand XOR, train the "
                        "scalar oracle, inspect its pattern voters, then freeze a portable model.",
                        id="intro",
                    )
                    with Horizontal(id="metrics"):
                        yield Static("RUN\nREADY", id="metric-run", classes="metric")
                        yield Static("ACCURACY\n--", id="metric-accuracy", classes="metric")
                        yield Static(
                            "CLAUSES\n20 configured", id="metric-clauses", classes="metric"
                        )
                        yield Static(
                            "ARTIFACT\nnot exported", id="metric-artifact", classes="metric"
                        )
                    with Horizontal(id="overview-grid"):
                        with Vertical(classes="card"):
                            yield Static("BUILT-IN DATA / XOR", classes="card-title")
                            yield DataTable(id="dataset", zebra_stripes=True)
                        with Vertical(classes="card"):
                            yield Static("ENVIRONMENT PREFLIGHT", classes="card-title")
                            yield DataTable(id="capabilities", zebra_stripes=True)
                    yield Static(
                        "NEXT  Press "
                        f"{_binding_key('train')} to train with reproducible defaults. "
                        "No native build is required.",
                        id="overview-next",
                    )
                with Horizontal(id="view-train", classes="view"):
                    with VerticalScroll(id="summary"):
                        yield Static("TRAIN / CONFIGURATION", classes="panel-heading")
                        yield Static(
                            "These settings are captured immutably for each run.",
                            id="config",
                        )
                        yield Label("Clauses — pattern voters", classes="field-label")
                        yield Input(id="config-clauses", type="integer")
                        yield Label("States per action — memory depth", classes="field-label")
                        yield Input(id="config-states", type="integer")
                        yield Label("Specificity — pattern detail (> 1)", classes="field-label")
                        yield Input(id="config-specificity", type="number")
                        yield Label("Threshold — signed vote scale", classes="field-label")
                        yield Input(id="config-threshold", type="integer")
                        yield Label("Epochs", classes="field-label")
                        yield Input(id="config-epochs", type="integer")
                        yield Label("Seed", classes="field-label")
                        yield Input(id="config-seed", type="integer")
                        yield Static("", id="validation")
                        with Horizontal(id="train-actions"):
                            yield Button(
                                _button_label("train"),
                                id="train-button",
                                variant="success",
                            )
                            yield Button(
                                _button_label("cancel"),
                                id="cancel-button",
                                disabled=True,
                            )
                    with Vertical(id="results"):
                        yield Static("READY", id="job")
                        yield ProgressBar(
                            total=150,
                            show_eta=False,
                            id="progress",
                        )
                        yield Static(
                            "Accuracy history appears during training.", id="accuracy-line"
                        )
                        yield Sparkline([0.0], id="accuracy-spark")
                        yield DataTable(id="predictions", zebra_stripes=True)
                        yield Static(
                            "Press "
                            f"{_binding_key('train')} to train the deterministic "
                            "scalar oracle.",
                            id="next-action",
                        )
                with Vertical(id="view-clauses", classes="view"):
                    yield Static("CLAUSES / LEARNED PATTERN VOTERS", classes="panel-heading")
                    yield Static(
                        "Even clauses vote positive; odd clauses vote negative. Select a row "
                        "to inspect exact automaton states.",
                        id="clauses-copy",
                    )
                    yield DataTable(
                        id="clauses",
                        zebra_stripes=True,
                        cursor_type="row",
                    )
                    yield Static(
                        "No snapshot yet. Press "
                        f"{_binding_key('train')} to train, then return here.",
                        id="clause-detail",
                    )
                with Horizontal(id="view-artifacts", classes="view"):
                    with VerticalScroll(id="export-form"):
                        yield Static("ARTIFACT / EXPORT", classes="panel-heading")
                        yield Static(
                            "Freeze the completed run as a portable, inference-only .ptm file. "
                            "Existing files are never overwritten silently."
                        )
                        yield Label("Output path", classes="field-label")
                        yield Input(id="artifact-path")
                        yield Label("Model name", classes="field-label")
                        yield Input(id="artifact-name")
                        yield Label("Author (optional)", classes="field-label")
                        yield Input(id="artifact-author")
                        yield Label("Description", classes="field-label")
                        yield Input(id="artifact-description")
                        yield Button(
                            "Export completed run "
                            f"[{_binding_key('export')}]",
                            id="export-button",
                            variant="primary",
                            disabled=True,
                        )
                        yield Static("WAITING FOR A COMPLETED RUN", id="artifact-status")
                    with VerticalScroll(id="artifact-result"):
                        yield Static("ARTIFACT / EXPLORE", classes="panel-heading")
                        yield Static(
                            "Open any portable artifact to inspect and verify it. Raw-record "
                            "controls appear when a preprocessing contract is embedded."
                        )
                        yield Label("Artifact path", classes="field-label")
                        yield Input(id="artifact-open-path")
                        with Horizontal(id="artifact-open-actions"):
                            yield Button(
                                f"Load + verify [{_binding_key('load_artifact')}]",
                                id="load-artifact-button",
                                variant="success",
                            )
                            yield Button(
                                "Verify again",
                                id="verify-artifact-button",
                                disabled=True,
                            )
                        yield Static("NO ARTIFACT LOADED", id="artifact-open-status")
                        yield Static(
                            "Choose a .ptm file to inspect its task, model, ports, producer, "
                            "size, preprocessing contract, and conformance cases.",
                            id="artifact-details",
                        )
                        yield Static("RAW RECORD", id="record-heading")
                        yield Vertical(id="record-fields")
                        yield Button(
                            f"Run typed record [{_binding_key('run_record')}]",
                            id="run-record-button",
                            variant="primary",
                            disabled=True,
                        )
                        yield Static(
                            "Load a raw-record-capable packed TM to run inference.",
                            id="artifact-inference",
                        )
                        yield DataTable(id="feature-trace", zebra_stripes=True)
                with Horizontal(id="view-search", classes="view"):
                    with Vertical(id="search-form"):
                        yield Static("PROLOG / BOUNDED SEARCH", classes="panel-heading")
                        yield Static(
                            "Every request declares a finite candidate space and deadline "
                            "before GNU Prolog starts. Edit the JSON or use a built-in demo."
                        )
                        yield Label("Search kind", classes="field-label")
                        yield Select(
                            (
                                ("Masked threshold", SearchKind.THRESHOLD.value),
                                ("Typed feature template", SearchKind.FEATURE_TEMPLATE.value),
                                ("Signed TA clause", SearchKind.TA_CLAUSE.value),
                                ("Decision tree", SearchKind.DECISION_TREE.value),
                                ("Counterexample repair", SearchKind.REPAIR.value),
                            ),
                            value=SearchKind.DECISION_TREE.value,
                            allow_blank=False,
                            id="search-kind",
                        )
                        yield Label("Deadline in seconds", classes="field-label")
                        yield Input("30", type="number", id="search-timeout")
                        yield TextArea(
                            language="json",
                            show_line_numbers=True,
                            id="search-json",
                        )
                        with Horizontal(id="search-actions"):
                            yield Button(
                                _button_label("search"),
                                id="search-button",
                                variant="success",
                            )
                            yield Button(
                                _button_label("cancel_search"),
                                id="search-cancel-button",
                                disabled=True,
                            )
                    with Vertical(id="search-result-panel"):
                        yield Static("READY / NO SEARCH RUN", id="search-status")
                        yield TextArea(
                            "The typed result and declared candidate bound appear here.",
                            read_only=True,
                            show_line_numbers=True,
                            language="json",
                            id="search-result",
                        )
                        yield DataTable(id="counterexamples", zebra_stripes=True)
                        yield Input(
                            str(
                                self.workspace
                                / "out"
                                / "artifacts"
                                / "prolog-search.ptm"
                            ),
                            id="search-export-path",
                        )
                        yield Button(
                            "Export fixed-Logic artifact",
                            id="search-export-button",
                            disabled=True,
                        )
        yield Log(id="events", max_lines=300, auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        request = self.session.configured_request
        if request is None:
            request = TrainingRequest()
            self.session.configured_request = request
        values = {
            "#config-clauses": request.number_of_clauses,
            "#config-states": request.states_per_action,
            "#config-specificity": request.specificity,
            "#config-threshold": request.threshold,
            "#config-epochs": request.epochs,
            "#config-seed": request.seed,
        }
        for selector, value in values.items():
            self.query_one(selector, Input).value = str(value)
        default_artifact = self.workspace / "out" / "artifacts" / "xor-explorer.ptm"
        self.query_one("#artifact-path", Input).value = str(default_artifact)
        self.query_one("#artifact-open-path", Input).value = str(default_artifact)
        self.query_one("#artifact-name", Input).value = "xor-explorer"
        self.query_one("#artifact-description", Input).value = (
            "XOR model trained in the PTM terminal workbench"
        )

        predictions = self.query_one("#predictions", DataTable)
        predictions.add_columns("x0", "x1", "target", "prediction", "status")
        clauses = self.query_one("#clauses", DataTable)
        clauses.add_columns("clause", "polarity", "included", "literals")
        feature_trace = self.query_one("#feature-trace", DataTable)
        feature_trace.add_columns("field", "transform", "value", "literal ID")
        counterexamples = self.query_one("#counterexamples", DataTable)
        counterexamples.add_columns(
            "example", "expected", "parent", "required flip"
        )
        self.query_one("#search-json", TextArea).text = json.dumps(
            demo_search_document(SearchKind.DECISION_TREE), indent=2
        )
        dataset = self.query_one("#dataset", DataTable)
        dataset.add_columns("x0", "x1", "target", "meaning")
        for x0, x1 in ((0, 0), (0, 1), (1, 0), (1, 1)):
            dataset.add_row(x0, x1, x0 ^ x1, "different" if x0 != x1 else "same")
        capabilities = self.query_one("#capabilities", DataTable)
        capabilities.add_columns("component", "state", "detail")
        for capability in inspect_environment(self.workspace):
            capabilities.add_row(
                capability.component,
                capability.status,
                capability.detail,
            )
        self._hydrating = False
        self._apply_breakpoint(self.size.width, self.size.height)
        self._emit("session", "session", message="workbench ready; built-in XOR selected")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "load-artifact-button":
            await self.action_load_artifact()
            return
        if button_id == "verify-artifact-button":
            self.action_verify_artifact()
            return
        if button_id == "run-record-button":
            self.action_run_record()
            return
        actions = {
            "nav-overview": self.action_show_overview,
            "nav-train": self.action_show_train,
            "nav-clauses": self.action_show_clauses,
            "nav-artifacts": self.action_show_artifacts,
            "nav-search": self.action_show_search,
            "train-button": self.action_train,
            "cancel-button": self.action_cancel,
            "export-button": self.action_export,
            "search-button": self.action_search,
            "search-cancel-button": self.action_cancel_search,
            "search-export-button": self.action_export_search,
        }
        action = actions.get(button_id)
        if action is not None:
            action()

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._hydrating or not event.input.id:
            return
        if event.input.id == "artifact-open-path":
            if self.session.loaded_artifact_path is not None:
                self.query_one("#artifact-open-status", Static).update(
                    "PATH CHANGED / Load the artifact before running it."
                )
                self.query_one("#run-record-button", Button).disabled = True
                self.query_one("#verify-artifact-button", Button).disabled = True
            return
        if not event.input.id.startswith("config-"):
            return
        try:
            current_request = self._request_from_form()
            current_request.validate()
        except ValueError:
            stale = self.training.synchronize_configuration(None)
        else:
            stale = self.training.synchronize_configuration(current_request)
        if self.session.last_completed_run is None:
            return
        self._sync_training_export_control()
        if stale:
            self.query_one("#job", Static).update("SUCCEEDED / STALE CONFIGURATION")
            self.query_one("#metric-run", Static).update("RUN\nSTALE CONFIG")
            self.query_one("#next-action", Static).update(
                "The visible results belong to the previous settings. Press "
                f"{_binding_key('train')} to retrain."
            )
        else:
            run = self.session.last_completed_run
            self.query_one("#job", Static).update(
                f"SUCCEEDED / accuracy {run.accuracy:.0%}"
            )
            self.query_one("#metric-run", Static).update("RUN\nSUCCEEDED")

    def on_select_changed(self, event: Select.Changed) -> None:
        if self._hydrating or event.select.id != "search-kind":
            return
        if self.search.active:
            self._search_cancel.set()
            self._emit(
                "search",
                "job_state",
                message="active search invalidated by kind change",
            )
        self._search_generation += 1
        kind = SearchKind(str(event.value))
        document = demo_search_document(kind)
        self.query_one("#search-json", TextArea).text = json.dumps(document, indent=2)
        self.query_one("#search-timeout", Input).value = str(
            document.get("timeout_seconds", 30)
        )
        self.search.reset()
        self.query_one("#search-status", Static).update(
            f"READY / {kind.value.upper()} DEMO LOADED"
        )
        self.query_one("#search-result", TextArea).text = (
            f"Edit the bounded request or press {_binding_key('search')} to run it."
        )
        self.query_one("#counterexamples", DataTable).clear()
        self.query_one("#search-export-button", Button).disabled = True

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if (
            event.data_table.id == "clauses"
            and self.session.last_completed_run is not None
        ):
            try:
                self._show_clause_detail(event.cursor_row)
            except NoMatches:
                # A final table event may race normal screen teardown in tests.
                return

    def on_resize(self, event: Resize) -> None:
        self._apply_breakpoint(event.size.width, event.size.height)

    def on_unmount(self) -> None:
        self._cancel.set()
        self._search_cancel.set()

    def _apply_breakpoint(self, width: int, height: int) -> None:
        self.screen.set_class(width < 90 or height < 24, "compact")
        self.screen.set_class(width >= 120 and height >= 30, "wide")

    def _navigate(self, destination: str) -> None:
        self.query_one("#workspace", ContentSwitcher).current = f"view-{destination}"
        for name in ("overview", "train", "clauses", "artifacts", "search"):
            self.query_one(f"#nav-{name}", Button).set_class(name == destination, "active")
        state = (
            self.session.search_state.value.upper()
            if destination == "search"
            else self.session.job_state.value.upper()
        )
        self.query_one("#workspace-bar", Static).update(
            f"{destination.upper()}  /  {self.session.name.upper()}  /  {state}"
        )

    def action_show_overview(self) -> None:
        self._navigate("overview")

    def action_show_train(self) -> None:
        self.query_one("#predictions").remove_class("hidden")
        self.query_one("#clauses").add_class("hidden")
        self._navigate("train")

    def action_show_clauses(self) -> None:
        self.query_one("#predictions").add_class("hidden")
        self.query_one("#clauses").remove_class("hidden")
        self._navigate("clauses")

    def action_show_artifacts(self) -> None:
        self._navigate("artifacts")

    def action_show_search(self) -> None:
        self._navigate("search")

    def _artifact_path_from_form(self) -> Path:
        raw = self.query_one("#artifact-open-path", Input).value.strip()
        if not raw:
            raise ValueError("artifact path cannot be empty")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        return path.resolve()

    @staticmethod
    def _artifact_details_text(report: dict[str, Any]) -> str:
        task = report.get("task")
        model = report.get("model")
        producer = report.get("producer")
        task_kind = task.get("kind", "unknown") if isinstance(task, dict) else "unknown"
        model_text = (
            ", ".join(f"{key}={value}" for key, value in model.items())
            if isinstance(model, dict)
            else "not declared"
        )
        producer_text = (
            f"{producer.get('name', 'unknown')} {producer.get('version', '')}".strip()
            if isinstance(producer, dict)
            else "unknown"
        )
        preprocessing = report.get("preprocessing_schema") or "none"
        return (
            f"{report.get('title') or '(untitled)'}\n"
            f"Kind       {report.get('artifact_kind')}\n"
            f"Task       {task_kind}\n"
            f"Model      {model_text}\n"
            f"Producer   {producer_text}\n"
            f"Size       {int(report.get('size_bytes') or 0):,} bytes\n"
            f"Preprocess {preprocessing}\n"
            f"Artifact ID\n{report.get('artifact_id')}"
        )

    @staticmethod
    def _field_label(field: ArtifactInputField) -> str:
        requirement = "required" if field.required else "optional / blank means missing"
        transforms = ", ".join(field.transforms)
        accepted = ""
        if field.accepted_values:
            accepted = " / values: " + ", ".join(
                repr(value) for value in field.accepted_values
            )
        return (
            f"{field.name} — {field.kind.value} — {requirement}\n"
            f"transforms: {transforms}{accepted}"
        )

    @staticmethod
    def _field_placeholder(field: ArtifactInputField) -> str:
        if field.kind.value == "boolean":
            return "true or false"
        if field.kind.value == "number":
            return "JSON number"
        return 'category value; quote numeric strings such as "1"'

    async def action_load_artifact(self) -> None:
        self.action_show_artifacts()
        fields_container = self.query_one("#record-fields", Vertical)
        try:
            path = self._artifact_path_from_form()
            loaded = self.artifacts.load(path)
            report = loaded.inspection
            verification = loaded.verification
            fields = loaded.fields
        except (OSError, ValueError, RuntimeError) as error:
            await fields_container.remove_children()
            await fields_container.mount(
                Static("No raw-record schema is available until an artifact loads.")
            )
            self.query_one("#artifact-open-status", Static).update(
                f"OPEN FAILED / {error}"
            )
            self.query_one("#verify-artifact-button", Button).disabled = True
            self.query_one("#run-record-button", Button).disabled = True
            self.query_one("#feature-trace", DataTable).clear()
            self._emit("artifact", "failure", "error", message=str(error))
            return

        await fields_container.remove_children()
        if fields:
            widgets = []
            for index, field in enumerate(fields):
                widgets.extend(
                    (
                        Label(self._field_label(field), classes="record-label"),
                        Input(
                            id=f"record-field-{index}",
                            type="number" if field.kind.value == "number" else "text",
                            placeholder=self._field_placeholder(field),
                            classes="record-value",
                        ),
                    )
                )
            await fields_container.mount(*widgets)
        else:
            await fields_container.mount(
                Static(
                    "This artifact accepts precomputed inputs and has no portable raw-record "
                    "contract. Inspection and verification remain available."
                )
            )
        self.query_one("#artifact-details", Static).update(
            self._artifact_details_text(report)
        )
        case_count = int(verification["conformance_case_count"])
        self.query_one("#artifact-open-status", Static).update(
            f"OPENED / VERIFIED / {case_count} CONFORMANCE CASE(S)"
        )
        self.query_one("#verify-artifact-button", Button).disabled = False
        self.query_one("#run-record-button", Button).disabled = not bool(fields)
        self.query_one("#feature-trace", DataTable).clear()
        self.query_one("#artifact-inference", Static).update(
            "Enter one typed record, then press "
            f"{_binding_key('run_record')} to materialize and run it."
            if fields
            else "Raw-record inference is unavailable for this artifact."
        )
        self._emit(
            "artifact",
            "artifact",
            message=f"opened and verified {path} id={report['artifact_id']}",
        )

    def action_verify_artifact(self) -> None:
        self.action_show_artifacts()
        try:
            report = self.artifacts.verify_loaded(
                displayed_path=self._artifact_path_from_form()
            )
        except (OSError, ValueError, RuntimeError) as error:
            self.query_one("#artifact-open-status", Static).update(
                f"VERIFY FAILED / {error}"
            )
            self._emit("artifact", "failure", "error", message=str(error))
            return
        self.query_one("#artifact-open-status", Static).update(
            "VERIFIED / integrity, contracts, and "
            f"{report['conformance_case_count']} conformance case(s)"
        )
        self._emit(
            "artifact", "artifact", message=f"verified {report['artifact_id']}"
        )

    def action_run_record(self) -> None:
        self.action_show_artifacts()
        fields = self.session.artifact_fields
        try:
            if not fields:
                raise ValueError("load a raw-record-capable artifact first")
            values = {
                field.name: self.query_one(f"#record-field-{index}", Input).value
                for index, field in enumerate(fields)
            }
            result = self.artifacts.run_record(
                values, displayed_path=self._artifact_path_from_form()
            )
        except (OSError, ValueError, RuntimeError) as error:
            self.query_one("#artifact-inference", Static).update(
                f"INFERENCE FAILED / {error}"
            )
            self._emit("artifact", "failure", "error", message=str(error))
            return

        trace = self.query_one("#feature-trace", DataTable)
        trace.clear()
        for item in result.feature_trace:
            trace.add_row(
                item["field"],
                item["expression"],
                item["value"],
                item["literal_id"],
            )
        prediction = result.prediction
        label = result.label
        self.query_one("#artifact-inference", Static).update(
            f"PREDICTION {label} / class index {prediction}\n"
            f"Materialized features: {list(result.features)}"
        )
        self._emit(
            "artifact",
            "prediction",
            message=f"record prediction={prediction} features={list(result.features)}",
        )

    def action_events(self) -> None:
        self.query_one("#events").toggle_class("hidden")

    def action_help(self) -> None:
        current = self.query_one("#workspace", ContentSwitcher).current
        view = current.removeprefix("view-") if current else "overview"
        self.push_screen(HelpScreen(view))

    def action_search(self) -> None:
        if self.search.active:
            return
        self.action_show_search()
        try:
            kind = SearchKind(str(self.query_one("#search-kind", Select).value))
            document = json.loads(self.query_one("#search-json", TextArea).text)
            if not isinstance(document, dict):
                raise ValueError("search JSON must contain an object")
            timeout = float(self.query_one("#search-timeout", Input).value)
            prepared = self.search.prepare(
                document,
                expected_kind=kind,
                timeout_seconds=timeout,
            )
            request = prepared.request
            budget = prepared.budget
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.search.failed(str(error))
            self._set_search_state(JobState.FAILED, "INVALID REQUEST")
            self.query_one("#search-result", TextArea).text = str(error)
            self._emit("search", "failure", "error", message=str(error))
            return
        self._search_generation += 1
        generation = self._search_generation
        cancel = Event()
        self._search_cancel = cancel
        self.query_one("#counterexamples", DataTable).clear()
        self.query_one("#search-export-button", Button).disabled = True
        self.query_one("#search-result", TextArea).text = json.dumps(
            {"status": "queued", "budget": budget}, indent=2, sort_keys=True
        )
        self._set_search_state(
            JobState.QUEUED,
            "QUEUED / <= "
            f"{int(budget['candidate_upper_bound']):,} CANDIDATES / "
            f"{budget['timeout_seconds']:g}s DEADLINE",
        )
        self._emit(
            "search",
            "job_state",
            message=(
                f"queued kind={request.kind.value} "
                f"candidates<={budget['candidate_upper_bound']} "
                f"timeout={request.timeout_seconds:g}s"
            ),
        )
        self._search_worker(request, cancel, generation)

    def action_cancel_search(self) -> None:
        if not self.search.request_cancel():
            return
        self._search_cancel.set()
        self._set_search_state(JobState.CANCELLING, "CANCELLING PROLOG SEARCH")
        self._emit("search", "job_state", message="cancellation requested")

    @work(thread=True, exclusive=True, group="prolog-search")
    def _search_worker(self, request, cancel: Event, generation: int) -> None:
        self.call_from_thread(self._show_search_running, generation)
        try:
            result = self.search.run(request, cancel=cancel.is_set)
        except PrologSearchCancelled as error:
            self.call_from_thread(
                self._show_search_cancelled, str(error), generation
            )
        except (
            NoThresholdSolution,
            NoFeatureTemplateSolution,
            NoTAClauseSolution,
            NoDecisionTreeSolution,
        ) as error:
            self.call_from_thread(
                self._show_search_no_solution, str(error), generation
            )
        except (KeyError, TypeError, ValueError, OSError, PrologBridgeError) as error:
            self.call_from_thread(
                self._show_search_failure, str(error), generation
            )
        else:
            self.call_from_thread(self._show_search_result, result, generation)

    def _search_callback_is_current(self, generation: int | None) -> bool:
        return generation is None or generation == self._search_generation

    def _show_search_running(self, generation: int | None = None) -> None:
        if not self._search_callback_is_current(generation):
            return
        if self.session.search_state is JobState.QUEUED:
            self.search.mark_running()
            self._set_search_state(JobState.RUNNING, "SEARCHING / GNU PROLOG")
            self._emit("search", "job_state", message="GNU Prolog started")

    def _show_search_cancelled(
        self, message: str, generation: int | None = None
    ) -> None:
        if not self._search_callback_is_current(generation):
            return
        self.search.cancelled()
        self._set_search_state(JobState.CANCELLED, "CANCELLED")
        self.query_one("#search-result", TextArea).text = message.splitlines()[0]
        self._emit("search", "job_state", message="search cancelled")

    def _show_search_no_solution(
        self, message: str, generation: int | None = None
    ) -> None:
        if not self._search_callback_is_current(generation):
            return
        self.search.no_solution()
        self._set_search_state(JobState.SUCCEEDED, "COMPLETE / NO EXACT SOLUTION")
        self.query_one("#search-result", TextArea).text = json.dumps(
            {"status": "no_solution", "message": message}, indent=2
        )
        self._emit("search", "search_result", message=f"no solution: {message}")

    def _show_search_failure(
        self, message: str, generation: int | None = None
    ) -> None:
        if not self._search_callback_is_current(generation):
            return
        self.search.failed(message)
        self._set_search_state(JobState.FAILED, "SEARCH FAILED")
        self.query_one("#search-result", TextArea).text = message
        self._emit("search", "failure", "error", message=message)

    def _show_search_result(
        self,
        result: BoundedSearchResult,
        generation: int | None = None,
    ) -> None:
        if not self._search_callback_is_current(generation):
            return
        self.search.complete(result)
        report = result.to_dict()
        candidate_bound = int(result.report.get("candidate_upper_bound", 0))
        self._set_search_state(
            JobState.SUCCEEDED,
            f"SOLVED / <= {candidate_bound:,} CANDIDATES / "
            f"{result.elapsed_seconds:.2f}s",
        )
        self.query_one("#search-result", TextArea).text = json.dumps(
            report, indent=2, sort_keys=True
        )
        counterexamples = self.query_one("#counterexamples", DataTable)
        counterexamples.clear()
        raw_counterexamples = result.report.get("counterexamples", ())
        if isinstance(raw_counterexamples, list):
            for item in raw_counterexamples:
                counterexamples.add_row(
                    str(item.get("example")),
                    str(item.get("expected")),
                    str(item.get("parent_prediction")),
                    str(item.get("required_flip")),
                )
        self.query_one("#search-export-button", Button).disabled = not result.exportable
        self._emit(
            "search",
            "search_result",
            message=(
                f"solved kind={result.kind.value} candidates<={candidate_bound} "
                f"elapsed={result.elapsed_seconds:.3f}s"
            ),
        )

    def action_export_search(self) -> None:
        result = self.session.search_result
        if result is None:
            self.query_one("#search-status", Static).update(
                "NO SEARCH RESULT TO EXPORT"
            )
            return
        raw = self.query_one("#search-export-path", Input).value.strip()
        path = Path(raw) if raw else Path("")
        if not path.is_absolute():
            path = self.workspace / path
        try:
            report = self.search.export(
                path, name=f"prolog-{result.kind.value}"
            )
        except FileExistsError:
            self.query_one("#search-status", Static).update(
                "EXPORT REFUSED / Choose a new path; no file was replaced."
            )
            return
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            self.query_one("#search-status", Static).update(
                f"EXPORT FAILED / {error}"
            )
            return
        self.query_one("#search-status", Static).update(
            f"EXPORTED / {report['size_bytes']:,} BYTES / CONFORMANCE VERIFIED"
        )
        self.query_one("#artifact-open-path", Input).value = str(report["output"])
        self._emit(
            "search", "artifact", message=f"exported {report['output']}"
        )

    def action_train(self) -> None:
        if self.training.active:
            return
        self.action_show_train()
        try:
            request = self._request_from_form()
            self.training.begin(request)
        except ValueError as error:
            self.training.synchronize_configuration(None)
            self._sync_training_export_control()
            message = str(error)
            self.session.error = message
            self.query_one("#validation", Static).update(message)
            self._emit("training", "failure", "error", message=f"configuration: {message}")
            return
        self._sync_training_export_control()
        self.query_one("#validation", Static).update("")
        self.query_one("#predictions", DataTable).clear()
        self.query_one("#clauses", DataTable).clear()
        self.query_one("#clause-detail", Static).update("Training is in progress.")
        self.query_one("#accuracy-spark", Sparkline).data = [0.0]
        self.query_one("#accuracy-line", Static).update("Waiting for epoch 1...")
        self.query_one("#progress", ProgressBar).update(total=request.epochs, progress=0)
        self._cancel = Event()
        self.telemetry.begin_run()
        self._run_started = perf_counter()
        self._set_job_state(JobState.QUEUED, "QUEUED")
        self._emit("training", "job_state", message="training queued")
        self._train_worker()

    def _request_from_form(self) -> TrainingRequest:
        def integer(selector: str, label: str) -> int:
            value = self.query_one(selector, Input).value.strip()
            try:
                return int(value)
            except ValueError as error:
                raise ValueError(f"{label} must be a whole number") from error

        value = self.query_one("#config-specificity", Input).value.strip()
        try:
            specificity = float(value)
        except ValueError as error:
            raise ValueError("specificity must be a number") from error
        return TrainingRequest(
            number_of_clauses=integer("#config-clauses", "clauses"),
            states_per_action=integer("#config-states", "states per action"),
            specificity=specificity,
            threshold=integer("#config-threshold", "threshold"),
            epochs=integer("#config-epochs", "epochs"),
            seed=integer("#config-seed", "seed"),
        )

    def _current_request_or_none(self) -> TrainingRequest | None:
        try:
            current_request = self._request_from_form()
            current_request.validate()
        except ValueError:
            return None
        return current_request

    def _sync_training_export_control(self) -> None:
        self.query_one("#export-button", Button).disabled = (
            self.session.last_completed_run is None
            or self.training.active
            or self.session.configuration_dirty
        )

    def action_cancel(self) -> None:
        if self.session.search_state in (JobState.QUEUED, JobState.RUNNING):
            self.action_cancel_search()
            return
        if not self.training.request_cancel():
            return
        self._cancel.set()
        self._set_job_state(JobState.CANCELLING, "CANCELLING")
        self._emit("training", "job_state", message="cancellation requested")

    @work(thread=True, exclusive=True)
    def _train_worker(self) -> None:
        self.call_from_thread(self._show_running)

        def report(value: TrainingProgress) -> None:
            if value.epoch == 1 or value.epoch == value.epochs or value.epoch % 10 == 0:
                self.call_from_thread(self._show_progress, value)

        try:
            result = self.training.run(progress=report, cancel=self._cancel)
        except TrainingCancelled as error:
            self.call_from_thread(self._show_cancelled, str(error))
        except (ValueError, RuntimeError) as error:
            self.call_from_thread(self._show_failure, str(error))
        else:
            self.call_from_thread(self._show_result, result)

    def _show_running(self) -> None:
        if self.session.job_state is JobState.QUEUED:
            self._set_job_state(JobState.RUNNING, "TRAINING")
            self._emit("training", "job_state", message="training started")

    def _show_progress(self, value: TrainingProgress) -> None:
        self.training.record_progress(value.epoch, value.accuracy)
        self.query_one("#progress", ProgressBar).update(
            total=value.epochs,
            progress=value.epoch,
        )
        self.query_one("#accuracy-line", Static).update(
            f"epoch {value.epoch:,}/{value.epochs:,}  /  accuracy {value.accuracy:.0%}"
        )
        self.query_one("#accuracy-spark", Sparkline).data = self.session.accuracy_history
        self._emit(
            "training",
            "progress",
            message=f"epoch {value.epoch}/{value.epochs} accuracy={value.accuracy:.0%}",
        )

    def _show_failure(self, message: str) -> None:
        self.training.failed(
            message, current_request=self._current_request_or_none()
        )
        self._sync_training_export_control()
        self._set_job_state(JobState.FAILED, "FAILED")
        self._emit("training", "failure", "error", message=message)

    def _show_cancelled(self, message: str) -> None:
        self.training.cancelled(
            current_request=self._current_request_or_none()
        )
        self._sync_training_export_control()
        self._set_job_state(JobState.CANCELLED, "CANCELLED")
        self.query_one("#next-action", Static).update(
            "Adjust the configuration or press "
            f"{_binding_key('train')} to try again."
        )
        self._emit("training", "job_state", message=message)

    def _show_result(self, result: TrainingRun) -> None:
        self.training.complete(
            result, current_request=self._current_request_or_none()
        )
        elapsed = perf_counter() - self._run_started
        completion_label = (
            "SUCCEEDED / STALE CONFIGURATION"
            if self.session.configuration_dirty
            else f"SUCCEEDED / accuracy {result.accuracy:.0%} / {elapsed:.2f}s"
        )
        self._set_job_state(
            JobState.SUCCEEDED,
            completion_label,
        )
        self.query_one("#progress", ProgressBar).update(
            total=result.request.epochs,
            progress=result.request.epochs,
        )
        self.query_one("#next-action", Static).update(
            f"Press {_binding_key('show_clauses')} to inspect learned clauses or "
            f"{_binding_key('show_artifacts')} to export a portable model."
        )
        predictions = self.query_one("#predictions", DataTable)
        for row, target, prediction in zip(result.rows, result.targets, result.predictions):
            predictions.add_row(
                int(row[0]),
                int(row[1]),
                target,
                prediction,
                "MATCH" if target == prediction else "MISS",
            )
        clauses = self.query_one("#clauses", DataTable)
        literal_names = ("x0", "not x0", "x1", "not x1")
        boundary = result.snapshot.states_per_action
        for index, states in enumerate(result.snapshot.states):
            included = tuple(
                literal_names[literal]
                for literal, state in enumerate(states)
                if state > boundary
            )
            clauses.add_row(
                index,
                "positive" if index % 2 == 0 else "negative",
                len(included),
                ", ".join(included) or "(empty)",
                key=str(index),
            )
        self._show_clause_detail(0)
        self._sync_training_export_control()
        self._update_overview()
        if self.session.configuration_dirty:
            self.query_one("#metric-run", Static).update("RUN\nSTALE CONFIG")
            self.query_one("#next-action", Static).update(
                "The completed run belongs to the settings that started training. "
                f"Press {_binding_key('train')} to train the visible configuration."
            )
        self._emit(
            "training",
            "metric",
            message=f"completed accuracy={result.accuracy:.0%} elapsed={elapsed:.2f}s",
        )

    def _show_clause_detail(self, index: int) -> None:
        run = self.session.last_completed_run
        if run is None or index < 0 or index >= len(run.snapshot.states):
            return
        states = run.snapshot.states[index]
        names = ("x0", "not x0", "x1", "not x1")
        boundary = run.snapshot.states_per_action
        facts = "  ".join(
            f"{name}={state} ({'INCLUDE' if state > boundary else 'exclude'})"
            for name, state in zip(names, states)
        )
        self.query_one("#clause-detail", Static).update(
            f"CLAUSE {index} / {'POSITIVE' if index % 2 == 0 else 'NEGATIVE'} VOTE\n"
            f"Action boundary: state > {boundary}\n{facts}"
        )

    def action_export(self) -> None:
        self.action_show_artifacts()
        try:
            current_request = self._request_from_form()
            current_request.validate()
        except ValueError as error:
            self.training.synchronize_configuration(None)
            self.query_one("#artifact-status", Static).update(
                f"EXPORT BLOCKED / Invalid visible configuration: {error}"
            )
            return
        raw_path = self.query_one("#artifact-path", Input).value.strip()
        path = Path(raw_path) if raw_path else Path("")
        if not path.is_absolute():
            path = self.workspace / path
        request = ArtifactExportRequest(
            path=path,
            name=self.query_one("#artifact-name", Input).value,
            author=self.query_one("#artifact-author", Input).value,
            description=self.query_one("#artifact-description", Input).value,
        )
        try:
            summary = self.training.export(current_request, request)
        except FileExistsError:
            message = "REFUSED / That file already exists. Choose a new path; no data was replaced."
            self.query_one("#artifact-status", Static).update(message)
            self._emit("artifact", "failure", "error", message=message)
            return
        except (OSError, ValueError, RuntimeError) as error:
            message = str(error)
            self.query_one("#artifact-status", Static).update(f"EXPORT FAILED / {message}")
            self._emit("artifact", "failure", "error", message=message)
            return
        self.query_one("#artifact-status", Static).update("EXPORTED / CONFORMANCE VERIFIED")
        self.query_one("#artifact-open-path", Input).value = str(summary.path)
        self.query_one("#artifact-details", Static).update(
            f"EXPORTED ARTIFACT READY TO LOAD\n{summary.path}\n\n"
            f"Artifact ID\n{summary.artifact_id}\n\n"
            f"Press {_binding_key('load_artifact')} to open its schema-driven "
            "record form."
        )
        self._update_overview()
        self._emit(
            "artifact",
            "artifact",
            message=f"exported {summary.path} id={summary.artifact_id}",
        )

    def _set_job_state(self, state: JobState, label: str) -> None:
        self.session.job_state = state
        self.query_one("#job", Static).update(label)
        active = state in (JobState.QUEUED, JobState.RUNNING, JobState.CANCELLING)
        self.query_one("#train-button", Button).disabled = active
        self.query_one("#cancel-button", Button).disabled = not active
        current = self.query_one("#workspace", ContentSwitcher).current
        destination = current.removeprefix("view-") if current else "overview"
        display_state = (
            self.session.search_state.value if destination == "search" else state.value
        )
        self.query_one("#workspace-bar", Static).update(
            f"{destination.upper()}  /  {self.session.name.upper()}  /  "
            f"{display_state.upper()}"
        )

    def _set_search_state(self, state: JobState, label: str) -> None:
        self.session.search_state = state
        self.query_one("#search-status", Static).update(label)
        active = state in (JobState.QUEUED, JobState.RUNNING, JobState.CANCELLING)
        self.query_one("#search-button", Button).disabled = active
        self.query_one("#search-cancel-button", Button).disabled = not active
        current = self.query_one("#workspace", ContentSwitcher).current
        if current == "view-search":
            self.query_one("#workspace-bar", Static).update(
                f"SEARCH  /  BOUNDED-PROLOG  /  {state.value.upper()}"
            )

    def _update_overview(self) -> None:
        run = self.session.last_completed_run
        artifact = self.session.artifact
        if run is not None:
            self.query_one("#metric-run", Static).update("RUN\nSUCCEEDED")
            correct = sum(a == b for a, b in zip(run.predictions, run.targets))
            self.query_one("#metric-accuracy", Static).update(
                f"ACCURACY\n{run.accuracy:.0%} ({correct}/4)"
            )
            self.query_one("#metric-clauses", Static).update(
                f"CLAUSES\n{run.request.number_of_clauses} learned"
            )
            self.query_one("#overview-next", Static).update(
                "NEXT  Inspect clauses with 3, or freeze this exact run with 4."
            )
        if artifact is not None:
            self.query_one("#metric-artifact", Static).update(
                f"ARTIFACT\n{artifact.byte_count:,} bytes"
            )

    def _emit(self, source: str, kind: str, level: str = "info", **payload: object) -> None:
        event = self.telemetry.emit(source, kind, level, **payload)
        self.session.events.append(event)
        if len(self.session.events) > 300:
            del self.session.events[:-300]
        self.query_one("#events", Log).write_line(event.display_line)
