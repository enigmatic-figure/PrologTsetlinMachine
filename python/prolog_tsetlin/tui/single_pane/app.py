from __future__ import annotations
import json
from pathlib import Path
from time import monotonic
from threading import Event

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.css.query import NoMatches
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    Footer,
    Input,
    Label,
    Static,
    TextArea,
)
from textual import work
from textual.binding import Binding

from .panels.dashboard import DashboardPanel
from .panels.training_graphs import TrainingGraphsPanel
from .panels.clause_inspector import ClauseInspectorPanel
from .panels.ta_histogram import TAHistogramPanel
from .panels.literal_view import LiteralViewPanel
from .panels.predictions import PredictionsPanel
from .panels.events import EventsPanel
from .panels.clause_detail import ClauseDetailPanel
from .panels.search_panel import SearchPanel
from .panels.training_config import TrainingConfigPanel
from .panels.system_info import SystemInfoPanel
from .panels.artifact_panel import ArtifactPanel
from .panels.temporal_inspector import (
    TemporalInspectorPanel,
    TemporalSampleSelected,
)
from .panels.help_screen import SinglePaneHelpScreen
from .widgets.tab_bar import TabBar, TabChanged
from .modules import clause_health, clause_rows, ta_histogram

from ...services.artifacts import ArtifactExportRequest
from ...services.diagnostics import (
    RunDiagnostics,
    SampledTrainingDiagnostics,
    analyze_training_run,
)
from ...services.training import (
    TrainingCancelled,
    TrainingDiagnosticSample,
    TrainingDiagnosticSampling,
    TrainingRequest,
)
from ...services.search import SearchKind, demo_search_document
from ...services.environment import inspect_environment
from ..models import SessionState, JobState
from ..controllers import (
    ArtifactSessionController,
    SearchSessionController,
    SessionContractError,
    TrainingInspection,
    TrainingSessionController,
)
from ...services.telemetry import TelemetrySession
from ...help_topics import TUI_SEMANTIC_BINDINGS

class SinglePaneApp(App[None]):
    TITLE = 'PTM Workbench single-pane mode'
    CSS_PATH = 'ptm.tcss'
    DIAGNOSTIC_SAMPLE_BUDGET = 25
    # PTM commands are shared; numeric navigation belongs to this shell.
    BINDINGS = [
        Binding(binding.key, binding.action, binding.label, show=binding.show)
        for binding in TUI_SEMANTIC_BINDINGS
    ] + [
        Binding('1', 'tab_1', 'System', show=False),
        Binding('2', 'tab_2', 'Dashboard', show=False),
        Binding('3', 'tab_3', 'Clauses', show=False),
        Binding('4', 'tab_4', 'TA States', show=False),
        Binding('5', 'tab_5', 'Literals', show=False),
        Binding('6', 'tab_6', 'Graphs', show=False),
        Binding('7', 'tab_7', 'Artifacts', show=False),
        Binding('c', 'show_config', 'Config', show=False),
        Binding('p', 'show_predictions', 'Predictions', show=False),
        Binding('ctrl+l', 'show_events', 'Events', show=False),
        Binding('d', 'show_detail', 'Detail', show=False),
        Binding('v', 'show_timeline', 'Timeline', show=False),
        Binding('s', 'show_search', 'Search', show=False),
        Binding('slash', 'filter', 'Filter'),
        Binding('k', 'prune', 'Mark hidden'),
        Binding('enter', 'inspect', 'Inspect', show=False),
    ]

    def __init__(self, workspace: Path | None = None, demo: str = 'xor') -> None:
        super().__init__()
        self.workspace = (workspace or Path.cwd()).resolve()
        self.demo = demo
        # One PTM session model, multiple presentation shells
        self.session = SessionState()
        self.telemetry = TelemetrySession()
        self.training = TrainingSessionController(self.session)
        self.artifacts = ArtifactSessionController(self.session)
        self.search = SearchSessionController(self.session)
        self._cancel = Event()
        self._run_generation = 0
        self._current_run_id: str | None = None
        self._start = monotonic()
        self._tab = '2'
        self._filter = ''
        self._snapshot = None
        self._diagnostics: RunDiagnostics | None = None
        self._snapshot_footer_details: str | None = None
        self._search_cancel = Event()
        self._search_generation = 0
        self._inspection_generation = 0

    def compose(self) -> ComposeResult:
        yield Static('PTM  prolog-tsetlin-machine  single-pane mode    TRAINING    epoch --/--', id='top-bar')
        yield DashboardPanel(id='top-cards')
        yield TabBar(id='tab-bar', active=self._tab)
        yield TrainingGraphsPanel(id='graphs')
        with ContentSwitcher(initial='view-split', id='bottom-switcher'):
            with Horizontal(id='view-split'):
                yield ClauseInspectorPanel(id='clause-inspector')
                yield TAHistogramPanel(id='ta-histogram')
            with Horizontal(id='view-clauses'):
                yield ClauseInspectorPanel(id='clause-inspector-full')
            with Horizontal(id='view-ta'):
                yield TAHistogramPanel(id='ta-histogram-full')
            with Horizontal(id='view-literals'):
                yield LiteralViewPanel(id='literal-view')
            with Vertical(id='view-system'):
                yield SystemInfoPanel(id='system-panel')
            with Vertical(id='view-artifacts'):
                yield ArtifactPanel(id='artifact-panel')
            with Horizontal(id='view-graphs'):
                yield TrainingGraphsPanel(id='graphs-full')
            with Vertical(id='view-predictions'):
                yield PredictionsPanel(id='predictions-panel')
            with Vertical(id='view-events'):
                yield EventsPanel(id='events-panel')
            with VerticalScroll(id='view-detail'):
                yield ClauseDetailPanel(id='clause-detail')
            with Vertical(id='view-config'):
                yield TrainingConfigPanel(id='config-panel')
            with Vertical(id='view-search'):
                yield SearchPanel(id='search-panel')
            with Vertical(id='view-timeline'):
                yield TemporalInspectorPanel(id='temporal-panel')
        yield Input(placeholder='Filter clauses (regex)  Enter to apply  Esc to clear', id='filter-input', classes='hidden')
        yield Static('CLAUSELIST 0/0  AVG STATE --  UTIL --  THR --', id='footer-bar')
        yield Static('1:System 2:Dashboard 3:Clauses 4:TA 5:Literals 6:Graphs 7:Artifacts v:Timeline s:Search c:Config p:Predictions Ctrl+L:Events d:Detail  t:train e:export /:filter k:hide ?:help q:quit', id='footer-sub')
        yield Footer()

    def on_mount(self) -> None:
        self.query_one('#filter-input', Input).can_focus = False
        self.set_focus(None)
        capabilities = {
            capability.component: capability
            for capability in inspect_environment(self.workspace)
        }
        from ..._version import __version__
        self.query_one(DashboardPanel).update_header('host-info', {
            'Version': f'PTM {__version__}',
            'Runtime': capabilities['Scalar oracle'].status.lower(),
            'Prolog': capabilities['GNU Prolog'].status.lower(),
            'Device': 'cpu',
        })
        self.session.configured_request = self.query_one(
            '#config-panel', TrainingConfigPanel
        ).get_request()
        self._apply_breakpoint(self.size.width, self.size.height)
        self._emit('session', 'session', message='single-pane ready; single PTM session model')
        self.set_interval(0.5, self._tick_uptime)

    def on_resize(self, event: Resize) -> None:
        self._apply_breakpoint(event.size.width, event.size.height)

    def on_unmount(self) -> None:
        self._cancel.set()
        self._search_cancel.set()

    def _apply_breakpoint(self, width: int, height: int) -> None:
        self.screen.set_class(width < 90 or height < 30, 'compact')

    def _emit(self, source: str, kind: str, level: str = 'info', **payload):
        event = self.telemetry.emit(source, kind, level, **payload)
        self.session.events.append(event)
        if self.query_one('#bottom-switcher', ContentSwitcher).current == 'view-events':
            self.query_one('#events-panel', EventsPanel).set_events(self.session.events)

    def _tick_uptime(self) -> None:
        elapsed = int(monotonic() - self._start)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        values = {'Uptime': f'{h:02d}:{m:02d}:{s:02d}'}
        try:
            import psutil
        except ImportError:
            pass
        else:
            values['CPU'] = f'{psutil.cpu_percent(interval=None):.0f}%'
            values['RAM'] = f'{psutil.virtual_memory().percent:.0f}%'
        try:
            self.query_one(DashboardPanel).update_header('system-util', values)
        except NoMatches:
            # A scheduled uptime tick may race normal screen teardown.
            return

    def action_train(self) -> None:
        if self.training.active:
            self.notify('training already running - press x to cancel')
            return
        try:
            request = self.query_one('#config-panel', TrainingConfigPanel).get_request()
            sampling = TrainingDiagnosticSampling.bounded(
                request.epochs,
                maximum_samples=self.DIAGNOSTIC_SAMPLE_BUDGET,
            )
            self.training.begin(request, diagnostic_sampling=sampling)
        except ValueError as error:
            self.training.synchronize_configuration(None)
            self._sync_training_export_control()
            self.query_one('#config-panel', TrainingConfigPanel).set_status(
                f'config invalid: {error}', is_error=True
            )
            self.notify(f'config invalid: {error}', severity='error')
            return
        self._inspection_generation += 1
        retained_inspection = self.training.current_inspection()
        if retained_inspection is not None:
            self._project_training_inspection(retained_inspection)
        else:
            self._refresh_timeline()
        self._sync_training_export_control()
        self._update_training_graphs()
        self.query_one(DashboardPanel).update_header(
            'statistics', {'TA Incl': 'waiting', 'TA Near': 'waiting'}
        )
        retained = self.session.last_completed_run
        if retained is None:
            status = 'training active — model panels have no completed snapshot'
        elif retained.request != request:
            status = 'training active — model panels show a stale last-completed snapshot'
        else:
            status = 'training active — model panels show the last-completed snapshot'
        self.query_one('#config-panel', TrainingConfigPanel).set_status(status)
        self._refresh_snapshot_provenance()
        self.query_one(DashboardPanel).update_header('training-config', {
            'Clauses': f'{request.number_of_clauses} {request.number_of_clauses//2}/{request.number_of_clauses//2}',
            'T': f'{request.threshold} thr',
            's': f'{request.specificity} spec',
            'Features': '4 lits',
        })
        # Telemetry + run_id generation
        run_id = self.telemetry.begin_run()
        self._run_generation += 1
        gen = self._run_generation
        self._current_run_id = run_id
        self._cancel = Event()
        self._emit(
            'training',
            'job_state',
            message=(
                f'training queued run_id={run_id} gen={gen}; '
                f'diagnostics every {sampling.every_epochs} epoch(s), '
                f'{len(sampling.selected_epochs(request.epochs))} samples'
            ),
        )
        self.query_one('#top-bar', Static).update(
            f'PTM  single-pane mode  QUEUED  epoch --/{request.epochs}  '
            f'{self._active_snapshot_tag()}'
        )
        self._start_training(request, self._cancel, run_id, gen)

    def action_cancel(self) -> None:
        if self.search.active:
            self.action_cancel_search()
            return
        if not self.training.request_cancel():
            return
        self._cancel.set()
        self._emit('training', 'job_state', message=f'cancellation requested run_id={self._current_run_id}')
        self.query_one('#top-bar', Static).update(
            f'PTM  single-pane mode  CANCELLING  {self._active_snapshot_tag()}'
        )

    def action_filter(self) -> None:
        inp = self.query_one('#filter-input', Input)
        inp.can_focus = True
        inp.remove_class('hidden')
        inp.focus()

    def action_prune(self) -> None:
        current = self.query_one('#bottom-switcher', ContentSwitcher).current
        selector = (
            '#clause-inspector-full'
            if current == 'view-clauses'
            else '#clause-inspector'
        )
        clause_id = self.query_one(selector, ClauseInspectorPanel).mark_hidden()
        if clause_id is not None:
            self.notify(
                f'marked clause {clause_id} as hidden '
                '(local filter only, no TM change)'
            )
            return
        self.notify('mark: select a clause row and press k to hide locally')

    def action_inspect(self) -> None:
        current = self.query_one('#bottom-switcher', ContentSwitcher).current
        if (
            current not in ('view-clauses', 'view-split')
            or self._snapshot is None
            or self._diagnostics is None
        ):
            self.notify('inspect: no snapshot yet, train first')
            return
        selector = (
            '#clause-inspector'
            if current == 'view-split'
            else '#clause-inspector-full'
        )
        table = self.query_one(selector, ClauseInspectorPanel).query_one(
            '#clause-table', DataTable
        )
        if table.row_count == 0 or table.cursor_row is None:
            return
        row = table.get_row_at(table.cursor_row)
        clause_id = int(str(row[0]).strip().lstrip('0') or '0')
        states = self._snapshot.states[clause_id]
        boundary = self._snapshot.states_per_action
        literals = [
            f'{"x" if len(states) == 4 else "f"}{index // 2} '
            f'lit{index} state={state} '
            f'{"include" if state > boundary else "exclude"}'
            for index, state in enumerate(states)
        ]
        self.notify(
            f'clause {clause_id}: '
            + ' | '.join(literals[:6])
            + (' ...' if len(literals) > 6 else '')
        )
        names = [
            f'x{index // 2}={"true" if index % 2 == 0 else "false"}'
            for index in range(len(states))
        ]
        self.query_one('#clause-detail', ClauseDetailPanel).show_clause(
            self._diagnostics.clause(clause_id),
            self._diagnostics,
            tuple(states),
            boundary,
            names,
            provenance=self._inspection_provenance(
                self.training.current_inspection()
            ),
        )
        self._set_tab('detail')

    def _inspection_provenance(
        self, inspection: TrainingInspection | None
    ) -> str:
        if inspection is None:
            return 'NO COMPLETED SNAPSHOT'
        total = inspection.run.request.epochs
        if inspection.historical:
            return f'HISTORICAL EPOCH {inspection.epoch}/{total} READ ONLY'
        return f'FINAL EPOCH {total}/{total} EXPORT SNAPSHOT'

    def _project_training_inspection(
        self, inspection: TrainingInspection
    ) -> None:
        """Project exactly one immutable completed-run snapshot everywhere."""

        snapshot = inspection.snapshot
        diagnostics = inspection.diagnostics
        provenance = self._inspection_provenance(inspection)
        rows = clause_rows(diagnostics)
        histogram = ta_histogram(diagnostics)
        health = clause_health(diagnostics)
        maximum_state = 2 * snapshot.states_per_action

        self._snapshot = snapshot
        self._diagnostics = diagnostics
        for inspector_id in ('clause-inspector', 'clause-inspector-full'):
            self.query_one(
                f'#{inspector_id}', ClauseInspectorPanel
            ).set_rows(rows, provenance=provenance, immediate=True)
        for histogram_id in ('ta-histogram', 'ta-histogram-full'):
            self.query_one(
                f'#{histogram_id}', TAHistogramPanel
            ).update_hist(
                histogram,
                states_per_action=snapshot.states_per_action,
                diagnostics=diagnostics.ta_population,
                provenance=provenance,
            )
        self.query_one('#literal-view', LiteralViewPanel).set_literals(
            snapshot, provenance=provenance
        )
        self.query_one('#predictions-panel', PredictionsPanel).set_predictions(
            inspection.rows,
            inspection.targets,
            inspection.predictions,
            provenance=provenance,
        )
        self.query_one('#clause-detail', ClauseDetailPanel).reset(
            provenance=provenance
        )
        self.query_one('#events-panel', EventsPanel).set_events(
            self.session.events
        )
        self.query_one(DashboardPanel).update_header(
            'clause-health',
            {
                'Avg TA': f'{health["avg_ta"]}/{maximum_state}',
                'Empty': f'{health["empty"]} {health["empty_pct"]}%',
                'Nonempty': (
                    f'{health["nonempty"]} {health["nonempty_pct"]}%'
                ),
                'Unique': str(health['unique']),
            },
        )
        self.query_one(DashboardPanel).update_header(
            'statistics',
            {
                'Acc': f'{inspection.accuracy * 100:.1f}%',
                'Epoch': f'{inspection.epoch}/{inspection.run.request.epochs}',
                'TA Incl': (
                    f'{diagnostics.ta_population.included_fraction:.1%}'
                ),
                'TA Near': (
                    f'{diagnostics.ta_population.near_boundary_fraction:.1%}'
                ),
            },
        )
        self._snapshot_footer_details = (
            f'CLAUSELIST {len(rows)}/{snapshot.number_of_clauses}  '
            f'AVG STATE {health["avg_ta"]}/{maximum_state}  '
            f'INCLUDE {diagnostics.ta_population.included_fraction:.1%}  '
            f'TEMPORAL {len(self.session.last_completed_diagnostics)} SAMPLES  '
            f'ACTION BOUNDARY > {snapshot.states_per_action}'
        )
        self._refresh_timeline()
        self._refresh_snapshot_provenance()

    def _refresh_timeline(self) -> None:
        panel = self.query_one('#temporal-panel', TemporalInspectorPanel)
        run = self.session.last_completed_run
        history = self.session.last_completed_diagnostics
        if run is None or not history:
            panel.clear_history()
            return
        panel.set_history(
            history,
            completed_run=run,
            generation=self._inspection_generation,
            selected_epoch=self.session.inspected_sample_epoch,
            training_active=self.training.active,
        )

    def on_temporal_sample_selected(
        self, event: TemporalSampleSelected
    ) -> None:
        if event.generation != self._inspection_generation:
            self._refresh_timeline()
            return
        if self.training.active:
            self.notify(
                'timeline is locked during training; model panels retain the '
                'last completed final snapshot'
            )
            self._refresh_timeline()
            return
        try:
            inspection = self.training.inspect_completed_epoch(
                event.epoch, expected_run=event.run
            )
        except SessionContractError as error:
            self.notify(str(error), severity='error')
            return
        self._project_training_inspection(inspection)
        total = inspection.run.request.epochs
        if inspection.historical:
            status = (
                f'INSPECTING HISTORICAL SAMPLE epoch '
                f'{inspection.epoch}/{total}  '
                f'acc {inspection.accuracy * 100:.1f}%  READ ONLY  '
                f'EXPORT FINAL {total}'
            )
        else:
            status = (
                f'COMPLETED FINAL SNAPSHOT epoch {total}/{total}  '
                f'acc {inspection.accuracy * 100:.1f}%'
            )
        self.query_one('#top-bar', Static).update(
            f'PTM  single-pane mode  {status}'
        )
        self._emit(
            'training',
            'inspection',
            message=(
                f'projected completed-run epoch {inspection.epoch}/{total}; '
                f'historical={inspection.historical}'
            ),
            epoch=inspection.epoch,
        )

    def action_help(self) -> None:
        self.push_screen(SinglePaneHelpScreen())

    def _refresh_configuration_state(self) -> None:
        try:
            panel = self.query_one('#config-panel', TrainingConfigPanel)
        except NoMatches:
            # Input.Changed messages can already be queued when the screen unmounts.
            return
        try:
            current = panel.get_request()
            current.validate()
        except ValueError as error:
            stale = self.training.synchronize_configuration(None)
            panel.set_status(f'config invalid: {error}', is_error=True)
        else:
            stale = self.training.synchronize_configuration(current)
            panel.set_status(
                'config changed — press t to retrain (stale)' if stale else '',
                is_error=False,
            )
        if self.session.last_completed_run is not None and not self.training.active:
            inspection = self.training.current_inspection()
            if inspection is not None and inspection.historical:
                self.query_one(DashboardPanel).update_header(
                    'statistics',
                    {
                        'Acc': f'HIST {inspection.accuracy * 100:.1f}%',
                        'Epoch': (
                            f'{inspection.epoch}/'
                            f'{inspection.run.request.epochs}'
                        ),
                    },
                )
            elif stale:
                self.query_one(DashboardPanel).update_header(
                    'statistics', {'Acc': 'STALE', 'Epoch': '--'}
                )
            else:
                run = self.session.last_completed_run
                self.query_one(DashboardPanel).update_header(
                    'statistics',
                    {
                        'Acc': f'{run.accuracy * 100:.1f}%',
                        'Epoch': f'{run.request.epochs}/{run.request.epochs}',
                    },
                )
        self._refresh_snapshot_provenance()
        self._sync_training_export_control()

    def _active_snapshot_tag(self) -> str:
        if self.session.last_completed_run is None:
            return 'MODEL PANELS EMPTY'
        return 'MODEL PANELS LAST COMPLETED'

    def _refresh_snapshot_provenance(self) -> None:
        run = self.session.last_completed_run
        active_request = self.session.active_request
        footer = self.query_one('#footer-bar', Static)
        footer.remove_class('historic')
        if run is None:
            if active_request is None:
                footer.update('MODEL PANELS EMPTY  /  NO COMPLETED RUN')
            else:
                footer.update(
                    'MODEL PANELS EMPTY  /  ACTIVE TRAINING HAS NO COMPLETED SNAPSHOT'
                )
            return

        details = self._snapshot_footer_details or (
            f'clauses {run.request.number_of_clauses}  '
            f'epochs {run.request.epochs}  seed {run.request.seed}'
        )
        inspection = self.training.current_inspection()
        if inspection is not None and inspection.historical:
            footer.add_class('historic')
            dirty = (
                '  /  STALE FOR VISIBLE CONFIGURATION'
                if self.session.configuration_dirty
                else ''
            )
            active = (
                '  /  ACTIVE TRAINING IS SEPARATE'
                if active_request is not None
                else ''
            )
            footer.update(
                f'HISTORICAL SAMPLE {inspection.epoch}/{run.request.epochs}  '
                f'/  READ ONLY  /  EXPORT REMAINS FINAL EPOCH '
                f'{run.request.epochs}{dirty}{active}  /  {details}'
            )
            return
        if active_request is not None:
            relation = (
                'STALE FOR ACTIVE REQUEST'
                if run.request != active_request
                else 'SAME REQUEST, RETRAINING'
            )
            visible = (
                ''
                if self.training.active_request_matches_configuration
                else '  /  VISIBLE CONFIG DIFFERS FROM ACTIVE REQUEST'
            )
            footer.update(
                f'LAST COMPLETED SNAPSHOT  /  {relation}{visible}  /  {details}'
            )
            return

        qualifier = (
            'LAST COMPLETED SNAPSHOT  /  STALE FOR VISIBLE CONFIGURATION'
            if self.session.configuration_dirty
            else 'CURRENT COMPLETED SNAPSHOT'
        )
        footer.update(f'{qualifier}  /  {details}')

    def _current_training_request(self) -> TrainingRequest | None:
        try:
            current_request = self.query_one(
                '#config-panel', TrainingConfigPanel
            ).get_request()
            current_request.validate()
        except ValueError:
            return None
        return current_request

    def _sync_training_export_control(self) -> None:
        self.query_one('#artifact-export', Button).disabled = (
            self.session.last_completed_run is None
            or self.training.active
            or self.session.configuration_dirty
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id and event.input.id.startswith('cfg-'):
            self._refresh_configuration_state()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Handle training config inputs dirty detection
        if event.input.id and event.input.id.startswith('cfg-'):
            self._refresh_configuration_state()
            return
        if event.input.id == 'filter-input':
            self._filter = event.value.strip()
            inp = self.query_one('#filter-input', Input)
            inp.add_class('hidden')
            inp.can_focus = False
            self.set_focus(None)
            for panel_id in ('clause-inspector', 'clause-inspector-full'):
                self.query_one(
                    f'#{panel_id}', ClauseInspectorPanel
                ).set_filter(self._filter)

    def on_key(self, event) -> None:
        if event.key == 'escape':
            field = self.query_one('#filter-input', Input)
            if 'hidden' not in field.classes:
                field.value = ''
                field.add_class('hidden')
                field.can_focus = False
                self.set_focus(None)
                self._filter = ''
                for panel_id in ('clause-inspector', 'clause-inspector-full'):
                    self.query_one(
                        f'#{panel_id}', ClauseInspectorPanel
                    ).set_filter('')
                event.stop()

    def on_button_pressed(self, event) -> None:
        actions = {
            'artifact-export': self._export_artifact,
            'artifact-verify': self._verify_artifact,
            'run-record': self._run_record,
            'cfg-train': self.action_train,
            'cfg-cancel': self.action_cancel,
            'search-run': self.action_search,
            'search-cancel': self.action_cancel_search,
            'search-export': self.action_export_search,
        }
        action = actions.get(event.button.id or '')
        if action is not None:
            action()

    def action_export(self) -> None:
        self._export_artifact()

    def action_load_artifact(self) -> None:
        self._verify_artifact()

    def action_run_record(self) -> None:
        self._run_record()

    def _export_artifact(self) -> None:
        panel = self.query_one('#artifact-panel', ArtifactPanel)
        try:
            current_request = self.query_one(
                '#config-panel', TrainingConfigPanel
            ).get_request()
            current_request.validate()
        except ValueError as error:
            self.training.synchronize_configuration(None)
            panel.set_status(
                f'EXPORT BLOCKED: invalid configuration: {error}'
            )
            return
        try:
            raw = panel.query_one('#artifact-path', Input).value.strip()
            path = Path(raw or 'out/single-pane-xor.ptm')
            if not path.is_absolute():
                path = self.workspace / path
            request = ArtifactExportRequest(
                path=path,
                name='single-pane-xor',
                description='Single-pane XOR export',
            )
            summary = self.training.export(current_request, request)
            panel.set_status(f'EXPORTED {summary.artifact_id[:8]}  {summary.byte_count} bytes  {summary.conformance_examples} cases  -> {path}')
            self.notify(f'exported {path}')
        except FileExistsError:
            panel.set_status('REFUSED - file exists, choose new path')
        except (OSError, ValueError, RuntimeError) as error:
            panel.set_status(f'EXPORT FAILED: {error}')
            self._emit('artifact', 'failure', level='error', message=str(error))

    def _verify_artifact(self) -> None:
        panel = self.query_one('#artifact-panel', ArtifactPanel)
        try:
            raw = panel.query_one('#artifact-path', Input).value.strip()
            path = Path(raw or 'out/single-pane-xor.ptm')
            if not path.is_absolute():
                path = self.workspace / path
            loaded = self.artifacts.load(path)
        except (OSError, ValueError, RuntimeError) as error:
            panel.set_status(f'VERIFY FAILED: {error}')
            panel.set_record_result('')
            self._emit('artifact', 'failure', level='error', message=str(error))
            return

        count = loaded.verification.get('conformance_case_count', '?')
        panel.set_status(f'LOADED + VERIFIED {count} cases  OK')
        table = panel.query_one('#artifact-table', DataTable)
        table.clear()
        details = loaded.inspection
        task = details.get('task')
        table.add_row('Artifact ID', str(details.get('artifact_id', '--')))
        table.add_row('Kind', str(details.get('artifact_kind', '--')))
        table.add_row('Size', f"{details.get('size_bytes', 0)} bytes")
        table.add_row(
            'Task',
            str(task.get('kind', '--') if isinstance(task, dict) else '--'),
        )
        container = panel.query_one('#record-fields', Vertical)

        async def populate_fields() -> None:
            await container.remove_children()
            if not loaded.fields:
                await container.mount(
                    Static('No raw-record schema (precomputed inputs only)')
                )
                return
            for index, field in enumerate(loaded.fields):
                requirement = 'required' if field.required else 'optional'
                await container.mount(
                    Label(
                        f'{field.name} ({field.kind.value}, {requirement})',
                        classes='card_label',
                    ),
                    Input(
                        placeholder='value',
                        id=f'record-{index}',
                        classes='card_label',
                    ),
                )

        self.call_later(populate_fields)
        self._emit(
            'artifact',
            'artifact',
            message=f"loaded {loaded.path} id={details.get('artifact_id')}",
        )

    def _run_record(self) -> None:
        panel = self.query_one('#artifact-panel', ArtifactPanel)
        try:
            raw_path = panel.query_one('#artifact-path', Input).value.strip()
            displayed_path = Path(raw_path or 'out/single-pane-xor.ptm')
            if not displayed_path.is_absolute():
                displayed_path = self.workspace / displayed_path
            values = {
                field.name: panel.query_one(f'#record-{index}', Input).value
                for index, field in enumerate(self.session.artifact_fields)
            }
            result = self.artifacts.run_record(
                values, displayed_path=displayed_path
            )
        except (OSError, ValueError, RuntimeError) as error:
            panel.set_record_result(f'INFERENCE FAILED: {error}')
            self._emit('artifact', 'failure', level='error', message=str(error))
            return

        trace = panel.query_one('#feature-trace', DataTable)
        trace.clear()
        for item in result.feature_trace:
            trace.add_row(
                str(item.get('field', '')),
                str(item.get('expression', '')),
                str(item.get('value', '')),
                str(item.get('literal_id', '')),
            )
        panel.set_record_result(
            f'PREDICTION {result.label} / class {result.prediction}  '
            f'features={list(result.features)}'
        )
        self._emit(
            'artifact',
            'prediction',
            message=(
                f'record prediction={result.prediction} '
                f'features={list(result.features)}'
            ),
        )

    def on_data_table_header_selected(self, event) -> None:
        if event.data_table.id != 'clause-table':
            return
        key = event.column_key.value
        if key not in {
            'id',
            'polarity',
            'support',
            'vote',
            'alignment',
            'outcome',
            'lits',
            'avg',
            'near',
            'similarity',
        }:
            return
        panel = event.data_table.parent
        if not isinstance(panel, ClauseInspectorPanel):
            return
        if panel._sort_key == key:
            panel._sort_rev = not panel._sort_rev
        else:
            panel._sort_key = key
            panel._sort_rev = True
        panel._render_filtered()

    def action_tab_1(self): self._set_tab('1')
    def action_tab_2(self): self._set_tab('2')
    def action_tab_3(self): self._set_tab('3')
    def action_tab_4(self): self._set_tab('4')
    def action_tab_5(self): self._set_tab('5')
    def action_tab_6(self): self._set_tab('6')
    def action_tab_7(self): self._set_tab('7')
    def action_show_config(self): self._set_tab('config')
    def action_show_predictions(self): self._set_tab('predictions')
    def action_show_events(self): self._set_tab('events')
    def action_show_detail(self): self._set_tab('detail')
    def action_show_timeline(self) -> None:
        self._set_tab('timeline')
        self.query_one('#temporal-table', DataTable).focus()
    def action_show_search(self): self._set_tab('search')

    def _set_tab(self, key: str):
        self._tab = key
        self.query_one(TabBar).active = key
        for tab_key in ('1', '2', '3', '4', '5', '6', '7'):
            self.query_one(f'#tab-{tab_key}', Button).set_class(
                tab_key == key, 'active'
            )
        destinations = {
            '1': 'view-system',
            '2': 'view-split',
            '3': 'view-clauses',
            '4': 'view-ta',
            '5': 'view-literals',
            '6': 'view-graphs',
            '7': 'view-artifacts',
            'config': 'view-config',
            'search': 'view-search',
            'predictions': 'view-predictions',
            'events': 'view-events',
            'detail': 'view-detail',
            'timeline': 'view-timeline',
        }
        self.query_one('#bottom-switcher', ContentSwitcher).current = (
            destinations.get(key, 'view-split')
        )

    def action_search(self) -> None:
        if self.search.active:
            self.notify('search already running')
            return
        panel = self.query_one('#search-panel', SearchPanel)
        try:
            kind = panel.selected_kind()
            prepared = self.search.prepare(
                panel.get_request_dict(),
                expected_kind=kind,
                timeout_seconds=panel.timeout_seconds(),
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            self.search.failed(str(error))
            panel.set_result(str(error), status='INVALID REQUEST')
            panel.set_controls(active=False, exportable=False)
            self._emit('search', 'failure', level='error', message=str(error))
            self.notify(f'search invalid: {error}', severity='error')
            return
        self._search_generation += 1
        generation = self._search_generation
        cancel = Event()
        self._search_cancel = cancel
        panel.set_result(
            json.dumps(
                {'status': 'queued', 'budget': dict(prepared.budget)},
                indent=2,
                sort_keys=True,
            ),
            status=f'QUEUED {kind.value}',
        )
        panel.set_counterexamples([])
        panel.set_controls(active=True, exportable=False)
        self._emit(
            'search',
            'job_state',
            message=(
                f"queued {kind.value} "
                f"candidates<={prepared.budget['candidate_upper_bound']}"
            ),
        )
        self._search_worker(prepared.request, cancel, generation)

    def action_cancel_search(self) -> None:
        if not self.search.request_cancel():
            return
        self._search_cancel.set()
        self._emit('search', 'job_state', message='cancel requested')
        panel = self.query_one('#search-panel', SearchPanel)
        panel.set_result('', status='CANCELLING')
        panel.set_controls(active=True, exportable=False)

    @work(thread=True, exclusive=True, group='search')
    def _search_worker(self, request, cancel: Event, generation: int) -> None:
        from ...prolog_bridge import PrologSearchCancelled, PrologBridgeError, NoThresholdSolution, NoFeatureTemplateSolution, NoTAClauseSolution, NoDecisionTreeSolution
        self.call_from_thread(self._show_search_running, generation)
        try:
            result = self.search.run(request, cancel=cancel.is_set)
        except PrologSearchCancelled as e:
            self.call_from_thread(self._show_search_cancelled, str(e), generation)
        except (NoThresholdSolution, NoFeatureTemplateSolution, NoTAClauseSolution, NoDecisionTreeSolution) as e:
            self.call_from_thread(self._show_search_no_solution, str(e), generation)
        except (KeyError, TypeError, ValueError, OSError, PrologBridgeError) as e:
            self.call_from_thread(self._show_search_failed, str(e), generation)
        else:
            self.call_from_thread(self._show_search_result, result, generation)

    def _search_callback_is_current(self, generation: int | None) -> bool:
        return generation is None or generation == self._search_generation

    def _show_search_running(self, generation: int | None = None) -> None:
        if not self._search_callback_is_current(generation):
            return
        if self.session.search_state is JobState.QUEUED:
            self.search.mark_running()
            self._emit('search', 'job_state', message='search running')
            panel = self.query_one('#search-panel', SearchPanel)
            panel.set_result('', status='RUNNING')
            panel.set_controls(active=True, exportable=False)

    def _show_search_cancelled(
        self, msg: str, generation: int | None = None
    ) -> None:
        if not self._search_callback_is_current(generation):
            return
        self.search.cancelled()
        self._emit('search', 'job_state', message='cancelled')
        panel = self.query_one('#search-panel', SearchPanel)
        panel.set_result(msg, status='CANCELLED')
        panel.set_controls(active=False, exportable=False)

    def _show_search_no_solution(
        self, msg: str, generation: int | None = None
    ) -> None:
        if not self._search_callback_is_current(generation):
            return
        self.search.no_solution()
        self._emit('search', 'search_result', message=f'no solution: {msg}')
        panel = self.query_one('#search-panel', SearchPanel)
        panel.set_result(
            json.dumps({'status': 'no_solution', 'message': msg}, indent=2),
            status='NO SOLUTION',
        )
        panel.set_controls(active=False, exportable=False)

    def _show_search_failed(
        self, msg: str, generation: int | None = None
    ) -> None:
        if not self._search_callback_is_current(generation):
            return
        self.search.failed(msg)
        self._emit('search', 'failure', level='error', message=msg)
        panel = self.query_one('#search-panel', SearchPanel)
        panel.set_result(msg, status='FAILED')
        panel.set_controls(active=False, exportable=False)

    def _show_search_result(self, result, generation: int | None = None) -> None:
        if not self._search_callback_is_current(generation):
            return
        self.search.complete(result)
        self._emit('search', 'search_result', message=f'solved {result.kind.value}')
        panel = self.query_one('#search-panel', SearchPanel)
        panel.set_result(
            json.dumps(result.to_dict(), indent=2, sort_keys=True),
            status=f'SOLVED {result.kind.value}',
        )
        counterexamples = result.report.get('counterexamples', [])
        panel.set_counterexamples(
            counterexamples if isinstance(counterexamples, list) else []
        )
        panel.set_controls(active=False, exportable=result.exportable)

    def action_export_search(self) -> None:
        panel = self.query_one('#search-panel', SearchPanel)
        raw = panel.query_one('#search-export-path', Input).value.strip()
        path = Path(raw or 'out/search.ptm')
        if not path.is_absolute():
            path = self.workspace / path
        result = self.session.search_result
        name = f'prolog-{result.kind.value}' if result is not None else 'prolog-search'
        try:
            report = self.search.export(path, name=name)
        except FileExistsError:
            panel.set_result(
                panel.query_one('#search-result', TextArea).text,
                status='EXPORT REFUSED — file exists',
            )
            return
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            panel.set_result(
                panel.query_one('#search-result', TextArea).text,
                status=f'EXPORT FAILED: {error}',
            )
            self._emit('search', 'failure', level='error', message=str(error))
            return
        panel.set_result(
            panel.query_one('#search-result', TextArea).text,
            status=f"EXPORTED {report['size_bytes']} bytes -> {report['output']}",
        )
        self._emit('search', 'artifact', message=f"exported {report['output']}")

    def on_select_changed(self, event) -> None:
        # Update demo JSON when search kind changes
        if event.select.id != 'search-kind':
            return
        if self.search.active:
            self._search_cancel.set()
            self._emit(
                'search',
                'job_state',
                message='active search invalidated by kind change',
            )
        self._search_generation += 1
        kind = SearchKind(str(event.value))
        document = demo_search_document(kind)
        self.query_one('#search-json', TextArea).text = json.dumps(
            document, indent=2
        )
        self.query_one('#search-timeout', Input).value = str(
            document.get('timeout_seconds', 30)
        )
        self.search.reset()
        panel = self.query_one('#search-panel', SearchPanel)
        panel.set_result('', status=f'READY {kind.value}')
        panel.set_counterexamples([])
        panel.set_controls(active=False, exportable=False)

    def on_training_progress(self, p, run_id: str | None = None, gen: int | None = None) -> None:
        # Discard stale messages that do not belong to current run
        if gen is not None and gen != self._run_generation:
            return
        if run_id is not None and run_id != self._current_run_id:
            return
        self.training.record_progress(p.epoch, p.accuracy)
        self._emit('training', 'progress', message=f'epoch {p.epoch}/{p.epochs} acc={p.accuracy:.0%}', epoch=p.epoch, run_id=run_id, gen=gen)
        self.query_one('#top-bar', Static).update(
            f'PTM  single-pane mode  TRAINING  epoch {p.epoch}/{p.epochs}  '
            f'acc {p.accuracy*100:.1f}%  {self._active_snapshot_tag()}'
        )
        self._update_training_graphs()
        self.query_one(DashboardPanel).update_header('statistics', {'Acc': f'{p.accuracy*100:.1f}%', 'Epoch': f'{p.epoch}/{p.epochs}'})
        self._refresh_snapshot_provenance()

    def on_training_diagnostic(
        self,
        sample: TrainingDiagnosticSample,
        run_id: str | None = None,
        gen: int | None = None,
    ) -> None:
        if gen is not None and gen != self._run_generation:
            return
        if run_id is not None and run_id != self._current_run_id:
            return
        sampled = self.training.record_diagnostic_sample(sample)
        population = sampled.diagnostics.ta_population
        delta = sampled.delta_from_previous
        movement = (
            'first sample'
            if delta is None
            else (
                f'window={delta.earlier_epoch}->{delta.later_epoch}, '
                f'TA state differs={delta.changed_automata_fraction:.1%}, '
                f'action flips={delta.action_flip_fraction:.1%}, '
                f'clause firing differs='
                f'{delta.clause_behavior_flip_fraction:.1%}'
            )
        )
        self._emit(
            'training',
            'diagnostic_sample',
            message=(
                f'epoch {sample.epoch}/{sample.request.epochs}; '
                f'include={population.included_fraction:.1%}, '
                f'near={population.near_boundary_fraction:.1%}; {movement}'
            ),
            epoch=sample.epoch,
            run_id=run_id,
            gen=gen,
        )
        self.query_one(DashboardPanel).update_header(
            'statistics',
            {
                'TA Incl': f'{population.included_fraction:.1%}',
                'TA Near': f'{population.near_boundary_fraction:.1%}',
            },
        )
        self._update_training_graphs()
        self._refresh_timeline()

    def _visible_temporal_diagnostics(
        self,
    ) -> tuple[SampledTrainingDiagnostics, ...]:
        if self.training.active:
            return tuple(self.session.active_diagnostics)
        return self.session.last_completed_diagnostics

    def _update_training_graphs(self) -> None:
        sampled = self._visible_temporal_diagnostics()
        accuracy_history = (
            tuple(self.session.accuracy_history)
            if self.training.active
            else self.session.last_completed_accuracy_history
        )
        sample_epochs = [item.sample.epoch for item in sampled]
        include_fractions = [
            item.diagnostics.ta_population.included_fraction for item in sampled
        ]
        deltas = [
            item.delta_from_previous
            for item in sampled
            if item.delta_from_previous is not None
        ]
        delta_epochs = [delta.later_epoch for delta in deltas]
        ta_changed = [delta.changed_automata_fraction for delta in deltas]
        clause_changed = [
            delta.clause_behavior_flip_fraction for delta in deltas
        ]
        for graph in self.query(TrainingGraphsPanel):
            graph.update_series(
                accuracy_history,
                sample_epochs=sample_epochs,
                ta_include_fractions=include_fractions,
                delta_epochs=delta_epochs,
                ta_changed_fractions=ta_changed,
                clause_changed_fractions=clause_changed,
            )

    def on_training_complete(self, result, run_id: str | None = None, gen: int | None = None) -> None:
        if gen is not None and gen != self._run_generation:
            return
        if run_id is not None and run_id != self._current_run_id:
            return
        try:
            analyze_training_run(result)
        except ValueError as error:
            self.on_training_failed(
                f'diagnostic validation failed: {error}', run_id, gen
            )
            return
        self.training.complete(
            result, current_request=self._current_training_request()
        )
        self._inspection_generation += 1
        inspection = self.training.inspect_completed_epoch()
        self._project_training_inspection(inspection)
        self.query_one('#top-bar', Static).update(
            f'PTM  single-pane mode  TRAINED  FINAL SNAPSHOT '
            f'epoch {inspection.epoch}/{result.request.epochs}'
        )
        self._emit('training', 'job_state', message=f'training succeeded run_id={run_id} acc={result.accuracy:.0%}')
        self._update_training_graphs()
        self._refresh_configuration_state()

    def on_training_cancelled(self, msg: str, run_id: str | None = None, gen: int | None = None) -> None:
        if gen is not None and gen != self._run_generation:
            return
        self.training.cancelled(
            current_request=self._current_training_request()
        )
        self._inspection_generation += 1
        self._update_training_graphs()
        inspection = self.training.current_inspection()
        if inspection is not None:
            self._project_training_inspection(inspection)
        else:
            self._restore_completed_diagnostic_summary()
            self._refresh_timeline()
        self.query_one('#top-bar', Static).update(
            'PTM  single-pane mode  CANCELLED  MODEL PANELS LAST COMPLETED'
            if self.session.last_completed_run is not None
            else 'PTM  single-pane mode  CANCELLED  MODEL PANELS EMPTY'
        )
        self._refresh_configuration_state()
        self._emit('training', 'job_state', message=f'cancelled run_id={run_id}: {msg}')
        self.notify(f'cancelled: {msg}')

    def on_training_failed(self, msg: str, run_id: str | None = None, gen: int | None = None) -> None:
        if gen is not None and gen != self._run_generation:
            return
        self.training.failed(
            msg, current_request=self._current_training_request()
        )
        self._inspection_generation += 1
        self._update_training_graphs()
        inspection = self.training.current_inspection()
        if inspection is not None:
            self._project_training_inspection(inspection)
        else:
            self._restore_completed_diagnostic_summary()
            self._refresh_timeline()
        self.query_one('#top-bar', Static).update(
            'PTM  single-pane mode  FAILED  MODEL PANELS LAST COMPLETED'
            if self.session.last_completed_run is not None
            else 'PTM  single-pane mode  FAILED  MODEL PANELS EMPTY'
        )
        self._refresh_configuration_state()
        self._emit('training', 'failure', level='error', message=msg, run_id=run_id, gen=gen)
        self.notify(f'failed: {msg}', severity='error')

    def on_tab_changed(self, event: TabChanged) -> None:
        self._set_tab(event.tab_id)

    def _restore_completed_diagnostic_summary(self) -> None:
        if self.session.last_completed_diagnostics:
            population = self.session.last_completed_diagnostics[
                -1
            ].diagnostics.ta_population
            values = {
                'TA Incl': f'{population.included_fraction:.1%}',
                'TA Near': f'{population.near_boundary_fraction:.1%}',
            }
        else:
            values = {'TA Incl': 'n/a', 'TA Near': 'n/a'}
        self.query_one(DashboardPanel).update_header('statistics', values)

    @work(exclusive=True, thread=True)
    def _start_training(self, request, cancel, run_id, gen) -> None:
        def report(p):
            self.call_from_thread(self.on_training_progress, p, run_id, gen)

        def report_diagnostic(sample):
            self.call_from_thread(
                self.on_training_diagnostic, sample, run_id, gen
            )

        try:
            result = self.training.run(
                request,
                progress=report,
                diagnostic=report_diagnostic,
                cancel=cancel,
            )
            self.call_from_thread(self.on_training_complete, result, run_id, gen)
        except TrainingCancelled as e:
            self.call_from_thread(self.on_training_cancelled, str(e), run_id, gen)
        except (ValueError, RuntimeError) as e:
            self.call_from_thread(self.on_training_failed, str(e), run_id, gen)
