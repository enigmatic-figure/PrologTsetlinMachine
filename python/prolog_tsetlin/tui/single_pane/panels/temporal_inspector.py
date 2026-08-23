from __future__ import annotations

from collections.abc import Sequence

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import DataTable, Static

from ....services.diagnostics import SampledTrainingDiagnostics
from ....services.training import TrainingRun


class TemporalSampleSelected(Message):
    def __init__(
        self, epoch: int, run: TrainingRun, generation: int
    ) -> None:
        super().__init__()
        self.epoch = epoch
        self.run = run
        self.generation = generation


class TemporalInspectorPanel(Vertical):
    """Select an immutable diagnostic sample from the completed run."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._history: tuple[SampledTrainingDiagnostics, ...] = ()
        self._run: TrainingRun | None = None
        self._generation = 0
        self._training_active = False
        self._selected_epoch: int | None = None
        self._completed_epochs = 0

    def compose(self) -> ComposeResult:
        yield Static(
            'TEMPORAL INSPECTOR  completed-run samples only',
            id='temporal-title',
            classes='card_title',
        )
        yield Static(
            'n/a — train a model to retain sampled diagnostics',
            id='temporal-status',
            classes='graph-legend',
        )
        yield DataTable(
            id='temporal-table', zebra_stripes=True, cursor_type='row'
        )

    def on_mount(self) -> None:
        self.query_one('#temporal-table', DataTable).add_columns(
            'EPOCH',
            'ACC',
            'TA INCLUDE',
            'NEAR BND',
            'SATURATED',
            'TA STATE DIFF',
            'ACTION FLIP',
            'CLAUSE FIRING DIFF',
            'PRED DIFF',
        )

    def set_history(
        self,
        history: Sequence[SampledTrainingDiagnostics],
        *,
        completed_run: TrainingRun,
        generation: int,
        selected_epoch: int | None,
        training_active: bool,
    ) -> None:
        self._history = tuple(history)
        self._run = completed_run
        self._generation = generation
        self._training_active = training_active
        self._completed_epochs = completed_run.request.epochs
        self._selected_epoch = selected_epoch or self._completed_epochs
        table = self.query_one('#temporal-table', DataTable)
        table.clear()
        self.query_one('#temporal-title', Static).update(
            f'TEMPORAL INSPECTOR  completed run seed '
            f'{completed_run.request.seed}  '
            f'{len(self._history)} retained samples'
        )
        for item in self._history:
            sample = item.sample
            population = item.diagnostics.ta_population
            delta = item.delta_from_previous
            table.add_row(
                f'{sample.epoch}/{self._completed_epochs}',
                f'{sample.accuracy:.1%}',
                f'{population.included_fraction:.1%}',
                f'{population.near_boundary_fraction:.1%}',
                f'{population.saturated_fraction:.1%}',
                'n/a'
                if delta is None
                else f'{delta.changed_automata_fraction:.1%}',
                'n/a'
                if delta is None
                else f'{delta.action_flip_fraction:.1%}',
                'n/a'
                if delta is None
                else f'{delta.clause_behavior_flip_fraction:.1%}',
                'n/a'
                if delta is None
                else f'{delta.prediction_flip_fraction:.1%}',
                key=str(sample.epoch),
            )
        selected_index = next(
            (
                index
                for index, item in enumerate(self._history)
                if item.sample.epoch == self._selected_epoch
            ),
            None,
        )
        if selected_index is not None:
            table.move_cursor(row=selected_index, column=0, animate=False)
        self._update_status(training_active=training_active)

    def clear_history(self) -> None:
        self._history = ()
        self._run = None
        self._training_active = False
        self._selected_epoch = None
        self._completed_epochs = 0
        self.query_one('#temporal-table', DataTable).clear()
        self.query_one('#temporal-title', Static).update(
            'TEMPORAL INSPECTOR  completed-run samples only'
        )
        self.query_one('#temporal-status', Static).update(
            'n/a — train a model to retain sampled diagnostics'
        )

    def _update_status(self, *, training_active: bool) -> None:
        selected = self._selected_epoch
        if selected is None:
            return
        if selected == self._completed_epochs:
            status = (
                '↑/↓ choose, Enter project  /  '
                f'PROJECTING FINAL EPOCH {selected}/{self._completed_epochs}  '
                'this is the completed export snapshot'
            )
        else:
            status = (
                '↑/↓ choose, Enter project  /  '
                f'PROJECTING HISTORICAL EPOCH {selected}/'
                f'{self._completed_epochs}  READ ONLY  '
                f'export remains final epoch {self._completed_epochs}'
            )
        if training_active:
            status += '  /  active training is separate'
        self.query_one('#temporal-status', Static).update(status)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != 'temporal-table':
            return
        if self._training_active:
            return
        run = self._run
        if run is None:
            return
        row = event.data_table.get_row_at(event.cursor_row)
        epoch = int(str(row[0]).split('/', 1)[0])
        self.post_message(
            TemporalSampleSelected(epoch, run, self._generation)
        )
