from __future__ import annotations
from collections.abc import Sequence
from textual.containers import Vertical
from textual.widgets import Static, Sparkline
from textual.app import ComposeResult

try:
    from textual_plotext import PlotextPlot
    HAS_PLOTEXT = True
except ImportError:
    PlotextPlot = None
    HAS_PLOTEXT = False

class TrainingGraphsPanel(Vertical):
    def compose(self) -> ComposeResult:
        header = (
            'TRAINING + SAMPLED MODEL DIAGNOSTICS  percent'
            if HAS_PLOTEXT
            else 'TRAINING ACCURACY ONLY  percent  (Plotext unavailable)'
        )
        legend = (
            'x: epoch  accuracy: each epoch  TA include: sampled  '
            'TA state/clause firing diff: prior sample'
            if HAS_PLOTEXT
            else 'x: epoch  y: accuracy percent  sampled diagnostics not displayed'
        )
        yield Static(header, id='graphs-header')
        if HAS_PLOTEXT:
            yield PlotextPlot(id='braille-graph')
        else:
            yield Sparkline([0], id='braille-graph-fallback')
        yield Static(legend, classes='graph-legend')

    def on_mount(self) -> None:
        if not HAS_PLOTEXT:
            return
        plot = self.query_one('#braille-graph', PlotextPlot)
        plot.plt.title('')
        plot.plt.xlabel('epochs')
        plot.plt.ylabel('percent')
        plot.plt.grid(True)
        plot.plt.plotsize(115, 9)
        plot.plt.theme('clear')

    @staticmethod
    def accuracy_percentages(values: Sequence[float] | None) -> list[float]:
        return [value * 100.0 for value in values or ()]

    @staticmethod
    def _paired_series(
        epochs: Sequence[int] | None,
        values: Sequence[float] | None,
        *,
        label: str,
    ) -> tuple[list[int], list[float]]:
        epoch_values = list(epochs or ())
        percentages = TrainingGraphsPanel.accuracy_percentages(values)
        if len(epoch_values) != len(percentages):
            raise ValueError(f'{label} epochs and values must align')
        return epoch_values, percentages

    def update_series(
        self,
        accuracy_fractions: Sequence[float] | None = None,
        *,
        sample_epochs: Sequence[int] | None = None,
        ta_include_fractions: Sequence[float] | None = None,
        delta_epochs: Sequence[int] | None = None,
        ta_changed_fractions: Sequence[float] | None = None,
        clause_changed_fractions: Sequence[float] | None = None,
    ) -> None:
        accuracy = self.accuracy_percentages(accuracy_fractions)
        include_epochs, ta_include = self._paired_series(
            sample_epochs, ta_include_fractions, label='TA include'
        )
        ta_delta_epochs, ta_changed = self._paired_series(
            delta_epochs, ta_changed_fractions, label='TA change'
        )
        clause_delta_epochs, clause_changed = self._paired_series(
            delta_epochs, clause_changed_fractions, label='clause change'
        )
        if HAS_PLOTEXT:
            plot = self.query_one('#braille-graph', PlotextPlot)
            plot.plt.clear_data()
            if accuracy:
                plot.plt.plot(
                    list(range(1, len(accuracy) + 1)),
                    accuracy,
                    marker='braille',
                    color='cyan',
                    label='Accuracy %',
                )
            if ta_include:
                plot.plt.plot(
                    include_epochs,
                    ta_include,
                    marker='braille',
                    color='yellow',
                    label='TA include %',
                )
            if ta_changed:
                plot.plt.plot(
                    ta_delta_epochs,
                    ta_changed,
                    marker='braille',
                    color='green',
                    label='TA state differs %',
                )
            if clause_changed:
                plot.plt.plot(
                    clause_delta_epochs,
                    clause_changed,
                    marker='braille',
                    color='red',
                    label='Clause firing differs %',
                )
            if not any([accuracy, ta_include, ta_changed, clause_changed]):
                plot.plt.plot([0], marker='braille', color='cyan')
            plot.refresh()
            return
        self.query_one('#braille-graph-fallback', Sparkline).data = (
            accuracy or [0]
        )
