from __future__ import annotations

from textual.widgets import Static, Sparkline
from textual.containers import Vertical
from textual.app import ComposeResult

from ....services.diagnostics import TAPopulationDiagnostics


class TAHistogramPanel(Vertical):
    def compose(self) -> ComposeResult:
        yield Static('TA STATE HISTOGRAM', id='ta-title', classes='card_title')
        yield Sparkline([0], id='ta-hist')
        yield Static('n/a — train a model to populate this histogram', id='ta-stats', classes='graph-legend')

    def update_hist(
        self,
        hist: list[int],
        *,
        states_per_action: int | None = None,
        diagnostics: TAPopulationDiagnostics | None = None,
        provenance: str = 'CURRENT',
    ) -> None:
        self.query_one('#ta-hist', Sparkline).data = hist or [0]
        if states_per_action is not None:
            self.query_one('#ta-title', Static).update(
                f'[{provenance}] TA STATE HISTOGRAM  '
                f'1-{2 * states_per_action}  '
                f'ACTION BOUNDARY > {states_per_action}'
            )
        if diagnostics is not None:
            self.query_one('#ta-stats', Static).update(
                f'INCLUDE {diagnostics.included_fraction:.1%}  '
                f'NEAR±{diagnostics.boundary_window} '
                f'{diagnostics.near_boundary_fraction:.1%}  '
                f'SATURATED {diagnostics.saturated_fraction:.1%}  '
                f'MEAN DIST {diagnostics.average_distance_to_boundary:.1f}  '
                f'{diagnostics.total_automata} TA'
            )
        elif hist and any(hist):
            self.query_one('#ta-stats', Static).update(
                f'HISTOGRAM  {len(hist)} bins  total {sum(hist)} automata'
            )
        else:
            self.query_one('#ta-stats', Static).update(
                'n/a — train a model to populate this histogram'
            )

    def set_unavailable(self, reason: str) -> None:
        self.query_one('#ta-hist', Sparkline).data = [0]
        self.query_one('#ta-title', Static).update('[UNAVAILABLE] TA STATE HISTOGRAM')
        self.query_one('#ta-stats', Static).update(f'n/a — {reason}')
