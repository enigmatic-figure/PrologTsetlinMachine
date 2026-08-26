from __future__ import annotations
from textual.widgets import DataTable, Static
from textual.containers import Vertical
from textual.app import ComposeResult

class LiteralViewPanel(Vertical):
    def compose(self) -> ComposeResult:
        yield Static(
            'LITERAL SUMMARY  n/a — train a model',
            id='literal-title',
            classes='card_title',
        )
        yield DataTable(id='literal-table', zebra_stripes=True, cursor_type='row')

    def on_mount(self) -> None:
        self.query_one('#literal-table', DataTable).add_columns(
            'FEATURE', 'LITERAL', 'INCLUDE RATE', 'AVG STATE', 'RANK'
        )

    def set_literals(self, snapshot, *, provenance: str = 'CURRENT') -> None:
        table = self.query_one('#literal-table', DataTable)
        table.clear()
        states = snapshot.states
        boundary = snapshot.states_per_action
        self.query_one('#literal-title', Static).update(
            f'[{provenance}] LITERAL SUMMARY  action boundary > {boundary}  '
            f'state range 1-{2 * boundary}'
        )
        if not states:
            return
        literal_count = len(states[0])
        for literal in range(literal_count):
            values = [clause[literal] for clause in states]
            average = sum(values) / len(values) if values else 0
            included = sum(1 for value in values if value > boundary)
            rate = f'{included / len(values) * 100:.0f}%' if values else 'n/a'
            feature = f'x{literal // 2}' if literal_count == 4 else f'f{literal}'
            literal_name = f'{feature}={"true" if literal % 2 == 0 else "false"}'
            table.add_row(
                feature,
                literal_name,
                rate,
                f'{average:.1f}',
                str(literal),
            )

    def set_unavailable(self, reason: str) -> None:
        self.query_one('#literal-table', DataTable).clear()
        self.query_one('#literal-title', Static).update(
            f'LITERAL SUMMARY  n/a — {reason}'
        )
