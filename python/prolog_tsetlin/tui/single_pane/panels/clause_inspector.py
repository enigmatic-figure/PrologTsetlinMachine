from __future__ import annotations
import re
from textual.widgets import DataTable, Static
from textual.containers import Vertical
from textual.app import ComposeResult

class ClauseInspectorPanel(Vertical):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rows: list[dict] = []
        self._hidden_ids: set[str] = set()
        self._filter = ''
        self._sort_key = 'support'
        self._sort_rev = True
        self._last_update = 0.0

    def compose(self) -> ComposeResult:
        yield Static(
            'CLAUSE BEHAVIOR  FIRE support  ΣVOTE signed contribution  '
            'A/O target alignment  C/E model outcome',
            id='clause-title',
            classes='card_title',
        )
        yield DataTable(id='clause-table', zebra_stripes=True, cursor_type='row')

    def on_mount(self) -> None:
        table = self.query_one('#clause-table', DataTable)
        for key, label in (
            ('id', 'CLAUSE ID'),
            ('polarity', 'POL'),
            ('support', 'FIRE'),
            ('vote', 'ΣVOTE'),
            ('alignment', 'A/O'),
            ('outcome', 'C/E'),
            ('lits', 'ACTIVE LITS'),
            ('avg', 'STATE AVG'),
            ('near', 'NEAR BND'),
            ('similarity', 'SAME-POL PEER L/B'),
        ):
            table.add_column(label, key=key)
        table.cursor_type = 'row'
        table.zebra_stripes = True

    def set_rows(
        self,
        rows: list[dict],
        *,
        provenance: str = 'CURRENT',
        immediate: bool = False,
    ) -> None:
        import time
        now = time.monotonic()
        self._rows = rows
        self.query_one('#clause-title', Static).update(
            f'[{provenance}] CLAUSE BEHAVIOR  FIRE support  '
            'ΣVOTE signed contribution  A/O target alignment  '
            'C/E model outcome'
        )
        if immediate:
            self._last_update = now
            self._render_filtered()
            return
        # coalesce to 200ms: if within window, schedule deferred render
        if now - self._last_update < 0.2:
            # schedule a single deferred render if not already scheduled
            if not getattr(self, '_pending', False):
                self._pending = True
                self.set_timer(0.2, self._flush_pending)
            return
        self._last_update = now
        self._render_filtered()

    def _flush_pending(self) -> None:
        self._pending = False
        import time
        self._last_update = time.monotonic()
        self._render_filtered()

    def set_filter(self, pattern: str) -> None:
        self._filter = pattern.strip()
        self._render_filtered()

    def _render_filtered(self) -> None:
        table = self.query_one('#clause-table', DataTable)
        table.clear()
        rows = [
            row
            for row in self._rows
            if str(row.get('id', '')) not in self._hidden_ids
        ]
        if self._filter:
            try:
                regex = re.compile(self._filter, re.I)
                rows = [
                    row
                    for row in rows
                    if regex.search(str(row.get('id', '')))
                    or regex.search(row.get('polarity', ''))
                    or regex.search(str(row.get('avg_state', '')))
                    or regex.search(str(row.get('support', '')))
                ]
            except re.error:
                pattern = self._filter.lower()
                rows = [
                    row
                    for row in rows
                    if pattern in str(row.get('id', '')).lower()
                    or pattern in row.get('polarity', '').lower()
                ]
        sort_keys = {
            'id': 'id',
            'polarity': 'polarity',
            'support': 'support_rate',
            'vote': 'vote_sum',
            'alignment': 'aligned',
            'outcome': 'correct_support',
            'lits': 'lits',
            'avg': 'avg_state',
            'near': 'near_boundary',
            'similarity': 'activation_similarity',
        }
        if self._sort_key in sort_keys:
            sort_key = sort_keys[self._sort_key]

            def value_for(row):
                value = row.get(sort_key)
                comparable = (
                    value
                    if isinstance(value, (int, float))
                    else value.lower()
                    if isinstance(value, str)
                    else str(value)
                )
                return value is not None, comparable

            rows = sorted(rows, key=value_for, reverse=self._sort_rev)
        for row in rows[:20]:
            literal_peer = row.get('literal_peer')
            activation_peer = row.get('activation_peer')
            literal_similarity = row.get('literal_similarity')
            activation_similarity = row.get('activation_similarity')
            literal_text = (
                'n/a'
                if literal_peer is None or literal_similarity is None
                else f'L{literal_peer}:{literal_similarity:.0%}'
            )
            activation_text = (
                'n/a'
                if activation_peer is None or activation_similarity is None
                else f'B{activation_peer}:{activation_similarity:.0%}'
            )
            vote_sum = int(row.get('vote_sum', 0))
            table.add_row(
                str(row.get('id', '')).rjust(3, '0'),
                row.get('polarity', ''),
                f"{row.get('support', 0)}/{row.get('sample_count', 0)} "
                f"{row.get('support_rate', 0.0):.0%}",
                f'{vote_sum:+d}',
                f"{row.get('aligned', 0)}/{row.get('opposed', 0)}",
                f"{row.get('correct_support', 0)}/{row.get('error_support', 0)}",
                str(row.get('lits', '')),
                f"{row.get('avg_state', 0):.1f}",
                f"{row.get('near_boundary', 0.0):.0%}",
                f'{literal_text} {activation_text}',
            )

    def mark_hidden(self) -> str | None:
        table = self.query_one('#clause-table', DataTable)
        if table.row_count == 0 or table.cursor_row is None:
            return None
        row = table.get_row_at(table.cursor_row)
        clause_id = str(row[0]).strip().lstrip('0') or '0'
        self._hidden_ids.add(clause_id)
        self._render_filtered()
        return clause_id
