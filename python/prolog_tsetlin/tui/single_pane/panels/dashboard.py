from __future__ import annotations
from textual.containers import Horizontal, Vertical
from textual.widgets import Static
from textual.app import ComposeResult

class DashboardPanel(Horizontal):
    def compose(self) -> ComposeResult:
        cards = [
            ('host-info', 'HOST INFO', [('Version','PTM 0.1.0'),('Runtime','scalar oracle'),('Prolog','detecting'),('Device','cpu:x86_64')]),
            ('system-util', 'SYSTEM UTIL', [('Uptime','00:00:00'),('CPU','n/a'),('RAM','n/a'),('Load','n/a')]),
            ('training-config', 'TRAINING CONFIG', [('Clauses','20 10/10'),('T','10 thr'),('s','3.0 spec'),('Features','4 lits')]),
            ('clause-health', 'CLAUSE HEALTH', [('Avg TA','n/a'),('Empty','n/a'),('Nonempty','n/a'),('Unique','n/a')]),
            ('data-ingest', 'DATA INGEST', [('Throughput','n/a'),('Batches','n/a'),('Lag','n/a'),('Cache','n/a')]),
            ('statistics', 'STATISTICS', [('Acc','n/a'),('TA Incl','n/a'),('TA Near','n/a'),('Epoch','n/a')]),
        ]
        for key, title, lines in cards:
            with Vertical(classes='card', id=f'card-{key}'):
                yield Static(title, classes='card_title')
                for k,v in lines:
                    safe = k.lower().replace(' ', '-')
                    yield Static(f'{k} {v}', classes='card_label', id=f'card-{key}-{safe}')

    def update_header(self, key: str, values: dict[str,str]) -> None:
        for k,v in values.items():
            safe = k.lower().replace(' ', '-')
            widget = self.query_one(f'#card-{key}-{safe}', Static)
            widget.update(f'{k} {v}')
