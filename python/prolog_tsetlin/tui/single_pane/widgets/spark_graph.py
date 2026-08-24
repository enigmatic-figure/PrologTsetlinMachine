from __future__ import annotations

from textual.widgets import Sparkline, Static
from textual.containers import Vertical
from textual.app import ComposeResult

class SparkGraph(Vertical):
    def __init__(self, title: str, color: str = '#5af2ff', **kwargs) -> None:
        super().__init__(**kwargs)
        self.title = title
        self.color = color
        self._data: list[float] = [0.0]

    def compose(self) -> ComposeResult:
        safe = self.title.lower().replace(' ', '-')
        yield Static(self.title, classes='card_title')
        yield Sparkline(self._data, id=f'spark-{safe}')

    def update_data(self, data: list[float]) -> None:
        self._data = data or [0.0]
        self.query_one(Sparkline).data = self._data
