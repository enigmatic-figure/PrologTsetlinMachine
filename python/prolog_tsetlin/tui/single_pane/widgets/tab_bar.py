from __future__ import annotations

from textual.widgets import Static, Button
from textual.containers import Horizontal
from textual.app import ComposeResult
from textual.message import Message

class TabChanged(Message):
    def __init__(self, tab_id: str) -> None:
        super().__init__()
        self.tab_id = tab_id

class TabBar(Horizontal):
    TABS = [
        ('1', 'System'),
        ('2', 'Dashboard'),
        ('3', 'Clauses'),
        ('4', 'TA States'),
        ('5', 'Literals'),
        ('6', 'Graphs'),
        ('7', 'Artifacts'),
    ]
    def __init__(self, active: str = '2', **kwargs) -> None:
        super().__init__(**kwargs)
        self.active = active

    def compose(self) -> ComposeResult:
        for key, label in self.TABS:
            yield Button(f'{key} {label}', id=f'tab-{key}', classes='tab active' if key==self.active else 'tab')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith('tab-'):
            key = event.button.id.split('-')[1]
            self.active = key
            for k,_ in self.TABS:
                self.query_one(f'#tab-{k}', Button).set_class(k==key, 'active')
            self.post_message(TabChanged(key))
