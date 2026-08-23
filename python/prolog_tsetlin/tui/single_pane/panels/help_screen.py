from __future__ import annotations
from textual.screen import ModalScreen
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Button
from textual.app import ComposeResult
from textual.binding import Binding

class SinglePaneHelpScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "dismiss", "Close"), Binding("question_mark", "dismiss", "Close", show=False)]
    def compose(self) -> ComposeResult:
        from ....help_topics import TUI_SEMANTIC_BINDINGS

        shared_commands = "\n".join(
            f'{binding.display_key:<6}{binding.description}'
            for binding in TUI_SEMANTIC_BINDINGS
        )
        navigation = (
            '1 System  2 Dashboard  3 Clauses  4 TA States\n'
            '5 Literals  6 Graphs  7 Artifacts\n'
            'v Timeline  s Search  c Config  p Predictions\n'
            'Ctrl+L Events  d Detail'
        )
        title = 'PTM WORKBENCH / SINGLE-PANE HELP'
        with Vertical(id='help-dialog'):
            yield Static(title + '  ? to close  Esc to close', id='help-title')
            with Horizontal(id='help-columns'):
                with VerticalScroll(classes='help-col'):
                    yield Static('COMMANDS', classes='card_title')
                    yield Static(
                        navigation
                        + '\n\n'
                        + shared_commands
                        + '\n/     Filter clauses\nk     Hide selected clause locally\nEnter Inspect selected clause',
                        classes='card_label',
                    )
                with VerticalScroll(classes='help-col'):
                    yield Static('DATAPOINTS', classes='card_title')
                    yield Static(
                        'TA state  1..2*states-per-action; include when state exceeds the action boundary\n'
                        'Empty clause  no included literals\n'
                        'Active literals  states above the action boundary\n'
                        'FIRE  clause support count/fraction across completed-run rows\n'
                        'ΣVOTE  sum of +1 even-clause or -1 odd-clause firing contributions\n'
                        'A/O  activations aligned/opposed to the target class\n'
                        'C/E  activations on correct/incorrect model predictions\n'
                        'Unique  rows fired by no other same-polarity clause\n'
                        'Peer L/B  same-polarity literal/behavior Jaccard similarity\n'
                        'Near boundary  TA states in the configured nearest-state window\n'
                        'Saturated  TA states at 1 or 2*states-per-action\n'
                        'Accuracy  measured XOR training accuracy, displayed as percent\n'
                        'Diagnostic sample  immutable evaluated model snapshot at a bounded epoch cadence\n'
                        'Timeline selection  projects one completed-run sample read-only; export remains the final snapshot\n'
                        'TA state differs  fraction whose sampled integer state differs from the prior sample\n'
                        'Action flip  fraction crossing the TA include/exclude boundary since the prior sample\n'
                        'Clause firing differs  fraction of clause/example outcomes differing from the prior sample\n'
                        'Histogram  20 bins across the configured TA-state range\n'
                        'Clause age  n/a; select a clause-detail example to bind TA input truth',
                        classes='card_label',
                    )
            yield Button('Close [Esc]', id='help-close', variant='primary')
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'help-close':
            self.dismiss()
    def action_dismiss(self) -> None:
        self.dismiss()
