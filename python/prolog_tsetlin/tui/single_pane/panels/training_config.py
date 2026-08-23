from __future__ import annotations
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Input, Button
from textual.app import ComposeResult
from ....services.training import TrainingRequest

class TrainingConfigPanel(Vertical):
    """Dense single-pane training config: 6 inputs in a row, inline validation, stale detection via SessionState."""

    def compose(self) -> ComposeResult:
        yield Static("TRAINING CONFIG  clauses/states/spec/T/epochs/seed  Enter to apply, t to train", classes="card_title")
        with Horizontal(id="config-row"):
            yield Input(placeholder="Clauses", id="cfg-clauses", value="20", type="integer")
            yield Input(placeholder="States", id="cfg-states", value="50", type="integer")
            yield Input(placeholder="s", id="cfg-spec", value="3.0", type="number")
            yield Input(placeholder="T", id="cfg-thr", value="10", type="integer")
            yield Input(placeholder="Epochs", id="cfg-epochs", value="80", type="integer")
            yield Input(placeholder="Seed", id="cfg-seed", value="7", type="integer")
            yield Button("Train [t]", id="cfg-train", variant="success")
            yield Button("Cancel [x]", id="cfg-cancel", variant="error")
        yield Static("", id="cfg-status", classes="card_label")

    def get_request(self) -> TrainingRequest:
        def _int(id_, label):
            raw = self.query_one(id_, Input).value.strip()
            try:
                return int(raw)
            except ValueError:
                raise ValueError(f"{label} must be integer")
        def _float(id_, label):
            raw = self.query_one(id_, Input).value.strip()
            try:
                return float(raw)
            except ValueError:
                raise ValueError(f"{label} must be number")
        return TrainingRequest(
            number_of_clauses=_int("#cfg-clauses", "Clauses"),
            states_per_action=_int("#cfg-states", "States"),
            specificity=_float("#cfg-spec", "s"),
            threshold=_int("#cfg-thr", "T"),
            epochs=_int("#cfg-epochs", "Epochs"),
            seed=_int("#cfg-seed", "Seed"),
        )

    def set_status(self, msg: str, is_error: bool = False) -> None:
        status = self.query_one("#cfg-status", Static)
        status.update(msg)
        status.styles.color = "#ff5a6a" if is_error else "#6b7a99"

    def set_from_request(self, req: TrainingRequest) -> None:
        self.query_one("#cfg-clauses", Input).value = str(req.number_of_clauses)
        self.query_one("#cfg-states", Input).value = str(req.states_per_action)
        self.query_one("#cfg-spec", Input).value = str(req.specificity)
        self.query_one("#cfg-thr", Input).value = str(req.threshold)
        self.query_one("#cfg-epochs", Input).value = str(req.epochs)
        self.query_one("#cfg-seed", Input).value = str(req.seed)
