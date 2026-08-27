from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Select, Static

from ....services.training import TrainingRequest, TrainingWorkload


WORKLOAD_DEFAULTS = {
    TrainingWorkload.XOR: TrainingRequest(
        states_per_action=50,
        epochs=80,
    ),
    TrainingWorkload.MNIST: TrainingRequest(
        workload=TrainingWorkload.MNIST,
        number_of_clauses=100,
        states_per_action=128,
        specificity=8.0,
        threshold=10,
        epochs=20,
        seed=20260826,
        boost_true_positive_feedback=True,
    ),
}


class TrainingConfigPanel(VerticalScroll):
    """Labeled workload configuration with validation and stale-state detection."""

    def compose(self) -> ComposeResult:
        yield Static(
            "TRAINING CONFIG  edit values, then press t or choose Train",
            classes="card_title",
        )
        with Grid(id="config-grid"):
            with Vertical(classes="config-field"):
                yield Static("WORKLOAD", classes="config-label")
                yield Select(
                    [("XOR smoke", "xor"), ("MNIST bits", "mnist")],
                    value="xor",
                    id="cfg-workload",
                    allow_blank=False,
                )
            with Vertical(classes="config-field"):
                yield Static("CLAUSES / CLASS", classes="config-label")
                yield Input(id="cfg-clauses", value="20", type="integer")
            with Vertical(classes="config-field"):
                yield Static("STATES / ACTION", classes="config-label")
                yield Input(id="cfg-states", value="50", type="integer")
            with Vertical(classes="config-field"):
                yield Static("SPECIFICITY (s)", classes="config-label")
                yield Input(id="cfg-spec", value="3.0", type="number")
            with Vertical(classes="config-field"):
                yield Static("VOTE THRESHOLD (T)", classes="config-label")
                yield Input(id="cfg-thr", value="10", type="integer")
            with Vertical(classes="config-field"):
                yield Static("EPOCHS", classes="config-label")
                yield Input(id="cfg-epochs", value="80", type="integer")
            with Vertical(classes="config-field"):
                yield Static("RANDOM SEED", classes="config-label")
                yield Input(id="cfg-seed", value="7", type="integer")
            with Vertical(classes="config-field"):
                yield Static("ACTIONS", classes="config-label")
                with Horizontal(id="config-buttons"):
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
            workload=TrainingWorkload(
                str(self.query_one("#cfg-workload", Select).value)
            ),
            boost_true_positive_feedback=(
                self.query_one("#cfg-workload", Select).value == "mnist"
            ),
        )

    def set_status(self, msg: str, is_error: bool = False) -> None:
        status = self.query_one("#cfg-status", Static)
        status.update(msg)
        status.styles.color = "#ff5a6a" if is_error else "#6b7a99"

    def set_from_request(self, req: TrainingRequest) -> None:
        self.query_one("#cfg-workload", Select).value = req.workload.value
        self.query_one("#cfg-clauses", Input).value = str(req.number_of_clauses)
        self.query_one("#cfg-states", Input).value = str(req.states_per_action)
        self.query_one("#cfg-spec", Input).value = str(req.specificity)
        self.query_one("#cfg-thr", Input).value = str(req.threshold)
        self.query_one("#cfg-epochs", Input).value = str(req.epochs)
        self.query_one("#cfg-seed", Input).value = str(req.seed)

    def apply_workload_defaults(
        self, workload: TrainingWorkload
    ) -> TrainingRequest:
        request = WORKLOAD_DEFAULTS[workload]
        self.set_from_request(request)
        return request
