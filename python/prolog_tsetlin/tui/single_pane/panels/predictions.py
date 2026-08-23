from __future__ import annotations
from textual.containers import Vertical
from textual.widgets import Static, DataTable
from textual.app import ComposeResult

class PredictionsPanel(Vertical):
    """Shows XOR predictions after training - the classic TUI had this in Train view."""

    def compose(self) -> ComposeResult:
        yield Static(
            "PREDICTIONS  x0/x1 -> target/prediction  (last completed snapshot)",
            id="predictions-title",
            classes="card_title",
        )
        yield DataTable(id="predictions-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#predictions-table", DataTable)
        table.add_columns("x0", "x1", "TARGET", "PREDICTION", "STATUS")
        for x0, x1 in ((0, 0), (0, 1), (1, 0), (1, 1)):
            table.add_row(str(x0), str(x1), str(x0 ^ x1), "--", "--")

    def set_predictions(
        self,
        rows,
        targets,
        predictions,
        *,
        provenance: str = "CURRENT COMPLETED SNAPSHOT",
    ) -> None:
        self.query_one("#predictions-title", Static).update(
            f"[{provenance}] PREDICTIONS  x0/x1 -> target/prediction"
        )
        table = self.query_one("#predictions-table", DataTable)
        table.clear()
        for (x0, x1), target, prediction in zip(rows, targets, predictions):
            status = "OK" if target == prediction else "WRONG"
            table.add_row(
                str(int(x0)),
                str(int(x1)),
                str(target),
                str(prediction),
                status,
            )
