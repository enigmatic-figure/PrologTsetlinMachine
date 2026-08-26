from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from ....services.training import MulticlassTrainingRun


class PredictionsPanel(Vertical):
    """Show binary predictions or multiclass validation outcomes."""

    def compose(self) -> ComposeResult:
        yield Static(
            "PREDICTIONS  x0/x1 -> target/prediction  (last completed snapshot)",
            id="predictions-title",
            classes="card_title",
        )
        yield DataTable(id="predictions-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#predictions-table", DataTable)
        self._binary_columns(table)
        for x0, x1 in ((0, 0), (0, 1), (1, 0), (1, 1)):
            table.add_row(str(x0), str(x1), str(x0 ^ x1), "--", "--")

    @staticmethod
    def _binary_columns(table: DataTable) -> None:
        table.clear(columns=True)
        table.add_columns("x0", "x1", "TARGET", "PREDICTION", "STATUS")

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
        self._binary_columns(table)
        for (x0, x1), target, prediction in zip(rows, targets, predictions):
            status = "OK" if target == prediction else "WRONG"
            table.add_row(
                str(int(x0)),
                str(int(x1)),
                str(target),
                str(prediction),
                status,
            )

    def set_multiclass(self, run: MulticlassTrainingRun) -> None:
        run.validate()
        self.query_one("#predictions-title", Static).update(
            "[CURRENT COMPLETED RUN] MNIST VALIDATION BY CLASS  "
            "rows are truth labels"
        )
        table = self.query_one("#predictions-table", DataTable)
        table.clear(columns=True)
        table.add_columns(
            "CLASS",
            "CORRECT",
            "TOTAL",
            "ACCURACY",
            "MOST CONFUSED AS",
        )
        for label, row in zip(run.class_labels, run.confusion_matrix):
            total = sum(row)
            correct = row[label]
            wrong = [
                (count, prediction)
                for prediction, count in enumerate(row)
                if prediction != label
            ]
            confused_count, confused_class = max(wrong)
            confused = (
                "--"
                if confused_count == 0
                else f"{confused_class} ({confused_count})"
            )
            table.add_row(
                str(label),
                str(correct),
                str(total),
                f"{correct / total:.1%}" if total else "n/a",
                confused,
            )
