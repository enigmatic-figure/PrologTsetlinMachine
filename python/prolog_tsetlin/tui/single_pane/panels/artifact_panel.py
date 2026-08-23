from __future__ import annotations
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.widgets import Static, Input, Button, DataTable, Label
from textual.app import ComposeResult

class ArtifactPanel(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Static("ARTIFACTS  .ptm export/verify/run-record", classes="card_title")
        yield Static("Last training snapshot can be exported. Load any .ptm to inspect and run a typed record.", classes="card_label")
        with Horizontal(id="artifact-controls"):
            yield Input(placeholder="out/model.ptm", id="artifact-path", value="out/single-pane-xor.ptm")
            yield Button(
                "Export [e]",
                id="artifact-export",
                variant="primary",
                disabled=True,
            )
            yield Button("Load + Verify [l]", id="artifact-verify")
        yield Static("", id="artifact-status", classes="card_label")
        yield DataTable(id="artifact-table", zebra_stripes=True)
        yield Static("RECORD  schema-driven inputs appear after Load", id="record-heading", classes="card_title")
        yield Vertical(id="record-fields")
        yield Button("Run record [r]", id="run-record", variant="success")
        yield Static("", id="record-result", classes="card_label")
        yield DataTable(id="feature-trace", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#artifact-table", DataTable)
        table.add_columns("FIELD", "VALUE")
        table.add_row("Artifact ID", "--")
        table.add_row("Kind", "--")
        table.add_row("Size", "--")
        self.query_one("#feature-trace", DataTable).add_columns(
            "FIELD", "TRANSFORM", "VALUE", "LITERAL ID"
        )

    def set_status(self, msg: str) -> None:
        self.query_one("#artifact-status", Static).update(msg)

    def set_record_result(self, msg: str) -> None:
        self.query_one("#record-result", Static).update(msg)
