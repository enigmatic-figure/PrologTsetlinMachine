from __future__ import annotations
import json
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Static, Button, Select, Input, TextArea, DataTable
from textual.app import ComposeResult
from ....services.search import SearchKind, demo_search_document

class SearchPanel(VerticalScroll):
    """Dense workbench search over the shared search service."""

    def compose(self) -> ComposeResult:
        yield Static("BOUNDED SEARCH  Prolog  threshold/template/clause/tree/repair  30s default", classes="card_title")
        with Horizontal(id="search-controls"):
            yield Select(
                [(k.value, k.value) for k in SearchKind],
                value=SearchKind.DECISION_TREE.value,
                id="search-kind",
                allow_blank=False,
            )
            yield Input("30", placeholder="Timeout s", id="search-timeout", type="number")
            yield Button("Run [F5]", id="search-run", variant="success")
            yield Button("Cancel [F6]", id="search-cancel", variant="error", disabled=True)
        yield TextArea(json.dumps(demo_search_document(SearchKind.DECISION_TREE), indent=2), id="search-json", language="json")
        yield Static("", id="search-status", classes="card_label")
        yield TextArea("", id="search-result", language="json", read_only=True)
        yield DataTable(id="search-counterexamples", zebra_stripes=True)
        yield Static("Export: path for fixed-Logic artifact", classes="card_label")
        with Horizontal(id="search-export-controls"):
            yield Input("out/search.ptm", id="search-export-path")
            yield Button("Export Logic", id="search-export", disabled=True)

    def on_mount(self) -> None:
        self.query_one("#search-counterexamples", DataTable).add_columns(
            "EXAMPLE", "EXPECTED", "PARENT", "FLIP"
        )

    def get_request_dict(self) -> dict:
        try:
            txt = self.query_one("#search-json", TextArea).text
            doc = json.loads(txt)
            if not isinstance(doc, dict):
                raise ValueError("search JSON must contain an object")
            return doc
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"search JSON invalid: {error}") from error

    def selected_kind(self) -> SearchKind:
        return SearchKind(str(self.query_one("#search-kind", Select).value))

    def timeout_seconds(self) -> float:
        raw = self.query_one("#search-timeout", Input).value.strip() or "30"
        try:
            return float(raw)
        except ValueError as error:
            raise ValueError("search timeout must be a number") from error

    def set_result(self, text: str, status: str = "") -> None:
        self.query_one("#search-result", TextArea).text = text
        self.query_one("#search-status", Static).update(status)

    def set_counterexamples(self, rows: list[dict]) -> None:
        tbl = self.query_one("#search-counterexamples", DataTable)
        tbl.clear()
        for row in rows:
            tbl.add_row(
                str(row.get("example", "")),
                str(row.get("expected", "")),
                str(row.get("parent_prediction", "")),
                str(row.get("required_flip", "")),
            )

    def set_controls(self, *, active: bool, exportable: bool = False) -> None:
        self.query_one("#search-run", Button).disabled = active
        self.query_one("#search-cancel", Button).disabled = not active
        self.query_one("#search-export", Button).disabled = active or not exportable
