from __future__ import annotations
from textual.containers import Vertical
from textual.widgets import Static, DataTable
from textual.app import ComposeResult

from ....services.diagnostics import ClauseDiagnostics, RunDiagnostics


class ClauseDetailPanel(Vertical):
    """Explain one clause against the completed run and its TA population."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._states: tuple[int, ...] = ()
        self._states_per_action = 0
        self._literal_names: list[str] | None = None
        self._run_diagnostics: RunDiagnostics | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            "CLAUSE DETAIL  select a row in Clauses, press Enter",
            id="clause-detail-title",
            classes="card_title",
        )
        yield Static(
            "No completed clause evaluation.",
            id="clause-behavior-summary",
            classes="graph-legend",
        )
        yield DataTable(
            id="clause-example-table", zebra_stripes=True, cursor_type="row"
        )
        yield Static(
            "TA STATES  input truth unbound — select an example row",
            id="clause-ta-title",
            classes="card_title",
        )
        yield DataTable(id="clause-detail-table", zebra_stripes=True)

    def on_mount(self) -> None:
        self.query_one("#clause-example-table", DataTable).add_columns(
            "ROW",
            "FEATURES",
            "TARGET",
            "MODEL",
            "SCORE",
            "OUTCOME",
            "FIRES",
            "VOTE",
            "INFLUENCE",
        )
        self.query_one("#clause-detail-table", DataTable).add_columns(
            "LITERAL", "STATE", "ACTION", "INPUT TRUTH"
        )

    def show_clause(
        self,
        diagnostic: ClauseDiagnostics,
        run_diagnostics: RunDiagnostics,
        states: tuple[int, ...],
        states_per_action: int,
        literal_names: list[str] | None = None,
        *,
        provenance: str = "CURRENT",
    ) -> None:
        clause_id = diagnostic.clause_id
        self._states = states
        self._states_per_action = states_per_action
        self._literal_names = literal_names
        self._run_diagnostics = run_diagnostics
        examples = self.query_one("#clause-example-table", DataTable)
        examples.clear()
        for clause_example in diagnostic.examples:
            run_example = run_diagnostics.examples[clause_example.example_index]
            influence = (
                "inactive"
                if clause_example.target_aligned is None
                else "aligned"
                if clause_example.target_aligned
                else "opposed"
            )
            examples.add_row(
                str(run_example.example_index),
                "".join("1" if value else "0" for value in run_example.features),
                str(run_example.target),
                str(run_example.prediction),
                f"{run_example.score:+d}",
                "correct" if run_example.correct else "error",
                "yes" if clause_example.fires else "no",
                f"{clause_example.signed_contribution:+d}",
                influence,
            )

        def peer(clause: int | None, similarity: float | None) -> str:
            return (
                "n/a"
                if clause is None or similarity is None
                else f"C{clause} {similarity:.0%}"
            )

        self.query_one("#clause-behavior-summary", Static).update(
            f"SUPPORT {diagnostic.support_count}/{len(run_diagnostics.examples)} "
            f"({diagnostic.support_fraction:.1%})  "
            f"ΣVOTE {diagnostic.signed_vote_sum:+d}  "
            f"ALIGNED/OPPOSED {diagnostic.aligned_count}/{diagnostic.opposed_count}  "
            f"CORRECT/ERROR {diagnostic.correct_activation_count}/"
            f"{diagnostic.incorrect_activation_count}  "
            f"UNIQUE {diagnostic.unique_support_count}  "
            f"LITERAL PEER {peer(diagnostic.literal_peer_clause_id, diagnostic.max_literal_jaccard)}  "
            f"BEHAVIOR PEER {peer(diagnostic.activation_peer_clause_id, diagnostic.max_activation_jaccard)}"
        )

        self._render_states()
        self.query_one("#clause-detail-title", Static).update(
            f"[{provenance}] CLAUSE DETAIL  clause {clause_id}  "
            f"{diagnostic.polarity_label}  "
            f"{len(diagnostic.included_literals)} active literals  "
            f"ACTION BOUNDARY > {states_per_action}"
        )

    def reset(self, *, provenance: str = "CURRENT") -> None:
        self._states = ()
        self._states_per_action = 0
        self._literal_names = None
        self._run_diagnostics = None
        self.query_one("#clause-example-table", DataTable).clear()
        self.query_one("#clause-detail-table", DataTable).clear()
        self.query_one("#clause-detail-title", Static).update(
            f"[{provenance}] CLAUSE DETAIL  select a clause and press Enter"
        )
        self.query_one("#clause-behavior-summary", Static).update(
            "No clause selected for this snapshot."
        )
        self.query_one("#clause-ta-title", Static).update(
            "TA STATES  input truth unbound — select an example row"
        )

    def select_example(self, example_index: int) -> None:
        diagnostics = self._run_diagnostics
        if diagnostics is None or not 0 <= example_index < len(diagnostics.examples):
            raise IndexError(example_index)
        features = diagnostics.examples[example_index].features
        truths: list[bool] = []
        for feature in features:
            truths.extend((feature, not feature))
        self._render_states(tuple(truths))
        self.query_one("#clause-ta-title", Static).update(
            f"TA STATES  input truth bound to example row {example_index}"
        )

    def _render_states(self, input_truths: tuple[bool, ...] | None = None) -> None:
        table = self.query_one("#clause-detail-table", DataTable)
        table.clear()
        if not self._states:
            table.add_row("--", "--", "--", "n/a — select an input")
            return
        for index, state in enumerate(self._states):
            action = "include" if state > self._states_per_action else "exclude"
            literal = (
                self._literal_names[index]
                if self._literal_names and index < len(self._literal_names)
                else f"lit{index}"
            )
            truth = (
                "n/a"
                if input_truths is None or index >= len(input_truths)
                else "true"
                if input_truths[index]
                else "false"
            )
            table.add_row(literal, str(state), action, truth)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "clause-example-table":
            return
        row = event.data_table.get_row_at(event.cursor_row)
        self.select_example(int(str(row[0])))
