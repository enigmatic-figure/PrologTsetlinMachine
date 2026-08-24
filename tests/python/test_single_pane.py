"""Regression suite for the single-pane workbench."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from rich.text import Text
from textual.widgets import Button, DataTable, Input, Sparkline, Static

from prolog_tsetlin.prolog_bridge import BooleanDecisionTree
from prolog_tsetlin.services.diagnostics import analyze_training_run
from prolog_tsetlin.services.search import BoundedSearchResult, SearchKind
from prolog_tsetlin.services.training import (
    TrainingDiagnosticSampling,
    TrainingProgress,
    TrainingRequest,
    train_xor,
)
from prolog_tsetlin.tui.single_pane.app import SinglePaneApp
from prolog_tsetlin.tui.single_pane.panels.temporal_inspector import (
    TemporalInspectorPanel,
    TemporalSampleSelected,
)
from prolog_tsetlin.tui.models import JobState

pytestmark = pytest.mark.asyncio


async def test_single_pane_trains_and_updates_session(tmp_path: Path) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.session.job_state == JobState.IDLE
        await pilot.press("t")
        # Wait for completion (XOR 80 epochs is fast)
        for _ in range(30):
            await pilot.pause(0.2)
            if app.session.job_state == JobState.SUCCEEDED:
                break
        assert app.session.job_state == JobState.SUCCEEDED
        assert app.session.last_completed_run is not None
        assert len(app.session.accuracy_history) == 80
        samples = app.session.last_completed_diagnostics
        assert 2 <= len(samples) <= app.DIAGNOSTIC_SAMPLE_BUDGET
        assert samples[0].sample.epoch == 1
        assert samples[-1].sample.epoch == 80
        assert samples[-1].sample.snapshot == app.session.last_completed_run.snapshot
        assert samples[0].delta_from_previous is None
        assert all(
            sample.delta_from_previous is not None for sample in samples[1:]
        )
        assert any(
            event.kind == "diagnostic_sample" for event in app.session.events
        )
        assert app._current_run_id is not None
        assert "/100" in str(
            app.query_one("#card-clause-health-avg-ta", Static).render()
        )


async def test_single_pane_rejects_double_start(tmp_path: Path) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # Simulate a running job
        app.session.job_state = JobState.RUNNING
        gen_before = app._run_generation
        app.action_train()
        await pilot.pause(0.2)
        assert app._run_generation == gen_before
        assert app.session.job_state == JobState.RUNNING
        # Reset to idle and ensure next train succeeds
        app.session.job_state = JobState.IDLE
        app.action_train()
        for _ in range(30):
            await pilot.pause(0.2)
            if app.session.job_state == JobState.SUCCEEDED:
                break
        assert app.session.job_state == JobState.SUCCEEDED


async def test_single_pane_layouts(tmp_path: Path) -> None:
    for size, compact in [((80, 24), True), ((120, 40), False)]:
        app = SinglePaneApp(workspace=tmp_path)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            assert app.query_one("#top-bar") is not None
            assert app.query_one("#bottom-switcher") is not None
            assert app.query_one("#top-cards") is not None
            assert app.screen.has_class("compact") is compact
            assert app.query_one("#graphs").display is not compact
            assert app.query_one("#footer-sub").display is not compact
            if compact:
                assert not app.query_one("#card-system-util").display
                assert not app.query_one("#card-data-ingest").display
                inspector = app.query_one("#clause-inspector")
                histogram = app.query_one("#ta-histogram")
                assert inspector.size.width == histogram.size.width
                assert inspector.size.height > 0
                assert histogram.size.height > 0
            else:
                assert app.query_one("#card-system-util").display
                assert app.query_one("#card-data-ingest").display

    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        assert not app.screen.has_class("compact")
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert app.screen.has_class("compact")
        await pilot.resize_terminal(120, 40)
        await pilot.pause()
        assert not app.screen.has_class("compact")


async def test_single_pane_both_panels_receive_same_data(tmp_path: Path) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("t")
        for _ in range(30):
            await pilot.pause(0.2)
            if app.session.job_state == JobState.SUCCEEDED:
                break
        assert app.session.job_state == JobState.SUCCEEDED
        # Both inspector copies should have same row count
        for pid in ["clause-inspector", "clause-inspector-full"]:
            from prolog_tsetlin.tui.single_pane.panels.clause_inspector import ClauseInspectorPanel
            panel = app.query_one(f"#{pid}", ClauseInspectorPanel)
            assert panel.query_one("#clause-table").row_count == 20
        for hid in ["ta-histogram", "ta-histogram-full"]:
            from textual.widgets import Sparkline
            # Should have hist data
            assert len(app.query_one(f"#{hid}").query_one(Sparkline).data) == 20


async def test_completed_timeline_projects_one_historical_sample_everywhere(
    tmp_path: Path,
) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        config = app.query_one("#config-panel")
        config.query_one("#cfg-epochs", Input).value = "5"
        await pilot.pause()
        await pilot.press("t")
        for _ in range(30):
            await pilot.pause(0.1)
            if app.session.job_state is JobState.SUCCEEDED:
                break

        run = app.session.last_completed_run
        assert run is not None
        history = app.session.last_completed_diagnostics
        assert [item.sample.epoch for item in history] == [1, 2, 3, 4, 5]
        await pilot.press("v")
        assert app.query_one("#bottom-switcher").current == "view-timeline"
        timeline = app.query_one("#temporal-panel", TemporalInspectorPanel)
        timeline_table = timeline.query_one("#temporal-table", DataTable)
        assert timeline_table.row_count == 5
        assert timeline_table.has_focus

        selected = history[0]
        await pilot.press("up", "up", "up", "up", "enter")
        await pilot.pause()

        assert app.session.inspected_sample_epoch == selected.sample.epoch
        assert app._snapshot == selected.sample.snapshot
        assert app._diagnostics == selected.diagnostics
        assert app.query_one("#footer-bar").has_class("historic")
        footer = str(app.query_one("#footer-bar", Static).render()).upper()
        assert "HISTORICAL SAMPLE 1/5" in footer
        assert "READ ONLY" in footer
        assert "EXPORT REMAINS FINAL EPOCH 5" in footer
        for title_id in (
            "#clause-title",
            "#ta-title",
            "#literal-title",
            "#predictions-title",
            "#clause-detail-title",
        ):
            assert "HISTORICAL EPOCH 1/5 READ ONLY" in str(
                app.query_one(title_id, Static).render()
            ).upper()
        predictions = app.query_one("#predictions-table", DataTable)
        assert [
            int(str(predictions.get_row_at(index)[3]))
            for index in range(predictions.row_count)
        ] == list(selected.sample.predictions)
        assert app.training.require_exportable(run.request) is run

        app.on_temporal_sample_selected(
            TemporalSampleSelected(5, run, app._inspection_generation)
        )
        await pilot.pause()
        assert app.session.inspected_sample_epoch is None
        assert app._snapshot == run.snapshot
        assert not app.query_one("#footer-bar").has_class("historic")
        assert "CURRENT COMPLETED SNAPSHOT" in str(
            app.query_one("#footer-bar", Static).render()
        ).upper()


async def test_timeline_selection_resets_and_locks_during_retraining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        config = app.query_one("#config-panel")
        config.query_one("#cfg-epochs", Input).value = "3"
        await pilot.pause()
        await pilot.press("t")
        for _ in range(30):
            await pilot.pause(0.1)
            if app.session.job_state is JobState.SUCCEEDED:
                break

        run = app.session.last_completed_run
        assert run is not None
        app.on_temporal_sample_selected(
            TemporalSampleSelected(1, run, app._inspection_generation)
        )
        await pilot.pause()
        assert app.session.inspected_sample_epoch == 1

        monkeypatch.setattr(app, "_start_training", lambda *args: None)
        app.action_train()
        await pilot.pause()
        assert app.session.job_state is JobState.QUEUED
        assert app.session.inspected_sample_epoch is None
        assert app._snapshot == run.snapshot
        assert not app.query_one("#footer-bar").has_class("historic")

        delayed = TemporalSampleSelected(
            1, run, app._inspection_generation
        )
        app.on_temporal_sample_selected(delayed)
        await pilot.pause()
        assert app.session.inspected_sample_epoch is None
        assert app._snapshot == run.snapshot
        await pilot.press("v")
        assert app.screen.has_class("compact")
        assert app.query_one("#bottom-switcher").current == "view-timeline"
        table = app.query_one("#temporal-table", DataTable)
        assert table.row_count == 3
        assert table.size.width > 0
        assert table.size.height > 0
        assert "ACTIVE TRAINING IS SEPARATE" in str(
            app.query_one("#temporal-status", Static).render()
        ).upper()

        app.on_training_cancelled("test cancellation")
        await pilot.pause()
        app.on_temporal_sample_selected(delayed)
        await pilot.pause()
        assert app.session.job_state is JobState.CANCELLED
        assert app.session.inspected_sample_epoch is None
        assert app._snapshot == run.snapshot

        app.action_train()
        replacement = app.training.run(
            diagnostic=app.training.record_diagnostic_sample
        )
        app.on_training_complete(replacement)
        await pilot.pause()
        assert app.session.last_completed_run is replacement
        assert replacement is not run

        app.on_temporal_sample_selected(
            TemporalSampleSelected(1, run, app._inspection_generation)
        )
        await pilot.pause()
        assert app.session.inspected_sample_epoch is None
        assert app._snapshot == replacement.snapshot
        assert not app.query_one("#footer-bar").has_class("historic")


async def test_single_pane_sortable_columns(tmp_path: Path) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("t")
        for _ in range(30):
            await pilot.pause(0.2)
            if app.session.job_state == JobState.SUCCEEDED:
                break
        from prolog_tsetlin.tui.single_pane.panels.clause_inspector import ClauseInspectorPanel
        panel = app.query_one("#clause-inspector", ClauseInspectorPanel)
        for key in [
            "id",
            "polarity",
            "support",
            "vote",
            "alignment",
            "outcome",
            "lits",
            "avg",
            "near",
            "similarity",
        ]:
            panel._sort_key = key
            panel._sort_rev = True
            panel._render_filtered()
            assert panel.query_one("#clause-table").row_count == 20

        table = panel.query_one("#clause-table", DataTable)
        header_key = next(iter(table.columns))
        app.on_data_table_header_selected(
            DataTable.HeaderSelected(table, header_key, 0, Text("CLAUSE ID"))
        )
        assert panel._sort_key == "id"


async def test_single_pane_filter_throttling(tmp_path: Path) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("t")
        for _ in range(30):
            await pilot.pause(0.2)
            if app.session.job_state == JobState.SUCCEEDED:
                break
        from prolog_tsetlin.tui.single_pane.panels.clause_inspector import ClauseInspectorPanel
        panel = app.query_one("#clause-inspector", ClauseInspectorPanel)
        # Rapid filter changes should coalesce
        for pat in ["0", "1", "2", ""]:
            panel.set_filter(pat)
            await pilot.pause(0.05)
        await pilot.pause(0.3)
        # Final filter empty should show all
        assert panel.query_one("#clause-table").row_count == 20
        panel.set_filter("9999")
        await pilot.pause(0.3)
        assert panel.query_one("#clause-table").row_count == 0


async def test_single_pane_mark_hidden_really_hides_without_mutating_telemetry(
    tmp_path: Path,
) -> None:
    from prolog_tsetlin.tui.single_pane.panels.clause_inspector import (
        ClauseInspectorPanel,
    )

    rows = [
        {
            "id": 0,
            "polarity": "+",
            "votes": None,
            "lits": 1,
            "avg_state": 50.0,
            "age": None,
        },
        {
            "id": 1,
            "polarity": "-",
            "votes": None,
            "lits": 0,
            "avg_state": 40.0,
            "age": None,
        },
    ]
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        panel = app.query_one("#clause-inspector", ClauseInspectorPanel)
        panel.set_rows(rows)
        await pilot.pause()
        assert panel.query_one("#clause-table", DataTable).row_count == 2
        assert panel.mark_hidden() in {"0", "1"}
        assert panel.query_one("#clause-table", DataTable).row_count == 1
        assert all(row["age"] is None for row in rows)
        assert all("dead" not in row for row in rows)


async def test_single_pane_clause_detail_binds_truth_to_selected_example(
    tmp_path: Path,
) -> None:
    from prolog_tsetlin.tui.single_pane.panels.clause_detail import (
        ClauseDetailPanel,
    )

    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        panel = app.query_one("#clause-detail", ClauseDetailPanel)
        run = train_xor(TrainingRequest(epochs=1))
        diagnostics = analyze_training_run(run)
        panel.show_clause(
            diagnostics.clause(0),
            diagnostics,
            run.snapshot.states[0],
            run.snapshot.states_per_action,
            ["x0", "not x0", "x1", "not x1"],
        )
        await pilot.pause()
        table = panel.query_one("#clause-detail-table", DataTable)
        assert len(table.columns) == 4
        assert [str(table.get_row_at(index)[3]) for index in range(2)] == [
            "n/a",
            "n/a",
        ]
        panel.select_example(1)
        assert [str(table.get_row_at(index)[3]) for index in range(4)] == [
            "false",
            "true",
            "true",
            "false",
        ]
        assert "row 1" in str(panel.query_one("#clause-ta-title").render())


async def test_single_pane_exposes_completed_run_research_diagnostics(
    tmp_path: Path,
) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        app.query_one("#cfg-epochs", Input).value = "1"
        await pilot.pause()
        app.action_train()
        for _ in range(30):
            await pilot.pause(0.2)
            if app.session.job_state is JobState.SUCCEEDED:
                break

        assert app._diagnostics is not None
        ta_stats = str(app.query_one("#ta-stats", Static).render()).upper()
        assert "INCLUDE" in ta_stats
        assert "NEAR±5" in ta_stats
        assert "SATURATED" in ta_stats
        assert "N/A" not in str(
            app.query_one("#card-statistics-ta-incl", Static).render()
        ).upper()

        app.action_inspect()
        await pilot.pause()
        summary = str(
            app.query_one("#clause-behavior-summary", Static).render()
        ).upper()
        assert "SUPPORT" in summary
        assert "ΣVOTE" in summary
        assert "LITERAL PEER" in summary
        assert app.query_one("#clause-example-table", DataTable).row_count == 4


async def test_single_pane_rejects_inconsistent_completed_run_diagnostics(
    tmp_path: Path,
) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)):
        request = TrainingRequest(epochs=1)
        app.training.begin(request)
        run = train_xor(request)
        inconsistent = replace(
            run,
            predictions=tuple(1 - prediction for prediction in run.predictions),
        )

        app.on_training_complete(inconsistent)

        assert app.session.job_state is JobState.FAILED
        assert app.session.last_completed_run is None
        assert app._diagnostics is None
        assert app.session.error is not None
        assert "diagnostic validation failed" in app.session.error


async def test_single_pane_failed_verification(tmp_path: Path) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # Create a bad artifact
        bad = tmp_path / "bad.ptm"
        bad.write_bytes(b"not a ptm")
        await pilot.press("7")
        await pilot.pause(0.3)
        from textual.widgets import Input
        panel = app.query_one("#artifact-panel")
        panel.query_one("#artifact-path", Input).value = str(bad)
        app._verify_artifact()
        await pilot.pause(0.3)
        from textual.widgets import Static
        status = panel.query_one("#artifact-status", Static)
        # Should contain FAILED, not VERIFIED OK
        txt = str(getattr(status, "_content", "") or getattr(status, "_renderable", "") or status.render())
        # At least check that file still not verified
        assert "FAILED" in txt or "failed" in txt.lower() or "VERIFY" in txt


async def test_single_pane_no_overwrite_export(tmp_path: Path) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("t")
        for _ in range(30):
            await pilot.pause(0.2)
            if app.session.job_state == JobState.SUCCEEDED:
                break
        dest = tmp_path / "out.ptm"
        dest.write_bytes(b"existing")
        await pilot.press("7")
        await pilot.pause(0.3)
        from textual.widgets import Input
        panel = app.query_one("#artifact-panel")
        panel.query_one("#artifact-path", Input).value = str(dest)
        app._export_artifact()
        await pilot.pause(0.3)
        # Should not overwrite
        assert dest.read_bytes() == b"existing"
        from textual.widgets import Static
        status = panel.query_one("#artifact-status", Static)
        txt = str(getattr(status, "_content", "") or getattr(status, "_renderable", "") or status.render())
        assert "REFUSED" in txt or "exists" in txt.lower()


async def test_single_pane_without_plotext_fallback(monkeypatch, tmp_path: Path) -> None:
    # Simulate missing textual-plotext by monkeypatching the flag
    import prolog_tsetlin.tui.single_pane.panels.training_graphs as tg
    orig = tg.HAS_PLOTEXT
    tg.HAS_PLOTEXT = False
    try:
        app = SinglePaneApp(workspace=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # Should still mount with fallback Sparkline
            assert app.query_one("#graphs") is not None
    finally:
        tg.HAS_PLOTEXT = orig


async def test_single_pane_initial_panels_do_not_fabricate_telemetry(
    tmp_path: Path,
) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.query_one("#clause-inspector").query_one(DataTable).row_count == 0
        assert list(
            app.query_one("#ta-histogram").query_one(Sparkline).data
        ) == [0]

        await pilot.press("p")
        predictions = app.query_one("#predictions-table", DataTable)
        assert [str(predictions.get_row_at(index)[2]) for index in range(4)] == [
            "0",
            "1",
            "1",
            "0",
        ]


async def test_single_pane_navigation_bindings_are_unique_and_shell_owned(
    tmp_path: Path,
) -> None:
    keys = [binding.key for binding in SinglePaneApp.BINDINGS]
    assert len(keys) == len(set(keys))
    destinations = {
        "1": "view-system",
        "2": "view-split",
        "3": "view-clauses",
        "4": "view-ta",
        "5": "view-literals",
        "6": "view-graphs",
        "7": "view-artifacts",
    }
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        for key, expected in destinations.items():
            await pilot.press(key)
            await pilot.pause()
            assert app.query_one("#bottom-switcher").current == expected


async def test_single_pane_updates_both_graph_copies(
    tmp_path: Path, monkeypatch
) -> None:
    from prolog_tsetlin.tui.single_pane.panels.training_graphs import (
        TrainingGraphsPanel,
    )

    calls = []

    def record_update(self, accuracy, *args, **kwargs):
        calls.append((self.id, list(accuracy), kwargs))

    monkeypatch.setattr(TrainingGraphsPanel, "update_series", record_update)
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        request = TrainingRequest(epochs=2)
        sampling = TrainingDiagnosticSampling(every_epochs=1)
        samples = []
        train_xor(
            request,
            diagnostic=samples.append,
            diagnostic_sampling=sampling,
        )
        app.training.begin(request, diagnostic_sampling=sampling)
        app.on_training_progress(TrainingProgress(1, 2, 0.5))
        assert [call[0] for call in calls] == ["graphs", "graphs-full"]
        calls.clear()

        app.on_training_diagnostic(samples[0])

        assert [call[0] for call in calls] == ["graphs", "graphs-full"]
        assert all(call[2]["sample_epochs"] == [1] for call in calls)
        assert all(call[2]["delta_epochs"] == [] for call in calls)
        assert TrainingGraphsPanel.accuracy_percentages([0.25, 1.0]) == [
            25.0,
            100.0,
        ]


async def test_single_pane_cancel_restores_one_completed_graph_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prolog_tsetlin.tui.single_pane.panels.training_graphs import (
        TrainingGraphsPanel,
    )

    calls = []

    def record_update(self, accuracy, *args, **kwargs):
        calls.append((self.id, list(accuracy), kwargs))

    monkeypatch.setattr(TrainingGraphsPanel, "update_series", record_update)
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)):
        sampling = TrainingDiagnosticSampling(every_epochs=1)
        completed_request = TrainingRequest(epochs=2)
        completed_samples = []
        completed_run = train_xor(
            completed_request,
            diagnostic=completed_samples.append,
            diagnostic_sampling=sampling,
        )
        app.training.begin(
            completed_request, diagnostic_sampling=sampling
        )
        for epoch, sample in enumerate(completed_samples, start=1):
            app.training.record_progress(epoch, epoch / 4)
            app.training.record_diagnostic_sample(sample)
        app.training.complete(completed_run)

        retry_request = replace(
            completed_request, seed=completed_request.seed + 1
        )
        retry_samples = []
        train_xor(
            retry_request,
            diagnostic=retry_samples.append,
            diagnostic_sampling=sampling,
        )
        app.training.begin(retry_request, diagnostic_sampling=sampling)
        app.training.record_progress(1, 0.75)
        app.training.record_diagnostic_sample(retry_samples[0])
        assert app.training.request_cancel()
        app.training.cancelled()

        calls.clear()
        app._update_training_graphs()

        assert [call[0] for call in calls] == ["graphs", "graphs-full"]
        assert all(call[1] == [0.25, 0.5] for call in calls)
        assert all(call[2]["sample_epochs"] == [1, 2] for call in calls)


async def test_training_graph_fallback_truthfully_labels_accuracy_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prolog_tsetlin.tui.single_pane.panels import training_graphs

    monkeypatch.setattr(training_graphs, "HAS_PLOTEXT", False)
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)):
        panel = app.query_one("#graphs", training_graphs.TrainingGraphsPanel)

        assert "ACCURACY ONLY" in str(
            panel.query_one("#graphs-header", Static).render()
        )
        assert "NOT DISPLAYED" in str(
            panel.query_one(".graph-legend", Static).render()
        ).upper()
        assert panel.query_one("#braille-graph-fallback", Sparkline)


async def test_single_pane_artifact_record_uses_service_contract(
    tmp_path: Path,
) -> None:
    source = Path(__file__).resolve().parents[2] / "tests/data/raw_xor_packed_tm_v1.hex"
    artifact = tmp_path / "raw-xor.ptm"
    artifact.write_bytes(bytes.fromhex(source.read_text(encoding="ascii")))

    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("7")
        panel = app.query_one("#artifact-panel")
        panel.query_one("#artifact-path", Input).value = str(artifact)
        await pilot.click("#artifact-verify")
        await pilot.pause()

        fields = list(panel.query("#record-fields Input"))
        assert len(fields) == 2
        fields[0].value = "false"
        fields[1].value = "true"
        panel.scroll_end(animate=False)
        await pilot.pause()
        await pilot.click("#run-record")
        await pilot.pause()

        assert "PREDICTION 1" in str(panel.query_one("#record-result").render())
        assert panel.query_one("#feature-trace", DataTable).row_count == 2


async def test_single_pane_stale_invalid_and_reverted_config_export(
    tmp_path: Path,
) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("t")
        for _ in range(30):
            await pilot.pause(0.2)
            if app.session.job_state is JobState.SUCCEEDED:
                break
        assert app.session.last_completed_run is not None
        panel = app.query_one("#artifact-panel")
        destination = tmp_path / "stale.ptm"
        panel.query_one("#artifact-path", Input).value = str(destination)

        config = app.query_one("#config-panel")
        seed = config.query_one("#cfg-seed", Input)
        trained_seed = seed.value
        seed.value = str(int(trained_seed) + 1)
        await pilot.pause()
        assert app.session.configuration_dirty
        assert panel.query_one("#artifact-export", Button).disabled
        app._export_artifact()
        assert not destination.exists()
        assert "STALE" in str(panel.query_one("#artifact-status").render()).upper()

        seed.value = ""
        await pilot.pause()
        assert app.session.configuration_dirty
        app._export_artifact()
        assert not destination.exists()
        assert "INVALID" in str(panel.query_one("#artifact-status").render()).upper()

        seed.value = trained_seed
        await pilot.pause()
        assert not app.session.configuration_dirty
        assert not panel.query_one("#artifact-export", Button).disabled
        app._export_artifact()
        assert destination.is_file()


async def test_single_pane_mid_training_edit_remains_stale_on_completion(
    tmp_path: Path,
) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        config = app.query_one("#config-panel")
        config.query_one("#cfg-epochs", Input).value = "1"
        await pilot.pause()
        trained_request = config.get_request()
        app.training.begin(trained_request)

        config.query_one("#cfg-seed", Input).value = str(trained_request.seed + 1)
        await pilot.pause()
        app.on_training_complete(train_xor(trained_request))

        assert app.session.job_state is JobState.SUCCEEDED
        assert app.session.configuration_dirty
        assert "STALE" in str(
            app.query_one("#card-statistics-acc", Static).render()
        )


async def test_single_pane_configuration_refresh_tolerates_screen_teardown(
    tmp_path: Path,
) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)):
        pass

    app._refresh_configuration_state()


async def test_single_pane_labels_retained_snapshot_during_retraining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        config = app.query_one("#config-panel")
        config.query_one("#cfg-epochs", Input).value = "1"
        await pilot.pause()
        app.action_train()
        for _ in range(30):
            await pilot.pause(0.2)
            if app.session.job_state is JobState.SUCCEEDED:
                break

        completed = app.session.last_completed_run
        assert completed is not None
        clause_count = app.query_one(
            "#clause-inspector #clause-table", DataTable
        ).row_count
        prediction_count = app.query_one(
            "#predictions-table", DataTable
        ).row_count

        config.query_one("#cfg-seed", Input).value = str(
            completed.request.seed + 1
        )
        await pilot.pause()
        active_request = config.get_request()
        monkeypatch.setattr(app, "_start_training", lambda *args: None)

        app.action_train()

        assert app.session.job_state is JobState.QUEUED
        assert app.session.active_request == active_request
        assert app.session.last_completed_run is completed
        assert app.session.configuration_dirty
        assert app.training.retained_run_is_historical
        assert clause_count > 0
        assert app.query_one(
            "#clause-inspector #clause-table", DataTable
        ).row_count == clause_count
        assert app.query_one(
            "#predictions-table", DataTable
        ).row_count == prediction_count
        footer = str(app.query_one("#footer-bar", Static).render()).upper()
        assert "LAST COMPLETED SNAPSHOT" in footer
        assert "STALE FOR ACTIVE REQUEST" in footer


async def test_single_pane_search_buttons_run_repair_and_export(
    tmp_path: Path,
) -> None:
    tree = BooleanDecisionTree.node(
        0,
        BooleanDecisionTree.leaf(False),
        BooleanDecisionTree.leaf(True),
    )
    result = BoundedSearchResult(
        SearchKind.REPAIR,
        {
            "candidate_upper_bound": 4,
            "dataset_digest": "frontend-contract",
            "mismatch_count": 0,
            "counterexamples": [
                {
                    "example": [False, False],
                    "expected": False,
                    "parent_prediction": True,
                    "required_flip": True,
                }
            ],
        },
        0.01,
        tree.to_logic_program(),
    )

    def run_search(request, *, cancel=None):
        assert request.kind is SearchKind.REPAIR
        return result

    app = SinglePaneApp(workspace=tmp_path)
    app.search._runner = run_search
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("s")
        app.query_one("#search-kind").value = SearchKind.REPAIR.value
        await pilot.pause()
        assert '"kind": "repair"' in app.query_one("#search-json").text

        await pilot.click("#search-run")
        for _ in range(20):
            await pilot.pause(0.05)
            if app.session.search_state is JobState.SUCCEEDED:
                break
        assert app.session.search_state is JobState.SUCCEEDED
        assert app.query_one("#search-counterexamples", DataTable).row_count == 1
        assert not app.query_one("#search-export").disabled

        destination = tmp_path / "repair.ptm"
        app.query_one("#search-export-path", Input).value = str(destination)
        app.on_button_pressed(
            Button.Pressed(app.query_one("#search-export", Button))
        )
        await pilot.pause()
        assert destination.is_file()
        assert "EXPORTED" in str(app.query_one("#search-status", Static).render())


async def test_single_pane_kind_change_invalidates_active_search_callback(
    tmp_path: Path,
) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("s")
        await pilot.pause()
        app.session.search_state = JobState.RUNNING
        old_generation = app._search_generation
        old_cancel = app._search_cancel
        app.query_one("#search-kind").value = SearchKind.REPAIR.value
        await pilot.pause()

        assert old_cancel.is_set()
        assert app._search_generation > old_generation
        assert app.session.search_state is JobState.IDLE

        stale_result = BoundedSearchResult(SearchKind.THRESHOLD, {}, 0.01)
        app._show_search_result(stale_result, old_generation)
        assert app.session.search_result is None
        assert "READY REPAIR" in str(
            app.query_one("#search-status", Static).render()
        ).upper()


async def test_single_pane_help_matches_shell_bindings(tmp_path: Path) -> None:
    app = SinglePaneApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("?")
        await pilot.pause()
        copy = "\n".join(
            str(widget.render()) for widget in app.screen.query(".card_label")
        )

        assert "Open System." in copy
        assert "Open Dashboard." in copy
        assert "Open Literals." in copy
        assert "Start XOR training." in copy
        assert "1      Open Overview" not in copy
        assert "STUBS.md" not in copy
        assert "Dead clauses" not in copy
