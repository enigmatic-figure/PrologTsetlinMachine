"""Behavioral contracts that every PTM Textual shell must satisfy."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import DataTable, Input, Select, Static

from prolog_tsetlin.prolog_bridge import BooleanDecisionTree
from prolog_tsetlin.services.search import BoundedSearchResult, SearchKind
from prolog_tsetlin.tui.app import PTMApp
from prolog_tsetlin.tui.models import JobState
from prolog_tsetlin.tui.single_pane.app import SinglePaneApp


SHELLS = pytest.mark.parametrize("shell", ("classic", "single-pane"))


def make_app(shell: str, workspace: Path):
    return PTMApp(workspace=workspace) if shell == "classic" else SinglePaneApp(
        workspace=workspace
    )


async def wait_for_job(pilot, app, attribute: str) -> JobState:
    for _ in range(60):
        state = getattr(app.session, attribute)
        if state in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED):
            return state
        await pilot.pause(0.05)
    return getattr(app.session, attribute)


def raw_xor_artifact(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[2] / "tests/data/raw_xor_packed_tm_v1.hex"
    artifact = tmp_path / "raw-xor.ptm"
    artifact.write_bytes(bytes.fromhex(source.read_text(encoding="ascii")))
    return artifact


@SHELLS
@pytest.mark.asyncio
async def test_frontend_contract_train_xor_has_four_predictions(
    shell: str, tmp_path: Path
) -> None:
    app = make_app(shell, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        app.action_train()
        assert await wait_for_job(pilot, app, "job_state") is JobState.SUCCEEDED
        assert app.session.last_completed_run is not None
        assert len(app.session.last_completed_run.predictions) == 4
        assert app.session.last_completed_run.targets == (0, 1, 1, 0)


@SHELLS
@pytest.mark.asyncio
async def test_frontend_contract_dirty_or_invalid_config_forbids_export(
    shell: str, tmp_path: Path
) -> None:
    app = make_app(shell, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        app.action_train()
        assert await wait_for_job(pilot, app, "job_state") is JobState.SUCCEEDED
        seed_selector = "#config-seed" if shell == "classic" else "#cfg-seed"
        seed = app.query_one(seed_selector, Input)
        trained_seed = seed.value
        destination = tmp_path / f"{shell}.ptm"
        if shell == "classic":
            app.query_one("#artifact-path", Input).value = str(destination)
        else:
            app.query_one("#artifact-panel").query_one(
                "#artifact-path", Input
            ).value = str(destination)

        seed.value = str(int(trained_seed) + 1)
        await pilot.pause()
        assert app.session.configuration_dirty
        app.action_export()
        assert not destination.exists()

        seed.value = ""
        await pilot.pause()
        assert app.session.configuration_dirty
        app.action_export()
        assert not destination.exists()

        seed.value = trained_seed
        await pilot.pause()
        assert not app.session.configuration_dirty
        app.action_export()
        assert destination.is_file()


@SHELLS
@pytest.mark.asyncio
async def test_frontend_contract_schema_record_predicts_one(
    shell: str, tmp_path: Path
) -> None:
    artifact = raw_xor_artifact(tmp_path)
    app = make_app(shell, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        if shell == "classic":
            app.query_one("#artifact-open-path", Input).value = str(artifact)
            await app.action_load_artifact()
            fields = list(app.query("#record-fields Input"))
        else:
            panel = app.query_one("#artifact-panel")
            panel.query_one("#artifact-path", Input).value = str(artifact)
            app.action_load_artifact()
            await pilot.pause()
            fields = list(panel.query("#record-fields Input"))

        assert len(fields) == 2
        artifact.write_bytes(b"replacement after the verified session was opened")
        fields[0].value = "false"
        fields[1].value = "true"
        app.action_run_record()
        await pilot.pause()

        if shell == "classic":
            message = str(app.query_one("#artifact-inference", Static).render())
            trace = app.query_one("#feature-trace", DataTable)
        else:
            panel = app.query_one("#artifact-panel")
            message = str(panel.query_one("#record-result", Static).render())
            trace = panel.query_one("#feature-trace", DataTable)
        assert "PREDICTION 1" in message
        assert trace.row_count == 2


@SHELLS
@pytest.mark.asyncio
async def test_frontend_contract_repair_search_shows_counterexample_and_exports(
    shell: str, tmp_path: Path
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

    app = make_app(shell, tmp_path)
    app.search._runner = run_search

    async with app.run_test(size=(120, 40)) as pilot:
        app.query_one("#search-kind", Select).value = SearchKind.REPAIR.value
        await pilot.pause()
        app.action_search()
        assert await wait_for_job(pilot, app, "search_state") is JobState.SUCCEEDED

        counterexamples = app.query_one(
            "#counterexamples" if shell == "classic" else "#search-counterexamples",
            DataTable,
        )
        assert counterexamples.row_count == 1

        destination = tmp_path / f"{shell}-repair.ptm"
        app.query_one(
            "#search-export-path", Input
        ).value = str(destination)
        if shell == "classic":
            app.action_export_search()
        else:
            app.action_export_search()
        assert destination.is_file()


@SHELLS
@pytest.mark.asyncio
async def test_frontend_contract_kind_change_discards_active_search_result(
    shell: str, tmp_path: Path
) -> None:
    app = make_app(shell, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        app.action_show_search()
        app.session.search_state = JobState.RUNNING
        old_generation = app._search_generation
        old_cancel = app._search_cancel

        app.query_one("#search-kind", Select).value = SearchKind.REPAIR.value
        await pilot.pause()

        assert old_cancel.is_set()
        assert app._search_generation > old_generation
        assert app.session.search_state is JobState.IDLE

        stale_result = BoundedSearchResult(SearchKind.THRESHOLD, {}, 0.01)
        app._show_search_result(stale_result, old_generation)
        assert app.session.search_result is None
        assert app.query_one(
            "#search-export-button" if shell == "classic" else "#search-export"
        ).disabled
