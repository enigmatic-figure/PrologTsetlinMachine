from unittest import SkipTest
from pathlib import Path
import os
import shutil

try:
    import pytest
except ImportError as error:  # dependency-free unittest verification path
    raise SkipTest("Textual tests require the optional pytest stack") from error

textual = pytest.importorskip("textual")
from textual.widgets import Button, Input, Select, TextArea

from prolog_tsetlin.model_artifact import load_model_artifact
from prolog_tsetlin.tui.app import PTMApp
from prolog_tsetlin.tui.models import JobState
from prolog_tsetlin.services.search import SearchKind
from prolog_tsetlin.services.training import TrainingRequest, train_xor


GPROLOG = Path(
    os.environ.get("PTM_GPROLOG")
    or shutil.which("gprolog")
    or r"C:\GNU-Prolog\bin\gprolog.exe"
)


@pytest.mark.asyncio
async def test_tui_trains_xor_from_keyboard() -> None:
    app = PTMApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("t")
        await pilot.pause(1)
        assert "SUCCEEDED" in str(app.query_one("#job").render())
        assert app.query_one("#predictions").row_count == 4
        assert app.query_one("#clauses").row_count == 20

        await pilot.press("c")
        assert app.query_one("#clauses").display
        assert not app.query_one("#predictions").display


@pytest.mark.asyncio
async def test_tui_can_cancel_training() -> None:
    app = PTMApp()
    app.session.configured_request = TrainingRequest(epochs=100_000)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("t")
        await pilot.press("x")
        for _ in range(20):
            await pilot.pause(0.05)
            if app.session.job_state is JobState.CANCELLED:
                break
        assert app.session.job_state is JobState.CANCELLED
        assert "CANCELLED" in str(app.query_one("#job").render())


@pytest.mark.asyncio
async def test_tui_validates_training_configuration() -> None:
    app = PTMApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.query_one("#config-clauses", Input).value = "0"
        await pilot.press("t")
        await pilot.pause()

        assert app.session.job_state is JobState.IDLE
        assert "number_of_clauses must be positive" in str(
            app.query_one("#validation").render()
        )


@pytest.mark.asyncio
async def test_tui_marks_a_completed_run_stale_when_configuration_changes() -> None:
    app = PTMApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("t")
        await pilot.pause(1)
        app.query_one("#config-seed", Input).value = "8"
        await pilot.pause()

        assert app.session.configuration_dirty
        assert app.query_one("#export-button", Button).disabled


@pytest.mark.asyncio
async def test_tui_mid_training_edit_is_stale_when_run_completes() -> None:
    app = PTMApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.query_one("#config-epochs", Input).value = "1"
        await pilot.pause()
        trained_request = app._request_from_form()
        app.training.begin(trained_request)

        app.query_one("#config-seed", Input).value = str(trained_request.seed + 1)
        await pilot.pause()
        app._show_result(train_xor(trained_request))

        assert app.session.configuration_dirty
        assert "STALE CONFIGURATION" in str(app.query_one("#job").render())
        assert app.query_one("#export-button", Button).disabled
        assert "STALE CONFIGURATION" in str(app.query_one("#job").render())


@pytest.mark.asyncio
async def test_tui_exports_completed_run_without_overwriting(tmp_path) -> None:
    app = PTMApp(workspace=tmp_path)
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press("t")
        await pilot.pause(1)
        await pilot.press("4")
        await pilot.press("e")
        await pilot.pause()

        assert app.session.artifact is not None
        destination = tmp_path / "out" / "artifacts" / "xor-explorer.ptm"
        artifact = load_model_artifact(destination)
        assert artifact.artifact_id == app.session.artifact.artifact_id
        assert artifact.verify_conformance()
        assert artifact.preprocessing is not None

        original = destination.read_bytes()
        await pilot.press("e")
        await pilot.pause()
        assert destination.read_bytes() == original
        assert "REFUSED" in str(app.query_one("#artifact-status").render())


@pytest.mark.asyncio
async def test_tui_opens_artifact_and_runs_schema_driven_record(tmp_path) -> None:
    source = Path(__file__).resolve().parents[2] / "tests/data/raw_xor_packed_tm_v1.hex"
    artifact = tmp_path / "raw-xor.ptm"
    artifact.write_bytes(bytes.fromhex(source.read_text(encoding="ascii")))

    app = PTMApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("4")
        app.query_one("#artifact-open-path", Input).value = str(artifact)
        await pilot.click("#load-artifact-button")
        await pilot.pause()

        assert "OPENED / VERIFIED" in str(
            app.query_one("#artifact-open-status").render()
        )
        fields = list(app.query("#record-fields Input"))
        assert len(fields) == 2
        fields[0].value = "false"
        fields[1].value = "true"
        app.query_one("#run-record-button").focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert "PREDICTION 1" in str(app.query_one("#artifact-inference").render())
        assert app.query_one("#feature-trace").row_count == 2


@pytest.mark.asyncio
async def test_tui_navigation_and_help_remain_available_at_80x24() -> None:
    app = PTMApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("2")
        assert app.query_one("#workspace").current == "view-train"
        await pilot.press("3")
        assert app.query_one("#workspace").current == "view-clauses"
        await pilot.press("5")
        app.query_one("#search-kind", Select).value = SearchKind.REPAIR.value
        await pilot.pause()
        search_hint = app.query_one("#search-result", TextArea).text
        assert "press F5" in search_hint
        assert "press s" not in search_hint
        await pilot.press("question_mark")
        await pilot.pause()
        assert app.screen.query_one("#help-dialog") is not None
        assert "BOUNDED SYMBOLIC SEARCH" in str(
            app.screen.query_one("#help-title").render()
        )
        help_copy = str(app.screen.query_one("#help-copy").render())
        assert "F5" in help_copy
        assert "GNU Prolog" in help_copy
        await pilot.press("escape")


@pytest.mark.asyncio
@pytest.mark.skipif(not GPROLOG.is_file(), reason="GNU Prolog is not installed")
async def test_tui_runs_repair_shows_counterexamples_and_exports(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PTM_GPROLOG", str(GPROLOG))
    app = PTMApp(workspace=tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("5")
        app.query_one("#search-kind", Select).value = SearchKind.REPAIR.value
        await pilot.pause()
        assert '"kind": "repair"' in app.query_one("#search-json", TextArea).text

        app.query_one("#search-button").focus()
        await pilot.press("enter")
        for _ in range(400):
            await pilot.pause(0.05)
            if app.session.search_state in (
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.CANCELLED,
            ):
                break

        assert app.session.search_state is JobState.SUCCEEDED
        assert app.query_one("#counterexamples").row_count == 4
        assert '"mismatch_count": 0' in app.query_one("#search-result", TextArea).text
        assert not app.query_one("#search-export-button").disabled

        output = tmp_path / "repair.ptm"
        app.query_one("#search-export-path", Input).value = str(output)
        app.action_export_search()
        artifact = load_model_artifact(output)
        assert artifact.verify_conformance()
        assert "EXPORTED" in str(app.query_one("#search-status").render())
