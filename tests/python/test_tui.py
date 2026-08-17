import pytest

textual = pytest.importorskip("textual")
from textual.widgets import Input

from prolog_tsetlin.tui.app import PTMApp
from prolog_tsetlin.tui.models import JobState


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
    app.session.request = app.session.request.__class__(epochs=100_000)
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
        assert "STALE CONFIGURATION" in str(app.query_one("#job").render())
