import pytest

textual = pytest.importorskip("textual")

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
