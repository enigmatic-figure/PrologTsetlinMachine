import pytest

textual = pytest.importorskip("textual")

from prolog_tsetlin.tui.app import PTMApp


@pytest.mark.asyncio
async def test_tui_trains_xor_from_keyboard() -> None:
    app = PTMApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("t")
        await pilot.pause(1)
        assert "SUCCEEDED" in str(app.query_one("#job").render())
        assert app.query_one("#predictions").row_count == 4
