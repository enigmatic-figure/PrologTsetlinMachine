from __future__ import annotations

from pathlib import Path

import pytest

from prolog_tsetlin import cli
from prolog_tsetlin import tui


def test_tui_launcher_selects_each_shell(monkeypatch, tmp_path: Path) -> None:
    from prolog_tsetlin.tui import app as classic
    from prolog_tsetlin.tui.single_pane import app as single_pane

    launched: list[tuple[str, Path | None, str]] = []

    class ClassicStub:
        def __init__(self, *, workspace: Path | None, demo: str) -> None:
            launched.append(("classic", workspace, demo))

        def run(self) -> None:
            return None

    class SinglePaneStub:
        def __init__(self, *, workspace: Path | None, demo: str) -> None:
            launched.append(("single_pane", workspace, demo))

        def run(self) -> None:
            return None

    monkeypatch.setattr(classic, "PTMApp", ClassicStub)
    monkeypatch.setattr(single_pane, "SinglePaneApp", SinglePaneStub)

    tui.run(workspace=tmp_path, demo="xor", style="classic")
    tui.run(workspace=tmp_path, demo="xor", style="single_pane")

    assert launched == [
        ("classic", tmp_path, "xor"),
        ("single_pane", tmp_path, "xor"),
    ]

    with pytest.raises(ValueError, match="unsupported TUI style"):
        tui.run(workspace=tmp_path, demo="xor", style="unknown")


def test_cli_forwards_single_pane_style(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def run_stub(**options: object) -> None:
        captured.update(options)

    monkeypatch.setattr(tui, "run", run_stub)

    assert (
        cli.main(
            [
                "tui",
                "--workspace",
                str(tmp_path),
                "--style",
                "single_pane",
            ]
        )
        == 0
    )
    assert captured == {
        "workspace": tmp_path,
        "demo": "xor",
        "style": "single_pane",
    }
