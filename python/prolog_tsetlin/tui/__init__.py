"""Optional Textual workbench launcher."""

from __future__ import annotations

from pathlib import Path


def run(
    *, workspace: Path | None = None, demo: str = "xor", style: str = "workbench"
) -> None:
    if style in ("workbench", "single_pane"):
        from .single_pane.app import PTMWorkbenchApp

        PTMWorkbenchApp(workspace=workspace, demo=demo).run()
    elif style == "classic":
        from .app import PTMApp

        PTMApp(workspace=workspace, demo=demo).run()
    else:
        raise ValueError(f"unsupported TUI style: {style}")


def main() -> None:
    run()
