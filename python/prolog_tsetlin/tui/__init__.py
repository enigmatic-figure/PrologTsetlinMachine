"""Optional Textual workbench launcher."""

from __future__ import annotations

from pathlib import Path


def run(*, workspace: Path | None = None, demo: str = "xor") -> None:
    from .app import PTMApp

    PTMApp(workspace=workspace, demo=demo).run()


def main() -> None:
    run()
