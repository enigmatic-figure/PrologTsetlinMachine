"""Visual regression for single-pane at 80x24 and 120x40."""

from __future__ import annotations

import pytest

from prolog_tsetlin.services.environment import Capability
from prolog_tsetlin.tui.single_pane.app import PTMWorkbenchApp


@pytest.fixture(autouse=True)
def deterministic_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep snapshots independent of host tools and operating system."""

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(PTMWorkbenchApp, "_tick_uptime", lambda self: None)
    monkeypatch.setattr(
        "prolog_tsetlin.tui.single_pane.app.inspect_environment",
        lambda workspace: (
            Capability("Scalar oracle", "READY", "reference backend"),
            Capability("GNU Prolog", "READY", "snapshot fixture"),
        ),
    )


def test_single_pane_snapshot_compact(snap_compare):
    app = PTMWorkbenchApp()
    assert snap_compare(app, terminal_size=(80, 24))


def test_single_pane_snapshot_wide(snap_compare):
    app = PTMWorkbenchApp()
    assert snap_compare(app, terminal_size=(120, 40))
