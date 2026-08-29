from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button, DataTable

from prolog_tsetlin.services.environment import Capability
from prolog_tsetlin.services.training import (
    MulticlassTrainingRun,
    TrainingWorkload,
)
from prolog_tsetlin.tui.models import JobState
from prolog_tsetlin.tui.single_pane.app import PTMWorkbenchApp
from prolog_tsetlin.tui.single_pane.panels.training_config import (
    TrainingConfigPanel,
)


@pytest.mark.asyncio
async def test_workbench_projects_mnist_without_binary_snapshot_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(PTMWorkbenchApp, "_tick_uptime", lambda self: None)
    monkeypatch.setattr(
        "prolog_tsetlin.tui.single_pane.app.inspect_environment",
        lambda workspace: (
            Capability("Scalar oracle", "READY", "reference backend"),
            Capability("GNU Prolog", "READY", "test fixture"),
        ),
    )
    app = PTMWorkbenchApp(workspace=tmp_path, demo="mnist")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        request = app.query_one(
            "#config-panel", TrainingConfigPanel
        ).get_request()
        assert request.workload is TrainingWorkload.MNIST
        assert request.number_of_clauses == 100
        assert request.boost_true_positive_feedback

        matrix = tuple(
            tuple(1000 if row == column else 0 for column in range(10))
            for row in range(10)
        )
        result = MulticlassTrainingRun(
            request=request,
            class_labels=tuple(range(10)),
            validation_rows=10_000,
            confusion_matrix=matrix,
            accuracy=1.0,
            training_seconds=2.5,
            backend="test-native",
            material_manifest=str(tmp_path / "manifest.json"),
        )
        app.training.begin(request)
        app.session.job_state = JobState.RUNNING
        app.on_training_complete(result)
        await pilot.pause()

        assert app._snapshot is None
        assert app._diagnostics is None
        assert app.training.current_inspection() is None
        assert app.query_one("#predictions-table", DataTable).row_count == 10
        assert app.query_one("#artifact-export", Button).disabled
        assert "ARTIFACT EXPORT BLOCKED" in str(
            app.query_one("#footer-bar").render()
        )
