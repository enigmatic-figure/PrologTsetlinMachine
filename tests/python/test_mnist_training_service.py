from __future__ import annotations

import json
from pathlib import Path
import sys
from threading import Event

import pytest

from prolog_tsetlin.services.mnist_training import train_mnist_native
from prolog_tsetlin.services.training import (
    TrainingCancelled,
    TrainingRequest,
    TrainingWorkload,
)


def _material(tmp_path: Path) -> Path:
    output = tmp_path / "material"
    output.mkdir()
    manifest = {
        "schema": "ptm.mnist-bits.v1",
        "threshold": 0.3,
        "threshold_rule": "pixel > threshold",
        "source_digest": "sha256:test",
        "splits": {
            "train": {"path": "train.ptmb", "rows": 50_000},
            "validation": {"path": "validation.ptmb", "rows": 10_000},
            "test": {"path": "test.ptmb", "rows": 10_000},
        },
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _runner(
    tmp_path: Path, *, valid: bool = True, delay_seconds: float = 0.0
) -> Path:
    path = tmp_path / "runner.py"
    schema = "ptm.mnist-ovr-epoch.v1" if valid else "wrong.schema"
    path.write_text(
        """
import json
import sys
import time

epochs = int(sys.argv[7])
matrix = [[0 for _ in range(10)] for _ in range(10)]
for label in range(10):
    matrix[label][label] = 1000
for epoch in range(1, epochs + 1):
    print(json.dumps({
        "schema": SCHEMA,
        "epoch": epoch,
        "clauses_per_class": int(sys.argv[3]),
        "threshold": int(sys.argv[6]),
        "specificity": float(sys.argv[5]),
        "training_policy": sys.argv[9],
        "epoch_shuffle": True,
        "parallel_class_training": True,
        "parallel_validation": True,
        "boost_true_positive_feedback": sys.argv[10] == "boost",
        "training_seconds": 0.25,
        "cumulative_training_seconds": 0.25 * epoch,
        "validation_accuracy": 1.0,
        "confusion_matrix": matrix,
    }), flush=True)
    if epoch < epochs:
        time.sleep(DELAY_SECONDS)
""".replace("SCHEMA", repr(schema)).replace(
            "DELAY_SECONDS", repr(delay_seconds)
        ),
        encoding="utf-8",
    )
    return path


def _request() -> TrainingRequest:
    return TrainingRequest(
        workload=TrainingWorkload.MNIST,
        number_of_clauses=4,
        states_per_action=8,
        specificity=3.0,
        threshold=5,
        epochs=2,
        seed=11,
        boost_true_positive_feedback=True,
    )


def test_native_mnist_service_validates_progress_and_class_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _material(tmp_path)
    monkeypatch.setattr(
        "prolog_tsetlin.services.mnist_training.ensure_mnist_material",
        lambda *args, **kwargs: manifest,
    )
    progress = []

    run = train_mnist_native(
        _request(),
        workspace=tmp_path,
        progress=progress.append,
        runner_command=(sys.executable, str(_runner(tmp_path))),
    )

    assert run.accuracy == 1.0
    assert run.validation_rows == 10_000
    assert run.training_seconds == 0.5
    assert len(run.confusion_matrix) == 10
    assert [item.epoch for item in progress] == [1, 2]
    assert progress[-1].cumulative_training_seconds == 0.5


def test_native_mnist_service_rejects_untrusted_runner_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _material(tmp_path)
    monkeypatch.setattr(
        "prolog_tsetlin.services.mnist_training.ensure_mnist_material",
        lambda *args, **kwargs: manifest,
    )

    with pytest.raises(RuntimeError, match="violates the request"):
        train_mnist_native(
            _request(),
            workspace=tmp_path,
            runner_command=(
                sys.executable,
                str(_runner(tmp_path, valid=False)),
            ),
        )


def test_native_mnist_service_honors_prelaunch_cancellation(tmp_path: Path) -> None:
    cancel = Event()
    cancel.set()

    with pytest.raises(TrainingCancelled, match="before launch"):
        train_mnist_native(_request(), workspace=tmp_path, cancel=cancel)


def test_native_mnist_service_terminates_a_cancelled_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _material(tmp_path)
    monkeypatch.setattr(
        "prolog_tsetlin.services.mnist_training.ensure_mnist_material",
        lambda *args, **kwargs: manifest,
    )
    cancel = Event()

    def cancel_after_first_epoch(progress) -> None:
        if progress.epoch == 1:
            cancel.set()

    with pytest.raises(TrainingCancelled, match="cancelled"):
        train_mnist_native(
            _request(),
            workspace=tmp_path,
            progress=cancel_after_first_epoch,
            cancel=cancel,
            runner_command=(
                sys.executable,
                str(_runner(tmp_path, delay_seconds=5.0)),
            ),
        )
