from threading import Event

import pytest

from prolog_tsetlin.services.training import (
    TrainingCancelled,
    TrainingRequest,
    train_xor,
)


def test_xor_training_is_deterministic_and_reports_progress() -> None:
    request = TrainingRequest(epochs=20)
    progress = []
    first = train_xor(request, progress=progress.append)
    second = train_xor(request)

    assert first.snapshot.states == second.snapshot.states
    assert first.predictions == second.predictions
    assert [item.epoch for item in progress] == list(range(1, 21))


def test_training_request_rejects_invalid_epochs() -> None:
    with pytest.raises(ValueError, match="epochs must be positive"):
        train_xor(TrainingRequest(epochs=0))


def test_training_honors_pre_set_cancellation() -> None:
    cancel = Event()
    cancel.set()
    with pytest.raises(TrainingCancelled, match="before epoch 1"):
        train_xor(TrainingRequest(), cancel=cancel)
