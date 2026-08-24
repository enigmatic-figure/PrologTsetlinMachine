from threading import Event
from unittest import TestCase

from prolog_tsetlin.services.training import (
    TrainingCancelled,
    TrainingDiagnosticSampling,
    TrainingRequest,
    train_xor,
)


ASSERTIONS = TestCase()


def test_xor_training_is_deterministic_and_reports_progress() -> None:
    request = TrainingRequest(epochs=20)
    progress = []
    first = train_xor(request, progress=progress.append)
    second = train_xor(request)

    assert first.snapshot.states == second.snapshot.states
    assert first.predictions == second.predictions
    assert [item.epoch for item in progress] == list(range(1, 21))


def test_training_request_rejects_invalid_epochs() -> None:
    with ASSERTIONS.assertRaisesRegex(ValueError, "epochs must be positive"):
        train_xor(TrainingRequest(epochs=0))


def test_training_request_rejects_non_finite_specificity() -> None:
    for specificity in (float("nan"), float("inf"), float("-inf")):
        with ASSERTIONS.assertRaisesRegex(
            ValueError, "specificity must be finite and greater than one"
        ):
            TrainingRequest(specificity=specificity).validate()


def test_training_honors_pre_set_cancellation() -> None:
    cancel = Event()
    cancel.set()
    with ASSERTIONS.assertRaisesRegex(TrainingCancelled, "before epoch 1"):
        train_xor(TrainingRequest(), cancel=cancel)


def test_training_emits_only_explicitly_sampled_immutable_diagnostics() -> None:
    request = TrainingRequest(epochs=20)
    samples = []
    sampling = TrainingDiagnosticSampling(every_epochs=6)

    sampled_run = train_xor(
        request,
        diagnostic=samples.append,
        diagnostic_sampling=sampling,
    )
    ordinary_run = train_xor(request)

    assert [sample.epoch for sample in samples] == [1, 6, 12, 18, 20]
    assert all(sample.request == request for sample in samples)
    assert samples[-1].predictions == sampled_run.predictions
    assert samples[-1].accuracy == sampled_run.accuracy
    assert samples[-1].snapshot == sampled_run.snapshot
    assert sampled_run == ordinary_run
    assert len({id(sample.snapshot) for sample in samples}) == len(samples)


def test_training_diagnostics_require_callback_and_policy_together() -> None:
    request = TrainingRequest(epochs=1)
    sampling = TrainingDiagnosticSampling(every_epochs=1)

    with ASSERTIONS.assertRaisesRegex(ValueError, "supplied together"):
        train_xor(request, diagnostic=lambda sample: None)
    with ASSERTIONS.assertRaisesRegex(ValueError, "supplied together"):
        train_xor(request, diagnostic_sampling=sampling)


def test_bounded_diagnostic_sampling_respects_sample_budget() -> None:
    for epochs in (1, 2, 24, 150, 10_000):
        sampling = TrainingDiagnosticSampling.bounded(
            epochs, maximum_samples=25
        )
        selected = [
            epoch
            for epoch in range(1, epochs + 1)
            if sampling.includes(epoch, epochs)
        ]

        assert len(selected) <= 25
        assert selected[0] == 1
        assert selected[-1] == epochs


def test_training_diagnostic_callback_failures_are_visible() -> None:
    def fail(_sample) -> None:
        raise RuntimeError("diagnostic consumer failed")

    with ASSERTIONS.assertRaisesRegex(RuntimeError, "consumer failed"):
        train_xor(
            TrainingRequest(epochs=1),
            diagnostic=fail,
            diagnostic_sampling=TrainingDiagnosticSampling(every_epochs=1),
        )
