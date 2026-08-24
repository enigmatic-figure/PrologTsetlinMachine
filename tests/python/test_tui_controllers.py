"""UI-neutral session contracts shared by Textual presentation shells."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from prolog_tsetlin.model_artifact import export_packed_tm
from prolog_tsetlin.preprocessing import PreprocessingContract
from prolog_tsetlin.reference import ScalarBinaryTsetlinMachine
from prolog_tsetlin.representation import FeatureSchema, FieldKind, LiteralCatalog
from prolog_tsetlin.services.artifacts import ArtifactExportRequest
from prolog_tsetlin.services.search import (
    BoundedSearchResult,
    SearchKind,
    demo_search_document,
)
from prolog_tsetlin.services.training import (
    TrainingDiagnosticSampling,
    TrainingRequest,
    train_xor,
)
from prolog_tsetlin.tui.controllers import (
    ArtifactSessionController,
    SearchSessionController,
    SessionContractError,
    TrainingSessionController,
)
from prolog_tsetlin.tui.models import JobState, SessionState


def _raw_xor_artifact(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[2] / "tests/data/raw_xor_packed_tm_v1.hex"
    destination = tmp_path / "raw-xor.ptm"
    destination.write_bytes(bytes.fromhex(source.read_text(encoding="ascii")))
    return destination


def _replace_with_compatible_zero_artifact(path: Path):
    machine = ScalarBinaryTsetlinMachine(
        4, 2, states_per_action=3, specificity=3.0, threshold=8, seed=41
    )
    for clause in range(4):
        for literal in range(4):
            machine.set_state(clause, literal, 3)
    schema = FeatureSchema.from_fields(
        left=FieldKind.BOOLEAN, right=FieldKind.BOOLEAN
    )
    catalog = LiteralCatalog(schema)
    catalog.category_eq("left", True)
    catalog.category_eq("right", True)
    preprocessing = PreprocessingContract.from_catalog(catalog)
    records = (
        {"left": False, "right": False},
        {"left": False, "right": True},
        {"left": True, "right": False},
        {"left": True, "right": True},
    )
    return export_packed_tm(
        machine.snapshot(),
        name="Compatible all-zero replacement",
        path=path,
        preprocessing=preprocessing,
        validation_records=records,
        feature_names=("left", "right"),
        feature_catalog_version="replacement-v1",
    )


def test_training_configuration_staleness_is_derived_and_reversible() -> None:
    request = TrainingRequest(epochs=1)
    session = SessionState(
        configured_request=request,
        last_completed_run=train_xor(request),
    )
    controller = TrainingSessionController(session)

    changed = TrainingRequest(epochs=1, seed=request.seed + 1)
    assert controller.synchronize_configuration(changed)
    assert session.configuration_dirty

    assert not controller.synchronize_configuration(request)
    assert not session.configuration_dirty

    assert controller.synchronize_configuration(None)
    assert session.configuration_dirty


def test_training_completion_preserves_mid_run_form_staleness() -> None:
    trained_request = TrainingRequest(epochs=1)
    visible_request = TrainingRequest(epochs=1, seed=trained_request.seed + 1)
    session = SessionState(
        configured_request=visible_request,
        active_request=trained_request,
        job_state=JobState.RUNNING,
    )
    controller = TrainingSessionController(session)

    controller.complete(
        train_xor(trained_request), current_request=visible_request
    )

    assert session.job_state is JobState.SUCCEEDED
    assert session.last_completed_run is not None
    assert session.configured_request == visible_request
    assert session.active_request is None
    assert session.configuration_dirty


def test_training_begin_retains_last_completed_snapshot_with_provenance() -> None:
    completed_request = TrainingRequest(epochs=1)
    active_request = TrainingRequest(
        epochs=1, seed=completed_request.seed + 1
    )
    completed_run = train_xor(completed_request)
    session = SessionState(
        configured_request=completed_request,
        last_completed_run=completed_run,
        job_state=JobState.SUCCEEDED,
    )
    controller = TrainingSessionController(session)

    controller.begin(active_request)

    assert session.configured_request == active_request
    assert session.active_request == active_request
    assert session.last_completed_run is completed_run
    assert session.configuration_dirty
    assert controller.retained_run_is_historical
    assert controller.active_request_matches_configuration

    controller.synchronize_configuration(completed_request)
    assert not session.configuration_dirty
    assert not controller.active_request_matches_configuration

    controller.cancelled(current_request=completed_request)
    assert session.active_request is None
    assert session.last_completed_run is completed_run


def test_training_controller_owns_sampled_diagnostic_history() -> None:
    request = TrainingRequest(epochs=5)
    sampling = TrainingDiagnosticSampling(every_epochs=2)
    session = SessionState()
    controller = TrainingSessionController(session)
    controller.begin(request, diagnostic_sampling=sampling)

    run = controller.run(diagnostic=controller.record_diagnostic_sample)

    assert [
        item.sample.epoch for item in session.active_diagnostics
    ] == [1, 2, 4, 5]
    assert session.last_completed_diagnostics == ()
    assert session.active_diagnostics[0].delta_from_previous is None
    assert session.active_diagnostics[-1].delta_from_previous is not None

    controller.complete(run)

    assert session.active_diagnostics == []
    assert session.active_diagnostic_sampling is None
    assert [
        item.sample.epoch for item in session.last_completed_diagnostics
    ] == [1, 2, 4, 5]
    assert session.last_completed_diagnostics[-1].sample.snapshot == run.snapshot


def test_training_controller_selects_only_retained_completed_samples() -> None:
    request = TrainingRequest(epochs=5)
    sampling = TrainingDiagnosticSampling(every_epochs=2)
    session = SessionState()
    controller = TrainingSessionController(session)
    controller.begin(request, diagnostic_sampling=sampling)
    run = controller.run(diagnostic=controller.record_diagnostic_sample)
    controller.complete(run)

    current = controller.current_inspection()
    assert current is not None
    assert current.epoch == request.epochs
    assert not current.historical
    assert current.snapshot == run.snapshot

    historical = controller.inspect_completed_epoch(2)
    assert historical.historical
    assert historical.epoch == 2
    assert historical.sampled is session.last_completed_diagnostics[1]
    assert historical.diagnostics is historical.sampled.diagnostics
    assert session.inspected_sample_epoch == 2
    assert controller.require_exportable(request) is run

    with pytest.raises(SessionContractError, match="not a retained"):
        controller.inspect_completed_epoch(3)
    assert session.inspected_sample_epoch == 2

    final = controller.inspect_completed_epoch()
    assert final.epoch == request.epochs
    assert not final.historical
    assert session.inspected_sample_epoch is None


def test_historical_inspection_does_not_change_staleness_or_export_source() -> None:
    request = TrainingRequest(epochs=3)
    sampling = TrainingDiagnosticSampling(every_epochs=1)
    session = SessionState()
    controller = TrainingSessionController(session)
    controller.begin(request, diagnostic_sampling=sampling)
    run = controller.run(diagnostic=controller.record_diagnostic_sample)
    controller.complete(run)
    controller.inspect_completed_epoch(1)

    assert controller.require_exportable(request) is run
    assert not session.configuration_dirty
    changed = replace(request, seed=request.seed + 1)
    assert controller.synchronize_configuration(changed)
    assert session.inspected_sample_epoch == 1
    with pytest.raises(SessionContractError, match="stale"):
        controller.require_exportable(changed)
    assert controller.synchronize_configuration(request) is False
    assert session.inspected_sample_epoch == 1
    assert controller.require_exportable(request) is run


def test_training_controller_rejects_selection_from_replaced_completed_run() -> None:
    request = TrainingRequest(epochs=3)
    sampling = TrainingDiagnosticSampling(every_epochs=1)
    session = SessionState()
    controller = TrainingSessionController(session)
    controller.begin(request, diagnostic_sampling=sampling)
    first_run = controller.run(diagnostic=controller.record_diagnostic_sample)
    controller.complete(first_run)

    controller.begin(request, diagnostic_sampling=sampling)
    replacement = controller.run(diagnostic=controller.record_diagnostic_sample)
    controller.complete(replacement)

    assert replacement == first_run
    assert replacement is not first_run
    with pytest.raises(SessionContractError, match="run changed"):
        controller.inspect_completed_epoch(1, expected_run=first_run)
    assert session.inspected_sample_epoch is None
    assert controller.current_inspection().run is replacement


def test_training_inspection_resets_to_completed_final_on_terminal_transitions() -> None:
    request = TrainingRequest(epochs=3)
    sampling = TrainingDiagnosticSampling(every_epochs=1)
    session = SessionState()
    controller = TrainingSessionController(session)
    controller.begin(request, diagnostic_sampling=sampling)
    run = controller.run(diagnostic=controller.record_diagnostic_sample)
    controller.complete(run)
    controller.inspect_completed_epoch(1)

    retry = replace(request, seed=request.seed + 1)
    controller.begin(retry, diagnostic_sampling=sampling)
    assert session.inspected_sample_epoch is None
    assert controller.current_inspection().epoch == request.epochs

    controller.cancelled()
    assert session.inspected_sample_epoch is None
    assert controller.current_inspection().epoch == request.epochs


def test_training_controller_rejects_missing_or_out_of_order_samples() -> None:
    request = TrainingRequest(epochs=3)
    sampling = TrainingDiagnosticSampling(every_epochs=2)
    samples = []
    run = train_xor(
        request,
        diagnostic=samples.append,
        diagnostic_sampling=sampling,
    )
    session = SessionState()
    controller = TrainingSessionController(session)
    controller.begin(request, diagnostic_sampling=sampling)

    with pytest.raises(SessionContractError, match="callback"):
        controller.run()

    controller.record_diagnostic_sample(samples[0])
    with pytest.raises(SessionContractError, match="expected 2"):
        controller.record_diagnostic_sample(replace(samples[1], epoch=3))
    with pytest.raises(SessionContractError, match="missing requested"):
        controller.complete(run)


def test_training_controller_retains_completed_history_during_retraining() -> None:
    request = TrainingRequest(epochs=2)
    sampling = TrainingDiagnosticSampling(every_epochs=1)
    session = SessionState()
    controller = TrainingSessionController(session)
    controller.begin(request, diagnostic_sampling=sampling)
    controller.record_progress(1, 0.25)
    controller.record_progress(2, 0.5)
    run = controller.run(diagnostic=controller.record_diagnostic_sample)
    controller.complete(run)
    retained = session.last_completed_diagnostics
    retained_accuracy = session.last_completed_accuracy_history

    controller.begin(
        replace(request, seed=request.seed + 1),
        diagnostic_sampling=sampling,
    )

    assert session.last_completed_diagnostics is retained
    assert session.active_diagnostics == []

    retraining_samples = []
    retraining_request = session.active_request
    assert retraining_request is not None
    train_xor(
        retraining_request,
        diagnostic=retraining_samples.append,
        diagnostic_sampling=sampling,
    )
    controller.record_progress(1, 0.75)
    controller.record_diagnostic_sample(retraining_samples[0])
    assert controller.request_cancel()
    controller.cancelled()

    assert session.active_diagnostics == []
    assert session.active_diagnostic_sampling is None
    assert session.last_completed_diagnostics is retained
    assert session.last_completed_accuracy_history is retained_accuracy
    assert session.accuracy_history == [0.75]
    assert session.progress_epoch == 1
    assert session.progress_accuracy == 0.75


def test_training_controller_failure_retains_completed_accuracy() -> None:
    completed_request = TrainingRequest(epochs=2)
    completed_run = train_xor(completed_request)
    retry_request = replace(
        completed_request, seed=completed_request.seed + 1
    )
    session = SessionState(
        configured_request=retry_request,
        active_request=retry_request,
        job_state=JobState.RUNNING,
        last_completed_run=completed_run,
        accuracy_history=[0.25],
        last_completed_accuracy_history=(0.5, completed_run.accuracy),
    )
    controller = TrainingSessionController(session)

    controller.failed("worker failed")

    assert session.last_completed_accuracy_history == (
        0.5,
        completed_run.accuracy,
    )
    assert session.accuracy_history == [0.25]


def test_training_rejects_result_from_a_different_request() -> None:
    active_request = TrainingRequest(epochs=1)
    other_request = TrainingRequest(epochs=1, seed=active_request.seed + 1)
    session = SessionState()
    controller = TrainingSessionController(session)
    controller.begin(active_request)

    with pytest.raises(SessionContractError, match="active training request"):
        controller.complete(train_xor(other_request))

    assert session.active_request == active_request
    assert session.last_completed_run is None
    assert session.job_state is JobState.QUEUED


def test_training_cancel_uses_the_visible_configuration() -> None:
    completed_request = TrainingRequest(epochs=1)
    retrain_request = TrainingRequest(epochs=1, seed=completed_request.seed + 1)
    session = SessionState(
        configured_request=completed_request,
        last_completed_run=train_xor(completed_request),
        job_state=JobState.SUCCEEDED,
    )
    controller = TrainingSessionController(session)
    controller.begin(retrain_request)

    controller.cancelled(current_request=completed_request)
    assert not session.configuration_dirty

    controller.begin(retrain_request)
    controller.cancelled(current_request=None)
    assert session.configuration_dirty


def test_training_export_enforces_current_valid_inactive_request(
    tmp_path: Path,
) -> None:
    request = TrainingRequest(epochs=1)
    run = train_xor(request)
    session = SessionState(
        configured_request=request,
        last_completed_run=run,
        job_state=JobState.SUCCEEDED,
    )
    controller = TrainingSessionController(session)
    destination = tmp_path / "current.ptm"
    export_request = ArtifactExportRequest(path=destination)

    session.job_state = JobState.RUNNING
    with pytest.raises(SessionContractError, match="training is active"):
        controller.export(request, export_request)
    assert not destination.exists()

    session.job_state = JobState.SUCCEEDED
    changed = TrainingRequest(epochs=1, seed=request.seed + 1)
    with pytest.raises(SessionContractError, match="stale"):
        controller.export(changed, export_request)
    assert session.configuration_dirty
    assert not destination.exists()

    summary = controller.export(request, export_request)
    assert summary.path == destination.resolve()
    assert session.artifact == summary
    assert not session.configuration_dirty


def test_artifact_controller_parses_raw_strings_and_normalizes_result(
    tmp_path: Path,
) -> None:
    session = SessionState()
    controller = ArtifactSessionController(session)
    artifact = _raw_xor_artifact(tmp_path)

    loaded = controller.load(artifact)
    assert [field.name for field in loaded.fields] == ["left", "right"]

    result = controller.run_record(
        {"left": "false", "right": "true"}, displayed_path=artifact
    )
    assert result.record == {"left": False, "right": True}
    assert result.prediction == 1
    assert len(result.feature_trace) == 2
    assert result.features == (0, 1)


def test_artifact_controller_rejects_changed_visible_path(tmp_path: Path) -> None:
    session = SessionState()
    controller = ArtifactSessionController(session)
    artifact = _raw_xor_artifact(tmp_path)
    controller.load(artifact)

    with pytest.raises(SessionContractError, match="path changed"):
        controller.run_record(
            {"left": "false", "right": "true"},
            displayed_path=tmp_path / "different.ptm",
        )


def test_artifact_controller_executes_pinned_bytes_after_same_path_replacement(
    tmp_path: Path,
) -> None:
    session = SessionState()
    controller = ArtifactSessionController(session)
    path = _raw_xor_artifact(tmp_path)
    loaded = controller.load(path)

    replacement = _replace_with_compatible_zero_artifact(path)
    assert replacement.artifact_id != loaded.inspection["artifact_id"]
    assert replacement.predict_records(
        ({"left": False, "right": True},)
    ) == (0,)

    result = controller.run_record(
        {"left": "false", "right": "true"}, displayed_path=path
    )
    assert result.prediction == 1
    assert session.artifact_inspection is not None
    assert session.artifact_inspection["artifact_id"] == loaded.inspection["artifact_id"]


def test_search_controller_validates_budget_and_owns_result_state() -> None:
    expected = BoundedSearchResult(
        SearchKind.REPAIR,
        {"counterexamples": [{"example": [], "expected": True}]},
        0.01,
    )

    def runner(request, *, cancel=None):
        assert request.kind is SearchKind.REPAIR
        assert cancel is not None and not cancel()
        return expected

    session = SessionState()
    controller = SearchSessionController(session, runner=runner)
    prepared = controller.prepare(
        demo_search_document(SearchKind.REPAIR),
        expected_kind=SearchKind.REPAIR,
        timeout_seconds=2.0,
    )

    assert prepared.budget["candidate_upper_bound"] > 0
    assert session.search_state is JobState.QUEUED
    controller.mark_running()
    assert session.search_state is JobState.RUNNING

    result = controller.run(cancel=lambda: False)
    controller.complete(result)
    assert session.search_state is JobState.SUCCEEDED
    assert session.search_result is expected


def test_search_controller_can_run_captured_request_after_ui_reset() -> None:
    expected = BoundedSearchResult(SearchKind.REPAIR, {}, 0.01)

    def runner(request, *, cancel=None):
        assert request.kind is SearchKind.REPAIR
        return expected

    session = SessionState()
    controller = SearchSessionController(session, runner=runner)
    prepared = controller.prepare(
        demo_search_document(SearchKind.REPAIR),
        expected_kind=SearchKind.REPAIR,
        timeout_seconds=2.0,
    )
    controller.reset()

    assert controller.run(prepared.request) is expected


def test_search_controller_rejects_selector_document_mismatch() -> None:
    session = SessionState()
    controller = SearchSessionController(session)

    with pytest.raises(ValueError, match="does not match"):
        controller.prepare(
            demo_search_document(SearchKind.THRESHOLD),
            expected_kind=SearchKind.REPAIR,
            timeout_seconds=2.0,
        )
    assert session.search_state is JobState.IDLE
