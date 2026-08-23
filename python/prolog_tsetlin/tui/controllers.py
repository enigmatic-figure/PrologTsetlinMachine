"""UI-neutral session orchestration shared by interactive frontends.

The controllers in this module own service contracts and state transitions.  A
frontend remains responsible for collecting widget values and rendering the
returned, typed outcomes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

from ..services.artifacts import (
    ArtifactExportRequest,
    ArtifactSummary,
    export_training_run,
)
from ..services.inference import (
    ArtifactInferenceSession,
    ArtifactInputField,
    open_artifact_session,
    parse_typed_record,
    run_session_artifact_records,
)
from ..services.diagnostics import (
    RunDiagnostics,
    SampledTrainingDiagnostics,
    analyze_training_run,
    analyze_training_sample,
    compare_training_samples,
)
from ..services.search import (
    BoundedSearchRequest,
    BoundedSearchResult,
    SearchKind,
    export_search_artifact,
    run_bounded_search,
    search_request_budget,
)
from ..services.training import (
    DiagnosticCallback,
    ProgressCallback,
    TrainingDiagnosticSample,
    TrainingDiagnosticSampling,
    TrainingRequest,
    TrainingRun,
    train_xor,
)
from ..reference import TMSnapshot
from .models import JobState, SessionState


ACTIVE_JOB_STATES = frozenset(
    (JobState.QUEUED, JobState.RUNNING, JobState.CANCELLING)
)


class _CurrentRequestUnset:
    pass


_CURRENT_REQUEST_UNSET = _CurrentRequestUnset()


class SessionContractError(RuntimeError):
    """Raised when a requested UI operation violates session state."""


@dataclass(frozen=True, slots=True)
class TrainingInspection:
    """One completed-run snapshot selected for read-only inspection."""

    run: TrainingRun
    diagnostics: RunDiagnostics
    sampled: SampledTrainingDiagnostics | None = None

    @property
    def epoch(self) -> int:
        return (
            self.sampled.sample.epoch
            if self.sampled is not None
            else self.run.request.epochs
        )

    @property
    def historical(self) -> bool:
        return self.epoch != self.run.request.epochs

    @property
    def snapshot(self) -> TMSnapshot:
        return (
            self.sampled.sample.snapshot
            if self.sampled is not None
            else self.run.snapshot
        )

    @property
    def rows(self) -> tuple[tuple[bool, ...], ...]:
        return (
            self.sampled.sample.rows
            if self.sampled is not None
            else self.run.rows
        )

    @property
    def targets(self) -> tuple[int, ...]:
        return (
            self.sampled.sample.targets
            if self.sampled is not None
            else self.run.targets
        )

    @property
    def predictions(self) -> tuple[int, ...]:
        return (
            self.sampled.sample.predictions
            if self.sampled is not None
            else self.run.predictions
        )

    @property
    def accuracy(self) -> float:
        return (
            self.sampled.sample.accuracy
            if self.sampled is not None
            else self.run.accuracy
        )


class TrainingSessionController:
    """Own training state transitions and the training-export invariant."""

    def __init__(
        self,
        session: SessionState,
        *,
        trainer: Callable[..., TrainingRun] = train_xor,
        exporter: Callable[
            [TrainingRun, ArtifactExportRequest], ArtifactSummary
        ] = export_training_run,
    ) -> None:
        self.session = session
        self._trainer = trainer
        self._exporter = exporter

    @property
    def active(self) -> bool:
        return self.session.job_state in ACTIVE_JOB_STATES

    @property
    def retained_run_is_historical(self) -> bool:
        """Whether the retained run predates an active training job."""

        return self.active and self.session.last_completed_run is not None

    @property
    def active_request_matches_configuration(self) -> bool:
        """Whether the visible parsed configuration is the executing request."""

        return (
            self.session.active_request is not None
            and self.session.active_request == self.session.configured_request
        )

    def synchronize_configuration(
        self, current_request: TrainingRequest | None
    ) -> bool:
        """Recompute staleness, treating an unparseable form as stale."""

        self.session.configured_request = current_request
        run = self.session.last_completed_run
        self.session.configuration_dirty = run is not None and (
            current_request is None or run.request != current_request
        )
        return self.session.configuration_dirty

    def begin(
        self,
        request: TrainingRequest,
        *,
        diagnostic_sampling: TrainingDiagnosticSampling | None = None,
    ) -> None:
        if self.active:
            raise SessionContractError("training is already active")
        request.validate()
        if diagnostic_sampling is not None:
            diagnostic_sampling.validate()
        self.session.configured_request = request
        self.session.active_request = request
        self.session.configuration_dirty = (
            self.session.last_completed_run is not None
            and self.session.last_completed_run.request != request
        )
        self.session.error = None
        self.session.progress_epoch = 0
        self.session.progress_accuracy = 0.0
        self.session.accuracy_history.clear()
        self.session.active_diagnostic_sampling = diagnostic_sampling
        self.session.active_diagnostics.clear()
        self.session.inspected_sample_epoch = None
        self.session.job_state = JobState.QUEUED

    def mark_running(self) -> None:
        if self.session.job_state is JobState.QUEUED:
            self.session.job_state = JobState.RUNNING

    def record_progress(self, epoch: int, accuracy: float) -> None:
        self.mark_running()
        self.session.progress_epoch = epoch
        self.session.progress_accuracy = accuracy
        self.session.accuracy_history.append(accuracy)

    def record_diagnostic_sample(
        self, sample: TrainingDiagnosticSample
    ) -> SampledTrainingDiagnostics:
        active_request = self.session.active_request
        sampling = self.session.active_diagnostic_sampling
        if not self.active or active_request is None:
            raise SessionContractError("no active training request is available")
        if sampling is None:
            raise SessionContractError("active training did not request diagnostics")
        if sample.request != active_request:
            raise SessionContractError(
                "diagnostic sample does not belong to the active training request"
            )
        expected_epochs = sampling.selected_epochs(active_request.epochs)
        sample_index = len(self.session.active_diagnostics)
        if (
            sample_index >= len(expected_epochs)
            or sample.epoch != expected_epochs[sample_index]
        ):
            expected = (
                str(expected_epochs[sample_index])
                if sample_index < len(expected_epochs)
                else "no further sample"
            )
            raise SessionContractError(
                f"unexpected diagnostic epoch {sample.epoch}; expected {expected}"
            )
        diagnostics = analyze_training_sample(sample)
        previous = (
            self.session.active_diagnostics[-1].sample
            if self.session.active_diagnostics
            else None
        )
        delta = (
            compare_training_samples(previous, sample)
            if previous is not None
            else None
        )
        sampled = SampledTrainingDiagnostics(sample, diagnostics, delta)
        self.session.active_diagnostics.append(sampled)
        return sampled

    def complete(
        self,
        run: TrainingRun,
        *,
        current_request: TrainingRequest | None | _CurrentRequestUnset = (
            _CURRENT_REQUEST_UNSET
        ),
    ) -> None:
        active_request = self.session.active_request
        if active_request is None:
            raise SessionContractError("no active training request is available")
        if run.request != active_request:
            raise SessionContractError(
                "completed run does not belong to the active training request"
            )
        sampling = self.session.active_diagnostic_sampling
        if sampling is not None:
            expected_epochs = sampling.selected_epochs(active_request.epochs)
            actual_epochs = tuple(
                item.sample.epoch for item in self.session.active_diagnostics
            )
            if actual_epochs != expected_epochs:
                raise SessionContractError(
                    "completed run is missing requested diagnostic samples"
                )
            if expected_epochs and expected_epochs[-1] == active_request.epochs:
                final_sample = self.session.active_diagnostics[-1].sample
                if (
                    final_sample.rows != run.rows
                    or final_sample.targets != run.targets
                    or final_sample.predictions != run.predictions
                    or final_sample.accuracy != run.accuracy
                    or final_sample.snapshot != run.snapshot
                ):
                    raise SessionContractError(
                        "final diagnostic sample does not match the completed run"
                    )
        self.session.last_completed_run = run
        self.session.last_completed_accuracy_history = tuple(
            self.session.accuracy_history
        )
        self.session.last_completed_diagnostics = tuple(
            self.session.active_diagnostics
        )
        self.session.active_diagnostics.clear()
        self.session.active_diagnostic_sampling = None
        self.session.inspected_sample_epoch = None
        self.session.active_request = None
        if isinstance(current_request, _CurrentRequestUnset):
            self.synchronize_configuration(self.session.configured_request)
        else:
            self.synchronize_configuration(current_request)
        self.session.error = None
        self.session.job_state = JobState.SUCCEEDED

    def request_cancel(self) -> bool:
        if self.session.job_state not in (JobState.QUEUED, JobState.RUNNING):
            return False
        self.session.job_state = JobState.CANCELLING
        return True

    def cancelled(
        self,
        *,
        current_request: TrainingRequest | None | _CurrentRequestUnset = (
            _CURRENT_REQUEST_UNSET
        ),
    ) -> None:
        self.session.active_request = None
        self._discard_active_diagnostics()
        self.session.inspected_sample_epoch = None
        self.session.job_state = JobState.CANCELLED
        self._synchronize_terminal_configuration(current_request)

    def failed(
        self,
        message: str,
        *,
        current_request: TrainingRequest | None | _CurrentRequestUnset = (
            _CURRENT_REQUEST_UNSET
        ),
    ) -> None:
        self.session.active_request = None
        self._discard_active_diagnostics()
        self.session.inspected_sample_epoch = None
        self.session.error = message
        self.session.job_state = JobState.FAILED
        self._synchronize_terminal_configuration(current_request)

    def _synchronize_terminal_configuration(
        self,
        current_request: TrainingRequest | None | _CurrentRequestUnset,
    ) -> None:
        if isinstance(current_request, _CurrentRequestUnset):
            self.synchronize_configuration(self.session.configured_request)
        else:
            self.synchronize_configuration(current_request)

    def _discard_active_diagnostics(self) -> None:
        self.session.active_diagnostics.clear()
        self.session.active_diagnostic_sampling = None

    def inspect_completed_epoch(
        self,
        epoch: int | None = None,
        *,
        expected_run: TrainingRun | None = None,
    ) -> TrainingInspection:
        if (
            expected_run is not None
            and self.session.last_completed_run is not expected_run
        ):
            raise SessionContractError(
                "completed training run changed before inspection"
            )
        inspection = self._completed_inspection(epoch)
        self.session.inspected_sample_epoch = (
            inspection.epoch if inspection.historical else None
        )
        return inspection

    def current_inspection(self) -> TrainingInspection | None:
        if self.session.last_completed_run is None:
            return None
        return self._completed_inspection(self.session.inspected_sample_epoch)

    def _completed_inspection(
        self, epoch: int | None
    ) -> TrainingInspection:
        run = self.session.last_completed_run
        if run is None:
            raise SessionContractError("no completed training run is available")
        requested_epoch = run.request.epochs if epoch is None else epoch
        sampled = next(
            (
                item
                for item in self.session.last_completed_diagnostics
                if item.sample.epoch == requested_epoch
            ),
            None,
        )
        if sampled is None and requested_epoch != run.request.epochs:
            raise SessionContractError(
                f"epoch {requested_epoch} is not a retained diagnostic sample"
            )
        diagnostics = (
            sampled.diagnostics if sampled is not None else analyze_training_run(run)
        )
        return TrainingInspection(run, diagnostics, sampled)

    def run(
        self,
        request: TrainingRequest | None = None,
        *,
        progress: ProgressCallback | None = None,
        diagnostic: DiagnosticCallback | None = None,
        cancel: Event | None = None,
    ) -> TrainingRun:
        active_request = self.session.active_request
        if active_request is None:
            raise SessionContractError("no active training request is available")
        if request is not None and request != active_request:
            raise SessionContractError(
                "worker request does not match the active training request"
            )
        sampling = self.session.active_diagnostic_sampling
        if (diagnostic is None) != (sampling is None):
            raise SessionContractError(
                "diagnostic callback does not match the active sampling policy"
            )
        options: dict[str, object] = {
            "progress": progress,
            "cancel": cancel,
        }
        if sampling is not None:
            options["diagnostic"] = diagnostic
            options["diagnostic_sampling"] = sampling
        return self._trainer(active_request, **options)

    def require_exportable(self, current_request: TrainingRequest) -> TrainingRun:
        try:
            current_request.validate()
        except ValueError:
            self.synchronize_configuration(None)
            raise
        if self.active:
            raise SessionContractError("training is active; wait or cancel before exporting")
        run = self.session.last_completed_run
        if run is None:
            raise SessionContractError("no completed training run is available")
        self.synchronize_configuration(current_request)
        if run.request != current_request:
            raise SessionContractError(
                "visible configuration is stale; retrain before exporting"
            )
        return run

    def export(
        self,
        current_request: TrainingRequest,
        request: ArtifactExportRequest,
    ) -> ArtifactSummary:
        run = self.require_exportable(current_request)
        summary = self._exporter(run, request)
        self.session.artifact = summary
        return summary


@dataclass(frozen=True, slots=True)
class ArtifactLoadResult:
    path: Path
    inspection: Mapping[str, Any]
    verification: Mapping[str, Any]
    fields: tuple[ArtifactInputField, ...]


@dataclass(frozen=True, slots=True)
class ArtifactRecordResult:
    record: Mapping[str, object]
    prediction: int
    label: object
    features: tuple[int, ...]
    feature_trace: tuple[Mapping[str, Any], ...]


class ArtifactSessionController:
    """Own artifact loading and raw-record inference service contracts."""

    def __init__(self, session: SessionState) -> None:
        self.session = session
        self._loaded: ArtifactInferenceSession | None = None

    def clear_loaded(self) -> None:
        self._loaded = None
        self.session.loaded_artifact_path = None
        self.session.artifact_inspection = None
        self.session.artifact_fields = ()

    def load(self, path: str | Path) -> ArtifactLoadResult:
        resolved = Path(path).expanduser().resolve()
        try:
            loaded = open_artifact_session(resolved)
            inspection = loaded.inspection
            verification = loaded.verification
            if not verification.get("verified", False):
                raise SessionContractError("artifact conformance verification failed")
            fields = loaded.fields
        except Exception:
            self.clear_loaded()
            raise
        self._loaded = loaded
        self.session.loaded_artifact_path = resolved
        self.session.artifact_inspection = dict(inspection)
        self.session.artifact_fields = tuple(fields)
        return ArtifactLoadResult(
            resolved, inspection, verification, tuple(fields)
        )

    def verify_loaded(
        self, *, displayed_path: str | Path | None = None
    ) -> Mapping[str, Any]:
        loaded = self._require_loaded(displayed_path)
        return loaded.verification

    def run_record(
        self,
        values: Mapping[str, str],
        *,
        displayed_path: str | Path | None = None,
    ) -> ArtifactRecordResult:
        loaded = self._require_loaded(displayed_path)
        path = self.session.loaded_artifact_path
        fields = self.session.artifact_fields
        if path is None or not fields:
            raise SessionContractError("load a raw-record-capable artifact first")

        record = parse_typed_record(fields, values)
        report = run_session_artifact_records(loaded, (record,))
        results = report.get("results")
        if not isinstance(results, Sequence) or len(results) != 1:
            raise SessionContractError("artifact service returned an invalid result envelope")
        result = results[0]
        if not isinstance(result, Mapping):
            raise SessionContractError("artifact service returned an invalid record result")
        trace = result.get("feature_trace")
        features = result.get("features")
        prediction = result.get("prediction")
        if (
            not isinstance(trace, Sequence)
            or isinstance(trace, (str, bytes))
            or not isinstance(features, Sequence)
            or isinstance(features, (str, bytes))
            or type(prediction) is not int
        ):
            raise SessionContractError("artifact service omitted required inference fields")
        return ArtifactRecordResult(
            record=record,
            prediction=prediction,
            label=result.get("label", prediction),
            features=tuple(int(value) for value in features),
            feature_trace=tuple(
                item for item in trace if isinstance(item, Mapping)
            ),
        )

    def _require_loaded(
        self, displayed_path: str | Path | None
    ) -> ArtifactInferenceSession:
        loaded = self._loaded
        path = self.session.loaded_artifact_path
        if loaded is None or path is None:
            raise SessionContractError("load an artifact first")
        if loaded.source != path:
            self.clear_loaded()
            raise SessionContractError("artifact session state is inconsistent; reload it")
        if (
            displayed_path is not None
            and Path(displayed_path).expanduser().resolve() != path
        ):
            raise SessionContractError("artifact path changed; load it before running")
        return loaded


@dataclass(frozen=True, slots=True)
class PreparedSearch:
    request: BoundedSearchRequest
    budget: Mapping[str, Any]


class SearchSessionController:
    """Own bounded-search validation, state transitions, and export."""

    def __init__(
        self,
        session: SessionState,
        *,
        runner: Callable[..., BoundedSearchResult] = run_bounded_search,
        exporter: Callable[..., Mapping[str, Any]] = export_search_artifact,
    ) -> None:
        self.session = session
        self._runner = runner
        self._exporter = exporter

    @property
    def active(self) -> bool:
        return self.session.search_state in ACTIVE_JOB_STATES

    def reset(self) -> None:
        self.session.search_state = JobState.IDLE
        self.session.search_request = None
        self.session.search_result = None
        self.session.search_error = None

    def prepare(
        self,
        document: Mapping[str, Any],
        *,
        expected_kind: SearchKind | str,
        timeout_seconds: float,
    ) -> PreparedSearch:
        if self.active:
            raise SessionContractError("search is already active")
        request = BoundedSearchRequest.from_dict(
            document,
            expected_kind=expected_kind,
            timeout_seconds=timeout_seconds,
        )
        budget = search_request_budget(request)
        self.session.search_request = request
        self.session.search_result = None
        self.session.search_error = None
        self.session.search_state = JobState.QUEUED
        return PreparedSearch(request, budget)

    def mark_running(self) -> None:
        if self.session.search_state is JobState.QUEUED:
            self.session.search_state = JobState.RUNNING

    def request_cancel(self) -> bool:
        if self.session.search_state not in (JobState.QUEUED, JobState.RUNNING):
            return False
        self.session.search_state = JobState.CANCELLING
        return True

    def cancelled(self) -> None:
        self.session.search_state = JobState.CANCELLED

    def no_solution(self) -> None:
        self.session.search_result = None
        self.session.search_error = None
        self.session.search_state = JobState.SUCCEEDED

    def failed(self, message: str) -> None:
        self.session.search_result = None
        self.session.search_error = message
        self.session.search_state = JobState.FAILED

    def complete(self, result: BoundedSearchResult) -> None:
        self.session.search_result = result
        self.session.search_error = None
        self.session.search_state = JobState.SUCCEEDED

    def run(
        self,
        request: BoundedSearchRequest | None = None,
        *,
        cancel: Callable[[], bool] | None = None,
    ) -> BoundedSearchResult:
        request = request or self.session.search_request
        if request is None:
            raise SessionContractError("no prepared search request is available")
        return self._runner(request, cancel=cancel)

    def export(self, path: str | Path, *, name: str) -> Mapping[str, Any]:
        if self.active:
            raise SessionContractError("search is active; wait or cancel before exporting")
        result = self.session.search_result
        if result is None:
            raise SessionContractError("no completed search result is available")
        if not result.exportable:
            raise SessionContractError("this search result is not exportable")
        return self._exporter(result, path, name=name)
