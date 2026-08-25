"""UI-neutral orchestration for trained-parent model generations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from itertools import combinations
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from ..model_artifact import PackedTMInferenceArtifact, export_packed_tm
from ..model_generation import (
    AdaptedChild,
    AdaptiveBehaviorIdentity,
    AdaptiveRestorationBundle,
    AdaptiveSnapshotEnvelope,
    CorpusRole,
    ContractedParent,
    DeescalationCorpora,
    DriftAuditPolicy,
    EvidenceUsage,
    EvidenceUsagePurpose,
    ExtendedParent,
    LabeledCorpus,
    LiveRuntimeConformanceEvidence,
    LifecycleCorpora,
    LiteralContractionLineage,
    MAX_DEESCALATION_CANDIDATES,
    ModelGeneration,
    ModelGenerationError,
    ModelGenerationLineage,
    CONTRACTION_LINEAGE_SCHEMA,
    LINEAGE_SCHEMA,
    GenerationKind,
    OrderedLiteralManifest,
    PromotionAuditPolicy,
    PromotionAuditSnapshot,
    PromotionRuntimeConformanceEvidence,
    PrologInventionEvidence,
    PrologDeescalationEvidence,
    PrologThresholdCandidateSet,
    RuntimeConformanceReport,
    ThresholdCandidateBudget,
    ThresholdCandidateOutcome,
    ThresholdCandidateProposal,
    ThresholdCandidateSelection,
    ThresholdCandidateSelectionPolicy,
    adapt_extended_parent,
    audit_parent_child,
    audit_parent_child_snapshots,
    audit_runtime_conformance,
    audit_snapshot_runtime_conformance,
    canonical_json_bytes,
    content_digest,
    contract_parent_with_equivalent_literal,
    drift_requires_reopen,
    extend_parent_with_threshold,
    preprocessing_contract_id,
)
from ..preprocessing import PreprocessingContract
from ..prolog_resources import prolog_process_environment
from ..pta import (
    PTACollectiveBudget,
    PTACollectiveQuery,
    PTACollectiveService,
    PTAEscalationProposal,
    PTAReasoningSession,
    ReviewedThresholdProposal,
    review_threshold_proposal,
)
from ..reference import ScalarBinaryTsetlinMachine, TMSnapshot
from ._atomic import publish_bytes
from .telemetry import TelemetryEvent, TelemetrySession


_EVENT_ANCHOR = "sha256:" + "0" * 64
_EVENT_HEAD_SCHEMA = "ptm.model-generation-event-head.v1"
_MAX_EVENT_LOG_BYTES = 16 * 1024 * 1024
_MAX_EVENT_HEAD_BYTES = 4 * 1024
_MAX_STORED_JSON_BYTES = 4 * 1024 * 1024
_MAX_ATTESTED_EXECUTABLE_BYTES = 128 * 1024 * 1024
_STORE_LOCKS_GUARD = RLock()
_STORE_EVENT_LOCKS: dict[Path, RLock] = {}
_CANDIDATE_EVIDENCE_PURPOSES = frozenset(
    {
        EvidenceUsagePurpose.CANDIDATE_EPISODE,
        EvidenceUsagePurpose.DEESCALATION_EPISODE,
    }
)


def _event_lock_for_root(root: Path) -> RLock:
    with _STORE_LOCKS_GUARD:
        return _STORE_EVENT_LOCKS.setdefault(root, RLock())


def _file_digest(path: Path, *, maximum_bytes: int | None = None) -> str:
    size = path.stat().st_size
    if maximum_bytes is not None and size > maximum_bytes:
        raise ModelGenerationError("attested executable exceeds its size bound")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _gprolog_version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "--version"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10.0,
        check=False,
        env=prolog_process_environment(),
        creationflags=(
            getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        ),
    )
    first_line = next(
        (
            line.strip()
            for line in (completed.stdout + "\n" + completed.stderr).splitlines()
            if line.strip()
        ),
        "",
    )
    if completed.returncode != 0 or not first_line:
        raise ModelGenerationError("GNU Prolog version attestation failed")
    return first_line


def _collective_attestation(
    service: PTACollectiveService,
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """Measure one stable GNU Prolog collective implementation image."""

    first_executable_digest = _file_digest(
        service.executable, maximum_bytes=_MAX_ATTESTED_EXECUTABLE_BYTES
    )
    first_module_digests = tuple(
        sorted((name, _file_digest(path)) for name, path in service.module_paths.items())
    )
    version = _gprolog_version(service.executable)
    second_executable_digest = _file_digest(
        service.executable, maximum_bytes=_MAX_ATTESTED_EXECUTABLE_BYTES
    )
    second_module_digests = tuple(
        sorted((name, _file_digest(path)) for name, path in service.module_paths.items())
    )
    if (
        first_executable_digest != second_executable_digest
        or first_module_digests != second_module_digests
    ):
        raise ModelGenerationError(
            "GNU Prolog collective changed during implementation attestation"
        )
    return version, second_executable_digest, second_module_digests


def _content_path(root: Path, namespace: str, identifier: str, suffix: str) -> Path:
    if (
        not namespace
        or any(character not in "abcdefghijklmnopqrstuvwxyz-_" for character in namespace)
    ):
        raise ModelGenerationError("content namespace is invalid")
    if (
        not identifier.startswith("sha256:")
        or len(identifier) != 71
        or any(character not in "0123456789abcdef" for character in identifier[7:])
    ):
        raise ModelGenerationError("content identifier is invalid")
    return root / "objects" / namespace / f"{identifier[7:]}{suffix}"


class LifecycleEventKind(str, Enum):
    PARENT_REGISTERED = "parent_registered"
    EVIDENCE_RESERVED = "evidence_reserved"
    EVIDENCE_ABANDONED = "evidence_abandoned"
    CANDIDATE_CREATED = "candidate_created"
    CANDIDATE_REJECTED = "candidate_rejected"
    PROMOTION_APPROVED = "promotion_approved"
    ACTIVATED = "activated"
    REOPEN_REQUESTED = "reopen_requested"
    PARENT_RESTORED = "parent_restored"


@dataclass(frozen=True, slots=True)
class GenerationLifecycleEvent:
    sequence: int
    previous_event_id: str
    kind: LifecycleEventKind
    generation_id: str
    observed_at_utc: str
    details: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence <= 0:
            raise ModelGenerationError("lifecycle event sequence must be positive")
        if not isinstance(self.kind, LifecycleEventKind):
            raise TypeError("lifecycle event kind is invalid")
        for label, identifier in (
            ("previous event", self.previous_event_id),
            ("generation", self.generation_id),
        ):
            if (
                type(identifier) is not str
                or not identifier.startswith("sha256:")
                or len(identifier) != 71
            ):
                raise ModelGenerationError(f"{label} ID is invalid")
        if type(self.observed_at_utc) is not str or not self.observed_at_utc.endswith("Z"):
            raise ModelGenerationError("lifecycle event timestamp is invalid")
        if not isinstance(self.details, Mapping):
            raise TypeError("lifecycle event details must be a mapping")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @property
    def event_id(self) -> str:
        return content_digest(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "previous_event_id": self.previous_event_id,
            "kind": self.kind.value,
            "generation_id": self.generation_id,
            "observed_at_utc": self.observed_at_utc,
            "details": dict(self.details),
        }

    def to_dict(self) -> dict[str, object]:
        result = self.canonical_payload()
        result["event_id"] = self.event_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "GenerationLifecycleEvent":
        if not isinstance(value, Mapping) or set(value) != {
            "sequence",
            "previous_event_id",
            "kind",
            "generation_id",
            "observed_at_utc",
            "details",
            "event_id",
        }:
            raise ModelGenerationError("lifecycle event is malformed")
        details = value["details"]
        if not isinstance(details, Mapping):
            raise ModelGenerationError("lifecycle event details are malformed")
        if (
            type(value["sequence"]) is not int
            or any(
                type(value[key]) is not str
                for key in (
                    "previous_event_id",
                    "kind",
                    "generation_id",
                    "observed_at_utc",
                    "event_id",
                )
            )
        ):
            raise ModelGenerationError("lifecycle event field types are malformed")
        try:
            event = cls(
                value["sequence"],
                value["previous_event_id"],
                LifecycleEventKind(value["kind"]),
                value["generation_id"],
                value["observed_at_utc"],
                details,
            )
        except (TypeError, ValueError) as error:
            raise ModelGenerationError("lifecycle event is malformed") from error
        if event.event_id != value["event_id"]:
            raise ModelGenerationError("lifecycle event digest mismatch")
        return event


class ModelGenerationStore:
    """Content-addressed objects plus an atomically replaced hash-chain log."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._event_lock = _event_lock_for_root(self.root)

    def _put_bytes(
        self,
        namespace: str,
        identifier: str,
        suffix: str,
        data: bytes,
    ) -> Path:
        path = _content_path(self.root, namespace, identifier, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise ModelGenerationError("content-addressed object collision")
            return path
        publish_bytes(path, data, overwrite=False)
        return path

    def _put_json(
        self,
        namespace: str,
        identifier: str,
        value: Mapping[str, object],
    ) -> Path:
        encoded = canonical_json_bytes(value) + b"\n"
        if len(encoded) > _MAX_STORED_JSON_BYTES:
            raise ModelGenerationError("stored generation object is too large")
        return self._put_bytes(namespace, identifier, ".json", encoded)

    def _read_json(self, namespace: str, identifier: str) -> Mapping[str, object]:
        path = _content_path(self.root, namespace, identifier, ".json")
        data = path.read_bytes()
        if len(data) > _MAX_STORED_JSON_BYTES:
            raise ModelGenerationError("stored generation object is too large")
        try:
            decoded = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ModelGenerationError("stored generation object is not canonical JSON") from error
        if not isinstance(decoded, dict):
            raise ModelGenerationError("stored generation object must be a JSON object")
        if data != canonical_json_bytes(decoded) + b"\n":
            raise ModelGenerationError("stored generation object is not canonical JSON")
        return decoded

    def contains(self, namespace: str, identifier: str, suffix: str = ".json") -> bool:
        return _content_path(self.root, namespace, identifier, suffix).is_file()

    def put_snapshot(self, value: AdaptiveSnapshotEnvelope) -> Path:
        return self._put_json("snapshots", value.snapshot_id, value.to_dict())

    def load_snapshot(self, identifier: str) -> AdaptiveSnapshotEnvelope:
        result = AdaptiveSnapshotEnvelope.from_dict(
            self._read_json("snapshots", identifier)
        )
        if result.snapshot_id != identifier:
            raise ModelGenerationError("snapshot content does not match its address")
        return result

    def put_manifest(self, value: OrderedLiteralManifest) -> Path:
        return self._put_json("literal-manifests", value.manifest_id, value.to_dict())

    def load_manifest(self, identifier: str) -> OrderedLiteralManifest:
        result = OrderedLiteralManifest.from_dict(
            self._read_json("literal-manifests", identifier)
        )
        if result.manifest_id != identifier:
            raise ModelGenerationError("literal manifest does not match its address")
        return result

    def put_preprocessing(self, value: PreprocessingContract) -> tuple[str, Path]:
        identifier = preprocessing_contract_id(value)
        return identifier, self._put_json("preprocessing", identifier, value.to_dict())

    def load_preprocessing(self, identifier: str) -> PreprocessingContract:
        result = PreprocessingContract.from_dict(
            self._read_json("preprocessing", identifier)
        )
        if preprocessing_contract_id(result) != identifier:
            raise ModelGenerationError("preprocessing contract does not match its address")
        return result

    def put_artifact(self, value: PackedTMInferenceArtifact) -> Path:
        if not value.verify_conformance():
            raise ModelGenerationError("inference artifact failed conformance")
        return self._put_bytes("artifacts", value.artifact_id, ".ptm", value.serialized)

    def load_artifact(self, identifier: str) -> PackedTMInferenceArtifact:
        path = _content_path(self.root, "artifacts", identifier, ".ptm")
        artifact = PackedTMInferenceArtifact.from_bytes(path.read_bytes())
        if artifact.artifact_id != identifier or not artifact.verify_conformance():
            raise ModelGenerationError("stored inference artifact failed verification")
        return artifact

    def artifact_path(self, identifier: str) -> Path:
        path = _content_path(self.root, "artifacts", identifier, ".ptm")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def put_generation(self, value: ModelGeneration) -> Path:
        return self._put_json("generations", value.generation_id, value.to_dict())

    def load_generation(self, identifier: str) -> ModelGeneration:
        result = ModelGeneration.from_dict(self._read_json("generations", identifier))
        if result.generation_id != identifier:
            raise ModelGenerationError("model generation does not match its address")
        return result

    def put_restoration_bundle(self, value: AdaptiveRestorationBundle) -> Path:
        return self._put_json("restoration-bundles", value.bundle_id, value.to_dict())

    def load_restoration_bundle(self, identifier: str) -> AdaptiveRestorationBundle:
        result = AdaptiveRestorationBundle.from_dict(
            self._read_json("restoration-bundles", identifier)
        )
        if result.bundle_id != identifier:
            raise ModelGenerationError("restoration bundle does not match its address")
        return result

    def put_invention_evidence(self, value: PrologInventionEvidence) -> Path:
        return self._put_json("invention-evidence", value.evidence_id, value.to_dict())

    def load_invention_evidence(
        self, identifier: str
    ) -> PrologInventionEvidence | PrologThresholdCandidateSet:
        if self.contains("threshold-candidate-sets", identifier):
            return self.load_threshold_candidate_set(identifier)
        result = PrologInventionEvidence.from_dict(
            self._read_json("invention-evidence", identifier)
        )
        if result.evidence_id != identifier:
            raise ModelGenerationError("invention evidence does not match its address")
        return result

    def put_threshold_candidate_set(self, value: PrologThresholdCandidateSet) -> Path:
        return self._put_json(
            "threshold-candidate-sets", value.candidate_set_id, value.to_dict()
        )

    def load_threshold_candidate_set(
        self, identifier: str
    ) -> PrologThresholdCandidateSet:
        result = PrologThresholdCandidateSet.from_dict(
            self._read_json("threshold-candidate-sets", identifier)
        )
        if result.candidate_set_id != identifier:
            raise ModelGenerationError("threshold candidate set does not match its address")
        return result

    def put_threshold_candidate_selection(
        self, value: ThresholdCandidateSelection
    ) -> Path:
        return self._put_json(
            "threshold-candidate-selections", value.selection_id, value.to_dict()
        )

    def load_threshold_candidate_selection(
        self, identifier: str
    ) -> ThresholdCandidateSelection:
        result = ThresholdCandidateSelection.from_dict(
            self._read_json("threshold-candidate-selections", identifier)
        )
        if result.selection_id != identifier:
            raise ModelGenerationError(
                "threshold candidate selection does not match its address"
            )
        return result

    def put_deescalation_evidence(
        self, value: PrologDeescalationEvidence
    ) -> Path:
        return self._put_json(
            "deescalation-evidence", value.evidence_id, value.to_dict()
        )

    def load_deescalation_evidence(
        self, identifier: str
    ) -> PrologDeescalationEvidence:
        result = PrologDeescalationEvidence.from_dict(
            self._read_json("deescalation-evidence", identifier)
        )
        if result.evidence_id != identifier:
            raise ModelGenerationError(
                "de-escalation evidence does not match its address"
            )
        return result

    def put_evidence_usage(self, value: EvidenceUsage) -> Path:
        return self._put_json("evidence-usage", value.usage_id, value.to_dict())

    def load_evidence_usage(self, identifier: str) -> EvidenceUsage:
        result = EvidenceUsage.from_dict(
            self._read_json("evidence-usage", identifier)
        )
        if result.usage_id != identifier:
            raise ModelGenerationError("evidence usage does not match its address")
        return result

    def put_audit(self, value: PromotionAuditSnapshot) -> Path:
        return self._put_json("audits", value.audit_id, value.to_dict())

    def load_audit(self, identifier: str) -> PromotionAuditSnapshot:
        result = PromotionAuditSnapshot.from_dict(
            self._read_json("audits", identifier)
        )
        if result.audit_id != identifier:
            raise ModelGenerationError("promotion audit does not match its address")
        return result

    def put_promotion_conformance(
        self, value: PromotionRuntimeConformanceEvidence
    ) -> Path:
        return self._put_json(
            "promotion-conformance", value.evidence_id, value.to_dict()
        )

    def load_promotion_conformance(
        self, identifier: str
    ) -> PromotionRuntimeConformanceEvidence:
        try:
            result = PromotionRuntimeConformanceEvidence.from_dict(
                self._read_json("promotion-conformance", identifier)
            )
        except OSError as error:
            raise ModelGenerationError(
                "promotion conformance evidence is unavailable"
            ) from error
        if result.evidence_id != identifier:
            raise ModelGenerationError(
                "promotion conformance evidence does not match its address"
            )
        return result

    def put_live_conformance(self, value: LiveRuntimeConformanceEvidence) -> Path:
        return self._put_json(
            "live-conformance", value.evidence_id, value.to_dict()
        )

    def load_live_conformance(
        self, identifier: str
    ) -> LiveRuntimeConformanceEvidence:
        try:
            result = LiveRuntimeConformanceEvidence.from_dict(
                self._read_json("live-conformance", identifier)
            )
        except OSError as error:
            raise ModelGenerationError(
                "live conformance evidence is unavailable"
            ) from error
        if result.evidence_id != identifier:
            raise ModelGenerationError(
                "live conformance evidence does not match its address"
            )
        return result

    def put_lineage(
        self, value: ModelGenerationLineage | LiteralContractionLineage
    ) -> Path:
        return self._put_json("lineage", value.lineage_id, value.to_dict())

    def load_lineage(
        self, identifier: str
    ) -> ModelGenerationLineage | LiteralContractionLineage:
        raw = self._read_json("lineage", identifier)
        if raw.get("schema") == CONTRACTION_LINEAGE_SCHEMA:
            result = LiteralContractionLineage.from_dict(raw)
        else:
            result = ModelGenerationLineage.from_dict(raw)
        if result.lineage_id != identifier:
            raise ModelGenerationError("model-generation lineage does not match its address")
        return result

    @property
    def event_log_path(self) -> Path:
        return self.root / "lifecycle" / "events.jsonl"

    @property
    def event_head_path(self) -> Path:
        return self.root / "lifecycle" / "events.head.json"

    @staticmethod
    def _event_head_bytes(
        data: bytes, events: Sequence[GenerationLifecycleEvent]
    ) -> bytes:
        return canonical_json_bytes(
            {
                "schema": _EVENT_HEAD_SCHEMA,
                "sequence": len(events),
                "event_id": events[-1].event_id if events else _EVENT_ANCHOR,
                "log_digest": "sha256:" + hashlib.sha256(data).hexdigest(),
            }
        )

    def read_events(self) -> tuple[GenerationLifecycleEvent, ...]:
        path = self.event_log_path
        head_path = self.event_head_path
        if not path.exists() and not head_path.exists():
            return ()
        if not path.is_file() or not head_path.is_file():
            raise ModelGenerationError("model-generation event checkpoint is incomplete")
        data = path.read_bytes()
        if len(data) > _MAX_EVENT_LOG_BYTES:
            raise ModelGenerationError("model-generation event log is too large")
        events: list[GenerationLifecycleEvent] = []
        previous = _EVENT_ANCHOR
        for sequence, line in enumerate(data.splitlines(), start=1):
            if not line:
                raise ModelGenerationError("model-generation event log has an empty frame")
            try:
                decoded = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ModelGenerationError("model-generation event log is corrupt") from error
            if line != canonical_json_bytes(decoded):
                raise ModelGenerationError("model-generation event frame is not canonical")
            event = GenerationLifecycleEvent.from_dict(decoded)
            if event.sequence != sequence or event.previous_event_id != previous:
                raise ModelGenerationError("model-generation event chain is discontinuous")
            events.append(event)
            previous = event.event_id
        head_data = head_path.read_bytes()
        if len(head_data) > _MAX_EVENT_HEAD_BYTES:
            raise ModelGenerationError("model-generation event head is too large")
        try:
            head = json.loads(head_data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ModelGenerationError("model-generation event head is corrupt") from error
        if (
            head_data != canonical_json_bytes(head)
            or not isinstance(head, Mapping)
            or set(head) != {"schema", "sequence", "event_id", "log_digest"}
            or head.get("schema") != _EVENT_HEAD_SCHEMA
            or type(head.get("sequence")) is not int
            or type(head.get("event_id")) is not str
            or type(head.get("log_digest")) is not str
            or head_data != self._event_head_bytes(data, events)
        ):
            raise ModelGenerationError("model-generation event head does not match the log")
        return tuple(events)

    def append_event(
        self,
        kind: LifecycleEventKind,
        generation_id: str,
        **details: object,
    ) -> GenerationLifecycleEvent:
        with self._event_lock:
            events = self.read_events()
            previous_data = (
                self.event_log_path.read_bytes()
                if self.event_log_path.is_file()
                else b""
            )
            previous_head = self._event_head_bytes(previous_data, events)
            previous = events[-1].event_id if events else _EVENT_ANCHOR
            event = GenerationLifecycleEvent(
                len(events) + 1,
                previous,
                kind,
                generation_id,
                datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
                    "+00:00", "Z"
                ),
                details,
            )
            encoded = b"".join(
                canonical_json_bytes(item.to_dict()) + b"\n"
                for item in (*events, event)
            )
            if len(encoded) > _MAX_EVENT_LOG_BYTES:
                raise ModelGenerationError("model-generation event log is too large")
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                publish_bytes(self.event_log_path, encoded, overwrite=True)
                publish_bytes(
                    self.event_head_path,
                    self._event_head_bytes(encoded, (*events, event)),
                    overwrite=True,
                )
            except Exception:
                # Restore the prior complete checkpoint when an ordinary write
                # fails. A process/power loss between these publications leaves
                # a detectable log/head mismatch and therefore fails closed.
                try:
                    publish_bytes(self.event_log_path, previous_data, overwrite=True)
                    publish_bytes(
                        self.event_head_path, previous_head, overwrite=True
                    )
                except Exception:
                    pass
                raise
            return event

TelemetrySink = Callable[[TelemetryEvent], None]
LifecycleLineage = ModelGenerationLineage | LiteralContractionLineage


@dataclass(frozen=True, slots=True)
class RestoredAdaptiveParent:
    generation_id: str
    snapshot: AdaptiveSnapshotEnvelope
    manifest: OrderedLiteralManifest
    preprocessing: PreprocessingContract
    artifact: PackedTMInferenceArtifact
    machine: ScalarBinaryTsetlinMachine


@dataclass(frozen=True, slots=True)
class _LiveRuntimeVectors:
    artifact_id: str
    scalar_features: tuple[tuple[bool, ...], ...]
    scalar_scores: tuple[int, ...]
    scalar_predictions: tuple[int, ...]
    packed_predictions: tuple[int, ...]
    native_features: tuple[tuple[int, ...], ...]
    native_scores: tuple[int, ...]
    native_predictions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _LifecycleReplayState:
    events: tuple[GenerationLifecycleEvent, ...]
    active_generation_id: str | None
    registered_dataset_id: str | None
    pending_candidate_usage: EvidenceUsage | None
    candidate: LifecycleLineage | None
    approved: LifecycleLineage | None
    activated: LifecycleLineage | None
    pending_live_usage: EvidenceUsage | None
    reopen: tuple[
        LifecycleLineage,
        AdaptiveRestorationBundle,
        PromotionAuditSnapshot,
    ] | None
    activation_count: int
    last_activated_lineage_id: str | None
    activated_behavior_ids: frozenset[str]
    spent_example_keys: frozenset[tuple[str, str, str | int]]
    spent_record_fingerprints: frozenset[str]


class ModelGenerationController:
    """Own durable active-generation routing above each generation's Class II registry."""

    def __init__(
        self,
        store: ModelGenerationStore,
        *,
        ptmrt_executable: str | Path | None = None,
        telemetry: TelemetrySession | None = None,
        event_sink: TelemetrySink | None = None,
    ) -> None:
        self.store = store
        self.telemetry = telemetry
        self.event_sink = event_sink
        self._control_lock = RLock()
        self._last_telemetry_error: Exception | None = None
        self._ptmrt_executable: Path | None = None
        if ptmrt_executable is not None:
            self._bind_ptmrt_executable(ptmrt_executable)
        self._active_generation_id: str | None = None
        with self.store._event_lock:
            state = self._reconcile_replay_state_under_event_lock()
            self._active_generation_id = state.active_generation_id

    def _reconcile_replay_state_under_event_lock(self) -> _LifecycleReplayState:
        """Terminalize or complete every crash-interrupted durable transition."""

        while True:
            state = self._replay_lifecycle_state()
            pending = (
                state.pending_candidate_usage
                if state.pending_candidate_usage is not None
                else state.pending_live_usage
            )
            if pending is not None:
                self._append_evidence_abandoned_under_event_lock(pending)
                continue
            if state.candidate is not None:
                lineage = state.candidate
                *_, audit, _ = self._validate_lineage_graph(lineage)
                if audit.accepted and audit.conformance.exact:
                    self.store.append_event(
                        LifecycleEventKind.PROMOTION_APPROVED,
                        lineage.child_generation_id,
                        lineage_id=lineage.lineage_id,
                        audit_id=audit.audit_id,
                    )
                else:
                    self.store.append_event(
                        LifecycleEventKind.CANDIDATE_REJECTED,
                        lineage.child_generation_id,
                        lineage_id=lineage.lineage_id,
                        audit_id=audit.audit_id,
                        parent_errors=audit.parent_errors,
                        child_errors=audit.child_errors,
                        improvements=audit.improvements,
                        regressions=audit.regressions,
                    )
                continue
            if state.approved is not None:
                lineage = state.approved
                parent, _, child, _, audit, _ = self._validate_lineage_graph(
                    lineage
                )
                self.store.append_event(
                    LifecycleEventKind.ACTIVATED,
                    child.generation_id,
                    previous_generation_id=parent.generation_id,
                    lineage_id=lineage.lineage_id,
                    adaptive_behavior_id=lineage.adaptive_behavior_id,
                    audit_id=audit.audit_id,
                )
                continue
            if state.reopen is not None:
                lineage, bundle, drift = state.reopen
                active_child_id = state.active_generation_id
                if active_child_id is None:
                    raise ModelGenerationError(
                        "reopen recovery lacks an active child"
                    )
                self._resolve_restoration_bundle(bundle)
                self.store.append_event(
                    LifecycleEventKind.PARENT_RESTORED,
                    bundle.parent_generation_id,
                    restoration_bundle_id=bundle.bundle_id,
                    previous_generation_id=active_child_id,
                    lineage_id=lineage.lineage_id,
                    drift_audit_id=drift.audit_id,
                )
                continue
            return state

    @property
    def active_generation_id(self) -> str | None:
        with self._control_lock:
            return self._active_generation_id

    @property
    def last_telemetry_error(self) -> Exception | None:
        with self._control_lock:
            return self._last_telemetry_error

    def _emit(self, kind: str, **payload: object) -> None:
        if self.telemetry is None:
            return
        try:
            event = self.telemetry.emit("generation", kind, **payload)
            if self.event_sink is not None:
                self.event_sink(event)
        except Exception as error:
            # Telemetry is observational. In particular, a sink failure after
            # ACTIVATED or PARENT_RESTORED must not turn a committed control
            # transition into a reported lifecycle failure.
            with self._control_lock:
                self._last_telemetry_error = error

    def _bind_ptmrt_executable(
        self, executable: str | Path | None = None
    ) -> Path:
        if executable is not None:
            try:
                candidate = Path(executable).resolve(strict=True)
            except OSError as error:
                raise ModelGenerationError(
                    "trusted ptmrt executable is unavailable"
                ) from error
            if not candidate.is_file():
                raise ModelGenerationError("trusted ptmrt executable is not a file")
            if (
                self._ptmrt_executable is not None
                and candidate != self._ptmrt_executable
            ):
                raise ModelGenerationError(
                    "reopen requested a different ptmrt executable"
                )
            self._ptmrt_executable = candidate
        if self._ptmrt_executable is None:
            raise ModelGenerationError(
                "durable runtime conformance requires a trusted ptmrt executable"
            )
        return self._ptmrt_executable

    @staticmethod
    def _evidence_usage_id(event: GenerationLifecycleEvent) -> str | None:
        if event.kind not in (
            LifecycleEventKind.PARENT_REGISTERED,
            LifecycleEventKind.EVIDENCE_RESERVED,
        ):
            return None
        identifier = event.details.get("evidence_usage_id")
        if type(identifier) is not str:
            raise ModelGenerationError("lifecycle event lacks durable evidence usage")
        return identifier

    def _ensure_evidence_available(
        self,
        usage: EvidenceUsage,
        events: Sequence[GenerationLifecycleEvent],
    ) -> None:
        for event in events:
            identifier = self._evidence_usage_id(event)
            if identifier is None:
                continue
            previous = self.store.load_evidence_usage(identifier)
            if previous.usage_id == usage.usage_id:
                raise ModelGenerationError("evidence usage has already been reserved")
            if previous.dataset_id != usage.dataset_id:
                continue
            if previous.example_keys & usage.example_keys:
                raise ModelGenerationError(
                    "evidence observation identity has already been used"
                )
            if previous.record_fingerprints & usage.record_fingerprints:
                raise ModelGenerationError(
                    "evidence labeled-record fingerprint has already been used"
                )

    def _authoritative_state_under_event_lock(self) -> _LifecycleReplayState:
        state = self._replay_lifecycle_state()
        if state.active_generation_id != self._active_generation_id:
            raise ModelGenerationError(
                "durable lifecycle route differs from the running controller"
            )
        return state

    def _append_evidence_abandoned_under_event_lock(
        self, usage: EvidenceUsage
    ) -> None:
        state = self._replay_lifecycle_state()
        pending = (
            state.pending_candidate_usage
            if usage.purpose in _CANDIDATE_EVIDENCE_PURPOSES
            else state.pending_live_usage
        )
        if pending != usage:
            raise ModelGenerationError(
                "evidence abandonment lacks the current pending reservation"
            )
        self.store.append_event(
            LifecycleEventKind.EVIDENCE_ABANDONED,
            usage.subject_generation_id,
            evidence_usage_id=usage.usage_id,
            purpose=usage.purpose.value,
        )

    def _abandon_evidence_if_pending(self, usage: EvidenceUsage) -> None:
        with self._control_lock, self.store._event_lock:
            state = self._authoritative_state_under_event_lock()
            pending = (
                state.pending_candidate_usage
                if usage.purpose in _CANDIDATE_EVIDENCE_PURPOSES
                else state.pending_live_usage
            )
            if pending == usage:
                self._append_evidence_abandoned_under_event_lock(usage)

    def _abandon_pending_for_purpose_under_event_lock(
        self,
        state: _LifecycleReplayState,
        purpose: EvidenceUsagePurpose,
    ) -> _LifecycleReplayState:
        pending = (
            state.pending_candidate_usage
            if purpose in _CANDIDATE_EVIDENCE_PURPOSES
            else state.pending_live_usage
        )
        if pending is None:
            return state
        self._append_evidence_abandoned_under_event_lock(pending)
        return self._authoritative_state_under_event_lock()

    def _reserve_evidence_under_event_lock(
        self,
        usage: EvidenceUsage,
        state: _LifecycleReplayState,
    ) -> None:
        if usage.subject_generation_id != state.active_generation_id:
            raise ModelGenerationError("evidence subject is not the active generation")
        if not state.events or state.registered_dataset_id is None:
            raise ModelGenerationError("evidence reservation requires a registered route")
        if state.registered_dataset_id != usage.dataset_id:
            raise ModelGenerationError(
                "evidence usage belongs to a different registered dataset"
            )
        if usage.purpose in _CANDIDATE_EVIDENCE_PURPOSES:
            subject = self.store.load_generation(usage.subject_generation_id)
            if (
                subject.kind is not GenerationKind.TRAINED_PARENT
                or state.activated is not None
                or state.candidate is not None
                or state.approved is not None
                or state.reopen is not None
                or state.pending_candidate_usage is not None
            ):
                raise ModelGenerationError(
                    "candidate evidence reservation is invalid in the current state"
                )
        elif usage.purpose is EvidenceUsagePurpose.LIVE_DRIFT:
            subject = self.store.load_generation(usage.subject_generation_id)
            if (
                subject.kind
                not in (
                    GenerationKind.ADAPTED_CHILD,
                    GenerationKind.CONTRACTED_CHILD,
                )
                or state.activated is None
                or state.candidate is not None
                or state.approved is not None
                or state.reopen is not None
                or state.pending_live_usage is not None
            ):
                raise ModelGenerationError(
                    "live evidence reservation is invalid in the current state"
                )
        else:
            raise ModelGenerationError("parent evidence is committed during registration")
        self._ensure_evidence_available(usage, state.events)
        self.store.put_evidence_usage(usage)
        self.store.append_event(
            LifecycleEventKind.EVIDENCE_RESERVED,
            usage.subject_generation_id,
            evidence_usage_id=usage.usage_id,
            purpose=usage.purpose.value,
            dataset_id=usage.dataset_id,
        )

    def reserve_candidate_evidence(
        self, corpora: LifecycleCorpora
    ) -> tuple[EvidenceUsage, int, str | None]:
        with self._control_lock:
            if self._active_generation_id is None:
                raise ModelGenerationError(
                    "candidate evidence requires an active trained parent"
                )
            usage = EvidenceUsage(
                EvidenceUsagePurpose.CANDIDATE_EPISODE,
                self._active_generation_id,
                (corpora.invention, corpora.adaptation, corpora.promotion),
            )
            with self.store._event_lock:
                state = self._authoritative_state_under_event_lock()
                state = self._abandon_pending_for_purpose_under_event_lock(
                    state, EvidenceUsagePurpose.CANDIDATE_EPISODE
                )
                self._reserve_evidence_under_event_lock(usage, state)
                return (
                    usage,
                    state.activation_count + 1,
                    state.last_activated_lineage_id,
                )

    def reserve_deescalation_evidence(
        self, corpora: DeescalationCorpora
    ) -> tuple[EvidenceUsage, int, str | None]:
        with self._control_lock:
            if self._active_generation_id is None:
                raise ModelGenerationError(
                    "de-escalation evidence requires an active trained parent"
                )
            usage = EvidenceUsage(
                EvidenceUsagePurpose.DEESCALATION_EPISODE,
                self._active_generation_id,
                (corpora.proof, corpora.confirmation, corpora.promotion),
            )
            with self.store._event_lock:
                state = self._authoritative_state_under_event_lock()
                state = self._abandon_pending_for_purpose_under_event_lock(
                    state, EvidenceUsagePurpose.DEESCALATION_EPISODE
                )
                self._reserve_evidence_under_event_lock(usage, state)
                return (
                    usage,
                    state.activation_count + 1,
                    state.last_activated_lineage_id,
                )
    def _validate_deployable_generation(
        self, generation: ModelGeneration
    ) -> tuple[
        AdaptiveSnapshotEnvelope,
        OrderedLiteralManifest,
        PreprocessingContract,
        PackedTMInferenceArtifact,
    ]:
        if generation.kind not in (
            GenerationKind.TRAINED_PARENT,
            GenerationKind.ADAPTED_CHILD,
            GenerationKind.CONTRACTED_CHILD,
        ) or generation.inference_artifact_id is None:
            raise ModelGenerationError("generation is not deployable")
        if self.store.load_generation(generation.generation_id) != generation:
            raise ModelGenerationError("deployable generation changed after publication")
        snapshot = self.store.load_snapshot(generation.snapshot_id)
        manifest = self.store.load_manifest(generation.literal_manifest_id)
        preprocessing = self.store.load_preprocessing(
            generation.preprocessing_contract_id
        )
        artifact = self.store.load_artifact(generation.inference_artifact_id)
        validation = artifact.manifest.get("validation")
        signature = (
            validation.get("signature")
            if isinstance(validation, Mapping)
            else None
        )
        artifact_preprocessing = artifact.preprocessing
        feature_contract = artifact.manifest.get("features")
        word_count = (snapshot.snapshot.number_of_features + 63) // 64
        expected_positive = [0] * (snapshot.snapshot.number_of_clauses * word_count)
        expected_negative = [0] * (snapshot.snapshot.number_of_clauses * word_count)
        for clause, states in enumerate(snapshot.snapshot.states):
            for feature in range(snapshot.snapshot.number_of_features):
                mask_index = clause * word_count + feature // 64
                bit = 1 << (feature % 64)
                if states[feature * 2] > snapshot.snapshot.states_per_action:
                    expected_positive[mask_index] |= bit
                if states[feature * 2 + 1] > snapshot.snapshot.states_per_action:
                    expected_negative[mask_index] |= bit
        expected_stage = {
            GenerationKind.TRAINED_PARENT: "trained_parent",
            GenerationKind.ADAPTED_CHILD: "adapted_child",
            GenerationKind.CONTRACTED_CHILD: "contracted_child",
        }[generation.kind]
        if (
            snapshot.snapshot.number_of_features != len(manifest.literals)
            or tuple(preprocessing.literal_ids) != manifest.literal_ids
            or artifact_preprocessing != preprocessing
            or not isinstance(feature_contract, Mapping)
            or feature_contract.get("catalog_version") != manifest.manifest_id
            or feature_contract.get("literal_ids")
            != [str(value) for value in manifest.literal_ids]
            or artifact.number_of_clauses != snapshot.snapshot.number_of_clauses
            or artifact.number_of_features != len(manifest.literals)
            or artifact.threshold != snapshot.snapshot.threshold
            or artifact.positive_include_masks != tuple(expected_positive)
            or artifact.negative_include_masks != tuple(expected_negative)
            or not artifact.verify_conformance()
            or artifact_preprocessing is None
            or preprocessing_contract_id(artifact_preprocessing)
            != generation.preprocessing_contract_id
            or not isinstance(signature, Mapping)
            or signature.get("generation_stage") != expected_stage
            or signature.get("adaptive_snapshot_id") != generation.snapshot_id
            or signature.get("ordered_literal_manifest_id")
            != generation.literal_manifest_id
        ):
            raise ModelGenerationError("deployable generation object graph is inconsistent")
        if generation.kind is GenerationKind.TRAINED_PARENT:
            corpora = dict(generation.corpus_digests)
            if (
                len(corpora) != 1
                or signature.get("training_corpus_digest")
                != corpora.get(CorpusRole.PARENT_TRAINING.value)
            ):
                raise ModelGenerationError("trained parent deployment evidence is inconsistent")
        else:
            behavior_id = AdaptiveBehaviorIdentity.from_generation(
                generation
            ).behavior_id
            if signature.get("adaptive_behavior_id") != behavior_id:
                raise ModelGenerationError("child artifact names a different adaptive behavior")
        return snapshot, manifest, preprocessing, artifact

    def _parent_training_corpus(
        self, generation: ModelGeneration
    ) -> LabeledCorpus:
        registration = next(
            (
                event
                for event in self.store.read_events()
                if event.kind is LifecycleEventKind.PARENT_REGISTERED
                and event.generation_id == generation.generation_id
            ),
            None,
        )
        usage_id = (
            registration.details.get("evidence_usage_id")
            if registration is not None
            else None
        )
        if type(usage_id) is not str:
            raise ModelGenerationError(
                "trained parent lacks durable registration evidence"
            )
        usage = self.store.load_evidence_usage(usage_id)
        if (
            usage.purpose is not EvidenceUsagePurpose.PARENT_REGISTRATION
            or usage.subject_generation_id != generation.generation_id
            or len(usage.corpora) != 1
            or usage.corpora[0].role is not CorpusRole.PARENT_TRAINING
        ):
            raise ModelGenerationError(
                "trained parent registration evidence is inconsistent"
            )
        return usage.corpora[0]

    def _validate_promotion_conformance_evidence(
        self,
        evidence_id: str,
        child: ModelGeneration,
        child_snapshot: AdaptiveSnapshotEnvelope,
        child_manifest: OrderedLiteralManifest,
        artifact: PackedTMInferenceArtifact,
        corpus: LabeledCorpus,
    ) -> RuntimeConformanceReport:
        evidence = self.store.load_promotion_conformance(evidence_id)
        if corpus.role is not CorpusRole.PROMOTION:
            raise ModelGenerationError(
                "promotion conformance requires the promotion holdout"
            )
        if (
            evidence.child_generation_id != child.generation_id
            or evidence.artifact_id != child.inference_artifact_id
            or evidence.snapshot_id != child.snapshot_id
            or evidence.literal_manifest_id != child.literal_manifest_id
            or evidence.corpus_digest != corpus.digest
            or evidence.case_count != len(corpus.examples)
        ):
            raise ModelGenerationError(
                "promotion conformance identity binding is inconsistent"
            )
        trusted_ptmrt = self._bind_ptmrt_executable()
        executable_digest = _file_digest(
            trusted_ptmrt,
            maximum_bytes=_MAX_ATTESTED_EXECUTABLE_BYTES,
        )
        if executable_digest != evidence.ptmrt_binary_digest:
            raise ModelGenerationError(
                "promotion conformance ptmrt executable digest does not match"
            )
        native = _verify_snapshot_records_with_ptmrt(
            trusted_ptmrt,
            self.store.artifact_path(artifact.artifact_id),
            child_snapshot,
            child_manifest,
            artifact,
            corpus.records,
        )
        if (
            _file_digest(
                trusted_ptmrt,
                maximum_bytes=_MAX_ATTESTED_EXECUTABLE_BYTES,
            )
            != evidence.ptmrt_binary_digest
            or native.artifact_id != evidence.artifact_id
            or native.scalar_features != evidence.scalar_features
            or native.scalar_scores != evidence.scalar_scores
            or native.scalar_predictions != evidence.scalar_predictions
            or native.packed_predictions != evidence.packed_predictions
            or native.native_features != evidence.native_features
            or native.native_scores != evidence.native_scores
            or native.native_predictions != evidence.native_predictions
        ):
            raise ModelGenerationError(
                "promotion native execution cannot be reproduced"
            )
        report = audit_snapshot_runtime_conformance(
            child_snapshot,
            child_manifest,
            artifact,
            corpus.records,
            ptmrt_verified=True,
            ptmrt_artifact_id=evidence.artifact_id,
        )
        if not report.exact:
            raise ModelGenerationError(
                "promotion runtime conformance is not exact"
            )
        return report

    def _validate_lineage_graph(
        self, lineage: LifecycleLineage
    ) -> tuple[
        ModelGeneration,
        ModelGeneration,
        ModelGeneration,
        AdaptiveRestorationBundle,
        PromotionAuditSnapshot,
        PrologInventionEvidence
        | PrologThresholdCandidateSet
        | PrologDeescalationEvidence,
    ]:
        if isinstance(lineage, LiteralContractionLineage):
            return self._validate_contraction_lineage_graph(lineage)
        if isinstance(lineage, ModelGenerationLineage):
            return self._validate_threshold_lineage_graph(lineage)
        raise TypeError("unsupported lifecycle lineage type")

    def _validate_threshold_lineage_graph(
        self, lineage: ModelGenerationLineage
    ) -> tuple[
        ModelGeneration,
        ModelGeneration,
        ModelGeneration,
        AdaptiveRestorationBundle,
        PromotionAuditSnapshot,
        PrologInventionEvidence | PrologThresholdCandidateSet,
    ]:
        if self.store.load_lineage(lineage.lineage_id) != lineage:
            raise ModelGenerationError("lineage differs from its durable object")
        parent = self.store.load_generation(lineage.parent_generation_id)
        extended = self.store.load_generation(lineage.extended_generation_id)
        child = self.store.load_generation(lineage.child_generation_id)
        bundle = self.store.load_restoration_bundle(lineage.restoration_bundle_id)
        audit = self.store.load_audit(lineage.promotion_audit_id)
        selection: ThresholdCandidateSelection | None = None
        if lineage.candidate_selection_id is None:
            evidence: PrologInventionEvidence | PrologThresholdCandidateSet = (
                self.store.load_invention_evidence(lineage.invention_evidence_id)
            )
        else:
            evidence = self.store.load_threshold_candidate_set(
                lineage.invention_evidence_id
            )
            selection = self.store.load_threshold_candidate_selection(
                lineage.candidate_selection_id
            )
        usage = self.store.load_evidence_usage(lineage.evidence_usage_id)
        parent_manifest = self.store.load_manifest(parent.literal_manifest_id)
        extended_manifest = self.store.load_manifest(extended.literal_manifest_id)
        child_manifest = self.store.load_manifest(child.literal_manifest_id)
        parent_snapshot = self.store.load_snapshot(parent.snapshot_id).snapshot
        extended_snapshot = self.store.load_snapshot(extended.snapshot_id).snapshot
        child_snapshot = self.store.load_snapshot(child.snapshot_id).snapshot
        replayed_session: PTAReasoningSession | None = None
        replayed_candidates: Mapping[
            str, tuple[PTAEscalationProposal, ReviewedThresholdProposal]
        ] = MappingProxyType({})
        parent_training_corpus: LabeledCorpus | None = None
        if child.inference_artifact_id is None:
            raise ModelGenerationError("adapted child lacks an inference artifact")
        _, _, _, child_artifact = self._validate_deployable_generation(child)
        child_validation = child_artifact.manifest.get("validation")
        child_signature = (
            child_validation.get("signature")
            if isinstance(child_validation, Mapping)
            else None
        )
        child_signature_map = (
            child_signature if isinstance(child_signature, Mapping) else {}
        )
        child_preprocessing = child_artifact.preprocessing
        child_restoration = child_artifact.manifest.get("restoration_reference")
        invention_digests = ((CorpusRole.INVENTION.value, lineage.invention_corpus_digest),)
        child_digests = (
            (CorpusRole.INVENTION.value, lineage.invention_corpus_digest),
            (CorpusRole.ADAPTATION.value, lineage.adaptation_corpus_digest),
            (CorpusRole.PROMOTION.value, lineage.promotion_corpus_digest),
        )
        usage_digests = tuple(
            (corpus.role.value, corpus.digest) for corpus in usage.corpora
        )
        promotion_corpus = next(
            (
                corpus
                for corpus in usage.corpora
                if corpus.role is CorpusRole.PROMOTION
            ),
            None,
        )
        promotion_conformance: RuntimeConformanceReport | None = None
        if lineage.promotion_conformance_evidence_id is not None:
            if promotion_corpus is None:
                raise ModelGenerationError(
                    "lineage lacks its promotion holdout"
                )
            promotion_conformance = (
                self._validate_promotion_conformance_evidence(
                    lineage.promotion_conformance_evidence_id,
                    child,
                    AdaptiveSnapshotEnvelope(child_snapshot),
                    child_manifest,
                    child_artifact,
                    promotion_corpus,
                )
            )
        if selection is None:
            invention_consistent = (
                isinstance(evidence, PrologInventionEvidence)
                and evidence.proposal_semantic_id
                == lineage.origin_proposal_semantic_id
                and evidence.proposal_provenance_id
                == lineage.origin_proposal_provenance_id
            )
            selection_consistent = (
                child_signature_map.get("threshold_candidate_set_id") is None
                and child_signature_map.get("threshold_candidate_selection_id") is None
            )
        else:
            invention_corpus = next(
                (
                    corpus
                    for corpus in usage.corpora
                    if corpus.role is CorpusRole.INVENTION
                ),
                None,
            )
            if invention_corpus is None:
                raise ModelGenerationError(
                    "threshold candidate selection lacks invention evidence"
                )
            replayed_session, replayed_candidates = (
                _validate_threshold_candidate_derivation(
                    evidence, invention_corpus, parent_manifest
                )
            )
            parent_training_corpus = self._parent_training_corpus(parent)
            selected = selection.selected_outcome
            candidate_ids = {
                (item.proposal_semantic_id, item.proposal_provenance_id)
                for item in evidence.candidates
            }
            outcome_ids = {
                (item.proposal_semantic_id, item.proposal_provenance_id)
                for item in selection.outcomes
            }
            invention_consistent = (
                isinstance(evidence, PrologThresholdCandidateSet)
                and selection.candidate_set_id == evidence.candidate_set_id
                and candidate_ids == outcome_ids
                and selection.parent_generation_id == parent.generation_id
                and selection.parent_snapshot_id == parent.snapshot_id
                and selection.parent_manifest_id == parent.literal_manifest_id
                and selection.adaptation_corpus_digest
                == lineage.adaptation_corpus_digest
                and selection.selected_proposal_semantic_id
                == lineage.origin_proposal_semantic_id
                and selection.selected_proposal_provenance_id
                == lineage.origin_proposal_provenance_id
                and selected.invented_literal_id == lineage.invented_literal_id
                and selected.extended_snapshot_id == extended.snapshot_id
                and selected.extended_manifest_id == extended.literal_manifest_id
                and selected.child_snapshot_id == child.snapshot_id
                and selected.child_manifest_id == child.literal_manifest_id
                and selected.child_preprocessing_id
                == child.preprocessing_contract_id
                and selected.adaptive_behavior_id == lineage.adaptive_behavior_id
            )
            selection_consistent = (
                child_signature_map.get("threshold_candidate_set_id")
                == evidence.candidate_set_id
                and child_signature_map.get("threshold_candidate_selection_id")
                == selection.selection_id
            )
        if (
            parent.kind is not GenerationKind.TRAINED_PARENT
            or extended.kind is not GenerationKind.EXTENDED_PARENT
            or child.kind is not GenerationKind.ADAPTED_CHILD
            or extended.parent_generation_id != parent.generation_id
            or child.parent_generation_id != extended.generation_id
            or extended.restoration_bundle_id != bundle.bundle_id
            or child.restoration_bundle_id != bundle.bundle_id
            or bundle.parent_generation_id != parent.generation_id
            or extended.corpus_digests != invention_digests
            or child.corpus_digests != child_digests
            or extended.origin_proposal_semantic_id
            != lineage.origin_proposal_semantic_id
            or extended.origin_proposal_provenance_id
            != lineage.origin_proposal_provenance_id
            or child.origin_proposal_semantic_id
            != lineage.origin_proposal_semantic_id
            or child.origin_proposal_provenance_id
            != lineage.origin_proposal_provenance_id
            or evidence.invention_corpus_digest != lineage.invention_corpus_digest
            or not invention_consistent
            or not selection_consistent
            or usage.purpose is not EvidenceUsagePurpose.CANDIDATE_EPISODE
            or usage.subject_generation_id != parent.generation_id
            or usage_digests != child_digests
            or audit.corpus_role is not CorpusRole.PROMOTION
            or audit.corpus_digest != lineage.promotion_corpus_digest
            or audit.conformance.artifact_id != child.inference_artifact_id
            or (
                lineage.schema == LINEAGE_SCHEMA
                and promotion_conformance is None
            )
            or (
                promotion_conformance is not None
                and audit.conformance != promotion_conformance
            )
            or lineage.adaptive_behavior_id
            != AdaptiveBehaviorIdentity.from_generation(child).behavior_id
            or not isinstance(child_signature, Mapping)
            or child_signature.get("generation_stage") != "adapted_child"
            or child_signature.get("adaptive_behavior_id")
            != lineage.adaptive_behavior_id
            or child_signature.get("adaptive_snapshot_id") != child.snapshot_id
            or child_signature.get("ordered_literal_manifest_id")
            != child.literal_manifest_id
            or child_signature.get("invention_corpus_digest")
            != lineage.invention_corpus_digest
            or child_signature.get("adaptation_corpus_digest")
            != lineage.adaptation_corpus_digest
            or child_signature.get("promotion_corpus_digest")
            != lineage.promotion_corpus_digest
            or child_signature.get("origin_proposal_semantic_id")
            != lineage.origin_proposal_semantic_id
            or child_signature.get("origin_proposal_provenance_id")
            != lineage.origin_proposal_provenance_id
            or child_preprocessing is None
            or preprocessing_contract_id(child_preprocessing)
            != child.preprocessing_contract_id
            or child_restoration != bundle.to_dict()
        ):
            raise ModelGenerationError("lineage object graph is inconsistent")
        if lineage.schema == LINEAGE_SCHEMA:
            if promotion_corpus is None or promotion_conformance is None:
                raise ModelGenerationError(
                    "current lineage lacks replayable promotion evidence"
                )
            reconstructed_promotion = audit_parent_child_snapshots(
                parent_snapshot,
                parent_manifest,
                AdaptiveSnapshotEnvelope(child_snapshot),
                child_manifest,
                promotion_corpus,
                promotion_conformance,
                PromotionAuditPolicy(
                    minimum_observations=audit.observations,
                    require_strict_improvement=True,
                    maximum_regressions=0,
                ),
            )
            if reconstructed_promotion != audit:
                raise ModelGenerationError(
                    "promotion audit cannot be reconstructed"
                )
        if selection is not None:
            if replayed_session is None or parent_training_corpus is None:
                raise ModelGenerationError(
                    "threshold candidate derivation replay is unavailable"
                )
            adaptation_corpus = next(
                corpus
                for corpus in usage.corpora
                if corpus.role is CorpusRole.ADAPTATION
            )
            candidates_by_id = {
                item.proposal_semantic_id: item for item in evidence.candidates
            }
            parent_predictions = _snapshot_predictions(
                parent_snapshot, parent_manifest, adaptation_corpus.records
            )
            for outcome in selection.outcomes:
                candidate = candidates_by_id[outcome.proposal_semantic_id]
                proposal, reviewed = replayed_candidates[
                    outcome.proposal_semantic_id
                ]
                alternative_extended = self.store.load_snapshot(
                    outcome.extended_snapshot_id
                ).snapshot
                alternative_manifest = self.store.load_manifest(
                    outcome.extended_manifest_id
                )
                alternative_child = self.store.load_snapshot(
                    outcome.child_snapshot_id
                ).snapshot
                alternative_child_manifest = self.store.load_manifest(
                    outcome.child_manifest_id
                )
                alternative_preprocessing = self.store.load_preprocessing(
                    outcome.child_preprocessing_id
                )
                replayed_extended = extend_parent_with_threshold(
                    parent_snapshot,
                    parent_manifest,
                    reviewed,
                    session=replayed_session,
                    equivalence_records=parent_training_corpus.records,
                )
                replayed_child = adapt_extended_parent(
                    replayed_extended,
                    adaptation_corpus,
                    epochs=selection.adaptation_epochs,
                )
                descriptor = alternative_manifest.literals[-1]
                if (
                    proposal.semantic_id() != candidate.proposal_semantic_id
                    or candidate.proposal_provenance_id
                    != outcome.proposal_provenance_id
                    or candidate.invented_literal_id != outcome.invented_literal_id
                    or len(alternative_manifest.literals)
                    != len(parent_manifest.literals) + 1
                    or alternative_manifest.literals[:-1] != parent_manifest.literals
                    or descriptor.literal_id != candidate.invented_literal_id
                    or descriptor.source_field != candidate.field
                    or descriptor.transform.value != "numeric_ge"
                    or dict(descriptor.parameters).get("threshold")
                    != candidate.threshold
                    or alternative_child_manifest != alternative_manifest
                    or tuple(alternative_preprocessing.literal_ids)
                    != alternative_child_manifest.literal_ids
                    or preprocessing_contract_id(alternative_preprocessing)
                    != outcome.child_preprocessing_id
                    or alternative_extended.number_of_features
                    != parent_snapshot.number_of_features + 1
                    or alternative_extended.number_of_clauses
                    != parent_snapshot.number_of_clauses
                    or alternative_extended.states_per_action
                    != parent_snapshot.states_per_action
                    or alternative_extended.specificity != parent_snapshot.specificity
                    or alternative_extended.threshold != parent_snapshot.threshold
                    or alternative_extended.rng_state != parent_snapshot.rng_state
                    or alternative_child.number_of_features
                    != alternative_extended.number_of_features
                    or alternative_child.number_of_clauses
                    != alternative_extended.number_of_clauses
                    or alternative_child.states_per_action
                    != alternative_extended.states_per_action
                    or alternative_child.specificity
                    != alternative_extended.specificity
                    or alternative_child.threshold != alternative_extended.threshold
                    or replayed_extended.snapshot.snapshot_id
                    != outcome.extended_snapshot_id
                    or replayed_extended.manifest.manifest_id
                    != outcome.extended_manifest_id
                    or replayed_child.snapshot.snapshot_id
                    != outcome.child_snapshot_id
                    or replayed_child.manifest.manifest_id
                    != outcome.child_manifest_id
                    or replayed_child.adaptation_corpus_digest
                    != selection.adaptation_corpus_digest
                    or replayed_child.epochs != selection.adaptation_epochs
                    or preprocessing_contract_id(
                        PreprocessingContract.from_catalog(
                            replayed_child.manifest.build_catalog()
                        )
                    )
                    != outcome.child_preprocessing_id
                    or AdaptiveBehaviorIdentity(
                        outcome.child_snapshot_id,
                        outcome.child_manifest_id,
                        outcome.child_preprocessing_id,
                    ).behavior_id
                    != outcome.adaptive_behavior_id
                ):
                    raise ModelGenerationError(
                        "threshold selection alternative graph is inconsistent"
                    )
                for old_states, new_states in zip(
                    parent_snapshot.states, alternative_extended.states
                ):
                    if new_states[:-2] != old_states or new_states[-2:] != (
                        parent_snapshot.states_per_action,
                        parent_snapshot.states_per_action,
                    ):
                        raise ModelGenerationError(
                            "threshold selection alternative is not an exact extension"
                        )
                child_predictions = _snapshot_predictions(
                    alternative_child,
                    alternative_child_manifest,
                    adaptation_corpus.records,
                )
                both_correct = both_wrong = improvements = regressions = 0
                for truth, parent_prediction, child_prediction in zip(
                    adaptation_corpus.labels,
                    parent_predictions,
                    child_predictions,
                ):
                    parent_correct = parent_prediction == truth
                    child_correct = child_prediction == truth
                    if parent_correct and child_correct:
                        both_correct += 1
                    elif not parent_correct and not child_correct:
                        both_wrong += 1
                    elif not parent_correct and child_correct:
                        improvements += 1
                    else:
                        regressions += 1
                if (
                    outcome.observations != len(adaptation_corpus.examples)
                    or outcome.parent_errors != both_wrong + improvements
                    or outcome.child_errors != both_wrong + regressions
                    or outcome.disagreements != improvements + regressions
                    or outcome.improvements != improvements
                    or outcome.regressions != regressions
                    or outcome.both_correct != both_correct
                    or outcome.both_wrong != both_wrong
                ):
                    raise ModelGenerationError(
                        "threshold selection metrics differ from adaptation evidence"
                    )
        if (
            len(extended_manifest.literals) != len(parent_manifest.literals) + 1
            or extended_manifest.literals[:-1] != parent_manifest.literals
            or extended_manifest.literals[-1].literal_id != lineage.invented_literal_id
            or child_manifest != extended_manifest
            or extended_snapshot.number_of_features
            != parent_snapshot.number_of_features + 1
            or extended_snapshot.number_of_clauses
            != parent_snapshot.number_of_clauses
            or extended_snapshot.states_per_action
            != parent_snapshot.states_per_action
            or extended_snapshot.specificity != parent_snapshot.specificity
            or extended_snapshot.threshold != parent_snapshot.threshold
            or extended_snapshot.rng_state != parent_snapshot.rng_state
            or child_snapshot.number_of_features != len(child_manifest.literals)
            or child_snapshot.number_of_clauses
            != extended_snapshot.number_of_clauses
            or child_snapshot.states_per_action
            != extended_snapshot.states_per_action
            or child_snapshot.specificity != extended_snapshot.specificity
            or child_snapshot.threshold != extended_snapshot.threshold
        ):
            raise ModelGenerationError("lineage representation extension is inconsistent")
        for old_states, new_states in zip(
            parent_snapshot.states, extended_snapshot.states
        ):
            if new_states[:-2] != old_states or new_states[-2:] != (
                parent_snapshot.states_per_action,
                parent_snapshot.states_per_action,
            ):
                raise ModelGenerationError("lineage P+ snapshot is not an exact extension")
        parent_digests = dict(parent.corpus_digests)
        if (
            parent.snapshot_id != bundle.adaptive_snapshot_id
            or parent.literal_manifest_id != bundle.ordered_literal_manifest_id
            or parent.preprocessing_contract_id != bundle.preprocessing_contract_id
            or parent.inference_artifact_id != bundle.deployed_parent_artifact_id
            or parent_digests.get(CorpusRole.PARENT_TRAINING.value)
            != bundle.parent_training_corpus_digest
            or len(parent_digests) != 1
        ):
            raise ModelGenerationError("lineage restoration bundle is inconsistent")
        for generation, manifest in (
            (parent, parent_manifest),
            (extended, extended_manifest),
            (child, child_manifest),
        ):
            preprocessing = self.store.load_preprocessing(
                generation.preprocessing_contract_id
            )
            if tuple(preprocessing.literal_ids) != manifest.literal_ids:
                raise ModelGenerationError(
                    "generation preprocessing differs from its literal manifest"
                )
        self._resolve_restoration_bundle(bundle)
        return parent, extended, child, bundle, audit, evidence

    def _validate_contraction_lineage_graph(
        self, lineage: LiteralContractionLineage
    ) -> tuple[
        ModelGeneration,
        ModelGeneration,
        ModelGeneration,
        AdaptiveRestorationBundle,
        PromotionAuditSnapshot,
        PrologDeescalationEvidence,
    ]:
        if self.store.load_lineage(lineage.lineage_id) != lineage:
            raise ModelGenerationError("lineage differs from its durable object")
        parent = self.store.load_generation(lineage.parent_generation_id)
        contracted = self.store.load_generation(lineage.contracted_generation_id)
        child = self.store.load_generation(lineage.child_generation_id)
        bundle = self.store.load_restoration_bundle(lineage.restoration_bundle_id)
        audit = self.store.load_audit(lineage.promotion_audit_id)
        evidence = self.store.load_deescalation_evidence(
            lineage.deescalation_evidence_id
        )
        usage = self.store.load_evidence_usage(lineage.evidence_usage_id)
        if (
            len(usage.corpora) != 3
            or tuple(corpus.role for corpus in usage.corpora)
            != (
                CorpusRole.DEESCALATION_PROOF,
                CorpusRole.DEESCALATION_CONFIRMATION,
                CorpusRole.PROMOTION,
            )
        ):
            raise ModelGenerationError(
                "literal-contraction evidence roles are inconsistent"
            )
        proof, confirmation, promotion = usage.corpora
        parent_manifest = self.store.load_manifest(parent.literal_manifest_id)
        parent_snapshot = self.store.load_snapshot(parent.snapshot_id).snapshot
        contracted_manifest = self.store.load_manifest(
            contracted.literal_manifest_id
        )
        contracted_snapshot = self.store.load_snapshot(contracted.snapshot_id).snapshot
        child_snapshot, child_manifest, child_preprocessing, child_artifact = (
            self._validate_deployable_generation(child)
        )
        validation = child_artifact.manifest.get("validation")
        signature = (
            validation.get("signature")
            if isinstance(validation, Mapping)
            else None
        )
        restoration_reference = child_artifact.manifest.get(
            "restoration_reference"
        )
        contracted_digests = (
            (CorpusRole.DEESCALATION_PROOF.value, lineage.proof_corpus_digest),
            (
                CorpusRole.DEESCALATION_CONFIRMATION.value,
                lineage.confirmation_corpus_digest,
            ),
        )
        child_digests = contracted_digests + (
            (CorpusRole.PROMOTION.value, lineage.promotion_corpus_digest),
        )
        promotion_conformance = self._validate_promotion_conformance_evidence(
            lineage.promotion_conformance_evidence_id,
            child,
            child_snapshot,
            child_manifest,
            child_artifact,
            promotion,
        )
        if (
            parent.kind is not GenerationKind.TRAINED_PARENT
            or contracted.kind is not GenerationKind.CONTRACTED_PARENT
            or child.kind is not GenerationKind.CONTRACTED_CHILD
            or contracted.parent_generation_id != parent.generation_id
            or child.parent_generation_id != contracted.generation_id
            or contracted.restoration_bundle_id != bundle.bundle_id
            or child.restoration_bundle_id != bundle.bundle_id
            or bundle.parent_generation_id != parent.generation_id
            or contracted.inference_artifact_id is not None
            or contracted.corpus_digests != contracted_digests
            or child.corpus_digests != child_digests
            or any(
                value is not None
                for value in (
                    contracted.origin_proposal_semantic_id,
                    contracted.origin_proposal_provenance_id,
                    child.origin_proposal_semantic_id,
                    child.origin_proposal_provenance_id,
                )
            )
            or usage.purpose is not EvidenceUsagePurpose.DEESCALATION_EPISODE
            or usage.subject_generation_id != parent.generation_id
            or tuple((corpus.role.value, corpus.digest) for corpus in usage.corpora)
            != child_digests
            or evidence.proof_corpus_digest != lineage.proof_corpus_digest
            or proof.digest != lineage.proof_corpus_digest
            or confirmation.digest != lineage.confirmation_corpus_digest
            or promotion.digest != lineage.promotion_corpus_digest
            or evidence.parent_snapshot_id != parent.snapshot_id
            or evidence.parent_manifest_id != parent.literal_manifest_id
            or evidence.surviving_literal_id != lineage.surviving_literal_id
            or evidence.removed_literal_id != lineage.removed_literal_id
            or contracted.snapshot_id != child.snapshot_id
            or contracted.literal_manifest_id != child.literal_manifest_id
            or contracted.preprocessing_contract_id
            != child.preprocessing_contract_id
            or contracted_snapshot != child_snapshot.snapshot
            or contracted_manifest != child_manifest
            or tuple(child_preprocessing.literal_ids) != child_manifest.literal_ids
            or audit.corpus_role is not CorpusRole.PROMOTION
            or audit.corpus_digest != lineage.promotion_corpus_digest
            or audit.conformance.artifact_id != child.inference_artifact_id
            or not audit.conformance.exact
            or audit.conformance != promotion_conformance
            or lineage.adaptive_behavior_id
            != AdaptiveBehaviorIdentity.from_generation(child).behavior_id
            or not isinstance(signature, Mapping)
            or signature.get("generation_stage") != "contracted_child"
            or signature.get("adaptive_behavior_id")
            != lineage.adaptive_behavior_id
            or signature.get("adaptive_snapshot_id") != child.snapshot_id
            or signature.get("ordered_literal_manifest_id")
            != child.literal_manifest_id
            or signature.get("deescalation_proof_corpus_digest")
            != lineage.proof_corpus_digest
            or signature.get("deescalation_confirmation_corpus_digest")
            != lineage.confirmation_corpus_digest
            or signature.get("promotion_corpus_digest")
            != lineage.promotion_corpus_digest
            or signature.get("deescalation_evidence_id")
            != evidence.evidence_id
            or signature.get("surviving_literal_id")
            != str(lineage.surviving_literal_id)
            or signature.get("removed_literal_id")
            != str(lineage.removed_literal_id)
            or restoration_reference != bundle.to_dict()
        ):
            raise ModelGenerationError(
                "literal-contraction lineage object graph is inconsistent"
            )

        _validate_literal_contraction_derivation(
            evidence,
            proof,
            parent_snapshot,
            parent_manifest,
        )
        reconstructed = contract_parent_with_equivalent_literal(
            parent_snapshot,
            parent_manifest,
            evidence,
            proof_records=proof.records,
            confirmation_records=confirmation.records,
        )
        if (
            reconstructed.snapshot.snapshot_id != contracted.snapshot_id
            or reconstructed.manifest.manifest_id
            != contracted.literal_manifest_id
            or reconstructed.snapshot.snapshot != contracted_snapshot
            or reconstructed.manifest != contracted_manifest
            or reconstructed.proof_case_count != len(proof.examples)
            or reconstructed.confirmation_case_count
            != len(confirmation.examples)
        ):
            raise ModelGenerationError(
                "literal-contraction construction cannot be reconstructed"
            )
        reconstructed_audit = audit_parent_child_snapshots(
            parent_snapshot,
            parent_manifest,
            child_snapshot,
            child_manifest,
            promotion,
            promotion_conformance,
            PromotionAuditPolicy(
                minimum_observations=audit.observations,
                require_strict_improvement=False,
                maximum_regressions=0,
            ),
        )
        if reconstructed_audit != audit:
            raise ModelGenerationError(
                "literal-contraction promotion audit cannot be reconstructed"
            )
        contracted_preprocessing = self.store.load_preprocessing(
            contracted.preprocessing_contract_id
        )
        if (
            tuple(contracted_preprocessing.literal_ids)
            != contracted_manifest.literal_ids
            or preprocessing_contract_id(contracted_preprocessing)
            != contracted.preprocessing_contract_id
        ):
            raise ModelGenerationError(
                "contracted preprocessing differs from its literal manifest"
            )
        parent_digests = dict(parent.corpus_digests)
        if (
            parent.snapshot_id != bundle.adaptive_snapshot_id
            or parent.literal_manifest_id != bundle.ordered_literal_manifest_id
            or parent.preprocessing_contract_id != bundle.preprocessing_contract_id
            or parent.inference_artifact_id != bundle.deployed_parent_artifact_id
            or parent_digests.get(CorpusRole.PARENT_TRAINING.value)
            != bundle.parent_training_corpus_digest
            or len(parent_digests) != 1
        ):
            raise ModelGenerationError(
                "literal-contraction restoration bundle is inconsistent"
            )
        self._resolve_restoration_bundle(bundle)
        return parent, contracted, child, bundle, audit, evidence

    def _validate_live_conformance_evidence(
        self,
        lineage: LifecycleLineage,
        drift: PromotionAuditSnapshot,
        evidence: LiveRuntimeConformanceEvidence,
    ) -> None:
        if self.store.load_live_conformance(evidence.evidence_id) != evidence:
            raise ModelGenerationError("live conformance evidence changed after publication")
        parent, _, child, _, _, _ = self._validate_lineage_graph(lineage)
        child_snapshot, child_manifest, _, artifact = (
            self._validate_deployable_generation(child)
        )
        if (
            evidence.child_generation_id != child.generation_id
            or evidence.artifact_id != child.inference_artifact_id
            or evidence.snapshot_id != child.snapshot_id
            or evidence.literal_manifest_id != child.literal_manifest_id
            or evidence.corpus.digest != drift.corpus_digest
            or len(evidence.corpus.examples) != drift.observations
        ):
            raise ModelGenerationError("live conformance identity binding is inconsistent")
        batch = child_manifest.build_catalog().encode(evidence.corpus.records).ta
        rows = tuple(
            batch.row_values(index) for index in range(batch.row_count)
        )
        machine = ScalarBinaryTsetlinMachine(
            child_snapshot.snapshot.number_of_clauses,
            child_snapshot.snapshot.number_of_features,
            states_per_action=child_snapshot.snapshot.states_per_action,
            specificity=child_snapshot.snapshot.specificity,
            threshold=child_snapshot.snapshot.threshold,
            seed=0,
        )
        machine.restore(child_snapshot.snapshot)
        scalar_scores = tuple(machine.score(row) for row in rows)
        scalar_predictions = tuple(int(score > 0) for score in scalar_scores)
        packed_predictions = artifact.predict_records(evidence.corpus.records)
        if (
            rows != evidence.scalar_features
            or scalar_scores != evidence.scalar_scores
            or scalar_predictions != evidence.scalar_predictions
            or packed_predictions != evidence.packed_predictions
            or tuple(tuple(int(value) for value in row) for row in rows)
            != evidence.native_features
            or scalar_scores != evidence.native_scores
            or scalar_predictions != evidence.native_predictions
        ):
            raise ModelGenerationError("live conformance evidence cannot be reconstructed")
        ptmrt_executable = self._bind_ptmrt_executable()
        executable_digest = _file_digest(
            ptmrt_executable,
            maximum_bytes=_MAX_ATTESTED_EXECUTABLE_BYTES,
        )
        if executable_digest != evidence.ptmrt_binary_digest:
            raise ModelGenerationError(
                "live conformance ptmrt executable digest does not match"
            )
        native = _verify_snapshot_records_with_ptmrt(
            ptmrt_executable,
            self.store.artifact_path(artifact.artifact_id),
            child_snapshot,
            child_manifest,
            artifact,
            evidence.corpus.records,
        )
        if (
            _file_digest(
                ptmrt_executable,
                maximum_bytes=_MAX_ATTESTED_EXECUTABLE_BYTES,
            )
            != evidence.ptmrt_binary_digest
            or native.artifact_id != evidence.artifact_id
            or native.scalar_features != evidence.scalar_features
            or native.scalar_scores != evidence.scalar_scores
            or native.scalar_predictions != evidence.scalar_predictions
            or native.packed_predictions != evidence.packed_predictions
            or native.native_features != evidence.native_features
            or native.native_scores != evidence.native_scores
            or native.native_predictions != evidence.native_predictions
        ):
            raise ModelGenerationError(
                "live conformance native execution cannot be reproduced"
            )
        conformance = RuntimeConformanceReport(
            artifact.artifact_id,
            len(rows),
            0,
            True,
            artifact.artifact_id,
        )
        parent_snapshot = self.store.load_snapshot(parent.snapshot_id).snapshot
        parent_manifest = self.store.load_manifest(parent.literal_manifest_id)
        reconstructed = audit_parent_child_snapshots(
            parent_snapshot,
            parent_manifest,
            child_snapshot,
            child_manifest,
            evidence.corpus,
            conformance,
            PromotionAuditPolicy(minimum_observations=1),
        )
        if reconstructed != drift:
            raise ModelGenerationError("live drift audit cannot be reconstructed")

    def _replay_lifecycle_state(self) -> _LifecycleReplayState:
        """Replay every durable transition into the authoritative semantic state."""

        events = self.store.read_events()
        if not events:
            return _LifecycleReplayState(
                events=(),
                active_generation_id=None,
                registered_dataset_id=None,
                pending_candidate_usage=None,
                candidate=None,
                approved=None,
                activated=None,
                pending_live_usage=None,
                reopen=None,
                activation_count=0,
                last_activated_lineage_id=None,
                activated_behavior_ids=frozenset(),
                spent_example_keys=frozenset(),
                spent_record_fingerprints=frozenset(),
            )
        active: str | None = None
        candidate: LifecycleLineage | None = None
        approved: LifecycleLineage | None = None
        activated: LifecycleLineage | None = None
        pending_candidate_usage: EvidenceUsage | None = None
        pending_live_usage: EvidenceUsage | None = None
        reopen: tuple[
            LifecycleLineage,
            AdaptiveRestorationBundle,
            PromotionAuditSnapshot,
        ] | None = None
        activated_behaviors: set[str] = set()
        activation_count = 0
        last_activated_lineage_id: str | None = None
        registered_dataset_id: str | None = None
        used_example_keys: set[tuple[str, str, str | int]] = set()
        used_record_fingerprints: set[str] = set()
        graph_cache: dict[
            str,
            tuple[
                ModelGeneration,
                ModelGeneration,
                ModelGeneration,
                AdaptiveRestorationBundle,
                PromotionAuditSnapshot,
                PrologInventionEvidence
                | PrologThresholdCandidateSet
                | PrologDeescalationEvidence,
            ],
        ] = {}

        def graph(lineage_id: object):
            if type(lineage_id) is not str:
                raise ModelGenerationError("lifecycle event lacks a lineage identity")
            if lineage_id not in graph_cache:
                lineage = self.store.load_lineage(lineage_id)
                graph_cache[lineage_id] = self._validate_lineage_graph(lineage)
            return self.store.load_lineage(lineage_id), graph_cache[lineage_id]

        def consume_usage(usage: EvidenceUsage) -> None:
            if used_example_keys & usage.example_keys:
                raise ModelGenerationError(
                    "lifecycle evidence reuses an observation identity"
                )
            if used_record_fingerprints & usage.record_fingerprints:
                raise ModelGenerationError(
                    "lifecycle evidence reuses a labeled-record fingerprint"
                )
            used_example_keys.update(usage.example_keys)
            used_record_fingerprints.update(usage.record_fingerprints)

        for index, event in enumerate(events):
            details = dict(event.details)
            if event.kind is LifecycleEventKind.PARENT_REGISTERED:
                usage_id = details.get("evidence_usage_id")
                if index != 0 or active is not None or type(usage_id) is not str:
                    raise ModelGenerationError("invalid parent registration transition")
                parent = self.store.load_generation(event.generation_id)
                usage = self.store.load_evidence_usage(usage_id)
                expected = {"evidence_usage_id": usage.usage_id}
                if parent.kind is not GenerationKind.TRAINED_PARENT:
                    raise ModelGenerationError("registered route is not a trained parent")
                parent_digests = dict(parent.corpus_digests)
                if (
                    details != expected
                    or usage.purpose
                    is not EvidenceUsagePurpose.PARENT_REGISTRATION
                    or usage.subject_generation_id != parent.generation_id
                    or len(usage.corpora) != 1
                    or usage.corpora[0].digest
                    != parent_digests.get(CorpusRole.PARENT_TRAINING.value)
                ):
                    raise ModelGenerationError(
                        "registered parent evidence usage is inconsistent"
                    )
                consume_usage(usage)
                self._validate_deployable_generation(parent)
                active = parent.generation_id
                registered_dataset_id = usage.dataset_id
                continue

            if event.kind is LifecycleEventKind.EVIDENCE_RESERVED:
                usage_id = details.get("evidence_usage_id")
                if type(usage_id) is not str:
                    raise ModelGenerationError("evidence reservation lacks an identity")
                usage = self.store.load_evidence_usage(usage_id)
                expected = {
                    "evidence_usage_id": usage.usage_id,
                    "purpose": usage.purpose.value,
                    "dataset_id": usage.dataset_id,
                }
                if (
                    details != expected
                    or active is None
                    or event.generation_id != active
                    or usage.subject_generation_id != active
                    or usage.dataset_id != registered_dataset_id
                    or candidate is not None
                    or approved is not None
                    or reopen is not None
                ):
                    raise ModelGenerationError("invalid evidence-reserved transition")
                if usage.purpose in _CANDIDATE_EVIDENCE_PURPOSES:
                    subject = self.store.load_generation(active)
                    if (
                        subject.kind is not GenerationKind.TRAINED_PARENT
                        or activated is not None
                        or pending_candidate_usage is not None
                    ):
                        raise ModelGenerationError(
                            "candidate evidence lacks an active trained parent"
                        )
                    pending_candidate_usage = usage
                elif usage.purpose is EvidenceUsagePurpose.LIVE_DRIFT:
                    subject = self.store.load_generation(active)
                    if (
                        subject.kind
                        not in (
                            GenerationKind.ADAPTED_CHILD,
                            GenerationKind.CONTRACTED_CHILD,
                        )
                        or activated is None
                        or pending_live_usage is not None
                    ):
                        raise ModelGenerationError(
                            "live evidence lacks an active child"
                        )
                    pending_live_usage = usage
                else:
                    raise ModelGenerationError(
                        "parent evidence cannot use a reservation event"
                    )
                consume_usage(usage)
                continue

            if event.kind is LifecycleEventKind.EVIDENCE_ABANDONED:
                usage_id = details.get("evidence_usage_id")
                purpose = details.get("purpose")
                if type(usage_id) is not str or type(purpose) is not str:
                    raise ModelGenerationError(
                        "evidence abandonment lacks durable identity"
                    )
                usage = self.store.load_evidence_usage(usage_id)
                expected = {
                    "evidence_usage_id": usage.usage_id,
                    "purpose": usage.purpose.value,
                }
                if (
                    details != expected
                    or active is None
                    or event.generation_id != active
                    or usage.subject_generation_id != active
                    or purpose != usage.purpose.value
                    or candidate is not None
                    or approved is not None
                    or reopen is not None
                ):
                    raise ModelGenerationError(
                        "invalid evidence-abandoned transition"
                    )
                if usage.purpose in _CANDIDATE_EVIDENCE_PURPOSES:
                    if pending_candidate_usage != usage or activated is not None:
                        raise ModelGenerationError(
                            "candidate evidence abandonment lacks a pending reservation"
                        )
                    pending_candidate_usage = None
                elif usage.purpose is EvidenceUsagePurpose.LIVE_DRIFT:
                    if pending_live_usage != usage or activated is None:
                        raise ModelGenerationError(
                            "live evidence abandonment lacks a pending reservation"
                        )
                    pending_live_usage = None
                else:
                    raise ModelGenerationError(
                        "parent registration evidence cannot be abandoned"
                    )
                continue

            if event.kind is LifecycleEventKind.CANDIDATE_CREATED:
                lineage, values = graph(details.get("lineage_id"))
                parent, extended, child, _, _, _ = values
                expected = {
                    "lineage_id": lineage.lineage_id,
                    "parent_generation_id": parent.generation_id,
                    "extended_generation_id": extended.generation_id,
                    "evidence_usage_id": lineage.evidence_usage_id,
                    "activation_sequence": lineage.activation_sequence,
                }
                if (
                    active != parent.generation_id
                    or candidate is not None
                    or approved is not None
                    or activated is not None
                    or reopen is not None
                    or event.generation_id != child.generation_id
                    or details != expected
                    or lineage.adaptive_behavior_id in activated_behaviors
                    or pending_candidate_usage is None
                    or pending_candidate_usage.usage_id
                    != lineage.evidence_usage_id
                    or lineage.activation_sequence != activation_count + 1
                    or lineage.previous_activated_lineage_id
                    != last_activated_lineage_id
                ):
                    raise ModelGenerationError("invalid candidate-created transition")
                candidate = lineage
                pending_candidate_usage = None
                continue

            if event.kind is LifecycleEventKind.CANDIDATE_REJECTED:
                if candidate is None:
                    raise ModelGenerationError("candidate rejection lacks a candidate")
                lineage, values = graph(details.get("lineage_id"))
                _, _, child, _, audit, _ = values
                expected = {
                    "lineage_id": lineage.lineage_id,
                    "audit_id": audit.audit_id,
                    "parent_errors": audit.parent_errors,
                    "child_errors": audit.child_errors,
                    "improvements": audit.improvements,
                    "regressions": audit.regressions,
                }
                if (
                    lineage != candidate
                    or event.generation_id != child.generation_id
                    or audit.accepted
                    or details != expected
                    or lineage.activation_sequence != activation_count + 1
                    or lineage.previous_activated_lineage_id
                    != last_activated_lineage_id
                ):
                    raise ModelGenerationError("invalid candidate-rejected transition")
                candidate = None
                continue

            if event.kind is LifecycleEventKind.PROMOTION_APPROVED:
                if candidate is None:
                    raise ModelGenerationError("promotion approval lacks a candidate")
                lineage, values = graph(details.get("lineage_id"))
                _, _, child, _, audit, _ = values
                expected = {
                    "lineage_id": lineage.lineage_id,
                    "audit_id": audit.audit_id,
                }
                if (
                    lineage != candidate
                    or event.generation_id != child.generation_id
                    or not audit.accepted
                    or not audit.conformance.exact
                    or details != expected
                ):
                    raise ModelGenerationError("invalid promotion-approved transition")
                candidate = None
                approved = lineage
                continue

            if event.kind is LifecycleEventKind.ACTIVATED:
                if approved is None:
                    raise ModelGenerationError("activation lacks promotion approval")
                lineage, values = graph(details.get("lineage_id"))
                parent, _, child, _, audit, _ = values
                expected = {
                    "previous_generation_id": parent.generation_id,
                    "lineage_id": lineage.lineage_id,
                    "adaptive_behavior_id": lineage.adaptive_behavior_id,
                    "audit_id": audit.audit_id,
                }
                if (
                    lineage != approved
                    or active != parent.generation_id
                    or event.generation_id != child.generation_id
                    or not audit.accepted
                    or not audit.conformance.exact
                    or lineage.adaptive_behavior_id in activated_behaviors
                    or lineage.activation_sequence != activation_count + 1
                    or lineage.previous_activated_lineage_id
                    != last_activated_lineage_id
                    or details != expected
                ):
                    raise ModelGenerationError("invalid activated transition")
                active = child.generation_id
                activated_behaviors.add(lineage.adaptive_behavior_id)
                activation_count += 1
                last_activated_lineage_id = lineage.lineage_id
                approved = None
                activated = lineage
                continue

            if event.kind is LifecycleEventKind.REOPEN_REQUESTED:
                if activated is None or reopen is not None:
                    raise ModelGenerationError("reopen request lacks an active child")
                lineage, values = graph(details.get("lineage_id"))
                _, _, child, bundle, _, _ = values
                drift_id = details.get("drift_audit_id")
                live_evidence_id = details.get("live_conformance_evidence_id")
                evidence_usage_id = details.get("evidence_usage_id")
                raw_policy = details.get("drift_policy")
                if (
                    type(drift_id) is not str
                    or type(live_evidence_id) is not str
                    or type(evidence_usage_id) is not str
                    or not isinstance(raw_policy, Mapping)
                ):
                    raise ModelGenerationError("reopen request lacks durable evidence")
                drift = self.store.load_audit(drift_id)
                live_evidence = self.store.load_live_conformance(live_evidence_id)
                policy = DriftAuditPolicy.from_dict(raw_policy)
                self._validate_live_conformance_evidence(
                    lineage, drift, live_evidence
                )
                expected = {
                    "drift_audit_id": drift.audit_id,
                    "lineage_id": lineage.lineage_id,
                    "restoration_bundle_id": bundle.bundle_id,
                    "parent_errors": drift.parent_errors,
                    "child_errors": drift.child_errors,
                    "drift_policy": policy.to_dict(),
                    "live_conformance_evidence_id": live_evidence.evidence_id,
                    "evidence_usage_id": (
                        pending_live_usage.usage_id
                        if pending_live_usage is not None
                        else None
                    ),
                }
                if (
                    lineage != activated
                    or active != child.generation_id
                    or event.generation_id != child.generation_id
                    or drift.corpus_role is not CorpusRole.LIVE
                    or not drift.conformance.exact
                    or drift.conformance.artifact_id != child.inference_artifact_id
                    or not drift_requires_reopen(drift, policy)
                    or pending_live_usage is None
                    or len(pending_live_usage.corpora) != 1
                    or pending_live_usage.corpora[0] != live_evidence.corpus
                    or details != expected
                ):
                    raise ModelGenerationError("invalid reopen-requested transition")
                reopen = lineage, bundle, drift
                pending_live_usage = None
                continue

            if event.kind is LifecycleEventKind.PARENT_RESTORED:
                if reopen is None or activated is None:
                    raise ModelGenerationError("parent restoration lacks a reopen request")
                lineage, bundle, drift = reopen
                expected = {
                    "restoration_bundle_id": bundle.bundle_id,
                    "previous_generation_id": active,
                    "lineage_id": lineage.lineage_id,
                    "drift_audit_id": drift.audit_id,
                }
                if (
                    event.generation_id != bundle.parent_generation_id
                    or details != expected
                ):
                    raise ModelGenerationError("invalid parent-restored transition")
                self._resolve_restoration_bundle(bundle)
                active = bundle.parent_generation_id
                activated = None
                reopen = None
                continue

            raise ModelGenerationError("unsupported lifecycle event transition")

        if active is None:
            raise ModelGenerationError("lifecycle history has no active parent")
        return _LifecycleReplayState(
            events=events,
            active_generation_id=active,
            registered_dataset_id=registered_dataset_id,
            pending_candidate_usage=pending_candidate_usage,
            candidate=candidate,
            approved=approved,
            activated=activated,
            pending_live_usage=pending_live_usage,
            reopen=reopen,
            activation_count=activation_count,
            last_activated_lineage_id=last_activated_lineage_id,
            activated_behavior_ids=frozenset(activated_behaviors),
            spent_example_keys=frozenset(used_example_keys),
            spent_record_fingerprints=frozenset(used_record_fingerprints),
        )

    def _resolve_restoration_bundle(
        self, bundle: AdaptiveRestorationBundle
    ) -> RestoredAdaptiveParent:
        stored_bundle = self.store.load_restoration_bundle(bundle.bundle_id)
        if stored_bundle != bundle:
            raise ModelGenerationError("restoration bundle changed after publication")
        parent = self.store.load_generation(bundle.parent_generation_id)
        if parent.kind is not GenerationKind.TRAINED_PARENT:
            raise ModelGenerationError("restoration target is not a trained parent")
        parent_corpora = dict(parent.corpus_digests)
        if (
            len(parent_corpora) != 1
            or parent.snapshot_id != bundle.adaptive_snapshot_id
            or parent.literal_manifest_id != bundle.ordered_literal_manifest_id
            or parent.preprocessing_contract_id != bundle.preprocessing_contract_id
            or parent.inference_artifact_id != bundle.deployed_parent_artifact_id
            or parent_corpora.get(CorpusRole.PARENT_TRAINING.value)
            != bundle.parent_training_corpus_digest
        ):
            raise ModelGenerationError("restoration bundle differs from its parent generation")
        snapshot, manifest, preprocessing, artifact = (
            self._validate_deployable_generation(parent)
        )
        if snapshot.snapshot.number_of_features != len(manifest.literals):
            raise ModelGenerationError("restoration snapshot and manifest widths differ")
        if tuple(preprocessing.literal_ids) != manifest.literal_ids:
            raise ModelGenerationError("restoration preprocessing order differs from manifest")
        validation = artifact.manifest.get("validation")
        signature = validation.get("signature") if isinstance(validation, Mapping) else None
        if not isinstance(signature, Mapping) or (
            signature.get("adaptive_snapshot_id") != bundle.adaptive_snapshot_id
            or signature.get("ordered_literal_manifest_id")
            != bundle.ordered_literal_manifest_id
            or signature.get("training_corpus_digest")
            != bundle.parent_training_corpus_digest
        ):
            raise ModelGenerationError("parent artifact is not bound to the restoration state")
        machine = ScalarBinaryTsetlinMachine(
            snapshot.snapshot.number_of_clauses,
            snapshot.snapshot.number_of_features,
            states_per_action=snapshot.snapshot.states_per_action,
            specificity=snapshot.snapshot.specificity,
            threshold=snapshot.snapshot.threshold,
            seed=0,
        )
        machine.restore(snapshot.snapshot)
        return RestoredAdaptiveParent(
            bundle.parent_generation_id,
            snapshot,
            manifest,
            preprocessing,
            artifact,
            machine,
        )

    def register_parent(
        self,
        generation: ModelGeneration,
        parent_training_corpus: LabeledCorpus,
    ) -> None:
        if generation.kind is not GenerationKind.TRAINED_PARENT:
            raise ModelGenerationError("initial active generation must be a trained parent")
        if parent_training_corpus.role is not CorpusRole.PARENT_TRAINING:
            raise ModelGenerationError("parent registration evidence has the wrong role")
        usage = EvidenceUsage(
            EvidenceUsagePurpose.PARENT_REGISTRATION,
            generation.generation_id,
            (parent_training_corpus,),
        )
        if dict(generation.corpus_digests) != {
            CorpusRole.PARENT_TRAINING.value: parent_training_corpus.digest
        }:
            raise ModelGenerationError(
                "parent registration evidence differs from the generation"
            )
        self._validate_deployable_generation(generation)
        with self._control_lock, self.store._event_lock:
            state = self._authoritative_state_under_event_lock()
            if state.active_generation_id is not None or state.events:
                raise ModelGenerationError("a trained parent is already registered")
            self.store.put_evidence_usage(usage)
            self.store.append_event(
                LifecycleEventKind.PARENT_REGISTERED,
                generation.generation_id,
                evidence_usage_id=usage.usage_id,
            )
            self._active_generation_id = generation.generation_id
        self._emit("parent_registered", generation_id=generation.generation_id)

    def record_candidate(self, lineage: LifecycleLineage) -> None:
        if lineage.schema not in (LINEAGE_SCHEMA, CONTRACTION_LINEAGE_SCHEMA):
            raise ModelGenerationError(
                "new candidate admission requires the current lineage schema"
            )
        with self._control_lock, self.store._event_lock:
            self._validate_lineage_graph(lineage)
            state = self._authoritative_state_under_event_lock()
            if state.active_generation_id != lineage.parent_generation_id:
                raise ModelGenerationError("candidate parent is not the active generation")
            usage = self.store.load_evidence_usage(lineage.evidence_usage_id)
            if lineage.adaptive_behavior_id in state.activated_behavior_ids:
                raise ModelGenerationError(
                    "candidate adaptive behavior has already been activated"
                )
            if (
                state.pending_candidate_usage != usage
                or state.candidate is not None
                or state.approved is not None
                or state.activated is not None
                or state.reopen is not None
                or usage.subject_generation_id != lineage.parent_generation_id
                or lineage.activation_sequence != state.activation_count + 1
                or lineage.previous_activated_lineage_id
                != state.last_activated_lineage_id
            ):
                raise ModelGenerationError(
                    "candidate creation lacks matching reserved evidence"
                )
            self.store.append_event(
                LifecycleEventKind.CANDIDATE_CREATED,
                lineage.child_generation_id,
                lineage_id=lineage.lineage_id,
                parent_generation_id=lineage.parent_generation_id,
                extended_generation_id=lineage.extended_generation_id,
                evidence_usage_id=usage.usage_id,
                activation_sequence=lineage.activation_sequence,
            )
        self._emit(
            "proposal_created",
            generation_id=lineage.child_generation_id,
            lineage_id=lineage.lineage_id,
        )

    def reject_candidate(
        self, generation_id: str, audit: PromotionAuditSnapshot
    ) -> None:
        with self._control_lock, self.store._event_lock:
            if audit.accepted:
                raise ModelGenerationError("an accepted promotion audit cannot be rejected")
            if self.store.load_audit(audit.audit_id) != audit:
                raise ModelGenerationError("rejection audit differs from durable evidence")
            state = self._authoritative_state_under_event_lock()
            if state.candidate is None:
                raise ModelGenerationError("candidate rejection is invalid in the current state")
            lineage = state.candidate
            *_, stored_audit, _ = self._validate_lineage_graph(lineage)
            if (
                state.active_generation_id != lineage.parent_generation_id
                or lineage.child_generation_id != generation_id
                or lineage.promotion_audit_id != audit.audit_id
                or stored_audit != audit
            ):
                raise ModelGenerationError("candidate rejection differs from its durable state")
            self.store.append_event(
                LifecycleEventKind.CANDIDATE_REJECTED,
                generation_id,
                lineage_id=lineage.lineage_id,
                audit_id=audit.audit_id,
                parent_errors=audit.parent_errors,
                child_errors=audit.child_errors,
                improvements=audit.improvements,
                regressions=audit.regressions,
            )
        self._emit(
            "proposal_rejected",
            generation_id=generation_id,
            audit_id=audit.audit_id,
        )

    def approve_promotion(
        self, lineage: LifecycleLineage, audit: PromotionAuditSnapshot
    ) -> None:
        if not audit.accepted or not audit.conformance.exact:
            raise ModelGenerationError("only an accepted exact audit may approve promotion")
        if lineage.promotion_audit_id != audit.audit_id:
            raise ModelGenerationError("lineage references a different promotion audit")
        with self._control_lock, self.store._event_lock:
            *_, stored_audit, _ = self._validate_lineage_graph(lineage)
            if stored_audit != audit:
                raise ModelGenerationError("promotion audit differs from durable audit")
            state = self._authoritative_state_under_event_lock()
            if (
                state.active_generation_id != lineage.parent_generation_id
                or state.candidate != lineage
            ):
                raise ModelGenerationError("promotion approval is invalid in the current state")
            self.store.append_event(
                LifecycleEventKind.PROMOTION_APPROVED,
                lineage.child_generation_id,
                lineage_id=lineage.lineage_id,
                audit_id=audit.audit_id,
            )
        self._emit(
            "shadow_completed",
            generation_id=lineage.child_generation_id,
            audit_id=audit.audit_id,
            accepted=True,
        )

    def activate_child(
        self, lineage: LifecycleLineage, audit: PromotionAuditSnapshot
    ) -> None:
        if not audit.accepted or lineage.promotion_audit_id != audit.audit_id:
            raise ModelGenerationError("child activation lacks an accepted promotion audit")
        with self._control_lock, self.store._event_lock:
            *_, stored_audit, _ = self._validate_lineage_graph(lineage)
            if stored_audit != audit:
                raise ModelGenerationError("activation audit differs from durable audit")
            state = self._authoritative_state_under_event_lock()
            if state.approved != lineage:
                raise ModelGenerationError("activation lacks the durable promotion decision")
            previous = state.active_generation_id
            if previous != lineage.parent_generation_id:
                raise ModelGenerationError("active generation is not the lineage parent")
            self.store.append_event(
                LifecycleEventKind.ACTIVATED,
                lineage.child_generation_id,
                previous_generation_id=previous,
                lineage_id=lineage.lineage_id,
                adaptive_behavior_id=lineage.adaptive_behavior_id,
                audit_id=audit.audit_id,
            )
            self._active_generation_id = lineage.child_generation_id
        self._emit(
            "artifact_published",
            generation_id=lineage.child_generation_id,
            lineage_id=lineage.lineage_id,
        )
        self._emit("activated", generation_id=lineage.child_generation_id)

    def request_reopen(
        self,
        child_generation_id: str,
        live_corpus: LabeledCorpus,
        policy: DriftAuditPolicy,
        ptmrt_executable: str | Path,
    ) -> PromotionAuditSnapshot:
        if live_corpus.role is not CorpusRole.LIVE:
            raise ModelGenerationError("reopen evaluation requires the live/drift corpus")
        with self._control_lock:
            trusted_ptmrt = self._bind_ptmrt_executable(ptmrt_executable)
            live_usage = EvidenceUsage(
                EvidenceUsagePurpose.LIVE_DRIFT,
                child_generation_id,
                (live_corpus,),
            )
            with self.store._event_lock:
                state = self._authoritative_state_under_event_lock()
                if (
                    state.active_generation_id != child_generation_id
                    or state.activated is None
                ):
                    raise ModelGenerationError(
                        "reopen lacks the active child's durable activation"
                    )
                state = self._abandon_pending_for_purpose_under_event_lock(
                    state, EvidenceUsagePurpose.LIVE_DRIFT
                )
                lineage = state.activated
                if lineage is None:
                    raise ModelGenerationError(
                        "reopen lacks the active child's durable activation"
                    )
                parent, _, child, bundle, _, _ = self._validate_lineage_graph(
                    lineage
                )
                if child.generation_id != child_generation_id:
                    raise ModelGenerationError(
                        "active lineage names a different child"
                    )
                if live_corpus.dataset_id != state.registered_dataset_id:
                    raise ModelGenerationError(
                        "live/drift corpus belongs to a different dataset"
                    )
                self._reserve_evidence_under_event_lock(live_usage, state)
            try:
                child_snapshot, child_manifest, _, child_artifact = (
                    self._validate_deployable_generation(child)
                )
                parent_snapshot = self.store.load_snapshot(parent.snapshot_id).snapshot
                parent_manifest = self.store.load_manifest(parent.literal_manifest_id)
                vectors = _verify_snapshot_records_with_ptmrt(
                    trusted_ptmrt,
                    self.store.artifact_path(child_artifact.artifact_id),
                    child_snapshot,
                    child_manifest,
                    child_artifact,
                    live_corpus.records,
                )
                live_evidence = LiveRuntimeConformanceEvidence(
                    child.generation_id,
                    child_artifact.artifact_id,
                    child.snapshot_id,
                    child.literal_manifest_id,
                    live_corpus,
                    vectors.scalar_features,
                    vectors.scalar_scores,
                    vectors.scalar_predictions,
                    vectors.packed_predictions,
                    vectors.native_features,
                    vectors.native_scores,
                    vectors.native_predictions,
                    _file_digest(
                        trusted_ptmrt,
                        maximum_bytes=_MAX_ATTESTED_EXECUTABLE_BYTES,
                    ),
                )
                conformance = audit_snapshot_runtime_conformance(
                    child_snapshot,
                    child_manifest,
                    child_artifact,
                    live_corpus.records,
                    ptmrt_verified=True,
                    ptmrt_artifact_id=vectors.artifact_id,
                )
                drift = audit_parent_child_snapshots(
                    parent_snapshot,
                    parent_manifest,
                    child_snapshot,
                    child_manifest,
                    live_corpus,
                    conformance,
                    PromotionAuditPolicy(minimum_observations=1),
                )
                if (
                    not drift.conformance.exact
                    or not drift_requires_reopen(drift, policy)
                ):
                    raise ModelGenerationError("labeled drift does not justify reopen")
                self.store.put_live_conformance(live_evidence)
                self.store.put_audit(drift)
                with self.store._event_lock:
                    commit_state = self._authoritative_state_under_event_lock()
                    if (
                        commit_state.active_generation_id != child_generation_id
                        or commit_state.activated != lineage
                        or commit_state.pending_live_usage != live_usage
                        or commit_state.reopen is not None
                    ):
                        raise ModelGenerationError(
                            "live evidence is no longer the pending reopen authorization"
                        )
                    self.store.append_event(
                        LifecycleEventKind.REOPEN_REQUESTED,
                        child_generation_id,
                        drift_audit_id=drift.audit_id,
                        lineage_id=lineage.lineage_id,
                        restoration_bundle_id=bundle.bundle_id,
                        parent_errors=drift.parent_errors,
                        child_errors=drift.child_errors,
                        drift_policy=policy.to_dict(),
                        live_conformance_evidence_id=live_evidence.evidence_id,
                        evidence_usage_id=live_usage.usage_id,
                    )
            except BaseException as error:
                try:
                    self._abandon_evidence_if_pending(live_usage)
                except Exception as abandonment_error:
                    error.add_note(
                        "durable evidence abandonment also failed: "
                        f"{abandonment_error}"
                    )
                raise
        self._emit(
            "reopen_requested",
            generation_id=child_generation_id,
            drift_audit_id=drift.audit_id,
        )
        return drift

    def restore_parent(
        self, bundle: AdaptiveRestorationBundle
    ) -> RestoredAdaptiveParent:
        with self._control_lock, self.store._event_lock:
            state = self._authoritative_state_under_event_lock()
            active_child_id = state.active_generation_id
            if active_child_id is None or state.reopen is None:
                raise ModelGenerationError(
                    "restoration lacks a matching reopen request"
                )
            lineage, graph_bundle, drift = state.reopen
            if graph_bundle != bundle:
                raise ModelGenerationError(
                    "reopen evidence does not authorize restoration"
                )
            restored = self._resolve_restoration_bundle(bundle)
            self.store.append_event(
                LifecycleEventKind.PARENT_RESTORED,
                bundle.parent_generation_id,
                restoration_bundle_id=bundle.bundle_id,
                previous_generation_id=active_child_id,
                lineage_id=lineage.lineage_id,
                drift_audit_id=drift.audit_id,
            )
            self._active_generation_id = bundle.parent_generation_id
        self._emit(
            "artifact_reopened",
            generation_id=bundle.parent_generation_id,
            restoration_bundle_id=bundle.bundle_id,
        )
        return restored


def compile_generation_artifact(
    snapshot: TMSnapshot,
    manifest: OrderedLiteralManifest,
    *,
    name: str,
    validation_records: Sequence[Mapping[str, object]],
    validation_signature: Mapping[str, object],
    restoration_reference: Mapping[str, object] | None = None,
) -> tuple[PreprocessingContract, PackedTMInferenceArtifact]:
    if snapshot.number_of_features != len(manifest.literals):
        raise ModelGenerationError("snapshot and literal manifest widths differ")
    catalog = manifest.build_catalog()
    preprocessing = PreprocessingContract.from_catalog(catalog)
    signature = dict(validation_signature)
    identity_signature = {
        "adaptive_snapshot_id": AdaptiveSnapshotEnvelope(snapshot).snapshot_id,
        "ordered_literal_manifest_id": manifest.manifest_id,
    }
    for key, expected in identity_signature.items():
        if key in signature and signature[key] != expected:
            raise ModelGenerationError(f"validation signature conflicts with {key}")
        signature[key] = expected
    artifact = export_packed_tm(
        snapshot,
        name=name,
        description="Immutable packed inference image for one adaptive PTM model generation.",
        intended_use="trained-parent PTA promotion, shadow audit, and reversible activation",
        limitations="Binary research lifecycle; adaptive continuation requires its restoration bundle.",
        feature_names=tuple(f"literal:{value}" for value in manifest.literal_ids),
        feature_literal_ids=manifest.literal_ids,
        feature_catalog_version=manifest.manifest_id,
        preprocessing=preprocessing,
        validation_records=validation_records,
        validation_signature=signature,
        restoration_reference=restoration_reference,
    )
    return preprocessing, artifact


def _invention_session(corpus: LabeledCorpus) -> PTAReasoningSession:
    session = PTAReasoningSession(corpus.dataset_id)
    for example in corpus.examples:
        for name in sorted(example.record):
            session.add_observation(
                "pta:input", example.example_id, name, example.record[name]
            )
        session.add_example_label(example.example_id, example.label)
    return session


def _run_complete_threshold_collective(
    session: PTAReasoningSession,
    numeric_fields: tuple[str, ...],
    budget: ThresholdCandidateBudget,
    *,
    expected_evidence: PrologThresholdCandidateSet | None = None,
) -> tuple[
    tuple[PTAEscalationProposal, ...],
    tuple[str, str, tuple[tuple[str, str], ...]],
]:
    """Run one stable, complete bounded threshold query and attest its bytes."""

    service = PTACollectiveService()
    before = _collective_attestation(service)
    if expected_evidence is not None and before != (
        expected_evidence.gprolog_version,
        expected_evidence.gprolog_binary_digest,
        expected_evidence.module_digests,
    ):
        raise ModelGenerationError(
            "threshold candidate set names a different GNU Prolog implementation"
        )
    query = PTACollectiveQuery(
        numeric_fields=numeric_fields,
        discover_intervals=False,
        derive_deescalation=False,
        derive_escalation=True,
    )
    result = service.run(
        session,
        query=query,
        budget=PTACollectiveBudget(max_results_per_product=budget.maximum_candidates),
    )
    after = _collective_attestation(service)
    if after != before:
        raise ModelGenerationError("GNU Prolog collective changed during execution")
    proposals = tuple(
        item
        for item in result.proposals
        if item.native_target == "threshold"
        and item.structure.get("field") in numeric_fields
    )
    proposal_counts = result.product_counts["threshold_proposals"]
    insight_counts = result.product_counts["threshold_insights"]
    if (
        proposal_counts.available > budget.maximum_candidates
        or proposal_counts.truncated
        or insight_counts.truncated
    ):
        raise ModelGenerationError(
            "the complete threshold candidate set exceeds its explicit budget"
        )
    if (
        not proposals
        or proposal_counts.available != len(proposals)
        or proposal_counts.emitted != len(proposals)
        or insight_counts.available != len(proposals)
        or insight_counts.emitted != len(proposals)
        or len(proposals) != len(result.proposals)
    ):
        raise ModelGenerationError(
            "the bounded collective returned an incomplete threshold candidate set"
        )
    return tuple(sorted(proposals, key=lambda item: item.semantic_id())), before


def _deescalation_session(
    corpus: LabeledCorpus,
    parent_manifest: OrderedLiteralManifest,
) -> PTAReasoningSession:
    """Encode complete literal truth vectors for one bounded proof corpus."""

    if corpus.role is not CorpusRole.DEESCALATION_PROOF:
        raise ModelGenerationError(
            "de-escalation reasoning requires the proof corpus"
        )
    catalog = parent_manifest.build_catalog()
    batch = catalog.encode(corpus.records).ta
    session = PTAReasoningSession(corpus.dataset_id)
    for example_position, example in enumerate(corpus.examples):
        session.add_example_label(example_position, example.label)
        for literal_position, literal_id in enumerate(parent_manifest.literal_ids):
            session.add_literal_truth(
                literal_id,
                example_position,
                int(batch.bit(example_position, literal_position)),
            )
    return session


def _run_complete_deescalation_collective(
    session: PTAReasoningSession,
    maximum_candidates: int,
    *,
    expected_evidence: PrologDeescalationEvidence | None = None,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[str, str, tuple[tuple[str, str], ...]],
]:
    """Run and attest a complete bounded literal-equivalence derivation."""

    service = PTACollectiveService()
    before = _collective_attestation(service)
    if expected_evidence is not None and before != (
        expected_evidence.gprolog_version,
        expected_evidence.gprolog_binary_digest,
        expected_evidence.module_digests,
    ):
        raise ModelGenerationError(
            "de-escalation evidence names a different GNU Prolog implementation"
        )
    query = PTACollectiveQuery(
        numeric_fields=(),
        discover_thresholds=False,
        discover_intervals=False,
        derive_deescalation=True,
        derive_escalation=False,
    )
    result = service.run(
        session,
        query=query,
        budget=PTACollectiveBudget(max_results_per_product=maximum_candidates),
    )
    after = _collective_attestation(service)
    if after != before:
        raise ModelGenerationError("GNU Prolog collective changed during execution")
    for name in (
        "literal_redundancies",
        "literal_subsumptions",
        "clause_subsumptions",
    ):
        if result.product_counts[name].truncated:
            raise ModelGenerationError(
                "the complete de-escalation result exceeds its explicit budget"
            )
    redundancies = tuple(
        item for item in result.insights if item.kind == "literal_redundant"
    )
    count = result.product_counts["literal_redundancies"]
    if (
        not redundancies
        or count.available != len(redundancies)
        or count.emitted != len(redundancies)
        or len(redundancies) > maximum_candidates
    ):
        raise ModelGenerationError(
            "the bounded collective returned an incomplete literal-equivalence set"
        )
    pairs = tuple(
        sorted(
            {
                tuple(sorted((int(item.evidence[0]), int(item.evidence[1]))))
                for item in redundancies
            }
        )
    )
    if len(pairs) != len(redundancies):
        raise ModelGenerationError(
            "GNU Prolog returned duplicate literal-equivalence products"
        )
    domain = set(session.example_domains)
    vectors: dict[int, dict[int, int]] = {}
    for literal, example, truth in session.literal_truths:
        values = vectors.setdefault(literal, {})
        if example in values:
            raise ModelGenerationError(
                "de-escalation proof contains duplicate literal truth rows"
            )
        values[example] = truth
    if not domain or any(set(vector) != domain for vector in vectors.values()):
        raise ModelGenerationError(
            "de-escalation proof lacks complete Python truth vectors"
        )
    expected_pairs = tuple(
        (left, right)
        for left, right in combinations(sorted(vectors), 2)
        if vectors[left] == vectors[right]
    )
    if pairs != expected_pairs:
        raise ModelGenerationError(
            "GNU Prolog literal-equivalence set differs from independent Python enumeration"
        )
    return pairs, before


@dataclass(frozen=True, slots=True)
class LiteralContractionInvention:
    """In-memory proof session plus its durable De-escalation attestation."""

    session: PTAReasoningSession
    evidence: PrologDeescalationEvidence


def invent_literal_contraction_for_corpus(
    proof_corpus: LabeledCorpus,
    parent_snapshot: TMSnapshot,
    parent_manifest: OrderedLiteralManifest,
    *,
    maximum_candidates: int = MAX_DEESCALATION_CANDIDATES,
) -> LiteralContractionInvention:
    parent_envelope = AdaptiveSnapshotEnvelope(parent_snapshot)
    if parent_snapshot.number_of_features != len(parent_manifest.literals):
        raise ModelGenerationError(
            "de-escalation parent snapshot and manifest widths differ"
        )
    session = _deescalation_session(proof_corpus, parent_manifest)
    pairs, attestation = _run_complete_deescalation_collective(
        session, maximum_candidates
    )
    positions = {
        literal_id: position
        for position, literal_id in enumerate(parent_manifest.literal_ids)
    }
    candidates: list[tuple[int, int, int, int]] = []
    for left, right in pairs:
        try:
            left_position = positions[left]
            right_position = positions[right]
        except KeyError as error:
            raise ModelGenerationError(
                "de-escalation result names a literal outside the parent manifest"
            ) from error
        if left_position < right_position:
            candidates.append((left_position, right_position, left, right))
        else:
            candidates.append((right_position, left_position, right, left))
    _, _, survivor, removed = min(candidates)
    version, binary_digest, module_digests = attestation
    evidence = PrologDeescalationEvidence(
        proof_corpus_digest=proof_corpus.digest,
        session_digest=content_digest(session.to_dict()),
        parent_snapshot_id=parent_envelope.snapshot_id,
        parent_manifest_id=parent_manifest.manifest_id,
        maximum_candidates=maximum_candidates,
        equivalent_pairs=pairs,
        surviving_literal_id=survivor,
        removed_literal_id=removed,
        collective_protocol="PTM_PTA_COLLECTIVE_V1",
        gprolog_version=version,
        gprolog_binary_digest=binary_digest,
        module_digests=module_digests,
    )
    return LiteralContractionInvention(session, evidence)


def _validate_literal_contraction_derivation(
    evidence: PrologDeescalationEvidence,
    proof_corpus: LabeledCorpus,
    parent_snapshot: TMSnapshot,
    parent_manifest: OrderedLiteralManifest,
) -> PTAReasoningSession:
    if (
        evidence.proof_corpus_digest != proof_corpus.digest
        or evidence.parent_snapshot_id
        != AdaptiveSnapshotEnvelope(parent_snapshot).snapshot_id
        or evidence.parent_manifest_id != parent_manifest.manifest_id
    ):
        raise ModelGenerationError(
            "de-escalation evidence differs from its proof inputs"
        )
    session = _deescalation_session(proof_corpus, parent_manifest)
    if content_digest(session.to_dict()) != evidence.session_digest:
        raise ModelGenerationError(
            "de-escalation proof session cannot be reconstructed"
        )
    pairs, _ = _run_complete_deescalation_collective(
        session,
        evidence.maximum_candidates,
        expected_evidence=evidence,
    )
    if pairs != evidence.equivalent_pairs:
        raise ModelGenerationError(
            "de-escalation equivalence products cannot be reconstructed"
        )
    positions = {
        literal_id: position
        for position, literal_id in enumerate(parent_manifest.literal_ids)
    }
    ordered: list[tuple[int, int, int, int]] = []
    for left, right in pairs:
        if left not in positions or right not in positions:
            raise ModelGenerationError(
                "de-escalation result names an unknown parent literal"
            )
        if positions[left] < positions[right]:
            ordered.append((positions[left], positions[right], left, right))
        else:
            ordered.append((positions[right], positions[left], right, left))
    *_, expected_survivor, expected_removed = min(ordered)
    if (
        evidence.surviving_literal_id != expected_survivor
        or evidence.removed_literal_id != expected_removed
    ):
        raise ModelGenerationError(
            "de-escalation selection differs from the deterministic policy"
        )
    return session


@dataclass(frozen=True, slots=True)
class ThresholdCandidateInvention:
    """In-memory reviewed candidates plus their durable collective attestation."""

    session: PTAReasoningSession
    proposals: tuple[PTAEscalationProposal, ...]
    reviewed: tuple[ReviewedThresholdProposal, ...]
    evidence: PrologThresholdCandidateSet

    def __post_init__(self) -> None:
        if not isinstance(self.session, PTAReasoningSession):
            raise TypeError("threshold candidate session is invalid")
        if not isinstance(self.evidence, PrologThresholdCandidateSet):
            raise TypeError("threshold candidate evidence is invalid")
        if (
            type(self.proposals) is not tuple
            or type(self.reviewed) is not tuple
            or not self.proposals
            or len(self.proposals) != len(self.reviewed)
            or len(self.proposals) != len(self.evidence.candidates)
            or any(
                not isinstance(proposal, PTAEscalationProposal)
                for proposal in self.proposals
            )
            or any(
                not isinstance(reviewed, ReviewedThresholdProposal)
                for reviewed in self.reviewed
            )
        ):
            raise ModelGenerationError("threshold candidate invention is inconsistent")
        for proposal, reviewed, candidate in zip(
            self.proposals, self.reviewed, self.evidence.candidates
        ):
            if (
                reviewed.proposal != proposal
                or proposal.semantic_id() != candidate.proposal_semantic_id
                or proposal.provenance_id() != candidate.proposal_provenance_id
                or reviewed.descriptor.literal_id != candidate.invented_literal_id
                or reviewed.evidence.field != candidate.field
                or reviewed.evidence.threshold != candidate.threshold
                or content_digest(reviewed.evidence.to_dict())
                != candidate.boundary_evidence_digest
                or canonical_json_bytes(proposal.to_dict())
                != canonical_json_bytes(candidate.proposal_payload)
            ):
                raise ModelGenerationError(
                    "threshold candidate invention differs from its evidence"
                )


def invent_threshold_candidates_for_corpus(
    corpus: LabeledCorpus,
    parent_manifest: OrderedLiteralManifest,
    *,
    numeric_fields: tuple[str, ...],
    budget: ThresholdCandidateBudget,
) -> ThresholdCandidateInvention:
    """Return a complete bounded set of independently reviewed thresholds."""

    if corpus.role is not CorpusRole.INVENTION:
        raise ModelGenerationError("threshold invention requires the invention corpus")
    if not isinstance(budget, ThresholdCandidateBudget):
        raise TypeError("threshold candidate budget is invalid")
    if (
        type(numeric_fields) is not tuple
        or not numeric_fields
        or any(type(field) is not str or not field for field in numeric_fields)
        or len(set(numeric_fields)) != len(numeric_fields)
    ):
        raise ModelGenerationError("threshold candidate fields must be unique strings")
    canonical_fields = tuple(sorted(numeric_fields))
    if len(canonical_fields) > budget.maximum_fields:
        raise ModelGenerationError("threshold candidate field budget was exceeded")
    parent_catalog = parent_manifest.build_catalog()
    for numeric_field in canonical_fields:
        try:
            field = parent_catalog.schema.field(numeric_field)
        except KeyError as error:
            raise ModelGenerationError(
                "threshold field is absent from the parent schema"
            ) from error
        if field.kind.value != "number":
            raise ModelGenerationError("threshold invention fields must be numeric")
    session = _invention_session(corpus)
    proposals, attestation = _run_complete_threshold_collective(
        session, canonical_fields, budget
    )
    reviewed_pairs = tuple(
        (
            proposal,
            review_threshold_proposal(
                proposal, session=session, catalog=parent_catalog
            ),
        )
        for proposal in proposals
    )
    ordered = tuple(sorted(reviewed_pairs, key=lambda item: item[0].semantic_id()))
    ordered_proposals = tuple(item[0] for item in ordered)
    ordered_reviewed = tuple(item[1] for item in ordered)
    summaries = tuple(
        ThresholdCandidateProposal(
            proposal_semantic_id=proposal.semantic_id(),
            proposal_provenance_id=proposal.provenance_id(),
            field=reviewed.evidence.field,
            threshold=reviewed.evidence.threshold,
            invented_literal_id=reviewed.descriptor.literal_id,
            boundary_evidence_digest=content_digest(reviewed.evidence.to_dict()),
            proposal_payload=proposal.to_dict(),
        )
        for proposal, reviewed in ordered
    )
    evidence = PrologThresholdCandidateSet(
        invention_corpus_digest=corpus.digest,
        session_digest=content_digest(session.to_dict()),
        numeric_fields=canonical_fields,
        budget=budget,
        available_candidates=len(proposals),
        candidates=summaries,
        collective_protocol="PTM_PTA_COLLECTIVE_V1",
        gprolog_version=attestation[0],
        gprolog_binary_digest=attestation[1],
        module_digests=attestation[2],
    )
    return ThresholdCandidateInvention(
        session,
        ordered_proposals,
        ordered_reviewed,
        evidence,
    )


def _validate_threshold_candidate_derivation(
    evidence: PrologThresholdCandidateSet,
    corpus: LabeledCorpus,
    parent_manifest: OrderedLiteralManifest,
) -> tuple[
    PTAReasoningSession,
    Mapping[str, tuple[PTAEscalationProposal, ReviewedThresholdProposal]],
]:
    """Re-prove a v5 candidate population from its durable invention corpus."""

    if (
        corpus.role is not CorpusRole.INVENTION
        or corpus.digest != evidence.invention_corpus_digest
    ):
        raise ModelGenerationError(
            "threshold candidate set has different invention evidence"
        )
    session = _invention_session(corpus)
    replayed, _ = _run_complete_threshold_collective(
        session,
        evidence.numeric_fields,
        evidence.budget,
        expected_evidence=evidence,
    )
    if content_digest(session.to_dict()) != evidence.session_digest:
        raise ModelGenerationError(
            "threshold candidate reasoning session digest is inconsistent"
        )
    if len(replayed) != evidence.available_candidates:
        raise ModelGenerationError(
            "threshold candidate count differs from deterministic GNU Prolog replay"
        )
    catalog = parent_manifest.build_catalog()
    reviewed_by_id: dict[
        str, tuple[PTAEscalationProposal, ReviewedThresholdProposal]
    ] = {}
    for proposal, candidate in zip(replayed, evidence.candidates):
        reviewed = review_threshold_proposal(
            proposal, session=session, catalog=catalog
        )
        if (
            proposal.semantic_id() != candidate.proposal_semantic_id
            or proposal.provenance_id() != candidate.proposal_provenance_id
            or canonical_json_bytes(proposal.to_dict())
            != canonical_json_bytes(candidate.proposal_payload)
            or reviewed.evidence.field != candidate.field
            or reviewed.evidence.threshold != candidate.threshold
            or reviewed.descriptor.literal_id != candidate.invented_literal_id
            or content_digest(reviewed.evidence.to_dict())
            != candidate.boundary_evidence_digest
        ):
            raise ModelGenerationError(
                "threshold candidate differs from deterministic GNU Prolog replay"
            )
        reviewed_by_id[proposal.semantic_id()] = (proposal, reviewed)
    return session, MappingProxyType(reviewed_by_id)


def invent_threshold_for_corpus(
    corpus: LabeledCorpus,
    parent_manifest: OrderedLiteralManifest,
    *,
    numeric_field: str,
) -> tuple[
    PTAReasoningSession,
    PTAEscalationProposal,
    ReviewedThresholdProposal,
    PrologInventionEvidence,
]:
    """Compatibility wrapper preserving the original exactly-one contract."""

    invention = invent_threshold_candidates_for_corpus(
        corpus,
        parent_manifest,
        numeric_fields=(numeric_field,),
        budget=ThresholdCandidateBudget(maximum_fields=1, maximum_candidates=1),
    )
    proposal = invention.proposals[0]
    reviewed = invention.reviewed[0]
    evidence = _legacy_invention_evidence(invention.evidence)
    return invention.session, proposal, reviewed, evidence


def _legacy_invention_evidence(
    candidate_set: PrologThresholdCandidateSet,
) -> PrologInventionEvidence:
    if len(candidate_set.numeric_fields) != 1 or len(candidate_set.candidates) != 1:
        raise ModelGenerationError(
            "legacy invention evidence requires exactly one field and candidate"
        )
    candidate = candidate_set.candidates[0]
    return PrologInventionEvidence(
        invention_corpus_digest=candidate_set.invention_corpus_digest,
        session_digest=candidate_set.session_digest,
        numeric_field=candidate_set.numeric_fields[0],
        collective_protocol=candidate_set.collective_protocol,
        gprolog_version=candidate_set.gprolog_version,
        gprolog_binary_digest=candidate_set.gprolog_binary_digest,
        module_digests=candidate_set.module_digests,
        proposal_semantic_id=candidate.proposal_semantic_id,
        proposal_provenance_id=candidate.proposal_provenance_id,
    )


def verify_artifact_with_ptmrt(
    executable: str | Path,
    artifact_path: str | Path,
    expected_artifact_id: str,
    *,
    timeout_seconds: float = 30.0,
) -> str:
    command = [str(Path(executable)), "verify", str(Path(artifact_path))]
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
        creationflags=(
            getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        ),
    )
    if completed.returncode != 0:
        raise ModelGenerationError(
            "ptmrt conformance verification failed: " + completed.stderr.strip()
        )
    if f"verified {expected_artifact_id} (" not in completed.stdout:
        raise ModelGenerationError("ptmrt verified an unexpected artifact identity")
    return expected_artifact_id


def _ptmrt_record_field(name: str, value: object) -> str:
    if not name or ":" in name or any(ord(character) < 0x20 for character in name):
        raise ModelGenerationError("record field name is not portable to ptmrt")
    if value is None:
        return f"{name}:null"
    if type(value) is bool:
        return f"{name}:bool={'true' if value else 'false'}"
    if type(value) is int:
        if not -(2**63) <= value < 2**63:
            raise ModelGenerationError("record integer is outside ptmrt int64 range")
        return f"{name}:int={value}"
    if type(value) is float:
        if not math.isfinite(value):
            raise ModelGenerationError("record float must be finite")
        return f"{name}:float={repr(value)}"
    if type(value) is str:
        if "\x00" in value:
            raise ModelGenerationError("record string contains a null byte")
        return f"{name}:string={value}"
    raise ModelGenerationError("record value type is not portable to ptmrt")


def verify_records_with_ptmrt(
    executable: str | Path,
    artifact_path: str | Path,
    child: AdaptedChild,
    artifact: PackedTMInferenceArtifact,
    records: Sequence[Mapping[str, object]],
    *,
    timeout_seconds: float = 30.0,
) -> str:
    """Run raw live records through ptmrt and require exact child semantics."""

    return _verify_snapshot_records_with_ptmrt(
        executable,
        artifact_path,
        child.snapshot,
        child.manifest,
        artifact,
        records,
        timeout_seconds=timeout_seconds,
    ).artifact_id


def _verify_snapshot_records_with_ptmrt(
    executable: str | Path,
    artifact_path: str | Path,
    child_snapshot: AdaptiveSnapshotEnvelope,
    child_manifest: OrderedLiteralManifest,
    artifact: PackedTMInferenceArtifact,
    records: Sequence[Mapping[str, object]],
    *,
    timeout_seconds: float = 30.0,
) -> _LiveRuntimeVectors:
    """Run raw records through ptmrt against one durable child state."""

    if not records:
        raise ModelGenerationError("live ptmrt conformance requires records")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ModelGenerationError("live ptmrt timeout must be positive and finite")
    catalog = child_manifest.build_catalog()
    batch = catalog.encode(records).ta
    rows = tuple(batch.row_values(index) for index in range(batch.row_count))
    machine = ScalarBinaryTsetlinMachine(
        child_snapshot.snapshot.number_of_clauses,
        child_snapshot.snapshot.number_of_features,
        states_per_action=child_snapshot.snapshot.states_per_action,
        specificity=child_snapshot.snapshot.specificity,
        threshold=child_snapshot.snapshot.threshold,
        seed=0,
    )
    machine.restore(child_snapshot.snapshot)
    scalar_scores = tuple(machine.score(row) for row in rows)
    scalar_predictions = tuple(int(score > 0) for score in scalar_scores)
    packed_predictions = artifact.predict_records(records)
    native_features: list[tuple[int, ...]] = []
    native_scores: list[int] = []
    native_predictions: list[int] = []
    deadline = time.monotonic() + timeout_seconds
    for record, row, expected_score, packed_prediction in zip(
        records, rows, scalar_scores, packed_predictions
    ):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModelGenerationError("live ptmrt conformance timed out")
        command = [
            str(Path(executable)),
            "run-record",
            str(Path(artifact_path)),
            *(
                _ptmrt_record_field(name, value)
                for name, value in record.items()
            ),
        ]
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=remaining,
                check=False,
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    if os.name == "nt"
                    else 0
                ),
            )
        except subprocess.TimeoutExpired as error:
            raise ModelGenerationError("live ptmrt conformance timed out") from error
        if completed.returncode != 0:
            raise ModelGenerationError(
                "live ptmrt conformance failed: " + completed.stderr.strip()
            )
        try:
            output = json.loads(completed.stdout)
        except (json.JSONDecodeError, TypeError) as error:
            raise ModelGenerationError("live ptmrt returned malformed output") from error
        if (
            not isinstance(output, Mapping)
            or output.get("artifact_id") != artifact.artifact_id
            or output.get("features") != [int(value) for value in row]
            or output.get("prediction") != int(expected_score > 0)
            or output.get("prediction") != packed_prediction
            or output.get("score") != expected_score
        ):
            raise ModelGenerationError("live ptmrt disagrees with child semantics")
        native_features.append(tuple(output["features"]))
        native_scores.append(output["score"])
        native_predictions.append(output["prediction"])
    return _LiveRuntimeVectors(
        artifact.artifact_id,
        rows,
        scalar_scores,
        scalar_predictions,
        packed_predictions,
        tuple(native_features),
        tuple(native_scores),
        tuple(native_predictions),
    )


def _promotion_conformance_evidence(
    child: ModelGeneration,
    corpus: LabeledCorpus,
    vectors: _LiveRuntimeVectors,
    ptmrt_executable: str | Path,
) -> PromotionRuntimeConformanceEvidence:
    if corpus.role is not CorpusRole.PROMOTION:
        raise ModelGenerationError(
            "promotion native evidence requires the promotion holdout"
        )
    return PromotionRuntimeConformanceEvidence(
        child_generation_id=child.generation_id,
        artifact_id=vectors.artifact_id,
        snapshot_id=child.snapshot_id,
        literal_manifest_id=child.literal_manifest_id,
        corpus_digest=corpus.digest,
        scalar_features=vectors.scalar_features,
        scalar_scores=vectors.scalar_scores,
        scalar_predictions=vectors.scalar_predictions,
        packed_predictions=vectors.packed_predictions,
        native_features=vectors.native_features,
        native_scores=vectors.native_scores,
        native_predictions=vectors.native_predictions,
        ptmrt_binary_digest=_file_digest(
            Path(ptmrt_executable).resolve(strict=True),
            maximum_bytes=_MAX_ATTESTED_EXECUTABLE_BYTES,
        ),
    )


@dataclass(frozen=True, slots=True)
class TrainedParentLifecycleResult:
    parent_generation: ModelGeneration
    extended_generation: ModelGeneration
    child_generation: ModelGeneration
    restoration_bundle: AdaptiveRestorationBundle
    extended_parent: ExtendedParent
    child: AdaptedChild
    child_artifact: PackedTMInferenceArtifact
    conformance: RuntimeConformanceReport
    promotion_conformance_evidence: PromotionRuntimeConformanceEvidence
    promotion_audit: PromotionAuditSnapshot
    lineage: ModelGenerationLineage
    invention_evidence: PrologInventionEvidence | PrologThresholdCandidateSet
    candidate_set: PrologThresholdCandidateSet
    candidate_selection: ThresholdCandidateSelection
    dataset_id: str
    preactivation_example_ids: frozenset[str | int]
    ptmrt_executable: Path
    controller: ModelGenerationController


@dataclass(frozen=True, slots=True)
class DeescalationLifecycleResult:
    parent_generation: ModelGeneration
    contracted_generation: ModelGeneration
    child_generation: ModelGeneration
    restoration_bundle: AdaptiveRestorationBundle
    contracted_parent: ContractedParent
    child_artifact: PackedTMInferenceArtifact
    conformance: RuntimeConformanceReport
    promotion_conformance_evidence: PromotionRuntimeConformanceEvidence
    promotion_audit: PromotionAuditSnapshot
    lineage: LiteralContractionLineage
    deescalation_evidence: PrologDeescalationEvidence
    dataset_id: str
    preactivation_example_ids: frozenset[str | int]
    ptmrt_executable: Path
    controller: ModelGenerationController


@dataclass(frozen=True, slots=True)
class _PreparedThresholdCandidate:
    proposal: PTAEscalationProposal
    reviewed: ReviewedThresholdProposal
    extended: ExtendedParent
    child: AdaptedChild
    outcome: ThresholdCandidateOutcome


def _snapshot_predictions(
    snapshot: TMSnapshot,
    manifest: OrderedLiteralManifest,
    records: Sequence[Mapping[str, object]],
) -> tuple[int, ...]:
    batch = manifest.build_catalog().encode(records).ta
    rows = tuple(batch.row_values(index) for index in range(batch.row_count))
    machine = ScalarBinaryTsetlinMachine(
        snapshot.number_of_clauses,
        snapshot.number_of_features,
        states_per_action=snapshot.states_per_action,
        specificity=snapshot.specificity,
        threshold=snapshot.threshold,
        seed=0,
    )
    machine.restore(snapshot)
    return tuple(machine.predict(rows))


def _threshold_candidate_outcome(
    parent_snapshot: TMSnapshot,
    parent_manifest: OrderedLiteralManifest,
    proposal: PTAEscalationProposal,
    reviewed: ReviewedThresholdProposal,
    extended: ExtendedParent,
    child: AdaptedChild,
    child_preprocessing_id: str,
    corpus: LabeledCorpus,
) -> ThresholdCandidateOutcome:
    if corpus.role is not CorpusRole.ADAPTATION:
        raise ModelGenerationError("threshold selection requires adaptation evidence")
    parent_predictions = _snapshot_predictions(
        parent_snapshot, parent_manifest, corpus.records
    )
    child_predictions = _snapshot_predictions(
        child.snapshot.snapshot, child.manifest, corpus.records
    )
    both_correct = both_wrong = improvements = regressions = disagreements = 0
    for truth, parent_prediction, child_prediction in zip(
        corpus.labels, parent_predictions, child_predictions
    ):
        parent_correct = parent_prediction == truth
        child_correct = child_prediction == truth
        disagreements += parent_prediction != child_prediction
        if parent_correct and child_correct:
            both_correct += 1
        elif not parent_correct and not child_correct:
            both_wrong += 1
        elif not parent_correct and child_correct:
            improvements += 1
        else:
            regressions += 1
    behavior = AdaptiveBehaviorIdentity.from_child(
        child, preprocessing_contract_id=child_preprocessing_id
    )
    return ThresholdCandidateOutcome(
        proposal_semantic_id=proposal.semantic_id(),
        proposal_provenance_id=proposal.provenance_id(),
        invented_literal_id=reviewed.descriptor.literal_id,
        extended_snapshot_id=extended.snapshot.snapshot_id,
        extended_manifest_id=extended.manifest.manifest_id,
        child_snapshot_id=child.snapshot.snapshot_id,
        child_manifest_id=child.manifest.manifest_id,
        child_preprocessing_id=child_preprocessing_id,
        adaptive_behavior_id=behavior.behavior_id,
        observations=len(corpus.examples),
        parent_errors=both_wrong + improvements,
        child_errors=both_wrong + regressions,
        disagreements=disagreements,
        improvements=improvements,
        regressions=regressions,
        both_correct=both_correct,
        both_wrong=both_wrong,
    )


def _select_threshold_candidate(
    *,
    parent_generation: ModelGeneration,
    parent_snapshot: TMSnapshot,
    parent_manifest: OrderedLiteralManifest,
    parent_training_corpus: LabeledCorpus,
    adaptation_corpus: LabeledCorpus,
    adaptation_epochs: int,
    invention: ThresholdCandidateInvention,
    policy: ThresholdCandidateSelectionPolicy,
    store: ModelGenerationStore,
) -> tuple[_PreparedThresholdCandidate, ThresholdCandidateSelection]:
    prepared: list[_PreparedThresholdCandidate] = []
    for proposal, reviewed in zip(invention.proposals, invention.reviewed):
        extended = extend_parent_with_threshold(
            parent_snapshot,
            parent_manifest,
            reviewed,
            session=invention.session,
            equivalence_records=parent_training_corpus.records,
        )
        store.put_snapshot(extended.snapshot)
        store.put_manifest(extended.manifest)
        child = adapt_extended_parent(
            extended, adaptation_corpus, epochs=adaptation_epochs
        )
        child_preprocessing = PreprocessingContract.from_catalog(
            child.manifest.build_catalog()
        )
        child_preprocessing_id, _ = store.put_preprocessing(child_preprocessing)
        store.put_snapshot(child.snapshot)
        store.put_manifest(child.manifest)
        outcome = _threshold_candidate_outcome(
            parent_snapshot,
            parent_manifest,
            proposal,
            reviewed,
            extended,
            child,
            child_preprocessing_id,
            adaptation_corpus,
        )
        prepared.append(
            _PreparedThresholdCandidate(proposal, reviewed, extended, child, outcome)
        )
    outcomes = tuple(sorted((item.outcome for item in prepared), key=lambda item: item.proposal_semantic_id))
    eligible = tuple(
        item
        for item in outcomes
        if item.observations >= policy.minimum_observations
        and (
            item.child_errors < item.parent_errors
            if policy.require_strict_improvement
            else item.child_errors <= item.parent_errors
        )
    )
    if not eligible:
        raise ModelGenerationError("no threshold candidate satisfies selection policy")
    winner = min(
        eligible,
        key=lambda item: (
            item.child_errors,
            item.regressions,
            -item.improvements,
            item.proposal_semantic_id,
        ),
    )
    selection = ThresholdCandidateSelection(
        candidate_set_id=invention.evidence.candidate_set_id,
        parent_generation_id=parent_generation.generation_id,
        parent_snapshot_id=AdaptiveSnapshotEnvelope(parent_snapshot).snapshot_id,
        parent_manifest_id=parent_manifest.manifest_id,
        adaptation_corpus_digest=adaptation_corpus.digest,
        adaptation_epochs=adaptation_epochs,
        policy=policy,
        outcomes=outcomes,
        selected_proposal_semantic_id=winner.proposal_semantic_id,
        selected_proposal_provenance_id=winner.proposal_provenance_id,
    )
    store.put_threshold_candidate_selection(selection)
    selected = next(
        item
        for item in prepared
        if item.outcome.proposal_semantic_id == winner.proposal_semantic_id
    )
    return selected, selection


def _execute_trained_parent_lifecycle_once(
    *,
    parent_snapshot: TMSnapshot,
    parent_manifest: OrderedLiteralManifest,
    parent_training_corpus: LabeledCorpus,
    corpora: LifecycleCorpora,
    numeric_fields: tuple[str, ...],
    candidate_budget: ThresholdCandidateBudget,
    selection_policy: ThresholdCandidateSelectionPolicy,
    adaptation_epochs: int,
    promotion_policy: PromotionAuditPolicy,
    store: ModelGenerationStore,
    ptmrt_executable: str | Path,
    telemetry: TelemetrySession | None = None,
    event_sink: TelemetrySink | None = None,
) -> TrainedParentLifecycleResult:
    """Execute one recurrent exact trained-parent PTA generation episode."""

    if parent_training_corpus.role is not CorpusRole.PARENT_TRAINING:
        raise ModelGenerationError("parent training evidence has the wrong corpus role")
    if parent_training_corpus.dataset_id != corpora.invention.dataset_id:
        raise ModelGenerationError("parent and lifecycle corpora use different datasets")
    if (
        not promotion_policy.require_strict_improvement
        or promotion_policy.maximum_regressions != 0
    ):
        raise ModelGenerationError(
            "the first trained-parent loop requires strict zero-regression promotion"
        )
    lifecycle_ids = {
        example.example_id
        for corpus in (
            corpora.invention,
            corpora.adaptation,
            corpora.promotion,
        )
        for example in corpus.examples
    }
    if lifecycle_ids & {
        example.example_id for example in parent_training_corpus.examples
    }:
        raise ModelGenerationError("parent training IDs overlap lifecycle corpora")
    preactivation_example_ids = frozenset(
        lifecycle_ids
        | {example.example_id for example in parent_training_corpus.examples}
    )

    parent_envelope = AdaptiveSnapshotEnvelope(parent_snapshot)
    parent_preprocessing, parent_artifact = compile_generation_artifact(
        parent_snapshot,
        parent_manifest,
        name="PTM trained adaptive parent",
        validation_records=parent_training_corpus.records,
        validation_signature={
            "generation_stage": "trained_parent",
            "training_corpus_digest": parent_training_corpus.digest,
        },
    )
    parent_preprocessing_id, _ = store.put_preprocessing(parent_preprocessing)
    store.put_snapshot(parent_envelope)
    store.put_manifest(parent_manifest)
    store.put_artifact(parent_artifact)
    parent_generation = ModelGeneration(
        GenerationKind.TRAINED_PARENT,
        parent_envelope.snapshot_id,
        parent_manifest.manifest_id,
        parent_preprocessing_id,
        parent_artifact.artifact_id,
        None,
        None,
        ((CorpusRole.PARENT_TRAINING.value, parent_training_corpus.digest),),
    )
    store.put_generation(parent_generation)
    restoration_bundle = AdaptiveRestorationBundle(
        parent_generation.generation_id,
        parent_envelope.snapshot_id,
        parent_manifest.manifest_id,
        parent_preprocessing_id,
        parent_artifact.artifact_id,
        parent_training_corpus.digest,
    )
    store.put_restoration_bundle(restoration_bundle)
    controller = ModelGenerationController(
        store,
        ptmrt_executable=ptmrt_executable,
        telemetry=telemetry,
        event_sink=event_sink,
    )
    if controller.active_generation_id is None:
        controller.register_parent(parent_generation, parent_training_corpus)
    elif controller.active_generation_id != parent_generation.generation_id:
        raise ModelGenerationError("store already routes a different active generation")
    else:
        events = store.read_events()
        parent_usage = EvidenceUsage(
            EvidenceUsagePurpose.PARENT_REGISTRATION,
            parent_generation.generation_id,
            (parent_training_corpus,),
        )
        registered_usage_id = (
            events[0].details.get("evidence_usage_id") if events else None
        )
        if (
            type(registered_usage_id) is not str
            or store.load_evidence_usage(registered_usage_id) != parent_usage
        ):
            raise ModelGenerationError(
                "active parent has different durable training evidence"
            )

    usage, activation_sequence, previous_lineage_id = (
        controller.reserve_candidate_evidence(corpora)
    )
    invention = invent_threshold_candidates_for_corpus(
        corpora.invention,
        parent_manifest,
        numeric_fields=numeric_fields,
        budget=candidate_budget,
    )
    store.put_threshold_candidate_set(invention.evidence)
    selected, candidate_selection = _select_threshold_candidate(
        parent_generation=parent_generation,
        parent_snapshot=parent_snapshot,
        parent_manifest=parent_manifest,
        parent_training_corpus=parent_training_corpus,
        adaptation_corpus=corpora.adaptation,
        adaptation_epochs=adaptation_epochs,
        invention=invention,
        policy=selection_policy,
        store=store,
    )
    proposal = selected.proposal
    reviewed = selected.reviewed
    extended = selected.extended
    child = selected.child
    extended_preprocessing = PreprocessingContract.from_catalog(
        extended.manifest.build_catalog()
    )
    extended_preprocessing_id, _ = store.put_preprocessing(extended_preprocessing)
    extended_generation = ModelGeneration(
        GenerationKind.EXTENDED_PARENT,
        extended.snapshot.snapshot_id,
        extended.manifest.manifest_id,
        extended_preprocessing_id,
        None,
        parent_generation.generation_id,
        restoration_bundle.bundle_id,
        ((CorpusRole.INVENTION.value, corpora.invention.digest),),
        proposal.semantic_id(),
        proposal.provenance_id(),
    )
    store.put_generation(extended_generation)

    expected_child_preprocessing = PreprocessingContract.from_catalog(
        child.manifest.build_catalog()
    )
    expected_child_preprocessing_id = preprocessing_contract_id(
        expected_child_preprocessing
    )
    behavior = AdaptiveBehaviorIdentity.from_child(
        child,
        preprocessing_contract_id=expected_child_preprocessing_id,
    )
    child_preprocessing, child_artifact = compile_generation_artifact(
        child.snapshot.snapshot,
        child.manifest,
        name="PTM PTA-adapted model-generation child",
        validation_records=corpora.promotion.records,
        validation_signature={
            "generation_stage": "adapted_child",
            "invention_corpus_digest": corpora.invention.digest,
            "adaptation_corpus_digest": corpora.adaptation.digest,
            "promotion_corpus_digest": corpora.promotion.digest,
            "origin_proposal_semantic_id": proposal.semantic_id(),
            "origin_proposal_provenance_id": proposal.provenance_id(),
            "adaptive_behavior_id": behavior.behavior_id,
            "threshold_candidate_set_id": invention.evidence.candidate_set_id,
            "threshold_candidate_selection_id": candidate_selection.selection_id,
        },
        restoration_reference=restoration_bundle.to_dict(),
    )
    child_preprocessing_id, _ = store.put_preprocessing(child_preprocessing)
    if (
        child_preprocessing != expected_child_preprocessing
        or child_preprocessing_id != expected_child_preprocessing_id
        or child_preprocessing_id != selected.outcome.child_preprocessing_id
        or behavior.behavior_id != selected.outcome.adaptive_behavior_id
    ):
        raise ModelGenerationError("child preprocessing changed during compilation")
    store.put_snapshot(child.snapshot)
    store.put_manifest(child.manifest)
    child_artifact_path = store.put_artifact(child_artifact)
    child_generation = ModelGeneration(
        GenerationKind.ADAPTED_CHILD,
        child.snapshot.snapshot_id,
        child.manifest.manifest_id,
        child_preprocessing_id,
        child_artifact.artifact_id,
        extended_generation.generation_id,
        restoration_bundle.bundle_id,
        (
            (CorpusRole.INVENTION.value, corpora.invention.digest),
            (CorpusRole.ADAPTATION.value, corpora.adaptation.digest),
            (CorpusRole.PROMOTION.value, corpora.promotion.digest),
        ),
        proposal.semantic_id(),
        proposal.provenance_id(),
    )
    store.put_generation(child_generation)

    verified_id = verify_artifact_with_ptmrt(
        ptmrt_executable, child_artifact_path, child_artifact.artifact_id
    )
    promotion_vectors = _verify_snapshot_records_with_ptmrt(
        ptmrt_executable,
        child_artifact_path,
        child.snapshot,
        child.manifest,
        child_artifact,
        corpora.promotion.records,
    )
    promotion_evidence = _promotion_conformance_evidence(
        child_generation,
        corpora.promotion,
        promotion_vectors,
        ptmrt_executable,
    )
    store.put_promotion_conformance(promotion_evidence)
    conformance = audit_runtime_conformance(
        child,
        child_artifact,
        corpora.promotion.records,
        ptmrt_verified=True,
        ptmrt_artifact_id=verified_id,
    )
    promotion = audit_parent_child(
        parent_snapshot,
        parent_manifest,
        child,
        corpora.promotion,
        conformance,
        promotion_policy,
    )
    store.put_audit(promotion)
    lineage = ModelGenerationLineage(
        parent_generation_id=parent_generation.generation_id,
        extended_generation_id=extended_generation.generation_id,
        child_generation_id=child_generation.generation_id,
        adaptive_behavior_id=behavior.behavior_id,
        restoration_bundle_id=restoration_bundle.bundle_id,
        promotion_audit_id=promotion.audit_id,
        promotion_conformance_evidence_id=promotion_evidence.evidence_id,
        invention_evidence_id=invention.evidence.evidence_id,
        evidence_usage_id=usage.usage_id,
        activation_sequence=activation_sequence,
        previous_activated_lineage_id=previous_lineage_id,
        invented_literal_id=reviewed.descriptor.literal_id,
        invention_corpus_digest=corpora.invention.digest,
        adaptation_corpus_digest=corpora.adaptation.digest,
        promotion_corpus_digest=corpora.promotion.digest,
        origin_proposal_semantic_id=proposal.semantic_id(),
        origin_proposal_provenance_id=proposal.provenance_id(),
        candidate_selection_id=candidate_selection.selection_id,
        schema=LINEAGE_SCHEMA,
    )
    store.put_lineage(lineage)
    controller.record_candidate(lineage)
    if not promotion.accepted:
        controller.reject_candidate(child_generation.generation_id, promotion)
        raise ModelGenerationError("strict promotion policy rejected the adapted child")
    controller.approve_promotion(lineage, promotion)
    controller.activate_child(lineage, promotion)
    return TrainedParentLifecycleResult(
        parent_generation,
        extended_generation,
        child_generation,
        restoration_bundle,
        extended,
        child,
        child_artifact,
        conformance,
        promotion_evidence,
        promotion,
        lineage,
        invention.evidence,
        invention.evidence,
        candidate_selection,
        parent_training_corpus.dataset_id,
        preactivation_example_ids,
        Path(ptmrt_executable),
        controller,
    )


def execute_trained_parent_lifecycle_with_candidates(
    *,
    parent_snapshot: TMSnapshot,
    parent_manifest: OrderedLiteralManifest,
    parent_training_corpus: LabeledCorpus,
    corpora: LifecycleCorpora,
    numeric_fields: tuple[str, ...],
    adaptation_epochs: int,
    promotion_policy: PromotionAuditPolicy,
    store: ModelGenerationStore,
    ptmrt_executable: str | Path,
    candidate_budget: ThresholdCandidateBudget | None = None,
    selection_policy: ThresholdCandidateSelectionPolicy | None = None,
    telemetry: TelemetrySession | None = None,
    event_sink: TelemetrySink | None = None,
) -> TrainedParentLifecycleResult:
    """Execute one bounded multi-candidate episode and terminalize failures."""

    resolved_budget = candidate_budget or ThresholdCandidateBudget()
    resolved_selection_policy = (
        selection_policy or ThresholdCandidateSelectionPolicy()
    )
    try:
        return _execute_trained_parent_lifecycle_once(
            parent_snapshot=parent_snapshot,
            parent_manifest=parent_manifest,
            parent_training_corpus=parent_training_corpus,
            corpora=corpora,
            numeric_fields=numeric_fields,
            candidate_budget=resolved_budget,
            selection_policy=resolved_selection_policy,
            adaptation_epochs=adaptation_epochs,
            promotion_policy=promotion_policy,
            store=store,
            ptmrt_executable=ptmrt_executable,
            telemetry=telemetry,
            event_sink=event_sink,
        )
    except BaseException as error:
        try:
            # A newly reconstructed controller treats any crash-orphaned
            # reservation as spent-but-terminal before returning a route.
            ModelGenerationController(
                store,
                ptmrt_executable=ptmrt_executable,
            )
        except Exception as recovery_error:
            error.add_note(
                "candidate evidence recovery also failed: "
                f"{recovery_error}"
            )
        raise


def execute_trained_parent_lifecycle(
    *,
    parent_snapshot: TMSnapshot,
    parent_manifest: OrderedLiteralManifest,
    parent_training_corpus: LabeledCorpus,
    corpora: LifecycleCorpora,
    numeric_field: str,
    adaptation_epochs: int,
    promotion_policy: PromotionAuditPolicy,
    store: ModelGenerationStore,
    ptmrt_executable: str | Path,
    telemetry: TelemetrySession | None = None,
    event_sink: TelemetrySink | None = None,
) -> TrainedParentLifecycleResult:
    """Compatibility entrypoint retaining the exactly-one threshold budget."""

    result = execute_trained_parent_lifecycle_with_candidates(
        parent_snapshot=parent_snapshot,
        parent_manifest=parent_manifest,
        parent_training_corpus=parent_training_corpus,
        corpora=corpora,
        numeric_fields=(numeric_field,),
        adaptation_epochs=adaptation_epochs,
        promotion_policy=promotion_policy,
        store=store,
        ptmrt_executable=ptmrt_executable,
        candidate_budget=ThresholdCandidateBudget(
            maximum_fields=1, maximum_candidates=1
        ),
        selection_policy=ThresholdCandidateSelectionPolicy(),
        telemetry=telemetry,
        event_sink=event_sink,
    )
    legacy_evidence = _legacy_invention_evidence(result.candidate_set)
    store.put_invention_evidence(legacy_evidence)
    return replace(result, invention_evidence=legacy_evidence)


def _execute_literal_deescalation_lifecycle_once(
    *,
    parent_snapshot: TMSnapshot,
    parent_manifest: OrderedLiteralManifest,
    parent_training_corpus: LabeledCorpus,
    corpora: DeescalationCorpora,
    maximum_candidates: int,
    promotion_policy: PromotionAuditPolicy,
    store: ModelGenerationStore,
    ptmrt_executable: str | Path,
    telemetry: TelemetrySession | None = None,
    event_sink: TelemetrySink | None = None,
) -> DeescalationLifecycleResult:
    """Execute one exact literal-equivalence contraction episode."""

    if parent_training_corpus.role is not CorpusRole.PARENT_TRAINING:
        raise ModelGenerationError("parent training evidence has the wrong corpus role")
    if parent_training_corpus.dataset_id != corpora.proof.dataset_id:
        raise ModelGenerationError(
            "parent and de-escalation corpora use different datasets"
        )
    if (
        promotion_policy.require_strict_improvement
        or promotion_policy.maximum_regressions != 0
    ):
        raise ModelGenerationError(
            "the first De-escalation loop requires non-strict zero-regression promotion"
        )
    lifecycle_ids = {
        example.example_id
        for corpus in (corpora.proof, corpora.confirmation, corpora.promotion)
        for example in corpus.examples
    }
    if lifecycle_ids & {
        example.example_id for example in parent_training_corpus.examples
    }:
        raise ModelGenerationError(
            "parent training IDs overlap de-escalation corpora"
        )
    preactivation_example_ids = frozenset(
        lifecycle_ids
        | {example.example_id for example in parent_training_corpus.examples}
    )

    parent_envelope = AdaptiveSnapshotEnvelope(parent_snapshot)
    parent_preprocessing, parent_artifact = compile_generation_artifact(
        parent_snapshot,
        parent_manifest,
        name="PTM trained adaptive parent",
        validation_records=parent_training_corpus.records,
        validation_signature={
            "generation_stage": "trained_parent",
            "training_corpus_digest": parent_training_corpus.digest,
        },
    )
    parent_preprocessing_id, _ = store.put_preprocessing(parent_preprocessing)
    store.put_snapshot(parent_envelope)
    store.put_manifest(parent_manifest)
    store.put_artifact(parent_artifact)
    parent_generation = ModelGeneration(
        GenerationKind.TRAINED_PARENT,
        parent_envelope.snapshot_id,
        parent_manifest.manifest_id,
        parent_preprocessing_id,
        parent_artifact.artifact_id,
        None,
        None,
        ((CorpusRole.PARENT_TRAINING.value, parent_training_corpus.digest),),
    )
    store.put_generation(parent_generation)
    restoration_bundle = AdaptiveRestorationBundle(
        parent_generation.generation_id,
        parent_envelope.snapshot_id,
        parent_manifest.manifest_id,
        parent_preprocessing_id,
        parent_artifact.artifact_id,
        parent_training_corpus.digest,
    )
    store.put_restoration_bundle(restoration_bundle)
    controller = ModelGenerationController(
        store,
        ptmrt_executable=ptmrt_executable,
        telemetry=telemetry,
        event_sink=event_sink,
    )
    if controller.active_generation_id is None:
        controller.register_parent(parent_generation, parent_training_corpus)
    elif controller.active_generation_id != parent_generation.generation_id:
        raise ModelGenerationError("store already routes a different active generation")
    else:
        events = store.read_events()
        parent_usage = EvidenceUsage(
            EvidenceUsagePurpose.PARENT_REGISTRATION,
            parent_generation.generation_id,
            (parent_training_corpus,),
        )
        registered_usage_id = (
            events[0].details.get("evidence_usage_id") if events else None
        )
        if (
            type(registered_usage_id) is not str
            or store.load_evidence_usage(registered_usage_id) != parent_usage
        ):
            raise ModelGenerationError(
                "active parent has different durable training evidence"
            )

    usage, activation_sequence, previous_lineage_id = (
        controller.reserve_deescalation_evidence(corpora)
    )
    invention = invent_literal_contraction_for_corpus(
        corpora.proof,
        parent_snapshot,
        parent_manifest,
        maximum_candidates=maximum_candidates,
    )
    evidence = invention.evidence
    store.put_deescalation_evidence(evidence)
    contracted_parent = contract_parent_with_equivalent_literal(
        parent_snapshot,
        parent_manifest,
        evidence,
        proof_records=corpora.proof.records,
        confirmation_records=corpora.confirmation.records,
    )
    contracted_preprocessing = PreprocessingContract.from_catalog(
        contracted_parent.manifest.build_catalog()
    )
    contracted_preprocessing_id, _ = store.put_preprocessing(
        contracted_preprocessing
    )
    store.put_snapshot(contracted_parent.snapshot)
    store.put_manifest(contracted_parent.manifest)
    contracted_generation = ModelGeneration(
        GenerationKind.CONTRACTED_PARENT,
        contracted_parent.snapshot.snapshot_id,
        contracted_parent.manifest.manifest_id,
        contracted_preprocessing_id,
        None,
        parent_generation.generation_id,
        restoration_bundle.bundle_id,
        (
            (CorpusRole.DEESCALATION_PROOF.value, corpora.proof.digest),
            (
                CorpusRole.DEESCALATION_CONFIRMATION.value,
                corpora.confirmation.digest,
            ),
        ),
    )
    store.put_generation(contracted_generation)

    behavior = AdaptiveBehaviorIdentity(
        contracted_parent.snapshot.snapshot_id,
        contracted_parent.manifest.manifest_id,
        contracted_preprocessing_id,
    )
    child_preprocessing, child_artifact = compile_generation_artifact(
        contracted_parent.snapshot.snapshot,
        contracted_parent.manifest,
        name="PTM De-escalation literal-contraction child",
        validation_records=corpora.promotion.records,
        validation_signature={
            "generation_stage": "contracted_child",
            "deescalation_proof_corpus_digest": corpora.proof.digest,
            "deescalation_confirmation_corpus_digest": corpora.confirmation.digest,
            "promotion_corpus_digest": corpora.promotion.digest,
            "deescalation_evidence_id": evidence.evidence_id,
            "surviving_literal_id": str(evidence.surviving_literal_id),
            "removed_literal_id": str(evidence.removed_literal_id),
            "adaptive_behavior_id": behavior.behavior_id,
        },
        restoration_reference=restoration_bundle.to_dict(),
    )
    child_preprocessing_id, _ = store.put_preprocessing(child_preprocessing)
    if (
        child_preprocessing != contracted_preprocessing
        or child_preprocessing_id != contracted_preprocessing_id
    ):
        raise ModelGenerationError(
            "contracted child preprocessing changed during compilation"
        )
    child_artifact_path = store.put_artifact(child_artifact)
    child_generation = ModelGeneration(
        GenerationKind.CONTRACTED_CHILD,
        contracted_parent.snapshot.snapshot_id,
        contracted_parent.manifest.manifest_id,
        child_preprocessing_id,
        child_artifact.artifact_id,
        contracted_generation.generation_id,
        restoration_bundle.bundle_id,
        (
            (CorpusRole.DEESCALATION_PROOF.value, corpora.proof.digest),
            (
                CorpusRole.DEESCALATION_CONFIRMATION.value,
                corpora.confirmation.digest,
            ),
            (CorpusRole.PROMOTION.value, corpora.promotion.digest),
        ),
    )
    if (
        AdaptiveBehaviorIdentity.from_generation(child_generation).behavior_id
        != behavior.behavior_id
    ):
        raise ModelGenerationError("contracted child behavior identity changed")
    store.put_generation(child_generation)

    verified_id = verify_artifact_with_ptmrt(
        ptmrt_executable, child_artifact_path, child_artifact.artifact_id
    )
    promotion_vectors = _verify_snapshot_records_with_ptmrt(
        ptmrt_executable,
        child_artifact_path,
        contracted_parent.snapshot,
        contracted_parent.manifest,
        child_artifact,
        corpora.promotion.records,
    )
    promotion_evidence = _promotion_conformance_evidence(
        child_generation,
        corpora.promotion,
        promotion_vectors,
        ptmrt_executable,
    )
    store.put_promotion_conformance(promotion_evidence)
    conformance = audit_snapshot_runtime_conformance(
        contracted_parent.snapshot,
        contracted_parent.manifest,
        child_artifact,
        corpora.promotion.records,
        ptmrt_verified=True,
        ptmrt_artifact_id=verified_id,
    )
    promotion = audit_parent_child_snapshots(
        parent_snapshot,
        parent_manifest,
        contracted_parent.snapshot,
        contracted_parent.manifest,
        corpora.promotion,
        conformance,
        promotion_policy,
    )
    store.put_audit(promotion)
    lineage = LiteralContractionLineage(
        parent_generation_id=parent_generation.generation_id,
        contracted_generation_id=contracted_generation.generation_id,
        child_generation_id=child_generation.generation_id,
        adaptive_behavior_id=behavior.behavior_id,
        restoration_bundle_id=restoration_bundle.bundle_id,
        promotion_audit_id=promotion.audit_id,
        promotion_conformance_evidence_id=promotion_evidence.evidence_id,
        deescalation_evidence_id=evidence.evidence_id,
        evidence_usage_id=usage.usage_id,
        activation_sequence=activation_sequence,
        previous_activated_lineage_id=previous_lineage_id,
        surviving_literal_id=evidence.surviving_literal_id,
        removed_literal_id=evidence.removed_literal_id,
        proof_corpus_digest=corpora.proof.digest,
        confirmation_corpus_digest=corpora.confirmation.digest,
        promotion_corpus_digest=corpora.promotion.digest,
    )
    store.put_lineage(lineage)
    controller.record_candidate(lineage)
    if not promotion.accepted:
        controller.reject_candidate(child_generation.generation_id, promotion)
        raise ModelGenerationError(
            "non-strict promotion policy rejected the contracted child"
        )
    controller.approve_promotion(lineage, promotion)
    controller.activate_child(lineage, promotion)
    return DeescalationLifecycleResult(
        parent_generation,
        contracted_generation,
        child_generation,
        restoration_bundle,
        contracted_parent,
        child_artifact,
        conformance,
        promotion_evidence,
        promotion,
        lineage,
        evidence,
        parent_training_corpus.dataset_id,
        preactivation_example_ids,
        Path(ptmrt_executable),
        controller,
    )


def execute_literal_deescalation_lifecycle(
    *,
    parent_snapshot: TMSnapshot,
    parent_manifest: OrderedLiteralManifest,
    parent_training_corpus: LabeledCorpus,
    corpora: DeescalationCorpora,
    promotion_policy: PromotionAuditPolicy,
    store: ModelGenerationStore,
    ptmrt_executable: str | Path,
    maximum_candidates: int = MAX_DEESCALATION_CANDIDATES,
    telemetry: TelemetrySession | None = None,
    event_sink: TelemetrySink | None = None,
) -> DeescalationLifecycleResult:
    """Run one bounded De-escalation PTA contraction and terminalize failures."""

    try:
        return _execute_literal_deescalation_lifecycle_once(
            parent_snapshot=parent_snapshot,
            parent_manifest=parent_manifest,
            parent_training_corpus=parent_training_corpus,
            corpora=corpora,
            maximum_candidates=maximum_candidates,
            promotion_policy=promotion_policy,
            store=store,
            ptmrt_executable=ptmrt_executable,
            telemetry=telemetry,
            event_sink=event_sink,
        )
    except BaseException as error:
        try:
            ModelGenerationController(
                store,
                ptmrt_executable=ptmrt_executable,
            )
        except Exception as recovery_error:
            error.add_note(
                "de-escalation evidence recovery also failed: "
                f"{recovery_error}"
            )
        raise


def reopen_and_restore_for_drift(
    result: TrainedParentLifecycleResult | DeescalationLifecycleResult,
    live_corpus: LabeledCorpus,
    drift_policy: DriftAuditPolicy,
) -> tuple[PromotionAuditSnapshot, RestoredAdaptiveParent]:
    if live_corpus.role is not CorpusRole.LIVE:
        raise ModelGenerationError("reopen evaluation requires the live/drift corpus")
    if live_corpus.dataset_id != result.dataset_id:
        raise ModelGenerationError("live/drift corpus belongs to a different dataset")
    if result.preactivation_example_ids & {
        example.example_id for example in live_corpus.examples
    }:
        raise ModelGenerationError("live/drift example IDs overlap pre-activation evidence")
    drift = result.controller.request_reopen(
        result.child_generation.generation_id,
        live_corpus,
        drift_policy,
        result.ptmrt_executable,
    )
    restored = result.controller.restore_parent(result.restoration_bundle)
    return drift, restored
