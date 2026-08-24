"""UI-neutral orchestration for trained-parent model generations."""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import subprocess
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from ..model_artifact import PackedTMInferenceArtifact, export_packed_tm
from ..model_generation import (
    AdaptedChild,
    AdaptiveRestorationBundle,
    AdaptiveSnapshotEnvelope,
    CorpusRole,
    ExtendedParent,
    LabeledCorpus,
    LifecycleCorpora,
    ModelGeneration,
    ModelGenerationError,
    ModelGenerationLineage,
    GenerationKind,
    OrderedLiteralManifest,
    PromotionAuditPolicy,
    PromotionAuditSnapshot,
    PrologInventionEvidence,
    RuntimeConformanceReport,
    adapt_extended_parent,
    audit_parent_child,
    audit_runtime_conformance,
    canonical_json_bytes,
    content_digest,
    drift_requires_reopen,
    extend_parent_with_threshold,
    preprocessing_contract_id,
)
from ..preprocessing import PreprocessingContract
from ..prolog_resources import prolog_process_environment
from ..pta import (
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

    def load_invention_evidence(self, identifier: str) -> PrologInventionEvidence:
        result = PrologInventionEvidence.from_dict(
            self._read_json("invention-evidence", identifier)
        )
        if result.evidence_id != identifier:
            raise ModelGenerationError("invention evidence does not match its address")
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

    def put_lineage(self, value: ModelGenerationLineage) -> Path:
        return self._put_json("lineage", value.lineage_id, value.to_dict())

    def load_lineage(self, identifier: str) -> ModelGenerationLineage:
        result = ModelGenerationLineage.from_dict(
            self._read_json("lineage", identifier)
        )
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

    def recover_active_generation(self) -> str | None:
        active: str | None = None
        for event in self.read_events():
            if event.kind in (
                LifecycleEventKind.PARENT_REGISTERED,
                LifecycleEventKind.ACTIVATED,
                LifecycleEventKind.PARENT_RESTORED,
            ):
                active = event.generation_id
        return active


TelemetrySink = Callable[[TelemetryEvent], None]


@dataclass(frozen=True, slots=True)
class RestoredAdaptiveParent:
    generation_id: str
    snapshot: AdaptiveSnapshotEnvelope
    manifest: OrderedLiteralManifest
    preprocessing: PreprocessingContract
    artifact: PackedTMInferenceArtifact
    machine: ScalarBinaryTsetlinMachine


class ModelGenerationController:
    """Own durable active-generation routing above each generation's Class II registry."""

    def __init__(
        self,
        store: ModelGenerationStore,
        *,
        telemetry: TelemetrySession | None = None,
        event_sink: TelemetrySink | None = None,
    ) -> None:
        self.store = store
        self.telemetry = telemetry
        self.event_sink = event_sink
        self._control_lock = RLock()
        self._last_telemetry_error: Exception | None = None
        self._active_generation_id = store.recover_active_generation()
        self._validate_recovered_state()

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

    def _validate_lineage_graph(
        self, lineage: ModelGenerationLineage
    ) -> tuple[
        ModelGeneration,
        ModelGeneration,
        ModelGeneration,
        AdaptiveRestorationBundle,
        PromotionAuditSnapshot,
        PrologInventionEvidence,
    ]:
        if self.store.load_lineage(lineage.lineage_id) != lineage:
            raise ModelGenerationError("lineage differs from its durable object")
        parent = self.store.load_generation(lineage.parent_generation_id)
        extended = self.store.load_generation(lineage.extended_generation_id)
        child = self.store.load_generation(lineage.child_generation_id)
        bundle = self.store.load_restoration_bundle(lineage.restoration_bundle_id)
        audit = self.store.load_audit(lineage.promotion_audit_id)
        evidence = self.store.load_invention_evidence(lineage.invention_evidence_id)
        parent_manifest = self.store.load_manifest(parent.literal_manifest_id)
        extended_manifest = self.store.load_manifest(extended.literal_manifest_id)
        child_manifest = self.store.load_manifest(child.literal_manifest_id)
        parent_snapshot = self.store.load_snapshot(parent.snapshot_id).snapshot
        extended_snapshot = self.store.load_snapshot(extended.snapshot_id).snapshot
        child_snapshot = self.store.load_snapshot(child.snapshot_id).snapshot
        if child.inference_artifact_id is None:
            raise ModelGenerationError("adapted child lacks an inference artifact")
        child_artifact = self.store.load_artifact(child.inference_artifact_id)
        child_validation = child_artifact.manifest.get("validation")
        child_signature = (
            child_validation.get("signature")
            if isinstance(child_validation, Mapping)
            else None
        )
        child_preprocessing = child_artifact.preprocessing
        child_restoration = child_artifact.manifest.get("restoration_reference")
        invention_digests = ((CorpusRole.INVENTION.value, lineage.invention_corpus_digest),)
        child_digests = (
            (CorpusRole.INVENTION.value, lineage.invention_corpus_digest),
            (CorpusRole.ADAPTATION.value, lineage.adaptation_corpus_digest),
            (CorpusRole.PROMOTION.value, lineage.promotion_corpus_digest),
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
            or evidence.proposal_semantic_id != lineage.origin_proposal_semantic_id
            or evidence.proposal_provenance_id != lineage.origin_proposal_provenance_id
            or audit.corpus_role is not CorpusRole.PROMOTION
            or audit.corpus_digest != lineage.promotion_corpus_digest
            or audit.conformance.artifact_id != child.inference_artifact_id
            or not isinstance(child_signature, Mapping)
            or child_signature.get("generation_stage") != "adapted_child"
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

    def _validate_recovered_state(self) -> None:
        if self._active_generation_id is None:
            return
        generation = self.store.load_generation(self._active_generation_id)
        events = self.store.read_events()
        route = next(
            (
                event
                for event in reversed(events)
                if event.kind
                in (
                    LifecycleEventKind.PARENT_REGISTERED,
                    LifecycleEventKind.ACTIVATED,
                    LifecycleEventKind.PARENT_RESTORED,
                )
            ),
            None,
        )
        if route is None or route.generation_id != generation.generation_id:
            raise ModelGenerationError("durable active-generation route is inconsistent")
        if generation.kind is GenerationKind.ADAPTED_CHILD:
            lineage_id = route.details.get("lineage_id")
            if route.kind is not LifecycleEventKind.ACTIVATED or type(lineage_id) is not str:
                raise ModelGenerationError("recovered child lacks its activation lineage")
            lineage = self.store.load_lineage(lineage_id)
            self._validate_lineage_graph(lineage)
        elif generation.kind is not GenerationKind.TRAINED_PARENT:
            raise ModelGenerationError("durable routing targets a non-deployable generation")
        elif route.kind is LifecycleEventKind.PARENT_RESTORED:
            bundle_id = route.details.get("restoration_bundle_id")
            if type(bundle_id) is not str:
                raise ModelGenerationError("recovered parent lacks its restoration bundle")
            bundle = self.store.load_restoration_bundle(bundle_id)
            if bundle.parent_generation_id != generation.generation_id:
                raise ModelGenerationError("recovered parent route names a different bundle")
            self._resolve_restoration_bundle(bundle)

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
        snapshot = self.store.load_snapshot(bundle.adaptive_snapshot_id)
        manifest = self.store.load_manifest(bundle.ordered_literal_manifest_id)
        preprocessing = self.store.load_preprocessing(bundle.preprocessing_contract_id)
        artifact = self.store.load_artifact(bundle.deployed_parent_artifact_id)
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

    def register_parent(self, generation: ModelGeneration) -> None:
        if generation.kind is not GenerationKind.TRAINED_PARENT:
            raise ModelGenerationError("initial active generation must be a trained parent")
        if self.store.load_generation(generation.generation_id) != generation:
            raise ModelGenerationError("trained parent differs from durable generation")
        with self._control_lock:
            if self._active_generation_id is not None or self.store.read_events():
                raise ModelGenerationError("a trained parent is already registered")
            self._active_generation_id = generation.generation_id
            try:
                self.store.append_event(
                    LifecycleEventKind.PARENT_REGISTERED, generation.generation_id
                )
            except Exception:
                self._active_generation_id = self.store.recover_active_generation()
                raise
        self._emit("parent_registered", generation_id=generation.generation_id)

    def record_candidate(self, lineage: ModelGenerationLineage) -> None:
        with self._control_lock:
            self._validate_lineage_graph(lineage)
            if self._active_generation_id != lineage.parent_generation_id:
                raise ModelGenerationError("candidate parent is not the active generation")
            events = self.store.read_events()
            if any(
                event.kind is LifecycleEventKind.ACTIVATED
                and event.generation_id == lineage.child_generation_id
                for event in events
            ):
                raise ModelGenerationError(
                    "candidate child generation has already been activated"
                )
            if not events or events[-1].kind not in (
                LifecycleEventKind.PARENT_REGISTERED,
                LifecycleEventKind.CANDIDATE_REJECTED,
                LifecycleEventKind.PARENT_RESTORED,
            ):
                raise ModelGenerationError("candidate creation is invalid in the current state")
            self.store.append_event(
                LifecycleEventKind.CANDIDATE_CREATED,
                lineage.child_generation_id,
                lineage_id=lineage.lineage_id,
                parent_generation_id=lineage.parent_generation_id,
                extended_generation_id=lineage.extended_generation_id,
            )
        self._emit(
            "proposal_created",
            generation_id=lineage.child_generation_id,
            lineage_id=lineage.lineage_id,
        )

    def reject_candidate(
        self, generation_id: str, audit: PromotionAuditSnapshot
    ) -> None:
        with self._control_lock:
            if audit.accepted:
                raise ModelGenerationError("an accepted promotion audit cannot be rejected")
            if self.store.load_audit(audit.audit_id) != audit:
                raise ModelGenerationError("rejection audit differs from durable evidence")
            events = self.store.read_events()
            lineage_id = events[-1].details.get("lineage_id") if events else None
            if (
                not events
                or events[-1].kind is not LifecycleEventKind.CANDIDATE_CREATED
                or events[-1].generation_id != generation_id
                or type(lineage_id) is not str
            ):
                raise ModelGenerationError("candidate rejection is invalid in the current state")
            lineage = self.store.load_lineage(lineage_id)
            *_, stored_audit, _ = self._validate_lineage_graph(lineage)
            if (
                self._active_generation_id != lineage.parent_generation_id
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
        self, lineage: ModelGenerationLineage, audit: PromotionAuditSnapshot
    ) -> None:
        if not audit.accepted or not audit.conformance.exact:
            raise ModelGenerationError("only an accepted exact audit may approve promotion")
        if lineage.promotion_audit_id != audit.audit_id:
            raise ModelGenerationError("lineage references a different promotion audit")
        with self._control_lock:
            *_, stored_audit, _ = self._validate_lineage_graph(lineage)
            if stored_audit != audit:
                raise ModelGenerationError("promotion audit differs from durable audit")
            events = self.store.read_events()
            if (
                self._active_generation_id != lineage.parent_generation_id
                or not events
                or events[-1].kind is not LifecycleEventKind.CANDIDATE_CREATED
                or events[-1].generation_id != lineage.child_generation_id
                or events[-1].details.get("lineage_id") != lineage.lineage_id
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
        self, lineage: ModelGenerationLineage, audit: PromotionAuditSnapshot
    ) -> None:
        if not audit.accepted or lineage.promotion_audit_id != audit.audit_id:
            raise ModelGenerationError("child activation lacks an accepted promotion audit")
        with self._control_lock:
            *_, stored_audit, _ = self._validate_lineage_graph(lineage)
            if stored_audit != audit:
                raise ModelGenerationError("activation audit differs from durable audit")
            events = self.store.read_events()
            if (
                not events
                or events[-1].kind is not LifecycleEventKind.PROMOTION_APPROVED
                or events[-1].generation_id != lineage.child_generation_id
                or events[-1].details.get("lineage_id") != lineage.lineage_id
                or events[-1].details.get("audit_id") != audit.audit_id
            ):
                raise ModelGenerationError("activation lacks the durable promotion decision")
            previous = self._active_generation_id
            if previous != lineage.parent_generation_id:
                raise ModelGenerationError("active generation is not the lineage parent")
            self._active_generation_id = lineage.child_generation_id
            try:
                self.store.append_event(
                    LifecycleEventKind.ACTIVATED,
                    lineage.child_generation_id,
                    previous_generation_id=previous,
                    lineage_id=lineage.lineage_id,
                    audit_id=audit.audit_id,
                )
            except Exception:
                self._active_generation_id = self.store.recover_active_generation()
                raise
        self._emit(
            "artifact_published",
            generation_id=lineage.child_generation_id,
            lineage_id=lineage.lineage_id,
        )
        self._emit("activated", generation_id=lineage.child_generation_id)

    def request_reopen(
        self, child_generation_id: str, drift: PromotionAuditSnapshot
    ) -> None:
        with self._control_lock:
            if self._active_generation_id != child_generation_id:
                raise ModelGenerationError("reopen target is not the active generation")
            if self.store.load_audit(drift.audit_id) != drift:
                raise ModelGenerationError("drift audit differs from durable evidence")
            if not drift.conformance.exact or not drift_requires_reopen(drift):
                raise ModelGenerationError("labeled drift does not justify reopen")
            events = self.store.read_events()
            if (
                not events
                or events[-1].kind is not LifecycleEventKind.ACTIVATED
                or events[-1].generation_id != child_generation_id
            ):
                raise ModelGenerationError("reopen lacks the active child's durable activation")
            lineage_id = events[-1].details.get("lineage_id")
            if type(lineage_id) is not str:
                raise ModelGenerationError("active child lacks durable lineage")
            lineage = self.store.load_lineage(lineage_id)
            _, _, child, bundle, _, _ = self._validate_lineage_graph(lineage)
            if (
                child.generation_id != child_generation_id
                or child.inference_artifact_id != drift.conformance.artifact_id
            ):
                raise ModelGenerationError("drift audit is not tied to the active child")
            self.store.append_event(
                LifecycleEventKind.REOPEN_REQUESTED,
                child_generation_id,
                drift_audit_id=drift.audit_id,
                lineage_id=lineage.lineage_id,
                restoration_bundle_id=bundle.bundle_id,
                parent_errors=drift.parent_errors,
                child_errors=drift.child_errors,
            )
        self._emit(
            "reopen_requested",
            generation_id=child_generation_id,
            drift_audit_id=drift.audit_id,
        )

    def restore_parent(
        self, bundle: AdaptiveRestorationBundle
    ) -> RestoredAdaptiveParent:
        with self._control_lock:
            active_child_id = self._active_generation_id
            if active_child_id is None:
                raise ModelGenerationError("restoration requires an active child")
            events = self.store.read_events()
            if (
                not events
                or events[-1].kind is not LifecycleEventKind.REOPEN_REQUESTED
                or events[-1].generation_id != active_child_id
                or events[-1].details.get("restoration_bundle_id") != bundle.bundle_id
            ):
                raise ModelGenerationError("restoration lacks a matching reopen request")
            drift_id = events[-1].details.get("drift_audit_id")
            lineage_id = events[-1].details.get("lineage_id")
            if type(drift_id) is not str or type(lineage_id) is not str:
                raise ModelGenerationError("reopen request lacks durable evidence")
            drift = self.store.load_audit(drift_id)
            lineage = self.store.load_lineage(lineage_id)
            _, _, child, graph_bundle, _, _ = self._validate_lineage_graph(lineage)
            if (
                child.generation_id != active_child_id
                or graph_bundle != bundle
                or not drift_requires_reopen(drift)
                or drift.conformance.artifact_id != child.inference_artifact_id
            ):
                raise ModelGenerationError("reopen evidence does not authorize restoration")
            restored = self._resolve_restoration_bundle(bundle)
            self._active_generation_id = bundle.parent_generation_id
            try:
                self.store.append_event(
                    LifecycleEventKind.PARENT_RESTORED,
                    bundle.parent_generation_id,
                    restoration_bundle_id=bundle.bundle_id,
                    previous_generation_id=active_child_id,
                    lineage_id=lineage.lineage_id,
                    drift_audit_id=drift.audit_id,
                )
            except Exception:
                self._active_generation_id = self.store.recover_active_generation()
                raise
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
    """Require GNU Prolog to produce the sole threshold considered for approval."""

    if corpus.role is not CorpusRole.INVENTION:
        raise ModelGenerationError("threshold invention requires the invention corpus")
    parent_catalog = parent_manifest.build_catalog()
    try:
        field = parent_catalog.schema.field(numeric_field)
    except KeyError as error:
        raise ModelGenerationError("threshold field is absent from the parent schema") from error
    if field.kind.value != "number":
        raise ModelGenerationError("threshold invention field must be numeric")
    session = PTAReasoningSession(corpus.dataset_id)
    for example in corpus.examples:
        for name, value in example.record.items():
            session.add_observation("pta:input", example.example_id, name, value)
        session.add_example_label(example.example_id, example.label)
    service = PTACollectiveService()
    query = PTACollectiveQuery(
        numeric_fields=(numeric_field,),
        discover_intervals=False,
        derive_deescalation=False,
        derive_escalation=True,
    )
    result = service.run(
        session,
        query=query,
    )
    proposals = tuple(
        item
        for item in result.proposals
        if item.native_target == "threshold"
        and item.structure.get("field") == numeric_field
    )
    if len(proposals) != 1:
        raise ModelGenerationError(
            "the bounded collective must yield exactly one threshold proposal"
        )
    proposal = proposals[0]
    reviewed = review_threshold_proposal(
        proposal, session=session, catalog=parent_catalog
    )
    evidence = PrologInventionEvidence(
        invention_corpus_digest=corpus.digest,
        session_digest=content_digest(session.to_dict()),
        numeric_field=numeric_field,
        collective_protocol="PTM_PTA_COLLECTIVE_V1",
        gprolog_version=_gprolog_version(service.executable),
        gprolog_binary_digest=_file_digest(
            service.executable, maximum_bytes=_MAX_ATTESTED_EXECUTABLE_BYTES
        ),
        module_digests=tuple(
            sorted(
                (name, _file_digest(path))
                for name, path in service.module_paths.items()
            )
        ),
        proposal_semantic_id=proposal.semantic_id(),
        proposal_provenance_id=proposal.provenance_id(),
    )
    return session, proposal, reviewed, evidence


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
    promotion_audit: PromotionAuditSnapshot
    lineage: ModelGenerationLineage
    invention_evidence: PrologInventionEvidence
    dataset_id: str
    preactivation_example_ids: frozenset[str | int]
    controller: ModelGenerationController


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
    """Execute and durably activate the first exact trained-parent PTA loop."""

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

    invention_session, proposal, reviewed, invention_evidence = (
        invent_threshold_for_corpus(
            corpora.invention,
            parent_manifest,
            numeric_field=numeric_field,
        )
    )
    store.put_invention_evidence(invention_evidence)

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
        store, telemetry=telemetry, event_sink=event_sink
    )
    if controller.active_generation_id is None:
        controller.register_parent(parent_generation)
    elif controller.active_generation_id != parent_generation.generation_id:
        raise ModelGenerationError("store already routes a different active generation")

    extended = extend_parent_with_threshold(
        parent_snapshot,
        parent_manifest,
        reviewed,
        session=invention_session,
        equivalence_records=parent_training_corpus.records,
    )
    store.put_snapshot(extended.snapshot)
    store.put_manifest(extended.manifest)
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

    child = adapt_extended_parent(
        extended, corpora.adaptation, epochs=adaptation_epochs
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
        },
        restoration_reference=restoration_bundle.to_dict(),
    )
    child_preprocessing_id, _ = store.put_preprocessing(child_preprocessing)
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
        parent_generation.generation_id,
        extended_generation.generation_id,
        child_generation.generation_id,
        restoration_bundle.bundle_id,
        promotion.audit_id,
        invention_evidence.evidence_id,
        reviewed.descriptor.literal_id,
        corpora.invention.digest,
        corpora.adaptation.digest,
        corpora.promotion.digest,
        proposal.semantic_id(),
        proposal.provenance_id(),
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
        promotion,
        lineage,
        invention_evidence,
        parent_training_corpus.dataset_id,
        preactivation_example_ids,
        controller,
    )


def reopen_and_restore_for_drift(
    result: TrainedParentLifecycleResult,
    live_corpus: LabeledCorpus,
) -> tuple[PromotionAuditSnapshot, RestoredAdaptiveParent]:
    if live_corpus.role is not CorpusRole.LIVE:
        raise ModelGenerationError("reopen evaluation requires the live/drift corpus")
    if live_corpus.dataset_id != result.dataset_id:
        raise ModelGenerationError("live/drift corpus belongs to a different dataset")
    if result.preactivation_example_ids & {
        example.example_id for example in live_corpus.examples
    }:
        raise ModelGenerationError("live/drift example IDs overlap pre-activation evidence")
    drift = audit_parent_child(
        result.extended_parent.parent_snapshot.snapshot,
        result.extended_parent.parent_manifest,
        result.child,
        live_corpus,
        result.conformance,
        PromotionAuditPolicy(minimum_observations=1),
    )
    result.controller.store.put_audit(drift)
    result.controller.request_reopen(result.child_generation.generation_id, drift)
    restored = result.controller.restore_parent(result.restoration_bundle)
    return drift, restored
