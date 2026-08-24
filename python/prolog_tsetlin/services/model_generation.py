"""UI-neutral orchestration for trained-parent model generations."""

from __future__ import annotations

import json
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
_MAX_EVENT_LOG_BYTES = 16 * 1024 * 1024
_MAX_STORED_JSON_BYTES = 4 * 1024 * 1024


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
        self._event_lock = RLock()

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
        return AdaptiveSnapshotEnvelope.from_dict(self._read_json("snapshots", identifier))

    def put_manifest(self, value: OrderedLiteralManifest) -> Path:
        return self._put_json("literal-manifests", value.manifest_id, value.to_dict())

    def load_manifest(self, identifier: str) -> OrderedLiteralManifest:
        return OrderedLiteralManifest.from_dict(
            self._read_json("literal-manifests", identifier)
        )

    def put_preprocessing(self, value: PreprocessingContract) -> tuple[str, Path]:
        identifier = preprocessing_contract_id(value)
        return identifier, self._put_json("preprocessing", identifier, value.to_dict())

    def load_preprocessing(self, identifier: str) -> PreprocessingContract:
        return PreprocessingContract.from_dict(self._read_json("preprocessing", identifier))

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
        return ModelGeneration.from_dict(self._read_json("generations", identifier))

    def put_restoration_bundle(self, value: AdaptiveRestorationBundle) -> Path:
        return self._put_json("restoration-bundles", value.bundle_id, value.to_dict())

    def load_restoration_bundle(self, identifier: str) -> AdaptiveRestorationBundle:
        return AdaptiveRestorationBundle.from_dict(
            self._read_json("restoration-bundles", identifier)
        )

    def put_audit(self, value: PromotionAuditSnapshot) -> Path:
        return self._put_json("audits", value.audit_id, value.to_dict())

    def load_audit(self, identifier: str) -> PromotionAuditSnapshot:
        return PromotionAuditSnapshot.from_dict(self._read_json("audits", identifier))

    def put_lineage(self, value: ModelGenerationLineage) -> Path:
        return self._put_json("lineage", value.lineage_id, value.to_dict())

    def load_lineage(self, identifier: str) -> ModelGenerationLineage:
        return ModelGenerationLineage.from_dict(self._read_json("lineage", identifier))

    @property
    def event_log_path(self) -> Path:
        return self.root / "lifecycle" / "events.jsonl"

    def read_events(self) -> tuple[GenerationLifecycleEvent, ...]:
        path = self.event_log_path
        if not path.exists():
            return ()
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
        return tuple(events)

    def append_event(
        self,
        kind: LifecycleEventKind,
        generation_id: str,
        **details: object,
    ) -> GenerationLifecycleEvent:
        with self._event_lock:
            events = self.read_events()
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
            publish_bytes(self.event_log_path, encoded, overwrite=True)
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
        self._active_generation_id = store.recover_active_generation()

    @property
    def active_generation_id(self) -> str | None:
        with self._control_lock:
            return self._active_generation_id

    def _emit(self, kind: str, **payload: object) -> None:
        if self.telemetry is None:
            return
        event = self.telemetry.emit("generation", kind, **payload)
        if self.event_sink is not None:
            self.event_sink(event)

    def register_parent(self, generation: ModelGeneration) -> None:
        if generation.kind is not GenerationKind.TRAINED_PARENT:
            raise ModelGenerationError("initial active generation must be a trained parent")
        if self.store.load_generation(generation.generation_id) != generation:
            raise ModelGenerationError("trained parent differs from durable generation")
        with self._control_lock:
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
        if self.store.load_lineage(lineage.lineage_id) != lineage:
            raise ModelGenerationError("candidate lineage differs from durable lineage")
        parent = self.store.load_generation(lineage.parent_generation_id)
        extended = self.store.load_generation(lineage.extended_generation_id)
        child = self.store.load_generation(lineage.child_generation_id)
        if (
            parent.kind is not GenerationKind.TRAINED_PARENT
            or extended.kind is not GenerationKind.EXTENDED_PARENT
            or child.kind is not GenerationKind.ADAPTED_CHILD
            or extended.parent_generation_id != parent.generation_id
            or child.parent_generation_id != extended.generation_id
            or extended.restoration_bundle_id != lineage.restoration_bundle_id
            or child.restoration_bundle_id != lineage.restoration_bundle_id
            or child.origin_proposal_semantic_id
            != lineage.origin_proposal_semantic_id
            or child.origin_proposal_provenance_id
            != lineage.origin_proposal_provenance_id
        ):
            raise ModelGenerationError("candidate generation chain is inconsistent")
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
        self.store.append_event(
            LifecycleEventKind.CANDIDATE_REJECTED,
            generation_id,
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
        if self.store.load_lineage(lineage.lineage_id) != lineage:
            raise ModelGenerationError("promotion lineage differs from durable lineage")
        if self.store.load_audit(audit.audit_id) != audit:
            raise ModelGenerationError("promotion audit differs from durable audit")
        child = self.store.load_generation(lineage.child_generation_id)
        if child.inference_artifact_id != audit.conformance.artifact_id:
            raise ModelGenerationError("promotion audit verified a different child artifact")
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
        if self.store.load_lineage(lineage.lineage_id) != lineage:
            raise ModelGenerationError("activation lineage differs from durable lineage")
        if self.store.load_audit(audit.audit_id) != audit:
            raise ModelGenerationError("activation audit differs from durable audit")
        child = self.store.load_generation(lineage.child_generation_id)
        bundle = self.store.load_restoration_bundle(lineage.restoration_bundle_id)
        if (
            child.inference_artifact_id != audit.conformance.artifact_id
            or child.restoration_bundle_id != bundle.bundle_id
            or bundle.parent_generation_id != lineage.parent_generation_id
        ):
            raise ModelGenerationError("activation objects do not form one generation chain")
        events = self.store.read_events()
        if (
            not events
            or events[-1].kind is not LifecycleEventKind.PROMOTION_APPROVED
            or events[-1].generation_id != lineage.child_generation_id
            or events[-1].details.get("lineage_id") != lineage.lineage_id
            or events[-1].details.get("audit_id") != audit.audit_id
        ):
            raise ModelGenerationError("activation lacks the durable promotion decision")
        with self._control_lock:
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
        if self.active_generation_id != child_generation_id:
            raise ModelGenerationError("reopen target is not the active generation")
        if not drift_requires_reopen(drift):
            raise ModelGenerationError("labeled drift does not justify reopen")
        self.store.append_event(
            LifecycleEventKind.REOPEN_REQUESTED,
            child_generation_id,
            drift_audit_id=drift.audit_id,
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
        stored_bundle = self.store.load_restoration_bundle(bundle.bundle_id)
        if stored_bundle != bundle:
            raise ModelGenerationError("restoration bundle changed after publication")
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
        with self._control_lock:
            self._active_generation_id = bundle.parent_generation_id
            try:
                self.store.append_event(
                    LifecycleEventKind.PARENT_RESTORED,
                    bundle.parent_generation_id,
                    restoration_bundle_id=bundle.bundle_id,
                )
            except Exception:
                self._active_generation_id = self.store.recover_active_generation()
                raise
        self._emit(
            "artifact_reopened",
            generation_id=bundle.parent_generation_id,
            restoration_bundle_id=bundle.bundle_id,
        )
        return RestoredAdaptiveParent(
            bundle.parent_generation_id,
            snapshot,
            manifest,
            preprocessing,
            artifact,
            machine,
        )


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
    collective: PTACollectiveService | None = None,
) -> tuple[PTAReasoningSession, PTAEscalationProposal, ReviewedThresholdProposal]:
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
    service = collective or PTACollectiveService()
    result = service.run(
        session,
        query=PTACollectiveQuery(
            numeric_fields=(numeric_field,),
            discover_intervals=False,
            derive_deescalation=False,
            derive_escalation=True,
        ),
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
    return session, proposal, reviewed


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
    controller: ModelGenerationController


def execute_trained_parent_lifecycle(
    *,
    parent_snapshot: TMSnapshot,
    parent_manifest: OrderedLiteralManifest,
    parent_training_corpus: LabeledCorpus,
    corpora: LifecycleCorpora,
    invention_session: PTAReasoningSession,
    reviewed: ReviewedThresholdProposal,
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
    if invention_session.dataset_id != corpora.invention.dataset_id:
        raise ModelGenerationError("threshold session does not belong to invention corpus")
    expected_observations = tuple(
        ("pta:input", example.example_id, field, value)
        for example in corpora.invention.examples
        for field, value in example.record.items()
    )
    expected_labels = tuple(
        (example.example_id, example.label)
        for example in corpora.invention.examples
    )
    if (
        tuple(invention_session.observations) != expected_observations
        or tuple(invention_session.example_labels) != expected_labels
    ):
        raise ModelGenerationError(
            "threshold session facts differ from the immutable invention corpus"
        )
    lifecycle_ids = {
        example.example_id
        for corpus in (
            corpora.invention,
            corpora.adaptation,
            corpora.promotion,
            corpora.live,
        )
        for example in corpus.examples
    }
    if lifecycle_ids & {
        example.example_id for example in parent_training_corpus.examples
    }:
        raise ModelGenerationError("parent training IDs overlap lifecycle corpora")

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

    equivalence_records = tuple(
        record
        for corpus in (
            parent_training_corpus,
            corpora.invention,
            corpora.adaptation,
            corpora.promotion,
            corpora.live,
        )
        for record in corpus.records
    )
    extended = extend_parent_with_threshold(
        parent_snapshot,
        parent_manifest,
        reviewed,
        session=invention_session,
        equivalence_records=equivalence_records,
    )
    store.put_snapshot(extended.snapshot)
    store.put_manifest(extended.manifest)
    extended_preprocessing = PreprocessingContract.from_catalog(
        extended.manifest.build_catalog()
    )
    extended_preprocessing_id, _ = store.put_preprocessing(extended_preprocessing)
    proposal = reviewed.proposal
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
        controller,
    )


def reopen_and_restore_for_drift(
    result: TrainedParentLifecycleResult,
    live_corpus: LabeledCorpus,
) -> tuple[PromotionAuditSnapshot, RestoredAdaptiveParent]:
    if live_corpus.role is not CorpusRole.LIVE:
        raise ModelGenerationError("reopen evaluation requires the live/drift corpus")
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
