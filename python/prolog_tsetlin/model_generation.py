"""Exact contracts for adaptive PTM model-generation lineage.

This layer sits above Class II consolidation.  A model generation may extend
its feature/source vocabulary, while Class II replacement deliberately keeps
the same source set.  Representation extension (``P -> P+``) is therefore
separate from behavior-changing adaptation (``P+ -> C``).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .model_artifact import PackedTMInferenceArtifact
from .preprocessing import PreprocessingContract
from .pta.threshold_artifact import (
    MaterializedThresholdClause,
    ReviewedThresholdProposal,
    materialize_threshold_clause,
)
from .pta.session import PTAReasoningSession
from .reference import (
    SNAPSHOT_SCHEMA_VERSION,
    ScalarBinaryTsetlinMachine,
    TMSnapshot,
    extend_snapshot_features,
)
from .representation import (
    FeatureSchema,
    FieldDefinition,
    FieldKind,
    LiteralCatalog,
    LiteralDescriptor,
    NullPolicy,
    TransformKind,
)


MODEL_GENERATION_SCHEMA_VERSION = 1
ORDERED_LITERAL_MANIFEST_SCHEMA = "ptm.ordered-literal-manifest.v1"
ADAPTIVE_SNAPSHOT_SCHEMA = "ptm.adaptive-snapshot.v1"
RESTORATION_BUNDLE_SCHEMA = "ptm.adaptive-restoration-bundle.v1"
ADAPTIVE_BEHAVIOR_SCHEMA = "ptm.adaptive-behavior.v2"
INVENTION_EVIDENCE_SCHEMA = "ptm.gnu-prolog-invention-evidence.v1"
GENERATION_SCHEMA = "ptm.model-generation.v1"
PROMOTION_AUDIT_SCHEMA = "ptm.promotion-audit.v1"
LIVE_CONFORMANCE_SCHEMA = "ptm.live-runtime-conformance.v1"
LINEAGE_SCHEMA = "ptm.model-generation-lineage.v3"
TRAINING_SEMANTICS_VERSION = "ptm.scalar-binary-training.v1"
PYTHON_RNG_ALGORITHM = "python.random-mt19937-state-v1"
MAX_CORPUS_EXAMPLES = 2_048
MAX_ADAPTATION_EPOCHS = 10_000


class ModelGenerationError(ValueError):
    """Raised when a generation contract fails closed."""


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("JSON object keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in (bool, int, float, str):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("non-finite JSON numbers are forbidden")
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _tuple_tree(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _thaw_json(value),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ModelGenerationError(f"{label} must be a canonical SHA-256 ID")
    return value


@dataclass(frozen=True, slots=True)
class CorpusExample:
    example_id: str | int
    record: Mapping[str, object]
    label: int

    def __post_init__(self) -> None:
        if type(self.example_id) not in (str, int) or self.example_id == "":
            raise ModelGenerationError("example_id must be a nonempty string or integer")
        if type(self.label) is not int or self.label not in (0, 1):
            raise ModelGenerationError("corpus labels must be strict integers 0 or 1")
        if not isinstance(self.record, Mapping):
            raise TypeError("corpus record must be a mapping")
        object.__setattr__(self, "record", _freeze_json(self.record))

    def to_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "record": _thaw_json(self.record),
            "label": self.label,
        }


class CorpusRole(str, Enum):
    PARENT_TRAINING = "parent_training"
    INVENTION = "invention"
    ADAPTATION = "adaptation"
    PROMOTION = "promotion_holdout"
    LIVE = "live_drift"


@dataclass(frozen=True, slots=True)
class LabeledCorpus:
    dataset_id: str
    role: CorpusRole
    examples: tuple[CorpusExample, ...]

    def __post_init__(self) -> None:
        if type(self.dataset_id) is not str or not self.dataset_id:
            raise ModelGenerationError("dataset_id must be a nonempty string")
        if not isinstance(self.role, CorpusRole):
            raise TypeError("role must be CorpusRole")
        if type(self.examples) is not tuple:
            raise TypeError("examples must be a tuple")
        if not 0 < len(self.examples) <= MAX_CORPUS_EXAMPLES:
            raise ModelGenerationError("corpus size lies outside the bounded contract")
        if any(not isinstance(item, CorpusExample) for item in self.examples):
            raise TypeError("examples must contain CorpusExample values")
        identifiers = tuple(item.example_id for item in self.examples)
        if len(set(identifiers)) != len(identifiers):
            raise ModelGenerationError("corpus example IDs must be unique")

    @property
    def digest(self) -> str:
        return content_digest(self.canonical_payload())

    @property
    def records(self) -> tuple[Mapping[str, object], ...]:
        return tuple(item.record for item in self.examples)

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(item.label for item in self.examples)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "role": self.role.value,
            "examples": [item.to_dict() for item in self.examples],
        }


@dataclass(frozen=True, slots=True)
class LifecycleCorpora:
    invention: LabeledCorpus
    adaptation: LabeledCorpus
    promotion: LabeledCorpus

    def __post_init__(self) -> None:
        expected = (
            (self.invention, CorpusRole.INVENTION),
            (self.adaptation, CorpusRole.ADAPTATION),
            (self.promotion, CorpusRole.PROMOTION),
        )
        if any(corpus.role is not role for corpus, role in expected):
            raise ModelGenerationError("lifecycle corpus role is misplaced")
        dataset_ids = {corpus.dataset_id for corpus, _ in expected}
        if len(dataset_ids) != 1:
            raise ModelGenerationError("lifecycle corpora must share one dataset ID")
        seen: set[str | int] = set()
        for corpus, _ in expected:
            identifiers = {item.example_id for item in corpus.examples}
            if seen & identifiers:
                raise ModelGenerationError("lifecycle corpus example IDs overlap")
            seen.update(identifiers)
        # Invention, adaptation, and promotion must not reuse identical labeled
        # rows. Live observations are deliberately absent until post-activation.
        fingerprints: set[str] = set()
        for corpus in (self.invention, self.adaptation, self.promotion):
            current = {
                content_digest({"record": item.record, "label": item.label})
                for item in corpus.examples
            }
            if fingerprints & current:
                raise ModelGenerationError(
                    "invention, adaptation, and promotion rows must be independent"
                )
            fingerprints.update(current)

    @property
    def digests(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (corpus.role.value, corpus.digest)
            for corpus in (
                self.invention,
                self.adaptation,
                self.promotion,
            )
        )


@dataclass(frozen=True, slots=True)
class PrologInventionEvidence:
    invention_corpus_digest: str
    session_digest: str
    numeric_field: str
    collective_protocol: str
    gprolog_version: str
    gprolog_binary_digest: str
    module_digests: tuple[tuple[str, str], ...]
    proposal_semantic_id: str
    proposal_provenance_id: str
    schema: str = INVENTION_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != INVENTION_EVIDENCE_SCHEMA:
            raise ModelGenerationError("GNU Prolog invention evidence schema is unsupported")
        for label, value in (
            ("invention corpus", self.invention_corpus_digest),
            ("reasoning session", self.session_digest),
            ("GNU Prolog executable", self.gprolog_binary_digest),
            ("proposal semantic", self.proposal_semantic_id),
            ("proposal provenance", self.proposal_provenance_id),
        ):
            _require_digest(value, label)
        if type(self.numeric_field) is not str or not self.numeric_field:
            raise ModelGenerationError("invention evidence numeric field is invalid")
        if self.collective_protocol != "PTM_PTA_COLLECTIVE_V1":
            raise ModelGenerationError("invention evidence collective protocol is unsupported")
        if (
            type(self.gprolog_version) is not str
            or not self.gprolog_version
            or len(self.gprolog_version) > 1_024
            or any(ord(character) < 0x20 for character in self.gprolog_version)
        ):
            raise ModelGenerationError("GNU Prolog version evidence is invalid")
        if (
            type(self.module_digests) is not tuple
            or not self.module_digests
            or tuple(sorted(self.module_digests)) != self.module_digests
            or len({name for name, _ in self.module_digests})
            != len(self.module_digests)
        ):
            raise ModelGenerationError("Prolog module evidence is not canonical")
        for name, digest in self.module_digests:
            if type(name) is not str or not name:
                raise ModelGenerationError("Prolog module name is invalid")
            _require_digest(digest, f"{name} module")

    @property
    def query(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "numeric_fields": (self.numeric_field,),
                "discover_thresholds": True,
                "discover_intervals": False,
                "derive_deescalation": False,
                "derive_escalation": True,
            }
        )

    @property
    def evidence_id(self) -> str:
        return content_digest(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "invention_corpus_digest": self.invention_corpus_digest,
            "session_digest": self.session_digest,
            "query": _thaw_json(self.query),
            "collective_protocol": self.collective_protocol,
            "gprolog_version": self.gprolog_version,
            "gprolog_binary_digest": self.gprolog_binary_digest,
            "module_digests": [list(item) for item in self.module_digests],
            "proposal_semantic_id": self.proposal_semantic_id,
            "proposal_provenance_id": self.proposal_provenance_id,
        }

    def to_dict(self) -> dict[str, object]:
        result = self.canonical_payload()
        result["evidence_id"] = self.evidence_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PrologInventionEvidence":
        expected = {
            "schema",
            "invention_corpus_digest",
            "session_digest",
            "query",
            "collective_protocol",
            "gprolog_version",
            "gprolog_binary_digest",
            "module_digests",
            "proposal_semantic_id",
            "proposal_provenance_id",
            "evidence_id",
        }
        string_fields = expected - {"query", "module_digests"}
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or any(type(value[key]) is not str for key in string_fields)
            or not isinstance(value["query"], Mapping)
            or not isinstance(value["module_digests"], list)
            or any(
                not isinstance(item, list)
                or len(item) != 2
                or any(type(part) is not str for part in item)
                for item in value["module_digests"]
            )
        ):
            raise ModelGenerationError("GNU Prolog invention evidence is malformed")
        query = value["query"]
        if (
            set(query)
            != {
                "numeric_fields",
                "discover_thresholds",
                "discover_intervals",
                "derive_deescalation",
                "derive_escalation",
            }
            or not isinstance(query["numeric_fields"], list)
            or len(query["numeric_fields"]) != 1
            or type(query["numeric_fields"][0]) is not str
            or query["discover_thresholds"] is not True
            or query["discover_intervals"] is not False
            or query["derive_deescalation"] is not False
            or query["derive_escalation"] is not True
        ):
            raise ModelGenerationError("GNU Prolog invention query is malformed")
        result = cls(
            invention_corpus_digest=value["invention_corpus_digest"],
            session_digest=value["session_digest"],
            numeric_field=query["numeric_fields"][0],
            collective_protocol=value["collective_protocol"],
            gprolog_version=value["gprolog_version"],
            gprolog_binary_digest=value["gprolog_binary_digest"],
            module_digests=tuple(tuple(item) for item in value["module_digests"]),
            proposal_semantic_id=value["proposal_semantic_id"],
            proposal_provenance_id=value["proposal_provenance_id"],
            schema=value["schema"],
        )
        if result.evidence_id != value["evidence_id"]:
            raise ModelGenerationError("GNU Prolog invention evidence digest mismatch")
        return result


@dataclass(frozen=True, slots=True)
class OrderedLiteralManifest:
    fields: tuple[FieldDefinition, ...]
    literals: tuple[LiteralDescriptor, ...]
    schema: str = ORDERED_LITERAL_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ORDERED_LITERAL_MANIFEST_SCHEMA:
            raise ModelGenerationError("ordered literal manifest schema is unsupported")
        for field in self.fields:
            if (
                not isinstance(field, FieldDefinition)
                or FieldDefinition.create(field.name, field.kind) != field
            ):
                raise ModelGenerationError("feature field identity is not canonical")
        catalog = LiteralCatalog.from_descriptors(
            FeatureSchema(self.fields), self.literals
        )
        if catalog.literals != self.literals:
            raise ModelGenerationError("literal manifest lost positional identity")

    @classmethod
    def from_catalog(cls, catalog: LiteralCatalog) -> "OrderedLiteralManifest":
        if not isinstance(catalog, LiteralCatalog):
            raise TypeError("catalog must be LiteralCatalog")
        return cls(catalog.schema.fields, catalog.literals)

    @property
    def manifest_id(self) -> str:
        return content_digest(self.canonical_payload())

    @property
    def literal_ids(self) -> tuple[int, ...]:
        return tuple(item.literal_id for item in self.literals)

    def build_catalog(self) -> LiteralCatalog:
        return LiteralCatalog.from_descriptors(
            FeatureSchema(self.fields), self.literals
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fields": [
                {
                    "source_field_id": str(field.source_field_id),
                    "name": field.name,
                    "kind": field.kind.value,
                }
                for field in self.fields
            ],
            "literals": [
                {
                    "literal_id": str(item.literal_id),
                    "source_field_id": str(item.source_field_id),
                    "source_field": item.source_field,
                    "transform": item.transform.value,
                    "parameters": _thaw_json(dict(item.parameters)),
                    "null_policy": item.null_policy.value,
                    "catalog_version": item.catalog_version,
                }
                for item in self.literals
            ],
        }

    def to_dict(self) -> dict[str, object]:
        result = self.canonical_payload()
        result["manifest_id"] = self.manifest_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "OrderedLiteralManifest":
        if not isinstance(value, Mapping):
            raise TypeError("literal manifest must be a mapping")
        if set(value) != {"schema", "fields", "literals", "manifest_id"}:
            raise ModelGenerationError("literal manifest fields are not canonical")
        raw_fields = value["fields"]
        raw_literals = value["literals"]
        if not isinstance(raw_fields, list) or not isinstance(raw_literals, list):
            raise ModelGenerationError("literal manifest arrays are malformed")
        fields: list[FieldDefinition] = []
        for raw in raw_fields:
            if not isinstance(raw, Mapping) or set(raw) != {
                "source_field_id", "name", "kind"
            }:
                raise ModelGenerationError("literal manifest field is malformed")
            if any(type(raw[key]) is not str for key in ("source_field_id", "name", "kind")):
                raise ModelGenerationError("literal manifest field types are malformed")
            try:
                field = FieldDefinition.create(raw["name"], FieldKind(raw["kind"]))
            except (TypeError, ValueError) as error:
                raise ModelGenerationError("literal manifest field is malformed") from error
            if str(field.source_field_id) != raw["source_field_id"]:
                raise ModelGenerationError("source field identity does not match semantics")
            fields.append(field)
        catalog = LiteralCatalog(FeatureSchema(fields))
        for raw in raw_literals:
            if not isinstance(raw, Mapping) or set(raw) != {
                "literal_id",
                "source_field_id",
                "source_field",
                "transform",
                "parameters",
                "null_policy",
                "catalog_version",
            }:
                raise ModelGenerationError("literal manifest descriptor is malformed")
            parameters = raw["parameters"]
            if not isinstance(parameters, Mapping):
                raise ModelGenerationError("literal parameters are malformed")
            if (
                any(
                    type(raw[key]) is not str
                    for key in (
                        "literal_id",
                        "source_field_id",
                        "source_field",
                        "transform",
                        "null_policy",
                    )
                )
                or type(raw["catalog_version"]) is not int
            ):
                raise ModelGenerationError("literal manifest descriptor types are malformed")
            try:
                descriptor = catalog.preview(
                    raw["source_field"],
                    TransformKind(raw["transform"]),
                    parameters,
                    NullPolicy(raw["null_policy"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ModelGenerationError("literal manifest descriptor is malformed") from error
            if (
                str(descriptor.literal_id) != raw["literal_id"]
                or str(descriptor.source_field_id) != raw["source_field_id"]
                or descriptor.catalog_version != raw["catalog_version"]
            ):
                raise ModelGenerationError("literal identity does not match semantics")
            catalog.register_descriptor(descriptor)
        if type(value["schema"]) is not str or type(value["manifest_id"]) is not str:
            raise ModelGenerationError("literal manifest identity fields are malformed")
        result = cls.from_catalog(catalog)
        if result.schema != value["schema"] or result.manifest_id != value["manifest_id"]:
            raise ModelGenerationError("literal manifest digest mismatch")
        return result


def _validate_adaptive_snapshot(snapshot: TMSnapshot) -> None:
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ModelGenerationError("adaptive snapshot schema is unsupported")
    machine = ScalarBinaryTsetlinMachine(
        snapshot.number_of_clauses,
        snapshot.number_of_features,
        states_per_action=snapshot.states_per_action,
        specificity=snapshot.specificity,
        threshold=snapshot.threshold,
        seed=0,
    )
    try:
        machine.restore(snapshot)
    except (TypeError, ValueError) as error:
        raise ModelGenerationError("adaptive snapshot is invalid") from error


@dataclass(frozen=True, slots=True)
class AdaptiveSnapshotEnvelope:
    snapshot: TMSnapshot
    rng_algorithm: str = PYTHON_RNG_ALGORITHM
    schema: str = ADAPTIVE_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ADAPTIVE_SNAPSHOT_SCHEMA:
            raise ModelGenerationError("adaptive snapshot envelope is unsupported")
        if self.rng_algorithm != PYTHON_RNG_ALGORITHM:
            raise ModelGenerationError("adaptive snapshot RNG algorithm is unsupported")
        _validate_adaptive_snapshot(self.snapshot)

    @property
    def snapshot_id(self) -> str:
        return content_digest(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        snapshot = self.snapshot
        return {
            "schema": self.schema,
            "rng_algorithm": self.rng_algorithm,
            "snapshot": {
                "schema_version": snapshot.schema_version,
                "number_of_clauses": snapshot.number_of_clauses,
                "number_of_features": snapshot.number_of_features,
                "states_per_action": snapshot.states_per_action,
                "specificity": snapshot.specificity,
                "threshold": snapshot.threshold,
                "states": [list(row) for row in snapshot.states],
                "rng_state": _thaw_json(snapshot.rng_state),
            },
        }

    def to_dict(self) -> dict[str, object]:
        result = self.canonical_payload()
        result["snapshot_id"] = self.snapshot_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AdaptiveSnapshotEnvelope":
        if not isinstance(value, Mapping) or set(value) != {
            "schema", "rng_algorithm", "snapshot", "snapshot_id"
        }:
            raise ModelGenerationError("adaptive snapshot envelope is malformed")
        raw = value["snapshot"]
        if not isinstance(raw, Mapping) or set(raw) != {
            "schema_version",
            "number_of_clauses",
            "number_of_features",
            "states_per_action",
            "specificity",
            "threshold",
            "states",
            "rng_state",
        }:
            raise ModelGenerationError("adaptive snapshot payload is malformed")
        integer_fields = (
            "schema_version",
            "number_of_clauses",
            "number_of_features",
            "states_per_action",
            "threshold",
        )
        if (
            any(type(raw[name]) is not int for name in integer_fields)
            or type(raw["specificity"]) not in (int, float)
            or not math.isfinite(float(raw["specificity"]))
            or not isinstance(raw["states"], list)
            or not isinstance(raw["rng_state"], list)
            or type(value["schema"]) is not str
            or type(value["rng_algorithm"]) is not str
            or type(value["snapshot_id"]) is not str
        ):
            raise ModelGenerationError("adaptive snapshot payload types are malformed")
        if any(
            not isinstance(row, list)
            or any(type(state) is not int for state in row)
            for row in raw["states"]
        ):
            raise ModelGenerationError("adaptive snapshot states are malformed")
        try:
            snapshot = TMSnapshot(
                schema_version=raw["schema_version"],
                number_of_clauses=raw["number_of_clauses"],
                number_of_features=raw["number_of_features"],
                states_per_action=raw["states_per_action"],
                # Preserve the serialized numeric type. Canonical JSON (and
                # therefore the content address) intentionally distinguishes
                # an integer specificity from its floating-point spelling.
                specificity=raw["specificity"],
                threshold=raw["threshold"],
                states=tuple(tuple(row) for row in raw["states"]),  # type: ignore[arg-type]
                rng_state=_tuple_tree(raw["rng_state"]),
            )
        except (TypeError, ValueError) as error:
            raise ModelGenerationError("adaptive snapshot payload is malformed") from error
        result = cls(snapshot, value["rng_algorithm"], value["schema"])
        if result.snapshot_id != value["snapshot_id"]:
            raise ModelGenerationError("adaptive snapshot digest mismatch")
        return result


class GenerationKind(str, Enum):
    TRAINED_PARENT = "trained_parent"
    EXTENDED_PARENT = "extended_parent"
    ADAPTED_CHILD = "adapted_child"


@dataclass(frozen=True, slots=True)
class ModelGeneration:
    kind: GenerationKind
    snapshot_id: str
    literal_manifest_id: str
    preprocessing_contract_id: str
    inference_artifact_id: str | None
    parent_generation_id: str | None
    restoration_bundle_id: str | None
    corpus_digests: tuple[tuple[str, str], ...]
    origin_proposal_semantic_id: str | None = None
    origin_proposal_provenance_id: str | None = None
    schema: str = GENERATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GENERATION_SCHEMA or not isinstance(self.kind, GenerationKind):
            raise ModelGenerationError("model generation schema or kind is invalid")
        for label, value in (
            ("snapshot_id", self.snapshot_id),
            ("literal_manifest_id", self.literal_manifest_id),
            ("preprocessing_contract_id", self.preprocessing_contract_id),
        ):
            _require_digest(value, label)
        for label, value in (
            ("inference_artifact_id", self.inference_artifact_id),
            ("parent_generation_id", self.parent_generation_id),
            ("restoration_bundle_id", self.restoration_bundle_id),
            ("origin proposal semantic ID", self.origin_proposal_semantic_id),
            ("origin proposal provenance ID", self.origin_proposal_provenance_id),
        ):
            if value is not None:
                _require_digest(value, label)
        if type(self.corpus_digests) is not tuple:
            raise TypeError("corpus_digests must be a tuple")
        if len({name for name, _ in self.corpus_digests}) != len(self.corpus_digests):
            raise ModelGenerationError("corpus digest roles must be unique")
        for name, digest in self.corpus_digests:
            if type(name) is not str or not name:
                raise ModelGenerationError("corpus digest role must be nonempty")
            _require_digest(digest, f"{name} corpus digest")

    @property
    def generation_id(self) -> str:
        return content_digest(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "kind": self.kind.value,
            "snapshot_id": self.snapshot_id,
            "literal_manifest_id": self.literal_manifest_id,
            "preprocessing_contract_id": self.preprocessing_contract_id,
            "inference_artifact_id": self.inference_artifact_id,
            "parent_generation_id": self.parent_generation_id,
            "restoration_bundle_id": self.restoration_bundle_id,
            "corpus_digests": [list(item) for item in self.corpus_digests],
            "origin_proposal_semantic_id": self.origin_proposal_semantic_id,
            "origin_proposal_provenance_id": self.origin_proposal_provenance_id,
        }

    def to_dict(self) -> dict[str, object]:
        result = self.canonical_payload()
        result["generation_id"] = self.generation_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ModelGeneration":
        expected = {
            "schema",
            "kind",
            "snapshot_id",
            "literal_manifest_id",
            "preprocessing_contract_id",
            "inference_artifact_id",
            "parent_generation_id",
            "restoration_bundle_id",
            "corpus_digests",
            "origin_proposal_semantic_id",
            "origin_proposal_provenance_id",
            "generation_id",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ModelGenerationError("model generation is malformed")
        required_strings = (
            "schema",
            "kind",
            "snapshot_id",
            "literal_manifest_id",
            "preprocessing_contract_id",
            "generation_id",
        )
        optional_strings = (
            "inference_artifact_id",
            "parent_generation_id",
            "restoration_bundle_id",
            "origin_proposal_semantic_id",
            "origin_proposal_provenance_id",
        )
        raw_corpora = value["corpus_digests"]
        if (
            any(type(value[key]) is not str for key in required_strings)
            or any(
                value[key] is not None and type(value[key]) is not str
                for key in optional_strings
            )
            or not isinstance(raw_corpora, list)
            or any(
                not isinstance(item, list)
                or len(item) != 2
                or any(type(part) is not str for part in item)
                for item in raw_corpora
            )
        ):
            raise ModelGenerationError("model generation field types are malformed")
        try:
            result = cls(
                kind=GenerationKind(value["kind"]),
                snapshot_id=value["snapshot_id"],
                literal_manifest_id=value["literal_manifest_id"],
                preprocessing_contract_id=value["preprocessing_contract_id"],
                inference_artifact_id=value["inference_artifact_id"],
                parent_generation_id=value["parent_generation_id"],
                restoration_bundle_id=value["restoration_bundle_id"],
                corpus_digests=tuple(tuple(item) for item in raw_corpora),
                origin_proposal_semantic_id=value["origin_proposal_semantic_id"],
                origin_proposal_provenance_id=value["origin_proposal_provenance_id"],
                schema=value["schema"],
            )
        except (TypeError, ValueError) as error:
            raise ModelGenerationError("model generation is malformed") from error
        if result.generation_id != value["generation_id"]:
            raise ModelGenerationError("model generation digest mismatch")
        return result


@dataclass(frozen=True, slots=True)
class AdaptiveBehaviorIdentity:
    snapshot_id: str
    literal_manifest_id: str
    preprocessing_contract_id: str
    training_semantics_version: str = TRAINING_SEMANTICS_VERSION
    schema: str = ADAPTIVE_BEHAVIOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ADAPTIVE_BEHAVIOR_SCHEMA:
            raise ModelGenerationError("adaptive behavior schema is unsupported")
        if self.training_semantics_version != TRAINING_SEMANTICS_VERSION:
            raise ModelGenerationError("adaptive behavior training semantics are unsupported")
        for label, value in (
            ("snapshot", self.snapshot_id),
            ("literal manifest", self.literal_manifest_id),
            ("preprocessing contract", self.preprocessing_contract_id),
        ):
            _require_digest(value, label)

    @classmethod
    def from_generation(cls, generation: ModelGeneration) -> "AdaptiveBehaviorIdentity":
        if generation.kind is not GenerationKind.ADAPTED_CHILD:
            raise ModelGenerationError("adaptive behavior requires an adapted child")
        return cls(
            generation.snapshot_id,
            generation.literal_manifest_id,
            generation.preprocessing_contract_id,
        )

    @classmethod
    def from_child(
        cls,
        child: "AdaptedChild",
        *,
        preprocessing_contract_id: str,
    ) -> "AdaptiveBehaviorIdentity":
        return cls(
            child.snapshot.snapshot_id,
            child.manifest.manifest_id,
            preprocessing_contract_id,
        )

    @property
    def behavior_id(self) -> str:
        return content_digest(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "snapshot_id": self.snapshot_id,
            "literal_manifest_id": self.literal_manifest_id,
            "preprocessing_contract_id": self.preprocessing_contract_id,
            "training_semantics_version": self.training_semantics_version,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveRestorationBundle:
    parent_generation_id: str
    adaptive_snapshot_id: str
    ordered_literal_manifest_id: str
    preprocessing_contract_id: str
    deployed_parent_artifact_id: str
    parent_training_corpus_digest: str
    training_semantics_version: str = TRAINING_SEMANTICS_VERSION
    rng_algorithm: str = PYTHON_RNG_ALGORITHM
    schema: str = RESTORATION_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESTORATION_BUNDLE_SCHEMA:
            raise ModelGenerationError("restoration bundle schema is unsupported")
        if self.training_semantics_version != TRAINING_SEMANTICS_VERSION:
            raise ModelGenerationError("training semantics version is unsupported")
        if self.rng_algorithm != PYTHON_RNG_ALGORITHM:
            raise ModelGenerationError("restoration RNG algorithm is unsupported")
        for label, value in (
            ("parent_generation_id", self.parent_generation_id),
            ("adaptive_snapshot_id", self.adaptive_snapshot_id),
            ("ordered_literal_manifest_id", self.ordered_literal_manifest_id),
            ("preprocessing_contract_id", self.preprocessing_contract_id),
            ("deployed_parent_artifact_id", self.deployed_parent_artifact_id),
            ("parent_training_corpus_digest", self.parent_training_corpus_digest),
        ):
            _require_digest(value, label)

    @property
    def bundle_id(self) -> str:
        return content_digest(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "parent_generation_id": self.parent_generation_id,
            "adaptive_snapshot_id": self.adaptive_snapshot_id,
            "ordered_literal_manifest_id": self.ordered_literal_manifest_id,
            "preprocessing_contract_id": self.preprocessing_contract_id,
            "deployed_parent_artifact_id": self.deployed_parent_artifact_id,
            "parent_training_corpus_digest": self.parent_training_corpus_digest,
            "training_semantics_version": self.training_semantics_version,
            "rng_algorithm": self.rng_algorithm,
        }

    def to_dict(self) -> dict[str, object]:
        result = self.canonical_payload()
        result["bundle_id"] = self.bundle_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AdaptiveRestorationBundle":
        expected = {
            "schema",
            "parent_generation_id",
            "adaptive_snapshot_id",
            "ordered_literal_manifest_id",
            "preprocessing_contract_id",
            "deployed_parent_artifact_id",
            "parent_training_corpus_digest",
            "training_semantics_version",
            "rng_algorithm",
            "bundle_id",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ModelGenerationError("restoration bundle is malformed")
        if any(type(value[key]) is not str for key in expected):
            raise ModelGenerationError("restoration bundle fields must be strings")
        result = cls(
            parent_generation_id=value["parent_generation_id"],
            adaptive_snapshot_id=value["adaptive_snapshot_id"],
            ordered_literal_manifest_id=value["ordered_literal_manifest_id"],
            preprocessing_contract_id=value["preprocessing_contract_id"],
            deployed_parent_artifact_id=value["deployed_parent_artifact_id"],
            parent_training_corpus_digest=value["parent_training_corpus_digest"],
            training_semantics_version=value["training_semantics_version"],
            rng_algorithm=value["rng_algorithm"],
            schema=value["schema"],
        )
        if result.bundle_id != value["bundle_id"]:
            raise ModelGenerationError("restoration bundle digest mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ExtendedParent:
    parent_snapshot: AdaptiveSnapshotEnvelope
    parent_manifest: OrderedLiteralManifest
    snapshot: AdaptiveSnapshotEnvelope
    manifest: OrderedLiteralManifest
    materialized: MaterializedThresholdClause
    equivalence_case_count: int


@dataclass(frozen=True, slots=True)
class AdaptedChild:
    snapshot: AdaptiveSnapshotEnvelope
    manifest: OrderedLiteralManifest
    adaptation_corpus_digest: str
    epochs: int

    def __post_init__(self) -> None:
        _require_digest(self.adaptation_corpus_digest, "adaptation corpus digest")
        if type(self.epochs) is not int or not 0 < self.epochs <= MAX_ADAPTATION_EPOCHS:
            raise ModelGenerationError("adapted child epoch count is invalid")
        if self.snapshot.snapshot.number_of_features != len(self.manifest.literals):
            raise ModelGenerationError("adapted child snapshot and manifest widths differ")


def _machine_from_snapshot(snapshot: TMSnapshot) -> ScalarBinaryTsetlinMachine:
    machine = ScalarBinaryTsetlinMachine(
        snapshot.number_of_clauses,
        snapshot.number_of_features,
        states_per_action=snapshot.states_per_action,
        specificity=snapshot.specificity,
        threshold=snapshot.threshold,
        seed=0,
    )
    machine.restore(snapshot)
    return machine


def preprocessing_contract_id(contract: PreprocessingContract) -> str:
    if not isinstance(contract, PreprocessingContract):
        raise TypeError("contract must be PreprocessingContract")
    return content_digest(contract.to_dict())


def extend_parent_with_threshold(
    parent_snapshot: TMSnapshot,
    parent_manifest: OrderedLiteralManifest,
    reviewed: ReviewedThresholdProposal,
    *,
    session: PTAReasoningSession,
    equivalence_records: Sequence[Mapping[str, object]],
) -> ExtendedParent:
    """Approve one threshold into an append-only, behavior-identical ``P+``."""

    if parent_snapshot.number_of_features != len(parent_manifest.literals):
        raise ModelGenerationError("parent snapshot and literal manifest widths differ")
    parent_catalog = parent_manifest.build_catalog()
    if reviewed.descriptor.literal_id in parent_manifest.literal_ids:
        raise ModelGenerationError("approved literal already exists in the parent")
    extended_catalog = parent_catalog.clone()
    materialized = materialize_threshold_clause(
        reviewed, session=session, catalog=extended_catalog
    )
    extended_manifest = OrderedLiteralManifest.from_catalog(extended_catalog)
    if extended_manifest.literals[:-1] != parent_manifest.literals:
        raise ModelGenerationError("representation extension changed existing positions")
    if extended_manifest.literals[-1] != materialized.descriptor:
        raise ModelGenerationError("approved literal was not appended at the final position")

    extended_snapshot = extend_snapshot_features(parent_snapshot, 1)
    for old_row, new_row in zip(parent_snapshot.states, extended_snapshot.states):
        if new_row[: len(old_row)] != old_row:
            raise ModelGenerationError("representation extension changed existing TA state")
        if new_row[-2:] != (
            parent_snapshot.states_per_action,
            parent_snapshot.states_per_action,
        ):
            raise ModelGenerationError("new feature TAs are not deterministically excluded")
    if extended_snapshot.rng_state != parent_snapshot.rng_state:
        raise ModelGenerationError("representation extension changed the RNG state")

    records = tuple(equivalence_records)
    if not records:
        raise ModelGenerationError("P+ equivalence oracle requires at least one record")
    parent_batch = parent_catalog.encode(records).ta
    extended_batch = extended_catalog.encode(records).ta
    parent_machine = _machine_from_snapshot(parent_snapshot)
    extended_machine = _machine_from_snapshot(extended_snapshot)
    parent_rows = tuple(
        parent_batch.row_values(index) for index in range(parent_batch.row_count)
    )
    extended_rows = tuple(
        extended_batch.row_values(index) for index in range(extended_batch.row_count)
    )
    parent_results = tuple(
        (parent_machine.score(row), parent_machine.predict_one(row))
        for row in parent_rows
    )
    extended_results = tuple(
        (extended_machine.score(row), extended_machine.predict_one(row))
        for row in extended_rows
    )
    if parent_results != extended_results:
        raise ModelGenerationError("P+ behavioral equivalence oracle failed")
    return ExtendedParent(
        AdaptiveSnapshotEnvelope(parent_snapshot),
        parent_manifest,
        AdaptiveSnapshotEnvelope(extended_snapshot),
        extended_manifest,
        materialized,
        len(records),
    )


def adapt_extended_parent(
    extended: ExtendedParent,
    corpus: LabeledCorpus,
    *,
    epochs: int,
) -> AdaptedChild:
    """Create ``C`` by bounded adaptation of an immutable ``P+`` snapshot."""

    if corpus.role is not CorpusRole.ADAPTATION:
        raise ModelGenerationError("child adaptation requires the adaptation corpus")
    if type(epochs) is not int or not 0 < epochs <= MAX_ADAPTATION_EPOCHS:
        raise ModelGenerationError("adaptation epochs lie outside the bounded contract")
    catalog = extended.manifest.build_catalog()
    batch = catalog.encode(corpus.records).ta
    before = extended.snapshot.snapshot
    machine = _machine_from_snapshot(before)
    machine.fit_literal_batch(batch, corpus.labels, epochs=epochs)
    child = machine.snapshot()
    if extended.snapshot.snapshot != before:
        raise ModelGenerationError("adaptation mutated the frozen P+ snapshot")
    return AdaptedChild(
        AdaptiveSnapshotEnvelope(child),
        extended.manifest,
        corpus.digest,
        epochs,
    )


@dataclass(frozen=True, slots=True)
class ClassPromotionCounts:
    label: int
    observed: int
    both_correct: int
    both_wrong: int
    improvements: int
    regressions: int

    def __post_init__(self) -> None:
        if type(self.label) is not int or self.label not in (0, 1):
            raise ModelGenerationError("promotion class label must be 0 or 1")
        counts = (
            self.observed,
            self.both_correct,
            self.both_wrong,
            self.improvements,
            self.regressions,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ModelGenerationError("promotion class counts must be nonnegative integers")
        if sum(counts[1:]) != self.observed:
            raise ModelGenerationError("promotion class counts do not partition observations")

    def to_dict(self) -> dict[str, int]:
        return {
            "label": self.label,
            "observed": self.observed,
            "both_correct": self.both_correct,
            "both_wrong": self.both_wrong,
            "improvements": self.improvements,
            "regressions": self.regressions,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ClassPromotionCounts":
        expected = {
            "label",
            "observed",
            "both_correct",
            "both_wrong",
            "improvements",
            "regressions",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or any(type(value[key]) is not int for key in expected)
        ):
            raise ModelGenerationError("promotion class counts are malformed")
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class PromotionAuditPolicy:
    minimum_observations: int
    require_strict_improvement: bool = True
    maximum_regressions: int = 0

    def __post_init__(self) -> None:
        if type(self.minimum_observations) is not int or self.minimum_observations <= 0:
            raise ModelGenerationError("promotion minimum observations must be positive")
        if type(self.require_strict_improvement) is not bool:
            raise TypeError("require_strict_improvement must be boolean")
        if type(self.maximum_regressions) is not int or self.maximum_regressions < 0:
            raise ModelGenerationError("promotion regression budget cannot be negative")


@dataclass(frozen=True, slots=True)
class DriftAuditPolicy:
    minimum_observations: int
    minimum_regressions: int
    minimum_regression_rate: float
    minimum_error_increase: int
    minimum_observations_per_class: int

    def __post_init__(self) -> None:
        integer_fields = (
            self.minimum_observations,
            self.minimum_regressions,
            self.minimum_error_increase,
            self.minimum_observations_per_class,
        )
        if any(type(value) is not int or value <= 0 for value in integer_fields):
            raise ModelGenerationError("drift policy count thresholds must be positive")
        if (
            type(self.minimum_regression_rate) not in (int, float)
            or not math.isfinite(float(self.minimum_regression_rate))
            or not 0.0 <= float(self.minimum_regression_rate) <= 1.0
        ):
            raise ModelGenerationError("drift policy regression rate is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_observations": self.minimum_observations,
            "minimum_regressions": self.minimum_regressions,
            "minimum_regression_rate": self.minimum_regression_rate,
            "minimum_error_increase": self.minimum_error_increase,
            "minimum_observations_per_class": self.minimum_observations_per_class,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DriftAuditPolicy":
        expected = {
            "minimum_observations",
            "minimum_regressions",
            "minimum_regression_rate",
            "minimum_error_increase",
            "minimum_observations_per_class",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ModelGenerationError("drift policy is malformed")
        try:
            return cls(**value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ModelGenerationError("drift policy is malformed") from error


@dataclass(frozen=True, slots=True)
class RuntimeConformanceReport:
    artifact_id: str
    case_count: int
    reference_packed_mismatches: int
    ptmrt_verified: bool
    ptmrt_artifact_id: str | None

    def __post_init__(self) -> None:
        _require_digest(self.artifact_id, "artifact_id")
        if type(self.case_count) is not int or self.case_count <= 0:
            raise ModelGenerationError("conformance case count must be positive")
        if (
            type(self.reference_packed_mismatches) is not int
            or self.reference_packed_mismatches < 0
            or self.reference_packed_mismatches > self.case_count
        ):
            raise ModelGenerationError("conformance mismatch count is invalid")
        if type(self.ptmrt_verified) is not bool:
            raise TypeError("ptmrt_verified must be boolean")
        if self.ptmrt_verified:
            _require_digest(self.ptmrt_artifact_id, "ptmrt artifact ID")
            if self.ptmrt_artifact_id != self.artifact_id:
                raise ModelGenerationError("ptmrt verified a different artifact")
        elif self.ptmrt_artifact_id is not None:
            raise ModelGenerationError("unverified ptmrt evidence cannot name an artifact")

    @property
    def exact(self) -> bool:
        return self.reference_packed_mismatches == 0 and self.ptmrt_verified

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "case_count": self.case_count,
            "reference_packed_mismatches": self.reference_packed_mismatches,
            "ptmrt_verified": self.ptmrt_verified,
            "ptmrt_artifact_id": self.ptmrt_artifact_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RuntimeConformanceReport":
        expected = {
            "artifact_id",
            "case_count",
            "reference_packed_mismatches",
            "ptmrt_verified",
            "ptmrt_artifact_id",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or type(value["artifact_id"]) is not str
            or type(value["case_count"]) is not int
            or type(value["reference_packed_mismatches"]) is not int
            or type(value["ptmrt_verified"]) is not bool
            or (
                value["ptmrt_artifact_id"] is not None
                and type(value["ptmrt_artifact_id"]) is not str
            )
        ):
            raise ModelGenerationError("runtime conformance report is malformed")
        return cls(
            value["artifact_id"],
            value["case_count"],
            value["reference_packed_mismatches"],
            value["ptmrt_verified"],
            value["ptmrt_artifact_id"],
        )


@dataclass(frozen=True, slots=True)
class LiveRuntimeConformanceEvidence:
    child_generation_id: str
    artifact_id: str
    snapshot_id: str
    literal_manifest_id: str
    corpus: LabeledCorpus
    scalar_features: tuple[tuple[bool, ...], ...]
    scalar_scores: tuple[int, ...]
    scalar_predictions: tuple[int, ...]
    packed_predictions: tuple[int, ...]
    native_features: tuple[tuple[int, ...], ...]
    native_scores: tuple[int, ...]
    native_predictions: tuple[int, ...]
    ptmrt_binary_digest: str
    schema: str = LIVE_CONFORMANCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LIVE_CONFORMANCE_SCHEMA:
            raise ModelGenerationError("live conformance schema is unsupported")
        for label, value in (
            ("child generation", self.child_generation_id),
            ("artifact", self.artifact_id),
            ("snapshot", self.snapshot_id),
            ("literal manifest", self.literal_manifest_id),
            ("ptmrt executable", self.ptmrt_binary_digest),
        ):
            _require_digest(value, label)
        if not isinstance(self.corpus, LabeledCorpus) or (
            self.corpus.role is not CorpusRole.LIVE
        ):
            raise ModelGenerationError("live conformance requires a live corpus")
        count = len(self.corpus.examples)
        vectors = (
            self.scalar_scores,
            self.scalar_predictions,
            self.packed_predictions,
            self.native_scores,
            self.native_predictions,
        )
        if (
            type(self.scalar_features) is not tuple
            or type(self.native_features) is not tuple
            or any(type(vector) is not tuple for vector in vectors)
            or len(self.scalar_features) != count
            or len(self.native_features) != count
            or any(len(vector) != count for vector in vectors)
        ):
            raise ModelGenerationError("live conformance vector lengths differ")
        if (
            not self.scalar_features
            or any(
                type(row) is not tuple
                or not row
                or any(type(value) is not bool for value in row)
                for row in self.scalar_features
            )
            or len({len(row) for row in self.scalar_features}) != 1
            or any(
                type(row) is not tuple
                or any(type(value) is not int or value not in (0, 1) for value in row)
                for row in self.native_features
            )
            or tuple(tuple(int(value) for value in row) for row in self.scalar_features)
            != self.native_features
        ):
            raise ModelGenerationError("live conformance feature vectors are invalid")
        if any(type(value) is not int for value in (*self.scalar_scores, *self.native_scores)):
            raise ModelGenerationError("live conformance scores must be integers")
        if any(
            type(value) is not int or value not in (0, 1)
            for vector in (
                self.scalar_predictions,
                self.packed_predictions,
                self.native_predictions,
            )
            for value in vector
        ):
            raise ModelGenerationError("live conformance predictions are invalid")
        expected_predictions = tuple(int(score > 0) for score in self.scalar_scores)
        if (
            self.scalar_scores != self.native_scores
            or self.scalar_predictions != expected_predictions
            or self.packed_predictions != expected_predictions
            or self.native_predictions != expected_predictions
        ):
            raise ModelGenerationError("live runtimes do not agree exactly")

    @property
    def evidence_id(self) -> str:
        return content_digest(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "child_generation_id": self.child_generation_id,
            "artifact_id": self.artifact_id,
            "snapshot_id": self.snapshot_id,
            "literal_manifest_id": self.literal_manifest_id,
            "corpus": self.corpus.canonical_payload(),
            "corpus_digest": self.corpus.digest,
            "scalar_features": [list(row) for row in self.scalar_features],
            "scalar_scores": list(self.scalar_scores),
            "scalar_predictions": list(self.scalar_predictions),
            "packed_predictions": list(self.packed_predictions),
            "native_features": [list(row) for row in self.native_features],
            "native_scores": list(self.native_scores),
            "native_predictions": list(self.native_predictions),
            "ptmrt_binary_digest": self.ptmrt_binary_digest,
        }

    def to_dict(self) -> dict[str, object]:
        result = self.canonical_payload()
        result["evidence_id"] = self.evidence_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LiveRuntimeConformanceEvidence":
        expected = {
            "schema",
            "child_generation_id",
            "artifact_id",
            "snapshot_id",
            "literal_manifest_id",
            "corpus",
            "corpus_digest",
            "scalar_features",
            "scalar_scores",
            "scalar_predictions",
            "packed_predictions",
            "native_features",
            "native_scores",
            "native_predictions",
            "ptmrt_binary_digest",
            "evidence_id",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ModelGenerationError("live conformance evidence is malformed")
        string_fields = (
            "schema",
            "child_generation_id",
            "artifact_id",
            "snapshot_id",
            "literal_manifest_id",
            "corpus_digest",
            "ptmrt_binary_digest",
            "evidence_id",
        )
        raw_corpus = value["corpus"]
        matrix_fields = ("scalar_features", "native_features")
        vector_fields = (
            "scalar_scores",
            "scalar_predictions",
            "packed_predictions",
            "native_scores",
            "native_predictions",
        )
        if (
            any(type(value[name]) is not str for name in string_fields)
            or not isinstance(raw_corpus, Mapping)
            or set(raw_corpus) != {"dataset_id", "role", "examples"}
            or type(raw_corpus["dataset_id"]) is not str
            or type(raw_corpus["role"]) is not str
            or not isinstance(raw_corpus["examples"], list)
            or any(not isinstance(value[name], list) for name in (*matrix_fields, *vector_fields))
            or any(
                not isinstance(row, list)
                for name in matrix_fields
                for row in value[name]
            )
        ):
            raise ModelGenerationError("live conformance evidence types are malformed")
        examples: list[CorpusExample] = []
        for raw in raw_corpus["examples"]:
            if not isinstance(raw, Mapping) or set(raw) != {
                "example_id",
                "record",
                "label",
            }:
                raise ModelGenerationError("live conformance corpus is malformed")
            examples.append(
                CorpusExample(raw["example_id"], raw["record"], raw["label"])
            )
        try:
            corpus = LabeledCorpus(
                raw_corpus["dataset_id"],
                CorpusRole(raw_corpus["role"]),
                tuple(examples),
            )
            result = cls(
                child_generation_id=value["child_generation_id"],
                artifact_id=value["artifact_id"],
                snapshot_id=value["snapshot_id"],
                literal_manifest_id=value["literal_manifest_id"],
                corpus=corpus,
                scalar_features=tuple(tuple(row) for row in value["scalar_features"]),
                scalar_scores=tuple(value["scalar_scores"]),
                scalar_predictions=tuple(value["scalar_predictions"]),
                packed_predictions=tuple(value["packed_predictions"]),
                native_features=tuple(tuple(row) for row in value["native_features"]),
                native_scores=tuple(value["native_scores"]),
                native_predictions=tuple(value["native_predictions"]),
                ptmrt_binary_digest=value["ptmrt_binary_digest"],
                schema=value["schema"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ModelGenerationError("live conformance evidence is malformed") from error
        if corpus.digest != value["corpus_digest"] or result.evidence_id != value["evidence_id"]:
            raise ModelGenerationError("live conformance evidence digest mismatch")
        return result


@dataclass(frozen=True, slots=True)
class PromotionAuditSnapshot:
    corpus_role: CorpusRole
    corpus_digest: str
    observations: int
    parent_errors: int
    child_errors: int
    disagreements: int
    improvements: int
    regressions: int
    both_correct: int
    both_wrong: int
    parent_scores: tuple[int, ...]
    child_scores: tuple[int, ...]
    class_counts: tuple[ClassPromotionCounts, ...]
    conformance: RuntimeConformanceReport
    accepted: bool
    schema: str = PROMOTION_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROMOTION_AUDIT_SCHEMA:
            raise ModelGenerationError("promotion audit schema is unsupported")
        if self.corpus_role not in (CorpusRole.PROMOTION, CorpusRole.LIVE):
            raise ModelGenerationError("promotion audit corpus role is invalid")
        _require_digest(self.corpus_digest, "audit corpus digest")
        counts = (
            self.observations,
            self.parent_errors,
            self.child_errors,
            self.disagreements,
            self.improvements,
            self.regressions,
            self.both_correct,
            self.both_wrong,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ModelGenerationError("promotion audit counts must be nonnegative integers")
        if self.observations <= 0 or sum(
            (self.both_correct, self.both_wrong, self.improvements, self.regressions)
        ) != self.observations:
            raise ModelGenerationError("promotion outcomes do not partition observations")
        if self.parent_errors != self.improvements + self.both_wrong:
            raise ModelGenerationError("parent error count is inconsistent")
        if self.child_errors != self.regressions + self.both_wrong:
            raise ModelGenerationError("child error count is inconsistent")
        if self.disagreements != self.improvements + self.regressions:
            raise ModelGenerationError("disagreement count is inconsistent")
        if (
            type(self.parent_scores) is not tuple
            or type(self.child_scores) is not tuple
            or len(self.parent_scores) != self.observations
            or len(self.child_scores) != self.observations
            or any(type(score) is not int for score in (*self.parent_scores, *self.child_scores))
        ):
            raise ModelGenerationError("promotion score vectors are invalid")
        if (
            type(self.class_counts) is not tuple
            or tuple(item.label for item in self.class_counts) != (0, 1)
            or sum(item.observed for item in self.class_counts) != self.observations
        ):
            raise ModelGenerationError("promotion class strata are invalid")
        if not isinstance(self.conformance, RuntimeConformanceReport):
            raise TypeError("promotion audit conformance report is invalid")
        if self.conformance.case_count != self.observations:
            raise ModelGenerationError(
                "promotion audit and conformance observation counts differ"
            )
        if type(self.accepted) is not bool:
            raise TypeError("promotion audit decision must be boolean")
        if self.accepted and (
            self.corpus_role is not CorpusRole.PROMOTION or not self.conformance.exact
        ):
            raise ModelGenerationError("accepted promotion lacks exact holdout evidence")

    @property
    def audit_id(self) -> str:
        return content_digest(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "corpus_role": self.corpus_role.value,
            "corpus_digest": self.corpus_digest,
            "observations": self.observations,
            "parent_errors": self.parent_errors,
            "child_errors": self.child_errors,
            "disagreements": self.disagreements,
            "improvements": self.improvements,
            "regressions": self.regressions,
            "both_correct": self.both_correct,
            "both_wrong": self.both_wrong,
            "parent_scores": list(self.parent_scores),
            "child_scores": list(self.child_scores),
            "class_counts": [item.to_dict() for item in self.class_counts],
            "conformance": self.conformance.to_dict(),
            "accepted": self.accepted,
        }

    def to_dict(self) -> dict[str, object]:
        result = self.canonical_payload()
        result["audit_id"] = self.audit_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PromotionAuditSnapshot":
        expected = {
            "schema",
            "corpus_role",
            "corpus_digest",
            "observations",
            "parent_errors",
            "child_errors",
            "disagreements",
            "improvements",
            "regressions",
            "both_correct",
            "both_wrong",
            "parent_scores",
            "child_scores",
            "class_counts",
            "conformance",
            "accepted",
            "audit_id",
        }
        count_fields = (
            "observations",
            "parent_errors",
            "child_errors",
            "disagreements",
            "improvements",
            "regressions",
            "both_correct",
            "both_wrong",
        )
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or any(
                type(value[key]) is not str
                for key in ("schema", "corpus_role", "corpus_digest", "audit_id")
            )
            or any(type(value[key]) is not int for key in count_fields)
            or type(value["accepted"]) is not bool
            or not isinstance(value["parent_scores"], list)
            or not isinstance(value["child_scores"], list)
            or any(type(item) is not int for item in value["parent_scores"])
            or any(type(item) is not int for item in value["child_scores"])
            or not isinstance(value["class_counts"], list)
            or not isinstance(value["conformance"], Mapping)
        ):
            raise ModelGenerationError("promotion audit is malformed")
        try:
            result = cls(
                corpus_role=CorpusRole(value["corpus_role"]),
                corpus_digest=value["corpus_digest"],
                observations=value["observations"],
                parent_errors=value["parent_errors"],
                child_errors=value["child_errors"],
                disagreements=value["disagreements"],
                improvements=value["improvements"],
                regressions=value["regressions"],
                both_correct=value["both_correct"],
                both_wrong=value["both_wrong"],
                parent_scores=tuple(value["parent_scores"]),
                child_scores=tuple(value["child_scores"]),
                class_counts=tuple(
                    ClassPromotionCounts.from_dict(item)
                    for item in value["class_counts"]
                ),
                conformance=RuntimeConformanceReport.from_dict(value["conformance"]),
                accepted=value["accepted"],
                schema=value["schema"],
            )
        except (TypeError, ValueError) as error:
            raise ModelGenerationError("promotion audit is malformed") from error
        if result.audit_id != value["audit_id"]:
            raise ModelGenerationError("promotion audit digest mismatch")
        return result


def audit_runtime_conformance(
    child: AdaptedChild,
    artifact: PackedTMInferenceArtifact,
    records: Sequence[Mapping[str, object]],
    *,
    ptmrt_verified: bool,
    ptmrt_artifact_id: str | None,
) -> RuntimeConformanceReport:
    return audit_snapshot_runtime_conformance(
        child.snapshot,
        child.manifest,
        artifact,
        records,
        ptmrt_verified=ptmrt_verified,
        ptmrt_artifact_id=ptmrt_artifact_id,
    )


def audit_snapshot_runtime_conformance(
    child_snapshot: AdaptiveSnapshotEnvelope,
    child_manifest: OrderedLiteralManifest,
    artifact: PackedTMInferenceArtifact,
    records: Sequence[Mapping[str, object]],
    *,
    ptmrt_verified: bool,
    ptmrt_artifact_id: str | None,
) -> RuntimeConformanceReport:
    catalog = child_manifest.build_catalog()
    batch = catalog.encode(records).ta
    rows = tuple(batch.row_values(index) for index in range(batch.row_count))
    reference = _machine_from_snapshot(child_snapshot.snapshot).predict(rows)
    packed = list(artifact.predict_records(records))
    mismatches = abs(len(reference) - len(packed)) + sum(
        left != right for left, right in zip(reference, packed)
    )
    if not artifact.verify_conformance():
        mismatches = max(1, mismatches)
    return RuntimeConformanceReport(
        artifact.artifact_id,
        len(rows),
        mismatches,
        ptmrt_verified,
        ptmrt_artifact_id,
    )


def audit_parent_child(
    parent_snapshot: TMSnapshot,
    parent_manifest: OrderedLiteralManifest,
    child: AdaptedChild,
    corpus: LabeledCorpus,
    conformance: RuntimeConformanceReport,
    policy: PromotionAuditPolicy,
) -> PromotionAuditSnapshot:
    return audit_parent_child_snapshots(
        parent_snapshot,
        parent_manifest,
        child.snapshot,
        child.manifest,
        corpus,
        conformance,
        policy,
    )


def audit_parent_child_snapshots(
    parent_snapshot: TMSnapshot,
    parent_manifest: OrderedLiteralManifest,
    child_snapshot: AdaptiveSnapshotEnvelope,
    child_manifest: OrderedLiteralManifest,
    corpus: LabeledCorpus,
    conformance: RuntimeConformanceReport,
    policy: PromotionAuditPolicy,
) -> PromotionAuditSnapshot:
    if corpus.role not in (CorpusRole.PROMOTION, CorpusRole.LIVE):
        raise ModelGenerationError("paired audit requires promotion or live corpus")
    parent_catalog = parent_manifest.build_catalog()
    child_catalog = child_manifest.build_catalog()
    parent_batch = parent_catalog.encode(corpus.records).ta
    child_batch = child_catalog.encode(corpus.records).ta
    parent_machine = _machine_from_snapshot(parent_snapshot)
    child_machine = _machine_from_snapshot(child_snapshot.snapshot)
    parent_rows = tuple(
        parent_batch.row_values(index) for index in range(parent_batch.row_count)
    )
    child_rows = tuple(
        child_batch.row_values(index) for index in range(child_batch.row_count)
    )
    parent_scores = tuple(parent_machine.score(row) for row in parent_rows)
    child_scores = tuple(child_machine.score(row) for row in child_rows)
    parent_predictions = tuple(int(score > 0) for score in parent_scores)
    child_predictions = tuple(int(score > 0) for score in child_scores)
    both_correct = both_wrong = improvements = regressions = disagreements = 0
    per_class = {
        0: {"observed": 0, "both_correct": 0, "both_wrong": 0, "improvements": 0, "regressions": 0},
        1: {"observed": 0, "both_correct": 0, "both_wrong": 0, "improvements": 0, "regressions": 0},
    }
    for truth, parent_prediction, child_prediction in zip(
        corpus.labels, parent_predictions, child_predictions
    ):
        parent_correct = parent_prediction == truth
        child_correct = child_prediction == truth
        per_class[truth]["observed"] += 1
        if parent_prediction != child_prediction:
            disagreements += 1
        if parent_correct and child_correct:
            both_correct += 1
            per_class[truth]["both_correct"] += 1
        elif not parent_correct and not child_correct:
            both_wrong += 1
            per_class[truth]["both_wrong"] += 1
        elif not parent_correct and child_correct:
            improvements += 1
            per_class[truth]["improvements"] += 1
        else:
            regressions += 1
            per_class[truth]["regressions"] += 1
    parent_errors = improvements + both_wrong
    child_errors = regressions + both_wrong
    error_policy_satisfied = (
        child_errors < parent_errors
        if policy.require_strict_improvement
        else child_errors <= parent_errors
    )
    accepted = (
        corpus.role is CorpusRole.PROMOTION
        and len(corpus.examples) >= policy.minimum_observations
        and error_policy_satisfied
        and regressions <= policy.maximum_regressions
        and conformance.exact
    )
    class_counts = tuple(
        ClassPromotionCounts(label=label, **per_class[label]) for label in (0, 1)
    )
    return PromotionAuditSnapshot(
        corpus.role,
        corpus.digest,
        len(corpus.examples),
        parent_errors,
        child_errors,
        disagreements,
        improvements,
        regressions,
        both_correct,
        both_wrong,
        parent_scores,
        child_scores,
        class_counts,
        conformance,
        accepted,
    )


def drift_requires_reopen(
    report: PromotionAuditSnapshot, policy: DriftAuditPolicy
) -> bool:
    """Request reopen only from labeled evidence that the child is worse."""

    if report.corpus_role is not CorpusRole.LIVE:
        raise ModelGenerationError("drift decisions require the live/drift corpus")
    return (
        report.observations >= policy.minimum_observations
        and report.child_errors - report.parent_errors
        >= policy.minimum_error_increase
        and report.regressions >= policy.minimum_regressions
        and report.regressions / report.observations
        >= float(policy.minimum_regression_rate)
        and report.regressions > report.improvements
        and all(
            counts.observed >= policy.minimum_observations_per_class
            for counts in report.class_counts
        )
    )


@dataclass(frozen=True, slots=True)
class ModelGenerationLineage:
    parent_generation_id: str
    extended_generation_id: str
    child_generation_id: str
    adaptive_behavior_id: str
    restoration_bundle_id: str
    promotion_audit_id: str
    invention_evidence_id: str
    invented_literal_id: int
    invention_corpus_digest: str
    adaptation_corpus_digest: str
    promotion_corpus_digest: str
    origin_proposal_semantic_id: str
    origin_proposal_provenance_id: str
    schema: str = LINEAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LINEAGE_SCHEMA:
            raise ModelGenerationError("model-generation lineage schema is unsupported")
        for label, value in (
            ("parent generation", self.parent_generation_id),
            ("extended generation", self.extended_generation_id),
            ("child generation", self.child_generation_id),
            ("adaptive behavior", self.adaptive_behavior_id),
            ("restoration bundle", self.restoration_bundle_id),
            ("promotion audit", self.promotion_audit_id),
            ("invention evidence", self.invention_evidence_id),
            ("invention corpus", self.invention_corpus_digest),
            ("adaptation corpus", self.adaptation_corpus_digest),
            ("promotion corpus", self.promotion_corpus_digest),
            ("origin proposal semantic ID", self.origin_proposal_semantic_id),
            ("origin proposal provenance ID", self.origin_proposal_provenance_id),
        ):
            _require_digest(value, label)
        if type(self.invented_literal_id) is not int or not 0 <= self.invented_literal_id < 1 << 64:
            raise ModelGenerationError("invented literal ID must be unsigned 64-bit")

    @property
    def lineage_id(self) -> str:
        return content_digest(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "parent_generation_id": self.parent_generation_id,
            "extended_generation_id": self.extended_generation_id,
            "child_generation_id": self.child_generation_id,
            "adaptive_behavior_id": self.adaptive_behavior_id,
            "restoration_bundle_id": self.restoration_bundle_id,
            "promotion_audit_id": self.promotion_audit_id,
            "invention_evidence_id": self.invention_evidence_id,
            "invented_literal_id": str(self.invented_literal_id),
            "invention_corpus_digest": self.invention_corpus_digest,
            "adaptation_corpus_digest": self.adaptation_corpus_digest,
            "promotion_corpus_digest": self.promotion_corpus_digest,
            "origin_proposal_semantic_id": self.origin_proposal_semantic_id,
            "origin_proposal_provenance_id": self.origin_proposal_provenance_id,
        }

    def to_dict(self) -> dict[str, object]:
        result = self.canonical_payload()
        result["lineage_id"] = self.lineage_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ModelGenerationLineage":
        expected = {
            "schema",
            "parent_generation_id",
            "extended_generation_id",
            "child_generation_id",
            "adaptive_behavior_id",
            "restoration_bundle_id",
            "promotion_audit_id",
            "invention_evidence_id",
            "invented_literal_id",
            "invention_corpus_digest",
            "adaptation_corpus_digest",
            "promotion_corpus_digest",
            "origin_proposal_semantic_id",
            "origin_proposal_provenance_id",
            "lineage_id",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or any(type(value[key]) is not str for key in expected)
        ):
            raise ModelGenerationError("model-generation lineage is malformed")
        raw_literal_id = value["invented_literal_id"]
        if not raw_literal_id.isdigit():
            raise ModelGenerationError("invented literal ID is malformed")
        try:
            result = cls(
                parent_generation_id=value["parent_generation_id"],
                extended_generation_id=value["extended_generation_id"],
                child_generation_id=value["child_generation_id"],
                adaptive_behavior_id=value["adaptive_behavior_id"],
                restoration_bundle_id=value["restoration_bundle_id"],
                promotion_audit_id=value["promotion_audit_id"],
                invention_evidence_id=value["invention_evidence_id"],
                invented_literal_id=int(raw_literal_id),
                invention_corpus_digest=value["invention_corpus_digest"],
                adaptation_corpus_digest=value["adaptation_corpus_digest"],
                promotion_corpus_digest=value["promotion_corpus_digest"],
                origin_proposal_semantic_id=value["origin_proposal_semantic_id"],
                origin_proposal_provenance_id=value["origin_proposal_provenance_id"],
                schema=value["schema"],
            )
        except (TypeError, ValueError) as error:
            raise ModelGenerationError("model-generation lineage is malformed") from error
        if result.lineage_id != value["lineage_id"]:
            raise ModelGenerationError("model-generation lineage digest mismatch")
        return result
