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
    contract_snapshot_equivalent_feature,
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
THRESHOLD_CANDIDATE_SET_SCHEMA = "ptm.threshold-candidate-set.v1"
THRESHOLD_CANDIDATE_SELECTION_SCHEMA = "ptm.threshold-candidate-selection.v1"
DEESCALATION_EVIDENCE_SCHEMA = "ptm.deescalation-evidence.v1"
GENERATION_SCHEMA = "ptm.model-generation.v1"
PROMOTION_AUDIT_SCHEMA = "ptm.promotion-audit.v1"
LIVE_CONFORMANCE_SCHEMA = "ptm.live-runtime-conformance.v1"
EVIDENCE_USAGE_SCHEMA = "ptm.model-generation-evidence-usage.v1"
LEGACY_LINEAGE_SCHEMA = "ptm.model-generation-lineage.v4"
LINEAGE_SCHEMA = "ptm.model-generation-lineage.v5"
CONTRACTION_LINEAGE_SCHEMA = "ptm.literal-contraction-lineage.v1"
TRAINING_SEMANTICS_VERSION = "ptm.scalar-binary-training.v1"
PYTHON_RNG_ALGORITHM = "python.random-mt19937-state-v1"
MAX_CORPUS_EXAMPLES = 2_048
MAX_ADAPTATION_EPOCHS = 10_000
MAX_THRESHOLD_CANDIDATES = 64
MAX_THRESHOLD_FIELDS = 32
MAX_DEESCALATION_CANDIDATES = 64


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
    DEESCALATION_PROOF = "deescalation_proof"
    DEESCALATION_CONFIRMATION = "deescalation_confirmation"
    PROMOTION = "promotion_holdout"
    LIVE = "live_drift"


class EvidenceUsagePurpose(str, Enum):
    PARENT_REGISTRATION = "parent_registration"
    CANDIDATE_EPISODE = "candidate_episode"
    DEESCALATION_EPISODE = "deescalation_episode"
    LIVE_DRIFT = "live_drift"


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
class DeescalationCorpora:
    proof: LabeledCorpus
    confirmation: LabeledCorpus
    promotion: LabeledCorpus

    def __post_init__(self) -> None:
        expected = (
            (self.proof, CorpusRole.DEESCALATION_PROOF),
            (self.confirmation, CorpusRole.DEESCALATION_CONFIRMATION),
            (self.promotion, CorpusRole.PROMOTION),
        )
        if any(corpus.role is not role for corpus, role in expected):
            raise ModelGenerationError("de-escalation corpus role is misplaced")
        if len({corpus.dataset_id for corpus, _ in expected}) != 1:
            raise ModelGenerationError(
                "de-escalation corpora must share one dataset ID"
            )
        seen: set[str | int] = set()
        fingerprints: set[str] = set()
        for corpus, _ in expected:
            identifiers = {item.example_id for item in corpus.examples}
            current = {
                content_digest({"record": item.record, "label": item.label})
                for item in corpus.examples
            }
            if seen & identifiers:
                raise ModelGenerationError(
                    "de-escalation corpus example IDs overlap"
                )
            if fingerprints & current:
                raise ModelGenerationError(
                    "de-escalation proof, confirmation, and promotion rows "
                    "must be independent"
                )
            seen.update(identifiers)
            fingerprints.update(current)

    @property
    def digests(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (corpus.role.value, corpus.digest)
            for corpus in (self.proof, self.confirmation, self.promotion)
        )


def evidence_record_fingerprint(
    dataset_id: str, example: CorpusExample
) -> str:
    """Identify one labeled row independently of its observation identity."""

    if type(dataset_id) is not str or not dataset_id:
        raise ModelGenerationError("evidence dataset ID must be nonempty")
    if not isinstance(example, CorpusExample):
        raise TypeError("evidence fingerprint requires a corpus example")
    return content_digest(
        {
            "dataset_id": dataset_id,
            "record": example.record,
            "label": example.label,
        }
    )


@dataclass(frozen=True, slots=True)
class EvidenceUsage:
    """Exact labeled corpora durably spent for one bounded lifecycle purpose."""

    purpose: EvidenceUsagePurpose
    subject_generation_id: str
    corpora: tuple[LabeledCorpus, ...]
    schema: str = EVIDENCE_USAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EVIDENCE_USAGE_SCHEMA:
            raise ModelGenerationError("evidence usage schema is unsupported")
        if not isinstance(self.purpose, EvidenceUsagePurpose):
            raise TypeError("evidence usage purpose is invalid")
        _require_digest(self.subject_generation_id, "evidence subject generation")
        if type(self.corpora) is not tuple or any(
            not isinstance(corpus, LabeledCorpus) for corpus in self.corpora
        ):
            raise TypeError("evidence usage corpora must be a tuple of labeled corpora")
        expected_roles = {
            EvidenceUsagePurpose.PARENT_REGISTRATION: (
                CorpusRole.PARENT_TRAINING,
            ),
            EvidenceUsagePurpose.CANDIDATE_EPISODE: (
                CorpusRole.INVENTION,
                CorpusRole.ADAPTATION,
                CorpusRole.PROMOTION,
            ),
            EvidenceUsagePurpose.DEESCALATION_EPISODE: (
                CorpusRole.DEESCALATION_PROOF,
                CorpusRole.DEESCALATION_CONFIRMATION,
                CorpusRole.PROMOTION,
            ),
            EvidenceUsagePurpose.LIVE_DRIFT: (CorpusRole.LIVE,),
        }[self.purpose]
        if tuple(corpus.role for corpus in self.corpora) != expected_roles:
            raise ModelGenerationError("evidence usage corpus roles are misplaced")
        if len({corpus.dataset_id for corpus in self.corpora}) != 1:
            raise ModelGenerationError("evidence usage corpora must share one dataset")
        if self.purpose is EvidenceUsagePurpose.CANDIDATE_EPISODE:
            LifecycleCorpora(*self.corpora)
        elif self.purpose is EvidenceUsagePurpose.DEESCALATION_EPISODE:
            DeescalationCorpora(*self.corpora)
        identifiers: set[tuple[str, str, str | int]] = set()
        fingerprints: set[str] = set()
        for corpus in self.corpora:
            for example in corpus.examples:
                identifier = (
                    corpus.dataset_id,
                    type(example.example_id).__name__,
                    example.example_id,
                )
                fingerprint = evidence_record_fingerprint(
                    corpus.dataset_id, example
                )
                if identifier in identifiers or fingerprint in fingerprints:
                    raise ModelGenerationError(
                        "evidence usage contains repeated observations"
                    )
                identifiers.add(identifier)
                fingerprints.add(fingerprint)

    @property
    def dataset_id(self) -> str:
        return self.corpora[0].dataset_id

    @property
    def example_keys(self) -> frozenset[tuple[str, str, str | int]]:
        return frozenset(
            (
                corpus.dataset_id,
                type(example.example_id).__name__,
                example.example_id,
            )
            for corpus in self.corpora
            for example in corpus.examples
        )

    @property
    def record_fingerprints(self) -> frozenset[str]:
        return frozenset(
            evidence_record_fingerprint(corpus.dataset_id, example)
            for corpus in self.corpora
            for example in corpus.examples
        )

    @property
    def usage_id(self) -> str:
        return content_digest(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "purpose": self.purpose.value,
            "subject_generation_id": self.subject_generation_id,
            "dataset_id": self.dataset_id,
            "corpora": [
                {
                    "corpus": corpus.canonical_payload(),
                    "corpus_digest": corpus.digest,
                }
                for corpus in self.corpora
            ],
        }

    def to_dict(self) -> dict[str, object]:
        result = self.canonical_payload()
        result["usage_id"] = self.usage_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "EvidenceUsage":
        expected = {
            "schema",
            "purpose",
            "subject_generation_id",
            "dataset_id",
            "corpora",
            "usage_id",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or any(
                type(value[name]) is not str
                for name in expected - {"corpora"}
            )
            or not isinstance(value["corpora"], list)
        ):
            raise ModelGenerationError("evidence usage is malformed")
        corpora: list[LabeledCorpus] = []
        for raw_entry in value["corpora"]:
            if (
                not isinstance(raw_entry, Mapping)
                or set(raw_entry) != {"corpus", "corpus_digest"}
                or type(raw_entry["corpus_digest"]) is not str
                or not isinstance(raw_entry["corpus"], Mapping)
            ):
                raise ModelGenerationError("evidence usage corpus is malformed")
            raw_corpus = raw_entry["corpus"]
            if (
                set(raw_corpus) != {"dataset_id", "role", "examples"}
                or type(raw_corpus["dataset_id"]) is not str
                or type(raw_corpus["role"]) is not str
                or not isinstance(raw_corpus["examples"], list)
            ):
                raise ModelGenerationError("evidence usage corpus is malformed")
            examples: list[CorpusExample] = []
            for raw_example in raw_corpus["examples"]:
                if not isinstance(raw_example, Mapping) or set(raw_example) != {
                    "example_id",
                    "record",
                    "label",
                } or not isinstance(raw_example["record"], Mapping):
                    raise ModelGenerationError(
                        "evidence usage corpus example is malformed"
                    )
                examples.append(
                    CorpusExample(
                        raw_example["example_id"],
                        raw_example["record"],
                        raw_example["label"],
                    )
                )
            try:
                corpus = LabeledCorpus(
                    raw_corpus["dataset_id"],
                    CorpusRole(raw_corpus["role"]),
                    tuple(examples),
                )
            except (TypeError, ValueError) as error:
                raise ModelGenerationError(
                    "evidence usage corpus is malformed"
                ) from error
            if corpus.digest != raw_entry["corpus_digest"]:
                raise ModelGenerationError("evidence usage corpus digest mismatch")
            corpora.append(corpus)
        try:
            result = cls(
                EvidenceUsagePurpose(value["purpose"]),
                value["subject_generation_id"],
                tuple(corpora),
                value["schema"],
            )
        except (TypeError, ValueError) as error:
            raise ModelGenerationError("evidence usage is malformed") from error
        if (
            result.dataset_id != value["dataset_id"]
            or result.usage_id != value["usage_id"]
        ):
            raise ModelGenerationError("evidence usage digest mismatch")
        return result


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
class PrologDeescalationEvidence:
    """Complete attested literal-equivalence result for one proof corpus."""

    proof_corpus_digest: str
    session_digest: str
    parent_snapshot_id: str
    parent_manifest_id: str
    maximum_candidates: int
    equivalent_pairs: tuple[tuple[int, int], ...]
    surviving_literal_id: int
    removed_literal_id: int
    collective_protocol: str
    gprolog_version: str
    gprolog_binary_digest: str
    module_digests: tuple[tuple[str, str], ...]
    schema: str = DEESCALATION_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DEESCALATION_EVIDENCE_SCHEMA:
            raise ModelGenerationError("de-escalation evidence schema is unsupported")
        for label, value in (
            ("de-escalation proof corpus", self.proof_corpus_digest),
            ("de-escalation reasoning session", self.session_digest),
            ("de-escalation parent snapshot", self.parent_snapshot_id),
            ("de-escalation parent manifest", self.parent_manifest_id),
            ("GNU Prolog executable", self.gprolog_binary_digest),
        ):
            _require_digest(value, label)
        if (
            type(self.maximum_candidates) is not int
            or not 1 <= self.maximum_candidates <= MAX_DEESCALATION_CANDIDATES
        ):
            raise ModelGenerationError(
                "de-escalation candidate budget is outside its bounds"
            )
        if (
            type(self.equivalent_pairs) is not tuple
            or not 0 < len(self.equivalent_pairs) <= self.maximum_candidates
            or tuple(sorted(set(self.equivalent_pairs))) != self.equivalent_pairs
        ):
            raise ModelGenerationError(
                "de-escalation equivalent pairs are not complete and canonical"
            )
        for pair in self.equivalent_pairs:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or any(type(item) is not int or not 0 <= item < 1 << 64 for item in pair)
                or pair[0] >= pair[1]
            ):
                raise ModelGenerationError(
                    "de-escalation equivalent literal pair is invalid"
                )
        selected = tuple(sorted((self.surviving_literal_id, self.removed_literal_id)))
        if (
            any(
                type(item) is not int or not 0 <= item < 1 << 64
                for item in (self.surviving_literal_id, self.removed_literal_id)
            )
            or self.surviving_literal_id == self.removed_literal_id
            or selected not in self.equivalent_pairs
        ):
            raise ModelGenerationError("de-escalation selected literal pair is invalid")
        if self.collective_protocol != "PTM_PTA_COLLECTIVE_V1":
            raise ModelGenerationError("de-escalation collective protocol is unsupported")
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
                "numeric_fields": (),
                "discover_thresholds": False,
                "discover_intervals": False,
                "derive_deescalation": True,
                "derive_escalation": False,
            }
        )

    @property
    def evidence_id(self) -> str:
        return content_digest(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "proof_corpus_digest": self.proof_corpus_digest,
            "session_digest": self.session_digest,
            "parent_snapshot_id": self.parent_snapshot_id,
            "parent_manifest_id": self.parent_manifest_id,
            "query": _thaw_json(self.query),
            "maximum_candidates": self.maximum_candidates,
            "equivalent_pairs": [
                [str(left), str(right)] for left, right in self.equivalent_pairs
            ],
            "surviving_literal_id": str(self.surviving_literal_id),
            "removed_literal_id": str(self.removed_literal_id),
            "collective_protocol": self.collective_protocol,
            "gprolog_version": self.gprolog_version,
            "gprolog_binary_digest": self.gprolog_binary_digest,
            "module_digests": [list(item) for item in self.module_digests],
        }

    def to_dict(self) -> dict[str, object]:
        result = self.canonical_payload()
        result["evidence_id"] = self.evidence_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PrologDeescalationEvidence":
        expected = {
            "schema",
            "proof_corpus_digest",
            "session_digest",
            "parent_snapshot_id",
            "parent_manifest_id",
            "query",
            "maximum_candidates",
            "equivalent_pairs",
            "surviving_literal_id",
            "removed_literal_id",
            "collective_protocol",
            "gprolog_version",
            "gprolog_binary_digest",
            "module_digests",
            "evidence_id",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or any(
                type(value[key]) is not str
                for key in expected
                - {"query", "maximum_candidates", "equivalent_pairs", "module_digests"}
            )
            or type(value["maximum_candidates"]) is not int
            or not isinstance(value["query"], Mapping)
            or not isinstance(value["equivalent_pairs"], list)
            or not isinstance(value["module_digests"], list)
        ):
            raise ModelGenerationError("de-escalation evidence is malformed")
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
            or query["numeric_fields"] != []
            or query["discover_thresholds"] is not False
            or query["discover_intervals"] is not False
            or query["derive_deescalation"] is not True
            or query["derive_escalation"] is not False
            or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or any(type(item) is not str or not item.isdigit() for item in pair)
                for pair in value["equivalent_pairs"]
            )
            or any(
                not isinstance(item, list)
                or len(item) != 2
                or any(type(part) is not str for part in item)
                for item in value["module_digests"]
            )
        ):
            raise ModelGenerationError("de-escalation evidence query is malformed")
        try:
            result = cls(
                proof_corpus_digest=value["proof_corpus_digest"],
                session_digest=value["session_digest"],
                parent_snapshot_id=value["parent_snapshot_id"],
                parent_manifest_id=value["parent_manifest_id"],
                maximum_candidates=value["maximum_candidates"],
                equivalent_pairs=tuple(
                    (int(pair[0]), int(pair[1]))
                    for pair in value["equivalent_pairs"]
                ),
                surviving_literal_id=int(value["surviving_literal_id"]),
                removed_literal_id=int(value["removed_literal_id"]),
                collective_protocol=value["collective_protocol"],
                gprolog_version=value["gprolog_version"],
                gprolog_binary_digest=value["gprolog_binary_digest"],
                module_digests=tuple(tuple(item) for item in value["module_digests"]),
                schema=value["schema"],
            )
        except (TypeError, ValueError) as error:
            raise ModelGenerationError("de-escalation evidence is malformed") from error
        if result.evidence_id != value["evidence_id"]:
            raise ModelGenerationError("de-escalation evidence digest mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ThresholdCandidateBudget:
    """Explicit breadth bounds for one complete Input-PTA threshold search."""

    maximum_fields: int = 8
    maximum_candidates: int = 8

    def __post_init__(self) -> None:
        for name, value, upper in (
            ("maximum_fields", self.maximum_fields, MAX_THRESHOLD_FIELDS),
            ("maximum_candidates", self.maximum_candidates, MAX_THRESHOLD_CANDIDATES),
        ):
            if type(value) is not int or not 1 <= value <= upper:
                raise ModelGenerationError(f"{name} must be in 1..{upper}")

    def to_dict(self) -> dict[str, int]:
        return {
            "maximum_fields": self.maximum_fields,
            "maximum_candidates": self.maximum_candidates,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ThresholdCandidateBudget":
        if not isinstance(value, Mapping) or set(value) != {
            "maximum_fields",
            "maximum_candidates",
        }:
            raise ModelGenerationError("threshold candidate budget is malformed")
        try:
            return cls(value["maximum_fields"], value["maximum_candidates"])
        except (TypeError, ValueError) as error:
            raise ModelGenerationError("threshold candidate budget is malformed") from error


@dataclass(frozen=True, slots=True)
class ThresholdCandidateProposal:
    """Reviewed identity of one threshold returned by the bounded collective."""

    proposal_semantic_id: str
    proposal_provenance_id: str
    field: str
    threshold: int | float
    invented_literal_id: int
    boundary_evidence_digest: str
    proposal_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_digest(self.proposal_semantic_id, "candidate proposal semantic")
        _require_digest(self.proposal_provenance_id, "candidate proposal provenance")
        _require_digest(self.boundary_evidence_digest, "candidate boundary evidence")
        if type(self.field) is not str or not self.field:
            raise ModelGenerationError("threshold candidate field is invalid")
        if type(self.threshold) not in (int, float) or not math.isfinite(self.threshold):
            raise ModelGenerationError("threshold candidate boundary is invalid")
        if (
            type(self.invented_literal_id) is not int
            or not 0 <= self.invented_literal_id < 1 << 64
        ):
            raise ModelGenerationError("threshold candidate literal ID is invalid")
        expected_proposal_fields = {
            "proposal_id",
            "source_pta_ids",
            "supporting_insights",
            "counterexamples_addressed",
            "required_literals",
            "native_target",
            "structure",
            "weights",
            "output_assignments",
            "resource_bounds",
            "lowering_version",
            "validation_signature",
            "support_trace",
        }
        if (
            not isinstance(self.proposal_payload, Mapping)
            or set(self.proposal_payload) != expected_proposal_fields
            or self.proposal_payload.get("native_target") != "threshold"
            or not isinstance(self.proposal_payload.get("structure"), Mapping)
            or dict(self.proposal_payload["structure"])
            != {
                "field": self.field,
                "operator": "ge",
                "threshold": self.threshold,
            }
        ):
            raise ModelGenerationError(
                "threshold candidate canonical proposal payload is inconsistent"
            )
        frozen_payload = _freeze_json(self.proposal_payload)
        if not isinstance(frozen_payload, Mapping):
            raise TypeError("threshold candidate proposal payload is invalid")
        object.__setattr__(self, "proposal_payload", frozen_payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_semantic_id": self.proposal_semantic_id,
            "proposal_provenance_id": self.proposal_provenance_id,
            "field": self.field,
            "threshold": self.threshold,
            "invented_literal_id": str(self.invented_literal_id),
            "boundary_evidence_digest": self.boundary_evidence_digest,
            "proposal_payload": _thaw_json(self.proposal_payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ThresholdCandidateProposal":
        expected = {
            "proposal_semantic_id",
            "proposal_provenance_id",
            "field",
            "threshold",
            "invented_literal_id",
            "boundary_evidence_digest",
            "proposal_payload",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or any(
                type(value[name]) is not str
                for name in expected - {"threshold", "proposal_payload"}
            )
            or type(value["threshold"]) not in (int, float)
            or not isinstance(value["proposal_payload"], Mapping)
            or not value["invented_literal_id"].isdigit()
        ):
            raise ModelGenerationError("threshold candidate proposal is malformed")
        try:
            return cls(
                value["proposal_semantic_id"],
                value["proposal_provenance_id"],
                value["field"],
                value["threshold"],
                int(value["invented_literal_id"]),
                value["boundary_evidence_digest"],
                value["proposal_payload"],
            )
        except (TypeError, ValueError) as error:
            raise ModelGenerationError("threshold candidate proposal is malformed") from error


@dataclass(frozen=True, slots=True)
class PrologThresholdCandidateSet:
    """Complete, reviewed output of one bounded GNU Prolog threshold query."""

    invention_corpus_digest: str
    session_digest: str
    numeric_fields: tuple[str, ...]
    budget: ThresholdCandidateBudget
    available_candidates: int
    candidates: tuple[ThresholdCandidateProposal, ...]
    collective_protocol: str
    gprolog_version: str
    gprolog_binary_digest: str
    module_digests: tuple[tuple[str, str], ...]
    schema: str = THRESHOLD_CANDIDATE_SET_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != THRESHOLD_CANDIDATE_SET_SCHEMA:
            raise ModelGenerationError("threshold candidate-set schema is unsupported")
        _require_digest(self.invention_corpus_digest, "invention corpus")
        _require_digest(self.session_digest, "reasoning session")
        _require_digest(self.gprolog_binary_digest, "GNU Prolog executable")
        if (
            type(self.numeric_fields) is not tuple
            or not self.numeric_fields
            or any(type(field) is not str or not field for field in self.numeric_fields)
            or tuple(sorted(set(self.numeric_fields))) != self.numeric_fields
        ):
            raise ModelGenerationError("threshold candidate fields are not canonical")
        if not isinstance(self.budget, ThresholdCandidateBudget):
            raise TypeError("threshold candidate budget is invalid")
        if len(self.numeric_fields) > self.budget.maximum_fields:
            raise ModelGenerationError("threshold candidate field budget was exceeded")
        if (
            type(self.available_candidates) is not int
            or self.available_candidates != len(self.candidates)
            or not 0 < len(self.candidates) <= self.budget.maximum_candidates
        ):
            raise ModelGenerationError(
                "threshold candidate set must be complete, nonempty, and within budget"
            )
        if (
            type(self.candidates) is not tuple
            or any(not isinstance(item, ThresholdCandidateProposal) for item in self.candidates)
            or tuple(sorted(self.candidates, key=lambda item: item.proposal_semantic_id))
            != self.candidates
            or any(item.field not in self.numeric_fields for item in self.candidates)
            or len({item.proposal_semantic_id for item in self.candidates})
            != len(self.candidates)
            or len({item.proposal_provenance_id for item in self.candidates})
            != len(self.candidates)
            or len({item.invented_literal_id for item in self.candidates})
            != len(self.candidates)
        ):
            raise ModelGenerationError("threshold candidates are not canonical and unique")
        if self.collective_protocol != "PTM_PTA_COLLECTIVE_V1":
            raise ModelGenerationError("threshold collective protocol is unsupported")
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
            or len({name for name, _ in self.module_digests}) != len(self.module_digests)
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
                "numeric_fields": self.numeric_fields,
                "discover_thresholds": True,
                "discover_intervals": False,
                "derive_deescalation": False,
                "derive_escalation": True,
            }
        )

    @property
    def evidence_id(self) -> str:
        return content_digest(self.canonical_payload())

    @property
    def candidate_set_id(self) -> str:
        return self.evidence_id

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "invention_corpus_digest": self.invention_corpus_digest,
            "session_digest": self.session_digest,
            "query": _thaw_json(self.query),
            "budget": self.budget.to_dict(),
            "available_candidates": self.available_candidates,
            "candidates": [item.to_dict() for item in self.candidates],
            "collective_protocol": self.collective_protocol,
            "gprolog_version": self.gprolog_version,
            "gprolog_binary_digest": self.gprolog_binary_digest,
            "module_digests": [list(item) for item in self.module_digests],
        }

    def to_dict(self) -> dict[str, object]:
        result = self.canonical_payload()
        result["candidate_set_id"] = self.candidate_set_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PrologThresholdCandidateSet":
        expected = {
            "schema",
            "invention_corpus_digest",
            "session_digest",
            "query",
            "budget",
            "available_candidates",
            "candidates",
            "collective_protocol",
            "gprolog_version",
            "gprolog_binary_digest",
            "module_digests",
            "candidate_set_id",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or not isinstance(value["query"], Mapping)
            or not isinstance(value["budget"], Mapping)
            or not isinstance(value["candidates"], list)
            or not isinstance(value["module_digests"], list)
            or type(value["available_candidates"]) is not int
            or any(
                type(value[name]) is not str
                for name in expected
                - {"query", "budget", "available_candidates", "candidates", "module_digests"}
            )
        ):
            raise ModelGenerationError("threshold candidate set is malformed")
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
            or any(type(field) is not str for field in query["numeric_fields"])
            or query["discover_thresholds"] is not True
            or query["discover_intervals"] is not False
            or query["derive_deescalation"] is not False
            or query["derive_escalation"] is not True
            or any(
                not isinstance(item, list)
                or len(item) != 2
                or any(type(part) is not str for part in item)
                for item in value["module_digests"]
            )
        ):
            raise ModelGenerationError("threshold candidate query is malformed")
        try:
            result = cls(
                invention_corpus_digest=value["invention_corpus_digest"],
                session_digest=value["session_digest"],
                numeric_fields=tuple(query["numeric_fields"]),
                budget=ThresholdCandidateBudget.from_dict(value["budget"]),
                available_candidates=value["available_candidates"],
                candidates=tuple(
                    ThresholdCandidateProposal.from_dict(item)
                    for item in value["candidates"]
                ),
                collective_protocol=value["collective_protocol"],
                gprolog_version=value["gprolog_version"],
                gprolog_binary_digest=value["gprolog_binary_digest"],
                module_digests=tuple(tuple(item) for item in value["module_digests"]),
                schema=value["schema"],
            )
        except (TypeError, ValueError) as error:
            raise ModelGenerationError("threshold candidate set is malformed") from error
        if result.candidate_set_id != value["candidate_set_id"]:
            raise ModelGenerationError("threshold candidate-set digest mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ThresholdCandidateSelectionPolicy:
    """Adaptation-only admission policy for choosing one invention alternative."""

    minimum_observations: int = 1
    require_strict_improvement: bool = True

    def __post_init__(self) -> None:
        if type(self.minimum_observations) is not int or self.minimum_observations <= 0:
            raise ModelGenerationError("selection minimum observations must be positive")
        if type(self.require_strict_improvement) is not bool:
            raise TypeError("selection strict-improvement option must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_observations": self.minimum_observations,
            "require_strict_improvement": self.require_strict_improvement,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "ThresholdCandidateSelectionPolicy":
        if not isinstance(value, Mapping) or set(value) != {
            "minimum_observations",
            "require_strict_improvement",
        }:
            raise ModelGenerationError("threshold selection policy is malformed")
        try:
            return cls(
                value["minimum_observations"], value["require_strict_improvement"]
            )
        except (TypeError, ValueError) as error:
            raise ModelGenerationError("threshold selection policy is malformed") from error


@dataclass(frozen=True, slots=True)
class ThresholdCandidateOutcome:
    """Adaptation-corpus result for one reviewed threshold alternative."""

    proposal_semantic_id: str
    proposal_provenance_id: str
    invented_literal_id: int
    extended_snapshot_id: str
    extended_manifest_id: str
    child_snapshot_id: str
    child_manifest_id: str
    child_preprocessing_id: str
    adaptive_behavior_id: str
    observations: int
    parent_errors: int
    child_errors: int
    disagreements: int
    improvements: int
    regressions: int
    both_correct: int
    both_wrong: int

    def __post_init__(self) -> None:
        for label, value in (
            ("proposal semantic", self.proposal_semantic_id),
            ("proposal provenance", self.proposal_provenance_id),
            ("extended snapshot", self.extended_snapshot_id),
            ("extended manifest", self.extended_manifest_id),
            ("child snapshot", self.child_snapshot_id),
            ("child manifest", self.child_manifest_id),
            ("child preprocessing", self.child_preprocessing_id),
            ("adaptive behavior", self.adaptive_behavior_id),
        ):
            _require_digest(value, label)
        if (
            type(self.invented_literal_id) is not int
            or not 0 <= self.invented_literal_id < 1 << 64
        ):
            raise ModelGenerationError("candidate outcome literal ID is invalid")
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
            raise ModelGenerationError("candidate outcome counts must be nonnegative integers")
        if (
            self.observations <= 0
            or self.both_correct
            + self.both_wrong
            + self.improvements
            + self.regressions
            != self.observations
            or self.parent_errors != self.both_wrong + self.improvements
            or self.child_errors != self.both_wrong + self.regressions
            or self.disagreements != self.improvements + self.regressions
        ):
            raise ModelGenerationError("candidate outcome counts are inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_semantic_id": self.proposal_semantic_id,
            "proposal_provenance_id": self.proposal_provenance_id,
            "invented_literal_id": str(self.invented_literal_id),
            "extended_snapshot_id": self.extended_snapshot_id,
            "extended_manifest_id": self.extended_manifest_id,
            "child_snapshot_id": self.child_snapshot_id,
            "child_manifest_id": self.child_manifest_id,
            "child_preprocessing_id": self.child_preprocessing_id,
            "adaptive_behavior_id": self.adaptive_behavior_id,
            "observations": self.observations,
            "parent_errors": self.parent_errors,
            "child_errors": self.child_errors,
            "disagreements": self.disagreements,
            "improvements": self.improvements,
            "regressions": self.regressions,
            "both_correct": self.both_correct,
            "both_wrong": self.both_wrong,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ThresholdCandidateOutcome":
        expected = {
            "proposal_semantic_id",
            "proposal_provenance_id",
            "invented_literal_id",
            "extended_snapshot_id",
            "extended_manifest_id",
            "child_snapshot_id",
            "child_manifest_id",
            "child_preprocessing_id",
            "adaptive_behavior_id",
            "observations",
            "parent_errors",
            "child_errors",
            "disagreements",
            "improvements",
            "regressions",
            "both_correct",
            "both_wrong",
        }
        count_names = {
            "observations",
            "parent_errors",
            "child_errors",
            "disagreements",
            "improvements",
            "regressions",
            "both_correct",
            "both_wrong",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or any(type(value[name]) is not int for name in count_names)
            or any(type(value[name]) is not str for name in expected - count_names)
            or not value["invented_literal_id"].isdigit()
        ):
            raise ModelGenerationError("threshold candidate outcome is malformed")
        try:
            return cls(
                proposal_semantic_id=value["proposal_semantic_id"],
                proposal_provenance_id=value["proposal_provenance_id"],
                invented_literal_id=int(value["invented_literal_id"]),
                extended_snapshot_id=value["extended_snapshot_id"],
                extended_manifest_id=value["extended_manifest_id"],
                child_snapshot_id=value["child_snapshot_id"],
                child_manifest_id=value["child_manifest_id"],
                child_preprocessing_id=value["child_preprocessing_id"],
                adaptive_behavior_id=value["adaptive_behavior_id"],
                observations=value["observations"],
                parent_errors=value["parent_errors"],
                child_errors=value["child_errors"],
                disagreements=value["disagreements"],
                improvements=value["improvements"],
                regressions=value["regressions"],
                both_correct=value["both_correct"],
                both_wrong=value["both_wrong"],
            )
        except (TypeError, ValueError) as error:
            raise ModelGenerationError("threshold candidate outcome is malformed") from error


@dataclass(frozen=True, slots=True)
class ThresholdCandidateSelection:
    """Deterministic adaptation-only selection over a complete candidate set."""

    candidate_set_id: str
    parent_generation_id: str
    parent_snapshot_id: str
    parent_manifest_id: str
    adaptation_corpus_digest: str
    adaptation_epochs: int
    policy: ThresholdCandidateSelectionPolicy
    outcomes: tuple[ThresholdCandidateOutcome, ...]
    selected_proposal_semantic_id: str
    selected_proposal_provenance_id: str
    schema: str = THRESHOLD_CANDIDATE_SELECTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != THRESHOLD_CANDIDATE_SELECTION_SCHEMA:
            raise ModelGenerationError("threshold candidate-selection schema is unsupported")
        for label, value in (
            ("candidate set", self.candidate_set_id),
            ("parent generation", self.parent_generation_id),
            ("parent snapshot", self.parent_snapshot_id),
            ("parent manifest", self.parent_manifest_id),
            ("adaptation corpus", self.adaptation_corpus_digest),
            ("selected proposal semantic", self.selected_proposal_semantic_id),
            ("selected proposal provenance", self.selected_proposal_provenance_id),
        ):
            _require_digest(value, label)
        if (
            type(self.adaptation_epochs) is not int
            or not 0 < self.adaptation_epochs <= MAX_ADAPTATION_EPOCHS
        ):
            raise ModelGenerationError("selection adaptation epochs are invalid")
        if not isinstance(self.policy, ThresholdCandidateSelectionPolicy):
            raise TypeError("threshold candidate-selection policy is invalid")
        if (
            type(self.outcomes) is not tuple
            or not self.outcomes
            or any(not isinstance(item, ThresholdCandidateOutcome) for item in self.outcomes)
            or tuple(sorted(self.outcomes, key=lambda item: item.proposal_semantic_id))
            != self.outcomes
            or len({item.proposal_semantic_id for item in self.outcomes})
            != len(self.outcomes)
            or len({item.proposal_provenance_id for item in self.outcomes})
            != len(self.outcomes)
        ):
            raise ModelGenerationError("threshold candidate outcomes are not canonical")
        eligible = tuple(
            item
            for item in self.outcomes
            if item.observations >= self.policy.minimum_observations
            and (
                item.child_errors < item.parent_errors
                if self.policy.require_strict_improvement
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
        if (
            winner.proposal_semantic_id != self.selected_proposal_semantic_id
            or winner.proposal_provenance_id != self.selected_proposal_provenance_id
        ):
            raise ModelGenerationError("threshold candidate selection is not deterministic")

    @property
    def selected_outcome(self) -> ThresholdCandidateOutcome:
        return next(
            item
            for item in self.outcomes
            if item.proposal_semantic_id == self.selected_proposal_semantic_id
        )

    @property
    def selection_id(self) -> str:
        return content_digest(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "candidate_set_id": self.candidate_set_id,
            "parent_generation_id": self.parent_generation_id,
            "parent_snapshot_id": self.parent_snapshot_id,
            "parent_manifest_id": self.parent_manifest_id,
            "adaptation_corpus_digest": self.adaptation_corpus_digest,
            "adaptation_epochs": self.adaptation_epochs,
            "policy": self.policy.to_dict(),
            "outcomes": [item.to_dict() for item in self.outcomes],
            "selected_proposal_semantic_id": self.selected_proposal_semantic_id,
            "selected_proposal_provenance_id": self.selected_proposal_provenance_id,
        }

    def to_dict(self) -> dict[str, object]:
        result = self.canonical_payload()
        result["selection_id"] = self.selection_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ThresholdCandidateSelection":
        expected = {
            "schema",
            "candidate_set_id",
            "parent_generation_id",
            "parent_snapshot_id",
            "parent_manifest_id",
            "adaptation_corpus_digest",
            "adaptation_epochs",
            "policy",
            "outcomes",
            "selected_proposal_semantic_id",
            "selected_proposal_provenance_id",
            "selection_id",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or type(value["adaptation_epochs"]) is not int
            or not isinstance(value["policy"], Mapping)
            or not isinstance(value["outcomes"], list)
            or any(
                type(value[name]) is not str
                for name in expected - {"adaptation_epochs", "policy", "outcomes"}
            )
        ):
            raise ModelGenerationError("threshold candidate selection is malformed")
        try:
            result = cls(
                candidate_set_id=value["candidate_set_id"],
                parent_generation_id=value["parent_generation_id"],
                parent_snapshot_id=value["parent_snapshot_id"],
                parent_manifest_id=value["parent_manifest_id"],
                adaptation_corpus_digest=value["adaptation_corpus_digest"],
                adaptation_epochs=value["adaptation_epochs"],
                policy=ThresholdCandidateSelectionPolicy.from_dict(value["policy"]),
                outcomes=tuple(
                    ThresholdCandidateOutcome.from_dict(item)
                    for item in value["outcomes"]
                ),
                selected_proposal_semantic_id=value["selected_proposal_semantic_id"],
                selected_proposal_provenance_id=value[
                    "selected_proposal_provenance_id"
                ],
                schema=value["schema"],
            )
        except ModelGenerationError:
            raise
        except (TypeError, ValueError) as error:
            raise ModelGenerationError("threshold candidate selection is malformed") from error
        if result.selection_id != value["selection_id"]:
            raise ModelGenerationError("threshold candidate-selection digest mismatch")
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
    CONTRACTED_PARENT = "contracted_parent"
    CONTRACTED_CHILD = "contracted_child"


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
        if generation.kind not in (
            GenerationKind.ADAPTED_CHILD,
            GenerationKind.CONTRACTED_CHILD,
        ):
            raise ModelGenerationError("adaptive behavior requires a deployable child")
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
class ContractedParent:
    parent_snapshot: AdaptiveSnapshotEnvelope
    parent_manifest: OrderedLiteralManifest
    snapshot: AdaptiveSnapshotEnvelope
    manifest: OrderedLiteralManifest
    evidence: PrologDeescalationEvidence
    proof_case_count: int
    confirmation_case_count: int


def contract_parent_with_equivalent_literal(
    parent_snapshot: TMSnapshot,
    parent_manifest: OrderedLiteralManifest,
    evidence: PrologDeescalationEvidence,
    *,
    proof_records: Sequence[Mapping[str, object]],
    confirmation_records: Sequence[Mapping[str, object]],
) -> ContractedParent:
    """Build a smaller adaptive generation from an attested equivalent pair."""

    parent_envelope = AdaptiveSnapshotEnvelope(parent_snapshot)
    if (
        parent_snapshot.number_of_features != len(parent_manifest.literals)
        or evidence.parent_snapshot_id != parent_envelope.snapshot_id
        or evidence.parent_manifest_id != parent_manifest.manifest_id
    ):
        raise ModelGenerationError(
            "de-escalation evidence names a different parent representation"
        )
    try:
        survivor_position = parent_manifest.literal_ids.index(
            evidence.surviving_literal_id
        )
        removed_position = parent_manifest.literal_ids.index(
            evidence.removed_literal_id
        )
    except ValueError as error:
        raise ModelGenerationError(
            "de-escalation literal is absent from the parent manifest"
        ) from error
    if survivor_position >= removed_position:
        raise ModelGenerationError(
            "de-escalation must preserve the earliest equivalent feature position"
        )

    proof = tuple(proof_records)
    confirmation = tuple(confirmation_records)
    if not proof or not confirmation:
        raise ModelGenerationError(
            "de-escalation requires nonempty proof and confirmation records"
        )
    parent_catalog = parent_manifest.build_catalog()
    for label, records in (("proof", proof), ("confirmation", confirmation)):
        batch = parent_catalog.encode(records).ta
        survivor_column = tuple(
            batch.bit(row, survivor_position) for row in range(batch.row_count)
        )
        removed_column = tuple(
            batch.bit(row, removed_position) for row in range(batch.row_count)
        )
        if survivor_column != removed_column:
            raise ModelGenerationError(
                f"de-escalation literals differ on the {label} corpus"
            )

    contracted_literals = (
        parent_manifest.literals[:removed_position]
        + parent_manifest.literals[removed_position + 1 :]
    )
    contracted_manifest = OrderedLiteralManifest(
        parent_manifest.fields, contracted_literals
    )
    contracted_snapshot = contract_snapshot_equivalent_feature(
        parent_snapshot, survivor_position, removed_position
    )
    if (
        contracted_manifest.literals != contracted_literals
        or contracted_snapshot.number_of_features
        != parent_snapshot.number_of_features - 1
        or contracted_snapshot.number_of_clauses
        != parent_snapshot.number_of_clauses
        or contracted_snapshot.states_per_action != parent_snapshot.states_per_action
        or contracted_snapshot.specificity != parent_snapshot.specificity
        or contracted_snapshot.threshold != parent_snapshot.threshold
        or contracted_snapshot.rng_state != parent_snapshot.rng_state
    ):
        raise ModelGenerationError("de-escalation structural contraction failed")

    contracted_catalog = contracted_manifest.build_catalog()
    parent_machine = _machine_from_snapshot(parent_snapshot)
    contracted_machine = _machine_from_snapshot(contracted_snapshot)
    for label, records in (("proof", proof), ("confirmation", confirmation)):
        parent_batch = parent_catalog.encode(records).ta
        contracted_batch = contracted_catalog.encode(records).ta
        parent_rows = tuple(
            parent_batch.row_values(index) for index in range(parent_batch.row_count)
        )
        contracted_rows = tuple(
            contracted_batch.row_values(index)
            for index in range(contracted_batch.row_count)
        )
        parent_results = tuple(
            (
                tuple(
                    parent_machine.clause_output(clause, row)
                    for clause in range(parent_snapshot.number_of_clauses)
                ),
                parent_machine.score(row),
                parent_machine.predict_one(row),
            )
            for row in parent_rows
        )
        contracted_results = tuple(
            (
                tuple(
                    contracted_machine.clause_output(clause, row)
                    for clause in range(contracted_snapshot.number_of_clauses)
                ),
                contracted_machine.score(row),
                contracted_machine.predict_one(row),
            )
            for row in contracted_rows
        )
        if parent_results != contracted_results:
            raise ModelGenerationError(
                f"de-escalation behavioral equivalence failed on the {label} corpus"
            )

    return ContractedParent(
        parent_envelope,
        parent_manifest,
        AdaptiveSnapshotEnvelope(contracted_snapshot),
        contracted_manifest,
        evidence,
        len(proof),
        len(confirmation),
    )


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
    evidence_usage_id: str
    activation_sequence: int
    previous_activated_lineage_id: str | None
    invented_literal_id: int
    invention_corpus_digest: str
    adaptation_corpus_digest: str
    promotion_corpus_digest: str
    origin_proposal_semantic_id: str
    origin_proposal_provenance_id: str
    candidate_selection_id: str | None = None
    schema: str = LEGACY_LINEAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema not in (LEGACY_LINEAGE_SCHEMA, LINEAGE_SCHEMA):
            raise ModelGenerationError("model-generation lineage schema is unsupported")
        for label, value in (
            ("parent generation", self.parent_generation_id),
            ("extended generation", self.extended_generation_id),
            ("child generation", self.child_generation_id),
            ("adaptive behavior", self.adaptive_behavior_id),
            ("restoration bundle", self.restoration_bundle_id),
            ("promotion audit", self.promotion_audit_id),
            ("invention evidence", self.invention_evidence_id),
            ("evidence usage", self.evidence_usage_id),
            ("invention corpus", self.invention_corpus_digest),
            ("adaptation corpus", self.adaptation_corpus_digest),
            ("promotion corpus", self.promotion_corpus_digest),
            ("origin proposal semantic ID", self.origin_proposal_semantic_id),
            ("origin proposal provenance ID", self.origin_proposal_provenance_id),
        ):
            _require_digest(value, label)
        if (
            type(self.activation_sequence) is not int
            or self.activation_sequence <= 0
        ):
            raise ModelGenerationError("activation sequence must be positive")
        if self.previous_activated_lineage_id is not None:
            _require_digest(
                self.previous_activated_lineage_id,
                "previous activated lineage",
            )
        if self.schema == LINEAGE_SCHEMA:
            _require_digest(self.candidate_selection_id, "candidate selection")
        elif self.candidate_selection_id is not None:
            raise ModelGenerationError("legacy lineage cannot reference candidate selection")
        if type(self.invented_literal_id) is not int or not 0 <= self.invented_literal_id < 1 << 64:
            raise ModelGenerationError("invented literal ID must be unsigned 64-bit")

    @property
    def lineage_id(self) -> str:
        return content_digest(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        result = {
            "schema": self.schema,
            "parent_generation_id": self.parent_generation_id,
            "extended_generation_id": self.extended_generation_id,
            "child_generation_id": self.child_generation_id,
            "adaptive_behavior_id": self.adaptive_behavior_id,
            "restoration_bundle_id": self.restoration_bundle_id,
            "promotion_audit_id": self.promotion_audit_id,
            "invention_evidence_id": self.invention_evidence_id,
            "evidence_usage_id": self.evidence_usage_id,
            "activation_sequence": self.activation_sequence,
            "previous_activated_lineage_id": self.previous_activated_lineage_id,
            "invented_literal_id": str(self.invented_literal_id),
            "invention_corpus_digest": self.invention_corpus_digest,
            "adaptation_corpus_digest": self.adaptation_corpus_digest,
            "promotion_corpus_digest": self.promotion_corpus_digest,
            "origin_proposal_semantic_id": self.origin_proposal_semantic_id,
            "origin_proposal_provenance_id": self.origin_proposal_provenance_id,
        }
        if self.schema == LINEAGE_SCHEMA:
            result["candidate_selection_id"] = self.candidate_selection_id
        return result

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
            "evidence_usage_id",
            "activation_sequence",
            "previous_activated_lineage_id",
            "invented_literal_id",
            "invention_corpus_digest",
            "adaptation_corpus_digest",
            "promotion_corpus_digest",
            "origin_proposal_semantic_id",
            "origin_proposal_provenance_id",
            "lineage_id",
        }
        if isinstance(value, Mapping) and value.get("schema") == LINEAGE_SCHEMA:
            expected.add("candidate_selection_id")
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or any(
                type(value[key]) is not str
                for key in expected
                - {"activation_sequence", "previous_activated_lineage_id"}
            )
            or type(value["activation_sequence"]) is not int
            or (
                value["previous_activated_lineage_id"] is not None
                and type(value["previous_activated_lineage_id"]) is not str
            )
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
                evidence_usage_id=value["evidence_usage_id"],
                activation_sequence=value["activation_sequence"],
                previous_activated_lineage_id=value[
                    "previous_activated_lineage_id"
                ],
                invented_literal_id=int(raw_literal_id),
                invention_corpus_digest=value["invention_corpus_digest"],
                adaptation_corpus_digest=value["adaptation_corpus_digest"],
                promotion_corpus_digest=value["promotion_corpus_digest"],
                origin_proposal_semantic_id=value["origin_proposal_semantic_id"],
                origin_proposal_provenance_id=value["origin_proposal_provenance_id"],
                candidate_selection_id=value.get("candidate_selection_id"),
                schema=value["schema"],
            )
        except (TypeError, ValueError) as error:
            raise ModelGenerationError("model-generation lineage is malformed") from error
        if result.lineage_id != value["lineage_id"]:
            raise ModelGenerationError("model-generation lineage digest mismatch")
        return result


@dataclass(frozen=True, slots=True)
class LiteralContractionLineage:
    parent_generation_id: str
    contracted_generation_id: str
    child_generation_id: str
    adaptive_behavior_id: str
    restoration_bundle_id: str
    promotion_audit_id: str
    deescalation_evidence_id: str
    evidence_usage_id: str
    activation_sequence: int
    previous_activated_lineage_id: str | None
    surviving_literal_id: int
    removed_literal_id: int
    proof_corpus_digest: str
    confirmation_corpus_digest: str
    promotion_corpus_digest: str
    schema: str = CONTRACTION_LINEAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CONTRACTION_LINEAGE_SCHEMA:
            raise ModelGenerationError("literal-contraction lineage schema is unsupported")
        for label, value in (
            ("parent generation", self.parent_generation_id),
            ("contracted generation", self.contracted_generation_id),
            ("child generation", self.child_generation_id),
            ("adaptive behavior", self.adaptive_behavior_id),
            ("restoration bundle", self.restoration_bundle_id),
            ("promotion audit", self.promotion_audit_id),
            ("de-escalation evidence", self.deescalation_evidence_id),
            ("evidence usage", self.evidence_usage_id),
            ("de-escalation proof corpus", self.proof_corpus_digest),
            ("de-escalation confirmation corpus", self.confirmation_corpus_digest),
            ("promotion corpus", self.promotion_corpus_digest),
        ):
            _require_digest(value, label)
        if (
            type(self.activation_sequence) is not int
            or self.activation_sequence <= 0
        ):
            raise ModelGenerationError("activation sequence must be positive")
        if self.previous_activated_lineage_id is not None:
            _require_digest(
                self.previous_activated_lineage_id,
                "previous activated lineage",
            )
        if (
            any(
                type(item) is not int or not 0 <= item < 1 << 64
                for item in (self.surviving_literal_id, self.removed_literal_id)
            )
            or self.surviving_literal_id == self.removed_literal_id
        ):
            raise ModelGenerationError("literal-contraction identities are invalid")

    @property
    def extended_generation_id(self) -> str:
        """Compatibility name used by the generic lifecycle event envelope."""

        return self.contracted_generation_id

    @property
    def lineage_id(self) -> str:
        return content_digest(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "parent_generation_id": self.parent_generation_id,
            "contracted_generation_id": self.contracted_generation_id,
            "child_generation_id": self.child_generation_id,
            "adaptive_behavior_id": self.adaptive_behavior_id,
            "restoration_bundle_id": self.restoration_bundle_id,
            "promotion_audit_id": self.promotion_audit_id,
            "deescalation_evidence_id": self.deescalation_evidence_id,
            "evidence_usage_id": self.evidence_usage_id,
            "activation_sequence": self.activation_sequence,
            "previous_activated_lineage_id": self.previous_activated_lineage_id,
            "surviving_literal_id": str(self.surviving_literal_id),
            "removed_literal_id": str(self.removed_literal_id),
            "proof_corpus_digest": self.proof_corpus_digest,
            "confirmation_corpus_digest": self.confirmation_corpus_digest,
            "promotion_corpus_digest": self.promotion_corpus_digest,
        }

    def to_dict(self) -> dict[str, object]:
        result = self.canonical_payload()
        result["lineage_id"] = self.lineage_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LiteralContractionLineage":
        expected = {
            "schema",
            "parent_generation_id",
            "contracted_generation_id",
            "child_generation_id",
            "adaptive_behavior_id",
            "restoration_bundle_id",
            "promotion_audit_id",
            "deescalation_evidence_id",
            "evidence_usage_id",
            "activation_sequence",
            "previous_activated_lineage_id",
            "surviving_literal_id",
            "removed_literal_id",
            "proof_corpus_digest",
            "confirmation_corpus_digest",
            "promotion_corpus_digest",
            "lineage_id",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or any(
                type(value[key]) is not str
                for key in expected
                - {"activation_sequence", "previous_activated_lineage_id"}
            )
            or type(value["activation_sequence"]) is not int
            or (
                value["previous_activated_lineage_id"] is not None
                and type(value["previous_activated_lineage_id"]) is not str
            )
            or not value["surviving_literal_id"].isdigit()
            or not value["removed_literal_id"].isdigit()
        ):
            raise ModelGenerationError("literal-contraction lineage is malformed")
        try:
            result = cls(
                parent_generation_id=value["parent_generation_id"],
                contracted_generation_id=value["contracted_generation_id"],
                child_generation_id=value["child_generation_id"],
                adaptive_behavior_id=value["adaptive_behavior_id"],
                restoration_bundle_id=value["restoration_bundle_id"],
                promotion_audit_id=value["promotion_audit_id"],
                deescalation_evidence_id=value["deescalation_evidence_id"],
                evidence_usage_id=value["evidence_usage_id"],
                activation_sequence=value["activation_sequence"],
                previous_activated_lineage_id=value[
                    "previous_activated_lineage_id"
                ],
                surviving_literal_id=int(value["surviving_literal_id"]),
                removed_literal_id=int(value["removed_literal_id"]),
                proof_corpus_digest=value["proof_corpus_digest"],
                confirmation_corpus_digest=value["confirmation_corpus_digest"],
                promotion_corpus_digest=value["promotion_corpus_digest"],
                schema=value["schema"],
            )
        except (TypeError, ValueError) as error:
            raise ModelGenerationError(
                "literal-contraction lineage is malformed"
            ) from error
        if result.lineage_id != value["lineage_id"]:
            raise ModelGenerationError("literal-contraction lineage digest mismatch")
        return result
