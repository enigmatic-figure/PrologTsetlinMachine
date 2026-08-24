"""PTA proposal ontology — typed, deterministic, lowerability-checked.

This module implements the first Milestone 7 deliverable: the exact lowering
boundary between the PTA reasoning plane and the native Tsetlin plane. It is
deliberately small, dependency-free, and hostile-input tested. Prolog may be
arbitrarily more expressive during deliberation; only a proposal for which
`lower_exact()` constructs and validates an exact executable representation
becomes a lowered candidate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from ..representation import LiteralDescriptor

LOWERING_VERSION = "pta.lowering.v2"
MAX_LITERALS_PER_CLAUSE = 64
MAX_CLAUSES = 1024
MAX_GRAPH_DEPTH = 8

NativeTarget = Literal[
    "binary_clause",
    "logic_program",
    "threshold",
    "shared_weighted_clause",
    "regression_clause",
    "graph_clause",
    "patch_clause",
    "composite_gate",
]

NATIVE_TARGETS: tuple[NativeTarget, ...] = (
    "binary_clause",
    "logic_program",
    "threshold",
    "shared_weighted_clause",
    "regression_clause",
    "graph_clause",
    "patch_clause",
    "composite_gate",
)

_VALID_TARGETS = frozenset(NATIVE_TARGETS)


def _canon(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sequence_tuple(value: Any, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    return tuple(value)


def _printable_string(value: Any, field_name: str, *, nonempty: bool = True) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if (nonempty and not value) or any(ord(character) < 0x20 for character in value):
        qualifier = "nonempty " if nonempty else ""
        raise ValueError(f"{field_name} must be a {qualifier}printable string")
    return value


def _semantic_literal_identity(value: LiteralDescriptor | str) -> int | str:
    if isinstance(value, LiteralDescriptor):
        return value.literal_id
    prefix = "literal:"
    if value.startswith(prefix):
        raw = value[len(prefix) :]
        if raw and raw.isascii() and raw.isdecimal() and str(int(raw)) == raw:
            return int(raw)
    return value


def _deep_freeze(value: Any) -> Any:
    """Recursively freeze to immutable, JSON-compatible, canonical values."""
    if isinstance(value, Mapping):
        from types import MappingProxyType
        # Reject non-string keys — 1 and "1" must not collide
        for k in value.keys():
            if not isinstance(k, str):
                raise TypeError(f"mapping keys must be strings, got {type(k).__name__}: {k!r}")
        return MappingProxyType({k: _deep_freeze(v) for k, v in sorted(value.items(), key=lambda kv: kv[0])})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_deep_freeze(v) for v in value), key=lambda x: _canon(x)))
    if isinstance(value, (str, int, float, bool)) or value is None:
        # Use int for bool check: bool is subclass of int, so check bool first already done
        if isinstance(value, float) and not __import__("math").isfinite(value):
            raise ValueError("non-finite float not allowed in proposal")
        return value
    # Reject unsupported types rather than str(value) — unstable
    raise TypeError(f"unsupported proposal value type: {type(value).__name__}: {value!r}")


def _deep_canonicalize(value: Any) -> Any:
    return _deep_freeze(value)


def _freeze_mapping(m: Mapping[str, Any]) -> Mapping[str, Any]:
    from types import MappingProxyType
    frozen = {}
    for k, v in m.items():
        if not isinstance(k, str):
            raise TypeError(f"mapping keys must be strings, got {type(k).__name__}: {k!r}")
        frozen[k] = _deep_freeze(v)
    return MappingProxyType(frozen)


def _freeze_evidence(evidence: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(_deep_freeze(v) for v in evidence)


def _deep_thaw(value: Any) -> Any:
    from types import MappingProxyType
    if isinstance(value, MappingProxyType):
        return {k: _deep_thaw(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _deep_thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_thaw(v) for v in value]
    return value


def _thaw_for_json(value: Any) -> Any:
    # Recursively thaw to JSON-serializable
    from types import MappingProxyType
    if isinstance(value, MappingProxyType):
        return {k: _thaw_for_json(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _thaw_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_for_json(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class PTAInsight:
    source_pta: str
    kind: str  # e.g. literal_redundant, clause_subsumes, thresholds_equivalent
    subject: str
    evidence: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        _printable_string(self.source_pta, "source_pta")
        _printable_string(self.kind, "kind")
        _printable_string(self.subject, "subject")
        object.__setattr__(
            self,
            "evidence",
            _freeze_evidence(_sequence_tuple(self.evidence, "evidence")),
        )


@dataclass(frozen=True, slots=True)
class PTAEscalationProposal:
    """Typed escalation proposal — trust-boundary object, canonical and content-addressed.

    Two identities:
      semantic_id: native_target + exact literal descriptors/IDs + exact structure + weights/output_assignments + lowering schema
      provenance_id: semantic_id + source PTAs + insights/evidence + counterexamples + validation_signature + support_trace
    """

    proposal_id: str
    source_pta_ids: tuple[str, ...]
    supporting_insights: tuple[PTAInsight, ...]
    counterexamples_addressed: tuple[int, ...]
    required_literals: tuple[LiteralDescriptor | str, ...]  # existing IDs or "transform:field:params"
    native_target: NativeTarget
    structure: Mapping[str, Any]  # e.g. {"clause": [104,105,231,388]} — target-specific
    weights: tuple[int, ...] | None = None
    output_assignments: tuple[tuple[int, int], ...] | None = None  # for CoTM: (clause, class)->weight
    resource_bounds: Mapping[str, int] = field(default_factory=dict)
    lowering_version: str = LOWERING_VERSION
    validation_signature: Mapping[str, Any] = field(default_factory=dict)
    support_trace: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _printable_string(self.proposal_id, "proposal_id")
        if type(self.native_target) is not str:
            raise TypeError("native_target must be a string")
        if self.native_target not in _VALID_TARGETS:
            raise ValueError(f"unknown native_target {self.native_target}")
        if type(self.lowering_version) is not str:
            raise TypeError("lowering_version must be a string")
        if self.lowering_version != LOWERING_VERSION:
            raise ValueError("unsupported lowering_version")

        source_pta_ids = _sequence_tuple(self.source_pta_ids, "source_pta_ids")
        if not source_pta_ids:
            raise ValueError("source_pta_ids must be nonempty")
        for source_pta_id in source_pta_ids:
            _printable_string(source_pta_id, "source_pta_ids item")

        supporting_insights = _sequence_tuple(
            self.supporting_insights, "supporting_insights"
        )
        if any(not isinstance(value, PTAInsight) for value in supporting_insights):
            raise TypeError("supporting_insights items must be PTAInsight")

        counterexamples = _sequence_tuple(
            self.counterexamples_addressed, "counterexamples_addressed"
        )
        if any(type(value) is not int for value in counterexamples):
            raise TypeError("counterexamples_addressed items must be integers")
        if any(value < 0 for value in counterexamples):
            raise ValueError("counterexamples_addressed items must be nonnegative")

        required_literals = _sequence_tuple(
            self.required_literals, "required_literals"
        )
        for literal in required_literals:
            if isinstance(literal, LiteralDescriptor):
                continue
            _printable_string(literal, "required_literals item")

        weights: tuple[int, ...] | None = None
        if self.weights is not None:
            raw_weights = _sequence_tuple(self.weights, "weights")
            if any(type(weight) is not int for weight in raw_weights):
                raise TypeError("weights items must be integers")
            weights = raw_weights

        output_assignments: tuple[tuple[int, int], ...] | None = None
        if self.output_assignments is not None:
            assignments = []
            for assignment in _sequence_tuple(
                self.output_assignments, "output_assignments"
            ):
                pair = _sequence_tuple(assignment, "output_assignments item")
                if len(pair) != 2 or any(type(value) is not int for value in pair):
                    raise TypeError(
                        "output_assignments items must be pairs of nonnegative integers"
                    )
                if any(value < 0 for value in pair):
                    raise ValueError(
                        "output_assignments items must be pairs of nonnegative integers"
                    )
                assignments.append((pair[0], pair[1]))
            output_assignments = tuple(assignments)

        support_trace = _sequence_tuple(self.support_trace, "support_trace")
        for item in support_trace:
            _printable_string(item, "support_trace item", nonempty=False)

        for field_name, value in (
            ("structure", self.structure),
            ("resource_bounds", self.resource_bounds),
            ("validation_signature", self.validation_signature),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"{field_name} must be a mapping")

        # resource bounds must be positive ints within known ceilings
        for k, v in self.resource_bounds.items():
            if type(v) is not int or v <= 0:
                raise ValueError(f"resource_bounds[{k}] must be positive int")
        if self.resource_bounds.get("clause_count", 1) > MAX_CLAUSES:
            raise ValueError("clause_count exceeds MAX_CLAUSES")
        if self.resource_bounds.get("graph_depth", 1) > MAX_GRAPH_DEPTH:
            raise ValueError("graph_depth exceeds MAX_GRAPH_DEPTH")
        object.__setattr__(self, "source_pta_ids", source_pta_ids)
        object.__setattr__(self, "supporting_insights", supporting_insights)
        object.__setattr__(self, "counterexamples_addressed", counterexamples)
        object.__setattr__(self, "required_literals", required_literals)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "output_assignments", output_assignments)
        object.__setattr__(self, "support_trace", support_trace)
        # Deep-freeze mappings with strict key check (reject non-string keys, not str(k))
        def _strict_freeze(m):
            from types import MappingProxyType
            frozen = {}
            for k, v in m.items():
                if not isinstance(k, str):
                    raise TypeError(f"mapping keys must be strings, got {type(k).__name__}: {k!r}")
                frozen[k] = _deep_freeze(v)
            return MappingProxyType(frozen)
        object.__setattr__(self, "structure", _strict_freeze(dict(self.structure)))
        object.__setattr__(self, "resource_bounds", _strict_freeze(dict(self.resource_bounds)))
        object.__setattr__(self, "validation_signature", _strict_freeze(dict(self.validation_signature)))

    def _semantic_payload(self) -> dict[str, Any]:
        lits = [
            _semantic_literal_identity(literal)
            for literal in self.required_literals
        ]
        return {
            "native_target": self.native_target,
            "required_literals": sorted(lits, key=lambda x: str(x)),
            "structure": _thaw_for_json(dict(self.structure)),
            "weights": _thaw_for_json(list(self.weights)) if self.weights is not None else None,
            "output_assignments": _thaw_for_json([list(p) for p in self.output_assignments]) if self.output_assignments is not None else None,
            "resource_bounds": _thaw_for_json(dict(self.resource_bounds)),
            "lowering_version": self.lowering_version,
        }

    def _provenance_payload(self) -> dict[str, Any]:
        d = self._semantic_payload()
        d.update({
            "proposal_id": self.proposal_id,
            "source_pta_ids": sorted(_thaw_for_json(list(self.source_pta_ids))),
            "supporting_insights": sorted([{"source_pta": i.source_pta, "kind": i.kind, "subject": i.subject, "evidence": _thaw_for_json(list(i.evidence))} for i in self.supporting_insights], key=lambda x: _canon(x)),
            "counterexamples_addressed": sorted(self.counterexamples_addressed),
            "validation_signature": _thaw_for_json(dict(self.validation_signature)),
            "support_trace": sorted(_thaw_for_json(list(self.support_trace))),
        })
        return d

    def semantic_id(self) -> str:
        """Content address of exact native semantics."""
        return "sha256:" + hashlib.sha256(_canon(self._semantic_payload()).encode()).hexdigest()

    def provenance_id(self) -> str:
        """Content address of full provenance (semantic + evidence)."""
        return "sha256:" + hashlib.sha256(_canon(self._provenance_payload()).encode()).hexdigest()

    def proposal_hash(self) -> str:
        """Stable hash of the proposal content — for provenance (now includes all fields)."""
        # Backward compat: now covers required_literals, weights, insights, counterexamples, validation_signature, support_trace
        return self.provenance_id()

    def to_dict(self) -> dict[str, Any]:
        value = {
            "proposal_id": self.proposal_id,
            "source_pta_ids": list(self.source_pta_ids),
            "supporting_insights": [
                {"source_pta": i.source_pta, "kind": i.kind, "subject": i.subject, "evidence": list(i.evidence)}
                for i in self.supporting_insights
            ],
            "counterexamples_addressed": list(self.counterexamples_addressed),
            "required_literals": [lit if isinstance(lit, str) else lit.literal_id for lit in self.required_literals],
            "native_target": self.native_target,
            "structure": dict(self.structure),
            "weights": list(self.weights) if self.weights is not None else None,
            "output_assignments": [list(p) for p in self.output_assignments] if self.output_assignments is not None else None,
            "resource_bounds": dict(self.resource_bounds),
            "lowering_version": self.lowering_version,
            "validation_signature": dict(self.validation_signature),
            "support_trace": list(self.support_trace),
        }
        thawed = _thaw_for_json(value)
        assert isinstance(thawed, dict)
        return thawed


@dataclass(frozen=True, slots=True)
class PTAMorphologyProposal:
    """Class II lifecycle morphology — content-addressed, behavior-changing.

    Produces a candidate child model requiring oracle/shadow validation and
    restoration lineage. Identity is parent artifact + exact removed literals/clauses + candidate bank + insights + schema version.
    """

    morphology_id: str
    parent_artifact_id: str | None
    source_pta_ids: tuple[str, ...]
    supporting_insights: tuple[PTAInsight, ...]
    removed_literals: tuple[int, ...]
    removed_clause_ids: tuple[int, ...] = ()
    removed_clauses: int = 0
    morphed_bank: Any = None  # SparseClauseBank dict
    resource_bounds: Mapping[str, Any] = field(default_factory=dict)
    morphology_version: str = "pta.morphology.v1"

    def __post_init__(self) -> None:
        if not self.morphology_id or any(ord(c) < 0x20 for c in self.morphology_id):
            raise ValueError("morphology_id must be nonempty printable")
        # Recursively freeze
        object.__setattr__(self, "removed_literals", tuple(sorted(self.removed_literals)))
        object.__setattr__(self, "removed_clause_ids", tuple(sorted(self.removed_clause_ids)))
        object.__setattr__(self, "supporting_insights", tuple(self.supporting_insights))
        object.__setattr__(self, "morphed_bank", _deep_freeze(self.morphed_bank) if self.morphed_bank is not None else None)
        object.__setattr__(self, "resource_bounds", _freeze_mapping(dict(self.resource_bounds)))
        object.__setattr__(self, "source_pta_ids", tuple(sorted(self.source_pta_ids)))

    @staticmethod
    def content_address(parent_artifact_id: str | None, removed_literals: tuple[int, ...], removed_clause_ids: tuple[int, ...], morphed_bank: Any, supporting_insights: tuple, version: str = "pta.morphology.v1") -> str:
        import hashlib, json
        payload = {
            "parent_artifact_id": parent_artifact_id,
            "removed_literals": sorted(removed_literals),
            "removed_clause_ids": sorted(removed_clause_ids),
            "morphed_bank": _thaw_for_json(morphed_bank) if morphed_bank is not None else None,
            "supporting_insights": sorted([{"source_pta": i.source_pta, "kind": i.kind, "subject": i.subject, "evidence": _thaw_for_json(list(i.evidence))} for i in supporting_insights], key=lambda x: json.dumps(x, sort_keys=True)),
            "version": version,
        }
        return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
