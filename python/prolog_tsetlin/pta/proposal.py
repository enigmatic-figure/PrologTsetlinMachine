"""PTA proposal ontology — typed, deterministic, lowerability-checked.

This module implements the first Milestone 7 deliverable: the exact lowering
boundary between the PTA reasoning plane and the native Tsetlin plane. It is
deliberately small, dependency-free, and hostile-input tested. Prolog may be
arbitrarily more expressive during deliberation; only a proposal that passes
`lowerable()` becomes a native candidate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..representation import LiteralDescriptor

LOWERING_VERSION = "pta.lowering.v1"
MAX_LITERALS_PER_CLAUSE = 64
MAX_CLAUSES = 1024
MAX_GRAPH_DEPTH = 8

NativeTarget = str  # binary_clause | shared_weighted_clause | regression_clause | patch_clause | graph_clause | logic_program | threshold | composite_gate

_VALID_TARGETS = {
    "binary_clause",
    "shared_weighted_clause",
    "regression_clause",
    "patch_clause",
    "graph_clause",
    "logic_program",
    "threshold",
    "composite_gate",
}


def _canon(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _deep_canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _deep_canonicalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_deep_canonicalize(v) for v in sorted(value, key=lambda x: _canon(x)) if not isinstance(value, (list, tuple)) or True] if isinstance(value, (set, frozenset)) else [_deep_canonicalize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # For dataclasses / tuples containing mixed types, canon via json
    try:
        json.dumps(value, sort_keys=True)
        return value
    except Exception:
        return str(value)


def _freeze_mapping(m: Mapping[str, Any]) -> Mapping[str, Any]:
    from types import MappingProxyType
    # Deep copy and freeze
    frozen = {k: _deep_canonicalize(v) for k, v in m.items()}
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class PTAInsight:
    source_pta: str
    kind: str  # e.g. literal_redundant, clause_subsumes, thresholds_equivalent
    subject: str
    evidence: tuple[Any, ...] = ()


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
        if not self.proposal_id or any(ord(c) < 0x20 for c in self.proposal_id):
            raise ValueError("proposal_id must be nonempty printable")
        if not self.source_pta_ids or any(not s for s in self.source_pta_ids):
            raise ValueError("source_pta_ids must be nonempty")
        if self.native_target not in _VALID_TARGETS:
            raise ValueError(f"unknown native_target {self.native_target}")
        if self.lowering_version != LOWERING_VERSION:
            raise ValueError("unsupported lowering_version")
        # resource bounds must be positive ints within known ceilings
        for k, v in self.resource_bounds.items():
            if type(v) is not int or isinstance(v, bool) or v <= 0:
                raise ValueError(f"resource_bounds[{k}] must be positive int")
        if self.resource_bounds.get("clause_count", 1) > MAX_CLAUSES:
            raise ValueError("clause_count exceeds MAX_CLAUSES")
        if self.resource_bounds.get("graph_depth", 1) > MAX_GRAPH_DEPTH:
            raise ValueError("graph_depth exceeds MAX_GRAPH_DEPTH")
        # Deep-freeze mutable mappings to prevent post-construction mutation
        object.__setattr__(self, "structure", _freeze_mapping(dict(self.structure)))
        object.__setattr__(self, "resource_bounds", _freeze_mapping(dict(self.resource_bounds)))
        object.__setattr__(self, "validation_signature", _freeze_mapping(dict(self.validation_signature)))
        # Freeze required_literals strings are immutable; descriptors are frozen

    def _semantic_payload(self) -> dict[str, Any]:
        lits = []
        for lit in self.required_literals:
            if hasattr(lit, "literal_id"):
                lits.append(getattr(lit, "literal_id"))
            else:
                lits.append(str(lit))
        return {
            "native_target": self.native_target,
            "required_literals": sorted(lits, key=lambda x: str(x)),
            "structure": dict(self.structure),
            "weights": list(self.weights) if self.weights is not None else None,
            "output_assignments": [list(p) for p in self.output_assignments] if self.output_assignments is not None else None,
            "resource_bounds": dict(self.resource_bounds),
            "lowering_version": self.lowering_version,
        }

    def _provenance_payload(self) -> dict[str, Any]:
        d = self._semantic_payload()
        d.update({
            "proposal_id": self.proposal_id,
            "source_pta_ids": sorted(self.source_pta_ids),
            "supporting_insights": sorted([{"source_pta": i.source_pta, "kind": i.kind, "subject": i.subject, "evidence": list(i.evidence)} for i in self.supporting_insights], key=lambda x: _canon(x)),
            "counterexamples_addressed": sorted(self.counterexamples_addressed),
            "validation_signature": dict(self.validation_signature),
            "support_trace": sorted(self.support_trace),
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
        return {
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
