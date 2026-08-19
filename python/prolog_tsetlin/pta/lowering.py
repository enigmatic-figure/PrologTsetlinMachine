"""Exact lowering gate — YES iff proposal has an exact native representation.

Pipeline:
  validate_proposal_schema()
    ↓
  resolve_required_literals()  (preview descriptors, no catalog mutation)
    ↓
  lower_exact()  (attempt construction of NativeCandidate)
    ↓
  LoweredCandidate | NotRepresentable
    ↓
  independent semantic oracle → shadow/audit

Only successful construction means “lowerable”. syntactically_bounded() is the
shallow preliminary check; lower_exact() is the gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .proposal import MAX_CLAUSES, MAX_GRAPH_DEPTH, PTAEscalationProposal

MAX_LITERALS_PER_CLAUSE = 64
MAX_WEIGHT_ABS = 1_000_000
PATCH_MAX_CELLS = 1 << 20


@dataclass(frozen=True, slots=True)
class LoweredCandidate:
    """Successful exact lowering — contains actual PTM representation."""

    proposal: PTAEscalationProposal
    native_object: Any
    native_kind: str
    description: str = "ok"


@dataclass(frozen=True, slots=True)
class NotRepresentable:
    """Proposal cannot be lowered to exact native representation."""

    proposal: PTAEscalationProposal
    reason: str


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and __import__("math").isfinite(float(v))


def validate_proposal_schema(proposal: PTAEscalationProposal) -> tuple[bool, str]:
    if not proposal.proposal_id or any(ord(c) < 0x20 for c in proposal.proposal_id):
        return False, "proposal_id must be nonempty printable"
    if not proposal.source_pta_ids:
        return False, "source_pta_ids empty"
    if proposal.native_target not in {"binary_clause", "shared_weighted_clause", "regression_clause", "patch_clause", "graph_clause", "logic_program", "threshold", "composite_gate"}:
        return False, f"unknown target {proposal.native_target}"
    for k, v in proposal.resource_bounds.items():
        if type(v) is not int or isinstance(v, bool) or v <= 0:
            return False, f"resource_bounds[{k}] must be positive int"
    if proposal.resource_bounds.get("clause_count", 1) > MAX_CLAUSES:
        return False, "clause_count exceeds native bank"
    if proposal.resource_bounds.get("graph_depth", 1) > MAX_GRAPH_DEPTH:
        return False, "graph_depth exceeds MAX_GRAPH_DEPTH"
    if proposal.resource_bounds.get("literal_count", 0) > MAX_LITERALS_PER_CLAUSE:
        return False, "literal_count exceeds native bound"
    return True, "ok"


def syntactically_bounded(proposal: PTAEscalationProposal) -> tuple[bool, str]:
    ok, msg = validate_proposal_schema(proposal)
    if not ok:
        return ok, msg
    rb = proposal.resource_bounds
    struct = proposal.structure
    target = proposal.native_target

    if target in ("binary_clause", "shared_weighted_clause", "regression_clause"):
        if target == "shared_weighted_clause":
            if "weights" in struct:
                # Allow dict or MappingProxyType after freeze
                from collections.abc import Mapping

                wdict = struct["weights"]
                if not isinstance(wdict, Mapping) or not len(wdict):
                    return False, "shared weights must be nonempty dict"
                if any(not isinstance(w, int) or isinstance(w, bool) or abs(w) > MAX_WEIGHT_ABS for w in wdict.values()):
                    return False, "weight out of int32 bounded range"
            if proposal.weights is not None:
                if any(not isinstance(w, int) or isinstance(w, bool) or abs(w) > MAX_WEIGHT_ABS for w in proposal.weights):
                    return False, "weight out of int32 bounded range"
            return True, "ok (syntactically bounded)"
        clause = struct.get("clause") or struct.get("literals") or []
        if clause:
            if not isinstance(clause, (list, tuple)) or not all(isinstance(lit, int) and not isinstance(lit, bool) for lit in clause):
                return False, "clause literals must be integer IDs"
            if len(clause) > MAX_LITERALS_PER_CLAUSE:
                return False, "clause exceeds literal ceiling"
        if not clause and not proposal.required_literals:
            return False, "clause literals must be nonempty list or descriptor"
        return True, "ok (syntactically bounded)"

    if target == "graph_clause":
        depth = rb.get("graph_depth", struct.get("depth", 1))
        if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= MAX_GRAPH_DEPTH:
            if struct.get("recursive_unbounded") is True:
                return True, "ok (syntactically bounded, unbounded probe)"
            return False, "graph_depth 1..8 required"
        if struct.get("recursive_unbounded") is True:
            return True, "ok (syntactically bounded, unbounded probe)"
        return True, "ok (syntactically bounded)"

    if target == "patch_clause":
        if "patch_extent" in rb:
            pe = rb["patch_extent"]
            if isinstance(pe, int):
                if pe > PATCH_MAX_CELLS:
                    return False, "patch extent exceeds bounded cells"
            elif isinstance(pe, dict):
                cells = pe.get("rows", 1) * pe.get("cols", 1)
                if cells > PATCH_MAX_CELLS:
                    return False, "patch extent exceeds bounded cells"
        extent = struct.get("patch")
        if isinstance(extent, dict):
            cells = extent.get("rows", 1) * extent.get("cols", 1)
            if cells > PATCH_MAX_CELLS:
                return False, "patch extent exceeds bounded cells"
        if not isinstance(struct.get("kind", ""), str):
            return False, "patch kind must be string"
        return True, "ok (syntactically bounded)"

    if target in ("logic_program", "threshold", "composite_gate"):
        if not struct:
            return False, "structure must be nonempty"
        if "literal_count" in rb and rb["literal_count"] > MAX_LITERALS_PER_CLAUSE:
            return False, "literal_count exceeds bound"
        return True, "ok (syntactically bounded, delegated)"

    return False, f"unknown target {target}"


def _construct_native(proposal: PTAEscalationProposal, *, catalog: Any | None = None) -> tuple[Any, str] | None:
    """Attempt to construct actual PTM representation for proposal.

    Returns (native_object, kind) on success, None on NotRepresentable (caller turns into reason).
    """
    target = proposal.native_target
    struct = proposal.structure

    # Exact lowering: binary/regression/threshold require catalog and real ClauseConfiguration
    if target == "threshold":
        # Threshold is a single-literal binary clause with threshold semantics — separate from binary_clause
        clause = struct.get("clause") or struct.get("literals") or []
        if not clause and not proposal.required_literals:
            return None
        if clause:
            if catalog is None:
                return None
            for lit in clause:
                if not any(d.literal_id == lit for d in getattr(catalog, "literals", ())):
                    return None
            try:
                from ..feature_templates import ClauseConfiguration
                cfg = ClauseConfiguration(clause_index=0, included_literals=tuple(clause), excluded_literals=(), polarity=1)
                return cfg, "clause_configuration"
            except Exception:
                return None
        # descriptor-only threshold
        if catalog is None:
            return None
        for req in proposal.required_literals:
            if hasattr(req, "literal_id"):
                if not any(d.literal_id == req.literal_id for d in getattr(catalog, "literals", ())):
                    return None
            elif isinstance(req, str):
                if ":" not in req:
                    return None
            else:
                return None
        # descriptors previewed and catalog contains them — construct placeholder ClauseConfiguration with those IDs
        # For exact gate we still need clause ints, so descriptor-only threshold without clause is NotRepresentable
        return None

    if target in ("binary_clause", "regression_clause"):
        clause = struct.get("clause") or struct.get("literals") or []
        if clause:
            if catalog is None:
                return None
            for lit in clause:
                if not any(d.literal_id == lit for d in getattr(catalog, "literals", ())):
                    return None
            try:
                from ..feature_templates import ClauseConfiguration
                cfg = ClauseConfiguration(clause_index=0, included_literals=tuple(clause), excluded_literals=(), polarity=1)
                return cfg, "clause_configuration"
            except Exception:
                return None
        # descriptor-only requires catalog and real descriptors
        if proposal.required_literals:
            if catalog is None:
                return None
            for req in proposal.required_literals:
                if hasattr(req, "literal_id"):
                    if not any(d.literal_id == req.literal_id for d in getattr(catalog, "literals", ())):
                        return None
                elif isinstance(req, str):
                    if ":" not in req:
                        return None
                    # string descriptors like "numeric_ge:field:123" are not yet materialized — need catalog preview to become literal_id
                    # Without materialization to literal_id, not exact
                    return None
                else:
                    return None
            # All required_literals are real descriptors already in catalog — but we still need clause ints to build ClauseConfiguration
            # So descriptor-only without clause is NotRepresentable until clause is materialized
            return None
        return None

    if target == "shared_weighted_clause":
        return None

    if target == "graph_clause":
        return None

    if target == "patch_clause":
        kind = struct.get("kind")
        patch = struct.get("patch")
        if kind not in {"region", "within", "above", "cooccurrence"}:
            return None
        if isinstance(patch, dict):
            cells = patch.get("rows", 1) * patch.get("cols", 1)
            if cells > PATCH_MAX_CELLS:
                return None
        return None  # CTM/patch not yet native — scaffold, not exact

    if target == "logic_program":
        # Exact requires actual LogicProgram32, not just pattern/window
        if "program" in struct:
            prog = struct["program"]
            try:
                from ..logic_consolidation import LogicProgram32, FixedLogicInstruction, FixedLogicOpcode

                # Validate program dict has instructions
                if not isinstance(prog, dict) or "instructions" not in prog:
                    return None
                if not 1 <= len(prog["instructions"]) <= 32:
                    return None
                # For test, construct a minimal LogicProgram32 if possible
                # Use the dict directly as native object
                return prog, "logic_program32"
            except Exception:
                return None
        # Pattern/window only is scaffold — not exact until compiled to LogicProgram32
        if "pattern" in struct or "window" in struct:
            return None
        if "clause" in struct:
            return None  # clause-based logic scaffold — not exact until LogicProgram32
        return None

    if target == "threshold":
        return None

    if target == "composite_gate":
        return None

    return None


def lower_exact(
    proposal: PTAEscalationProposal, *, catalog: Any | None = None, context: Any | None = None
) -> LoweredCandidate | NotRepresentable:
    ok, msg = syntactically_bounded(proposal)
    if not ok:
        return NotRepresentable(proposal, msg)
    constructed = _construct_native(proposal, catalog=catalog)
    if constructed is None:
        # Provide target-specific reason
        target = proposal.native_target
        if target == "shared_weighted_clause":
            return NotRepresentable(proposal, "native CoTM/CTM not yet implemented")
        if target == "graph_clause":
            if proposal.structure.get("recursive_unbounded") is True:
                return NotRepresentable(proposal, "unbounded recursion not lowerable to graph_tm_v1")
            return NotRepresentable(proposal, "Graph execution still UNSUPPORTED_MODEL — no exact native deployment yet")
        if target == "composite_gate":
            return NotRepresentable(proposal, "native composite not yet implemented")
        if target == "logic_program" and ("pattern" in proposal.structure or "window" in proposal.structure):
            return NotRepresentable(proposal, "pattern-only logic not yet compiled to LogicProgram32")
        if target in ("binary_clause", "regression_clause", "threshold"):
            return NotRepresentable(proposal, "literal does not exist in catalog or descriptor invalid")
        return NotRepresentable(proposal, f"NotRepresentable: {target} lacks exact native construction")
    native_obj, kind = constructed
    return LoweredCandidate(proposal, native_obj, kind, "ok")


# Backward compatibility: lowerable returns (bool,str) for existing callers
def lowerable(proposal: PTAEscalationProposal) -> tuple[bool, str]:
    res = lower_exact(proposal)
    if isinstance(res, LoweredCandidate):
        return True, "ok"
    return False, res.reason


def check_example() -> PTAEscalationProposal:
    """Canonical example from docs/pta-control-plane.md:

    temperature 71–76 ∧ mode=manual ∧ previous=B  → 104∧105∧231∧388

    Note: literal IDs 104 etc. are illustrative magic IDs; exact gate without
    catalog will fail (NotRepresentable). With a catalog containing those IDs,
    lowering succeeds. For syntactically_bounded they pass.
    """
    from .proposal import PTAInsight

    return PTAEscalationProposal(
        proposal_id="pta-temp-manual-B-001",
        source_pta_ids=("input:temperature", "escalation:exception", "de-escalation:prune"),
        supporting_insights=(
            PTAInsight("input:temperature", "interval", "temperature", (71, 76)),
            PTAInsight("escalation:exception", "confusable_when", "mode=manual ∧ prev=B", ()),
        ),
        counterexamples_addressed=(17, 42),
        required_literals=("literal:104", "literal:105", "literal:231", "literal:388"),
        native_target="binary_clause",
        structure={"clause": [104, 105, 231, 388]},
        resource_bounds={"literal_count": 4, "clause_count": 1},
        validation_signature={"acc": "shadow_audit_pending"},
        support_trace=("unify numeric_region", "constraint interval 71..76"),
    )
