"""Exact lowerability checker — YES iff proposal has an exact native representation.

No approximation. A Prolog hypothesis may use recursion, continuous values,
and richer relations during deliberation; it becomes a native candidate only if
its conclusion fits the target's bounded grammar (literal count, clause count,
graph depth, integer weight ranges, patch extent, etc.).
"""

from __future__ import annotations

from typing import Any

from .proposal import MAX_CLAUSES, MAX_GRAPH_DEPTH, PTAEscalationProposal

MAX_LITERALS_PER_CLAUSE = 64
MAX_WEIGHT_ABS = 1_000_000
PATCH_MAX_CELLS = 1 << 20


def lowerable(proposal: PTAEscalationProposal) -> tuple[bool, str]:
    """Check proposal against native target's exact grammar.

    Returns (True, "ok") or (False, reason). Pure and deterministic.
    """
    rb = proposal.resource_bounds
    struct = proposal.structure

    # Common: literal count
    if "literal_count" in rb and rb["literal_count"] > MAX_LITERALS_PER_CLAUSE:
        return False, "literal_count exceeds native bound"

    target = proposal.native_target
    if target in ("binary_clause", "shared_weighted_clause", "regression_clause"):
        # structure must contain clause literal IDs list, all ints
        clause = struct.get("clause") or struct.get("literals") or []
        if not isinstance(clause, list) or not clause:
            return False, "clause literals must be nonempty list"
        if len(clause) > MAX_LITERALS_PER_CLAUSE:
            return False, "clause exceeds literal ceiling"
        if any(not isinstance(lit, int) for lit in clause):
            return False, "clause literals must be integer IDs"
        if proposal.weights is not None:
            if any(not isinstance(w, int) or abs(w) > MAX_WEIGHT_ABS for w in proposal.weights):
                return False, "weight out of int32 bounded range"
        if "clause_count" in rb and rb["clause_count"] > MAX_CLAUSES:
            return False, "clause_count exceeds native bank"
        return True, "ok"

    if target == "graph_clause":
        depth = rb.get("graph_depth", struct.get("depth", 1))
        if not isinstance(depth, int) or not 1 <= depth <= MAX_GRAPH_DEPTH:
            return False, "graph_depth 1..8 required"
        # Patch: allow recursive Prolog during search, but require bounded unrolling
        if struct.get("requires_recursion") is True and struct.get("recursive_unbounded") is True:
            return False, "unbounded recursion not lowerable to graph_tm_v1"
        if struct.get("requires_recursion") is True and depth > MAX_GRAPH_DEPTH:
            return False, "discovered relation requires depth beyond graph_tm_v1"
        return True, "ok"

    if target == "patch_clause":
        # resource_bounds patch_extent may be int cells or dict handling; check both
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
        return True, "ok"

    if target in ("logic_program", "threshold", "composite_gate"):
        # Defer to existing lowerers (bounded_structure_search, threshold search)
        # which already enforce capacity and verify via oracle.
        return True, "ok (delegated to existing lowerer)"

    return False, f"unknown target {target}"


def check_example() -> PTAEscalationProposal:
    """Canonical example from docs/pta-control-plane.md:

    temperature 71–76 ∧ mode=manual ∧ previous=B  → 104∧105∧231∧388
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
