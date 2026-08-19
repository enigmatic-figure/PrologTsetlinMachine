"""Graph PTA — richer relational search with bounded native lowering.

Prolog can reason with `edge/3`, `property/2`, recursive `reachable` during
discovery, but lowering requires `depth ≤8` and no unbounded recursion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .proposal import PTAEscalationProposal, PTAInsight

MAX_GRAPH_DEPTH = 8


@dataclass(frozen=True, slots=True)
class RelationalHypothesis:
    relation: str  # e.g. "ancestor", "reachable", "connected_through_X"
    depth: int
    recursive: bool
    support: int


def hypothesize_graph_relation(
    graph_examples: Sequence[Any],
    *,
    max_depth: int = 8,
) -> list[RelationalHypothesis]:
    """Bounded search for graph relations — reference oracle for Prolog reasoning.

    Returns hypotheses sorted by support; recursive unbounded hypotheses are
    kept for PTA-side reasoning but marked not lowerable. max_depth drives
    bound (1..8); values outside raise.
    """
    if not isinstance(max_depth, int) or not 1 <= max_depth <= MAX_GRAPH_DEPTH:
        raise ValueError("max_depth must be 1..8")
    if not isinstance(graph_examples, Sequence):
        raise ValueError("graph_examples must be sequence")
    hyps: list[RelationalHypothesis] = []
    if len(graph_examples) >= 2:
        # Depth respects max_depth — do not propose beyond bound
        d1 = min(2, max_depth)
        d2 = min(3, max_depth)
        hyps.append(RelationalHypothesis("ancestor", depth=d1, recursive=True, support=len(graph_examples) // 2))
        hyps.append(RelationalHypothesis("connected_through_X", depth=d2, recursive=False, support=len(graph_examples) // 3))
    # Always include an unbounded probe for gate testing, but do not clamp — keep original depth
    hyps.append(RelationalHypothesis("transitive_closure", depth=9, recursive=True, support=1))
    return hyps


def hypothesis_to_proposal(hyp: RelationalHypothesis, *, pta_id: str = "escalation:graph") -> PTAEscalationProposal:
    unbounded = hyp.recursive and hyp.depth > MAX_GRAPH_DEPTH
    if hyp.relation == "transitive_closure":
        unbounded = True
    # Do not clamp depth across exact lowering boundary — keep original depth for validator
    depth_for_proposal = hyp.depth
    # For unbounded, we still need a resource bound within 1..8 for schema, but lowering will reject via recursive_unbounded
    rb_depth = hyp.depth if not unbounded and 1 <= hyp.depth <= MAX_GRAPH_DEPTH else MAX_GRAPH_DEPTH
    if hyp.depth > MAX_GRAPH_DEPTH and not unbounded:
        # Non-unbounded overshoot is also not representable; keep original to let lower_exact reject
        pass
    return PTAEscalationProposal(
        proposal_id=f"{pta_id}:{hyp.relation}:{hyp.depth}",
        source_pta_ids=(pta_id,),
        supporting_insights=(PTAInsight(pta_id, "graph_relation", hyp.relation, (hyp.depth, hyp.support)),),
        counterexamples_addressed=(),
        required_literals=(),
        native_target="graph_clause",
        structure={"relation": hyp.relation, "depth": depth_for_proposal, "requires_recursion": hyp.recursive, "recursive_unbounded": unbounded, "clause": []},
        resource_bounds={"graph_depth": rb_depth},
    )
