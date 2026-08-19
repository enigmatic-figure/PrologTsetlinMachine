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
    """Bounded search for graph relations — stub for Prolog recursive reasoning.

    Returns hypotheses sorted by support; recursive unbounded hypotheses are
    kept for PTA-side reasoning but marked not lowerable.
    """
    hyps: list[RelationalHypothesis] = []
    if len(graph_examples) >= 2:
        hyps.append(RelationalHypothesis("ancestor", depth=2, recursive=True, support=len(graph_examples) // 2))
        hyps.append(RelationalHypothesis("connected_through_X", depth=3, recursive=False, support=len(graph_examples) // 3))
    # Filter depth bound for lowering, but keep recursive unbounded for demo
    hyps.append(RelationalHypothesis("transitive_closure", depth=9, recursive=True, support=1))  # exceeds max, not lowerable
    return hyps


def hypothesis_to_proposal(hyp: RelationalHypothesis, *, pta_id: str = "escalation:graph") -> PTAEscalationProposal:
    unbounded = hyp.recursive and hyp.depth > MAX_GRAPH_DEPTH
    # For demo, treat depth 9 as unbounded
    if hyp.relation == "transitive_closure":
        unbounded = True
    return PTAEscalationProposal(
        proposal_id=f"{pta_id}:{hyp.relation}:{hyp.depth}",
        source_pta_ids=(pta_id,),
        supporting_insights=(PTAInsight(pta_id, "graph_relation", hyp.relation, (hyp.depth, hyp.support)),),
        counterexamples_addressed=(),
        required_literals=(),
        native_target="graph_clause",
        structure={"relation": hyp.relation, "depth": min(hyp.depth, MAX_GRAPH_DEPTH), "requires_recursion": hyp.recursive, "recursive_unbounded": unbounded, "clause": [hash(hyp.relation) & 0xFFFFFFFF]},
        resource_bounds={"graph_depth": min(hyp.depth, MAX_GRAPH_DEPTH) if not unbounded else MAX_GRAPH_DEPTH},
    )
