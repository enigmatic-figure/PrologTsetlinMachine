"""Escalation PTA — structure invention + CoTM weight allocation.

Escalation PTAs propose richer but still lowerable structures when the current
Boolean clause substrate fails. They may invent literals (via Input PTA), change
clauses, share clauses, assign integer weights, add graph-message conditions,
patch patterns, or exception rules. They query de-escalated knowledge to avoid
re-discovery. Native executes `TA states + int weight matrix` for CoTM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .proposal import PTAEscalationProposal, PTAInsight

MAX_CLAUSES = 1024
MAX_WEIGHT = 1_000_000


@dataclass(frozen=True, slots=True)
class WeightProposal:
    clause: int
    target_class: int
    weight: int
    rationale: str


class EscalationPTA:
    """Proposes new clauses, shared clauses, and CoTM weights.

    Example CoTM allocation:
        C0 C1 C2
      A  2  0 -1
      B  0  3  1  ← constraint-solved by escalation PTAs, observed by de-escalation.
    """

    def __init__(self, pta_id: str = "escalation:invent") -> None:
        if not pta_id or any(ord(c) < 0x20 for c in pta_id):
            raise ValueError("pta_id invalid")
        self.pta_id = pta_id
        self._proposals: list[PTAEscalationProposal] = []

    def propose_exception_rule(
        self,
        failing_examples: Sequence[Mapping[str, Any]],
        existing_clause_support: Mapping[int, frozenset[int]],
        *,
        field: str,
        threshold: float,
    ) -> PTAEscalationProposal:
        """Propose an exception clause for failing examples — reference oracle."""
        import math
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        return PTAEscalationProposal(
            proposal_id=f"{self.pta_id}:exception:{field}:{threshold:.6g}",
            source_pta_ids=(self.pta_id,),
            supporting_insights=(PTAInsight(self.pta_id, "exception", field, (threshold, len(failing_examples))),),
            counterexamples_addressed=tuple(range(len(failing_examples))),
            required_literals=(f"numeric_ge:{field}:{threshold}",),
            native_target="binary_clause",
            structure={"exception_for": list(existing_clause_support.keys())[:1], "clause": [], "field": field, "threshold": threshold, "descriptor": {"field": field, "threshold": threshold}},
            resource_bounds={"literal_count": 1, "clause_count": 1},
        )

    def allocate_cotm_weights(
        self,
        clause_count: int,
        class_count: int,
        clause_class_support: Mapping[tuple[int, int], float],  # (clause,class) -> F1 or accuracy
        *,
        budget_per_class: int = 4,
    ) -> list[WeightProposal]:
        """Allocate integer CoTM weights via greedy bounded assignment.

        This is the constraint problem `Clause×Output → Weight` with `|weight|≤1e6`.
        Native will execute the resulting matrix; Prolog (future CLP(FD)) will
        replace the greedy solver but the lowering contract is identical.
        """
        if not 1 <= clause_count <= MAX_CLAUSES:
            raise ValueError("clause_count out of bounds")
        if not 2 <= class_count <= 256:
            raise ValueError("class_count out of bounds")
        proposals: list[WeightProposal] = []
        for cls in range(class_count):
            # Top-k clauses for this class by support
            ranked = sorted(
                [(c, s) for (c, cl), s in clause_class_support.items() if cl == cls],
                key=lambda p: -p[1],
            )[:budget_per_class]
            for clause, support in ranked:
                w = max(-MAX_WEIGHT, min(MAX_WEIGHT, int(round(support * 4))))
                if w == 0:
                    w = 1 if support > 0 else -1
                proposals.append(WeightProposal(clause, cls, w, f"support {support:.3f}"))
        return proposals

    def weights_to_proposal(
        self,
        weights: Sequence[WeightProposal],
        *,
        proposal_id: str = "cotm-weights",
    ) -> PTAEscalationProposal:
        """Lower weight proposals into a `shared_weighted_clause` escalation proposal."""
        # Build output_assignments as (clause, class) → weight
        assignments = tuple((w.clause, w.target_class) for w in weights)
        weight_vals = tuple(w.weight for w in weights)
        # Validate via lowerability: use resource_bounds clause_count = distinct clauses
        distinct_clauses = len({w.clause for w in weights})
        return PTAEscalationProposal(
            proposal_id=f"{self.pta_id}:{proposal_id}",
            source_pta_ids=(self.pta_id,),
            supporting_insights=tuple(PTAInsight(self.pta_id, "weight", f"{w.clause}->{w.target_class}", (w.weight,)) for w in weights),
            counterexamples_addressed=(),
            required_literals=(),
            native_target="shared_weighted_clause",
            structure={"weights": {f"{w.clause}->{w.target_class}": w.weight for w in weights}},
            weights=weight_vals,
            output_assignments=assignments,
            resource_bounds={"clause_count": distinct_clauses},
        )

    def propose_graph_depth_increase(
        self,
        current_depth: int,
        failing_graph_examples: Sequence[Any],
        *,
        max_depth: int = 8,
    ) -> PTAEscalationProposal | None:
        """Propose increasing Graph TM depth if failures need more hops.

        If discovered relation requires unbounded recursion, proposal is not
        lowerable (checker will reject).
        """
        if current_depth >= max_depth:
            return None
        new_depth = min(max_depth, current_depth + 1)
        return PTAEscalationProposal(
            proposal_id=f"{self.pta_id}:graph_depth:{new_depth}",
            source_pta_ids=(self.pta_id,),
            supporting_insights=(PTAInsight(self.pta_id, "graph_depth", str(new_depth), (len(failing_graph_examples),)),),
            counterexamples_addressed=tuple(range(len(failing_graph_examples))),
            required_literals=(),
            native_target="graph_clause",
            structure={"depth": new_depth, "requires_recursion": True, "recursive_unbounded": False},
            resource_bounds={"graph_depth": new_depth},
        )
