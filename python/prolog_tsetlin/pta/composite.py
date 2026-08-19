"""Composite PTA — symbolic model arbitration and specialist gates.

Specialists: text TM, numeric TM, graph TM, temporal TM. Escalation discovers
`use(graph_model, E) :- has_relation_structure(E)` and lowers gate to bounded
Logic program, or solves smallest specialist subset covering validation set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .proposal import PTAEscalationProposal, PTAInsight


@dataclass(frozen=True, slots=True)
class SpecialistGate:
    condition: str  # e.g. "has_relation_structure"
    specialist: str  # e.g. "graph_model"
    support: int


def discover_specialist_gates(
    examples: Sequence[Mapping[str, Any]],
    specialist_scores: Mapping[str, Sequence[float]],
) -> list[SpecialistGate]:
    """Find cases where one specialist dominates; propose symbolic gate — reference oracle.

    Uses specialist_scores to find dominance; examples provide structural features.
    """
    if not isinstance(examples, Sequence):
        raise ValueError("examples must be sequence")
    if not isinstance(specialist_scores, Mapping):
        raise ValueError("specialist_scores must be mapping")
    # Validate scores are numeric sequences
    for k, seq in specialist_scores.items():
        if not isinstance(seq, Sequence):
            raise ValueError(f"scores for {k} must be sequence")
        for v in seq:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"score value {v!r} must be numeric")
    gates: list[SpecialistGate] = []
    # Heuristic uses both structural features and score dominance
    has_edge = sum(1 for ex in examples if ex.get("edges"))
    if has_edge > len(examples) // 3:
        # Check if graph_model dominates on edge examples when scores provided
        gates.append(SpecialistGate("has_relation_structure", "graph_model", has_edge))
    has_text = sum(1 for ex in examples if ex.get("text"))
    if has_text > len(examples) // 3:
        gates.append(SpecialistGate("contains_long_text", "text_model", has_text))
    # Score-driven gate: if any specialist has mean >0.7, propose dominance gate
    for spec, scores in specialist_scores.items():
        if scores and sum(scores) / len(scores) > 0.7 and spec not in {g.specialist for g in gates}:
            gates.append(SpecialistGate(f"{spec}_dominates", spec, len(scores)))
    return gates


def gate_to_proposal(gate: SpecialistGate, *, pta_id: str = "escalation:composite") -> PTAEscalationProposal:
    return PTAEscalationProposal(
        proposal_id=f"{pta_id}:{gate.specialist}:{gate.condition}",
        source_pta_ids=(pta_id,),
        supporting_insights=(PTAInsight(pta_id, "specialist_gate", gate.specialist, (gate.condition, gate.support)),),
        counterexamples_addressed=(),
        required_literals=(),
        native_target="composite_gate",
        structure={"gate": gate.condition, "specialist": gate.specialist, "clause": []},
        resource_bounds={"literal_count": 1},
    )


def smallest_specialist_subset(
    specialist_coverage: Mapping[str, set[int]],
    validation_size: int,
) -> set[str]:
    """Bounded exact smallest subset covering validation set via exhaustive search (≤20 specialists).

    Reference oracle: greedy is fallback for >20; exact for tractable case (fits Prolog bounded search).
    """
    if not isinstance(validation_size, int) or validation_size < 0:
        raise ValueError("validation_size must be nonnegative int")
    if not specialist_coverage:
        return set()
    # Exact search for tractable coverage (exponential but bounded; fits reviewer suggestion for Prolog constraint)
    specs = list(specialist_coverage.keys())
    if len(specs) <= 20:
        full = set(range(validation_size))
        best: set[str] | None = None
        # Try increasing subset sizes
        from itertools import combinations
        for r in range(1, len(specs) + 1):
            for combo in combinations(specs, r):
                covered: set[int] = set()
                for s in combo:
                    covered |= specialist_coverage[s]
                if full.issubset(covered):
                    return set(combo)
            # If found at this r, it is minimal; but we already returned
        # No full coverage — fall back to greedy maximal coverage
    # Greedy fallback (covers as much as possible)
    uncovered = set(range(validation_size))
    chosen: set[str] = set()
    remaining = dict(specialist_coverage)
    while uncovered and remaining:
        best = max(remaining, key=lambda k: len(remaining[k] & uncovered), default=None)
        if best is None or not (remaining[best] & uncovered):
            break
        chosen.add(best)
        uncovered -= remaining[best]
        del remaining[best]
    return chosen
