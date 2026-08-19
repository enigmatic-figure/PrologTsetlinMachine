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
    """Find cases where one specialist dominates; propose symbolic gate."""
    gates: list[SpecialistGate] = []
    # Heuristic: if graph specialist scores high when example has edges, propose gate
    has_edge = sum(1 for ex in examples if ex.get("edges"))
    if has_edge > len(examples) // 3:
        gates.append(SpecialistGate("has_relation_structure", "graph_model", has_edge))
    # Text specialist gate
    has_text = sum(1 for ex in examples if ex.get("text"))
    if has_text > len(examples) // 3:
        gates.append(SpecialistGate("contains_long_text", "text_model", has_text))
    return gates


def gate_to_proposal(gate: SpecialistGate, *, pta_id: str = "escalation:composite") -> PTAEscalationProposal:
    return PTAEscalationProposal(
        proposal_id=f"{pta_id}:{gate.specialist}:{gate.condition}",
        source_pta_ids=(pta_id,),
        supporting_insights=(PTAInsight(pta_id, "specialist_gate", gate.specialist, (gate.condition, gate.support)),),
        counterexamples_addressed=(),
        required_literals=(),
        native_target="composite_gate",
        structure={"gate": gate.condition, "specialist": gate.specialist, "clause": [hash(gate.condition) & 0xFFFFFFFF]},
        resource_bounds={"literal_count": 1},
    )


def smallest_specialist_subset(
    specialist_coverage: Mapping[str, set[int]],
    validation_size: int,
) -> set[str]:
    """Greedy smallest subset covering validation set — classic bounded set cover."""
    uncovered = set(range(validation_size))
    chosen: set[str] = set()
    while uncovered:
        best = max(specialist_coverage, key=lambda k: len(specialist_coverage[k] & uncovered), default=None)
        if best is None or not (specialist_coverage[best] & uncovered):
            break
        chosen.add(best)
        uncovered -= specialist_coverage[best]
    return chosen
