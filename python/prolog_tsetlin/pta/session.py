"""PTAReasoningSession — shared knowledge base boundary for the collective.

Python owns validation and safe fact serialization; GNU Prolog consumes those
facts and lets Input, De-escalation and Escalation PTAs communicate through
the shared ontology. Output remains the narrow typed proposal protocol.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .proposal import PTAEscalationProposal, PTAInsight


@dataclass
class PTAReasoningSession:
    """Bounded shared reasoning session.

    Attributes:
      generation / dataset_id / bounds
      observations: (pta, example, field, raw_value)
      counterexamples: (model, example, expected, actual)
      literal_support: literal_id -> (pos, neg)
      clause_support / clause_conflict
      consolidated insights
      current native mappings
      pending proposals
    """

    dataset_id: str
    generation: int = 0
    max_observations: int = 1024
    max_insights: int = 256
    observations: list[tuple[str, int, str, Any]] = field(default_factory=list)
    counterexamples: list[tuple[str, int, int, int]] = field(default_factory=list)
    insights: list[PTAInsight] = field(default_factory=list)
    proposals: list[PTAEscalationProposal] = field(default_factory=list)

    def add_observation(self, pta: str, example: int, field: str, raw_value: Any) -> None:
        if len(self.observations) >= self.max_observations:
            raise ValueError("observation budget exceeded")
        self.observations.append((pta, example, field, raw_value))

    def add_counterexample(self, model: str, example: int, expected: int, actual: int) -> None:
        self.counterexamples.append((model, example, expected, actual))

    def add_insight(self, insight: PTAInsight) -> None:
        if len(self.insights) >= self.max_insights:
            raise ValueError("insight budget exceeded")
        self.insights.append(insight)

    def add_proposal(self, proposal: PTAEscalationProposal) -> None:
        self.proposals.append(proposal)

    def to_prolog_facts(self) -> str:
        """Serialize safe facts for GNU Prolog consult (bounded, validated)."""
        lines: list[str] = ["% PTAReasoningSession facts — auto-generated, bounded"]
        for pta, ex, fld, val in self.observations:
            # Only allow finite numbers, strings, bools as raw values
            if isinstance(val, float) and not __import__("math").isfinite(val):
                continue
            # Escape single quotes in strings
            val_repr = repr(val) if isinstance(val, str) else str(val) if isinstance(val, (int, float, bool)) else repr(str(val))
            lines.append(f"observation('{pta}',{ex},'{fld}',{val_repr}).")
        for ins in self.insights:
            lines.append(f"insight('{ins.source_pta}','{ins.kind}','{ins.subject}',{list(ins.evidence)!r}).")
        for model, ex, exp, act in self.counterexamples:
            lines.append(f"counterexample('{model}',{ex},{exp},{act}).")
        for prop in self.proposals:
            lines.append(f"proposal('{prop.proposal_id}','{prop.native_target}',{prop.proposal_hash()!r}).")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "generation": self.generation,
            "observations": list(self.observations),
            "counterexamples": list(self.counterexamples),
            "insights": [{"source_pta": i.source_pta, "kind": i.kind, "subject": i.subject, "evidence": list(i.evidence)} for i in self.insights],
            "proposals": [p.to_dict() for p in self.proposals],
        }
