"""Deep clauses Cj = ∧_i C^i_j  with message inboxes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DeepClauseComponent:
    """Conjunction over literals for one layer.

    - layer 0: literals over property IDs (e.g., frozenset containing 'P:prop:...')
    - layer >0: literals over message symbols (layer-1 messages bound with edge types)
    For simplicity, literals are string ids of symbols expected present.
    Negation is represented by `negated=True` meaning literal must be ABSENT.
    """

    layer: int
    literals: frozenset[str]  # positive literals (must be present)
    negated: frozenset[str] = field(default_factory=frozenset)  # must be absent
    # For layer>0, literals refer to messages "M:layer:clause" optionally bound to edge type
    # We store raw symbol strings; edge-type binding is handled by inbox encoding outside.

    def evaluate(self, present: frozenset[str]) -> bool:
        # All positive present and no negated present
        return self.literals.issubset(present) and self.negated.isdisjoint(present)

    def to_dict(self) -> dict[str, object]:
        return {"layer": self.layer, "literals": sorted(self.literals), "negated": sorted(self.negated)}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "DeepClauseComponent":
        return cls(
            layer=int(d["layer"]),  # type: ignore[index]
            literals=frozenset(str(x) for x in d["literals"]),  # type: ignore[index]
            negated=frozenset(str(x) for x in d.get("negated", [])),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DeepClause:
    """Full deep clause Cj = ∧ components."""

    components: tuple[DeepClauseComponent, ...]
    # For interpretation, store clause index weight etc outside.

    def depth(self) -> int:
        return len(self.components)

    def to_dict(self) -> dict[str, object]:
        return {"components": [c.to_dict() for c in self.components]}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "DeepClause":
        comps = tuple(DeepClauseComponent.from_dict(c) for c in d["components"])  # type: ignore[index]
        return cls(comps)
