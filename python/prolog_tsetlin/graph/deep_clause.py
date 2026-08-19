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

    MAX_LITERAL_LEN = 256
    MAX_LITERALS = 4096

    def __post_init__(self) -> None:
        if type(self.layer) is not int or not 0 <= self.layer < 8:
            raise ValueError("layer must be int 0..7")
        if not isinstance(self.literals, frozenset) or not isinstance(self.negated, frozenset):
            raise ValueError("literals/negated must be frozenset")
        if len(self.literals) + len(self.negated) > self.MAX_LITERALS:
            raise ValueError("too many literals in component")
        for lit in self.literals | self.negated:
            if not isinstance(lit, str) or not lit or len(lit) > self.MAX_LITERAL_LEN or any(ord(c) < 0x20 for c in lit):
                raise ValueError(f"literal string invalid: {lit!r}")
        if self.literals & self.negated:
            raise ValueError("positive and negated literals must not overlap")

    def evaluate(self, present: frozenset[str]) -> bool:
        # All positive present and no negated present
        return self.literals.issubset(present) and self.negated.isdisjoint(present)

    def to_dict(self) -> dict[str, object]:
        return {"layer": self.layer, "literals": sorted(self.literals), "negated": sorted(self.negated)}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "DeepClauseComponent":
        if not isinstance(d, dict):
            raise ValueError("component must be dict")
        if "layer" not in d or "literals" not in d:
            raise ValueError("component missing required keys")
        layer = d["layer"]
        raw_lits = d["literals"]
        raw_neg = d.get("negated", [])
        if type(layer) is not int or isinstance(layer, bool):
            raise ValueError("layer must be strict int")
        if not 0 <= layer < 8:
            raise ValueError("layer out of bounds")
        if not isinstance(raw_lits, (list, tuple)):
            raise ValueError("literals must be list")
        if not isinstance(raw_neg, (list, tuple)):
            raise ValueError("negated must be list")
        lits = []
        for x in raw_lits:
            if not isinstance(x, str) or not x or len(x) > 256 or any(ord(c) < 0x20 for c in x):
                raise ValueError(f"literal invalid: {x!r}")
            lits.append(x)
        negs = []
        for x in raw_neg:
            if not isinstance(x, str) or not x or len(x) > 256 or any(ord(c) < 0x20 for c in x):
                raise ValueError(f"negated literal invalid: {x!r}")
            negs.append(x)
        lit_set = frozenset(lits)
        neg_set = frozenset(negs)
        if len(lit_set) != len(lits) or len(neg_set) != len(negs):
            raise ValueError("duplicate literals not allowed")
        if lit_set & neg_set:
            raise ValueError("positive and negated literals must not overlap")
        # Caller (DeepClause.from_dict) will verify layer == index; also check here strictly
        return cls(layer=layer, literals=lit_set, negated=neg_set)


@dataclass(frozen=True, slots=True)
class DeepClause:
    """Full deep clause Cj = ∧ components."""

    components: tuple[DeepClauseComponent, ...]
    # For interpretation, store clause index weight etc outside.

    def __post_init__(self) -> None:
        if not isinstance(self.components, tuple):
            raise ValueError("components must be tuple")
        if not 1 <= len(self.components) <= 8:
            raise ValueError("depth must be 1..8")
        for idx, comp in enumerate(self.components):
            if not isinstance(comp, DeepClauseComponent):
                raise ValueError("component must be DeepClauseComponent")
            if comp.layer != idx:
                raise ValueError(f"component layer {comp.layer} must equal index {idx}")

    def depth(self) -> int:
        return len(self.components)

    def to_dict(self) -> dict[str, object]:
        return {"components": [c.to_dict() for c in self.components]}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "DeepClause":
        if not isinstance(d, dict) or "components" not in d:
            raise ValueError("DeepClause dict missing components")
        raw = d["components"]
        if not isinstance(raw, (list, tuple)):
            raise ValueError("components must be list")
        if not 1 <= len(raw) <= 8:
            raise ValueError("depth must be 1..8")
        comps = tuple(DeepClauseComponent.from_dict(c) for c in raw)  # type: ignore[index]
        return cls(comps)
