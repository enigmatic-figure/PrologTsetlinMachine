"""Deterministic mapping from graph symbols (P/M/T) to hypervectors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .hypervector import Hypervector, HypervectorSpace


def _symbol_key(kind: str, identifier: object) -> str:
    return f"{kind}:{identifier}"


@dataclass(frozen=True, slots=True)
class HypervectorEncoder:
    dim: int = 4096
    sparsity: float = 0.01
    seed_salt: str = "ptm-graph-v1"

    def __post_init__(self) -> None:
        if not 256 <= self.dim <= 65536 or self.dim % 64 != 0:
            raise ValueError("dim must be 256..65536 and multiple of 64")
        if not 0.001 <= self.sparsity <= 0.5:
            raise ValueError("sparsity must be 0.001..0.5")

    @property
    def space(self) -> HypervectorSpace:
        return HypervectorSpace(self.dim, self.sparsity)

    def _hv_for(self, kind: str, ident: object) -> Hypervector:
        # include salt to namespace graph HVs away from other uses
        key = f"{self.seed_salt}|{kind}:{ident}"
        return self.space.hv(key)

    def property_hv(self, property_id: str) -> Hypervector:
        return self._hv_for("P", property_id)

    def message_hv(self, layer: int, clause_index: int) -> Hypervector:
        if not 0 <= layer < 8:
            raise ValueError("layer must be 0..7")
        return self._hv_for("M", f"{layer}:{clause_index}")

    def edge_type_hv(self, edge_type: object) -> Hypervector:
        from .types import canonical_edge_type

        return self._hv_for("T", canonical_edge_type(edge_type))

    def node_hv(self, property_ids: frozenset[str]) -> Hypervector:
        if not property_ids:
            # empty node → zero-like HV (single reserved symbol)
            return self._hv_for("P", "__empty__")
        hvs = [self.property_hv(pid) for pid in sorted(property_ids)]
        # bundle per paper: ⊕ over properties
        from .hypervector import bundle
        return bundle(hvs)

    def inbox_hv(self, message_ids: set[tuple[int,int]]) -> Hypervector | None:
        """Bundle of messages present in inbox at a layer (None if empty)."""
        if not message_ids:
            return None
        hvs = [self.message_hv(layer, clause) for layer, clause in sorted(message_ids)]
        from .hypervector import bundle
        return bundle(hvs)

    def bound_message(self, layer: int, clause: int, edge_type: object) -> Hypervector:
        """M^i_j ⊗ t  (bind message to edge type)."""
        from .hypervector import bind
        return bind(self.message_hv(layer, clause), self.edge_type_hv(edge_type))
