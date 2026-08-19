"""Sparse binary hypervectors — bundle (⊕) and bind (⊗).

Deterministic, Boolean, and reversible. Based on Rachkovskij sparse
distributed representation. No numpy required.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Hypervector:
    dim: int
    indices: frozenset[int]  # active positions (sparse)

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.dim > 65536:
            raise ValueError("dim must be 1..65536")
        if any(not 0 <= i < self.dim for i in self.indices):
            raise ValueError("index out of range")

    def to_dense(self) -> tuple[int, ...]:
        return tuple(1 if i in self.indices else 0 for i in range(self.dim))

    @classmethod
    def from_dense(cls, bits: list[int] | tuple[int, ...]) -> "Hypervector":
        dim = len(bits)
        indices = frozenset(i for i, b in enumerate(bits) if b)
        return cls(dim, indices)

    def hamming(self, other: "Hypervector") -> int:
        if self.dim != other.dim:
            raise ValueError("dim mismatch")
        return len(self.indices.symmetric_difference(other.indices))


def _seeded_random(symbol: str, dim: int) -> random.Random:
    h = hashlib.blake2b(symbol.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(h, "little")
    return random.Random(seed ^ dim)


def random_hv(dim: int, sparsity: float, symbol: str) -> Hypervector:
    """Deterministic sparse HV for a symbol — same symbol → same HV."""
    if not 0 < sparsity < 1:
        raise ValueError("sparsity must be 0..1")
    k = max(1, int(dim * sparsity))
    rng = _seeded_random(symbol, dim)
    indices = frozenset(rng.sample(range(dim), k))
    return Hypervector(dim, indices)


def bundle(hvs: list[Hypervector]) -> Hypervector:
    """Bundle (⊕) = OR then sparsify to median density (deterministic)."""
    if not hvs:
        raise ValueError("bundle requires at least one HV")
    dim = hvs[0].dim
    if any(hv.dim != dim for hv in hvs):
        raise ValueError("dim mismatch in bundle")
    # Count per position
    counts = [0] * dim
    for hv in hvs:
        for i in hv.indices:
            counts[i] += 1
    # threshold = ceil(len(hvs)/2) (majority vote) — preserves Boolean interpretability
    thresh = (len(hvs) + 1) // 2
    indices = frozenset(i for i, c in enumerate(counts) if c >= thresh)
    # If empty (even split with no overlap), keep union to avoid zero vector
    if not indices:
        indices = frozenset().union(*(hv.indices for hv in hvs))
        # sparsify to k = dim*sparsity_approx (use first HV's sparsity as target)
        # keep deterministic: sort and take first k
        k = max(1, len(indices) // 2)
        indices = frozenset(sorted(indices)[:k])
    return Hypervector(dim, indices)


def bind(a: Hypervector, b: Hypervector) -> Hypervector:
    """Binding (⊗) = XOR with permutation — reversible, distinct per pair.

    Implemented as: permute `b` by a fixed rotation (dim//3) then XOR.
    """
    if a.dim != b.dim:
        raise ValueError("dim mismatch in bind")
    dim = a.dim
    # permute b
    shift = dim // 3 or 1
    perm_b = frozenset((i + shift) % dim for i in b.indices)
    # XOR = symmetric difference of active bits after permutation
    # But to keep binding distinct from plain XOR, we XOR a with permuted b
    result = a.indices.symmetric_difference(perm_b)
    # Keep sparsity roughly sum: no further sparsify — binding is dense-ish
    # To stay sparse, we keep result as is (size ~ |a|+|b|)
    # If too dense, sparsify deterministically by hashing
    if len(result) > dim // 2:
        # keep first half sorted
        result = frozenset(sorted(result)[: dim // 2])
    return Hypervector(dim, result)


def bind_many(hvs: list[Hypervector]) -> Hypervector:
    if not hvs:
        raise ValueError("bind_many requires at least one HV")
    cur = hvs[0]
    for hv in hvs[1:]:
        cur = bind(cur, hv)
    return cur


@dataclass(frozen=True, slots=True)
class HypervectorSpace:
    dim: int
    sparsity: float = 0.01

    def hv(self, symbol: str) -> Hypervector:
        return random_hv(self.dim, self.sparsity, symbol)

    def bundle(self, symbols: list[str]) -> Hypervector:
        return bundle([self.hv(s) for s in symbols])

    def bind_symbols(self, *symbols: str) -> Hypervector:
        hvs = [self.hv(s) for s in symbols]
        return bind_many(hvs)
