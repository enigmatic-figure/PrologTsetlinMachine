"""Sparse lowering — de-escalation knowledge → SparseClauseBank/ClauseIndex.

De-escalation PTAs compute:
  unused literals, permanently excluded, duplicate/subsumed clauses,
  functionally equivalent clauses, zero-weight outputs, unreferenced features.

This module lowers that knowledge into the native sparse representation that
C++/CUDA executes without knowing Prolog helped derive it.

Native execution stays `clause banks • sparse clauses • patch evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..representation import LiteralCatalog
from .deescalation import DeescalationPTA


@dataclass(frozen=True, slots=True)
class SparseClause:
    clause_id: int
    literal_ids: tuple[int, ...]  # only active, non-redundant
    is_sparse: bool = True


@dataclass(frozen=True, slots=True)
class SparseClauseBank:
    """Sparse native bank — only surviving clause/literal IDs."""

    clauses: tuple[SparseClause, ...]
    literal_ids: tuple[int, ...]  # union of surviving literals
    clause_index: Mapping[int, int]  # original clause_id → position in bank (for runtime dispatch)

    @property
    def clause_count(self) -> int:
        return len(self.clauses)

    @property
    def literal_count(self) -> int:
        return len(self.literal_ids)

    def to_proposal_structure(self) -> dict[str, Any]:
        return {
            "sparse_clauses": [{"clause_id": c.clause_id, "literals": list(c.literal_ids)} for c in self.clauses],
            "literal_ids": list(self.literal_ids),
        }


def lower_to_sparse(
    catalog: LiteralCatalog,
    clause_literals: Mapping[int, frozenset[int]],
    rows: Sequence[Mapping[str, Any]],
    *,
    pta: DeescalationPTA | None = None,
) -> SparseClauseBank:
    """Lower via de-escalation insights to sparse bank.

    Deterministic, bounded. Removes:
      - duplicate clauses (identical literal sets)
      - subsumed clauses (Si ⊂ Sj)
      - literals that are thresholds_equivalent (keep smallest threshold)
      - literals never true on observed rows (unused)
    """
    pta = pta or DeescalationPTA()
    # 1. Find redundant literals (thresholds_equivalent → keep one per equivalence class)
    all_lids = sorted({lid for s in clause_literals.values() for lid in s})
    redundant = pta.find_redundant_literals(catalog, all_lids, rows)
    # Build equivalence classes for thresholds_equivalent
    to_remove: set[int] = set()
    from collections import defaultdict

    equiv: dict[int, set[int]] = defaultdict(set)
    for ins in redundant:
        if ins.kind == "thresholds_equivalent":
            a, b = ins.evidence[0], ins.evidence[1]  # type: ignore[index]
            equiv[a].add(b)
            equiv[b].add(a)
    # For each equivalence class, keep smallest literal_id (deterministic)
    visited: set[int] = set()
    for lid in all_lids:
        if lid in visited:
            continue
        cls = {lid} | equiv.get(lid, set())
        # expand closure
        stack = list(cls)
        closure = set(cls)
        while stack:
            cur = stack.pop()
            for nb in equiv.get(cur, ()):
                if nb not in closure:
                    closure.add(nb)
                    stack.append(nb)
        if len(closure) > 1:
            keep = min(closure)
            for other in closure:
                if other != keep:
                    to_remove.add(other)
                visited.add(other)
        visited.add(lid)

    # Also remove permanently unused literals (column all False)
    descs = {d.literal_id: d for d in catalog.literals}
    for lid in all_lids:
        if lid in to_remove:
            continue
        desc = descs.get(lid)
        if desc is None:
            continue
        col = [bool(catalog.evaluate(desc, r.get(desc.source_field))) for r in rows]
        if not any(col):
            to_remove.add(lid)

    # 2. Build sparse clause literal sets minus removed
    sparse_map: dict[int, frozenset[int]] = {}
    for cid, lids in clause_literals.items():
        sparse_map[cid] = frozenset(l for l in lids if l not in to_remove)

    # 3. Remove duplicate clauses (identical sparse sets)
    seen: dict[frozenset[int], int] = {}
    deduped: dict[int, frozenset[int]] = {}
    for cid in sorted(sparse_map):
        s = sparse_map[cid]
        if s not in seen:
            seen[s] = cid
            deduped[cid] = s
        # else duplicate → skip (subsumed)

    # 4. Remove subsumed clauses (Si ⊂ Sj) — keep minimal
    cids = sorted(deduped)
    subsumed: set[int] = set()
    for i in cids:
        for j in cids:
            if i == j or i in subsumed or j in subsumed:
                continue
            si, sj = deduped[i], deduped[j]
            if si and si.issubset(sj) and si != sj:
                # i subsumes j is not correct; j is more specific, keep i
                # Actually Si ⊂ Sj means Sj is subsumed by Si (Si weaker) → Sj redundant
                subsumed.add(j)

    final = {cid: s for cid, s in deduped.items() if cid not in subsumed}
    # Build bank
    clauses = tuple(SparseClause(cid, tuple(sorted(s))) for cid, s in sorted(final.items()))
    all_lits = tuple(sorted({lid for c in clauses for lid in c.literal_ids}))
    index = {c.clause_id: idx for idx, c in enumerate(clauses)}
    return SparseClauseBank(clauses, all_lits, index)
