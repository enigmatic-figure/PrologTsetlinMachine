"""Sparse lowering — exact representation vs model morphology.

De-escalation PTAs compute:
  unused literals, permanently excluded, duplicate/subsumed clauses,
  functionally equivalent clauses, zero-weight outputs, unreferenced features.

Two distinct operations:
  Exact representation lowering:
    dense included-literal mask → sparse list of SAME included literals
    all clauses/weights/polarities retained, structurally exact
  Model morphology:
    remove/change literals or clauses → new behavioral model
    requires PTA proposal → oracle/shadow validation → child artifact lineage
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..representation import LiteralCatalog
from .deescalation import DeescalationPTA
from .proposal import PTAEscalationProposal, PTAInsight, PTAMorphologyProposal


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


def to_sparse_exact(
    clause_literals: Mapping[int, frozenset[int]],
) -> SparseClauseBank:
    """Exact representation lowering: dense mask → sparse list, SAME semantics.

    Retains all clauses, all literal_ids, all weights/polarities.
    Structurally exact; no literals or clauses removed.
    """
    clauses = tuple(SparseClause(cid, tuple(sorted(s))) for cid, s in sorted(clause_literals.items()))
    all_lits = tuple(sorted({lid for c in clauses for lid in c.literal_ids}))
    index = {c.clause_id: idx for idx, c in enumerate(clauses)}
    return SparseClauseBank(clauses, all_lits, index)


def lower_to_sparse(
    catalog: LiteralCatalog,
    clause_literals: Mapping[int, frozenset[int]],
    rows: Sequence[Mapping[str, Any]],
    *,
    pta: DeescalationPTA | None = None,
) -> SparseClauseBank:
    """Deprecated: previously conflated exact lowering with morphology.

    This wrapper now calls exact lowering only. Behavior-changing morphology
    (removing unused, dedup, subsumption) must go through morphology.py
    and produce a PTA proposal for oracle/shadow validation.
    """
    return to_sparse_exact(clause_literals)


def propose_sparse_morphology(
    catalog: LiteralCatalog,
    clause_literals: Mapping[int, frozenset[int]],
    rows: Sequence[Mapping[str, Any]],
    *,
    pta: DeescalationPTA | None = None,
) -> tuple[SparseClauseBank, PTAMorphologyProposal | None]:
    """Model morphology: propose removing redundancy, requires new artifact.

    Uses DeescalationPTA to find redundancy. Returns (exact_bank,
    morphology_proposal) where morphology_proposal is a PTAMorphologyProposal
    (Class II lifecycle, not NativeTarget) requiring oracle/shadow validation.
    If no redundancy found, morphology_proposal is None.
    """
    pta = pta or DeescalationPTA()
    all_lids = sorted({lid for s in clause_literals.values() for lid in s})
    redundant = pta.find_redundant_literals(catalog, all_lids, rows)
    from collections import defaultdict

    equiv: dict[int, set[int]] = defaultdict(set)
    for ins in redundant:
        if ins.kind == "thresholds_equivalent":
            a, b = ins.evidence[0], ins.evidence[1]  # type: ignore[index]
            equiv[a].add(b)
            equiv[b].add(a)
    to_remove: set[int] = set()
    visited: set[int] = set()
    for lid in all_lids:
        if lid in visited:
            continue
        cls = {lid} | equiv.get(lid, set())
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

    sparse_map: dict[int, frozenset[int]] = {}
    for cid, lids in clause_literals.items():
        sparse_map[cid] = frozenset(l for l in lids if l not in to_remove)

    seen: dict[frozenset[int], int] = {}
    deduped: dict[int, frozenset[int]] = {}
    for cid in sorted(sparse_map):
        s = sparse_map[cid]
        if s not in seen:
            seen[s] = cid
            deduped[cid] = s

    cids = sorted(deduped)
    subsumed: set[int] = set()
    for i in cids:
        for j in cids:
            if i == j or i in subsumed or j in subsumed:
                continue
            si, sj = deduped[i], deduped[j]
            if si and si.issubset(sj) and si != sj:
                subsumed.add(j)

    final = {cid: s for cid, s in deduped.items() if cid not in subsumed}
    # If no change, no morphology proposal needed
    if not to_remove and len(final) == len(clause_literals):
        exact = to_sparse_exact(clause_literals)
        return exact, None

    clauses = tuple(SparseClause(cid, tuple(sorted(s))) for cid, s in sorted(final.items()))
    all_lits = tuple(sorted({lid for c in clauses for lid in c.literal_ids}))
    index = {c.clause_id: idx for idx, c in enumerate(clauses)}
    morphed = SparseClauseBank(clauses, all_lits, index)
    # Content-addressed morphology_id
    bank_dict = morphed.to_proposal_structure()
    # Determine removed clause IDs
    removed_cids = tuple(sorted(set(clause_literals.keys()) - set(final.keys())))
    content_id = PTAMorphologyProposal.content_address(None, tuple(sorted(to_remove)), removed_cids, bank_dict, tuple(redundant))
    proposal = PTAMorphologyProposal(
        morphology_id=content_id,
        parent_artifact_id=None,
        source_pta_ids=(pta.pta_id,),
        supporting_insights=tuple(redundant),
        removed_literals=tuple(sorted(to_remove)),
        removed_clause_ids=removed_cids,
        removed_clauses=len(clause_literals) - len(final),
        morphed_bank=bank_dict,
        resource_bounds={"literal_count": max(1, len(all_lits))},
    )
    exact = to_sparse_exact(clause_literals)
    return exact, proposal
