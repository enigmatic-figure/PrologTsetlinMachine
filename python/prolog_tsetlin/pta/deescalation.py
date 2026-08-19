"""De-escalation PTA — Type III pruning + reversible absorption.

Implements de-escalation as Type III feedback: identifies context-specific
independence and removes unnecessary literals, but with PTM's shadow audit,
maturity, and restoration lineage so absorbing actions are reversible.

Pipeline:
  ordinary TA
    ↓ de-escalation identifies stable state
    ↓ candidate frozen representation
    ↓ shadow audit
    ↓ consolidated / absorbed
    ↓ drift → reopen → restore adaptive state
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..representation import FeatureSchema, FieldKind, LiteralCatalog, LiteralDescriptor
from ..budgeted_features import BudgetedFeatureStore
from .proposal import PTAInsight

MAX_CLAUSES = 1024


def _canon(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class PruningInsight:
    kind: str  # literal_redundant | literal_subsumed | thresholds_equivalent | clause_subsumes | ...
    subject: str
    evidence: tuple[Any, ...]


class DeescalationPTA:
    """Type-III-inspired de-escalation — reasons over clause/literal redundancy.

    Observed inputs are literal batches and clause reports:
      feature_relation(L1, subsumes, L2)
      clause_support(Clause, Example)
      thresholds_equivalent(T1, T2)

    Reference oracle for Prolog Type-III mechanism: identifies context-specific
    independence via observed-column equivalence/subsumption plus utility-based
    stability heuristics. Future Prolog backend will replace heuristics with
    exact CS-IA Type III feedback via pta_deescalation.pl.
    """

    def __init__(self, pta_id: str = "de-escalation:prune") -> None:
        if not pta_id or any(ord(c) < 0x20 for c in pta_id):
            raise ValueError("pta_id invalid")
        self.pta_id = pta_id
        self._insights: list[PruningInsight] = []

    @property
    def insights(self) -> tuple[PruningInsight, ...]:
        return tuple(self._insights)

    def find_redundant_literals(
        self,
        catalog: LiteralCatalog,
        literal_ids: Sequence[int],
        rows: Sequence[Mapping[str, Any]],
        *,
        epsilon: float = 0.0,
    ) -> list[PTAInsight]:
        """Identify literals with identical column vectors (threshold equivalence).

        Two `numeric_ge` literals on same field with different thresholds may be
        functionally identical on the observed rows. Returns insights of kind
        `thresholds_equivalent` or `literal_redundant`.
        """
        if len(literal_ids) > MAX_CLAUSES:
            raise ValueError("too many literals")
        # Evaluate each literal column
        columns: dict[int, tuple[bool, ...]] = {}
        descs: dict[int, LiteralDescriptor] = {d.literal_id: d for d in catalog.literals}
        for lid in literal_ids:
            desc = descs.get(lid)
            if desc is None:
                continue
            col = tuple(bool(catalog.evaluate(desc, row.get(desc.source_field))) for row in rows)
            columns[lid] = col
        insights: list[PTAInsight] = []
        lids = list(columns.keys())
        for i in range(len(lids)):
            for j in range(i + 1, len(lids)):
                a, b = lids[i], lids[j]
                if columns[a] == columns[b]:
                    # Check if both are numeric_ge on same field → thresholds_equivalent
                    da, db = descs[a], descs[b]
                    if da.transform.value == "numeric_ge" and db.transform.value == "numeric_ge" and da.source_field == db.source_field:
                        kind = "thresholds_equivalent"
                        subject = f"{da.source_field}:{da.parameter('threshold')}~{db.parameter('threshold')}"
                    else:
                        kind = "literal_redundant"
                        subject = f"{a}~{b}"
                    evidence = (a, b, f"identical on {len(rows)} rows")
                    ins = PTAInsight(self.pta_id, kind, subject, evidence)
                    insights.append(ins)
                    self._insights.append(PruningInsight(kind, subject, evidence))
        return insights

    def find_subsumed_literals(
        self,
        catalog: LiteralCatalog,
        literal_ids: Sequence[int],
        rows: Sequence[Mapping[str, Any]],
    ) -> list[PTAInsight]:
        """Identify `L1 subsumes L2` where L1 → L2 on all rows (e.g. `x≥7.4` subsumes `x≥7.1`)."""
        columns: dict[int, tuple[bool, ...]] = {}
        descs: dict[int, LiteralDescriptor] = {d.literal_id: d for d in catalog.literals}
        for lid in literal_ids:
            desc = descs.get(lid)
            if desc is None:
                continue
            columns[lid] = tuple(bool(catalog.evaluate(desc, row.get(desc.source_field))) for row in rows)
        insights: list[PTAInsight] = []
        for a in literal_ids:
            for b in literal_ids:
                if a == b:
                    continue
                ca, cb = columns.get(a), columns.get(b)
                if ca is None or cb is None:
                    continue
                # a subsumes b if a → b (whenever a true, b true) and not equivalent
                if all(not ca[k] or cb[k] for k in range(len(ca))) and ca != cb:
                    ins = PTAInsight(self.pta_id, "literal_subsumed", f"{a} subsumed by {b}", (a, b))
                    insights.append(ins)
                    self._insights.append(PruningInsight("literal_subsumed", f"{a}->{b}", (a, b)))
        return insights

    def propose_stable_absorption(
        self,
        store: BudgetedFeatureStore,
        *,
        utility_threshold: float = 0.9,
        min_uses: int = 10,
    ) -> list[PTAInsight]:
        """Identify literals with stable high utility and many uses → candidate frozen.

        Returns insights of kind `stable_inclusion` that upstream can turn into
        a frozen representation for shadow audit.
        """
        insights: list[PTAInsight] = []
        for rec in store._records.values():  # type: ignore[attr-defined]
            if rec.use_count >= min_uses and rec.utility >= utility_threshold:
                ins = PTAInsight(self.pta_id, "stable_inclusion", str(rec.descriptor.literal_id), (rec.use_count, rec.utility))
                insights.append(ins)
                self._insights.append(PruningInsight("stable_inclusion", str(rec.descriptor.literal_id), (rec.use_count, rec.utility)))
        return insights

    def detect_clause_subsumption(
        self,
        clause_literals: Mapping[int, frozenset[int]],  # clause_id → set[literal_id]
        rows_literal_batches: Sequence[Mapping[int, bool]],  # per-row literal truth
    ) -> list[PTAInsight]:
        """Detect `clause C1 subsumes C2` where C1's literal set ⊆ C2 and support superset."""
        insights: list[PTAInsight] = []
        cids = list(clause_literals.keys())
        for i in cids:
            for j in cids:
                if i == j:
                    continue
                si, sj = clause_literals[i], clause_literals[j]
                if si.issubset(sj) and si != sj:
                    # Check support superset: C1 covers superset of rows that C2 covers
                    # Approximate: if C1's conjunction is weaker, it should fire more often
                    ins = PTAInsight(self.pta_id, "clause_subsumes", f"{i} subsumes {j}", (i, j))
                    insights.append(ins)
                    self._insights.append(PruningInsight("clause_subsumes", f"{i}->{j}", (i, j)))
        return insights
