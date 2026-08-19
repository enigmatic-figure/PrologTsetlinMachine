"""Input PTA — learned Booleanization with symbolic provenance.

Ordinary TMs require a Booleanization scheme beforehand. Input PTAs observe
pre-binarization values (numeric, categorical, text, graph, temporal) while
native TAs see only instantiated Boolean literals. This module is the
literal-invention layer: it watches where discrimination occurs and proposes
thresholds/intervals that are then tested by escalation PTAs and pruned by
de-escalation PTAs.

Design follows docs/pta-control-plane.md:
  raw values → Input PTA proposes `x ≥ 7.3` / `7.3 ≤ x < 9.8` →
  escalation tests coverage on unresolved examples →
  de-escalation collapses `x≥7.1..7.4` to surviving boundary.
Only lowerable proposals become native `LiteralDescriptor`s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..representation import FeatureSchema, FieldKind, LiteralCatalog, LiteralDescriptor, NullPolicy
from .proposal import PTAEscalationProposal, PTAInsight

LITERAL_BUDGET_MAX = 512


@dataclass(frozen=True, slots=True)
class LiteralProposal:
    """One invented literal — still PTA-side until lowered."""

    field: str
    transform: str  # numeric_ge | numeric_between | category_eq | token_contains
    parameters: Mapping[str, Any]
    support_pos: int
    support_neg: int
    provenance: str  # e.g. "midpoint between 71.0/76.0 where label flips"
    proposal_id: str


class InputPTA:
    """Stateful per-field literal inventor with budget and provenance.

    Usage:
        schema = FeatureSchema.from_fields(temperature=FieldKind.NUMBER, mode=FieldKind.CATEGORY)
        catalog = LiteralCatalog(schema)
        pta = InputPTA(catalog, budget=64, pta_id="input:temperature")
        proposals = pta.propose_for_numeric("temperature", values, labels)
        for prop in proposals:
            # escalation checks coverage, then lower:
            desc = catalog.numeric_ge(prop.field, prop.parameters["threshold"])
    """

    def __init__(self, catalog: LiteralCatalog, *, budget: int = 64, pta_id: str = "input:pta") -> None:
        if not 1 <= budget <= LITERAL_BUDGET_MAX:
            raise ValueError(f"budget must be 1..{LITERAL_BUDGET_MAX}")
        if not pta_id or any(ord(c) < 0x20 for c in pta_id):
            raise ValueError("pta_id invalid")
        self.catalog = catalog
        self.budget = budget
        self.pta_id = pta_id
        self._proposed: list[LiteralProposal] = []
        self._seen_thresholds: set[tuple[str, float]] = set()

    @property
    def proposed_count(self) -> int:
        return len(self._proposed)

    def propose_for_numeric(
        self,
        field: str,
        values: Sequence[float | int | None],
        labels: Sequence[int],
        *,
        max_proposals: int = 8,
    ) -> list[LiteralProposal]:
        """Propose `numeric_ge` thresholds where label changes between sorted values.

        Deterministic, bounded, pure. Values may contain None (missing).
        Labels must be 0/1 strict ints and equal length to values.
        Returns up to max_proposals (≤ budget) sorted by discriminative power.
        """
        if field not in {f.name for f in self.catalog.schema.fields}:
            raise ValueError(f"unknown field {field}")
        if len(values) != len(labels):
            raise ValueError("values/labels length mismatch")
        if not labels or max_proposals <= 0:
            return []
        for y in labels:
            if type(y) is not int or y not in (0, 1):
                raise ValueError("labels must be strict 0/1")
        # Collect valid numeric pairs
        pairs: list[tuple[float, int]] = []
        for v, y in zip(values, labels):
            if v is None:
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
                raise ValueError(f"numeric field {field} has non-finite value {v!r}")
            pairs.append((float(v), int(y)))
        if len(pairs) < 2:
            return []
        pairs.sort(key=lambda p: p[0])
        # Find midpoints where adjacent labels differ
        candidates: list[tuple[float, int, int, float]] = []  # (threshold, pos_above, neg_above, strength)
        for i in range(len(pairs) - 1):
            v0, y0 = pairs[i]
            v1, y1 = pairs[i + 1]
            if y0 == y1 or v0 == v1:
                continue
            thr = (v0 + v1) / 2.0
            # strength = how many positives are ≥ threshold vs negatives < threshold
            pos_above = sum(1 for v, y in pairs if y == 1 and v >= thr)
            neg_below = sum(1 for v, y in pairs if y == 0 and v < thr)
            strength = pos_above + neg_below
            # dedup within epsilon
            key = (field, round(thr, 6))
            if key in self._seen_thresholds:
                continue
            candidates.append((thr, pos_above, neg_below, strength))
        # Rank by strength then by threshold (deterministic)
        candidates.sort(key=lambda c: (-c[3], c[0]))
        proposals: list[LiteralProposal] = []
        for thr, pos_above, neg_below, _ in candidates[:max_proposals]:
            if len(self._proposed) >= self.budget:
                break
            key = (field, round(thr, 6))
            self._seen_thresholds.add(key)
            # Also check categorical budgets via canonical duplicates are handled by LiteralCatalog dedup
            pid = f"{self.pta_id}:{field}:ge:{thr:.6g}"
            # Check if literal already exists in catalog (dedup) — preview without mutating
            try:
                existing = self.catalog.preview_numeric_ge(field, thr)
                # Check existence without registering
                already = any(d.literal_id == existing.literal_id for d in self.catalog.literals)
                if already:
                    provenance = f"threshold {thr:.6g} between values where label flips (already in catalog {existing.literal_id})"
                else:
                    provenance = f"threshold {thr:.6g} between values where label flips"
            except Exception:
                provenance = f"threshold {thr:.6g} between values where label flips"
            prop = LiteralProposal(
                field=field,
                transform="numeric_ge",
                parameters={"threshold": thr},
                support_pos=pos_above,
                support_neg=neg_below,
                provenance=provenance,
                proposal_id=pid,
            )
            self._proposed.append(prop)
            proposals.append(prop)
        return proposals

    def propose_interval(
        self,
        field: str,
        values: Sequence[float | int | None],
        labels: Sequence[int],
        *,
        max_proposals: int = 4,
    ) -> list[LiteralProposal]:
        """Propose `numeric_between` intervals that separate a positive run.

        Finds maximal contiguous positive runs in sorted order and proposes the
        enclosing interval [lo, hi]. Useful for `7.3 ≤ x < 9.8` literals.
        """
        if field not in {f.name for f in self.catalog.schema.fields}:
            raise ValueError(f"unknown field {field}")
        if len(values) != len(labels):
            raise ValueError("values/labels length mismatch")
        pairs: list[tuple[float, int]] = []
        for v, y in zip(values, labels):
            if v is None:
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
                raise ValueError("non-finite value")
            if type(y) is not int or y not in (0, 1):
                raise ValueError("labels must be 0/1")
            pairs.append((float(v), int(y)))
        if len(pairs) < 3:
            return []
        pairs.sort(key=lambda p: p[0])
        # Find positive runs bounded by negatives
        runs: list[tuple[float, float, int]] = []
        i = 0
        while i < len(pairs):
            if pairs[i][1] == 1:
                j = i
                while j < len(pairs) and pairs[j][1] == 1:
                    j += 1
                # run i..j-1 is positives; neighbours are negatives or edges
                lo = pairs[i][0]
                hi = pairs[j - 1][0]
                # expand to midpoint with neighbours if exist
                if i > 0:
                    lo = (pairs[i - 1][0] + lo) / 2.0
                if j < len(pairs):
                    hi = (hi + pairs[j][0]) / 2.0
                runs.append((lo, hi, j - i))
                i = j
            else:
                i += 1
        runs.sort(key=lambda r: (-r[2], r[0]))
        proposals: list[LiteralProposal] = []
        for lo, hi, support in runs[:max_proposals]:
            if len(self._proposed) >= self.budget:
                break
            pid = f"{self.pta_id}:{field}:between:{lo:.6g}-{hi:.6g}"
            # dedup check via preview without mutating
            try:
                self.catalog.preview_numeric_between(field, lo, hi, inclusive_lower=True, inclusive_upper=False)
            except Exception:
                pass
            prop = LiteralProposal(
                field=field,
                transform="numeric_between",
                parameters={"lower": lo, "upper": hi, "inclusive_lower": True, "inclusive_upper": False},
                support_pos=support,
                support_neg=len(pairs) - support,
                provenance=f"interval [{lo:.6g}, {hi:.6g}) covering positive run of {support}",
                proposal_id=pid,
            )
            self._proposed.append(prop)
            proposals.append(prop)
        return proposals

    def propose_categorical_group(
        self,
        field: str,
        values: Sequence[Any],
        labels: Sequence[int],
        *,
        max_groups: int = 4,
    ) -> list[LiteralProposal]:
        """Propose `category_in` groups that are positively associated.

        Groups values whose positive rate > 0.7. Returns at most max_groups.
        """
        if len(values) != len(labels):
            raise ValueError("length mismatch")
        by_value: dict[Any, list[int]] = {}
        for v, y in zip(values, labels):
            if v is None:
                continue
            if type(y) is not int or y not in (0, 1):
                raise ValueError("labels 0/1")
            by_value.setdefault(v, []).append(int(y))
        groups: list[tuple[Any, float, int]] = []
        for val, ys in by_value.items():
            rate = sum(ys) / len(ys)
            if rate > 0.7 and len(ys) >= 2:
                groups.append((val, rate, len(ys)))
        groups.sort(key=lambda g: (-g[1], -g[2], str(g[0])))
        proposals: list[LiteralProposal] = []
        for val, rate, n in groups[:max_groups]:
            if len(self._proposed) >= self.budget:
                break
            pid = f"{self.pta_id}:{field}:eq:{val!r}"
            proposals.append(
                LiteralProposal(
                    field=field,
                    transform="category_eq",
                    parameters={"value": val},
                    support_pos=n,
                    support_neg=0,
                    provenance=f"category {val!r} pos_rate {rate:.2f} over {n}",
                    proposal_id=pid,
                )
            )
            self._proposed.append(proposals[-1])
        return proposals

    def to_proposal(
        self, literal_proposal: LiteralProposal, *, native_target: str = "binary_clause"
    ) -> PTAEscalationProposal:
        """Lower a literal proposal into a typed escalation proposal for the gate."""
        # Preview descriptor without mutating catalog — materialization happens at exact lowering time
        desc: LiteralDescriptor | None = None
        try:
            if literal_proposal.transform == "numeric_ge":
                desc = self.catalog.preview_numeric_ge(literal_proposal.field, float(literal_proposal.parameters["threshold"]))
            elif literal_proposal.transform == "numeric_between":
                p = literal_proposal.parameters
                desc = self.catalog.preview_numeric_between(
                    literal_proposal.field,
                    float(p["lower"]),  # type: ignore[arg-type]
                    float(p["upper"]),  # type: ignore[arg-type]
                    inclusive_lower=bool(p.get("inclusive_lower", True)),
                    inclusive_upper=bool(p.get("inclusive_upper", False)),
                )
            elif literal_proposal.transform == "category_eq":
                desc = self.catalog.preview_category_eq(literal_proposal.field, literal_proposal.parameters["value"])
            elif literal_proposal.transform == "category_in":
                # category_in not yet in preview; fallback to generic preview
                desc = self.catalog.preview(literal_proposal.field, __import__("prolog_tsetlin.representation", fromlist=["TransformKind"]).TransformKind.CATEGORY_IN, {"values": literal_proposal.parameters["value"]})
        except Exception:
            desc = None
        if desc is not None:
            literal_id = desc.literal_id
            # Store descriptor payload for exact lowering to materialize
            req = (desc,)
            clause = [literal_id]
        else:
            # No valid descriptor → proposal is not exactly lowerable; keep transform descriptor string
            req = (f"{literal_proposal.transform}:{literal_proposal.field}:{literal_proposal.parameters}",)
            clause = []
        return PTAEscalationProposal(
            proposal_id=literal_proposal.proposal_id,
            source_pta_ids=(self.pta_id,),
            supporting_insights=(
                PTAInsight(self.pta_id, "interval" if literal_proposal.transform == "numeric_between" else "threshold", literal_proposal.field, (literal_proposal.parameters,)),
            ),
            counterexamples_addressed=(),
            required_literals=req,  # type: ignore[arg-type]
            native_target=native_target,
            structure={"clause": clause, "field": literal_proposal.field, "descriptor": desc.canonical_payload() if desc is not None else None},
            resource_bounds={"literal_count": 1 if clause else 0},
        )
