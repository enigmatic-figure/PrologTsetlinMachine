"""Regression PTA — error-directed symbolic feature engineering.

Native regression error (continuous summation + error-dependent feedback) stays
in C++/CUDA. Input PTAs see continuous target + raw features; escalation
proposes clauses for persistent residual; de-escalation consolidates regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .proposal import PTAInsight, PTAEscalationProposal


@dataclass(frozen=True, slots=True)
class ResidualRegion:
    field: str
    lo: float | None
    hi: float | None
    mean_residual: float
    count: int


def find_residual_regions(
    values: Sequence[float | int | None],
    targets: Sequence[float | int],
    predictions: Sequence[float | int],
    *,
    field: str,
    bins: int = 4,
) -> list[ResidualRegion]:
    """Partition field values into bins and find bins with persistent residual.

    Deterministic, bounded. Returns regions where |mean residual| > 0.5*std.
    """
    triples = [(float(v), float(t), float(p)) for v, t, p in zip(values, targets, predictions) if v is not None]
    if len(triples) < 4:
        return []
    triples.sort(key=lambda x: x[0])
    n = len(triples)
    # Simple quantile bins
    regions: list[ResidualRegion] = []
    for b in range(bins):
        lo_idx = b * n // bins
        hi_idx = (b + 1) * n // bins
        chunk = triples[lo_idx:hi_idx]
        if not chunk:
            continue
        residuals = [t - p for _, t, p in chunk]
        mean_r = sum(residuals) / len(residuals)
        # std
        var = sum((r - mean_r) ** 2 for r in residuals) / len(residuals) if len(residuals) > 1 else 0.0
        std = var ** 0.5
        if abs(mean_r) > 0.5 * (std + 1e-9) and abs(mean_r) > 0.1:
            lo = chunk[0][0] if b > 0 else None
            hi = chunk[-1][0] if b < bins - 1 else None
            regions.append(ResidualRegion(field, lo, hi, mean_r, len(chunk)))
    return regions


def residual_to_proposal(region: ResidualRegion, *, pta_id: str = "regression:residual") -> PTAEscalationProposal:
    """Lower residual region to regression_clause proposal (bounded interval)."""
    clause = []
    params: dict[str, Any] = {"field": region.field}
    if region.lo is not None:
        params["lower"] = region.lo
    if region.hi is not None:
        params["upper"] = region.hi
    # Native regression clause will be a numeric_between literal plus weight sign from residual
    return PTAEscalationProposal(
        proposal_id=f"{pta_id}:{region.field}:{region.lo}:{region.hi}",
        source_pta_ids=(pta_id,),
        supporting_insights=(PTAInsight(pta_id, "residual_region", region.field, (region.mean_residual, region.count)),),
        counterexamples_addressed=(),
        required_literals=(f"numeric_between:{region.field}:{params}",),
        native_target="regression_clause",
        structure={"field": region.field, "lower": region.lo, "upper": region.hi, "residual": region.mean_residual, "clause": [hash((region.field, region.lo, region.hi)) & 0xFFFFFFFF]},
        resource_bounds={"literal_count": 1},
    )
