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

    Reference oracle: deterministic, bounded, validates inputs strictly.
    Returns regions where |mean residual| > 0.5*std.
    """
    import math
    if not isinstance(field, str) or not field:
        raise ValueError("field must be nonempty string")
    if not isinstance(bins, int) or not 2 <= bins <= 32:
        raise ValueError("bins must be 2..32")
    if len(values) != len(targets) or len(values) != len(predictions):
        raise ValueError("values/targets/predictions length mismatch")
    for seq in (values, targets, predictions):
        for v in seq:
            if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v))):
                raise ValueError(f"non-finite or invalid value {v!r}")
    triples = [(float(v), float(t), float(p)) for v, t, p in zip(values, targets, predictions) if v is not None]
    if len(triples) < 4:
        return []
    triples.sort(key=lambda x: x[0])
    n = len(triples)
    regions: list[ResidualRegion] = []
    for b in range(bins):
        lo_idx = b * n // bins
        hi_idx = (b + 1) * n // bins
        chunk = triples[lo_idx:hi_idx]
        if not chunk:
            continue
        residuals = [t - p for _, t, p in chunk]
        mean_r = sum(residuals) / len(residuals)
        var = sum((r - mean_r) ** 2 for r in residuals) / len(residuals) if len(residuals) > 1 else 0.0
        std = var ** 0.5
        if abs(mean_r) > 0.5 * (std + 1e-9) and abs(mean_r) > 0.1:
            lo = chunk[0][0] if b > 0 else None
            hi = chunk[-1][0] if b < bins - 1 else None
            regions.append(ResidualRegion(field, lo, hi, mean_r, len(chunk)))
    return regions


def residual_to_proposal(region: ResidualRegion, *, pta_id: str = "regression:residual") -> PTAEscalationProposal:
    """Lower residual region to regression_clause proposal (bounded interval).

    Reference oracle: proposes interval descriptor; exact lowering must materialize
    via LiteralCatalog preview. No synthetic hash literal is invented.
    """
    params: dict[str, Any] = {"field": region.field}
    if region.lo is not None:
        params["lower"] = region.lo
    if region.hi is not None:
        params["upper"] = region.hi
    # Descriptor string for exact lowerer to preview; structure holds bounds without fake literal_id
    descriptor_str = f"numeric_between:{region.field}:{params}"
    # Validate bounds syntactically (not yet materialized)
    has_bounds = region.lo is not None or region.hi is not None
    return PTAEscalationProposal(
        proposal_id=f"{pta_id}:{region.field}:{region.lo}:{region.hi}",
        source_pta_ids=(pta_id,),
        supporting_insights=(PTAInsight(pta_id, "residual_region", region.field, (region.mean_residual, region.count)),),
        counterexamples_addressed=(),
        required_literals=(descriptor_str,) if has_bounds else (),
        native_target="regression_clause",
        structure={"field": region.field, "lower": region.lo, "upper": region.hi, "residual": region.mean_residual, "clause": [], "descriptor": params if has_bounds else None},
        resource_bounds={"literal_count": 1 if has_bounds else 0},
    )
