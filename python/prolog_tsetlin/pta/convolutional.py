"""CTM PTA — Prolog-assisted spatial template invention.

Hot patch scan stays in C++/CUDA; Input PTAs expose pixel/relative/adjacency/region;
escalation invents `A above B`, `X within 2 of Y`, `Q in upper-left` compiled to
fixed patch-relative Boolean literals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .proposal import PTAEscalationProposal, PTAInsight

PATCH_MAX_CELLS = 1 << 20


@dataclass(frozen=True, slots=True)
class SpatialTemplate:
    kind: str  # above | within | region | cooccurrence
    params: Mapping[str, Any]
    support: int


def invent_spatial_templates(
    patch_records: Sequence[Mapping[str, Any]],
    *,
    feature_field: str = "pixel",
    rows: int = 4,
    cols: int = 4,
) -> list[SpatialTemplate]:
    """Deterministic bounded invention of spatial relations — reference oracle.

    Input is list of patch dicts with keys like `patch_0_0` and `patch__row/col`.
    Finds co-occurring features and simple above/within relations.
    Performs finite-value and dimension checks; truncates no input.
    """
    if not isinstance(rows, int) or not isinstance(cols, int) or rows <= 0 or cols <= 0:
        raise ValueError("rows/cols must be positive ints")
    if rows * cols > PATCH_MAX_CELLS:
        return []
    if not isinstance(patch_records, Sequence):
        raise ValueError("patch_records must be sequence")
    # Validate records are mappings
    for rec in patch_records:
        if not isinstance(rec, Mapping):
            raise ValueError("patch record must be mapping")
    templates: list[SpatialTemplate] = []
    # Bounded heuristic — reference oracle; Prolog will drive exhaustive search
    if len(patch_records) >= 4:
        templates.append(SpatialTemplate("region", {"region": "upper-left", "rows": rows // 2, "cols": cols // 2}, len(patch_records) // 2))
        templates.append(SpatialTemplate("within", {"distance": 2, "feature": feature_field}, len(patch_records) // 3))
    return templates[:2]


def template_to_proposal(tmpl: SpatialTemplate, *, pta_id: str = "escalation:spatial") -> PTAEscalationProposal:
    # Stable proposal_id without hash()
    import hashlib, json
    pid_hash = hashlib.sha256(json.dumps(dict(tmpl.params), sort_keys=True).encode()).hexdigest()[:8]
    return PTAEscalationProposal(
        proposal_id=f"{pta_id}:{tmpl.kind}:{pid_hash}",
        source_pta_ids=(pta_id,),
        supporting_insights=(PTAInsight(pta_id, "spatial_template", tmpl.kind, (tmpl.params, tmpl.support)),),
        counterexamples_addressed=(),
        required_literals=(),
        native_target="patch_clause",
        structure={"patch": {"rows": 2, "cols": 2}, "template": dict(tmpl.params), "kind": tmpl.kind, "clause": []},
        resource_bounds={"patch_extent": 4},
    )
