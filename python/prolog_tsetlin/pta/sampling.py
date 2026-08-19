"""FNS + multigranularity — PTA-authored masks and per-clause specificity.

Escalation/de-escalation knowledge maintains:
  confusable(cat, fox) / confusable_when(cat, fox, context)
  clause 17 only covers 2 positives (over-specific)
  clause 22 covers 6 failure regions (over-broad)

Compiles to native bitmasks/policies:
  negative_candidates[class] = bitmask
  clause_specificity[clause] = s
Native still runs FNS cheaply; Prolog determines the mask.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


def fns_mask_from_confusable(
    class_count: int,
    confusable_pairs: Sequence[tuple[int, int]],
) -> dict[int, int]:
    """Compile confusable(class, other) facts to per-class bitmask.

    Output is bounded `class_count ≤256` → mask fits in Python int (or 256-bit).
    Example: confusable(0,1), confusable(0,2) → mask[0]=0b110
    """
    if not 2 <= class_count <= 256:
        raise ValueError("class_count 2..256")
    masks: dict[int, int] = {c: 0 for c in range(class_count)}
    for a, b in confusable_pairs:
        if not (0 <= a < class_count and 0 <= b < class_count):
            raise ValueError("confusable pair out of range")
        if a == b:
            continue
        masks[a] |= 1 << b
        # Symmetric confusability is common but not required; caller may add directed.
    return masks


def fns_mask_from_counterexamples(
    examples: Sequence[Mapping[str, Any]],
    *,
    class_field: str = "label",
    pred_field: str = "pred",
) -> tuple[int, dict[int, int]]:
    """Derive confusable pairs from counterexamples where pred ≠ label.

    Returns (class_count, masks). Deterministic, bounded.
    """
    pairs: set[tuple[int, int]] = set()
    classes: set[int] = set()
    for ex in examples:
        true = ex.get(class_field)
        pred = ex.get(pred_field)
        if not isinstance(true, int) or not isinstance(pred, int):
            continue
        classes.add(true)
        classes.add(pred)
        if true != pred:
            pairs.add((true, pred))
    if not classes:
        return 0, {}
    class_count = max(classes) + 1
    # Clamp to 256
    class_count = min(256, max(2, class_count))
    return class_count, fns_mask_from_confusable(class_count, list(pairs))


def multigranularity_schedule(
    clause_stats: Mapping[int, Mapping[str, Any]],
    *,
    default_s: float = 3.9,
) -> dict[int, float]:
    """Derive per-clause specificity `s` from de-escalation stats.

    Input: clause_id → {"support": int positives covered, "failure_regions": int, "literal_count": int}
    Heuristic (bounded, deterministic):
      support ≤2 and literal_count >4 → over-specific → s=8.0
      failure_regions ≥6 → over-broad → s=2.1
      else default_s
    Prolog would author the same via `clause 17 only covers two positives` rules;
    here is the lowered numeric policy native feedback uses.
    """
    schedule: dict[int, float] = {}
    for cid, stats in clause_stats.items():
        support = int(stats.get("support", 0))
        failures = int(stats.get("failure_regions", 0))
        lits = int(stats.get("literal_count", 0))
        if support <= 2 and lits > 4:
            s = 8.0
        elif failures >= 6:
            s = 2.1
        else:
            s = float(default_s)
        # Clamp to sensible TM range
        s = max(1.0, min(20.0, s))
        schedule[cid] = s
    return schedule
