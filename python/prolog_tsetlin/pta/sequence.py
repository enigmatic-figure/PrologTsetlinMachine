"""Sequence PTA — DCG-style bounded sequence patterns.

Escalation discovers `A followed by B`, `A ... B within 3`, `A before B unless C`,
then lowers to fixed positional/temporal literals or asks Input PTA to materialize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .proposal import PTAEscalationProposal, PTAInsight


@dataclass(frozen=True, slots=True)
class SequencePattern:
    pattern: str  # e.g. "A followed by B", "A ... B within 3"
    window: int
    support: int


def discover_sequence_patterns(
    sequences: Sequence[Sequence[Any]],
    labels: Sequence[int],
    *,
    max_window: int = 3,
) -> list[SequencePattern]:
    """Bounded DCG-style discovery — reference oracle (Prolog DCG will replace)."""
    if len(sequences) != len(labels):
        raise ValueError("sequences/labels length mismatch")
    if not isinstance(max_window, int) or max_window <= 0:
        raise ValueError("max_window must be positive int")
    for y in labels:
        if type(y) is not int or y not in (0, 1):
            raise ValueError("labels must be strict 0/1 ints")
    pos_seqs = [seq for seq, y in zip(sequences, labels) if y == 1]
    if not pos_seqs:
        return []
    from collections import Counter

    pair_counts: Counter[tuple[Any, Any]] = Counter()
    for seq in pos_seqs:
        for i in range(len(seq) - 1):
            pair_counts[(seq[i], seq[i + 1])] += 1
    patterns: list[SequencePattern] = []
    for (a, b), cnt in pair_counts.most_common(2):
        if cnt >= 2:
            patterns.append(SequencePattern(f"{a} followed by {b}", window=1, support=cnt))
            patterns.append(SequencePattern(f"{a} ... {b} within {max_window}", window=max_window, support=cnt))
    return patterns[:2]


def pattern_to_proposal(pat: SequencePattern, *, pta_id: str = "escalation:sequence") -> PTAEscalationProposal:
    return PTAEscalationProposal(
        proposal_id=f"{pta_id}:{pat.pattern}",
        source_pta_ids=(pta_id,),
        supporting_insights=(PTAInsight(pta_id, "sequence_pattern", pat.pattern, (pat.window, pat.support)),),
        counterexamples_addressed=(),
        required_literals=(),
        native_target="logic_program",
        structure={"pattern": pat.pattern, "window": pat.window, "clause": []},
        resource_bounds={"literal_count": pat.window + 1},
    )
