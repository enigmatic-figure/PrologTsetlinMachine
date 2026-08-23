from __future__ import annotations
from typing import Any

from ....services.diagnostics import RunDiagnostics


def clause_rows(diagnostics: RunDiagnostics) -> list[dict[str, Any]]:
    rows = [
        {
            'id': clause.clause_id,
            'polarity': '+' if clause.polarity > 0 else '-',
            'support': clause.support_count,
            'sample_count': len(diagnostics.examples),
            'support_rate': clause.support_fraction,
            'vote_sum': clause.signed_vote_sum,
            'aligned': clause.aligned_count,
            'opposed': clause.opposed_count,
            'correct_support': clause.correct_activation_count,
            'error_support': clause.incorrect_activation_count,
            'unique_support': clause.unique_support_count,
            'lits': len(clause.included_literals),
            'avg_state': round(clause.average_state, 1),
            'near_boundary': clause.near_boundary_fraction,
            'saturated': clause.saturated_fraction,
            'literal_peer': clause.literal_peer_clause_id,
            'literal_similarity': clause.max_literal_jaccard,
            'activation_peer': clause.activation_peer_clause_id,
            'activation_similarity': clause.max_activation_jaccard,
        }
        for clause in diagnostics.clauses
    ]
    rows.sort(key=lambda row: row['support_rate'], reverse=True)
    return rows


def ta_histogram(diagnostics: RunDiagnostics) -> list[int]:
    return list(diagnostics.ta_population.state_histogram)


def clause_health(diagnostics: RunDiagnostics) -> dict[str, Any]:
    rows = clause_rows(diagnostics)
    count = len(rows)
    empty = sum(1 for row in rows if row['lits'] == 0)
    nonempty = count - empty
    return {
        'avg_ta': round(diagnostics.ta_population.average_state, 1),
        'empty': empty,
        'empty_pct': round(empty / count * 100, 1) if count else 0,
        'nonempty': nonempty,
        'nonempty_pct': round(nonempty / count * 100, 1) if count else 0,
        'unique': sum(row['unique_support'] > 0 for row in rows),
    }
