"""PTA communication ontology — minimal shared predicates.

Prolog and Python share these terms; Python mirrors them as dataclasses for
deterministic reasoning. Free-form terms are intentionally not allowed — every
insight that survives must be an instance of this ontology so a graph PTA can
consume a numeric de-escalation PTA's insight without re-discovery.

Prolog source of truth is prolog/pta_ontology.pl (compiled); Python reads it.
"""

from pathlib import Path

_ONTOLOGY_PATH = Path(__file__).parents[3] / "prolog" / "pta_ontology.pl"
try:
    PROLOG_ONTOLOGY = _ONTOLOGY_PATH.read_text(encoding="utf-8")
except Exception:
    # Fallback stub for environments without prolog/ checkout
    PROLOG_ONTOLOGY = r"""
% PTA shared ontology — keep in sync with python/prolog_tsetlin/pta/proposal.py
:- dynamic observation/4, feature_support/3, feature_relation/3.
:- dynamic clause_support/2, clause_conflict/2, insight/4, counterexample/4, proposal/3.
similar_failure(A,B) :-
    clause_conflict(C,A), clause_conflict(C,B),
    insight(_, thresholds_equivalent, _, _).
"""

__all__ = ["PROLOG_ONTOLOGY"]
