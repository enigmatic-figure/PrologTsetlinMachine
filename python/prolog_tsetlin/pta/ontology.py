"""PTA communication ontology — minimal shared predicates.

Prolog and Python share these terms; Python mirrors them as dataclasses for
deterministic reasoning. Free-form terms are intentionally not allowed — every
insight that survives must be an instance of this ontology so a graph PTA can
consume a numeric de-escalation PTA's insight without re-discovery.

Prolog source of truth is prolog/pta_ontology.pl (compiled); Python reads it via shared resolver.
"""

from pathlib import Path
import sys

def _resolve_ontology() -> str:
    # Reuse mature resolver logic: check checkout then share/prolog-tsetlin-machine/prolog
    candidates = (
        Path(__file__).resolve().parents[3] / "prolog" / "pta_ontology.pl",
        Path(sys.prefix) / "share" / "prolog-tsetlin-machine" / "prolog" / "pta_ontology.pl",
    )
    for cand in candidates:
        if cand.is_file():
            return cand.read_text(encoding="utf-8")
    # Fallback stub
    return r"""
% PTA shared ontology — keep in sync with python/prolog_tsetlin/pta/proposal.py
:- dynamic(observation/4).
:- dynamic(feature_support/3).
:- dynamic(feature_relation/3).
:- dynamic(clause_support/2).
:- dynamic(clause_conflict/2).
:- dynamic(insight/4).
:- dynamic(counterexample/4).
:- dynamic(proposal/3).
similar_failure(A,B) :-
    clause_conflict(C,A), clause_conflict(C,B),
    insight(_, thresholds_equivalent, _, _).
"""

PROLOG_ONTOLOGY = _resolve_ontology()

__all__ = ["PROLOG_ONTOLOGY"]
