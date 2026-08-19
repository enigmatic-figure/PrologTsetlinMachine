"""PTA communication ontology — minimal shared predicates.

Prolog and Python share these terms; Python mirrors them as dataclasses for
deterministic reasoning. Free-form terms are intentionally not allowed — every
insight that survives must be an instance of this ontology so a graph PTA can
consume a numeric de-escalation PTA's insight without re-discovery.
"""

# Prolog stub — include via `:- include('pta_ontology.pl').`
PROLOG_ONTOLOGY = r"""
% PTA shared ontology — keep in sync with python/prolog_tsetlin/pta/proposal.py
% observation(PTA, Example, Field, RawValue).
% feature_support(Literal, Pos, Neg).
% feature_relation(L1, subsumes, L2).  % subsumes | equivalent | thresholds_equivalent
% clause_support(Clause, Example).
% clause_conflict(Clause, Example).
% insight(SourcePTA, Kind, Subject, Evidence).
% counterexample(Model, Example, Expected, Actual).
% proposal(PTA, Kind, Candidate).
% lowerable(Candidate, Target).  % exact, no approximation
:- dynamic observation/4, feature_support/3, feature_relation/3.
:- dynamic clause_support/2, clause_conflict/2, insight/4, counterexample/4, proposal/3.

similar_failure(A,B) :-
    clause_conflict(C,A), clause_conflict(C,B),
    insight(_, thresholds_equivalent, _, _).
"""

__all__ = ["PROLOG_ONTOLOGY"]
