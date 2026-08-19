% PTA shared ontology — compiled Prolog module, keep in sync with python/prolog_tsetlin/pta/proposal.py
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

% Bounded lowerability — exact gate (stubs, Python validates natively)
lowerable(Candidate, Target) :-
    nonvar(Candidate), nonvar(Target),
    member(Target, [binary_clause, shared_weighted_clause, regression_clause, patch_clause, graph_clause, logic_program, threshold, composite_gate]),
    % For now, delegate depth/bounds checks to Python lower_exact; Prolog ensures proposal well-formed
    true.

similar_failure(A,B) :-
    clause_conflict(C,A), clause_conflict(C,B),
    insight(_, thresholds_equivalent, _, _).

% Threshold inventiveness: midpoint where label flips (bounded, Python bounds width)
threshold_candidate(Field, Threshold) :-
    observation(_, E1, Field, V1), observation(_, E2, Field, V2),
    counterexample(_, E1, 0, 1), counterexample(_, E2, 1, 0),
    V1 \= V2, Threshold is (V1 + V2) / 2.

% Graph recursion bounded to 8 hops (exact lowering boundary)
bounded_depth(D) :- integer(D), D >= 1, D =< 8.
