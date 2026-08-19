% PTA De-escalation -- Type-III-inspired pruning + reversible absorption
% Python DeescalationPTA is reference oracle; Prolog drives equivalence/subsumption reasoning.

:- include('pta_ontology.pl').

% Rich factual primitives for exact reasoning
% literal_truth(Literal, Example, 0|1), clause_truth/3, clause_literal/2 etc. are asserted by Python

% Thresholds equivalent: truth columns identical across all examples
thresholds_equivalent(L1, L2) :-
    L1 \= L2,
    \+ (literal_truth(L1, E, V1), literal_truth(L2, E, V2), V1 \= V2),
    \+ (literal_truth(L2, E, V2), literal_truth(L1, E, V1), V1 \= V2),
    % At least one example exists
    literal_truth(L1, _, _), literal_truth(L2, _, _).

% L1 subsumes L2 if whenever L1 true then L2 true, and not equivalent
literal_subsumes(L1, L2) :-
    L1 \= L2,
    \+ (literal_truth(L1, E, 1), literal_truth(L2, E, 0)),
    \+ thresholds_equivalent(L1, L2).

% Clause C1 subsumes C2 where C1 literal set subset C2 and support superset
clause_subsumes(C1, C2) :-
    C1 \= C2,
    \+ (clause_literal(C1, L), \+ clause_literal(C2, L)),
    % C1 fires whenever C2 fires (support superset) -- derived from clause_truth
    \+ (clause_truth(C2, E, 1), clause_truth(C1, E, 0)).

% Stable inclusion for shadow audit (utility + use-count thresholds in Python)
stable_inclusion(Literal) :-
    insight(_, stable_inclusion, Literal, _).
