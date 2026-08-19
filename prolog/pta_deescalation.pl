% PTA De-escalation — Type-III-inspired pruning + reversible absorption
% Python DeescalationPTA is reference oracle; Prolog drives equivalence/subsumption reasoning.

:- include('pta_ontology.pl').

% Thresholds equivalent on observed rows (column equality)
thresholds_equivalent(L1, L2) :-
    feature_support(L1, Pos1, _), feature_support(L2, Pos2, _),
    Pos1 == Pos2,
    insight(_, thresholds_equivalent, _, _).

% L1 subsumes L2 if L1 -> L2 on all rows and not equivalent
literal_subsumes(L1, L2) :-
    feature_relation(L1, subsumes, L2).

% Clause C1 subsumes C2 where C1 literal set subset C2
clause_subsumes(C1, C2) :-
    clause_support(C1, _), clause_support(C2, _),
    insight(_, clause_subsumes, _, _).

% Stable inclusion for shadow audit (utility + use-count thresholds in Python)
stable_inclusion(Literal) :-
    insight(_, stable_inclusion, Literal, _).
