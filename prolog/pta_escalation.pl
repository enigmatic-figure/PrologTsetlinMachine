% PTA Escalation — structure invention + CoTM weight allocation
% Python EscalationPTA is reference oracle with greedy solver; Prolog CLP(FD) will replace.

:- include('pta_ontology.pl').

% Exception clause for failing examples differing on field
exception_clause(Field, Threshold, Clause) :-
    observation(_, _, Field, _),
    Threshold is 0, Clause = [Field, Threshold].

% CoTM weight proposal: Clause x Output -> Weight, bounded |Weight| =< 1e6
% Greedy reference in Python; Prolog CLP(FD) will enforce constraints exactly
cotm_weight(Clause, Class, Weight) :-
    integer(Clause), integer(Class), integer(Weight),
    Weight >= -1000000, Weight =< 1000000.

% Graph depth increase bounded to 8
graph_depth_increase(Current, New) :-
    integer(Current), Current < 8,
    New is Current + 1, bounded_depth(New).

% Specialist gate: has_relation_structure -> graph_model
specialist_gate(Condition, Specialist) :-
    member(Condition, [has_relation_structure, contains_long_text]),
    member(Specialist, [graph_model, text_model, numeric_model]).
