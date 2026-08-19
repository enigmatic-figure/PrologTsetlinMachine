% PTA Escalation -- structure invention + CoTM weight allocation
% Python EscalationPTA is reference oracle with greedy solver; Prolog CLP(FD) will replace.

:- include('pta_ontology.pl').

% Exception clause via example_label join (discovers real threshold)
exception_clause(Field, Threshold, Clause) :-
    findall(V-Y, (observation(_, E, Field, V), example_label(E, Y)), Pairs),
    sort(Pairs, Sorted),
    Sorted = [V1-Y1, V2-Y2 | _], Y1 \= Y2, Threshold is (V1 + V2) / 2,
    Clause = [Field, Threshold].

% CoTM weight via class_support and clause_class_score facts
cotm_weight(Clause, Class, Weight) :-
    clause_class_score(Clause, Class, Score),
    class_support(Class, _, _),
    Weight is truncate(Score * 4),
    Weight >= -1000000, Weight =< 1000000,
    Weight =\= 0.

% Graph depth increase bounded to 8
graph_depth_increase(Current, New) :-
    integer(Current), Current < 8,
    New is Current + 1, bounded_depth(New).

% Specialist gate
specialist_gate(Condition, Specialist) :-
    member(Condition, [has_relation_structure, contains_long_text]),
    member(Specialist, [graph_model, text_model, numeric_model]).
