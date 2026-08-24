% PTA Escalation -- structure invention + CoTM weight allocation
% Python EscalationPTA is reference oracle with greedy solver; Prolog CLP(FD) will replace.

% The collective driver loads pta_ontology.pl and pta_input.pl first.

% Exception clause via example_label join (discovers real threshold)
exception_clause(Field, Threshold, Clause) :-
    invent_threshold(Field, Threshold),
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
