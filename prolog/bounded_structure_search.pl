% Bounded Class III searches for typed templates, signed TA clauses, and
% read-once Boolean decision trees. Drivers contain numeric facts only.

ptm_contains(Value, [Value|_]) :- !.
ptm_contains(Value, [_|Tail]) :-
    ptm_contains(Value, Tail).

ptm_all_members([], _).
ptm_all_members([Value|Values], Set) :-
    ptm_contains(Value, Set),
    ptm_all_members(Values, Set).

ptm_no_members([], _).
ptm_no_members([Value|Values], Set) :-
    \+ ptm_contains(Value, Set),
    ptm_no_members(Values, Set).

ptm_nth0(0, [Value|_], Value).
ptm_nth0(Index, [_|Values], Value) :-
    ptm_nth0(TailIndex, Values, Value),
    Index is TailIndex + 1.

% A typed feature candidate is represented by its coverage over finite example
% identifiers. Python owns and validates the candidate's field, type, template
% identifier, and parameters; Prolog returns only its bounded numeric index.
ptm_search_feature_template(Positives, Negatives, Coverages, Candidate) :-
    ptm_nth0(Candidate, Coverages, Coverage),
    ptm_all_members(Positives, Coverage),
    ptm_no_members(Negatives, Coverage),
    !.

ptm_run_feature_template_problem :-
    feature_template_problem(Positives, Negatives, Coverages),
    ( ptm_search_feature_template(Positives, Negatives, Coverages, Candidate) ->
        write('PTM_RESULT v2 feature_template candidate='),
        write(Candidate),
        write(' mismatches=0'),
        nl
    ;
        write('PTM_RESULT v2 no_solution kind=feature_template'),
        nl
    ),
    halt.

ptm_integer_between(Lower, Upper, Lower) :-
    Lower =< Upper.
ptm_integer_between(Lower, Upper, Value) :-
    Lower < Upper,
    Next is Lower + 1,
    ptm_integer_between(Next, Upper, Value).

ptm_slots(Count, Slots) :-
    Count > 0,
    Last is Count - 1,
    ptm_slots_from(0, Last, Slots).

ptm_slots_from(Current, Last, []) :-
    Current > Last.
ptm_slots_from(Current, Last, [Current|Rest]) :-
    Current =< Last,
    Next is Current + 1,
    ptm_slots_from(Next, Last, Rest).

ptm_choose_exact(0, _, []).
ptm_choose_exact(Count, [Head|Tail], [Head|Chosen]) :-
    Count > 0,
    Remaining is Count - 1,
    ptm_choose_exact(Remaining, Tail, Chosen).
ptm_choose_exact(Count, [_|Tail], Chosen) :-
    Count > 0,
    ptm_choose_exact(Count, Tail, Chosen).

ptm_complement(Literal, Complement) :-
    Polarity is Literal mod 2,
    ( Polarity =:= 0 -> Complement is Literal + 1 ; Complement is Literal - 1 ).

ptm_consistent_literals([]).
ptm_consistent_literals([Literal|Literals]) :-
    ptm_complement(Literal, Complement),
    \+ ptm_contains(Complement, Literals),
    ptm_consistent_literals(Literals).

% Signed literal 2*N means feature N; 2*N+1 means NOT feature N.
ptm_literal_matches(Literal, Example) :-
    Feature is Literal // 2,
    Polarity is Literal mod 2,
    ( Polarity =:= 0 ->
        ptm_contains(Feature, Example)
    ;
        \+ ptm_contains(Feature, Example)
    ).

ptm_clause_matches([], _).
ptm_clause_matches([Literal|Literals], Example) :-
    ptm_literal_matches(Literal, Example),
    ptm_clause_matches(Literals, Example).

ptm_all_clause_matches([], _).
ptm_all_clause_matches([Example|Examples], Literals) :-
    ptm_clause_matches(Literals, Example),
    ptm_all_clause_matches(Examples, Literals).

ptm_all_clause_rejects([], _).
ptm_all_clause_rejects([Example|Examples], Literals) :-
    \+ ptm_clause_matches(Literals, Example),
    ptm_all_clause_rejects(Examples, Literals).

ptm_search_ta_clause(FeatureCount, MaxLiterals, Positives, Negatives, Literals) :-
    SignedCount is FeatureCount * 2,
    ptm_slots(SignedCount, SignedSlots),
    EffectiveMax is min(FeatureCount, MaxLiterals),
    ptm_integer_between(1, EffectiveMax, Width),
    ptm_choose_exact(Width, SignedSlots, Literals),
    ptm_consistent_literals(Literals),
    ptm_all_clause_matches(Positives, Literals),
    ptm_all_clause_rejects(Negatives, Literals),
    !.

ptm_write_csv([]).
ptm_write_csv([Value]) :-
    write(Value).
ptm_write_csv([Value,Next|Rest]) :-
    write(Value),
    write(','),
    ptm_write_csv([Next|Rest]).

ptm_run_ta_clause_problem :-
    ta_clause_problem(FeatureCount, MaxLiterals, Positives, Negatives),
    ( ptm_search_ta_clause(FeatureCount, MaxLiterals, Positives, Negatives,
                           Literals) ->
        write('PTM_RESULT v2 ta_clause literals='),
        ptm_write_csv(Literals),
        write(' mismatches=0'),
        nl
    ;
        write('PTM_RESULT v2 no_solution kind=ta_clause'),
        nl
    ),
    halt.

ptm_select(Head, [Head|Tail], Tail).
ptm_select(Value, [Head|Tail], [Head|Rest]) :-
    ptm_select(Value, Tail, Rest).

% Tree terms are l(Value) or n(Feature, FalseBranch, TrueBranch). Removing a
% selected feature from both branch domains makes every root-to-leaf path
% read-once while still allowing that feature in a disjoint branch.
ptm_tree(_, _, l(0)).
ptm_tree(_, _, l(1)).
ptm_tree(MaxDepth, Features, n(Feature, FalseBranch, TrueBranch)) :-
    MaxDepth > 0,
    ptm_select(Feature, Features, RemainingFeatures),
    ChildDepth is MaxDepth - 1,
    ptm_tree(ChildDepth, RemainingFeatures, FalseBranch),
    ptm_tree(ChildDepth, RemainingFeatures, TrueBranch),
    FalseBranch \= TrueBranch.

ptm_tree_value(l(Value), _, Value).
ptm_tree_value(n(Feature, FalseBranch, TrueBranch), Example, Value) :-
    ( ptm_contains(Feature, Example) ->
        ptm_tree_value(TrueBranch, Example, Value)
    ;
        ptm_tree_value(FalseBranch, Example, Value)
    ).

ptm_tree_classifies([], [], _).
ptm_tree_classifies([Example|Examples], [Label|Labels], Tree) :-
    ptm_tree_value(Tree, Example, Label),
    ptm_tree_classifies(Examples, Labels, Tree).

ptm_search_decision_tree(SlotCount, MaxDepth, Examples, Labels, Tree) :-
    ptm_slots(SlotCount, Features),
    ptm_tree(MaxDepth, Features, Tree),
    ptm_tree_classifies(Examples, Labels, Tree),
    !.

ptm_tree_nodes(l(_), 1).
ptm_tree_nodes(n(_, Left, Right), Count) :-
    ptm_tree_nodes(Left, LeftCount),
    ptm_tree_nodes(Right, RightCount),
    Count is LeftCount + RightCount + 1.

ptm_maximum(Left, Right, Left) :-
    Left >= Right,
    !.
ptm_maximum(_, Right, Right).

ptm_tree_depth(l(_), 0).
ptm_tree_depth(n(_, Left, Right), Depth) :-
    ptm_tree_depth(Left, LeftDepth),
    ptm_tree_depth(Right, RightDepth),
    ptm_maximum(LeftDepth, RightDepth, ChildDepth),
    Depth is ChildDepth + 1.

% Prefix encoding: leaf false=0, leaf true=1, node=2,Feature,Left,Right.
ptm_write_tree(l(Value)) :-
    write(Value).
ptm_write_tree(n(Feature, Left, Right)) :-
    write('2,'),
    write(Feature),
    write(','),
    ptm_write_tree(Left),
    write(','),
    ptm_write_tree(Right).

ptm_run_decision_tree_problem :-
    decision_tree_problem(SlotCount, MaxDepth, Examples, Labels),
    ( ptm_search_decision_tree(SlotCount, MaxDepth, Examples, Labels, Tree) ->
        ptm_tree_nodes(Tree, Nodes),
        ptm_tree_depth(Tree, Depth),
        write('PTM_RESULT v2 decision_tree nodes='),
        write(Nodes),
        write(' depth='),
        write(Depth),
        write(' tree='),
        ptm_write_tree(Tree),
        write(' mismatches=0'),
        nl
    ;
        write('PTM_RESULT v2 no_solution kind=decision_tree'),
        nl
    ),
    halt.
