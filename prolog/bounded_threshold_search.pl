% Bounded exact search for a masked k-of-n Boolean kernel.
%
% A problem is supplied by a driver as:
%   problem(SlotCount, MaxSelected, PositiveExamples, NegativeExamples).
%
% Each example is a sorted list of true, zero-based input slots. Search is
% deliberately finite: selected width is bounded by MaxSelected and every
% candidate is drawn from 0..SlotCount-1.

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

ptm_contains(Value, [Value|_]) :- !.
ptm_contains(Value, [_|Tail]) :-
    ptm_contains(Value, Tail).

ptm_match_count([], _, 0).
ptm_match_count([Slot|Slots], Example, Count) :-
    ptm_contains(Slot, Example),
    !,
    ptm_match_count(Slots, Example, TailCount),
    Count is TailCount + 1.
ptm_match_count([_|Slots], Example, Count) :-
    ptm_match_count(Slots, Example, Count).

ptm_all_positive([], _, _).
ptm_all_positive([Example|Examples], Selected, MinimumTrue) :-
    ptm_match_count(Selected, Example, Count),
    Count >= MinimumTrue,
    ptm_all_positive(Examples, Selected, MinimumTrue).

ptm_all_negative([], _, _).
ptm_all_negative([Example|Examples], Selected, MinimumTrue) :-
    ptm_match_count(Selected, Example, Count),
    Count < MinimumTrue,
    ptm_all_negative(Examples, Selected, MinimumTrue).

ptm_search_threshold(SlotCount, MaxSelected, Positives, Negatives,
                     Selected, MinimumTrue) :-
    SlotCount > 0,
    MaxSelected > 0,
    EffectiveMax is min(SlotCount, MaxSelected),
    ptm_slots(SlotCount, Slots),
    ptm_integer_between(1, EffectiveMax, Width),
    ptm_choose_exact(Width, Slots, Selected),
    ptm_integer_between(1, Width, MinimumTrue),
    ptm_all_positive(Positives, Selected, MinimumTrue),
    ptm_all_negative(Negatives, Selected, MinimumTrue),
    !.

ptm_write_csv([]).
ptm_write_csv([Value]) :-
    write(Value).
ptm_write_csv([Value,Next|Rest]) :-
    write(Value),
    write(','),
    ptm_write_csv([Next|Rest]).

ptm_emit_solution(Selected, MinimumTrue) :-
    write('PTM_RESULT v1 masked_threshold selected='),
    ptm_write_csv(Selected),
    write(' minimum='),
    write(MinimumTrue),
    write(' mismatches=0'),
    nl.

ptm_emit_no_solution :-
    write('PTM_RESULT v1 no_solution'),
    nl.

ptm_run_problem :-
    problem(SlotCount, MaxSelected, Positives, Negatives),
    ( ptm_search_threshold(SlotCount, MaxSelected, Positives, Negatives,
                           Selected, MinimumTrue) ->
        ptm_emit_solution(Selected, MinimumTrue)
    ;
        ptm_emit_no_solution
    ),
    halt.
