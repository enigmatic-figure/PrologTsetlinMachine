% PTA Input -- learned Booleanization with symbolic provenance
% Part of PTAReasoningSession collective; Python owns bounds/validation, Prolog drives invention.

% The collective driver loads pta_ontology.pl before this file. Keeping module
% composition in the driver avoids consulting the ontology three times.

% Invent numeric_ge thresholds where label flips between adjacent sorted values
% Requires both observation and example_label facts; Python bounds candidate width
invent_threshold(Field, Threshold) :-
    findall(V-Y, (observation(_, E, Field, V), example_label(E, Y)), Pairs),
    sort(Pairs, Sorted),
    numeric_states(Sorted, States),
    adjacent_flip(States, Threshold).

adjacent_flip([V1-Y1, V2-Y2 | _], Threshold) :-
    (Y1 = 0 ; Y1 = 1),
    (Y2 = 0 ; Y2 = 1),
    Y1 \= Y2,
    Threshold is (V1 + V2) / 2.
adjacent_flip([_ | Rest], Threshold) :-
    adjacent_flip(Rest, Threshold).

% Consolidate duplicate observations into one state per numeric value. A mixed
% value is a barrier: it cannot create a zero-width threshold or masquerade as
% part of an exact positive run.
numeric_states(Pairs, States) :-
    findall(V, member(V-_, Pairs), Values),
    sort(Values, UniqueValues),
    value_states(UniqueValues, Pairs, States).

value_states([], _, []).
value_states([V | Rest], Pairs, [V-State | States]) :-
    value_state(V, Pairs, State),
    value_states(Rest, Pairs, States).

value_state(V, Pairs, mixed) :-
    member(V-0, Pairs),
    member(V-1, Pairs), !.
value_state(V, Pairs, 1) :-
    member(V-1, Pairs), !.
value_state(_, _, 0).

% Interval [Lo,Hi) covering maximal positive run bounded by negatives — halfway to neighboring negatives like Python reference
invent_interval(Field, Lo, Hi) :-
    findall(V-Y, (observation(_, E, Field, V), example_label(E, Y)), Pairs),
    sort(Pairs, Sorted),
    numeric_states(Sorted, States),
    positive_run(States, Lo0, Hi0, Prev, Next),
    % A finite [Lo,Hi) interval is justified only when observations establish
    % both neighboring negative regions. Edge runs are threshold candidates,
    % not finite intervals.
    Prev \= none, Next \= none,
    Lo is (Prev + Lo0) / 2,
    Hi is (Hi0 + Next) / 2.

% positive_run finds maximal contiguous 1-run, also returns neighboring negative values
positive_run(Sorted, Lo, Hi, Prev, Next) :-
    % Enumerate only real run starts: the beginning of the series or a 0->1
    % transition. Allowing every positive suffix creates redundant search paths.
    ( Sorted = [V-1 | Rest], Prev = none
    ; append(_, [PrevV-0, V-1 | Rest], Sorted), Prev = PrevV
    ),
    Lo = V,
    run_end([V-1 | Rest], Hi, Next).

run_end([V-1 | Rest], Hi, Next) :-
    run_end_acc(Rest, V, Hi, Next).
run_end_acc([], Hi, Hi, none).
run_end_acc([V2-1 | Rest], _, Hi, Next) :-
    run_end_acc(Rest, V2, Hi, Next).
run_end_acc([V2-0 | _], CurrHi, CurrHi, V2).
run_end_acc([_-mixed | _], CurrHi, CurrHi, none).

% Alternative: discover thresholds via counterexample-guided join
threshold_via_counterexample(Field, Threshold) :-
    findall(V-Y, (observation(_, E, Field, V), example_label(E, Y)), Pairs),
    sort(Pairs, Sorted),
    numeric_states(Sorted, States),
    adjacent_flip(States, Threshold).
