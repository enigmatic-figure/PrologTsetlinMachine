% PTA Input -- learned Booleanization with symbolic provenance
% Part of PTAReasoningSession collective; Python owns bounds/validation, Prolog drives invention.

:- include('pta_ontology.pl').

% Invent numeric_ge thresholds where label flips between adjacent sorted values
% Requires both observation and example_label facts; Python bounds candidate width
invent_threshold(Field, Threshold) :-
    findall(V-Y, (observation(_, E, Field, V), example_label(E, Y)), Pairs),
    sort(Pairs, Sorted),
    adjacent_flip(Sorted, Threshold).

adjacent_flip([V1-Y1, V2-Y2 | _], Threshold) :-
    Y1 \= Y2, V1 \= V2, Threshold is (V1 + V2) / 2.
adjacent_flip([_ | Rest], Threshold) :-
    adjacent_flip(Rest, Threshold).

% Interval [Lo,Hi) covering maximal positive run bounded by negatives — halfway to neighboring negatives like Python reference
invent_interval(Field, Lo, Hi) :-
    findall(V-Y, (observation(_, E, Field, V), example_label(E, Y)), Pairs),
    sort(Pairs, Sorted),
    positive_run(Sorted, Lo0, Hi0, Prev, Next),
    ( Prev = none -> Lo = Lo0 ; Lo is (Prev + Lo0) / 2 ),
    ( Next = none -> Hi = Hi0 ; Hi is (Hi0 + Next) / 2 ).

% positive_run finds maximal contiguous 1-run, also returns neighboring negative values
positive_run(Sorted, Lo, Hi, Prev, Next) :-
    append(Before, [V-1 | Rest], Sorted),
    % V is first 1 of run, Before ends with 0 or empty
    ( Before = [] -> Prev = none ; my_last(Before, PrevV-PrevY), ( PrevY = 0 -> Prev = PrevV ; Prev = none ) ),
    run_end([V-1 | Rest], Hi, Next).

run_end([V-1 | Rest], Hi, Next) :-
    run_end_acc(Rest, V, Hi, Next).
run_end_acc([], Hi, Hi, none).
run_end_acc([V2-Y2 | Rest], CurrHi, Hi, Next) :-
    ( Y2 = 1 -> run_end_acc(Rest, V2, Hi, Next) ; Hi = CurrHi, Next = V2 ).

% Helper: last element (use built-in last/2 if available, else define)
:- dynamic(my_last/2).
my_last([X], X) :- !.
my_last([_ | T], X) :- my_last(T, X).

% Alternative: discover thresholds via counterexample-guided join
threshold_via_counterexample(Field, Threshold) :-
    findall(V-Y, (observation(_, E, Field, V), example_label(E, Y)), Pairs),
    sort(Pairs, Sorted),
    adjacent_flip(Sorted, Threshold).
