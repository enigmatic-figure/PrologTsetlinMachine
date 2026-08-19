% PTA Input — learned Booleanization with symbolic provenance
% Part of PTAReasoningSession collective; Python owns bounds/validation, Prolog drives invention.

:- include('pta_ontology.pl').

% Invent numeric_ge thresholds where label flips between adjacent sorted values
% Python bounds candidate width, total examples, and validates finiteness before consult
invent_threshold(Field, Threshold) :-
    findall(V-Y, observation(_, _, Field, V), Pairs),
    sort(Pairs, Sorted),
    adjacent_flip(Sorted, Threshold).

adjacent_flip([V1-Y1, V2-Y2 | _], Threshold) :-
    Y1 \= Y2, V1 \= V2, Threshold is (V1 + V2) / 2.
adjacent_flip([_ | Rest], Threshold) :-
    adjacent_flip(Rest, Threshold).

% Interval [Lo,Hi) covering maximal positive run bounded by negatives
invent_interval(Field, Lo, Hi) :-
    findall(V-Y, observation(_, _, Field, V), Pairs),
    sort(Pairs, Sorted),
    positive_run(Sorted, Lo0, Hi0),
    % Expand to midpoint with neighbours (Python computes exact midpoints)
    Lo = Lo0, Hi = Hi0.

positive_run([V-1 | Rest], Lo, Hi) :-
    run_end([V-1 | Rest], Hi),
    Lo = V.
positive_run([_ | Rest], Lo, Hi) :-
    positive_run(Rest, Lo, Hi).

run_end([V-1 | Rest], Hi) :-
    ( Rest = [V2-1 | _] -> run_end(Rest, Hi) ; Hi = V ).
