"""Truthful, snapshot-derived diagnostics for completed training runs."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Sequence

from ..reference import SNAPSHOT_SCHEMA_VERSION
from .training import TrainingDiagnosticSample, TrainingRun


@dataclass(frozen=True, slots=True)
class RunExampleDiagnostics:
    example_index: int
    features: tuple[bool, ...]
    target: int
    prediction: int
    score: int
    correct: bool


@dataclass(frozen=True, slots=True)
class ClauseExampleDiagnostics:
    example_index: int
    fires: bool
    signed_contribution: int
    target_aligned: bool | None


@dataclass(frozen=True, slots=True)
class ClauseDiagnostics:
    clause_id: int
    polarity: int
    included_literals: tuple[int, ...]
    average_state: float
    near_boundary_fraction: float
    saturated_fraction: float
    support_count: int
    support_fraction: float
    signed_vote_sum: int
    aligned_count: int
    opposed_count: int
    correct_activation_count: int
    incorrect_activation_count: int
    unique_support_count: int
    literal_peer_clause_id: int | None
    max_literal_jaccard: float | None
    activation_peer_clause_id: int | None
    max_activation_jaccard: float | None
    examples: tuple[ClauseExampleDiagnostics, ...]

    @property
    def polarity_label(self) -> str:
        return "positive" if self.polarity > 0 else "negative"


@dataclass(frozen=True, slots=True)
class TAPopulationDiagnostics:
    total_automata: int
    included_count: int
    included_fraction: float
    excluded_count: int
    excluded_fraction: float
    near_boundary_count: int
    near_boundary_fraction: float
    boundary_window: int
    saturated_count: int
    saturated_fraction: float
    average_state: float
    average_distance_to_boundary: float
    state_histogram: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RunDiagnostics:
    examples: tuple[RunExampleDiagnostics, ...]
    clauses: tuple[ClauseDiagnostics, ...]
    ta_population: TAPopulationDiagnostics

    def clause(self, clause_id: int) -> ClauseDiagnostics:
        if not 0 <= clause_id < len(self.clauses):
            raise IndexError(clause_id)
        return self.clauses[clause_id]


@dataclass(frozen=True, slots=True)
class TrainingSampleDelta:
    """Exact endpoint differences between compatible training snapshots."""

    earlier_epoch: int
    later_epoch: int
    epoch_span: int
    changed_automata_count: int
    changed_automata_fraction: float
    mean_absolute_state_change: float
    mean_absolute_state_change_per_epoch: float
    maximum_absolute_state_change: int
    action_flip_count: int
    action_flip_fraction: float
    clause_behavior_flip_count: int
    clause_behavior_flip_fraction: float
    prediction_flip_count: int
    prediction_flip_fraction: float


@dataclass(frozen=True, slots=True)
class SampledTrainingDiagnostics:
    """One sampled evaluation plus its change from the previous sample."""

    sample: TrainingDiagnosticSample
    diagnostics: RunDiagnostics
    delta_from_previous: TrainingSampleDelta | None


def _binary(value: object, *, label: str) -> bool:
    if value is True or value is False:
        return bool(value)
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError(f"{label} must be bool or integer 0/1")


def _literal_truths(row: Sequence[bool | int]) -> tuple[bool, ...]:
    truths: list[bool] = []
    for index, value in enumerate(row):
        truth = _binary(value, label=f"feature {index}")
        truths.extend((truth, not truth))
    return tuple(truths)


def _distance_to_action_boundary(state: int, boundary: int) -> int:
    """Return zero for the two states adjacent to the action boundary."""

    return boundary - state if state <= boundary else state - boundary - 1


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float | None:
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def _most_similar_same_polarity(
    clause_id: int,
    polarities: Sequence[int],
    values: Sequence[frozenset[int]],
) -> tuple[int | None, float | None]:
    candidates: list[tuple[float, int]] = []
    for other_id, other in enumerate(values):
        if other_id == clause_id or polarities[other_id] != polarities[clause_id]:
            continue
        similarity = _jaccard(values[clause_id], other)
        if similarity is not None:
            candidates.append((similarity, other_id))
    if not candidates:
        return None, None
    similarity, peer = max(candidates, key=lambda item: (item[0], -item[1]))
    return peer, similarity


def analyze_training_run(
    run: TrainingRun,
    *,
    boundary_window: int = 5,
    histogram_bins: int = 20,
) -> RunDiagnostics:
    """Derive clause and TA diagnostics from one immutable completed run.

    ``boundary_window`` counts the nearest states on each side. A value of five
    therefore includes ``N-4..N`` and ``N+1..N+5`` for an action boundary
    between ``N`` and ``N+1``.
    """

    if boundary_window <= 0:
        raise ValueError("boundary_window must be positive")
    if histogram_bins <= 0:
        raise ValueError("histogram_bins must be positive")
    run.request.validate()
    if not (len(run.rows) == len(run.targets) == len(run.predictions)):
        raise ValueError("training run rows, targets, and predictions must align")

    snapshot = run.snapshot
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported snapshot schema version")
    request_contract = (
        run.request.number_of_clauses,
        run.request.states_per_action,
        run.request.specificity,
        run.request.threshold,
    )
    snapshot_contract = (
        snapshot.number_of_clauses,
        snapshot.states_per_action,
        snapshot.specificity,
        snapshot.threshold,
    )
    if snapshot_contract != request_contract:
        raise ValueError("training request does not match snapshot configuration")
    if snapshot.number_of_features <= 0:
        raise ValueError("snapshot feature count must be positive")
    if len(snapshot.states) != snapshot.number_of_clauses:
        raise ValueError("snapshot clause count does not match its state matrix")
    expected_literals = 2 * snapshot.number_of_features
    boundary = snapshot.states_per_action
    maximum_state = 2 * boundary
    if boundary <= 0:
        raise ValueError("states_per_action must be positive")
    if snapshot.threshold <= 0:
        raise ValueError("snapshot threshold must be positive")

    rows: list[tuple[bool, ...]] = []
    literal_rows: list[tuple[bool, ...]] = []
    for row in run.rows:
        normalized = tuple(
            _binary(value, label=f"feature {index}")
            for index, value in enumerate(row)
        )
        if len(normalized) != snapshot.number_of_features:
            raise ValueError("training row width does not match snapshot features")
        rows.append(normalized)
        literal_rows.append(_literal_truths(normalized))

    targets = tuple(
        int(_binary(value, label=f"target {index}"))
        for index, value in enumerate(run.targets)
    )
    predictions = tuple(
        int(_binary(value, label=f"prediction {index}"))
        for index, value in enumerate(run.predictions)
    )
    included_sets: list[frozenset[int]] = []
    activation_sets: list[frozenset[int]] = []
    activations: list[tuple[bool, ...]] = []
    polarities: list[int] = []
    all_states: list[int] = []
    for clause_id, states in enumerate(snapshot.states):
        if len(states) != expected_literals:
            raise ValueError("snapshot clause has the wrong literal width")
        for state in states:
            if type(state) is not int or not 1 <= state <= maximum_state:
                raise ValueError("snapshot TA state lies outside its action regions")
        all_states.extend(states)
        included = frozenset(
            literal_id
            for literal_id, state in enumerate(states)
            if state > boundary
        )
        clause_activations = tuple(
            bool(included)
            and all(literals[literal_id] for literal_id in included)
            for literals in literal_rows
        )
        included_sets.append(included)
        activations.append(clause_activations)
        activation_sets.append(
            frozenset(
                index
                for index, fires in enumerate(clause_activations)
                if fires
            )
        )
        polarities.append(1 if clause_id % 2 == 0 else -1)

    raw_scores = tuple(
        sum(
            polarities[clause_id]
            for clause_id in range(snapshot.number_of_clauses)
            if activations[clause_id][example_index]
        )
        for example_index in range(len(rows))
    )
    scores = tuple(
        max(-snapshot.threshold, min(snapshot.threshold, score))
        for score in raw_scores
    )
    reconstructed = tuple(int(score > 0) for score in scores)
    if reconstructed != predictions:
        raise ValueError(
            "snapshot clause evaluations do not reproduce recorded predictions"
        )

    examples = tuple(
        RunExampleDiagnostics(
            example_index=index,
            features=row,
            target=targets[index],
            prediction=predictions[index],
            score=scores[index],
            correct=targets[index] == predictions[index],
        )
        for index, row in enumerate(rows)
    )
    measured_accuracy = (
        sum(example.correct for example in examples) / len(examples)
        if examples
        else 0.0
    )
    if not isclose(run.accuracy, measured_accuracy, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("recorded accuracy does not match run predictions")

    clauses: list[ClauseDiagnostics] = []
    for clause_id, states in enumerate(snapshot.states):
        polarity = polarities[clause_id]
        clause_activations = activations[clause_id]
        support_count = sum(clause_activations)
        aligned_count = sum(
            fires and target == int(polarity > 0)
            for fires, target in zip(clause_activations, targets)
        )
        opposed_count = support_count - aligned_count
        correct_activation_count = sum(
            fires and example.correct
            for fires, example in zip(clause_activations, examples)
        )
        incorrect_activation_count = support_count - correct_activation_count
        same_polarity = tuple(
            other_id
            for other_id, other_polarity in enumerate(polarities)
            if other_polarity == polarity
        )
        unique_support_count = sum(
            fires
            and sum(activations[other_id][example_index] for other_id in same_polarity)
            == 1
            for example_index, fires in enumerate(clause_activations)
        )
        literal_peer, literal_similarity = _most_similar_same_polarity(
            clause_id, polarities, included_sets
        )
        activation_peer, activation_similarity = _most_similar_same_polarity(
            clause_id, polarities, activation_sets
        )
        near_count = sum(
            _distance_to_action_boundary(state, boundary) < boundary_window
            for state in states
        )
        saturated_count = sum(state in (1, maximum_state) for state in states)
        clause_examples = tuple(
            ClauseExampleDiagnostics(
                example_index=index,
                fires=fires,
                signed_contribution=polarity if fires else 0,
                target_aligned=(
                    targets[index] == int(polarity > 0) if fires else None
                ),
            )
            for index, fires in enumerate(clause_activations)
        )
        clauses.append(
            ClauseDiagnostics(
                clause_id=clause_id,
                polarity=polarity,
                included_literals=tuple(sorted(included_sets[clause_id])),
                average_state=sum(states) / len(states),
                near_boundary_fraction=near_count / len(states),
                saturated_fraction=saturated_count / len(states),
                support_count=support_count,
                support_fraction=(support_count / len(rows) if rows else 0.0),
                signed_vote_sum=polarity * support_count,
                aligned_count=aligned_count,
                opposed_count=opposed_count,
                correct_activation_count=correct_activation_count,
                incorrect_activation_count=incorrect_activation_count,
                unique_support_count=unique_support_count,
                literal_peer_clause_id=literal_peer,
                max_literal_jaccard=literal_similarity,
                activation_peer_clause_id=activation_peer,
                max_activation_jaccard=activation_similarity,
                examples=clause_examples,
            )
        )

    total_automata = len(all_states)
    included_count = sum(state > boundary for state in all_states)
    near_boundary_count = sum(
        _distance_to_action_boundary(state, boundary) < boundary_window
        for state in all_states
    )
    saturated_count = sum(state in (1, maximum_state) for state in all_states)
    histogram = [0] * histogram_bins
    for state in all_states:
        bucket = min(
            histogram_bins - 1,
            (state - 1) * histogram_bins // maximum_state,
        )
        histogram[bucket] += 1
    ta_population = TAPopulationDiagnostics(
        total_automata=total_automata,
        included_count=included_count,
        included_fraction=(included_count / total_automata if total_automata else 0.0),
        excluded_count=total_automata - included_count,
        excluded_fraction=(
            (total_automata - included_count) / total_automata
            if total_automata
            else 0.0
        ),
        near_boundary_count=near_boundary_count,
        near_boundary_fraction=(
            near_boundary_count / total_automata if total_automata else 0.0
        ),
        boundary_window=boundary_window,
        saturated_count=saturated_count,
        saturated_fraction=(
            saturated_count / total_automata if total_automata else 0.0
        ),
        average_state=(sum(all_states) / total_automata if total_automata else 0.0),
        average_distance_to_boundary=(
            sum(
                _distance_to_action_boundary(state, boundary)
                for state in all_states
            )
            / total_automata
            if total_automata
            else 0.0
        ),
        state_histogram=tuple(histogram),
    )
    return RunDiagnostics(examples, tuple(clauses), ta_population)


def analyze_training_sample(
    sample: TrainingDiagnosticSample,
    *,
    boundary_window: int = 5,
    histogram_bins: int = 20,
) -> RunDiagnostics:
    """Apply completed-run diagnostics to an immutable in-training sample."""

    if not 1 <= sample.epoch <= sample.request.epochs:
        raise ValueError("diagnostic sample epoch lies outside its training request")
    return analyze_training_run(
        TrainingRun(
            request=sample.request,
            rows=sample.rows,
            targets=sample.targets,
            predictions=sample.predictions,
            accuracy=sample.accuracy,
            snapshot=sample.snapshot,
        ),
        boundary_window=boundary_window,
        histogram_bins=histogram_bins,
    )


def compare_training_samples(
    earlier: TrainingDiagnosticSample,
    later: TrainingDiagnosticSample,
) -> TrainingSampleDelta:
    """Measure exact TA, clause-behavior, and prediction movement."""

    if earlier.request != later.request:
        raise ValueError("diagnostic samples belong to different training requests")
    if later.epoch <= earlier.epoch:
        raise ValueError("diagnostic sample epochs must increase")
    if earlier.rows != later.rows or earlier.targets != later.targets:
        raise ValueError("diagnostic samples use different evaluation data")

    earlier_diagnostics = analyze_training_sample(earlier)
    later_diagnostics = analyze_training_sample(later)
    earlier_states = tuple(
        state for clause in earlier.snapshot.states for state in clause
    )
    later_states = tuple(
        state for clause in later.snapshot.states for state in clause
    )
    if len(earlier_states) != len(later_states):
        raise ValueError("diagnostic samples have incompatible TA populations")
    total_automata = len(earlier_states)
    absolute_changes = tuple(
        abs(after - before)
        for before, after in zip(earlier_states, later_states)
    )
    changed_automata_count = sum(change > 0 for change in absolute_changes)
    boundary = earlier.snapshot.states_per_action
    action_flip_count = sum(
        (before > boundary) != (after > boundary)
        for before, after in zip(earlier_states, later_states)
    )

    behavior_pairs = tuple(
        (before.fires, after.fires)
        for earlier_clause, later_clause in zip(
            earlier_diagnostics.clauses, later_diagnostics.clauses
        )
        for before, after in zip(earlier_clause.examples, later_clause.examples)
    )
    clause_behavior_flip_count = sum(
        before != after for before, after in behavior_pairs
    )
    prediction_pairs = tuple(zip(earlier.predictions, later.predictions))
    prediction_flip_count = sum(
        before != after for before, after in prediction_pairs
    )

    return TrainingSampleDelta(
        earlier_epoch=earlier.epoch,
        later_epoch=later.epoch,
        epoch_span=later.epoch - earlier.epoch,
        changed_automata_count=changed_automata_count,
        changed_automata_fraction=(
            changed_automata_count / total_automata if total_automata else 0.0
        ),
        mean_absolute_state_change=(
            sum(absolute_changes) / total_automata if total_automata else 0.0
        ),
        mean_absolute_state_change_per_epoch=(
            sum(absolute_changes)
            / total_automata
            / (later.epoch - earlier.epoch)
            if total_automata
            else 0.0
        ),
        maximum_absolute_state_change=max(absolute_changes, default=0),
        action_flip_count=action_flip_count,
        action_flip_fraction=(
            action_flip_count / total_automata if total_automata else 0.0
        ),
        clause_behavior_flip_count=clause_behavior_flip_count,
        clause_behavior_flip_fraction=(
            clause_behavior_flip_count / len(behavior_pairs)
            if behavior_pairs
            else 0.0
        ),
        prediction_flip_count=prediction_flip_count,
        prediction_flip_fraction=(
            prediction_flip_count / len(prediction_pairs)
            if prediction_pairs
            else 0.0
        ),
    )
