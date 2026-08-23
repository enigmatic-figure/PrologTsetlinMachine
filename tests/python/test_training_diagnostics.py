"""Completed-run research diagnostics derived from immutable snapshots."""

from __future__ import annotations

from dataclasses import replace

import pytest

from prolog_tsetlin.reference import SNAPSHOT_SCHEMA_VERSION, TMSnapshot
from prolog_tsetlin.services.diagnostics import (
    analyze_training_run,
    analyze_training_sample,
    compare_training_samples,
)
from prolog_tsetlin.services.training import (
    TrainingDiagnosticSample,
    TrainingRequest,
    TrainingRun,
)


def _controlled_run(*, predictions: tuple[int, ...] = (0, 0, 1, 1)) -> TrainingRun:
    request = TrainingRequest(
        number_of_clauses=4,
        states_per_action=3,
        specificity=3.0,
        threshold=8,
        epochs=1,
    )
    snapshot = TMSnapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        number_of_clauses=4,
        number_of_features=2,
        states_per_action=3,
        specificity=3.0,
        threshold=8,
        states=(
            (6, 3, 3, 1),  # positive: x0
            (3, 4, 3, 3),  # negative: not x0
            (3, 3, 4, 3),  # positive: x1
            (4, 3, 4, 3),  # negative: x0 and x1
        ),
        rng_state=None,
    )
    rows = (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    )
    targets = (0, 1, 1, 0)
    accuracy = sum(a == b for a, b in zip(predictions, targets)) / 4
    return TrainingRun(request, rows, targets, predictions, accuracy, snapshot)


def test_clause_diagnostics_reconstruct_behavior_and_influence() -> None:
    diagnostics = analyze_training_run(
        _controlled_run(), boundary_window=1, histogram_bins=6
    )

    assert [example.score for example in diagnostics.examples] == [-1, 0, 1, 1]
    assert [example.correct for example in diagnostics.examples] == [
        True,
        False,
        True,
        False,
    ]

    clause = diagnostics.clause(0)
    assert clause.polarity == 1
    assert clause.included_literals == (0,)
    assert clause.support_count == 2
    assert clause.support_fraction == pytest.approx(0.5)
    assert clause.signed_vote_sum == 2
    assert (clause.aligned_count, clause.opposed_count) == (1, 1)
    assert (
        clause.correct_activation_count,
        clause.incorrect_activation_count,
    ) == (1, 1)
    assert clause.unique_support_count == 1
    assert [example.signed_contribution for example in clause.examples] == [
        0,
        0,
        1,
        1,
    ]
    assert [example.target_aligned for example in clause.examples] == [
        None,
        None,
        True,
        False,
    ]


def test_clause_similarity_is_same_polarity_and_explicit() -> None:
    diagnostics = analyze_training_run(_controlled_run())

    positive = diagnostics.clause(0)
    assert positive.literal_peer_clause_id == 2
    assert positive.max_literal_jaccard == 0.0
    assert positive.activation_peer_clause_id == 2
    assert positive.max_activation_jaccard == pytest.approx(1 / 3)

    negative = diagnostics.clause(1)
    assert negative.literal_peer_clause_id == 3
    assert negative.max_literal_jaccard == 0.0
    assert negative.activation_peer_clause_id == 3
    assert negative.max_activation_jaccard == 0.0


def test_ta_population_diagnostics_use_action_boundary_distance() -> None:
    population = analyze_training_run(
        _controlled_run(), boundary_window=1, histogram_bins=6
    ).ta_population

    assert population.total_automata == 16
    assert population.included_count == 5
    assert population.included_fraction == pytest.approx(5 / 16)
    assert population.excluded_count == 11
    assert population.near_boundary_count == 14
    assert population.near_boundary_fraction == pytest.approx(14 / 16)
    assert population.saturated_count == 2
    assert population.saturated_fraction == pytest.approx(2 / 16)
    assert population.average_distance_to_boundary == pytest.approx(0.25)
    assert population.state_histogram == (1, 0, 10, 4, 0, 1)


def test_diagnostics_reject_snapshot_prediction_mismatch() -> None:
    with pytest.raises(ValueError, match="do not reproduce"):
        analyze_training_run(_controlled_run(predictions=(1, 1, 0, 0)))


def test_diagnostics_reject_recorded_accuracy_mismatch() -> None:
    with pytest.raises(ValueError, match="recorded accuracy"):
        analyze_training_run(replace(_controlled_run(), accuracy=1.0))


def _sample(run: TrainingRun, epoch: int) -> TrainingDiagnosticSample:
    return TrainingDiagnosticSample(
        request=run.request,
        epoch=epoch,
        rows=run.rows,
        targets=run.targets,
        predictions=run.predictions,
        accuracy=run.accuracy,
        snapshot=run.snapshot,
    )


def test_sample_diagnostics_use_the_same_validated_semantics_as_runs() -> None:
    run = _controlled_run()

    assert analyze_training_sample(_sample(run, 1)) == analyze_training_run(run)


def test_temporal_delta_measures_state_action_behavior_and_prediction_change() -> None:
    earlier_run = _controlled_run()
    earlier_run = replace(
        earlier_run,
        request=replace(earlier_run.request, epochs=2),
    )
    later_snapshot = replace(
        earlier_run.snapshot,
        states=(
            (3, 3, 3, 1),  # positive clause becomes empty
            *earlier_run.snapshot.states[1:],
        ),
    )
    later_predictions = (0, 0, 0, 0)
    later_run = replace(
        earlier_run,
        predictions=later_predictions,
        accuracy=0.5,
        snapshot=later_snapshot,
    )

    delta = compare_training_samples(
        _sample(earlier_run, 1), _sample(later_run, 2)
    )

    assert (delta.earlier_epoch, delta.later_epoch) == (1, 2)
    assert delta.epoch_span == 1
    assert delta.changed_automata_count == 1
    assert delta.changed_automata_fraction == pytest.approx(1 / 16)
    assert delta.mean_absolute_state_change == pytest.approx(3 / 16)
    assert delta.mean_absolute_state_change_per_epoch == pytest.approx(3 / 16)
    assert delta.maximum_absolute_state_change == 3
    assert delta.action_flip_count == 1
    assert delta.action_flip_fraction == pytest.approx(1 / 16)
    assert delta.clause_behavior_flip_count == 2
    assert delta.clause_behavior_flip_fraction == pytest.approx(2 / 16)
    assert delta.prediction_flip_count == 2
    assert delta.prediction_flip_fraction == pytest.approx(2 / 4)
