"""Deterministic scalar binary Tsetlin Machine semantic reference."""

from __future__ import annotations

import copy
from math import exp, isfinite
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from .representation import LiteralBatch


SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TMSnapshot:
    schema_version: int
    number_of_clauses: int
    number_of_features: int
    states_per_action: int
    specificity: float
    threshold: int
    states: tuple[tuple[int, ...], ...]
    rng_state: object


@dataclass(frozen=True, slots=True)
class ResidualUpdateResult:
    """Observable result of one experimental residual-guided TM update."""

    raw_score: int
    student_probability: float
    target_probability: float
    residual: float
    feedback_probability: float


def extend_snapshot_features(
    snapshot: TMSnapshot,
    added_features: int,
) -> TMSnapshot:
    """Append deterministically excluded feature TAs to an adaptive snapshot.

    Scalar TM state rows are interleaved by feature as positive/negative TA
    pairs.  Appending ``(states_per_action, states_per_action)`` therefore
    preserves every existing position and places each new TA at the highest
    exclude state immediately below the inclusion boundary.  The RNG state is
    copied without constructing or advancing a fresh random stream.
    """

    if not isinstance(snapshot, TMSnapshot):
        raise TypeError("snapshot must be TMSnapshot")
    if type(added_features) is not int:
        raise TypeError("added_features must be an integer")
    if added_features <= 0:
        raise ValueError("added_features must be positive")
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("snapshot schema version is unsupported")
    if (
        snapshot.number_of_clauses <= 0
        or snapshot.number_of_features <= 0
        or snapshot.states_per_action <= 0
    ):
        raise ValueError("snapshot configuration is invalid")
    ScalarBinaryTsetlinMachine._check_snapshot_states(
        snapshot,
        number_of_clauses=snapshot.number_of_clauses,
        number_of_features=snapshot.number_of_features,
        states_per_action=snapshot.states_per_action,
    )
    suffix = (snapshot.states_per_action,) * (2 * added_features)
    return TMSnapshot(
        schema_version=snapshot.schema_version,
        number_of_clauses=snapshot.number_of_clauses,
        number_of_features=snapshot.number_of_features + added_features,
        states_per_action=snapshot.states_per_action,
        specificity=snapshot.specificity,
        threshold=snapshot.threshold,
        states=tuple(tuple(row) + suffix for row in snapshot.states),
        rng_state=copy.deepcopy(snapshot.rng_state),
    )


def contract_snapshot_equivalent_feature(
    snapshot: TMSnapshot,
    survivor_position: int,
    removed_position: int,
) -> TMSnapshot:
    """Consolidate one equivalent feature into another and remove its TA pair.

    Scalar state rows are interleaved positive/negative TA pairs. For each
    polarity the survivor receives the stronger of its own state and the
    removed feature's state before the removed pair is spliced out. Therefore
    inclusion is retained whenever either equivalent literal was included.
    Behavioral equivalence still depends on the two feature columns being
    equal and is proved by the model-generation layer over its authorized
    evidence. The RNG trajectory and every unrelated TA state are preserved.
    """

    if not isinstance(snapshot, TMSnapshot):
        raise TypeError("snapshot must be TMSnapshot")
    if type(survivor_position) is not int or type(removed_position) is not int:
        raise TypeError("feature positions must be integers")
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("snapshot schema version is unsupported")
    if (
        snapshot.number_of_clauses <= 0
        or snapshot.number_of_features <= 1
        or snapshot.states_per_action <= 0
    ):
        raise ValueError("snapshot configuration cannot be contracted")
    if (
        survivor_position == removed_position
        or not 0 <= survivor_position < snapshot.number_of_features
        or not 0 <= removed_position < snapshot.number_of_features
    ):
        raise ValueError("feature contraction positions are invalid")
    ScalarBinaryTsetlinMachine._check_snapshot_states(
        snapshot,
        number_of_clauses=snapshot.number_of_clauses,
        number_of_features=snapshot.number_of_features,
        states_per_action=snapshot.states_per_action,
    )

    survivor_first = 2 * survivor_position
    removed_first = 2 * removed_position
    states: list[tuple[int, ...]] = []
    for source in snapshot.states:
        contracted = list(source)
        contracted[survivor_first] = max(
            source[survivor_first], source[removed_first]
        )
        contracted[survivor_first + 1] = max(
            source[survivor_first + 1], source[removed_first + 1]
        )
        del contracted[removed_first : removed_first + 2]
        states.append(tuple(contracted))

    return TMSnapshot(
        schema_version=snapshot.schema_version,
        number_of_clauses=snapshot.number_of_clauses,
        number_of_features=snapshot.number_of_features - 1,
        states_per_action=snapshot.states_per_action,
        specificity=snapshot.specificity,
        threshold=snapshot.threshold,
        states=tuple(states),
        rng_state=copy.deepcopy(snapshot.rng_state),
    )


class ScalarBinaryTsetlinMachine:
    """Readable training oracle; it is intentionally not performance code."""

    def __init__(
        self,
        number_of_clauses: int,
        number_of_features: int,
        *,
        states_per_action: int = 100,
        specificity: float = 3.9,
        threshold: int = 15,
        seed: int = 1,
    ) -> None:
        if number_of_clauses <= 0 or number_of_features <= 0:
            raise ValueError("clause and feature counts must be positive")
        if states_per_action <= 0:
            raise ValueError("states_per_action must be positive")
        if not isfinite(specificity) or specificity <= 1.0:
            raise ValueError("specificity must be finite and greater than one")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.number_of_clauses = number_of_clauses
        self.number_of_features = number_of_features
        self.states_per_action = states_per_action
        self.specificity = specificity
        self.threshold = threshold
        self._rng = random.Random(seed)
        self._states = [
            [
                states_per_action + self._rng.randrange(2)
                for _ in range(2 * number_of_features)
            ]
            for _ in range(number_of_clauses)
        ]

    def _check_clause_literal(self, clause: int, literal: int) -> None:
        if not 0 <= clause < self.number_of_clauses:
            raise IndexError(clause)
        if not 0 <= literal < 2 * self.number_of_features:
            raise IndexError(literal)

    def state(self, clause: int, literal: int) -> int:
        self._check_clause_literal(clause, literal)
        return self._states[clause][literal]

    def set_state(self, clause: int, literal: int, state: int) -> None:
        self._check_clause_literal(clause, literal)
        if not 1 <= state <= 2 * self.states_per_action:
            raise ValueError("TA state lies outside its two action regions")
        self._states[clause][literal] = state

    def action_include(self, clause: int, literal: int) -> bool:
        return self.state(clause, literal) > self.states_per_action

    def _literals(self, features: Sequence[bool | int]) -> tuple[bool, ...]:
        if len(features) != self.number_of_features:
            raise ValueError("feature vector has the wrong width")
        result: list[bool] = []
        for feature in features:
            truth = self._require_binary(feature)
            result.extend((truth, not truth))
        return tuple(result)

    def clause_output(
        self,
        clause: int,
        features: Sequence[bool | int],
        *,
        prediction: bool = True,
    ) -> bool:
        literals = self._literals(features)
        included = False
        for literal_index, truth in enumerate(literals):
            if self.action_include(clause, literal_index):
                included = True
                if not truth:
                    return False
        return included if prediction else True

    def raw_vote(self, features: Sequence[bool | int]) -> int:
        """Return the unclipped signed sum of positive and negative clauses."""

        total = 0
        for clause in range(self.number_of_clauses):
            if self.clause_output(clause, features):
                total += 1 if clause % 2 == 0 else -1
        return total

    def score(self, features: Sequence[bool | int]) -> int:
        """Return the signed clause vote clipped to the training margin."""

        return max(-self.threshold, min(self.threshold, self.raw_vote(features)))

    def predict_one(self, features: Sequence[bool | int]) -> int:
        return int(self.score(features) > 0)

    def predict(self, rows: Iterable[Sequence[bool | int]]) -> list[int]:
        return [self.predict_one(row) for row in rows]

    def standard_feedback_probability(
        self,
        features: Sequence[bool | int],
        target: int | bool,
    ) -> float:
        """Return the canonical shared outer feedback gate for one example."""

        target_value = int(self._require_binary(target))
        class_sum = self.score(features)
        return (
            self.threshold + (1 - 2 * target_value) * class_sum
        ) / (2 * self.threshold)

    def _increment(self, clause: int, literal: int) -> None:
        current = self._states[clause][literal]
        self._states[clause][literal] = min(2 * self.states_per_action, current + 1)

    def _decrement(self, clause: int, literal: int) -> None:
        current = self._states[clause][literal]
        self._states[clause][literal] = max(1, current - 1)

    def _type_i_feedback(self, clause: int, literals: tuple[bool, ...], output: bool) -> None:
        reward_probability = (self.specificity - 1.0) / self.specificity
        penalty_probability = 1.0 / self.specificity
        for literal, truth in enumerate(literals):
            if output and truth:
                if self._rng.random() <= reward_probability:
                    self._increment(clause, literal)
            elif self._rng.random() <= penalty_probability:
                self._decrement(clause, literal)

    def _type_ii_feedback(self, clause: int, literals: tuple[bool, ...], output: bool) -> None:
        if not output:
            return
        for literal, truth in enumerate(literals):
            if not truth and not self.action_include(clause, literal):
                self._increment(clause, literal)

    def update(self, features: Sequence[bool | int], target: int | bool) -> None:
        target_value = int(self._require_binary(target))
        literals = self._literals(features)
        probability = self.standard_feedback_probability(features, target_value)
        for clause in range(self.number_of_clauses):
            polarity = 1 if clause % 2 == 0 else -1
            if self._rng.random() > probability:
                continue
            output = self.clause_output(clause, features, prediction=False)
            target_polarity = (
                (target_value == 1 and polarity == 1)
                or (target_value == 0 and polarity == -1)
            )
            if target_polarity:
                self._type_i_feedback(clause, literals, output)
            else:
                self._type_ii_feedback(clause, literals, output)

    def update_residual(
        self,
        features: Sequence[bool | int],
        target_probability: float,
        *,
        temperature: float,
        learning_rate: float,
    ) -> ResidualUpdateResult:
        """Update from a probability residual instead of the standard margin gate.

        This experimental controller retains the existing literal-level Type I
        and Type II rules. It replaces only the outer feedback gate and routes
        feedback by clause polarity so positive residuals raise the signed vote
        while negative residuals lower it.
        """

        if isinstance(target_probability, bool) or not isinstance(
            target_probability, (int, float)
        ):
            raise TypeError("target_probability must be a real number")
        target_value = float(target_probability)
        if not isfinite(target_value) or not 0.0 <= target_value <= 1.0:
            raise ValueError("target_probability must be finite and in [0, 1]")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be a real number")
        temperature_value = float(temperature)
        if not isfinite(temperature_value) or temperature_value <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if isinstance(learning_rate, bool) or not isinstance(
            learning_rate, (int, float)
        ):
            raise TypeError("learning_rate must be a real number")
        learning_rate_value = float(learning_rate)
        if not isfinite(learning_rate_value) or learning_rate_value <= 0.0:
            raise ValueError("learning_rate must be finite and positive")

        literals = self._literals(features)
        raw_score = self.raw_vote(features)
        scaled_score = raw_score / temperature_value
        if scaled_score >= 0.0:
            student_probability = 1.0 / (1.0 + exp(-scaled_score))
        else:
            scaled_exp = exp(scaled_score)
            student_probability = scaled_exp / (1.0 + scaled_exp)
        residual = target_value - student_probability
        feedback_probability = min(1.0, learning_rate_value * abs(residual))

        result = ResidualUpdateResult(
            raw_score=raw_score,
            student_probability=student_probability,
            target_probability=target_value,
            residual=residual,
            feedback_probability=feedback_probability,
        )
        if residual == 0.0:
            return result

        for clause in range(self.number_of_clauses):
            if self._rng.random() > feedback_probability:
                continue
            positive_clause = clause % 2 == 0
            output = self.clause_output(clause, features, prediction=False)
            use_type_i = (residual > 0.0 and positive_clause) or (
                residual < 0.0 and not positive_clause
            )
            if use_type_i:
                self._type_i_feedback(clause, literals, output)
            else:
                self._type_ii_feedback(clause, literals, output)
        return result

    def fit(
        self,
        rows: Sequence[Sequence[bool | int]],
        targets: Sequence[int | bool],
        *,
        epochs: int = 1,
    ) -> "ScalarBinaryTsetlinMachine":
        if len(rows) != len(targets):
            raise ValueError("rows and targets must have equal length")
        if epochs < 0:
            raise ValueError("epochs cannot be negative")
        for _ in range(epochs):
            for row, target in zip(rows, targets):
                self.update(row, target)
        return self

    def fit_literal_batch(
        self,
        batch: LiteralBatch,
        targets: Sequence[int | bool],
        *,
        epochs: int = 1,
    ) -> "ScalarBinaryTsetlinMachine":
        if batch.literal_count != self.number_of_features:
            raise ValueError("literal batch width does not match TM feature count")
        rows = [batch.row_values(index) for index in range(batch.row_count)]
        return self.fit(rows, targets, epochs=epochs)

    @staticmethod
    def _require_binary(value: object) -> bool:
        if value is True or value is False:
            return bool(value)
        if type(value) is int and value in (0, 1):
            return bool(value)
        raise ValueError("binary value must be bool or integer 0/1")

    @staticmethod
    def _check_snapshot_states(
        snapshot: TMSnapshot,
        *,
        number_of_clauses: int,
        number_of_features: int,
        states_per_action: int,
    ) -> None:
        if not isinstance(snapshot.states, (tuple, list)):
            raise ValueError("snapshot states must be a sequence")
        if len(snapshot.states) != number_of_clauses:
            raise ValueError("snapshot clause count does not match configuration")
        upper = 2 * states_per_action
        for row in snapshot.states:
            if not isinstance(row, (tuple, list)):
                raise ValueError("snapshot state row must be a sequence")
            if len(row) != 2 * number_of_features:
                raise ValueError("snapshot state row has wrong width")
            for state in row:
                if not isinstance(state, int) or isinstance(state, bool):
                    raise ValueError("snapshot TA state must be an integer")
                if not 1 <= state <= upper:
                    raise ValueError(
                        "snapshot TA state lies outside its two action regions"
                    )

    def snapshot(self) -> TMSnapshot:
        return TMSnapshot(
            schema_version=SNAPSHOT_SCHEMA_VERSION,
            number_of_clauses=self.number_of_clauses,
            number_of_features=self.number_of_features,
            states_per_action=self.states_per_action,
            specificity=self.specificity,
            threshold=self.threshold,
            states=tuple(tuple(row) for row in self._states),
            rng_state=copy.deepcopy(self._rng.getstate()),
        )

    def restore(self, snapshot: TMSnapshot) -> None:
        expected = (
            SNAPSHOT_SCHEMA_VERSION,
            self.number_of_clauses,
            self.number_of_features,
            self.states_per_action,
            self.specificity,
            self.threshold,
        )
        actual = (
            snapshot.schema_version,
            snapshot.number_of_clauses,
            snapshot.number_of_features,
            snapshot.states_per_action,
            snapshot.specificity,
            snapshot.threshold,
        )
        if actual != expected:
            raise ValueError("snapshot configuration does not match this machine")
        self._check_snapshot_states(
            snapshot,
            number_of_clauses=self.number_of_clauses,
            number_of_features=self.number_of_features,
            states_per_action=self.states_per_action,
        )
        self._states = [list(row) for row in snapshot.states]
        self._rng.setstate(copy.deepcopy(snapshot.rng_state))
