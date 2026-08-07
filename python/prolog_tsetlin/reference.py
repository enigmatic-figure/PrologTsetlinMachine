"""Deterministic scalar binary Tsetlin Machine semantic reference."""

from __future__ import annotations

import copy
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
        if specificity <= 1.0:
            raise ValueError("specificity must be greater than one")
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
            truth = bool(feature)
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

    def score(self, features: Sequence[bool | int]) -> int:
        total = 0
        for clause in range(self.number_of_clauses):
            if self.clause_output(clause, features):
                total += 1 if clause % 2 == 0 else -1
        return max(-self.threshold, min(self.threshold, total))

    def predict_one(self, features: Sequence[bool | int]) -> int:
        return int(self.score(features) > 0)

    def predict(self, rows: Iterable[Sequence[bool | int]]) -> list[int]:
        return [self.predict_one(row) for row in rows]

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
        target_value = int(target)
        if target_value not in (0, 1):
            raise ValueError("binary target must be zero or one")
        literals = self._literals(features)
        class_sum = self.score(features)
        for clause in range(self.number_of_clauses):
            polarity = 1 if clause % 2 == 0 else -1
            probability = (
                self.threshold
                + (1 - 2 * target_value) * polarity * class_sum
            ) / (2 * self.threshold)
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
        self._states = [list(row) for row in snapshot.states]
        self._rng.setstate(copy.deepcopy(snapshot.rng_state))

