"""Coalesced Tsetlin Machine — shared clause pool with per-class weights (scalar reference)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from prolog_tsetlin.reference import ScalarBinaryTsetlinMachine, TMSnapshot

# Reuse TA logic from ScalarBinaryTsetlinMachine but coalesce clauses.


@dataclass
class CoalescedTsetlinMachine:
    """Minimal CoTM for GraphTM: shared clauses, per-class weights.

    This is intentionally scalar and simple. Each clause `j` has weight
    `weights[j][c]` for class `c`. Voting is weighted sum per class.
    """

    number_of_clauses: int
    number_of_features: int
    number_of_classes: int = 2
    states_per_action: int = 100
    specificity: float = 3.9
    threshold: int = 15
    seed: int = 1
    weights: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.number_of_clauses <= 0 or self.number_of_features <= 0:
            raise ValueError("clauses/features must be positive")
        if self.number_of_classes < 2:
            raise ValueError("classes >=2")
        self._rng = random.Random(self.seed)
        # reuse underlying binary TMs per class? Instead single pool
        self._states = [
            [self.states_per_action + self._rng.randrange(2) for _ in range(2 * self.number_of_features)]
            for _ in range(self.number_of_clauses)
        ]
        if not self.weights:
            # init weights to 1 per class
            self.weights = [[1 for _ in range(self.number_of_classes)] for _ in range(self.number_of_clauses)]

    def clause_output(self, clause: int, features: list[bool]) -> bool:
        # same logic as ScalarBinaryTsetlinMachine.clause_output (empty clause false)
        included = False
        for lit, truth in enumerate([features[i // 2] if i % 2 == 0 else not features[i // 2] for i in range(2 * self.number_of_features)]):
            # inefficient but clear: use states
            state = self._states[clause][lit]
            if state > self.states_per_action:
                included = True
                if not truth:
                    return False
        return included

    def predict(self, features: list[bool]) -> int:
        # weighted voting per class
        scores = [0] * self.number_of_classes
        for j in range(self.number_of_clauses):
            out = self.clause_output(j, features)
            if out:
                # clause polarity not used; weight per class
                # Simple: even clauses favour class 0, odd favour class 1? No, use weights
                # Use weights[j][c] as contribution if clause true
                for c in range(self.number_of_classes):
                    scores[c] += self.weights[j][c]
        # argmax, tie -> 0
        best = 0
        for c in range(1, self.number_of_classes):
            if scores[c] > scores[best]:
                best = c
        return best

    def snapshot(self):
        return {
            "states": tuple(tuple(r) for r in self._states),
            "weights": tuple(tuple(w) for w in self.weights),
            "seed": self.seed,
        }
