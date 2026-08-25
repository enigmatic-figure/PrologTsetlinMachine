from __future__ import annotations

import unittest

from prolog_tsetlin import (
    FeatureSchema,
    FieldKind,
    LiteralCatalog,
    ScalarBinaryTsetlinMachine,
)
from prolog_tsetlin.reference import contract_snapshot_equivalent_feature


def configured_xor_machine() -> ScalarBinaryTsetlinMachine:
    machine = ScalarBinaryTsetlinMachine(
        4,
        2,
        states_per_action=4,
        specificity=3.0,
        threshold=5,
        seed=9,
    )
    for clause in range(4):
        for literal in range(4):
            machine.set_state(clause, literal, 4)

    # Positive clauses x0 & !x1, !x0 & x1.
    machine.set_state(0, 0, 5)
    machine.set_state(0, 3, 5)
    machine.set_state(2, 1, 5)
    machine.set_state(2, 2, 5)
    # Negative clauses x0 & x1, !x0 & !x1.
    machine.set_state(1, 0, 5)
    machine.set_state(1, 2, 5)
    machine.set_state(3, 1, 5)
    machine.set_state(3, 3, 5)
    return machine


class ScalarTMTests(unittest.TestCase):
    def test_constructor_rejects_non_finite_specificity(self) -> None:
        for specificity in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(specificity=specificity):
                with self.assertRaisesRegex(
                    ValueError, "specificity must be finite and greater than one"
                ):
                    ScalarBinaryTsetlinMachine(
                        2, 2, specificity=specificity
                    )

    def test_class_i_output_feeds_xor_machine(self) -> None:
        schema = FeatureSchema.from_fields(
            x0=FieldKind.BOOLEAN,
            x1=FieldKind.BOOLEAN,
        )
        catalog = LiteralCatalog(schema)
        catalog.category_eq("x0", True)
        catalog.category_eq("x1", True)
        represented = catalog.encode(
            [
                {"x0": False, "x1": False},
                {"x0": False, "x1": True},
                {"x0": True, "x1": False},
                {"x0": True, "x1": True},
            ]
        )
        rows = [represented.ta.row_values(index) for index in range(4)]
        self.assertEqual(configured_xor_machine().predict(rows), [0, 1, 1, 0])

    def test_snapshot_restores_state_and_random_stream(self) -> None:
        machine = configured_xor_machine()
        snapshot = machine.snapshot()
        machine.update((True, False), 1)
        after_first_update = machine.snapshot()
        machine.restore(snapshot)
        machine.update((True, False), 1)
        after_replayed_update = machine.snapshot()
        self.assertEqual(after_first_update.states, after_replayed_update.states)
        self.assertEqual(after_first_update.rng_state, after_replayed_update.rng_state)

    def test_empty_clause_is_false_during_prediction(self) -> None:
        machine = ScalarBinaryTsetlinMachine(2, 2, states_per_action=4)
        for clause in range(2):
            for literal in range(4):
                machine.set_state(clause, literal, 4)
        self.assertFalse(machine.clause_output(0, (False, False)))
        self.assertEqual(machine.predict_one((False, False)), 0)

    def test_contract_equivalent_feature_merges_tas_and_preserves_rng(self) -> None:
        machine = ScalarBinaryTsetlinMachine(
            2,
            3,
            states_per_action=10,
            specificity=3.0,
            threshold=5,
            seed=9,
        )
        machine.set_state(0, 0, 8)
        machine.set_state(0, 1, 14)
        machine.set_state(0, 4, 17)
        machine.set_state(0, 5, 6)
        snapshot = machine.snapshot()

        contracted = contract_snapshot_equivalent_feature(snapshot, 0, 2)

        self.assertEqual(contracted.number_of_features, 2)
        self.assertEqual(contracted.states[0], (17, 14, *snapshot.states[0][2:4]))
        expected_second = list(snapshot.states[1])
        expected_second[0] = max(snapshot.states[1][0], snapshot.states[1][4])
        expected_second[1] = max(snapshot.states[1][1], snapshot.states[1][5])
        del expected_second[4:6]
        self.assertEqual(contracted.states[1], tuple(expected_second))
        self.assertEqual(contracted.rng_state, snapshot.rng_state)
        restored = ScalarBinaryTsetlinMachine(
            2,
            2,
            states_per_action=10,
            specificity=3.0,
            threshold=5,
            seed=0,
        )
        restored.restore(contracted)

    def test_contract_equivalent_feature_rejects_invalid_positions(self) -> None:
        snapshot = ScalarBinaryTsetlinMachine(2, 3, seed=4).snapshot()
        for survivor, removed in ((0, 0), (-1, 1), (0, 3)):
            with self.subTest(survivor=survivor, removed=removed):
                with self.assertRaisesRegex(ValueError, "positions"):
                    contract_snapshot_equivalent_feature(
                        snapshot, survivor, removed
                    )


if __name__ == "__main__":
    unittest.main()
