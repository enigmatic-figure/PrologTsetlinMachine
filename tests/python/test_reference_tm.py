from __future__ import annotations

import unittest

from prolog_tsetlin import (
    FeatureSchema,
    FieldKind,
    LiteralCatalog,
    ScalarBinaryTsetlinMachine,
)


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


if __name__ == "__main__":
    unittest.main()
