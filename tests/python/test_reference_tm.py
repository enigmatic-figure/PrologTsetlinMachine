from __future__ import annotations

import unittest

from prolog_tsetlin import (
    FeatureSchema,
    FieldKind,
    LiteralCatalog,
    ScalarBinaryTsetlinMachine,
)
from prolog_tsetlin.reference import contract_snapshot_equivalent_feature


class FeedbackRecordingMachine(ScalarBinaryTsetlinMachine):
    def __init__(self) -> None:
        super().__init__(
            4,
            2,
            states_per_action=4,
            specificity=3.0,
            threshold=5,
            seed=9,
        )
        for clause in range(4):
            for literal in range(4):
                self.set_state(clause, literal, 4)
        self.feedback: list[tuple[str, int]] = []

    def _type_i_feedback(
        self, clause: int, literals: tuple[bool, ...], output: bool
    ) -> None:
        self.feedback.append(("I", clause))

    def _type_ii_feedback(
        self, clause: int, literals: tuple[bool, ...], output: bool
    ) -> None:
        self.feedback.append(("II", clause))


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


def configured_saturated_vote_machine() -> ScalarBinaryTsetlinMachine:
    machine = ScalarBinaryTsetlinMachine(
        40,
        1,
        states_per_action=4,
        specificity=3.0,
        threshold=10,
        seed=9,
    )
    for clause in range(40):
        for literal in range(2):
            machine.set_state(clause, literal, 4)
        # On input 1, every positive clause fires and every negative clause fails.
        machine.set_state(clause, 0 if clause % 2 == 0 else 1, 5)
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

    def test_raw_vote_is_distinct_from_margin_clipped_score(self) -> None:
        machine = configured_saturated_vote_machine()
        self.assertEqual(machine.raw_vote((True,)), 20)
        self.assertEqual(machine.score((True,)), 10)
        self.assertEqual(machine.predict_one((True,)), 1)

    def test_standard_feedback_gate_is_shared_across_clause_polarities(self) -> None:
        machine = ScalarBinaryTsetlinMachine(
            12,
            1,
            states_per_action=4,
            specificity=3.0,
            threshold=10,
            seed=9,
        )
        for clause in range(12):
            for literal in range(2):
                machine.set_state(clause, literal, 4)
            machine.set_state(clause, 0 if clause % 2 == 0 else 1, 5)
        self.assertEqual(machine.raw_vote((True,)), 6)
        self.assertAlmostEqual(
            machine.standard_feedback_probability((True,), 1), 0.2
        )
        self.assertAlmostEqual(
            machine.standard_feedback_probability((True,), 0), 0.8
        )

    def test_residual_probability_uses_unclipped_vote_direction(self) -> None:
        machine = configured_saturated_vote_machine()
        result = machine.update_residual(
            (True,),
            0.98,
            temperature=3.0,
            learning_rate=1.0,
        )
        self.assertEqual(result.raw_score, 20)
        self.assertGreater(result.student_probability, 0.998)
        self.assertLess(result.residual, 0.0)

    def test_residual_feedback_routes_by_residual_and_clause_polarity(self) -> None:
        positive = FeedbackRecordingMachine()
        positive_result = positive.update_residual(
            (False, False),
            1.0,
            temperature=1.0,
            learning_rate=2.0,
        )
        self.assertEqual(
            positive.feedback,
            [("I", 0), ("II", 1), ("I", 2), ("II", 3)],
        )
        self.assertEqual(positive_result.raw_score, 0)
        self.assertEqual(positive_result.student_probability, 0.5)
        self.assertEqual(positive_result.base_feedback_probability, 1.0)
        self.assertEqual(positive_result.feedback_scale, 1.0)
        self.assertEqual(positive_result.feedback_probability, 1.0)

        negative = FeedbackRecordingMachine()
        negative_result = negative.update_residual(
            (False, False),
            0.0,
            temperature=1.0,
            learning_rate=2.0,
        )
        self.assertEqual(
            negative.feedback,
            [("II", 0), ("I", 1), ("II", 2), ("I", 3)],
        )
        self.assertEqual(negative_result.residual, -0.5)

    def test_zero_residual_does_not_advance_state_or_rng(self) -> None:
        machine = FeedbackRecordingMachine()
        before = machine.snapshot()
        result = machine.update_residual(
            (False, False),
            0.5,
            temperature=3.0,
            learning_rate=2.0,
        )
        self.assertEqual(result.residual, 0.0)
        self.assertEqual(result.feedback_probability, 0.0)
        self.assertEqual(machine.snapshot(), before)

    def test_residual_feedback_scale_changes_only_the_outer_probability(self) -> None:
        machine = FeedbackRecordingMachine()
        result = machine.update_residual(
            (False, False),
            1.0,
            temperature=1.0,
            learning_rate=1.0,
            feedback_scale=0.2,
        )

        self.assertEqual(result.student_probability, 0.5)
        self.assertEqual(result.target_probability, 1.0)
        self.assertEqual(result.residual, 0.5)
        self.assertEqual(result.base_feedback_probability, 0.5)
        self.assertEqual(result.feedback_scale, 0.2)
        self.assertAlmostEqual(result.feedback_probability, 0.1)

    def test_residual_feedback_validates_controller_parameters(self) -> None:
        machine = FeedbackRecordingMachine()
        invalid_cases = (
            (
                {"target_probability": -0.1, "temperature": 1.0, "learning_rate": 1.0},
                ValueError,
            ),
            (
                {"target_probability": 1.1, "temperature": 1.0, "learning_rate": 1.0},
                ValueError,
            ),
            (
                {
                    "target_probability": float("nan"),
                    "temperature": 1.0,
                    "learning_rate": 1.0,
                },
                ValueError,
            ),
            (
                {"target_probability": 0.5, "temperature": 0.0, "learning_rate": 1.0},
                ValueError,
            ),
            (
                {"target_probability": 0.5, "temperature": 1.0, "learning_rate": 0.0},
                ValueError,
            ),
            (
                {
                    "target_probability": 0.5,
                    "temperature": 1.0,
                    "learning_rate": 1.0,
                    "feedback_scale": -0.1,
                },
                ValueError,
            ),
            (
                {
                    "target_probability": 0.5,
                    "temperature": 1.0,
                    "learning_rate": 1.0,
                    "feedback_scale": 1.1,
                },
                ValueError,
            ),
            (
                {
                    "target_probability": 0.5,
                    "temperature": 1.0,
                    "learning_rate": 1.0,
                    "feedback_scale": True,
                },
                TypeError,
            ),
            (
                {"target_probability": True, "temperature": 1.0, "learning_rate": 1.0},
                TypeError,
            ),
        )
        for arguments, error in invalid_cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(error):
                    machine.update_residual((False, False), **arguments)

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
