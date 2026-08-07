from __future__ import annotations

import itertools
import unittest

from prolog_tsetlin import (
    FixedLogicInstruction,
    FixedLogicOpcode,
    LogicMorphology,
    LogicMorphologyArtifact,
    LogicProgram32,
    MorphologyOperation,
    parse_logic_tokens,
)


def compile_tokens(*tokens: str) -> LogicProgram32:
    return LogicProgram32.compile(parse_logic_tokens(tokens).lower())


class LogicMorphologyTests(unittest.TestCase):
    def test_counterexample_patches_change_exactly_one_assignment(self) -> None:
        base = LogicMorphology.input_program("A")
        false_row = (False, True, False, True, False)
        promoted = LogicMorphology.patch_counterexample(base, false_row, True)
        self.assertEqual(promoted.operation, MorphologyOperation.PATCH_TRUE)
        self.assertEqual(promoted.changed_assignments, (10,))
        for row in itertools.product((False, True), repeat=5):
            expected = bool(row[0]) or row == false_row
            self.assertEqual(promoted.program.evaluate(row).value, expected)

        true_row = (True, False, True, False, True)
        suppressed = LogicMorphology.patch_counterexample(base, true_row, False)
        self.assertEqual(suppressed.operation, MorphologyOperation.PATCH_FALSE)
        self.assertEqual(suppressed.changed_assignments, (21,))
        for row in itertools.product((False, True), repeat=5):
            expected = bool(row[0]) and row != true_row
            self.assertEqual(suppressed.program.evaluate(row).value, expected)

    def test_specialize_generalize_and_noop_obey_monotone_contracts(self) -> None:
        a = LogicMorphology.input_program("A")
        b = LogicMorphology.input_program("B")
        specialized = LogicMorphology.specialize(a, b)
        generalized = LogicMorphology.generalize(a, b)
        for row in itertools.product((False, True), repeat=5):
            self.assertEqual(
                specialized.program.evaluate(row).value, row[0] and row[1]
            )
            self.assertEqual(
                generalized.program.evaluate(row).value, row[0] or row[1]
            )
        noop = LogicMorphology.patch_counterexample(a, (True,) * 5, True)
        self.assertEqual(noop.operation, MorphologyOperation.NOOP)
        self.assertEqual(noop.program.program_id, a.program_id)

    def test_conditional_composition_factors_shared_subgraphs(self) -> None:
        condition = LogicMorphology.input_program("C")
        when_true = compile_tokens("A", "&", "B")
        when_false = compile_tokens("A", "&", "D")
        composed = LogicMorphology.compose_conditional(
            condition, when_true, when_false
        )
        self.assertGreater(composed.shared_instruction_savings, 0)
        for row in itertools.product((False, True), repeat=5):
            expected = (row[0] and row[1]) if row[2] else (row[0] and row[3])
            self.assertEqual(composed.program.evaluate(row).value, expected)

    def test_equivalence_merge_chooses_smallest_restorable_program(self) -> None:
        minimal = LogicMorphology.input_program("A")
        redundant = LogicProgram32(
            (
                FixedLogicInstruction(FixedLogicOpcode.INPUT, argument=0),
                FixedLogicInstruction(FixedLogicOpcode.INPUT, argument=1),
                FixedLogicInstruction(FixedLogicOpcode.AND, 0b11),
                FixedLogicInstruction(FixedLogicOpcode.OR, 0b0101),
            ),
            3,
        )
        merged = LogicMorphology.merge_equivalent((redundant, minimal))
        self.assertEqual(merged.operation, MorphologyOperation.EQUIVALENCE_MERGE)
        self.assertEqual(merged.program.program_id, minimal.program_id)
        with self.assertRaisesRegex(ValueError, "distinct"):
            LogicMorphology.merge_equivalent(
                (minimal, LogicMorphology.input_program("B"))
            )

    def test_lineage_artifact_is_content_addressed_and_tamper_evident(self) -> None:
        result = LogicMorphology.patch_counterexample(
            LogicMorphology.input_program("A"),
            (False, False, False, False, False),
            True,
        )
        artifact = result.to_artifact("logic-morphology-v1")
        restored = LogicMorphologyArtifact.from_dict(artifact.to_dict())
        self.assertEqual(restored, artifact)
        self.assertTrue(restored.verify_artifact_id())
        self.assertEqual(restored.parent_program_ids, result.parent_program_ids)

        tampered = artifact.to_dict()
        tampered["changed_assignments"] = [1]
        with self.assertRaises(ValueError):
            LogicMorphologyArtifact.from_dict(tampered)


if __name__ == "__main__":
    unittest.main()
