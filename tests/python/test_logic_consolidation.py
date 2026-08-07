from __future__ import annotations

import itertools
import unittest

from prolog_tsetlin import (
    LogicEvaluatorArtifact,
    LogicProgram32,
    NativeLogicKernel,
    RestorationHandle,
    ValidationSignature,
    find_native_library,
    parse_logic_tokens,
)


NATIVE_LIBRARY = find_native_library()


class LogicConsolidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.program = parse_logic_tokens(
            (
                "(",
                "A",
                "&",
                "-",
                "B",
                ")",
                "if",
                "C",
                "$",
                "(",
                "D",
                "x",
                "E",
                ")",
            )
        )

    def test_fixed_program_matches_both_independent_evaluators(self) -> None:
        primitive = self.program.lower()
        compiled = LogicProgram32.compile(primitive)
        self.assertLessEqual(len(compiled.instructions), 32)
        for bindings in itertools.product((False, True), repeat=5):
            expected = self.program.evaluate(bindings)
            self.assertEqual(primitive.evaluate(bindings), expected)
            self.assertEqual(compiled.evaluate(bindings).value, expected)

    def test_evaluator_artifact_is_content_addressed_and_tamper_evident(self) -> None:
        artifact = LogicEvaluatorArtifact.create(
            mapping_version="logic-ast-slots-v1",
            validation_signature=ValidationSignature(
                "sha256:dataset", 5000, 0
            ),
            restoration_handle=RestorationHandle(
                1, "snapshot:logic-flat-tm-before-evaluator"
            ),
        )
        restored = LogicEvaluatorArtifact.from_dict(artifact.to_dict())
        self.assertEqual(restored, artifact)
        self.assertTrue(restored.verify_artifact_id())
        self.assertEqual(restored.to_dict()["input_shape"], "32x32")
        self.assertEqual(
            [binding["source_id"] for binding in restored.to_dict()["slot_bindings"]],
            [f"logic_binding:{variable}" for variable in "ABCDE"],
        )

        tampered = artifact.to_dict()
        tampered["kernel"]["program_capacity"] = 64
        with self.assertRaises(ValueError):
            LogicEvaluatorArtifact.from_dict(tampered)

    @unittest.skipUnless(NATIVE_LIBRARY, "PTM native library is not built")
    def test_prepared_native_batch_matches_fixed_program(self) -> None:
        compiled = LogicProgram32.compile(self.program.lower())
        bindings = tuple(itertools.product((False, True), repeat=5))
        batch = NativeLogicKernel(NATIVE_LIBRARY).prepare(
            (compiled,) * len(bindings), bindings
        )
        with self.assertRaisesRegex(RuntimeError, "not been executed"):
            batch.results()
        native = batch.evaluate()
        reference = tuple(compiled.evaluate(row) for row in bindings)
        self.assertEqual(native, reference)


if __name__ == "__main__":
    unittest.main()
