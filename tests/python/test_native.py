from __future__ import annotations

import random
import unittest

from prolog_tsetlin import (
    FixedBitBlock,
    LiteralBatch,
    MaskedThresholdKernel,
    NativePAKernel,
    NativePackedTsetlinMachine,
    NativeRuntimeError,
    PackedTMBackend,
    PortSemantic,
    ScalarBinaryTsetlinMachine,
    find_native_library,
    native_cpu_capabilities,
)


@unittest.skipUnless(find_native_library() is not None, "PTM native library is not built")
class NativeKernelTests(unittest.TestCase):
    def test_native_matches_python_oracle_for_both_shapes(self) -> None:
        native = NativePAKernel()
        for bit_count in (1024, 4096):
            with self.subTest(bit_count=bit_count):
                selection = FixedBitBlock(bit_count, PortSemantic.CLAUSE_OUTPUT)
                inputs = FixedBitBlock(bit_count, PortSemantic.CLAUSE_OUTPUT)
                for slot in (0, 63, 64, bit_count - 1):
                    selection.set(slot, True)
                for slot in (0, 64, bit_count - 1):
                    inputs.set(slot, True)
                expected = MaskedThresholdKernel(selection, 3).evaluate(inputs)
                actual = native.evaluate(inputs, selection, 3)
                self.assertEqual(actual, expected)

    def test_native_reports_invalid_threshold(self) -> None:
        native = NativePAKernel()
        selection = FixedBitBlock(1024, PortSemantic.TA_ACTION)
        inputs = FixedBitBlock(1024, PortSemantic.TA_ACTION)
        selection.set(4, True)
        with self.assertRaisesRegex(NativeRuntimeError, "minimum_true"):
            native.evaluate(inputs, selection, 2)


def configured_xor_machine() -> ScalarBinaryTsetlinMachine:
    machine = ScalarBinaryTsetlinMachine(
        4, 2, states_per_action=3, specificity=3.0, threshold=8, seed=41
    )
    for clause in range(4):
        for literal in range(4):
            machine.set_state(clause, literal, 3)
    for clause, literals in enumerate(((0, 3), (0, 2), (1, 2), (1, 3))):
        for literal in literals:
            machine.set_state(clause, literal, 4)
    return machine


@unittest.skipUnless(find_native_library() is not None, "PTM native library is not built")
class NativePackedTMTests(unittest.TestCase):
    def test_capabilities_and_forced_backends_are_consistent(self) -> None:
        capabilities = native_cpu_capabilities()
        self.assertTrue(capabilities.brand)
        self.assertTrue(capabilities.supports(PackedTMBackend.SCALAR))
        machine = configured_xor_machine()
        rows = [(0, 0), (0, 1), (1, 0), (1, 1)] * 4
        with NativePackedTsetlinMachine(machine.snapshot()) as native:
            scalar = native.evaluate_rows(rows, backend=PackedTMBackend.SCALAR)
            automatic = native.evaluate_rows(rows, backend="automatic")
            self.assertEqual(automatic.backend, native.selected_backend)
            self.assertEqual(scalar.prediction_mask, automatic.prediction_mask)
            self.assertEqual(scalar.scores, automatic.scores)
            self.assertEqual(scalar.clause_outputs, automatic.clause_outputs)
            for backend in (PackedTMBackend.AVX2, PackedTMBackend.AVX512):
                if capabilities.supports(backend):
                    accelerated = native.evaluate_rows(rows, backend=backend)
                    self.assertEqual(accelerated.backend, backend)
                    self.assertEqual(accelerated.scores, scalar.scores)
                    self.assertEqual(
                        accelerated.clause_outputs, scalar.clause_outputs
                    )

    def assert_matches_scalar(
        self,
        machine: ScalarBinaryTsetlinMachine,
        rows: list[tuple[int, ...]],
    ) -> None:
        with NativePackedTsetlinMachine(machine.snapshot()) as native:
            actual = native.evaluate_rows(rows)
            self.assertEqual(actual.predictions(len(rows)), tuple(machine.predict(rows)))
            for lane, row in enumerate(rows):
                self.assertEqual(actual.scores[lane], machine.score(row))
                for clause in range(machine.number_of_clauses):
                    self.assertEqual(
                        (actual.clause_outputs[clause] >> lane) & 1,
                        int(machine.clause_output(clause, row)),
                    )
                    self.assertEqual(
                        (actual.feedback_clause_outputs[clause] >> lane) & 1,
                        int(machine.clause_output(clause, row, prediction=False)),
                    )
            self.assertTrue(all(score == 0 for score in actual.scores[len(rows) :]))

    def test_native_packed_xor_matches_all_intermediates(self) -> None:
        rows = [(0, 0), (0, 1), (1, 0), (1, 1)] * 16
        self.assert_matches_scalar(configured_xor_machine(), rows)

    def test_randomized_partial_batches_match_scalar_oracle(self) -> None:
        random_source = random.Random(20260806)
        machine = ScalarBinaryTsetlinMachine(
            9, 11, states_per_action=31, specificity=3.9, threshold=7, seed=5
        )
        for clause in range(machine.number_of_clauses):
            for literal in range(machine.number_of_features * 2):
                machine.set_state(clause, literal, random_source.randint(1, 62))
        for row_count in (1, 17, 63, 64):
            with self.subTest(row_count=row_count):
                rows = [
                    tuple(
                        random_source.randrange(2)
                        for _ in range(machine.number_of_features)
                    )
                    for _ in range(row_count)
                ]
                self.assert_matches_scalar(machine, rows)

    def test_packed_input_and_closed_model_validation(self) -> None:
        native = NativePackedTsetlinMachine(configured_xor_machine().snapshot())
        with self.assertRaisesRegex(ValueError, "wrong width"):
            native.evaluate_packed((0,))
        with self.assertRaisesRegex(ValueError, "cannot exceed 64"):
            native.evaluate_rows([(0, 0)] * 65)
        native.close()
        native.close()
        with self.assertRaisesRegex(NativeRuntimeError, "closed"):
            native.evaluate_rows([(0, 0)])

    def test_class_i_literal_batch_feeds_native_packed_abi(self) -> None:
        batch = LiteralBatch(
            row_ids=("00", "01", "10", "11"),
            literal_ids=(101, 102),
            words=((0,), (2,), (1,), (3,)),
        )
        with NativePackedTsetlinMachine(configured_xor_machine().snapshot()) as native:
            result = native.evaluate_literal_batch(batch)
        self.assertEqual(result.predictions(4), (0, 1, 1, 0))


if __name__ == "__main__":
    unittest.main()
