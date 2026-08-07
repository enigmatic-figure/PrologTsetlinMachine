from __future__ import annotations

import itertools
import unittest

from prolog_tsetlin import (
    FixedBitBlock,
    MaskedThresholdKernel,
    PortSemantic,
    fredkin_gate,
    fredkin_literal_condition,
)


class FredkinTests(unittest.TestCase):
    def test_gate_is_bijective_and_conservative(self) -> None:
        outputs = set()
        for inputs in itertools.product((False, True), repeat=3):
            result = fredkin_gate(*inputs)
            outputs.add(result.as_tuple())
            restored = fredkin_gate(result.control, result.first, result.second)
            self.assertEqual(restored.as_tuple(), inputs)
            self.assertEqual(sum(inputs), sum(result.as_tuple()))
        self.assertEqual(len(outputs), 8)

    def test_literal_condition_retains_garbage(self) -> None:
        excluded_false = fredkin_literal_condition(False, False)
        included_false = fredkin_literal_condition(True, False)
        included_true = fredkin_literal_condition(True, True)
        self.assertTrue(excluded_false.first)
        self.assertFalse(excluded_false.second)
        self.assertFalse(included_false.first)
        self.assertTrue(included_false.second)
        self.assertTrue(included_true.first)


class PAKernelTests(unittest.TestCase):
    def test_masked_threshold_and_diagnostics(self) -> None:
        selection = FixedBitBlock(1024, PortSemantic.TA_ACTION)
        for slot in (1, 7, 70):
            selection.set(slot, True)
        inputs = FixedBitBlock(1024, PortSemantic.TA_ACTION)
        inputs.set(1, True)
        inputs.set(70, True)

        result = MaskedThresholdKernel(selection, minimum_true=2).evaluate(inputs)
        self.assertTrue(result.value)
        self.assertEqual(result.matched_count, 2)
        self.assertEqual(result.selected_count, 3)
        self.assertEqual(sum(word.bit_count() for word in result.missing_words), 1)


if __name__ == "__main__":
    unittest.main()

