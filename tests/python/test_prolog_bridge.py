from __future__ import annotations

import os
import unittest
from pathlib import Path

from prolog_tsetlin import (
    GNUPrologThresholdSearch,
    FixedBitBlock,
    InputShape,
    NoThresholdSolution,
    NativePAKernel,
    PortSemantic,
    RestorationHandle,
    SlotBinding,
    SourceKind,
    ThresholdSearchProblem,
    find_native_library,
)


GPROLOG = Path(os.environ.get("PTM_GPROLOG", r"C:\GNU-Prolog\bin\gprolog.exe"))


@unittest.skipUnless(GPROLOG.is_file(), "GNU Prolog is not installed")
class GNUPrologBridgeTests(unittest.TestCase):
    def test_search_lowers_exact_or_rule_to_pa_artifact(self) -> None:
        problem = ThresholdSearchProblem.create(
            slot_count=3,
            max_selected=3,
            positive_examples=[{0}, {1}, {0, 1}, {0, 2}],
            negative_examples=[set(), {2}],
        )
        artifact = GNUPrologThresholdSearch(GPROLOG).search_artifact(
            problem,
            input_shape=InputShape.PA_32X32,
            port_semantic=PortSemantic.TA_ACTION,
            mapping_version="test-map-v1",
            slot_bindings=[
                SlotBinding(0, SourceKind.TA, "ta-0", (100,)),
                SlotBinding(1, SourceKind.TA, "ta-1", (101,)),
                SlotBinding(2, SourceKind.TA, "ta-2", (102,)),
            ],
            restoration_handle=RestorationHandle(1, "snapshot:test"),
            timeout_seconds=60.0,
        )
        self.assertEqual(artifact.payload.selected_slots, (0, 1))
        self.assertEqual(artifact.payload.minimum_true, 1)
        self.assertEqual(artifact.validation_signature.example_count, 6)
        self.assertEqual(artifact.validation_signature.mismatch_count, 0)
        self.assertTrue(artifact.verify_artifact_id())
        if find_native_library() is not None:
            native = NativePAKernel()
            positive = FixedBitBlock(1024, PortSemantic.TA_ACTION)
            positive.set(1, True)
            negative = FixedBitBlock(1024, PortSemantic.TA_ACTION)
            negative.set(2, True)
            self.assertTrue(native.evaluate_artifact(artifact, positive).value)
            self.assertFalse(native.evaluate_artifact(artifact, negative).value)

    def test_non_threshold_xor_is_reported_as_unsolved(self) -> None:
        problem = ThresholdSearchProblem.create(
            slot_count=2,
            max_selected=2,
            positive_examples=[{0}, {1}],
            negative_examples=[set(), {0, 1}],
        )
        with self.assertRaises(NoThresholdSolution):
            GNUPrologThresholdSearch(GPROLOG).search(
                problem, timeout_seconds=60.0
            )

    def test_combinatorial_search_budget_is_enforced_before_launch(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidates"):
            ThresholdSearchProblem.create(
                slot_count=256,
                max_selected=3,
                positive_examples=[{0}],
                negative_examples=[set()],
            )


if __name__ == "__main__":
    unittest.main()
