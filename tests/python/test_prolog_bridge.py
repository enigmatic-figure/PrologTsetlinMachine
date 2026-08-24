from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

from prolog_tsetlin import prolog_bridge
from prolog_tsetlin import (
    BooleanDecisionTree,
    DataType,
    DecisionTreeSearchProblem,
    FeatureSchema,
    FeatureTemplateCandidate,
    FeatureTemplateSearchProblem,
    FieldKind,
    GNUPrologSearch,
    GNUPrologThresholdSearch,
    FixedBitBlock,
    InputShape,
    NoThresholdSolution,
    NativePAKernel,
    NoDecisionTreeSolution,
    NoFeatureTemplateSolution,
    NoTAClauseSolution,
    PortSemantic,
    PrologBridgeError,
    RestorationHandle,
    SlotBinding,
    SourceKind,
    TAClauseSearchProblem,
    ThresholdSearchProblem,
    find_native_library,
)


GPROLOG = Path(
    os.environ.get("PTM_GPROLOG")
    or shutil.which("gprolog")
    or r"C:\GNU-Prolog\bin\gprolog.exe"
)


class BoundedStructureProblemTests(unittest.TestCase):
    def test_windows_prolog_environment_disables_gui_console_without_mutation(
        self,
    ) -> None:
        parent = {"LINEDIT": "gui=silent", "PTM_SENTINEL": "preserved"}

        child = prolog_bridge._prolog_process_environment(parent, windows=True)

        self.assertEqual(parent["LINEDIT"], "gui=silent")
        self.assertEqual(child["LINEDIT"], "gui=no")
        self.assertEqual(child["PTM_SENTINEL"], "preserved")

    def test_non_windows_prolog_environment_preserves_linedit(self) -> None:
        parent = {"LINEDIT": "ansi=no"}

        child = prolog_bridge._prolog_process_environment(parent, windows=False)

        self.assertEqual(child, parent)
        self.assertIsNot(child, parent)

    def test_bridge_rejects_a_child_output_flood_at_its_byte_ceiling(self) -> None:
        command = [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'x' * 1000000); "
            "sys.stdout.buffer.flush()",
        ]

        with self.assertRaisesRegex(PrologBridgeError, "output exceeded"):
            prolog_bridge._run_prolog_process(
                command,
                timeout_seconds=1.0,
                cancel=None,
            )

    def test_typed_template_candidate_must_match_registry_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "has type"):
            FeatureTemplateCandidate.create(
                field_name="temperature",
                template_id="numeric_threshold_v1",
                data_type=DataType.CATEGORICAL,
                parameters={"thresholds": [20.0]},
            )

    def test_ta_clause_and_tree_candidate_budgets_are_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidates"):
            TAClauseSearchProblem.create(
                feature_count=256,
                max_literals=3,
                examples=[{0}, set()],
                labels=[1, 0],
            )
        with self.assertRaisesRegex(ValueError, "candidates"):
            DecisionTreeSearchProblem.create(
                slot_count=5,
                max_depth=3,
                examples=[set()],
                labels=[0],
            )

    def test_decision_tree_problem_rejects_contradictory_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "contradictory"):
            DecisionTreeSearchProblem.create(
                slot_count=2,
                max_depth=2,
                examples=[{0}, {0}],
                labels=[0, 1],
            )

    def test_repair_passes_only_the_remaining_request_deadline(self) -> None:
        rows = [set(), {0}, {1}, {0, 1}]
        problem = DecisionTreeSearchProblem.create(
            slot_count=2,
            max_depth=2,
            examples=rows,
            labels=[0, 1, 1, 0],
        )
        first_guard = BooleanDecisionTree.node(
            0,
            BooleanDecisionTree.leaf(False),
            BooleanDecisionTree.leaf(True),
        )
        xor_guard = BooleanDecisionTree.node(
            0,
            BooleanDecisionTree.node(
                1,
                BooleanDecisionTree.leaf(False),
                BooleanDecisionTree.leaf(True),
            ),
            BooleanDecisionTree.node(
                1,
                BooleanDecisionTree.leaf(True),
                BooleanDecisionTree.leaf(False),
            ),
        )
        guards = iter((first_guard, xor_guard))
        observed_timeouts: list[float] = []

        def fake_search(*args: object, timeout_seconds: float, **kwargs: object):
            observed_timeouts.append(timeout_seconds)
            return mock.Mock(tree=next(guards))

        search = object.__new__(GNUPrologSearch)
        with (
            mock.patch.object(search, "search_decision_tree", side_effect=fake_search),
            mock.patch.object(
                prolog_bridge.time,
                "monotonic",
                side_effect=(10.0, 10.05, 10.2, 10.7, 10.9, 10.95, 10.99),
            ),
        ):
            result = search.repair_decision_tree(
                BooleanDecisionTree.leaf(False),
                problem,
                max_iterations=4,
                timeout_seconds=1.0,
            )

        self.assertEqual(result.mismatch_count, 0)
        self.assertEqual(len(observed_timeouts), 2)
        self.assertAlmostEqual(observed_timeouts[0], 0.8)
        self.assertAlmostEqual(observed_timeouts[1], 0.1)


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

    def test_search_selects_a_typed_feature_template(self) -> None:
        candidates = (
            FeatureTemplateCandidate.create(
                field_name="status",
                template_id="categorical_v1",
                data_type=DataType.CATEGORICAL,
                parameters={"categories": ["cold"]},
            ),
            FeatureTemplateCandidate.create(
                field_name="status",
                template_id="categorical_v1",
                data_type=DataType.CATEGORICAL,
                parameters={"categories": ["hot"]},
            ),
        )
        problem = FeatureTemplateSearchProblem.create(
            candidates=candidates,
            labels=[0, 1, 1, 0],
            coverage=[
                [0, 0, 1, 0],
                [0, 1, 1, 0],
            ],
        )
        result = GNUPrologSearch(GPROLOG).search_feature_template(problem)
        self.assertEqual(result.candidate_index, 1)
        self.assertEqual(result.dataset_digest, problem.dataset_digest())
        schema = FeatureSchema.from_fields(status=FieldKind.CATEGORY)
        catalog, generated = result.create_catalog(schema)
        self.assertEqual(len(catalog.literals), 1)
        self.assertEqual(len(generated["status"]), 1)

    def test_search_emits_a_signed_ta_clause_configuration(self) -> None:
        rows = [set(), {0}, {1}, {0, 1}]
        problem = TAClauseSearchProblem.create(
            feature_count=2,
            max_literals=2,
            examples=rows,
            labels=[0, 1, 0, 0],
        )
        result = GNUPrologSearch(GPROLOG).search_ta_clause(problem)
        self.assertEqual(result.included_literals, (0, 3))
        configuration = result.to_ta_configuration(
            problem,
            states_per_action=100,
            specificity=3.0,
            threshold=10,
        )
        self.assertEqual(configuration.clause_configs[0].included_literals, (0, 3))
        self.assertEqual(configuration.final_accuracy, 1.0)
        self.assertEqual(configuration.metadata["analysis_type"], "prolog_ta_clause_configuration")

    def test_searches_xor_tree_and_lowers_to_fixed_logic(self) -> None:
        rows = [set(), {0}, {1}, {0, 1}]
        problem = DecisionTreeSearchProblem.create(
            slot_count=2,
            max_depth=2,
            examples=rows,
            labels=[0, 1, 1, 0],
        )
        result = GNUPrologSearch(GPROLOG).search_decision_tree(problem)
        self.assertEqual(result.tree.depth, 2)
        self.assertEqual(
            tuple(result.tree.evaluate(row) for row in rows),
            (False, True, True, False),
        )
        program = result.tree.to_logic_program()
        self.assertEqual(
            tuple(
                program.evaluate((x0, x1, 0, 0, 0)).value
                for x0, x1 in ((0, 0), (1, 0), (0, 1), (1, 1))
            ),
            (False, True, True, False),
        )

    def test_counterexample_guided_repair_synthesizes_a_guard(self) -> None:
        rows = [set(), {0}, {1}, {0, 1}]
        problem = DecisionTreeSearchProblem.create(
            slot_count=2,
            max_depth=2,
            examples=rows,
            labels=[0, 1, 1, 0],
        )
        result = GNUPrologSearch(GPROLOG).repair_decision_tree(
            BooleanDecisionTree.leaf(False),
            problem,
            max_iterations=4,
        )
        self.assertEqual(result.mismatches_before, 2)
        self.assertEqual(result.mismatch_count, 0)
        self.assertEqual(len(result.counterexamples), 4)
        self.assertEqual(
            tuple(result.evaluate(row) for row in rows),
            (False, True, True, False),
        )
        program = result.to_logic_program()
        self.assertTrue(program.evaluate((1, 0, 0, 0, 0)).value)
        self.assertFalse(program.evaluate((1, 1, 0, 0, 0)).value)

    def test_structure_searches_report_typed_no_solution_results(self) -> None:
        search = GNUPrologSearch(GPROLOG)
        candidate = FeatureTemplateCandidate.create(
            field_name="status",
            template_id="categorical_v1",
            data_type=DataType.CATEGORICAL,
            parameters={"categories": ["hot"]},
        )
        with self.assertRaises(NoFeatureTemplateSolution):
            search.search_feature_template(
                FeatureTemplateSearchProblem.create(
                    candidates=[candidate],
                    labels=[0, 1],
                    coverage=[[0, 0]],
                )
            )
        rows = [set(), {0}, {1}, {0, 1}]
        with self.assertRaises(NoTAClauseSolution):
            search.search_ta_clause(
                TAClauseSearchProblem.create(
                    feature_count=2,
                    max_literals=2,
                    examples=rows,
                    labels=[0, 1, 1, 0],
                )
            )
        with self.assertRaises(NoDecisionTreeSolution):
            search.search_decision_tree(
                DecisionTreeSearchProblem.create(
                    slot_count=2,
                    max_depth=1,
                    examples=rows,
                    labels=[0, 1, 1, 0],
                )
            )


if __name__ == "__main__":
    unittest.main()
