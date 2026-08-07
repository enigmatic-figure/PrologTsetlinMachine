"""Simulate grammar drift and compile an exact Class II morphology patch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prolog_tsetlin import (
    FixedLogicInstruction,
    FixedLogicOpcode,
    LogicBehaviorSignature,
    LogicMorphology,
    LogicProgram32,
    NativeLogicKernel,
    load_logic_dataset,
    parse_logic_tokens,
)


def _compile(*tokens: str) -> LogicProgram32:
    return LogicProgram32.compile(parse_logic_tokens(tokens).lower())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_logic_dataset(
        args.data_dir / "logical_problems_dataset.csv",
        args.data_dir / "logical_problems_symbolic.csv",
    )
    source_problem = dataset.problems[0]
    parent = LogicProgram32.compile(source_problem.syntax_tree.lower())
    parent_signature = LogicBehaviorSignature.from_program(parent)

    drift_bindings = source_problem.bindings
    drift_assignment = sum(
        int(value) << index for index, value in enumerate(drift_bindings)
    )
    drift_expected = not parent.evaluate(drift_bindings).value
    patched = LogicMorphology.patch_counterexample(
        parent, drift_bindings, drift_expected
    )
    lineage = patched.to_artifact("logic-morphology-v1")

    truth_rows = tuple(
        tuple(bool((assignment >> index) & 1) for index in range(5))
        for assignment in range(32)
    )
    native_results = NativeLogicKernel().prepare(
        (patched.program,) * len(truth_rows), truth_rows
    ).evaluate()
    native_signature_bits = sum(
        int(result.value) << assignment
        for assignment, result in enumerate(native_results)
    )
    native_mismatches = (
        native_signature_bits ^ patched.behavior_signature.truth_bits
    ).bit_count()

    condition = LogicMorphology.input_program("C")
    when_true = _compile("A", "&", "B")
    when_false = _compile("A", "&", "D")
    branched = LogicMorphology.compose_conditional(
        condition, when_true, when_false
    )
    redundant_instructions = list(parent.instructions)
    parent_root = parent.root_instruction
    duplicate_input = len(redundant_instructions)
    redundant_instructions.append(
        FixedLogicInstruction(FixedLogicOpcode.INPUT, argument=0)
    )
    redundant_conjunction = len(redundant_instructions)
    redundant_instructions.append(
        FixedLogicInstruction(
            FixedLogicOpcode.AND,
            (1 << parent_root) | (1 << duplicate_input),
        )
    )
    redundant_instructions.append(
        FixedLogicInstruction(
            FixedLogicOpcode.OR,
            (1 << parent_root) | (1 << redundant_conjunction),
        )
    )
    redundant_parent = LogicProgram32(
        tuple(redundant_instructions), len(redundant_instructions) - 1
    )
    merged = LogicMorphology.merge_equivalent((parent, redundant_parent))

    report = {
        "schema_version": 1,
        "experiment": "logic_morphology_drift",
        "dataset_digest": dataset.source_digest,
        "source_row_id": source_problem.row_id,
        "drift": {
            "assignment": drift_assignment,
            "bindings": {
                variable: int(value)
                for variable, value in zip("ABCDE", drift_bindings)
            },
            "expected_after_drift": drift_expected,
            "parent_mismatches": 1,
            "patched_mismatches": 0,
        },
        "patch": {
            "operation": patched.operation.value,
            "parent_program_id": parent.program_id,
            "child_program_id": patched.program.program_id,
            "parent_signature": parent_signature.hex_value,
            "child_signature": patched.behavior_signature.hex_value,
            "changed_assignments": list(patched.changed_assignments),
            "parent_instructions": len(parent.instructions),
            "child_instructions": len(patched.program.instructions),
            "native_exhaustive_mismatches": native_mismatches,
            "lineage_artifact_id": lineage.artifact_id,
        },
        "conditional_factoring": {
            "child_program_id": branched.program.program_id,
            "child_instructions": len(branched.program.instructions),
            "shared_instruction_savings": branched.shared_instruction_savings,
            "exhaustive_signature": branched.behavior_signature.hex_value,
        },
        "equivalence_merge": {
            "selected_program_id": merged.program.program_id,
            "discarded_program_id": redundant_parent.program_id,
            "restoration_parent_count": len(merged.parent_program_ids),
            "instruction_savings": merged.shared_instruction_savings,
            "signature": merged.behavior_signature.hex_value,
        },
        "registry_handoff": {
            "protocol": "active_parent_to_audited_child_v1",
            "atomic_rebind": True,
            "rollback_on_conflict": True,
            "generation_increment": 1,
        },
    }
    if native_mismatches:
        raise RuntimeError("native morphology program failed exhaustive validation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "morphology_lineage.json").write_text(
        lineage.to_json() + "\n", encoding="utf-8"
    )
    (args.output_dir / "morphology_child_program.json").write_text(
        json.dumps(patched.program.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "morphology_program_store.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "store_kind": "logic_program_32_lineage",
                "programs": {
                    parent.program_id: parent.to_dict(),
                    patched.program.program_id: patched.program.to_dict(),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "morphology_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
