"""Compile and shadow-audit the Logic AST evaluator as a Class II artifact."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

from prolog_tsetlin import (
    LogicEvaluatorArtifact,
    LogicProgram32,
    NativeLogicKernel,
    RestorationHandle,
    ValidationSignature,
    load_logic_dataset,
    stratified_logic_split,
)


SPLIT_SEED = 20260806
FLAT_TM_FRONTIER = 0.638


def _accuracy(
    predictions: tuple[bool, ...],
    targets: tuple[int, ...],
    indices: tuple[int, ...],
) -> float:
    return sum(predictions[index] == bool(targets[index]) for index in indices) / len(indices)


def _binding_bits(bindings: tuple[bool, ...]) -> int:
    return sum(int(value) << index for index, value in enumerate(bindings))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    dataset = load_logic_dataset(
        args.data_dir / "logical_problems_dataset.csv",
        args.data_dir / "logical_problems_symbolic.csv",
    )
    split = stratified_logic_split(dataset, seed=SPLIT_SEED)
    targets = tuple(problem.target for problem in dataset.problems)
    bindings = tuple(problem.bindings for problem in dataset.problems)

    compile_start = time.perf_counter()
    primitive_graphs = tuple(problem.syntax_tree.lower() for problem in dataset.problems)
    programs = tuple(LogicProgram32.compile(graph) for graph in primitive_graphs)
    compile_seconds = time.perf_counter() - compile_start

    ast_predictions = tuple(
        problem.syntax_tree.evaluate(problem.bindings) for problem in dataset.problems
    )
    primitive_predictions = tuple(
        graph.evaluate(problem.bindings)
        for graph, problem in zip(primitive_graphs, dataset.problems)
    )
    fixed_predictions = tuple(
        program.evaluate(problem.bindings).value
        for program, problem in zip(programs, dataset.problems)
    )

    native_kernel = NativeLogicKernel()
    prepared = native_kernel.prepare(programs, bindings)
    native_results = prepared.evaluate()
    native_predictions = tuple(result.value for result in native_results)

    native_seconds = 0.0
    for _ in range(args.repeats):
        started = time.perf_counter()
        prepared.execute()
        native_seconds += time.perf_counter() - started
    materialize_started = time.perf_counter()
    measured = prepared.results()
    materialize_seconds = time.perf_counter() - materialize_started
    checksum = sum(result.value for result in measured) * args.repeats

    unique_programs = {program.program_id: program for program in programs}
    truth_rows = tuple(itertools.product((False, True), repeat=5))
    perturbation_programs = tuple(
        program for program in unique_programs.values() for _ in truth_rows
    )
    perturbation_bindings = truth_rows * len(unique_programs)
    perturbation_batch = native_kernel.prepare(
        perturbation_programs, perturbation_bindings
    )
    perturbation_native = perturbation_batch.evaluate()
    perturbation_mismatches = sum(
        native.value != program.evaluate(row).value
        for native, program, row in zip(
            perturbation_native, perturbation_programs, perturbation_bindings
        )
    )

    disagreements = {
        "typed_ast_vs_label": sum(
            prediction != bool(target)
            for prediction, target in zip(ast_predictions, targets)
        ),
        "primitive_graph_vs_label": sum(
            prediction != bool(target)
            for prediction, target in zip(primitive_predictions, targets)
        ),
        "fixed_python_vs_label": sum(
            prediction != bool(target)
            for prediction, target in zip(fixed_predictions, targets)
        ),
        "native_vs_label": sum(
            prediction != bool(target)
            for prediction, target in zip(native_predictions, targets)
        ),
        "native_vs_fixed_python": sum(
            native != fixed
            for native, fixed in zip(native_predictions, fixed_predictions)
        ),
    }
    artifact = LogicEvaluatorArtifact.create(
        mapping_version="logic-ast-program32-v1",
        validation_signature=ValidationSignature(
            dataset.source_digest,
            len(dataset.problems),
            disagreements["native_vs_label"],
        ),
        restoration_handle=RestorationHandle(
            1,
            "retrain:logic-flat-baseline-v1:seed-20260806:commit-74e455f",
        ),
    )

    state_content = {
        "schema_version": 1,
        "state_kind": "logic_program_32_matrix",
        "dataset_digest": dataset.source_digest,
        "unique_programs": {
            program_id: program.to_dict()
            for program_id, program in sorted(unique_programs.items())
        },
        "rows": [
            {
                "row_id": problem.row_id,
                "program_id": program.program_id,
                "binding_bits": _binding_bits(problem.bindings),
            }
            for problem, program in zip(dataset.problems, programs)
        ],
    }
    state_id = "sha256:" + hashlib.sha256(
        json.dumps(
            state_content,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    positive_count = sum(targets)
    expected_checksum = positive_count * args.repeats
    if checksum != expected_checksum:
        raise RuntimeError("prepared native benchmark checksum changed")

    report = {
        "schema_version": 1,
        "experiment": "logic_class_ii_consolidation",
        "dataset_digest": dataset.source_digest,
        "split_seed": SPLIT_SEED,
        "artifact_id": artifact.artifact_id,
        "compiled_state_id": state_id,
        "rows": len(dataset.problems),
        "training_rows": len(split.train_indices),
        "shadow_rows": len(split.evaluation_indices),
        "accuracy": {
            "training": _accuracy(native_predictions, targets, split.train_indices),
            "shadow_evaluation": _accuracy(
                native_predictions, targets, split.evaluation_indices
            ),
            "overall": _accuracy(
                native_predictions, targets, tuple(range(len(targets)))
            ),
        },
        "disagreements": disagreements,
        "shadow_decision": (
            "activate" if disagreements["native_vs_label"] == 0 else "reject"
        ),
        "program_shape": {
            "capacity": 32,
            "minimum_instructions": min(len(program.instructions) for program in programs),
            "maximum_instructions": max(len(program.instructions) for program in programs),
            "mean_instructions": sum(len(program.instructions) for program in programs)
            / len(programs),
            "unique_programs": len(unique_programs),
        },
        "candidate_maturity": {
            "candidate_kind": "specified_operator_semantics",
            "precision": 1.0 - disagreements["native_vs_label"] / len(targets),
            "support": len(targets),
            "recent_state_movement": 0.0,
            "feedback_rate": 0.0,
            "reuse_count": len(targets),
            "exhaustive_binding_cases": len(perturbation_programs),
            "exhaustive_binding_mismatches": perturbation_mismatches,
            "semantic_conformance_error_rate": (
                perturbation_mismatches / len(perturbation_programs)
            ),
        },
        "gap": {
            "flat_tm_frontier": FLAT_TM_FRONTIER,
            "class_ii_accuracy": _accuracy(
                native_predictions, targets, split.evaluation_indices
            ),
            "residual_error": 1.0
            - _accuracy(native_predictions, targets, split.evaluation_indices),
        },
        "timing": {
            "compile_seconds": compile_seconds,
            "prepared_native_repetitions": args.repeats,
            "prepared_native_seconds": native_seconds,
            "prepared_native_programs_per_second": (
                len(programs) * args.repeats / native_seconds
            ),
            "single_result_materialization_seconds": materialize_seconds,
            "checksum": checksum,
        },
    }

    state = dict(state_content)
    state["state_id"] = state_id

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logic_evaluator_artifact.json").write_text(
        artifact.to_json() + "\n", encoding="utf-8"
    )
    (args.output_dir / "logic_program_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "logic_consolidation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
