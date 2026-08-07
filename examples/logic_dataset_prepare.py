"""Prepare leakage-safe binary baselines for the paired Logic dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from prolog_tsetlin import (
    LogicEncoding,
    collision_report,
    encode_logic_split,
    evaluation_signature_report,
    load_logic_dataset,
    stratified_logic_split,
)


def write_binary_rows(
    path: Path,
    rows: Sequence[Sequence[int]],
    targets: Sequence[int],
) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="ascii", newline="\n") as stream:
        for row, target in zip(rows, targets):
            line = " ".join(str(int(value)) for value in (*row, target)) + "\n"
            stream.write(line)
            digest.update(line.encode("ascii"))
    return f"sha256:{digest.hexdigest()}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/Logic"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/logic-dataset"))
    parser.add_argument("--evaluation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--encodings",
        nargs="+",
        choices=[encoding.value for encoding in LogicEncoding],
        default=[encoding.value for encoding in LogicEncoding],
    )
    arguments = parser.parse_args()

    dataset = load_logic_dataset(
        arguments.data_dir / "logical_problems_dataset.csv",
        arguments.data_dir / "logical_problems_symbolic.csv",
    )
    split = stratified_logic_split(
        dataset,
        evaluation_fraction=arguments.evaluation_fraction,
        seed=arguments.seed,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    split_digest = hashlib.sha256()
    split_digest.update(
        ",".join(str(index) for index in split.train_indices).encode("ascii")
    )
    split_digest.update(b"|")
    split_digest.update(
        ",".join(str(index) for index in split.evaluation_indices).encode("ascii")
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_digest": dataset.source_digest,
        "row_count": len(dataset.problems),
        "class_counts": {
            "0": sum(problem.target == 0 for problem in dataset.problems),
            "1": sum(problem.target == 1 for problem in dataset.problems),
        },
        "semantic_validation": {
            "typed_ast_mismatches": 0,
            "primitive_graph_mismatches": 0,
            "validated_rows": len(dataset.problems),
            "minimum_ast_nodes": min(
                len(problem.syntax_tree.nodes) for problem in dataset.problems
            ),
            "maximum_ast_nodes": max(
                len(problem.syntax_tree.nodes) for problem in dataset.problems
            ),
            "maximum_ast_depth": max(
                problem.syntax_tree.maximum_depth for problem in dataset.problems
            ),
        },
        "split": {
            "seed": split.seed,
            "evaluation_fraction": split.evaluation_fraction,
            "train_rows": len(split.train_indices),
            "evaluation_rows": len(split.evaluation_indices),
            "index_digest": f"sha256:{split_digest.hexdigest()}",
        },
        "encodings": {},
    }

    print(
        f"Logic rows={len(dataset.problems)} train={len(split.train_indices)} "
        f"evaluation={len(split.evaluation_indices)} digest={dataset.source_digest}"
    )
    encoding_manifest = manifest["encodings"]
    assert isinstance(encoding_manifest, dict)
    for encoding_name in arguments.encodings:
        encoding = LogicEncoding(encoding_name)
        encoded = encode_logic_split(dataset, split, encoding)
        train_path = arguments.output_dir / f"{encoding.value}_train.txt"
        evaluation_path = arguments.output_dir / f"{encoding.value}_evaluation.txt"
        train_digest = write_binary_rows(
            train_path, encoded.train.rows, encoded.train.targets
        )
        evaluation_digest = write_binary_rows(
            evaluation_path, encoded.evaluation.rows, encoded.evaluation.targets
        )
        combined = collision_report(
            encoded.train.rows + encoded.evaluation.rows,
            encoded.train.targets + encoded.evaluation.targets,
        )
        signatures = evaluation_signature_report(
            encoded.train.rows,
            encoded.train.targets,
            encoded.evaluation.rows,
            encoded.evaluation.targets,
        )
        literal_schema = [
            {
                "feature_index": literal.feature_index,
                "literal_id": literal.literal_id,
                "name": literal.name,
                "kind": literal.kind.value,
                "token": literal.token,
                "count_at_least": literal.count_at_least,
                "position": literal.position,
                "depth": literal.depth,
                "variable": literal.variable,
                "relation": literal.relation,
            }
            for literal in encoded.literals
        ]
        encoding_manifest[encoding.value] = {
            "feature_count": encoded.feature_count,
            "train_file": train_path.name,
            "train_digest": train_digest,
            "evaluation_file": evaluation_path.name,
            "evaluation_digest": evaluation_digest,
            "train_truncated_rows": encoded.train.truncated_rows,
            "evaluation_truncated_rows": encoded.evaluation.truncated_rows,
            "collision_report": asdict(combined)
            | {"optimistic_ceiling": combined.optimistic_ceiling},
            "evaluation_signature_report": asdict(signatures)
            | {
                "seen_fraction": signatures.seen_fraction,
                "lookup_accuracy": signatures.lookup_accuracy,
            },
            "literal_schema": literal_schema,
        }
        print(
            f"{encoding.value}: features={encoded.feature_count} "
            f"unique={combined.unique_signatures} "
            f"collision_ceiling={combined.optimistic_ceiling:.4f} "
            f"eval_seen={signatures.seen_fraction:.4f} "
            f"lookup_eval={signatures.lookup_accuracy:.4f}"
        )

    manifest_path = arguments.output_dir / "logic_baseline_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
