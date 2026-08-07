"""Export the first portable PTM "Little Guy" and verify its truth table."""

from __future__ import annotations

import argparse
from pathlib import Path

from prolog_tsetlin import ScalarBinaryTsetlinMachine, export_packed_tm


def xor_machine() -> ScalarBinaryTsetlinMachine:
    machine = ScalarBinaryTsetlinMachine(
        4, 2, states_per_action=3, specificity=3.0, threshold=8, seed=41
    )
    for clause in range(4):
        for literal in range(4):
            machine.set_state(clause, literal, 3)
    machine.set_state(0, 0, 4)
    machine.set_state(0, 3, 4)
    machine.set_state(1, 0, 4)
    machine.set_state(1, 2, 4)
    machine.set_state(2, 1, 4)
    machine.set_state(2, 2, 4)
    machine.set_state(3, 1, 4)
    machine.set_state(3, 3, 4)
    return machine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        nargs="?",
        default="out/artifacts/xor-little-guy.ptm",
    )
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = ((0, 0), (0, 1), (1, 0), (1, 1))
    artifact = export_packed_tm(
        xor_machine().snapshot(),
        name="XOR Little Guy",
        path=output,
        description="Exact two-input XOR learned behavior.",
        authors=("Prolog Tsetlin Machine project",),
        license="CC0-1.0",
        intended_use="recreational math and runtime conformance",
        limitations="binary two-feature demonstration only",
        feature_names=("left", "right"),
        feature_literal_ids=(101, 102),
        feature_catalog_version="xor-v1",
        validation_rows=rows,
        validation_signature={
            "dataset_digest": "sha256:xor-truth-table",
            "example_count": 4,
            "mismatch_count": 0,
        },
    )
    predictions = artifact.predict_rows(rows)
    if predictions != (0, 1, 1, 0):
        raise RuntimeError("exported XOR artifact failed its truth table")
    print(f"wrote {output} ({len(artifact.serialized)} bytes)")
    print(f"artifact_id={artifact.artifact_id}")
    print(f"predictions={list(predictions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
