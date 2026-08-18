"""Export a packed XOR model that accepts typed raw Boolean records."""

from __future__ import annotations

import argparse
from pathlib import Path

from prolog_tsetlin import (
    FeatureSchema,
    FieldKind,
    LiteralCatalog,
    PreprocessingContract,
    ScalarBinaryTsetlinMachine,
    export_packed_tm,
)


def xor_machine() -> ScalarBinaryTsetlinMachine:
    machine = ScalarBinaryTsetlinMachine(
        4, 2, states_per_action=3, specificity=3.0, threshold=8, seed=41
    )
    for clause in range(4):
        for literal in range(4):
            machine.set_state(clause, literal, 3)
    for clause, literal in (
        (0, 0), (0, 3), (1, 0), (1, 2),
        (2, 1), (2, 2), (3, 1), (3, 3),
    ):
        machine.set_state(clause, literal, 4)
    return machine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output", nargs="?", default="out/artifacts/raw-xor-little-guy.ptm"
    )
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    schema = FeatureSchema.from_fields(
        left=FieldKind.BOOLEAN, right=FieldKind.BOOLEAN
    )
    catalog = LiteralCatalog(schema)
    catalog.category_eq("left", True)
    catalog.category_eq("right", True)
    preprocessing = PreprocessingContract.from_catalog(catalog)
    records = (
        {"left": False, "right": False},
        {"left": False, "right": True},
        {"left": True, "right": False},
        {"left": True, "right": True},
    )
    artifact = export_packed_tm(
        xor_machine().snapshot(),
        name="Raw-record XOR Little Guy",
        path=output,
        description="Exact XOR with deterministic typed record preprocessing.",
        preprocessing=preprocessing,
        validation_records=records,
        feature_names=("left", "right"),
        feature_catalog_version="raw-xor-v1",
        intended_use="raw-record preprocessing and runtime conformance",
        limitations="two Boolean fields and binary output",
    )
    predictions = artifact.predict_records(records)
    if predictions != (0, 1, 1, 0):
        raise RuntimeError("exported raw-record XOR failed its truth table")
    print(f"wrote {output} ({len(artifact.serialized)} bytes)")
    print(f"artifact_id={artifact.artifact_id}")
    print(f"predictions={list(predictions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
