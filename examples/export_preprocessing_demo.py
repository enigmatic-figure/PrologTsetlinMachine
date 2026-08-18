"""Export a packed TM exercising every portable preprocessing-v1 transform."""

from __future__ import annotations

import argparse
from pathlib import Path

from prolog_tsetlin import (
    FeatureSchema,
    FieldKind,
    LiteralCatalog,
    NullPolicy,
    PreprocessingContract,
    ScalarBinaryTsetlinMachine,
    export_packed_tm,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output", nargs="?", default="out/artifacts/preprocessing-demo.ptm"
    )
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    schema = FeatureSchema.from_fields(
        age=FieldKind.NUMBER,
        status=FieldKind.CATEGORY,
        active=FieldKind.BOOLEAN,
    )
    catalog = LiteralCatalog(schema)
    catalog.numeric_ge("age", 18, null_policy=NullPolicy.ERROR)
    catalog.numeric_between("age", 21, 65)
    catalog.category_eq("status", "ready", null_policy=NullPolicy.TRUE)
    catalog.category_in("status", ("ready", "running"))
    catalog.category_eq("active", True)
    catalog.is_missing("active")
    preprocessing = PreprocessingContract.from_catalog(catalog)

    # A two-clause binary TM whose positive clause selects the first
    # preprocessing output. It predicts whether the strict age gate is true;
    # the other outputs make every v1 transform inspectable in one artifact.
    machine = ScalarBinaryTsetlinMachine(
        2, 6, states_per_action=3, specificity=3.0, threshold=8, seed=73
    )
    for clause in range(2):
        for literal in range(12):
            machine.set_state(clause, literal, 3)
    machine.set_state(0, 0, 4)

    records = (
        {"age": 17, "status": "ready", "active": True},
        {"age": 30, "status": "running", "active": False},
        {"age": 70, "status": None},
    )
    artifact = export_packed_tm(
        machine.snapshot(),
        name="Portable preprocessing demo",
        path=output,
        description="Every deterministic ptm.preprocessing.v1 transform.",
        preprocessing=preprocessing,
        validation_records=records,
        feature_names=(
            "adult",
            "working_age",
            "status_ready",
            "status_live",
            "active",
            "active_missing",
        ),
        feature_catalog_version="preprocessing-demo-v1",
        intended_use="cross-language preprocessing conformance",
        limitations="binary demonstration model, not a fitted policy",
    )
    print(f"wrote {output} ({len(artifact.serialized)} bytes)")
    print(f"artifact_id={artifact.artifact_id}")
    print(f"predictions={list(artifact.predict_records(records))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
