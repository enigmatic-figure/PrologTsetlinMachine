"""Export a PA threshold feature as a portable PTM "Little Guy."""

from __future__ import annotations

import argparse
from pathlib import Path

from prolog_tsetlin import (
    InputShape,
    PAArtifact,
    PortSemantic,
    RestorationHandle,
    SlotBinding,
    SourceKind,
    ValidationSignature,
    export_masked_threshold,
)


def threshold_source() -> PAArtifact:
    return PAArtifact.create_masked_threshold(
        input_shape=InputShape.PA_32X32,
        port_semantic=PortSemantic.TA_ACTION,
        mapping_version="threshold-little-guy-v1",
        slot_bindings=(
            SlotBinding(1, SourceKind.TA, "first-signal", (301,)),
            SlotBinding(7, SourceKind.TA, "second-signal", (302,)),
            SlotBinding(70, SourceKind.TA, "third-signal", (303,)),
        ),
        selected_slots=(1, 7, 70),
        minimum_true=2,
        validation_signature=ValidationSignature(
            "sha256:exhaustive-three-selected-slots", 8, 0
        ),
        restoration_handle=RestorationHandle(1, "snapshot:pa-before-export"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        nargs="?",
        default="out/artifacts/threshold-little-guy.ptm",
    )
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = export_masked_threshold(
        threshold_source(),
        name="Threshold Little Guy",
        path=output,
        description="Returns true when at least two of three selected slots match.",
        authors=("Prolog Tsetlin Machine project",),
        license="CC0-1.0",
        intended_use="recreational math and reusable threshold features",
        limitations="three selected slots in a fixed 32x32 PA block",
    )
    rows = []
    for bits in range(8):
        row = [False] * artifact.slot_count
        for index, slot in enumerate(artifact.selected_slots):
            row[slot] = bool(bits & (1 << index))
        rows.append(row)
    values = artifact.predict_rows(rows)
    expected = (0, 0, 0, 1, 0, 1, 1, 1)
    if values != expected:
        raise RuntimeError("exported threshold artifact failed its exhaustive domain")
    print(f"wrote {output} ({len(artifact.serialized)} bytes)")
    print(f"artifact_id={artifact.artifact_id}")
    print(f"source_artifact_id={threshold_source().artifact_id}")
    print(f"truth_table_mask=0x{sum(value << lane for lane, value in enumerate(values)):02x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
