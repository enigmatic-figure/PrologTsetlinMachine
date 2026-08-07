"""Export a compiled Logic feature as a portable PTM "Little Guy."""

from __future__ import annotations

import argparse
from pathlib import Path

from prolog_tsetlin import LogicProgram32, export_logic_program, parse_logic_tokens


def conditional_program() -> LogicProgram32:
    parsed = parse_logic_tokens(
        (
            "(",
            "A",
            "&",
            "-",
            "B",
            ")",
            "if",
            "C",
            "$",
            "(",
            "D",
            "x",
            "E",
            ")",
        )
    )
    return LogicProgram32.compile(parsed.lower())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        nargs="?",
        default="out/artifacts/conditional-little-guy.ptm",
    )
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    program = conditional_program()
    artifact = export_logic_program(
        program,
        name="Conditional Little Guy",
        path=output,
        description="Returns A and not B when C, otherwise D or E.",
        authors=("Prolog Tsetlin Machine project",),
        license="CC0-1.0",
        intended_use="recreational math and reusable Boolean features",
        limitations="fixed five-binding demonstration only",
        binding_literal_ids=(201, 202, 203, 204, 205),
        binding_catalog_version="conditional-v1",
        validation_signature={
            "dataset_digest": "sha256:exhaustive-five-binding-domain",
            "example_count": 32,
            "mismatch_count": 0,
        },
    )
    rows = tuple(
        tuple(bool(bits & (1 << index)) for index in range(5))
        for bits in range(32)
    )
    expected = tuple(
        int((row[0] and not row[1]) if row[2] else (row[3] or row[4]))
        for row in rows
    )
    values = artifact.predict_rows(rows)
    if values != expected:
        raise RuntimeError("exported Logic artifact failed its exhaustive domain")
    print(f"wrote {output} ({len(artifact.serialized)} bytes)")
    print(f"artifact_id={artifact.artifact_id}")
    print(f"program_id={program.program_id}")
    print(f"truth_table_mask=0x{sum(value << lane for lane, value in enumerate(values)):08x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
