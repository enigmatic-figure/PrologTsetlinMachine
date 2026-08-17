"""Training-side PTM command line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .artifact import PAArtifact
from .logic_consolidation import (
    FixedLogicInstruction,
    FixedLogicOpcode,
    LogicProgram32,
)
from .model_artifact import (
    export_logic_program,
    export_masked_threshold,
    export_packed_tm,
)
from .reference import SNAPSHOT_SCHEMA_VERSION, TMSnapshot


def _snapshot_from_json(path: Path) -> TMSnapshot:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("snapshot JSON must contain an object")
    return TMSnapshot(
        schema_version=int(value.get("schema_version", SNAPSHOT_SCHEMA_VERSION)),
        number_of_clauses=int(value["number_of_clauses"]),
        number_of_features=int(value["number_of_features"]),
        states_per_action=int(value["states_per_action"]),
        specificity=float(value.get("specificity", 2.0)),
        threshold=int(value["threshold"]),
        states=tuple(
            tuple(int(state) for state in clause) for clause in value["states"]
        ),
        rng_state=None,
    )


def _export(arguments: argparse.Namespace) -> int:
    source = Path(arguments.snapshot)
    destination = Path(arguments.output)
    snapshot = _snapshot_from_json(source)
    feature_names = (
        tuple(arguments.feature_names.split(","))
        if arguments.feature_names is not None
        else None
    )
    feature_literal_ids = (
        tuple(int(value) for value in arguments.feature_literal_ids.split(","))
        if arguments.feature_literal_ids is not None
        else None
    )
    artifact = export_packed_tm(
        snapshot,
        name=arguments.name or source.stem,
        path=destination,
        description=arguments.description,
        authors=arguments.author,
        license=arguments.license,
        intended_use=arguments.intended_use,
        limitations=arguments.limitations,
        feature_names=feature_names,
        feature_literal_ids=feature_literal_ids,
        feature_catalog_version=arguments.feature_catalog_version,
    )
    print(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "artifact_kind": artifact.manifest["artifact_kind"],
                "output": str(destination),
            },
            sort_keys=True,
        )
    )
    return 0


def _logic_program_from_json(path: Path) -> LogicProgram32:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("instructions"), list):
        raise ValueError("Logic program JSON must contain an instructions array")
    instructions = []
    for item in value["instructions"]:
        if not isinstance(item, dict):
            raise ValueError("Logic instructions must be objects")
        opcode_value = item.get("opcode_value")
        if opcode_value is None:
            opcode_name = str(item["opcode"]).upper()
            opcode = FixedLogicOpcode[opcode_name]
        else:
            opcode = FixedLogicOpcode(int(opcode_value))
        instructions.append(
            FixedLogicInstruction(
                opcode,
                int(item.get("operand_mask", 0)),
                int(item.get("argument", 0)),
            )
        )
    return LogicProgram32(
        tuple(instructions),
        int(value["root_instruction"]),
        int(value.get("schema_version", 1)),
    )


def _export_logic(arguments: argparse.Namespace) -> int:
    source = Path(arguments.program)
    destination = Path(arguments.output)
    binding_names = (
        tuple(arguments.binding_names.split(","))
        if arguments.binding_names is not None
        else tuple("ABCDE")
    )
    binding_literal_ids = (
        tuple(int(value) for value in arguments.binding_literal_ids.split(","))
        if arguments.binding_literal_ids is not None
        else None
    )
    artifact = export_logic_program(
        _logic_program_from_json(source),
        name=arguments.name or source.stem,
        path=destination,
        description=arguments.description,
        authors=arguments.author,
        license=arguments.license,
        intended_use=arguments.intended_use,
        limitations=arguments.limitations,
        binding_names=binding_names,
        binding_literal_ids=binding_literal_ids,
        binding_catalog_version=arguments.binding_catalog_version,
    )
    print(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "artifact_kind": artifact.manifest["artifact_kind"],
                "output": str(destination),
            },
            sort_keys=True,
        )
    )
    return 0


def _export_pa(arguments: argparse.Namespace) -> int:
    source = Path(arguments.artifact)
    destination = Path(arguments.output)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("PA artifact JSON must contain an object")
    artifact = export_masked_threshold(
        PAArtifact.from_dict(value),
        name=arguments.name or source.stem,
        path=destination,
        description=arguments.description,
        authors=arguments.author,
        license=arguments.license,
        intended_use=arguments.intended_use,
        limitations=arguments.limitations,
    )
    print(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "artifact_kind": artifact.manifest["artifact_kind"],
                "output": str(destination),
            },
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ptm")
    commands = parser.add_subparsers(dest="command", required=True)
    tui = commands.add_parser("tui", help="launch the optional terminal workbench")
    tui.add_argument("--workspace", type=Path)
    tui.add_argument("--demo", choices=("xor",), default="xor")
    tui.set_defaults(handler=_tui)
    export = commands.add_parser(
        "export", help="freeze a scalar TM snapshot JSON file into a .ptm artifact"
    )
    export.add_argument("snapshot")
    export.add_argument("output")
    export.add_argument("--name")
    export.add_argument("--description", default="")
    export.add_argument("--author", action="append", default=[])
    export.add_argument("--license", default="unspecified")
    export.add_argument("--intended-use", default="research")
    export.add_argument("--limitations", default="research prototype")
    export.add_argument("--feature-names")
    export.add_argument("--feature-literal-ids")
    export.add_argument("--feature-catalog-version", default="anonymous-v1")
    export.set_defaults(handler=_export)

    export_logic = commands.add_parser(
        "export-logic",
        help="package a fixed LogicProgram32 JSON file into a .ptm artifact",
    )
    export_logic.add_argument("program")
    export_logic.add_argument("output")
    export_logic.add_argument("--name")
    export_logic.add_argument("--description", default="")
    export_logic.add_argument("--author", action="append", default=[])
    export_logic.add_argument("--license", default="unspecified")
    export_logic.add_argument("--intended-use", default="research")
    export_logic.add_argument("--limitations", default="research prototype")
    export_logic.add_argument("--binding-names")
    export_logic.add_argument("--binding-literal-ids")
    export_logic.add_argument(
        "--binding-catalog-version", default="logic-bindings-v1"
    )
    export_logic.set_defaults(handler=_export_logic)

    export_pa = commands.add_parser(
        "export-pa",
        help="package a validated Class II PA JSON artifact into a .ptm file",
    )
    export_pa.add_argument("artifact")
    export_pa.add_argument("output")
    export_pa.add_argument("--name")
    export_pa.add_argument("--description", default="")
    export_pa.add_argument("--author", action="append", default=[])
    export_pa.add_argument("--license", default="unspecified")
    export_pa.add_argument("--intended-use", default="research")
    export_pa.add_argument("--limitations", default="research prototype")
    export_pa.set_defaults(handler=_export_pa)
    return parser


def _tui(arguments: argparse.Namespace) -> int:
    """Import Textual only when the optional interface is requested."""
    try:
        from .tui import run
    except ModuleNotFoundError as error:
        if error.name == "textual":
            raise ValueError(
                "the TUI extra is not installed; run "
                "`python -m pip install 'prolog-tsetlin-machine[tui]'`"
            ) from None
        raise
    run(workspace=arguments.workspace, demo=arguments.demo)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
