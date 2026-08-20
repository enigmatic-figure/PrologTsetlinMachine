"""Training-side PTM command line entry points."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

from ._version import __version__
from .artifact import PAArtifact
from .help_topics import (
    PARSER_TOPICS,
    TOPIC_ORDER,
    parser_topic_kwargs,
    render_topic,
    render_topic_index,
)
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
from .prolog_bridge import (
    NoDecisionTreeSolution,
    NoFeatureTemplateSolution,
    NoTAClauseSolution,
    NoThresholdSolution,
    PrologBridgeError,
)
from .services.inference import (
    MAX_INTERACTIVE_RECORDS,
    inspect_artifact,
    run_artifact_records,
    verify_artifact,
)
from .services.search import (
    BoundedSearchRequest,
    SearchKind,
    demo_search_document,
    export_search_artifact,
    run_bounded_search,
)


def _print_json(value: object, *, pretty: bool = False) -> None:
    print(json.dumps(value, indent=2 if pretty else None, sort_keys=True))


def _artifact_inspect(arguments: argparse.Namespace) -> int:
    _print_json(
        inspect_artifact(arguments.model, include_manifest=arguments.manifest),
        pretty=arguments.pretty,
    )
    return 0


def _artifact_verify(arguments: argparse.Namespace) -> int:
    report = verify_artifact(arguments.model)
    _print_json(report, pretty=arguments.pretty)
    return 0 if report["verified"] else 1


def _parse_record(value: str, label: str) -> dict[str, object]:
    try:
        record = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {error.msg}") from error
    if not isinstance(record, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return record


def _record_arguments(arguments: argparse.Namespace) -> tuple[dict[str, object], ...]:
    if arguments.record is not None:
        return tuple(
            _parse_record(value, f"--record value {index}")
            for index, value in enumerate(arguments.record, start=1)
        )

    assert arguments.jsonl is not None
    if str(arguments.jsonl) == "-":
        lines = sys.stdin
    else:
        lines = arguments.jsonl.open("r", encoding="utf-8")
    records = []
    try:
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            records.append(_parse_record(line, f"JSONL line {line_number}"))
            if len(records) > MAX_INTERACTIVE_RECORDS:
                raise ValueError(
                    "raw-record command is limited to "
                    f"{MAX_INTERACTIVE_RECORDS} records"
                )
        return tuple(records)
    finally:
        if lines is not sys.stdin:
            lines.close()


def _artifact_run_record(arguments: argparse.Namespace) -> int:
    _print_json(
        run_artifact_records(arguments.model, _record_arguments(arguments)),
        pretty=arguments.pretty,
    )
    return 0


_NO_SEARCH_SOLUTION = (
    NoThresholdSolution,
    NoFeatureTemplateSolution,
    NoTAClauseSolution,
    NoDecisionTreeSolution,
)


def _search(arguments: argparse.Namespace) -> int:
    kind = SearchKind(arguments.search_kind)
    if arguments.demo and arguments.request is not None:
        raise ValueError("choose either a request JSON file or --demo, not both")
    if arguments.demo:
        document = demo_search_document(kind)
    else:
        if arguments.request is None:
            raise ValueError("provide a request JSON file or use --demo")
        document = json.loads(arguments.request.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("search request JSON must contain an object")
    request = BoundedSearchRequest.from_dict(
        document,
        expected_kind=kind,
        timeout_seconds=arguments.timeout,
    )
    try:
        result = run_bounded_search(
            request,
            executable=arguments.gprolog,
        )
    except _NO_SEARCH_SOLUTION as error:
        _print_json(
            {
                "schema": "ptm.search.result.v1",
                "kind": kind.value,
                "status": "no_solution",
                "message": str(error),
            },
            pretty=arguments.pretty,
        )
        return 3
    report = result.to_dict()
    if getattr(arguments, "output", None) is not None:
        report["artifact"] = export_search_artifact(
            result,
            arguments.output,
            name=arguments.name or f"prolog-{kind.value}",
        )
    _print_json(report, pretty=arguments.pretty)
    return 0


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


def _help(arguments: argparse.Namespace) -> int:
    print(render_topic(arguments.topic) if arguments.topic else render_topic_index())
    return 0


def _topic_parser(
    subparsers: argparse._SubParsersAction,
    name: str,
    command_path: str,
    **kwargs: object,
) -> argparse.ArgumentParser:
    """Create a parser whose conceptual owner comes from the shared registry."""

    topic_id = PARSER_TOPICS[command_path]
    child = subparsers.add_parser(name, **kwargs, **parser_topic_kwargs(topic_id))
    child.set_defaults(_help_topic=topic_id)
    return child


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptm",
        description="Train, inspect, search, and package deterministic PTM models.",
        epilog="Conceptual help and examples: ptm help [TOPIC]",
    )
    try:
        package_version = version("prolog-tsetlin-machine")
    except PackageNotFoundError:
        package_version = f"{__version__}+source"
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {package_version}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    help_command = _topic_parser(
        commands,
        "help",
        "help",
        help="list or explain conceptual help topics",
    )
    help_command.add_argument(
        "topic",
        nargs="?",
        choices=TOPIC_ORDER,
        metavar="TOPIC",
        help="conceptual topic to explain; omit to list available topics",
    )
    help_command.set_defaults(handler=_help)

    tui = _topic_parser(
        commands,
        "tui",
        "tui",
        help="launch the optional terminal workbench",
    )
    tui.add_argument(
        "--workspace",
        type=Path,
        help="directory used for artifacts and workbench state",
    )
    tui.add_argument(
        "--demo", choices=("xor",), default="xor", help="initial demonstration dataset"
    )
    tui.set_defaults(handler=_tui)

    artifact = _topic_parser(
        commands,
        "artifact",
        "artifact",
        help="inspect, verify, or run a portable .ptm artifact",
    )
    artifact_commands = artifact.add_subparsers(
        dest="artifact_command", required=True
    )
    artifact_inspect = _topic_parser(
        artifact_commands,
        "inspect",
        "artifact inspect",
        help="validate and describe an artifact",
    )
    artifact_inspect.add_argument("model", type=Path, help="portable .ptm artifact")
    artifact_inspect.add_argument(
        "--manifest", action="store_true", help="include the complete manifest"
    )
    artifact_inspect.add_argument(
        "--pretty", action="store_true", help="indent JSON output"
    )
    artifact_inspect.set_defaults(handler=_artifact_inspect)

    artifact_verify = _topic_parser(
        artifact_commands,
        "verify",
        "artifact verify",
        help="check integrity, contracts, and conformance vectors",
    )
    artifact_verify.add_argument("model", type=Path, help="portable .ptm artifact")
    artifact_verify.add_argument(
        "--pretty", action="store_true", help="indent JSON output"
    )
    artifact_verify.set_defaults(handler=_artifact_verify)

    artifact_run = _topic_parser(
        artifact_commands,
        "run-record",
        "artifact run-record",
        help="preprocess typed JSON records and run inference",
    )
    artifact_run.add_argument("model", type=Path, help="portable .ptm artifact")
    record_source = artifact_run.add_mutually_exclusive_group(required=True)
    record_source.add_argument(
        "--record",
        action="append",
        metavar="JSON",
        help="JSON object; repeat to run multiple records",
    )
    record_source.add_argument(
        "--jsonl",
        type=Path,
        metavar="PATH",
        help="newline-delimited JSON records; use - for standard input",
    )
    artifact_run.add_argument(
        "--pretty", action="store_true", help="indent JSON output"
    )
    artifact_run.set_defaults(handler=_artifact_run_record)

    search = _topic_parser(
        commands,
        "search",
        "search",
        help="run a resource-bounded GNU Prolog search",
    )
    search_commands = search.add_subparsers(dest="search_kind", required=True)
    search_help = {
        SearchKind.THRESHOLD: "find an exact monotone masked threshold",
        SearchKind.FEATURE_TEMPLATE: "select an exact typed feature template",
        SearchKind.TA_CLAUSE: "synthesize an exact signed TA clause",
        SearchKind.DECISION_TREE: "synthesize an exact bounded decision tree",
        SearchKind.REPAIR: "repair a parent tree from counterexamples",
    }
    for kind, help_text in search_help.items():
        command_path = f"search {kind.value}"
        search_command = _topic_parser(
            search_commands,
            kind.value,
            command_path,
            help=help_text,
        )
        search_command.add_argument(
            "request",
            nargs="?",
            type=Path,
            help="ptm.search.request.v1 JSON file",
        )
        search_command.add_argument(
            "--demo", action="store_true", help="run the built-in bounded example"
        )
        search_command.add_argument(
            "--timeout",
            type=float,
            help="override the request deadline in seconds (0.1 through 300)",
        )
        search_command.add_argument(
            "--gprolog",
            type=Path,
            help="GNU Prolog executable; otherwise use PTM_GPROLOG or PATH",
        )
        search_command.add_argument(
            "--pretty", action="store_true", help="indent JSON output"
        )
        if kind in (SearchKind.DECISION_TREE, SearchKind.REPAIR):
            search_command.add_argument(
                "--output", type=Path, help="export fixed-Logic .ptm artifact"
            )
            search_command.add_argument("--name", help="exported artifact title")
        search_command.set_defaults(handler=_search)

    export = _topic_parser(
        commands,
        "export",
        "export",
        help="freeze a scalar TM snapshot JSON file into a .ptm artifact",
    )
    export.add_argument("snapshot", help="scalar TM snapshot JSON file")
    export.add_argument("output", help="destination .ptm artifact")
    export.add_argument("--name", help="artifact title; defaults to the input stem")
    export.add_argument("--description", default="", help="artifact description")
    export.add_argument(
        "--author", action="append", default=[], help="author name; repeat as needed"
    )
    export.add_argument("--license", default="unspecified", help="license identifier")
    export.add_argument(
        "--intended-use", default="research", help="intended-use statement"
    )
    export.add_argument(
        "--limitations", default="research prototype", help="limitations statement"
    )
    export.add_argument(
        "--feature-names", help="comma-separated feature names in input order"
    )
    export.add_argument(
        "--feature-literal-ids", help="comma-separated integer literal IDs"
    )
    export.add_argument(
        "--feature-catalog-version",
        default="anonymous-v1",
        help="feature-catalog version label",
    )
    export.set_defaults(handler=_export)

    export_logic = _topic_parser(
        commands,
        "export-logic",
        "export-logic",
        help="package a fixed LogicProgram32 JSON file into a .ptm artifact",
    )
    export_logic.add_argument("program", help="fixed LogicProgram32 JSON file")
    export_logic.add_argument("output", help="destination .ptm artifact")
    export_logic.add_argument("--name", help="artifact title; defaults to the input stem")
    export_logic.add_argument("--description", default="", help="artifact description")
    export_logic.add_argument(
        "--author", action="append", default=[], help="author name; repeat as needed"
    )
    export_logic.add_argument(
        "--license", default="unspecified", help="license identifier"
    )
    export_logic.add_argument(
        "--intended-use", default="research", help="intended-use statement"
    )
    export_logic.add_argument(
        "--limitations", default="research prototype", help="limitations statement"
    )
    export_logic.add_argument(
        "--binding-names",
        default="A,B,C,D,E",
        help="comma-separated binding names",
    )
    export_logic.add_argument(
        "--binding-literal-ids", help="comma-separated integer literal IDs"
    )
    export_logic.add_argument(
        "--binding-catalog-version",
        default="logic-bindings-v1",
        help="binding-catalog version label",
    )
    export_logic.set_defaults(handler=_export_logic)

    export_pa = _topic_parser(
        commands,
        "export-pa",
        "export-pa",
        help="package a validated Class II PA JSON artifact into a .ptm file",
    )
    export_pa.add_argument("artifact", help="validated Class II PA JSON file")
    export_pa.add_argument("output", help="destination .ptm artifact")
    export_pa.add_argument("--name", help="artifact title; defaults to the input stem")
    export_pa.add_argument("--description", default="", help="artifact description")
    export_pa.add_argument(
        "--author", action="append", default=[], help="author name; repeat as needed"
    )
    export_pa.add_argument("--license", default="unspecified", help="license identifier")
    export_pa.add_argument(
        "--intended-use", default="research", help="intended-use statement"
    )
    export_pa.add_argument(
        "--limitations", default="research prototype", help="limitations statement"
    )
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
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
        PrologBridgeError,
    ) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print("ptm: interrupted; active child process was terminated", file=sys.stderr)
        return 130
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
