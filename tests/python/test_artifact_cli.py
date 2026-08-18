from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path

import pytest

from prolog_tsetlin.cli import main as cli_main
from prolog_tsetlin.services.inference import (
    artifact_input_fields,
    parse_typed_record,
)


ROOT = Path(__file__).resolve().parents[2]
RAW_XOR_HEX = ROOT / "tests" / "data" / "raw_xor_packed_tm_v1.hex"
PREPROCESSING_HEX = ROOT / "tests" / "data" / "preprocessing_demo_v1.hex"


def _artifact_path(directory: Path) -> Path:
    path = directory / "raw-xor.ptm"
    path.write_bytes(bytes.fromhex(RAW_XOR_HEX.read_text(encoding="ascii")))
    return path


def _run(arguments: list[str]) -> dict[str, object]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert cli_main(arguments) == 0
    return json.loads(stdout.getvalue())


def test_artifact_cli_inspects_and_verifies() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        artifact = _artifact_path(Path(temporary))
        inspected = _run(["artifact", "inspect", str(artifact), "--manifest"])
        verified = _run(["artifact", "verify", str(artifact)])

    assert inspected["artifact_kind"] == "packed_tm_binary_v1"
    assert inspected["has_preprocessing"] is True
    assert inspected["preprocessing_schema"] == "ptm.preprocessing.v1"
    assert inspected["manifest"]["features"]["names"] == ["left", "right"]
    assert verified["verified"] is True
    assert verified["conformance_case_count"] == 1


def test_artifact_cli_runs_repeated_typed_records() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        artifact = _artifact_path(Path(temporary))
        report = _run(
            [
                "artifact",
                "run-record",
                str(artifact),
                "--record",
                '{"left":false,"right":true}',
                "--record",
                '{"left":true,"right":true}',
            ]
        )

    assert report["record_count"] == 2
    assert [item["features"] for item in report["results"]] == [[0, 1], [1, 1]]
    assert [item["prediction"] for item in report["results"]] == [1, 0]
    assert report["results"][0]["feature_trace"] == [
        {
            "expression": "category_eq(value=true)",
            "field": "left",
            "literal_id": "7103032519624590305",
            "parameters": {"value": True},
            "transform": "category_eq",
            "value": 0,
        },
        {
            "expression": "category_eq(value=true)",
            "field": "right",
            "literal_id": "6433068973161788972",
            "parameters": {"value": True},
            "transform": "category_eq",
            "value": 1,
        },
    ]


def test_artifact_cli_reads_jsonl() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        artifact = _artifact_path(directory)
        records = directory / "records.jsonl"
        records.write_text(
            '{"left":false,"right":false}\n'
            '{"left":true,"right":false}\n',
            encoding="utf-8",
        )
        report = _run(
            ["artifact", "run-record", str(artifact), "--jsonl", str(records)]
        )

    assert [item["prediction"] for item in report["results"]] == [0, 1]


def test_artifact_service_describes_and_parses_typed_fields() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        artifact = _artifact_path(Path(temporary))
        fields = artifact_input_fields(artifact)

    assert [(field.name, field.kind.value) for field in fields] == [
        ("left", "boolean"),
        ("right", "boolean"),
    ]
    assert parse_typed_record(fields, {"left": "false", "right": "1"}) == {
        "left": False,
        "right": True,
    }
    with pytest.raises(ValueError, match="left must be true or false"):
        parse_typed_record(fields, {"left": "perhaps", "right": "true"})


def test_artifact_service_parses_numeric_category_and_missing_values() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        artifact = Path(temporary) / "preprocessing.ptm"
        artifact.write_bytes(
            bytes.fromhex(PREPROCESSING_HEX.read_text(encoding="ascii"))
        )
        fields = artifact_input_fields(artifact)

    assert [(field.name, field.required) for field in fields] == [
        ("age", True),
        ("status", False),
        ("active", False),
    ]
    assert parse_typed_record(
        fields, {"age": "30.5", "status": "ready", "active": ""}
    ) == {"age": 30.5, "status": "ready"}
    assert parse_typed_record(
        fields, {"age": "21", "status": '"1"', "active": "null"}
    ) == {"age": 21, "status": "1", "active": None}
    with pytest.raises(ValueError, match="age is required"):
        parse_typed_record(fields, {"age": "", "status": "ready"})
