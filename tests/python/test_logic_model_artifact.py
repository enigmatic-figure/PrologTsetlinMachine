from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from prolog_tsetlin import (
    LogicProgram32,
    LogicProgramInferenceArtifact,
    export_logic_program,
    load_model_artifact,
    load_model_artifact_from_bytes,
    parse_logic_tokens,
)
from prolog_tsetlin.model_artifact import ModelArtifactError
from prolog_tsetlin.cli import main as cli_main


ROOT = Path(__file__).resolve().parents[2]
GOLDEN_HEX = ROOT / "tests" / "data" / "conditional_logic_program_v1.hex"


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


def export_conditional() -> LogicProgramInferenceArtifact:
    return export_logic_program(
        conditional_program(),
        name="Conditional Little Guy",
        description="Five-binding conditional Boolean feature.",
        authors=("PTM tests",),
        license="CC0-1.0",
        intended_use="portable Logic inference conformance",
        limitations="fixed five-binding demonstration only",
        binding_literal_ids=(201, 202, 203, 204, 205),
        binding_catalog_version="conditional-v1",
        validation_signature={
            "dataset_digest": "sha256:exhaustive-five-binding-domain",
            "example_count": 32,
            "mismatch_count": 0,
        },
        restoration_reference={
            "snapshot_id": "snapshot:logic-before-export",
            "snapshot_schema_version": 1,
        },
    )


class LogicModelArtifactTests(unittest.TestCase):
    def test_export_is_deterministic_and_exhaustively_exact(self) -> None:
        first = export_conditional()
        second = export_conditional()
        self.assertEqual(first.serialized, second.serialized)
        self.assertEqual(first.artifact_id, second.artifact_id)
        self.assertTrue(first.verify_conformance())

        rows = tuple(
            tuple(bool(bits & (1 << index)) for index in range(5))
            for bits in range(32)
        )
        expected = tuple(
            int((row[0] and not row[1]) if row[2] else (row[3] or row[4]))
            for row in rows
        )
        self.assertEqual(first.predict_rows(rows), expected)
        self.assertEqual(first.manifest["bindings"]["names"], list("ABCDE"))
        self.assertEqual(
            first.manifest["bindings"]["literal_ids"],
            ["201", "202", "203", "204", "205"],
        )

    def test_generic_loader_dispatches_file_and_memory(self) -> None:
        artifact = export_conditional()
        loaded_memory = load_model_artifact_from_bytes(artifact.serialized)
        self.assertIsInstance(loaded_memory, LogicProgramInferenceArtifact)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "conditional.ptm"
            artifact.write(path)
            loaded_file = load_model_artifact(path)
        self.assertEqual(loaded_file.artifact_id, artifact.artifact_id)
        self.assertEqual(loaded_file.predict_rows(((1, 0, 1, 0, 0),)), (1,))

    def test_manifest_program_identity_cannot_disagree_with_payload(self) -> None:
        artifact = export_conditional()
        data = artifact.serialized
        manifest_size = int.from_bytes(data[24:32], "little")
        payload_size = int.from_bytes(data[32:40], "little")
        first = 64
        last = first + manifest_size
        manifest = json.loads(data[first:last])
        manifest["model"]["program_id"] = "sha256:" + "0" * 64
        manifest_bytes = json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        header = bytearray(data[:64])
        header[24:32] = len(manifest_bytes).to_bytes(8, "little")
        content = bytes(header) + manifest_bytes + data[last : last + payload_size]
        malformed = content + hashlib.sha256(content).digest()
        with self.assertRaisesRegex(ModelArtifactError, "semantics disagree"):
            LogicProgramInferenceArtifact.from_bytes(malformed)

    def test_export_logic_cli_accepts_program_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "conditional.json"
            output = Path(temporary) / "conditional.ptm"
            source.write_text(
                json.dumps(conditional_program().to_dict()), encoding="utf-8"
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = cli_main(
                    [
                        "export-logic",
                        str(source),
                        str(output),
                        "--name",
                        "CLI Conditional",
                        "--binding-literal-ids",
                        "201,202,203,204,205",
                    ]
                )
            self.assertEqual(status, 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["artifact_kind"], "logic_program32_v1")
            loaded = LogicProgramInferenceArtifact.from_file(output)
            self.assertEqual(loaded.predict_rows(((1, 0, 1, 0, 0),)), (1,))

    def test_python_export_matches_cross_language_golden_bytes(self) -> None:
        expected = bytes.fromhex(GOLDEN_HEX.read_text(encoding="ascii"))
        self.assertEqual(export_conditional().serialized, expected)


if __name__ == "__main__":
    unittest.main()
