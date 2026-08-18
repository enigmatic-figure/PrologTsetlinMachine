from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from prolog_tsetlin.cli import main as cli_main
from prolog_tsetlin import model_artifact
from prolog_tsetlin.model_artifact import (
    ModelArtifactError,
    PackedTMInferenceArtifact,
    export_packed_tm,
    load_model_artifact,
    load_model_artifact_from_bytes,
)
from prolog_tsetlin.reference import ScalarBinaryTsetlinMachine


ROOT = Path(__file__).resolve().parents[2]
GOLDEN_HEX = ROOT / "tests" / "data" / "xor_packed_tm_v1.hex"


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


def export_xor() -> PackedTMInferenceArtifact:
    return export_packed_tm(
        xor_machine().snapshot(),
        name="XOR Little Guy",
        description="Exact two-input XOR learned behavior.",
        authors=("PTM tests",),
        license="CC0-1.0",
        intended_use="portable inference conformance",
        limitations="binary demonstration only",
        feature_names=("left", "right"),
        feature_literal_ids=(101, 102),
        feature_catalog_version="xor-v1",
        validation_rows=((0, 0), (0, 1), (1, 0), (1, 1)),
        validation_signature={
            "dataset_digest": "sha256:xor-truth-table",
            "example_count": 4,
            "mismatch_count": 0,
        },
        restoration_reference={
            "snapshot_id": "snapshot:xor-before-export",
            "snapshot_schema_version": 1,
        },
    )


class ModelArtifactTests(unittest.TestCase):
    def test_xor_export_is_deterministic_and_exact(self) -> None:
        first = export_xor()
        second = export_xor()
        self.assertEqual(first.serialized, second.serialized)
        self.assertEqual(first.artifact_id, second.artifact_id)
        self.assertIsInstance(
            load_model_artifact_from_bytes(first.serialized),
            PackedTMInferenceArtifact,
        )
        self.assertTrue(first.verify_conformance())
        rows = ((0, 0), (0, 1), (1, 0), (1, 1))
        self.assertEqual(first.predict_rows(rows), (0, 1, 1, 0))
        self.assertEqual(
            first.predict_rows(rows), tuple(xor_machine().predict(rows))
        )
        self.assertNotIn("states", first.manifest)
        self.assertNotIn("rng_state", first.manifest)
        self.assertEqual(first.manifest["features"]["names"], ["left", "right"])
        self.assertEqual(first.manifest["features"]["literal_ids"], ["101", "102"])
        self.assertEqual(first.manifest["task"]["kind"], "binary_classification")

    def test_python_export_matches_cross_language_golden_bytes(self) -> None:
        expected = bytes.fromhex(GOLDEN_HEX.read_text(encoding="ascii"))
        self.assertEqual(export_xor().serialized, expected)

    def test_file_round_trip_and_tampering_rejection(self) -> None:
        artifact = export_xor()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "xor.ptm"
            artifact.write(path)
            loaded = PackedTMInferenceArtifact.from_file(path)
        self.assertEqual(loaded.artifact_id, artifact.artifact_id)
        self.assertEqual(loaded.predict_rows(((0, 1), (1, 1))), (1, 0))

        tampered = bytearray(artifact.serialized)
        tampered[len(tampered) // 2] ^= 1
        with self.assertRaisesRegex(ModelArtifactError, "SHA-256"):
            PackedTMInferenceArtifact.from_bytes(tampered)

    def test_generic_file_loader_round_trip(self) -> None:
        artifact = export_xor()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "xor.ptm"
            artifact.write(path)
            loaded = load_model_artifact(path)
        self.assertEqual(loaded.artifact_id, artifact.artifact_id)
        self.assertEqual(loaded.serialized, artifact.serialized)

    def test_oversized_sparse_file_is_rejected_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "oversized.ptm"
            with path.open("wb") as sparse_file:
                sparse_file.truncate(model_artifact._MAX_ARTIFACT_BYTES + 1)

            real_open = Path.open
            read_attempted = False

            class TrackedFile:
                def __init__(self) -> None:
                    self.file = real_open(path, "rb")

                def __enter__(self) -> "TrackedFile":
                    return self

                def __exit__(self, *args: object) -> None:
                    self.file.close()

                def fileno(self) -> int:
                    return self.file.fileno()

                def read(self, size: int = -1) -> bytes:
                    nonlocal read_attempted
                    read_attempted = True
                    return self.file.read(size)

            with mock.patch.object(Path, "open", return_value=TrackedFile()):
                with self.assertRaisesRegex(ModelArtifactError, "v1 size ceiling"):
                    load_model_artifact(path)
            self.assertFalse(read_attempted)

    def test_noncanonical_manifest_is_rejected_even_with_valid_digest(self) -> None:
        artifact = export_xor()
        data = bytearray(artifact.serialized)
        manifest_size = int.from_bytes(data[24:32], "little")
        first = 64
        last = first + manifest_size
        manifest = json.loads(data[first:last])
        noncanonical = json.dumps(manifest, sort_keys=False).encode("utf-8")
        self.assertNotEqual(noncanonical, bytes(data[first:last]))
        payload_size = int.from_bytes(data[32:40], "little")
        data[24:32] = len(noncanonical).to_bytes(8, "little")
        content = bytes(data[:64]) + noncanonical + bytes(data[last : last + payload_size])
        malformed = content + hashlib.sha256(content).digest()
        with self.assertRaisesRegex(ModelArtifactError, "not canonical"):
            PackedTMInferenceArtifact.from_bytes(malformed)

    def test_manifest_ports_cannot_disagree_with_payload(self) -> None:
        artifact = export_xor()
        data = artifact.serialized
        manifest_size = int.from_bytes(data[24:32], "little")
        payload_size = int.from_bytes(data[32:40], "little")
        first = 64
        last = first + manifest_size
        manifest = json.loads(data[first:last])
        manifest["ports"]["inputs"][0]["shape"] = [3]
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
        with self.assertRaisesRegex(ModelArtifactError, "port contract"):
            PackedTMInferenceArtifact.from_bytes(malformed)

    def test_export_ignores_training_only_random_state(self) -> None:
        snapshot = xor_machine().snapshot()
        first = export_packed_tm(snapshot, name="same")
        second = export_packed_tm(
            replace(snapshot, rng_state=("different", 42)), name="same"
        )
        self.assertEqual(first.serialized, second.serialized)

    def test_export_cli_accepts_frozen_snapshot_json(self) -> None:
        snapshot = xor_machine().snapshot()
        value = {
            "schema_version": snapshot.schema_version,
            "number_of_clauses": snapshot.number_of_clauses,
            "number_of_features": snapshot.number_of_features,
            "states_per_action": snapshot.states_per_action,
            "specificity": snapshot.specificity,
            "threshold": snapshot.threshold,
            "states": snapshot.states,
        }
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "snapshot.json"
            output = Path(temporary) / "little-guy.ptm"
            source.write_text(json.dumps(value), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = cli_main(
                    [
                        "export",
                        str(source),
                        str(output),
                        "--name",
                        "CLI XOR",
                        "--feature-names",
                        "left,right",
                        "--feature-literal-ids",
                        "101,102",
                    ]
                )
            self.assertEqual(status, 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["artifact_kind"], "packed_tm_binary_v1")
            loaded = PackedTMInferenceArtifact.from_file(output)
            self.assertEqual(
                loaded.manifest["features"]["literal_ids"], ["101", "102"]
            )
            self.assertEqual(
                loaded.predict_rows(
                    ((0, 0), (0, 1), (1, 0), (1, 1))
                ),
                (0, 1, 1, 0),
            )


if __name__ == "__main__":
    unittest.main()
