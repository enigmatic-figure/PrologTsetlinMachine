from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from prolog_tsetlin import (
    InputShape,
    MaskedThresholdInferenceArtifact,
    PAArtifact,
    PortSemantic,
    RestorationHandle,
    SlotBinding,
    SourceKind,
    ValidationSignature,
    export_masked_threshold,
    load_model_artifact,
)
from prolog_tsetlin.cli import main as cli_main
from prolog_tsetlin.model_artifact import ModelArtifactError


ROOT = Path(__file__).resolve().parents[2]
GOLDEN_HEX = ROOT / "tests" / "data" / "masked_threshold_v1.hex"


def source_artifact() -> PAArtifact:
    return PAArtifact.create_masked_threshold(
        input_shape=InputShape.PA_32X32,
        port_semantic=PortSemantic.TA_ACTION,
        mapping_version="pa-little-guy-v1",
        slot_bindings=(
            SlotBinding(1, SourceKind.TA, "ta-1", (301,)),
            SlotBinding(7, SourceKind.TA, "ta-7", (302,)),
            SlotBinding(70, SourceKind.TA, "ta-70", (303,)),
        ),
        selected_slots=(1, 7, 70),
        minimum_true=2,
        validation_signature=ValidationSignature(
            "sha256:exhaustive-three-selected-slots", 8, 0
        ),
        restoration_handle=RestorationHandle(1, "snapshot:pa-before-export"),
    )


def export_threshold() -> MaskedThresholdInferenceArtifact:
    return export_masked_threshold(
        source_artifact(),
        name="Threshold Little Guy",
        description="Two-of-three threshold over reusable TA-action slots.",
        authors=("PTM tests",),
        license="CC0-1.0",
        intended_use="portable PA inference conformance",
        limitations="three selected slots in a 32x32 PA block",
    )


class PAModelArtifactTests(unittest.TestCase):
    def test_export_is_deterministic_and_exhaustively_exact(self) -> None:
        first = export_threshold()
        second = export_threshold()
        self.assertEqual(first.serialized, second.serialized)
        self.assertEqual(first.artifact_id, second.artifact_id)
        self.assertEqual(first.selected_slots, (1, 7, 70))
        self.assertTrue(first.verify_conformance())

        rows = []
        for bits in range(8):
            row = [0] * 1024
            for index, slot in enumerate(first.selected_slots):
                row[slot] = (bits >> index) & 1
            rows.append(row)
        self.assertEqual(first.predict_rows(rows), (0, 0, 0, 1, 0, 1, 1, 1))
        packed = [0] * 1024
        packed[1] = 1
        packed[70] = 1
        result = first.evaluate_packed(packed, valid_example_mask=1)
        self.assertEqual(result.value_mask, 1)
        self.assertEqual(result.matched_counts[0], 2)
        self.assertEqual(result.matched_slot_words[1], 1)
        self.assertEqual(result.missing_slot_words[7], 1)

    def test_generic_file_loader_dispatches_pa(self) -> None:
        artifact = export_threshold()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "threshold.ptm"
            artifact.write(path)
            loaded = load_model_artifact(path)
        self.assertIsInstance(loaded, MaskedThresholdInferenceArtifact)
        self.assertEqual(loaded.artifact_id, artifact.artifact_id)

    def test_empty_selection_with_zero_threshold_is_always_true(self) -> None:
        source = PAArtifact.create_masked_threshold(
            input_shape=InputShape.PA_32X32,
            port_semantic=PortSemantic.TA_ACTION,
            mapping_version="empty-threshold-v1",
            slot_bindings=(),
            selected_slots=(),
            minimum_true=0,
            validation_signature=ValidationSignature("sha256:empty-domain", 1, 0),
            restoration_handle=RestorationHandle(1, "snapshot:empty-threshold"),
        )
        artifact = export_masked_threshold(source, name="Always True")
        rows = ([0] * 1024, [0] * 1024)
        self.assertEqual(artifact.predict_rows(rows), (1, 1))
        self.assertEqual(artifact.selected_slots, ())
        self.assertTrue(artifact.verify_conformance())

    def test_manifest_threshold_cannot_disagree_with_payload(self) -> None:
        artifact = export_threshold()
        data = artifact.serialized
        manifest_size = int.from_bytes(data[24:32], "little")
        payload_size = int.from_bytes(data[32:40], "little")
        first = 64
        last = first + manifest_size
        manifest = json.loads(data[first:last])
        manifest["model"]["minimum_true"] = 1
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
            MaskedThresholdInferenceArtifact.from_bytes(malformed)

    def test_export_pa_cli_accepts_class_ii_artifact_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source-pa.json"
            output = Path(temporary) / "threshold.ptm"
            source.write_text(source_artifact().to_json(indent=None), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = cli_main(
                    [
                        "export-pa",
                        str(source),
                        str(output),
                        "--name",
                        "CLI Threshold",
                    ]
                )
            self.assertEqual(status, 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["artifact_kind"], "masked_threshold_v1")
            self.assertTrue(MaskedThresholdInferenceArtifact.from_file(output))

    def test_python_export_matches_cross_language_golden_bytes(self) -> None:
        expected = bytes.fromhex(GOLDEN_HEX.read_text(encoding="ascii"))
        self.assertEqual(export_threshold().serialized, expected)


if __name__ == "__main__":
    unittest.main()
