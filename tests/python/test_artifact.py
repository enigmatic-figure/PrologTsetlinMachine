from __future__ import annotations

import copy
import unittest

from prolog_tsetlin import (
    InputShape,
    PAArtifact,
    PortSemantic,
    RestorationHandle,
    SlotBinding,
    SourceKind,
    ValidationSignature,
)


class PAArtifactTests(unittest.TestCase):
    def make_artifact(self) -> PAArtifact:
        return PAArtifact.create_masked_threshold(
            input_shape=InputShape.PA_64X64,
            port_semantic=PortSemantic.TA_ACTION,
            mapping_version="map-17",
            slot_bindings=[
                SlotBinding(9, SourceKind.TA, "ta-300", (481,)),
                SlotBinding(2, SourceKind.TA, "ta-101", (480,)),
            ],
            selected_slots=[9, 2],
            minimum_true=2,
            validation_signature=ValidationSignature(
                "sha256:validation", 2500, 0
            ),
            restoration_handle=RestorationHandle(1, "snapshot:before-compile"),
        )

    def test_round_trip_is_content_addressed_and_canonical(self) -> None:
        artifact = self.make_artifact()
        self.assertTrue(artifact.verify_artifact_id())
        restored = PAArtifact.from_dict(artifact.to_dict())
        self.assertEqual(restored, artifact)
        self.assertEqual([item.slot for item in artifact.slot_bindings], [2, 9])
        self.assertIsInstance(
            artifact.to_dict()["slot_bindings"][0]["provenance_literal_ids"][0],
            str,
        )

    def test_tampering_is_detected(self) -> None:
        value = copy.deepcopy(self.make_artifact().to_dict())
        value["payload"]["minimum_true"] = 1
        with self.assertRaises(ValueError):
            PAArtifact.from_dict(value)

    def test_selected_slot_requires_a_binding(self) -> None:
        with self.assertRaises(ValueError):
            PAArtifact.create_masked_threshold(
                input_shape=InputShape.PA_32X32,
                port_semantic=PortSemantic.CLAUSE_OUTPUT,
                mapping_version="map-1",
                slot_bindings=[],
                selected_slots=[8],
                minimum_true=1,
                validation_signature=ValidationSignature("digest", 1, 0),
                restoration_handle=RestorationHandle(1, "snapshot"),
            )


if __name__ == "__main__":
    unittest.main()
