"""Versioned Class II PA artifacts and their cold state partition."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .pa import PortSemantic


PA_ARTIFACT_SCHEMA_VERSION = 1


class InputShape(str, Enum):
    PA_32X32 = "32x32"
    PA_64X64 = "64x64"

    @property
    def bit_count(self) -> int:
        return 1024 if self is InputShape.PA_32X32 else 4096


class SourceKind(str, Enum):
    LITERAL = "literal"
    TA = "ta"
    LITERAL_CONDITION = "literal_condition"
    CLAUSE = "clause"
    ARTIFACT_OUTPUT = "artifact_output"


@dataclass(frozen=True, slots=True)
class SlotBinding:
    slot: int
    source_kind: SourceKind
    source_id: str
    provenance_literal_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.slot < 0:
            raise ValueError("slot cannot be negative")
        if not self.source_id:
            raise ValueError("source_id cannot be empty")
        if any(value < 0 or value >= 1 << 64 for value in self.provenance_literal_ids):
            raise ValueError("provenance literal IDs must be unsigned 64-bit values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "source_kind": self.source_kind.value,
            "source_id": self.source_id,
            "provenance_literal_ids": [
                str(value) for value in self.provenance_literal_ids
            ],
        }


@dataclass(frozen=True, slots=True)
class ValidationSignature:
    dataset_digest: str
    example_count: int
    mismatch_count: int

    def __post_init__(self) -> None:
        if not self.dataset_digest:
            raise ValueError("dataset_digest cannot be empty")
        if self.example_count <= 0:
            raise ValueError("example_count must be positive")
        if not 0 <= self.mismatch_count <= self.example_count:
            raise ValueError("mismatch_count must lie within the validation set")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_digest": self.dataset_digest,
            "example_count": self.example_count,
            "mismatch_count": self.mismatch_count,
        }


@dataclass(frozen=True, slots=True)
class RestorationHandle:
    snapshot_schema_version: int
    snapshot_id: str

    def __post_init__(self) -> None:
        if self.snapshot_schema_version <= 0:
            raise ValueError("snapshot_schema_version must be positive")
        if not self.snapshot_id:
            raise ValueError("snapshot_id cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_schema_version": self.snapshot_schema_version,
            "snapshot_id": self.snapshot_id,
        }


@dataclass(frozen=True, slots=True)
class MaskedThresholdPayload:
    selected_slots: tuple[int, ...]
    minimum_true: int

    def __post_init__(self) -> None:
        if len(set(self.selected_slots)) != len(self.selected_slots):
            raise ValueError("selected slots must be unique")
        if any(slot < 0 for slot in self.selected_slots):
            raise ValueError("selected slots cannot be negative")
        if not 0 <= self.minimum_true <= len(self.selected_slots):
            raise ValueError("minimum_true must be within selected slot count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_kind": "masked_threshold_v1",
            "selected_slots": list(self.selected_slots),
            "minimum_true": self.minimum_true,
        }


@dataclass(frozen=True, slots=True)
class PAArtifact:
    """Content-addressed artifact for the first compiled Class II kernel."""

    artifact_id: str
    input_shape: InputShape
    port_semantic: PortSemantic
    mapping_version: str
    slot_bindings: tuple[SlotBinding, ...]
    payload: MaskedThresholdPayload
    validation_signature: ValidationSignature
    restoration_handle: RestorationHandle
    schema_version: int = PA_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PA_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported PA artifact schema version")
        if not self.mapping_version:
            raise ValueError("mapping_version cannot be empty")
        slots = [binding.slot for binding in self.slot_bindings]
        if len(slots) != len(set(slots)):
            raise ValueError("a PA slot can have only one source binding")
        if any(slot >= self.input_shape.bit_count for slot in slots):
            raise ValueError("slot binding lies outside the PA input shape")
        if not set(self.payload.selected_slots).issubset(set(slots)):
            raise ValueError("every selected slot must have a source binding")

    @classmethod
    def create_masked_threshold(
        cls,
        *,
        input_shape: InputShape,
        port_semantic: PortSemantic,
        mapping_version: str,
        slot_bindings: Sequence[SlotBinding],
        selected_slots: Sequence[int],
        minimum_true: int,
        validation_signature: ValidationSignature,
        restoration_handle: RestorationHandle,
    ) -> "PAArtifact":
        bindings = tuple(sorted(slot_bindings, key=lambda binding: binding.slot))
        payload = MaskedThresholdPayload(tuple(sorted(selected_slots)), minimum_true)
        provisional = cls(
            artifact_id="sha256:pending",
            input_shape=input_shape,
            port_semantic=port_semantic,
            mapping_version=mapping_version,
            slot_bindings=bindings,
            payload=payload,
            validation_signature=validation_signature,
            restoration_handle=restoration_handle,
        )
        artifact_id = "sha256:" + hashlib.sha256(
            provisional.canonical_content().encode("utf-8")
        ).hexdigest()
        return cls(
            artifact_id=artifact_id,
            input_shape=input_shape,
            port_semantic=port_semantic,
            mapping_version=mapping_version,
            slot_bindings=bindings,
            payload=payload,
            validation_signature=validation_signature,
            restoration_handle=restoration_handle,
        )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_kind": "class_ii_kernel",
            "input_shape": self.input_shape.value,
            "port_semantic": self.port_semantic.value,
            "mapping_version": self.mapping_version,
            "slot_bindings": [binding.to_dict() for binding in self.slot_bindings],
            "payload": self.payload.to_dict(),
            "validation_signature": self.validation_signature.to_dict(),
            "restoration_handle": self.restoration_handle.to_dict(),
        }

    def canonical_content(self) -> str:
        return json.dumps(
            self._content_dict(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def verify_artifact_id(self) -> bool:
        expected = "sha256:" + hashlib.sha256(
            self.canonical_content().encode("utf-8")
        ).hexdigest()
        return self.artifact_id == expected

    def to_dict(self) -> dict[str, Any]:
        result = self._content_dict()
        result["artifact_id"] = self.artifact_id
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PAArtifact":
        if value.get("artifact_kind") != "class_ii_kernel":
            raise ValueError("unsupported artifact kind")
        payload_value = value["payload"]
        if payload_value.get("kernel_kind") != "masked_threshold_v1":
            raise ValueError("unsupported PA kernel kind")
        artifact = cls(
            schema_version=int(value["schema_version"]),
            artifact_id=str(value["artifact_id"]),
            input_shape=InputShape(value["input_shape"]),
            port_semantic=PortSemantic(value["port_semantic"]),
            mapping_version=str(value["mapping_version"]),
            slot_bindings=tuple(
                SlotBinding(
                    slot=int(binding["slot"]),
                    source_kind=SourceKind(binding["source_kind"]),
                    source_id=str(binding["source_id"]),
                    provenance_literal_ids=tuple(
                        int(item)
                        for item in binding.get("provenance_literal_ids", ())
                    ),
                )
                for binding in value["slot_bindings"]
            ),
            payload=MaskedThresholdPayload(
                selected_slots=tuple(
                    int(item) for item in payload_value["selected_slots"]
                ),
                minimum_true=int(payload_value["minimum_true"]),
            ),
            validation_signature=ValidationSignature(
                dataset_digest=str(value["validation_signature"]["dataset_digest"]),
                example_count=int(value["validation_signature"]["example_count"]),
                mismatch_count=int(value["validation_signature"]["mismatch_count"]),
            ),
            restoration_handle=RestorationHandle(
                snapshot_schema_version=int(
                    value["restoration_handle"]["snapshot_schema_version"]
                ),
                snapshot_id=str(value["restoration_handle"]["snapshot_id"]),
            ),
        )
        if not artifact.verify_artifact_id():
            raise ValueError("PA artifact content hash does not match artifact_id")
        return artifact
