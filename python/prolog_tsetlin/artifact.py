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
    """Supported input shapes for Prolog Automaton artifacts.
    
    Attributes:
        PA_32X32: 32x32 bit input shape (1024 bits total)
        PA_64X64: 64x64 bit input shape (4096 bits total)
    """
    PA_32X32 = "32x32"
    PA_64X64 = "64x64"

    @property
    def bit_count(self) -> int:
        """Return the total number of bits for this input shape."""
        return 1024 if self is InputShape.PA_32X32 else 4096


class SourceKind(str, Enum):
    """Types of sources that can bind to PA slots.
    
    Attributes:
        LITERAL: Direct literal truth value
        TA: Tsetlin Automaton action output
        LITERAL_CONDITION: Conditional literal evaluation
        CLAUSE: Clause output from TM
        ARTIFACT_OUTPUT: Output from another artifact
    """
    LITERAL = "literal"
    TA = "ta"
    LITERAL_CONDITION = "literal_condition"
    CLAUSE = "clause"
    ARTIFACT_OUTPUT = "artifact_output"


@dataclass(frozen=True, slots=True)
class SlotBinding:
    """Binds a PA slot to a source with optional provenance tracking.
    
    Attributes:
        slot: The slot index being bound
        source_kind: The type of source providing the value
        source_id: Identifier for the source
        provenance_literal_ids: Tuple of literal IDs for provenance tracking
    """
    slot: int
    source_kind: SourceKind
    source_id: str
    provenance_literal_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate slot binding constraints."""
        if self.slot < 0:
            raise ValueError("slot cannot be negative")
        if not self.source_id:
            raise ValueError("source_id cannot be empty")
        if any(value < 0 or value >= 1 << 64 for value in self.provenance_literal_ids):
            raise ValueError("provenance literal IDs must be unsigned 64-bit values")

    def to_dict(self) -> dict[str, Any]:
        """Convert slot binding to dictionary representation."""
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
    """Cryptographic validation signature for artifact testing.
    
    Attributes:
        dataset_digest: SHA256 digest of the validation dataset
        example_count: Number of examples in the validation set
        mismatch_count: Number of mismatches observed during validation
    """
    dataset_digest: str
    example_count: int
    mismatch_count: int

    def __post_init__(self) -> None:
        """Validate signature constraints."""
        if not self.dataset_digest:
            raise ValueError("dataset_digest cannot be empty")
        if self.example_count <= 0:
            raise ValueError("example_count must be positive")
        if not 0 <= self.mismatch_count <= self.example_count:
            raise ValueError("mismatch_count must lie within the validation set")

    def to_dict(self) -> dict[str, Any]:
        """Convert validation signature to dictionary representation."""
        return {
            "dataset_digest": self.dataset_digest,
            "example_count": self.example_count,
            "mismatch_count": self.mismatch_count,
        }


@dataclass(frozen=True, slots=True)
class RestorationHandle:
    """Handle for restoring TM state from a snapshot.
    
    Attributes:
        snapshot_schema_version: Version of the snapshot schema
        snapshot_id: Unique identifier for the snapshot
    """
    snapshot_schema_version: int
    snapshot_id: str

    def __post_init__(self) -> None:
        """Validate restoration handle constraints."""
        if self.snapshot_schema_version <= 0:
            raise ValueError("snapshot_schema_version must be positive")
        if not self.snapshot_id:
            raise ValueError("snapshot_id cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Convert restoration handle to dictionary representation."""
        return {
            "snapshot_schema_version": self.snapshot_schema_version,
            "snapshot_id": self.snapshot_id,
        }


@dataclass(frozen=True, slots=True)
class MaskedThresholdPayload:
    """Payload for masked threshold kernel evaluation.
    
    Attributes:
        selected_slots: Tuple of slot indices selected for the threshold operation
        minimum_true: Minimum number of selected slots that must be true
    """
    selected_slots: tuple[int, ...]
    minimum_true: int

    def __post_init__(self) -> None:
        """Validate payload constraints."""
        if len(set(self.selected_slots)) != len(self.selected_slots):
            raise ValueError("selected slots must be unique")
        if any(slot < 0 for slot in self.selected_slots):
            raise ValueError("selected slots cannot be negative")
        if not 0 <= self.minimum_true <= len(self.selected_slots):
            raise ValueError("minimum_true must be within selected slot count")

    def to_dict(self) -> dict[str, Any]:
        """Convert payload to dictionary representation."""
        return {
            "kernel_kind": "masked_threshold_v1",
            "selected_slots": list(self.selected_slots),
            "minimum_true": self.minimum_true,
        }


@dataclass(frozen=True, slots=True)
class PAArtifact:
    """Content-addressed artifact for the first compiled Class II kernel.
    
    Attributes:
        artifact_id: SHA256 content hash identifier
        input_shape: Input shape specification (32x32 or 64x64)
        port_semantic: Semantic type of the input port
        mapping_version: Version string for the slot binding mapping
        slot_bindings: Tuple of slot-to-source bindings
        payload: Masked threshold evaluation payload
        validation_signature: Validation metrics and dataset digest
        restoration_handle: Handle for TM state restoration
        schema_version: Artifact schema version number
    """
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
        """Validate artifact integrity constraints."""
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
        """Create a masked threshold PA artifact with computed content hash.
        
        Args:
            input_shape: Input shape specification
            port_semantic: Semantic type of the input port
            mapping_version: Version string for the slot binding mapping
            slot_bindings: Sequence of slot-to-source bindings
            selected_slots: Indices of slots selected for threshold operation
            minimum_true: Minimum number of selected slots that must be true
            validation_signature: Validation metrics and dataset digest
            restoration_handle: Handle for TM state restoration
            
        Returns:
            A new PAArtifact instance with computed SHA256 artifact_id
        """
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
        """Return dictionary representation excluding artifact_id for hashing."""
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
        """Return canonical JSON string for content hashing."""
        return json.dumps(
            self._content_dict(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def verify_artifact_id(self) -> bool:
        """Verify that artifact_id matches the computed content hash."""
        expected = "sha256:" + hashlib.sha256(
            self.canonical_content().encode("utf-8")
        ).hexdigest()
        return self.artifact_id == expected

    def to_dict(self) -> dict[str, Any]:
        """Convert artifact to dictionary including artifact_id."""
        result = self._content_dict()
        result["artifact_id"] = self.artifact_id
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize artifact to JSON string."""
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PAArtifact":
        """Deserialize a PAArtifact from a dictionary representation.
        
        Args:
            value: Dictionary containing artifact data
            
        Returns:
            A new PAArtifact instance with verified content hash
            
        Raises:
            ValueError: If artifact kind is unsupported, kernel kind is unknown,
                       or content hash verification fails
        """
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
