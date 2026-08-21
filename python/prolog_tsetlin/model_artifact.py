"""Deterministic portable inference artifacts for frozen PTM models."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ._version import __version__
from .artifact import PAArtifact
from .logic_ast import LOGIC_AST_VARIABLES
from .logic_consolidation import (
    LOGIC_PROGRAM_CAPACITY,
    FixedLogicInstruction,
    FixedLogicOpcode,
    LogicProgram32,
)
from .pa import FixedBitBlock, MaskedThresholdKernel
from .preprocessing import PreprocessingContract
from .reference import SNAPSHOT_SCHEMA_VERSION, TMSnapshot

try:  # optional graph extra; validated lazily so core stays importable without it
    from .graph.cotm import CoalescedTsetlinMachine as ClauseCoTM  # noqa: F401
    from .graph.deep_clause import DeepClause, DeepClauseComponent
    from .graph.graph_tm import GraphTsetlinMachine
    from .graph.types import GraphInput, MAX_GRAPH_CLAUSES, MAX_GRAPH_DEPTH
except Exception:  # pragma: no cover
    ClauseCoTM = DeepClause = DeepClauseComponent = GraphTsetlinMachine = GraphInput = None  # type: ignore[assignment]
    MAX_GRAPH_CLAUSES = 1024  # type: ignore[assignment]
    MAX_GRAPH_DEPTH = 8  # type: ignore[assignment]


MODEL_ARTIFACT_SCHEMA = "ptm.model.v1"
PACKED_TM_PAYLOAD_KIND = "packed_tm_binary_v1"
LOGIC_PROGRAM_PAYLOAD_KIND = "logic_program32_v1"
MASKED_THRESHOLD_PAYLOAD_KIND = "masked_threshold_v1"
GRAPH_TM_PAYLOAD_KIND = "graph_tm_v1"
CONTAINER_VERSION = 1
PACKED_TM_PAYLOAD_VERSION = 1
LOGIC_PROGRAM_PAYLOAD_VERSION = 1
MASKED_THRESHOLD_PAYLOAD_VERSION = 1
GRAPH_TM_PAYLOAD_VERSION = 1

_MAGIC = b"PTMODEL\0"
_MODEL_KIND_PACKED_TM_BINARY = 1
_MODEL_KIND_LOGIC_PROGRAM32 = 2
_MODEL_KIND_MASKED_THRESHOLD = 3
_MODEL_KIND_GRAPH_TM = 4
_CONTAINER_HEADER = struct.Struct("<8sIIIIQQ24s")
_PACKED_TM_HEADER = struct.Struct("<IIIIiIII")
_LOGIC_PROGRAM_HEADER = struct.Struct("<IIIIIIII")
_LOGIC_INSTRUCTION = struct.Struct("<IBBH")
_MASKED_THRESHOLD_HEADER = struct.Struct("<IIIIIIII")
_GRAPH_TM_HEADER = struct.Struct("<IIIIIIII")
_DIGEST_SIZE = 32
_MAX_ARTIFACT_BYTES = 256 << 20
_MAX_MANIFEST_BYTES = 16 << 20
_MAX_MANIFEST_DEPTH = 16
_MAX_MANIFEST_NODES = 1_000_000
_MAX_DIMENSION = 1 << 20
_MAX_CONFORMANCE_CASES = 16


class ModelArtifactError(ValueError):
    """Raised when a model artifact is malformed or incompatible."""


def _read_bounded_artifact(path: str | Path) -> bytes:
    """Read an artifact without allowing its file to exceed the v1 limit."""

    with Path(path).open("rb") as artifact_file:
        if os.fstat(artifact_file.fileno()).st_size > _MAX_ARTIFACT_BYTES:
            raise ModelArtifactError("model artifact exceeds the v1 size ceiling")
        serialized = artifact_file.read(_MAX_ARTIFACT_BYTES + 1)
    if len(serialized) > _MAX_ARTIFACT_BYTES:
        raise ModelArtifactError("model artifact exceeds the v1 size ceiling")
    return serialized


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not permitted")


def _validate_manifest_complexity(value: object) -> None:
    """Bound decoded JSON expansion before schema-specific construction."""

    nodes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_MANIFEST_NODES:
            raise ModelArtifactError("model artifact manifest is too complex")
        if depth > _MAX_MANIFEST_DEPTH:
            raise ModelArtifactError("model artifact manifest is nested too deeply")
        if isinstance(current, dict):
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            stack.extend((child, depth + 1) for child in current)


@dataclass(frozen=True, slots=True)
class _DecodedModelContainer:
    serialized: bytes
    artifact_id: str
    model_kind: int
    manifest: Mapping[str, Any]
    payload: memoryview


def _decode_model_container(
    data: bytes | bytearray | memoryview,
    *,
    expected_model_kind: int,
    expected_artifact_kind: str,
) -> _DecodedModelContainer:
    serialized = bytes(data)
    minimum_size = _CONTAINER_HEADER.size + _DIGEST_SIZE
    if len(serialized) > _MAX_ARTIFACT_BYTES:
        raise ModelArtifactError("model artifact exceeds the v1 size ceiling")
    if len(serialized) < minimum_size:
        raise ModelArtifactError("model artifact is truncated")

    content = serialized[:-_DIGEST_SIZE]
    stored_digest = serialized[-_DIGEST_SIZE:]
    calculated_digest = hashlib.sha256(content).digest()
    if stored_digest != calculated_digest:
        raise ModelArtifactError("model artifact SHA-256 check failed")

    (
        magic,
        container_version,
        header_size,
        model_kind,
        flags,
        manifest_size,
        payload_size,
        reserved,
    ) = _CONTAINER_HEADER.unpack_from(serialized)
    if magic != _MAGIC:
        raise ModelArtifactError("model artifact magic is invalid")
    if container_version != CONTAINER_VERSION:
        raise ModelArtifactError("unsupported model container version")
    if header_size != _CONTAINER_HEADER.size:
        raise ModelArtifactError("model artifact header size is invalid")
    if model_kind != expected_model_kind or flags != 0:
        raise ModelArtifactError("unsupported model artifact kind or flags")
    if reserved != bytes(len(reserved)):
        raise ModelArtifactError("model artifact reserved bytes are nonzero")
    if manifest_size > _MAX_MANIFEST_BYTES:
        raise ModelArtifactError("model artifact manifest exceeds the v1 size ceiling")
    expected_size = header_size + manifest_size + payload_size + _DIGEST_SIZE
    if expected_size != len(serialized):
        raise ModelArtifactError("model artifact section sizes are inconsistent")

    manifest_first = header_size
    manifest_last = manifest_first + manifest_size
    try:
        manifest = json.loads(
            serialized[manifest_first:manifest_last],
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ModelArtifactError("model artifact manifest is invalid JSON") from error
    if not isinstance(manifest, dict):
        raise ModelArtifactError("model artifact manifest must be an object")
    _validate_manifest_complexity(manifest)
    if manifest.get("artifact_schema") != MODEL_ARTIFACT_SCHEMA:
        raise ModelArtifactError("unsupported model artifact schema")
    if manifest.get("artifact_kind") != expected_artifact_kind:
        raise ModelArtifactError("manifest and payload kinds disagree")
    try:
        canonical_manifest = json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, UnicodeEncodeError, ValueError) as error:
        raise ModelArtifactError("model artifact manifest is invalid JSON") from error
    if canonical_manifest != serialized[manifest_first:manifest_last]:
        raise ModelArtifactError("model artifact manifest is not canonical")

    return _DecodedModelContainer(
        serialized,
        "sha256:" + calculated_digest.hex(),
        model_kind,
        manifest,
        memoryview(serialized)[manifest_last : manifest_last + payload_size],
    )


def _encode_model_container(
    model_kind: int, manifest: Mapping[str, Any], payload: bytes | bytearray
) -> bytes:
    manifest_bytes = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    header = _CONTAINER_HEADER.pack(
        _MAGIC,
        CONTAINER_VERSION,
        _CONTAINER_HEADER.size,
        model_kind,
        0,
        len(manifest_bytes),
        len(payload),
        bytes(24),
    )
    content = header + manifest_bytes + payload
    return content + hashlib.sha256(content).digest()


@dataclass(frozen=True, slots=True)
class PackedTMArtifactResult64:
    valid_example_mask: int
    prediction_mask: int
    scores: tuple[int, ...]

    def predictions(self, lane_count: int) -> tuple[int, ...]:
        if not 0 <= lane_count <= 64:
            raise ValueError("lane_count must be between zero and 64")
        expected = (1 << lane_count) - 1 if lane_count < 64 else (1 << 64) - 1
        if self.valid_example_mask != expected:
            raise ValueError("valid mask is not a contiguous lane prefix")
        return tuple((self.prediction_mask >> lane) & 1 for lane in range(lane_count))


@dataclass(frozen=True, slots=True)
class PackedTMConformanceCase:
    valid_example_mask: int
    prediction_mask: int
    feature_words: tuple[int, ...]
    scores: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PackedTMInferenceArtifact:
    """Loaded immutable `packed_tm_binary_v1` inference artifact."""

    artifact_id: str
    manifest: Mapping[str, Any]
    number_of_clauses: int
    number_of_features: int
    threshold: int
    positive_include_masks: tuple[int, ...]
    negative_include_masks: tuple[int, ...]
    conformance_cases: tuple[PackedTMConformanceCase, ...]
    _serialized: bytes

    @property
    def feature_word_count(self) -> int:
        return (self.number_of_features + 63) // 64

    @property
    def serialized(self) -> bytes:
        return self._serialized

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_bytes(self._serialized)
        return destination

    def evaluate_packed(
        self,
        feature_words: Sequence[int],
        *,
        valid_example_mask: int = (1 << 64) - 1,
    ) -> PackedTMArtifactResult64:
        return _evaluate_masks(
            self.number_of_clauses,
            self.number_of_features,
            self.threshold,
            self.positive_include_masks,
            self.negative_include_masks,
            feature_words,
            valid_example_mask,
        )

    def predict_rows(
        self, rows: Sequence[Sequence[bool | int]]
    ) -> tuple[int, ...]:
        predictions: list[int] = []
        for first in range(0, len(rows), 64):
            page = rows[first : first + 64]
            feature_words, valid = _pack_rows(page, self.number_of_features)
            result = self.evaluate_packed(
                feature_words, valid_example_mask=valid
            )
            predictions.extend(result.predictions(len(page)))
        return tuple(predictions)

    @property
    def preprocessing(self) -> PreprocessingContract | None:
        value = self.manifest.get("preprocessing")
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ModelArtifactError("artifact preprocessing contract is invalid")
        try:
            return PreprocessingContract.from_dict(value)
        except (KeyError, TypeError, ValueError) as error:
            raise ModelArtifactError("artifact preprocessing contract is invalid") from error

    def predict_records(
        self, records: Iterable[Mapping[str, object]]
    ) -> tuple[int, ...]:
        return tuple(self.iter_predict_records(records))

    def iter_predict_records(
        self, records: Iterable[Mapping[str, object]]
    ) -> Iterator[int]:
        """Predict a record stream using bounded 64-record packed pages."""

        preprocessing = self.preprocessing
        if preprocessing is None:
            raise ValueError("artifact requires precomputed Boolean features")
        page: list[tuple[bool, ...]] = []
        for record in records:
            page.append(preprocessing.materialize(record))
            if len(page) == 64:
                yield from self.predict_rows(page)
                page.clear()
        if page:
            yield from self.predict_rows(page)

    def verify_conformance(self) -> bool:
        for case in self.conformance_cases:
            actual = self.evaluate_packed(
                case.feature_words,
                valid_example_mask=case.valid_example_mask,
            )
            if (
                actual.prediction_mask != case.prediction_mask
                or actual.scores != case.scores
            ):
                return False
        return True

    @classmethod
    def from_file(cls, path: str | Path) -> "PackedTMInferenceArtifact":
        return cls.from_bytes(_read_bounded_artifact(path))

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "PackedTMInferenceArtifact":
        decoded = _decode_model_container(
            data,
            expected_model_kind=_MODEL_KIND_PACKED_TM_BINARY,
            expected_artifact_kind=PACKED_TM_PAYLOAD_KIND,
        )
        payload = decoded.payload
        if len(payload) < _PACKED_TM_HEADER.size:
            raise ModelArtifactError("packed-TM payload is truncated")
        (
            payload_version,
            clauses,
            features,
            feature_word_count,
            threshold,
            conformance_count,
            payload_flags,
            payload_reserved,
        ) = _PACKED_TM_HEADER.unpack_from(payload)
        if payload_version != PACKED_TM_PAYLOAD_VERSION:
            raise ModelArtifactError("unsupported packed-TM payload version")
        if (
            clauses == 0
            or features == 0
            or clauses > _MAX_DIMENSION
            or features > _MAX_DIMENSION
            or threshold <= 0
        ):
            raise ModelArtifactError("packed-TM dimensions are outside v1 bounds")
        if feature_word_count != (features + 63) // 64:
            raise ModelArtifactError("packed-TM feature-word count is inconsistent")
        if not 0 < conformance_count <= _MAX_CONFORMANCE_CASES:
            raise ModelArtifactError("invalid packed-TM conformance case count")
        if payload_flags != 0 or payload_reserved != 0:
            raise ModelArtifactError("packed-TM reserved fields are nonzero")

        mask_count = clauses * feature_word_count
        case_size = 16 + features * 8 + 64 * 4
        expected_payload_size = _PACKED_TM_HEADER.size + mask_count * 16 + (
            conformance_count * case_size
        )
        if expected_payload_size != len(payload):
            raise ModelArtifactError("packed-TM payload size is inconsistent")

        offset = _PACKED_TM_HEADER.size
        masks_size = mask_count * 8
        positive = struct.unpack_from(f"<{mask_count}Q", payload, offset)
        offset += masks_size
        negative = struct.unpack_from(f"<{mask_count}Q", payload, offset)
        offset += masks_size
        _validate_mask_tails(positive, clauses, features, feature_word_count)
        _validate_mask_tails(negative, clauses, features, feature_word_count)

        cases: list[PackedTMConformanceCase] = []
        for _ in range(conformance_count):
            valid, prediction = struct.unpack_from("<QQ", payload, offset)
            offset += 16
            feature_words = struct.unpack_from(f"<{features}Q", payload, offset)
            offset += features * 8
            scores = struct.unpack_from("<64i", payload, offset)
            offset += 64 * 4
            if prediction & ~valid:
                raise ModelArtifactError("conformance prediction uses an invalid lane")
            if any(
                score != 0
                for lane, score in enumerate(scores)
                if not (valid >> lane) & 1
            ):
                raise ModelArtifactError("conformance score uses an invalid lane")
            cases.append(
                PackedTMConformanceCase(valid, prediction, feature_words, scores)
            )

        _validate_packed_tm_manifest(
            decoded.manifest, clauses, features, threshold, tuple(cases)
        )

        artifact = cls(
            artifact_id=decoded.artifact_id,
            manifest=decoded.manifest,
            number_of_clauses=clauses,
            number_of_features=features,
            threshold=threshold,
            positive_include_masks=tuple(positive),
            negative_include_masks=tuple(negative),
            conformance_cases=tuple(cases),
            _serialized=decoded.serialized,
        )
        if not artifact.verify_conformance():
            raise ModelArtifactError("packed-TM conformance vectors do not match")
        return artifact


@dataclass(frozen=True, slots=True)
class LogicProgramArtifactResult64:
    valid_example_mask: int
    value_mask: int
    true_instruction_masks: tuple[int, ...]
    evaluated_instruction_masks: tuple[int, ...]

    def values(self, lane_count: int) -> tuple[int, ...]:
        if not 0 <= lane_count <= 64:
            raise ValueError("lane_count must be between zero and 64")
        expected = (1 << lane_count) - 1 if lane_count < 64 else (1 << 64) - 1
        if self.valid_example_mask != expected:
            raise ValueError("valid mask is not a contiguous lane prefix")
        return tuple((self.value_mask >> lane) & 1 for lane in range(lane_count))


@dataclass(frozen=True, slots=True)
class LogicProgramConformanceCase:
    valid_example_mask: int
    value_mask: int
    binding_words: tuple[int, ...]
    true_instruction_masks: tuple[int, ...]
    evaluated_instruction_masks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LogicProgramInferenceArtifact:
    """Loaded immutable `logic_program32_v1` inference artifact."""

    artifact_id: str
    manifest: Mapping[str, Any]
    program: LogicProgram32
    binding_names: tuple[str, ...]
    conformance_cases: tuple[LogicProgramConformanceCase, ...]
    _serialized: bytes

    @property
    def serialized(self) -> bytes:
        return self._serialized

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_bytes(self._serialized)
        return destination

    def evaluate_packed(
        self,
        binding_words: Sequence[int],
        *,
        valid_example_mask: int = (1 << 64) - 1,
    ) -> LogicProgramArtifactResult64:
        return _evaluate_logic_program_packed(
            self.program, binding_words, valid_example_mask
        )

    def predict_rows(
        self, rows: Sequence[Sequence[bool | int]]
    ) -> tuple[int, ...]:
        values: list[int] = []
        for first in range(0, len(rows), 64):
            page = rows[first : first + 64]
            binding_words, valid = _pack_rows(page, len(self.binding_names))
            result = self.evaluate_packed(
                binding_words, valid_example_mask=valid
            )
            values.extend(result.values(len(page)))
        return tuple(values)

    def verify_conformance(self) -> bool:
        for case in self.conformance_cases:
            actual = self.evaluate_packed(
                case.binding_words,
                valid_example_mask=case.valid_example_mask,
            )
            if (
                actual.value_mask != case.value_mask
                or actual.true_instruction_masks != case.true_instruction_masks
                or actual.evaluated_instruction_masks
                != case.evaluated_instruction_masks
            ):
                return False
        return True

    @classmethod
    def from_file(cls, path: str | Path) -> "LogicProgramInferenceArtifact":
        return cls.from_bytes(_read_bounded_artifact(path))

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray | memoryview
    ) -> "LogicProgramInferenceArtifact":
        decoded = _decode_model_container(
            data,
            expected_model_kind=_MODEL_KIND_LOGIC_PROGRAM32,
            expected_artifact_kind=LOGIC_PROGRAM_PAYLOAD_KIND,
        )
        payload = decoded.payload
        case_size = 16 + len(LOGIC_AST_VARIABLES) * 8 + 64 * 4 * 2
        if len(payload) < _LOGIC_PROGRAM_HEADER.size:
            raise ModelArtifactError("Logic-program payload is truncated")
        (
            payload_version,
            instruction_count,
            root_instruction,
            binding_count,
            conformance_count,
            payload_flags,
            reserved0,
            reserved1,
        ) = _LOGIC_PROGRAM_HEADER.unpack_from(payload)
        if payload_version != LOGIC_PROGRAM_PAYLOAD_VERSION:
            raise ModelArtifactError("unsupported Logic-program payload version")
        if (
            not 1 <= instruction_count <= LOGIC_PROGRAM_CAPACITY
            or root_instruction != instruction_count - 1
            or binding_count != len(LOGIC_AST_VARIABLES)
            or conformance_count != 1
            or payload_flags != 0
            or reserved0 != 0
            or reserved1 != 0
        ):
            raise ModelArtifactError("Logic-program payload header is invalid")
        expected_size = (
            _LOGIC_PROGRAM_HEADER.size
            + instruction_count * _LOGIC_INSTRUCTION.size
            + case_size
        )
        if len(payload) != expected_size:
            raise ModelArtifactError("Logic-program payload size is inconsistent")

        offset = _LOGIC_PROGRAM_HEADER.size
        instructions: list[FixedLogicInstruction] = []
        for _ in range(instruction_count):
            operand_mask, opcode_value, argument, reserved = (
                _LOGIC_INSTRUCTION.unpack_from(payload, offset)
            )
            offset += _LOGIC_INSTRUCTION.size
            if reserved != 0:
                raise ModelArtifactError("Logic instruction reserved field is nonzero")
            try:
                opcode = FixedLogicOpcode(opcode_value)
                instructions.append(
                    FixedLogicInstruction(opcode, operand_mask, argument)
                )
            except ValueError as error:
                raise ModelArtifactError("Logic instruction is invalid") from error
        try:
            program = LogicProgram32(tuple(instructions), root_instruction)
        except ValueError as error:
            raise ModelArtifactError("Logic program is invalid") from error

        valid, value_mask = struct.unpack_from("<QQ", payload, offset)
        offset += 16
        binding_words = struct.unpack_from(
            f"<{binding_count}Q", payload, offset
        )
        offset += binding_count * 8
        true_masks = struct.unpack_from("<64I", payload, offset)
        offset += 64 * 4
        evaluated_masks = struct.unpack_from("<64I", payload, offset)
        if value_mask & ~valid or any(word & ~valid for word in binding_words):
            raise ModelArtifactError("Logic conformance data uses an invalid lane")
        if any(
            (true_masks[lane] != 0 or evaluated_masks[lane] != 0)
            and not ((valid >> lane) & 1)
            for lane in range(64)
        ):
            raise ModelArtifactError("Logic diagnostics use an invalid lane")
        case = LogicProgramConformanceCase(
            valid,
            value_mask,
            tuple(binding_words),
            tuple(true_masks),
            tuple(evaluated_masks),
        )
        binding_names = _validate_logic_program_manifest(
            decoded.manifest, program, (case,)
        )
        artifact = cls(
            decoded.artifact_id,
            decoded.manifest,
            program,
            binding_names,
            (case,),
            decoded.serialized,
        )
        if not artifact.verify_conformance():
            raise ModelArtifactError("Logic-program conformance vectors do not match")
        return artifact


@dataclass(frozen=True, slots=True)
class MaskedThresholdArtifactResult64:
    valid_example_mask: int
    value_mask: int
    matched_counts: tuple[int, ...]
    matched_slot_words: tuple[int, ...]
    missing_slot_words: tuple[int, ...]

    def values(self, lane_count: int) -> tuple[int, ...]:
        if not 0 <= lane_count <= 64:
            raise ValueError("lane_count must be between zero and 64")
        expected = (1 << lane_count) - 1 if lane_count < 64 else (1 << 64) - 1
        if self.valid_example_mask != expected:
            raise ValueError("valid mask is not a contiguous lane prefix")
        return tuple((self.value_mask >> lane) & 1 for lane in range(lane_count))


@dataclass(frozen=True, slots=True)
class MaskedThresholdConformanceCase:
    valid_example_mask: int
    value_mask: int
    selected_input_words: tuple[int, ...]
    matched_counts: tuple[int, ...]
    matched_selected_words: tuple[int, ...]
    missing_selected_words: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MaskedThresholdInferenceArtifact:
    """Loaded immutable `masked_threshold_v1` inference artifact."""

    artifact_id: str
    manifest: Mapping[str, Any]
    slot_count: int
    minimum_true: int
    selected_count: int
    selection_words: tuple[int, ...]
    conformance_cases: tuple[MaskedThresholdConformanceCase, ...]
    _serialized: bytes

    @property
    def selected_slots(self) -> tuple[int, ...]:
        return tuple(
            word_index * 64 + bit
            for word_index, word in enumerate(self.selection_words)
            for bit in range(64)
            if word & (1 << bit)
        )

    @property
    def serialized(self) -> bytes:
        return self._serialized

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_bytes(self._serialized)
        return destination

    def evaluate_packed(
        self,
        slot_words: Sequence[int],
        *,
        valid_example_mask: int = (1 << 64) - 1,
    ) -> MaskedThresholdArtifactResult64:
        return _evaluate_masked_threshold_packed(
            self.slot_count,
            self.minimum_true,
            self.selection_words,
            slot_words,
            valid_example_mask,
        )

    def predict_rows(
        self, rows: Sequence[Sequence[bool | int]]
    ) -> tuple[int, ...]:
        values: list[int] = []
        for first in range(0, len(rows), 64):
            page = rows[first : first + 64]
            slot_words, valid = _pack_rows(page, self.slot_count)
            result = self.evaluate_packed(slot_words, valid_example_mask=valid)
            values.extend(result.values(len(page)))
        return tuple(values)

    def verify_conformance(self) -> bool:
        selected_slots = self.selected_slots
        for case in self.conformance_cases:
            slot_words = [0] * self.slot_count
            for slot, word in zip(selected_slots, case.selected_input_words):
                slot_words[slot] = word
            actual = self.evaluate_packed(
                slot_words, valid_example_mask=case.valid_example_mask
            )
            if (
                actual.value_mask != case.value_mask
                or actual.matched_counts != case.matched_counts
                or tuple(actual.matched_slot_words[slot] for slot in selected_slots)
                != case.matched_selected_words
                or tuple(actual.missing_slot_words[slot] for slot in selected_slots)
                != case.missing_selected_words
            ):
                return False
        return True

    @classmethod
    def from_file(cls, path: str | Path) -> "MaskedThresholdInferenceArtifact":
        return cls.from_bytes(_read_bounded_artifact(path))

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray | memoryview
    ) -> "MaskedThresholdInferenceArtifact":
        decoded = _decode_model_container(
            data,
            expected_model_kind=_MODEL_KIND_MASKED_THRESHOLD,
            expected_artifact_kind=MASKED_THRESHOLD_PAYLOAD_KIND,
        )
        payload = decoded.payload
        if len(payload) < _MASKED_THRESHOLD_HEADER.size:
            raise ModelArtifactError("masked-threshold payload is truncated")
        (
            payload_version,
            slot_count,
            selection_word_count,
            minimum_true,
            selected_count,
            conformance_count,
            payload_flags,
            payload_reserved,
        ) = _MASKED_THRESHOLD_HEADER.unpack_from(payload)
        if payload_version != MASKED_THRESHOLD_PAYLOAD_VERSION:
            raise ModelArtifactError("unsupported masked-threshold payload version")
        if (
            slot_count not in (1024, 4096)
            or selection_word_count != slot_count // 64
            or minimum_true > selected_count
            or selected_count > slot_count
            or conformance_count != 1
            or payload_flags != 0
            or payload_reserved != 0
        ):
            raise ModelArtifactError("masked-threshold payload header is invalid")
        case_size = 16 + selected_count * 24 + 64 * 4
        expected_size = (
            _MASKED_THRESHOLD_HEADER.size
            + selection_word_count * 8
            + case_size
        )
        if len(payload) != expected_size:
            raise ModelArtifactError("masked-threshold payload size is inconsistent")

        offset = _MASKED_THRESHOLD_HEADER.size
        selection_words = struct.unpack_from(
            f"<{selection_word_count}Q", payload, offset
        )
        offset += selection_word_count * 8
        if sum(word.bit_count() for word in selection_words) != selected_count:
            raise ModelArtifactError("masked-threshold selection count is inconsistent")
        valid, value_mask = struct.unpack_from("<QQ", payload, offset)
        offset += 16
        selected_inputs = struct.unpack_from(
            f"<{selected_count}Q", payload, offset
        )
        offset += selected_count * 8
        matched_counts = struct.unpack_from("<64I", payload, offset)
        offset += 64 * 4
        matched_selected = struct.unpack_from(
            f"<{selected_count}Q", payload, offset
        )
        offset += selected_count * 8
        missing_selected = struct.unpack_from(
            f"<{selected_count}Q", payload, offset
        )
        if value_mask & ~valid or any(word & ~valid for word in selected_inputs):
            raise ModelArtifactError("masked-threshold conformance uses an invalid lane")
        for lane, count in enumerate(matched_counts):
            if (not ((valid >> lane) & 1) and count != 0) or count > selected_count:
                raise ModelArtifactError("masked-threshold count is invalid")
        for input_word, matched_word, missing_word in zip(
            selected_inputs, matched_selected, missing_selected
        ):
            if (
                matched_word != (input_word & valid)
                or missing_word != ((~input_word) & valid & ((1 << 64) - 1))
            ):
                raise ModelArtifactError("masked-threshold diagnostics are invalid")
        case = MaskedThresholdConformanceCase(
            valid,
            value_mask,
            tuple(selected_inputs),
            tuple(matched_counts),
            tuple(matched_selected),
            tuple(missing_selected),
        )
        _validate_masked_threshold_manifest(
            decoded.manifest,
            slot_count,
            minimum_true,
            selected_count,
            tuple(selection_words),
            (case,),
        )
        artifact = cls(
            decoded.artifact_id,
            decoded.manifest,
            slot_count,
            minimum_true,
            selected_count,
            tuple(selection_words),
            (case,),
            decoded.serialized,
        )
        if not artifact.verify_conformance():
            raise ModelArtifactError(
                "masked-threshold conformance vectors do not match"
            )
        return artifact


@dataclass(frozen=True, slots=True)
class GraphTMInferenceArtifact:
    artifact_id: str
    manifest: Mapping[str, Any]
    graph_depth: int
    graph_clauses: int
    graph_hv_dim: int
    components: tuple[Any, ...]  # DeepClause, typed as Any to avoid import cycle
    weights: tuple[tuple[int, int], ...]
    conformance_graphs: tuple[Any, ...]  # GraphInput
    expected_labels: tuple[int, ...]
    serialized: bytes

    def write(self, path: str | Path) -> None:
        with open(path, "wb") as handle:
            handle.write(self.serialized)

    def verify_conformance(self) -> bool:
        if GraphTsetlinMachine is None or GraphInput is None:
            return False
        try:
            gtm = GraphTsetlinMachine(
                depth=self.graph_depth,
                clauses=self.graph_clauses,
                hv_dim=self.graph_hv_dim,
            )
            gtm._components = list(self.components)  # type: ignore[attr-defined]
            gtm._weights = [list(w) for w in self.weights]  # type: ignore[attr-defined]
            for graph, expected in zip(self.conformance_graphs, self.expected_labels):
                pred = int(gtm.predict(graph))
                if pred != int(expected):
                    return False
            return True
        except Exception:
            return False

    def predict(self, graph: Any) -> int:
        if GraphTsetlinMachine is None:
            raise ModelArtifactError("graph extra not available")
        gtm = GraphTsetlinMachine(
            depth=self.graph_depth,
            clauses=self.graph_clauses,
            hv_dim=self.graph_hv_dim,
        )
        gtm._components = list(self.components)  # type: ignore[attr-defined]
        gtm._weights = [list(w) for w in self.weights]  # type: ignore[attr-defined]
        return int(gtm.predict(graph))

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "GraphTMInferenceArtifact":
        decoded = _decode_model_container(
            bytes(data),
            expected_model_kind=_MODEL_KIND_GRAPH_TM,
            expected_artifact_kind=GRAPH_TM_PAYLOAD_KIND,
        )
        payload = decoded.payload
        if len(payload) < _GRAPH_TM_HEADER.size:
            raise ModelArtifactError("graph-TM payload is truncated")
        (
            payload_version,
            depth,
            clauses,
            hv_dim,
            edge_type_count,
            conformance_count,
            flags,
            reserved,
        ) = _GRAPH_TM_HEADER.unpack_from(payload, 0)
        if payload_version != GRAPH_TM_PAYLOAD_VERSION or reserved != 0 or flags != 0:
            raise ModelArtifactError("graph-TM payload has unsupported flags")
        if not 1 <= depth <= MAX_GRAPH_DEPTH or not 1 <= clauses <= MAX_GRAPH_CLAUSES or hv_dim not in (256, 512, 1024, 2048, 4096, 8192):
            raise ModelArtifactError("graph-TM payload describes an unsupported configuration")
        if not 1 <= conformance_count <= 256:
            raise ModelArtifactError("graph-TM conformance length is invalid")
        if edge_type_count != 16:
            raise ModelArtifactError("graph-TM edge_type_count must be 16")
        offset = _GRAPH_TM_HEADER.size
        # weights: clauses * 2 int32
        weight_bytes = clauses * 2 * 4
        if len(payload) < offset + weight_bytes:
            raise ModelArtifactError("graph-TM payload is truncated at weights")
        weights: list[tuple[int, int]] = []
        for idx in range(clauses):
            w0, w1 = struct.unpack_from("<ii", payload, offset + idx * 8)
            if not -1_000_000 <= w0 <= 1_000_000 or not -1_000_000 <= w1 <= 1_000_000:
                raise ModelArtifactError("graph-TM weight is out of range")
            weights.append((int(w0), int(w1)))
        offset += weight_bytes
        conformance_graphs: list[Any] = []
        expected_labels: list[int] = []
        for _ in range(conformance_count):
            if len(payload) < offset + 8:
                raise ModelArtifactError("graph-TM payload is truncated at conformance graph")
            graph_len, expected = struct.unpack_from("<II", payload, offset)
            offset += 8
            if graph_len > 1 << 20 or expected not in (0, 1):
                raise ModelArtifactError("graph-TM conformance entry is invalid")
            if len(payload) < offset + graph_len:
                raise ModelArtifactError("graph-TM payload is truncated at graph bytes")
            graph_bytes = bytes(payload[offset : offset + graph_len])
            offset += graph_len
            try:
                graph_dict = json.loads(graph_bytes.decode("utf-8"))
                if not isinstance(graph_dict, dict):
                    raise ModelArtifactError("graph-TM graph payload is not an object")
                # Strict ptm.graph.v1 grammar — no coercion
                raw_node_count = graph_dict.get("node_count")
                if type(raw_node_count) is not int:
                    raise ModelArtifactError("graph-TM node_count must be strict int")
                node_count = raw_node_count
                raw_schema = graph_dict.get("schema")
                if raw_schema != "ptm.graph.v1":
                    raise ModelArtifactError("graph-TM schema must be ptm.graph.v1")
                # Validate exact expected keys/types before construction
                if not isinstance(graph_dict.get("edges"), list):
                    raise ModelArtifactError("graph-TM edges must be list")
                if not isinstance(graph_dict.get("node_properties"), list):
                    raise ModelArtifactError("graph-TM node_properties must be list")
                if not isinstance(graph_dict.get("edge_types"), list):
                    raise ModelArtifactError("graph-TM edge_types must be list")
                edges = graph_dict.get("edges") or []
                node_props_raw = graph_dict.get("node_properties") or []
                # Strict: node_properties length must equal node_count (native requires exact)
                if len(node_props_raw) != node_count:
                    raise ModelArtifactError(f"graph-TM node_properties length {len(node_props_raw)} != node_count {node_count}")
                # Strict: edge_types values must match edge-derived set (native checks size and values)
                raw_edge_types = graph_dict.get("edge_types") or []
                # Build canonical edge-derived set
                derived_types: set[str] = set()
                for e in edges:
                    et = e[2] if len(e) == 3 else None
                    if isinstance(et, int) and type(et) is int:
                        derived_types.add(f"int:{et}")
                    elif isinstance(et, str):
                        derived_types.add(f"str:{et}")
                canonical_raw = set()
                for et in raw_edge_types:
                    if isinstance(et, int) and type(et) is int:
                        canonical_raw.add(f"int:{et}")
                    elif isinstance(et, str):
                        canonical_raw.add(f"str:{et}")
                    else:
                        raise ModelArtifactError("graph-TM edge_types entry must be int or str")
                if canonical_raw != derived_types:
                    raise ModelArtifactError(f"graph-TM edge_types {sorted(canonical_raw)} != derived {sorted(derived_types)}")
                # Validate edges are [int,int, int|str] with strict types; GraphInput.create will enforce bounds but we pre-check strictness
                for e in edges:
                    if not isinstance(e, (list, tuple)) or len(e) != 3:
                        raise ModelArtifactError("graph-TM edge must be 3-tuple")
                    s, d, et = e
                    if type(s) is not int or type(d) is not int:
                        raise ModelArtifactError("graph-TM edge endpoints must be strict int")
                    if not (isinstance(et, int) and type(et) is int) and not isinstance(et, str):
                        raise ModelArtifactError("graph-TM edge_type must be int or str")
                    if isinstance(et, bool):
                        raise ModelArtifactError("graph-TM edge_type bool not allowed")
                props: dict[int, list[object]] = {}
                for idx, plist in enumerate(node_props_raw):
                    if not isinstance(plist, list):
                        raise ModelArtifactError("graph-TM node_properties entry must be list")
                    if plist:
                        # Validate each property scalar strictness at trust boundary
                        for p in plist:
                            if isinstance(p, bool):
                                continue
                            if isinstance(p, int) and type(p) is int:
                                if not -(1 << 53) <= p <= (1 << 53):
                                    raise ModelArtifactError("graph-TM integer property out of 53-bit range")
                                continue
                            if isinstance(p, float):
                                import math as _math

                                if not _math.isfinite(p):
                                    raise ModelArtifactError("graph-TM float property must be finite")
                                continue
                            if isinstance(p, str):
                                if not p or len(p) > 256 or any(ord(c) < 0x20 for c in p):
                                    raise ModelArtifactError("graph-TM string property invalid")
                                continue
                            raise ModelArtifactError("graph-TM property must be str, int, bool, or finite float")
                        props[idx] = list(plist)  # type: ignore[assignment]
                graph = GraphInput.create(node_count=node_count, edges=[tuple(e) for e in edges], node_properties=props)  # type: ignore[arg-type]
            except ModelArtifactError:
                raise
            except Exception as error:
                raise ModelArtifactError("graph-TM graph payload is invalid") from error
            conformance_graphs.append(graph)
            expected_labels.append(int(expected))
        if offset != len(payload):
            raise ModelArtifactError("graph-TM payload has trailing bytes")
        # Validate manifest against payload — binary is authoritative
        _validate_graph_tm_manifest(
            decoded.manifest,
            depth,
            clauses,
            hv_dim,
            edge_type_count,
            tuple(weights),
            tuple(conformance_graphs),
            tuple(expected_labels),
        )
        # Reconstruct components from manifest graph.components (DeepClause.from_dict)
        raw_components = decoded.manifest.get("graph", {}).get("components", []) if isinstance(decoded.manifest.get("graph"), dict) else []
        components: list[Any] = []
        if DeepClause is not None:
            for raw in raw_components:
                try:
                    components.append(DeepClause.from_dict(raw))  # type: ignore[union-attr]
                except Exception as error:
                    raise ModelArtifactError("graph-TM component is invalid") from error
        else:
            components = list(raw_components)
        artifact = cls(
            decoded.artifact_id,
            decoded.manifest,
            depth,
            clauses,
            hv_dim,
            tuple(components),
            tuple(weights),
            tuple(conformance_graphs),
            tuple(expected_labels),
            decoded.serialized,
        )
        if not artifact.verify_conformance():
            raise ModelArtifactError("graph-TM conformance graphs do not match")
        return artifact


InferenceArtifact = (
    PackedTMInferenceArtifact
    | LogicProgramInferenceArtifact
    | MaskedThresholdInferenceArtifact
    | GraphTMInferenceArtifact
)


def load_model_artifact_from_bytes(
    data: bytes | bytearray | memoryview,
) -> InferenceArtifact:
    """Load and validate an inference artifact from serialized bytes."""
    serialized = bytes(data)
    if len(serialized) < _CONTAINER_HEADER.size:
        raise ModelArtifactError("model artifact is truncated")
    model_kind = int.from_bytes(serialized[16:20], "little")
    if model_kind == _MODEL_KIND_PACKED_TM_BINARY:
        return PackedTMInferenceArtifact.from_bytes(serialized)
    if model_kind == _MODEL_KIND_LOGIC_PROGRAM32:
        return LogicProgramInferenceArtifact.from_bytes(serialized)
    if model_kind == _MODEL_KIND_MASKED_THRESHOLD:
        return MaskedThresholdInferenceArtifact.from_bytes(serialized)
    if model_kind == _MODEL_KIND_GRAPH_TM:
        return GraphTMInferenceArtifact.from_bytes(serialized)
    raise ModelArtifactError("unsupported model artifact kind")


def load_model_artifact(path: str | Path) -> InferenceArtifact:
    """Load and validate an inference artifact from a filesystem path."""
    return load_model_artifact_from_bytes(_read_bounded_artifact(path))


def export_packed_tm(
    snapshot: TMSnapshot,
    *,
    name: str,
    path: str | Path | None = None,
    description: str = "",
    authors: Sequence[str] = (),
    license: str = "unspecified",
    citations: Sequence[str] = (),
    intended_use: str = "research",
    limitations: str = "research prototype",
    feature_names: Sequence[str] | None = None,
    feature_literal_ids: Sequence[int] | None = None,
    feature_catalog_version: str = "anonymous-v1",
    output_labels: Sequence[str] = ("0", "1"),
    validation_rows: Sequence[Sequence[bool | int]] | None = None,
    validation_signature: Mapping[str, Any] | None = None,
    restoration_reference: Mapping[str, Any] | None = None,
    preprocessing: PreprocessingContract | None = None,
    validation_records: Sequence[Mapping[str, object]] | None = None,
) -> PackedTMInferenceArtifact:
    """Freeze a scalar TM snapshot into a deterministic inference-only artifact."""

    _validate_snapshot(snapshot)
    if not name:
        raise ValueError("model artifact name cannot be empty")
    if not feature_catalog_version:
        raise ValueError("feature catalog version cannot be empty")
    if feature_names is None:
        names = tuple(f"x{index}" for index in range(snapshot.number_of_features))
    else:
        names = tuple(str(value) for value in feature_names)
    if len(names) != snapshot.number_of_features:
        raise ValueError("feature_names has the wrong width")
    if any(not value for value in names) or len(set(names)) != len(names):
        raise ValueError("feature names must be nonempty and unique")
    if preprocessing is not None and len(preprocessing.outputs) != (
        snapshot.number_of_features
    ):
        raise ValueError("preprocessing output width differs from the TM snapshot")
    preprocessing_ids = preprocessing.literal_ids if preprocessing is not None else ()
    literal_ids = tuple(feature_literal_ids or preprocessing_ids)
    if preprocessing_ids and literal_ids != preprocessing_ids:
        raise ValueError("feature literal IDs disagree with preprocessing outputs")
    if literal_ids and len(literal_ids) != snapshot.number_of_features:
        raise ValueError("feature_literal_ids has the wrong width")
    if any(value < 0 or value >= 1 << 64 for value in literal_ids):
        raise ValueError("feature literal IDs must be unsigned 64-bit values")
    if len(set(literal_ids)) != len(literal_ids):
        raise ValueError("feature literal IDs must be unique")
    labels = tuple(str(value) for value in output_labels)
    if len(labels) != 2 or any(not value for value in labels) or labels[0] == labels[1]:
        raise ValueError("binary artifacts require two distinct nonempty output labels")

    feature_word_count = (snapshot.number_of_features + 63) // 64
    mask_count = snapshot.number_of_clauses * feature_word_count
    positive = [0] * mask_count
    negative = [0] * mask_count
    for clause, states in enumerate(snapshot.states):
        for feature in range(snapshot.number_of_features):
            mask_index = clause * feature_word_count + feature // 64
            bit = 1 << (feature % 64)
            if states[feature * 2] > snapshot.states_per_action:
                positive[mask_index] |= bit
            if states[feature * 2 + 1] > snapshot.states_per_action:
                negative[mask_index] |= bit

    record_rows = (
        preprocessing.materialize_many(validation_records)
        if preprocessing is not None and validation_records is not None
        else None
    )
    if validation_records is not None and preprocessing is None:
        raise ValueError("validation_records require a preprocessing contract")
    supplied_rows = (
        tuple(tuple(bool(value) for value in row) for row in validation_rows)
        if validation_rows is not None
        else None
    )
    if supplied_rows is not None and record_rows is not None and supplied_rows != record_rows:
        raise ValueError("validation rows disagree with raw-record preprocessing")
    if supplied_rows is not None:
        rows = supplied_rows
    elif record_rows is not None:
        rows = record_rows
    else:
        rows = _default_conformance_rows(snapshot.number_of_features)
    if len(rows) > _MAX_CONFORMANCE_CASES * 64:
        raise ValueError("validation_rows exceeds the v1 conformance ceiling")
    if not rows:
        raise ValueError("at least one validation row is required")
    cases: list[PackedTMConformanceCase] = []
    for first in range(0, len(rows), 64):
        page = rows[first : first + 64]
        feature_words, valid = _pack_rows(page, snapshot.number_of_features)
        result = _evaluate_masks(
            snapshot.number_of_clauses,
            snapshot.number_of_features,
            snapshot.threshold,
            positive,
            negative,
            feature_words,
            valid,
        )
        for lane, row in enumerate(page):
            oracle_score, oracle_prediction = _evaluate_snapshot_row(snapshot, row)
            if (
                result.scores[lane] != oracle_score
                or ((result.prediction_mask >> lane) & 1) != oracle_prediction
            ):
                raise ModelArtifactError(
                    "packed-TM lowering differs from the training snapshot oracle"
                )
        cases.append(
            PackedTMConformanceCase(
                valid,
                result.prediction_mask,
                feature_words,
                result.scores,
            )
        )

    manifest: dict[str, Any] = {
        "artifact_kind": PACKED_TM_PAYLOAD_KIND,
        "artifact_schema": MODEL_ARTIFACT_SCHEMA,
        "container_digest": "sha256-trailer-v1",
        "description": description,
        "features": {
            "catalog_version": feature_catalog_version,
            "kind": "packed_boolean_features",
            "literal_ids": [str(value) for value in literal_ids],
            "materialization": (
                "precomputed_or_raw_record_v1"
                if preprocessing is not None
                else "precomputed"
            ),
            "names": list(names),
        },
        "model": {
            "clause_polarity": "even_positive_odd_negative",
            "number_of_clauses": snapshot.number_of_clauses,
            "number_of_features": snapshot.number_of_features,
            "score_clamp": "symmetric_threshold",
            "threshold": snapshot.threshold,
        },
        "ports": {
            "inputs": [
                {
                    "dtype": "uint64",
                    "layout": "feature_major_packed64",
                    "name": "features",
                    "shape": [snapshot.number_of_features],
                },
                {"dtype": "uint64", "name": "valid_mask", "shape": []},
            ],
            "outputs": [
                {"dtype": "uint64", "name": "predictions", "shape": []},
                {"dtype": "int32", "name": "scores", "shape": [64]},
            ],
        },
        "producer": {"name": "prolog-tsetlin-machine", "version": __version__},
        "research": {
            "authors": [str(value) for value in authors],
            "citations": [str(value) for value in citations],
            "intended_use": intended_use,
            "license": license,
            "limitations": limitations,
        },
        "task": {"kind": "binary_classification", "labels": list(labels)},
        "title": name,
        "validation": {
            "conformance_case_count": len(cases),
            "conformance_example_count": len(rows),
            "signature": dict(validation_signature or {}),
        },
    }
    if restoration_reference is not None:
        manifest["restoration_reference"] = dict(restoration_reference)
    if preprocessing is not None:
        manifest["preprocessing"] = preprocessing.to_dict()
    payload = bytearray(
        _PACKED_TM_HEADER.pack(
            PACKED_TM_PAYLOAD_VERSION,
            snapshot.number_of_clauses,
            snapshot.number_of_features,
            feature_word_count,
            snapshot.threshold,
            len(cases),
            0,
            0,
        )
    )
    payload.extend(struct.pack(f"<{mask_count}Q", *positive))
    payload.extend(struct.pack(f"<{mask_count}Q", *negative))
    for case in cases:
        payload.extend(struct.pack("<QQ", case.valid_example_mask, case.prediction_mask))
        payload.extend(struct.pack(f"<{snapshot.number_of_features}Q", *case.feature_words))
        payload.extend(struct.pack("<64i", *case.scores))

    serialized = _encode_model_container(
        _MODEL_KIND_PACKED_TM_BINARY, manifest, payload
    )
    artifact = PackedTMInferenceArtifact.from_bytes(serialized)
    if path is not None:
        artifact.write(path)
    return artifact


def export_logic_program(
    program: LogicProgram32,
    *,
    name: str,
    path: str | Path | None = None,
    description: str = "",
    authors: Sequence[str] = (),
    license: str = "unspecified",
    citations: Sequence[str] = (),
    intended_use: str = "research",
    limitations: str = "research prototype",
    binding_names: Sequence[str] = LOGIC_AST_VARIABLES,
    binding_literal_ids: Sequence[int] | None = None,
    binding_catalog_version: str = "logic-bindings-v1",
    output_labels: Sequence[str] = ("false", "true"),
    validation_signature: Mapping[str, Any] | None = None,
    restoration_reference: Mapping[str, Any] | None = None,
) -> LogicProgramInferenceArtifact:
    """Freeze a fixed Logic program into a deterministic inference artifact."""

    if not isinstance(program, LogicProgram32):
        raise TypeError("program must be a LogicProgram32")
    if not name:
        raise ValueError("model artifact name cannot be empty")
    names = tuple(str(value) for value in binding_names)
    if (
        len(names) != len(LOGIC_AST_VARIABLES)
        or any(not value for value in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Logic binding names must be five nonempty unique values")
    if not binding_catalog_version:
        raise ValueError("binding catalog version cannot be empty")
    literal_ids = tuple(binding_literal_ids or ())
    if literal_ids and len(literal_ids) != len(LOGIC_AST_VARIABLES):
        raise ValueError("binding_literal_ids has the wrong width")
    if any(value < 0 or value >= 1 << 64 for value in literal_ids):
        raise ValueError("binding literal IDs must be unsigned 64-bit values")
    if len(set(literal_ids)) != len(literal_ids):
        raise ValueError("binding literal IDs must be unique")
    labels = tuple(str(value) for value in output_labels)
    if len(labels) != 2 or any(not value for value in labels) or labels[0] == labels[1]:
        raise ValueError("Logic artifacts require two distinct nonempty labels")

    rows = tuple(
        tuple(bool(binding_bits & (1 << index)) for index in range(5))
        for binding_bits in range(32)
    )
    binding_words, valid = _pack_rows(rows, len(LOGIC_AST_VARIABLES))
    result = _evaluate_logic_program_packed(program, binding_words, valid)
    for lane, row in enumerate(rows):
        oracle = program.evaluate(row)
        if (
            ((result.value_mask >> lane) & 1) != int(oracle.value)
            or result.true_instruction_masks[lane]
            != oracle.true_instruction_mask
            or result.evaluated_instruction_masks[lane]
            != oracle.evaluated_instruction_mask
        ):
            raise ModelArtifactError(
                "Logic-program lowering differs from the program oracle"
            )
    case = LogicProgramConformanceCase(
        valid,
        result.value_mask,
        binding_words,
        result.true_instruction_masks,
        result.evaluated_instruction_masks,
    )
    manifest: dict[str, Any] = {
        "artifact_kind": LOGIC_PROGRAM_PAYLOAD_KIND,
        "artifact_schema": MODEL_ARTIFACT_SCHEMA,
        "bindings": {
            "catalog_version": binding_catalog_version,
            "kind": "boolean_bindings",
            "literal_ids": [str(value) for value in literal_ids],
            "materialization": "precomputed",
            "names": list(names),
        },
        "container_digest": "sha256-trailer-v1",
        "description": description,
        "model": {
            "binding_count": len(LOGIC_AST_VARIABLES),
            "instruction_count": len(program.instructions),
            "opcodes": {
                opcode.name.lower(): int(opcode) for opcode in FixedLogicOpcode
            },
            "program_id": program.program_id,
            "root_instruction": program.root_instruction,
        },
        "ports": {
            "inputs": [
                {
                    "dtype": "uint64",
                    "layout": "binding_major_packed64",
                    "name": "bindings",
                    "shape": [len(LOGIC_AST_VARIABLES)],
                },
                {"dtype": "uint64", "name": "valid_mask", "shape": []},
            ],
            "outputs": [
                {"dtype": "uint64", "name": "values", "shape": []},
                {
                    "dtype": "uint32",
                    "name": "true_instruction_masks",
                    "shape": [64],
                },
                {
                    "dtype": "uint32",
                    "name": "evaluated_instruction_masks",
                    "shape": [64],
                },
            ],
        },
        "producer": {"name": "prolog-tsetlin-machine", "version": __version__},
        "research": {
            "authors": [str(value) for value in authors],
            "citations": [str(value) for value in citations],
            "intended_use": intended_use,
            "license": license,
            "limitations": limitations,
        },
        "task": {"kind": "boolean_function", "labels": list(labels)},
        "title": name,
        "validation": {
            "conformance_case_count": 1,
            "conformance_example_count": 32,
            "signature": dict(validation_signature or {}),
        },
    }
    if restoration_reference is not None:
        manifest["restoration_reference"] = dict(restoration_reference)

    payload = bytearray(
        _LOGIC_PROGRAM_HEADER.pack(
            LOGIC_PROGRAM_PAYLOAD_VERSION,
            len(program.instructions),
            program.root_instruction,
            len(LOGIC_AST_VARIABLES),
            1,
            0,
            0,
            0,
        )
    )
    for instruction in program.instructions:
        payload.extend(
            _LOGIC_INSTRUCTION.pack(
                instruction.operand_mask,
                int(instruction.opcode),
                instruction.argument,
                0,
            )
        )
    payload.extend(struct.pack("<QQ", case.valid_example_mask, case.value_mask))
    payload.extend(struct.pack("<5Q", *case.binding_words))
    payload.extend(struct.pack("<64I", *case.true_instruction_masks))
    payload.extend(struct.pack("<64I", *case.evaluated_instruction_masks))

    serialized = _encode_model_container(
        _MODEL_KIND_LOGIC_PROGRAM32, manifest, payload
    )
    artifact = LogicProgramInferenceArtifact.from_bytes(serialized)
    if path is not None:
        artifact.write(path)
    return artifact


def export_masked_threshold(
    source: PAArtifact,
    *,
    name: str,
    path: str | Path | None = None,
    description: str = "",
    authors: Sequence[str] = (),
    license: str = "unspecified",
    citations: Sequence[str] = (),
    intended_use: str = "research",
    limitations: str = "research prototype",
    output_labels: Sequence[str] = ("false", "true"),
) -> MaskedThresholdInferenceArtifact:
    """Package a validated Class II PA threshold kernel for static inference."""

    if not isinstance(source, PAArtifact) or not source.verify_artifact_id():
        raise ValueError("source must be a valid content-addressed PA artifact")
    if not name:
        raise ValueError("model artifact name cannot be empty")
    labels = tuple(str(value) for value in output_labels)
    if len(labels) != 2 or any(not value for value in labels) or labels[0] == labels[1]:
        raise ValueError("PA artifacts require two distinct nonempty labels")

    slot_count = source.input_shape.bit_count
    selection_word_count = slot_count // 64
    selection_words = [0] * selection_word_count
    for slot in source.payload.selected_slots:
        selection_words[slot // 64] |= 1 << (slot % 64)
    selected_slots = tuple(source.payload.selected_slots)
    selected_count = len(selected_slots)
    if selected_count == 0:
        lane_count = 1
    elif selected_count <= 6:
        lane_count = 1 << selected_count
    else:
        lane_count = 64
    valid = (1 << lane_count) - 1 if lane_count < 64 else (1 << 64) - 1
    selected_inputs = []
    for position in range(selected_count):
        word = 0
        for lane in range(lane_count):
            if (lane >> (position % 6)) & 1:
                word |= 1 << lane
        selected_inputs.append(word)
    slot_words = [0] * slot_count
    for slot, word in zip(selected_slots, selected_inputs):
        slot_words[slot] = word
    result = _evaluate_masked_threshold_packed(
        slot_count,
        source.payload.minimum_true,
        selection_words,
        slot_words,
        valid,
    )

    selection = FixedBitBlock(slot_count, source.port_semantic)
    for slot in selected_slots:
        selection.set(slot, True)
    oracle = MaskedThresholdKernel(selection, source.payload.minimum_true)
    for lane in range(lane_count):
        inputs = FixedBitBlock(slot_count, source.port_semantic)
        for slot, word in zip(selected_slots, selected_inputs):
            inputs.set(slot, bool((word >> lane) & 1))
        expected = oracle.evaluate(inputs)
        if (
            ((result.value_mask >> lane) & 1) != int(expected.value)
            or result.matched_counts[lane] != expected.matched_count
            or any(
                ((result.matched_slot_words[slot] >> lane) & 1)
                != ((expected.matched_words[slot // 64] >> (slot % 64)) & 1)
                or ((result.missing_slot_words[slot] >> lane) & 1)
                != ((expected.missing_words[slot // 64] >> (slot % 64)) & 1)
                for slot in selected_slots
            )
        ):
            raise ModelArtifactError(
                "masked-threshold lowering differs from the PA kernel oracle"
            )
    matched_selected = tuple(
        result.matched_slot_words[slot] for slot in selected_slots
    )
    missing_selected = tuple(
        result.missing_slot_words[slot] for slot in selected_slots
    )
    case = MaskedThresholdConformanceCase(
        valid,
        result.value_mask,
        tuple(selected_inputs),
        result.matched_counts,
        matched_selected,
        missing_selected,
    )

    manifest: dict[str, Any] = {
        "artifact_kind": MASKED_THRESHOLD_PAYLOAD_KIND,
        "artifact_schema": MODEL_ARTIFACT_SCHEMA,
        "container_digest": "sha256-trailer-v1",
        "description": description,
        "model": {
            "minimum_true": source.payload.minimum_true,
            "selected_count": selected_count,
            "slot_count": slot_count,
            "source_artifact_id": source.artifact_id,
        },
        "ports": {
            "inputs": [
                {
                    "dtype": "uint64",
                    "layout": "slot_major_packed64",
                    "name": "slots",
                    "shape": [slot_count],
                },
                {"dtype": "uint64", "name": "valid_mask", "shape": []},
            ],
            "outputs": [
                {"dtype": "uint64", "name": "values", "shape": []},
                {"dtype": "uint32", "name": "matched_counts", "shape": [64]},
                {
                    "dtype": "uint64",
                    "layout": "slot_major_packed64",
                    "name": "matched_slots",
                    "shape": [slot_count],
                },
                {
                    "dtype": "uint64",
                    "layout": "slot_major_packed64",
                    "name": "missing_slots",
                    "shape": [slot_count],
                },
            ],
        },
        "producer": {"name": "prolog-tsetlin-machine", "version": __version__},
        "research": {
            "authors": [str(value) for value in authors],
            "citations": [str(value) for value in citations],
            "intended_use": intended_use,
            "license": license,
            "limitations": limitations,
        },
        "slots": {
            "bindings": [binding.to_dict() for binding in source.slot_bindings],
            "mapping_version": source.mapping_version,
            "materialization": "precomputed",
            "port_semantic": source.port_semantic.value,
        },
        "task": {"kind": "boolean_threshold", "labels": list(labels)},
        "title": name,
        "validation": {
            "conformance_case_count": 1,
            "conformance_example_count": lane_count,
            "signature": source.validation_signature.to_dict(),
        },
        "restoration_reference": source.restoration_handle.to_dict(),
    }

    payload = bytearray(
        _MASKED_THRESHOLD_HEADER.pack(
            MASKED_THRESHOLD_PAYLOAD_VERSION,
            slot_count,
            selection_word_count,
            source.payload.minimum_true,
            selected_count,
            1,
            0,
            0,
        )
    )
    payload.extend(struct.pack(f"<{selection_word_count}Q", *selection_words))
    payload.extend(struct.pack("<QQ", case.valid_example_mask, case.value_mask))
    payload.extend(struct.pack(f"<{selected_count}Q", *case.selected_input_words))
    payload.extend(struct.pack("<64I", *case.matched_counts))
    payload.extend(struct.pack(f"<{selected_count}Q", *case.matched_selected_words))
    payload.extend(struct.pack(f"<{selected_count}Q", *case.missing_selected_words))
    serialized = _encode_model_container(
        _MODEL_KIND_MASKED_THRESHOLD, manifest, payload
    )
    artifact = MaskedThresholdInferenceArtifact.from_bytes(serialized)
    if path is not None:
        artifact.write(path)
    return artifact


def export_graph_tm(
    gtm: Any,
    conformance_graphs: Sequence[Any],
    *,
    name: str,
    path: str | Path | None = None,
    description: str = "",
    authors: Sequence[str] = (),
    license: str = "unspecified",
    citations: Sequence[str] = (),
    intended_use: str = "research",
    limitations: str = "research prototype",
    output_labels: Sequence[str] = ("false", "true"),
) -> GraphTMInferenceArtifact:
    """Package a GraphTM for static inference (graph_tm_v1)."""
    if GraphTsetlinMachine is None or GraphInput is None or DeepClause is None:
        raise ModelArtifactError("graph extra not available")
    if not isinstance(gtm, GraphTsetlinMachine):
        raise ValueError("gtm must be a GraphTsetlinMachine")
    # GraphTsetlinMachine has no explicit trained flag; consider fit() having been called if any clause mutated or weights differ from init
    if not name:
        raise ValueError("model artifact name cannot be empty")
    labels = tuple(str(v) for v in output_labels)
    if len(labels) != 2 or any(not v for v in labels) or labels[0] == labels[1]:
        raise ValueError("graph-TM artifacts require two distinct nonempty labels")
    depth = int(getattr(gtm, "depth"))
    clauses = int(getattr(gtm, "clauses"))
    hv_dim = int(getattr(gtm, "hv_dim"))
    if not 1 <= depth <= MAX_GRAPH_DEPTH or not 1 <= clauses <= MAX_GRAPH_CLAUSES or hv_dim not in (256, 512, 1024, 2048, 4096, 8192):
        raise ValueError("graph-TM configuration is out of bounds")
    # Build components list from internal _components/_weights
    raw_components = []
    weights: list[tuple[int, int]] = []
    internal_components = getattr(gtm, "_components", None)
    internal_weights = getattr(gtm, "_weights", None)
    if internal_components is None or internal_weights is None:
        raise ValueError("GraphTM is missing internal components/weights")
    for idx, clause in enumerate(internal_components):
        if not isinstance(clause, DeepClause):
            raise ValueError(f"clause {idx} is not a DeepClause")
        raw_components.append(clause.to_dict())
        w = internal_weights[idx] if idx < len(internal_weights) else (1, 1)
        weights.append((int(w[0]), int(w[1])))
    # Validate conformance graphs via oracle
    if len(conformance_graphs) == 0 or len(conformance_graphs) > 256:
        raise ValueError("conformance_graphs must have 1..256 entries")
    expected_labels: list[int] = []
    graph_bytes_list: list[bytes] = []
    for graph in conformance_graphs:
        if not isinstance(graph, GraphInput):
            raise ValueError("conformance_graphs must be GraphInput instances")
        pred = int(gtm.predict(graph))
        if pred not in (0, 1):
            raise ValueError("GraphTM prediction must be 0 or 1")
        expected_labels.append(pred)
        graph_bytes_list.append(json.dumps(graph.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    # Build manifest
    manifest: dict[str, Any] = {
        "artifact_kind": GRAPH_TM_PAYLOAD_KIND,
        "artifact_schema": MODEL_ARTIFACT_SCHEMA,
        "container_digest": "sha256-trailer-v1",
        "graph": {
            "clauses": clauses,
            "components": raw_components,
            "depth": depth,
            "edge_type_count": 16,
            "hv_dim": hv_dim,
            "weights": [list(w) for w in weights],
        },
        "model": {
            "clauses": clauses,
            "depth": depth,
            "hv_dim": hv_dim,
            "payload_kind": GRAPH_TM_PAYLOAD_KIND,
            "payload_version": GRAPH_TM_PAYLOAD_VERSION,
        },
        "ports": {
            "inputs": [{"dtype": "json", "name": "graph", "shape": []}],
            "outputs": [{"dtype": "uint32", "name": "prediction", "shape": []}],
        },
        "producer": {"name": "prolog-tsetlin-machine", "version": __version__},
        "research": {
            "authors": [str(v) for v in authors],
            "citations": [str(v) for v in citations],
            "intended_use": intended_use,
            "license": license,
            "limitations": limitations,
        },
        "task": {"kind": "graph_classification", "labels": list(labels)},
        "title": name,
        "description": description,
        "validation": {
            "conformance_case_count": len(conformance_graphs),
            "conformance_example_count": len(conformance_graphs),
            "signature": {"graph_conformance": True},
        },
    }
    # Build payload
    payload = bytearray(
        _GRAPH_TM_HEADER.pack(
            GRAPH_TM_PAYLOAD_VERSION,
            depth,
            clauses,
            hv_dim,
            16,
            len(conformance_graphs),
            0,
            0,
        )
    )
    for w0, w1 in weights:
        payload.extend(struct.pack("<ii", int(w0), int(w1)))
    for graph_bytes, expected in zip(graph_bytes_list, expected_labels):
        payload.extend(struct.pack("<II", len(graph_bytes), int(expected)))
        payload.extend(graph_bytes)
    serialized = _encode_model_container(_MODEL_KIND_GRAPH_TM, manifest, payload)
    artifact = GraphTMInferenceArtifact.from_bytes(serialized)
    if path is not None:
        artifact.write(path)
    return artifact


def _validate_packed_tm_manifest(
    manifest: Mapping[str, Any],
    clauses: int,
    features: int,
    threshold: int,
    cases: tuple[PackedTMConformanceCase, ...],
) -> None:
    if manifest.get("container_digest") != "sha256-trailer-v1":
        raise ModelArtifactError("unsupported model artifact digest contract")

    model = manifest.get("model")
    if not isinstance(model, dict) or (
        model.get("number_of_clauses") != clauses
        or model.get("number_of_features") != features
        or model.get("threshold") != threshold
        or model.get("clause_polarity") != "even_positive_odd_negative"
        or model.get("score_clamp") != "symmetric_threshold"
    ):
        raise ModelArtifactError("manifest and packed-TM semantics disagree")

    feature_contract = manifest.get("features")
    if not isinstance(feature_contract, dict):
        raise ModelArtifactError("model artifact lacks its feature contract")
    names = feature_contract.get("names")
    literal_ids = feature_contract.get("literal_ids")
    materialization = feature_contract.get("materialization")
    if (
        feature_contract.get("kind") != "packed_boolean_features"
        or materialization
        not in ("precomputed", "precomputed_or_raw_record_v1")
        or not isinstance(feature_contract.get("catalog_version"), str)
        or not feature_contract["catalog_version"]
        or not isinstance(names, list)
        or len(names) != features
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or not isinstance(literal_ids, list)
        or len(literal_ids) not in (0, features)
        or any(
            not isinstance(value, str)
            or not value.isascii()
            or not value.isdigit()
            or str(int(value)) != value
            or int(value) >= 1 << 64
            for value in literal_ids
        )
        or len(set(literal_ids)) != len(literal_ids)
    ):
        raise ModelArtifactError("feature contract is invalid")
    raw_preprocessing = manifest.get("preprocessing")
    if materialization == "precomputed_or_raw_record_v1":
        if not isinstance(raw_preprocessing, dict):
            raise ModelArtifactError("raw-record artifact lacks preprocessing")
        try:
            preprocessing = PreprocessingContract.from_dict(raw_preprocessing)
        except (KeyError, TypeError, ValueError) as error:
            raise ModelArtifactError("preprocessing contract is invalid") from error
        if len(preprocessing.outputs) != features or tuple(
            str(value) for value in preprocessing.literal_ids
        ) != tuple(literal_ids):
            raise ModelArtifactError(
                "preprocessing outputs disagree with the feature contract"
            )
    elif raw_preprocessing is not None:
        raise ModelArtifactError(
            "precomputed-only artifact unexpectedly contains preprocessing"
        )

    expected_ports = {
        "inputs": [
            {
                "dtype": "uint64",
                "layout": "feature_major_packed64",
                "name": "features",
                "shape": [features],
            },
            {"dtype": "uint64", "name": "valid_mask", "shape": []},
        ],
        "outputs": [
            {"dtype": "uint64", "name": "predictions", "shape": []},
            {"dtype": "int32", "name": "scores", "shape": [64]},
        ],
    }
    if manifest.get("ports") != expected_ports:
        raise ModelArtifactError("manifest port contract disagrees with the payload")

    task = manifest.get("task")
    labels = task.get("labels") if isinstance(task, dict) else None
    if (
        not isinstance(task, dict)
        or task.get("kind") != "binary_classification"
        or not isinstance(labels, list)
        or len(labels) != 2
        or any(not isinstance(value, str) or not value for value in labels)
        or labels[0] == labels[1]
    ):
        raise ModelArtifactError("binary task contract is invalid")

    validation = manifest.get("validation")
    example_count = sum(case.valid_example_mask.bit_count() for case in cases)
    if (
        not isinstance(validation, dict)
        or validation.get("conformance_case_count") != len(cases)
        or validation.get("conformance_example_count") != example_count
        or not isinstance(validation.get("signature"), dict)
    ):
        raise ModelArtifactError("manifest validation contract is invalid")

    producer = manifest.get("producer")
    research = manifest.get("research")
    if (
        not isinstance(manifest.get("title"), str)
        or not manifest["title"]
        or not isinstance(manifest.get("description"), str)
        or not isinstance(producer, dict)
        or any(not isinstance(producer.get(key), str) or not producer[key]
               for key in ("name", "version"))
        or not isinstance(research, dict)
        or any(not isinstance(research.get(key), str)
               for key in ("intended_use", "license", "limitations"))
        or any(
            not isinstance(research.get(key), list)
            or any(not isinstance(value, str) for value in research[key])
            for key in ("authors", "citations")
        )
    ):
        raise ModelArtifactError("model artifact metadata contract is invalid")
    if "restoration_reference" in manifest and not isinstance(
        manifest["restoration_reference"], dict
    ):
        raise ModelArtifactError("restoration reference must be an object")


def _validate_logic_program_manifest(
    manifest: Mapping[str, Any],
    program: LogicProgram32,
    cases: tuple[LogicProgramConformanceCase, ...],
) -> tuple[str, ...]:
    if manifest.get("container_digest") != "sha256-trailer-v1":
        raise ModelArtifactError("unsupported model artifact digest contract")
    expected_opcodes = {
        opcode.name.lower(): int(opcode) for opcode in FixedLogicOpcode
    }
    model = manifest.get("model")
    if not isinstance(model, dict) or (
        model.get("binding_count") != len(LOGIC_AST_VARIABLES)
        or model.get("instruction_count") != len(program.instructions)
        or model.get("root_instruction") != program.root_instruction
        or model.get("program_id") != program.program_id
        or model.get("opcodes") != expected_opcodes
    ):
        raise ModelArtifactError("manifest and Logic-program semantics disagree")

    binding_contract = manifest.get("bindings")
    if not isinstance(binding_contract, dict):
        raise ModelArtifactError("model artifact lacks its binding contract")
    names = binding_contract.get("names")
    literal_ids = binding_contract.get("literal_ids")
    if (
        binding_contract.get("kind") != "boolean_bindings"
        or binding_contract.get("materialization") != "precomputed"
        or not isinstance(binding_contract.get("catalog_version"), str)
        or not binding_contract["catalog_version"]
        or not isinstance(names, list)
        or len(names) != len(LOGIC_AST_VARIABLES)
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or not isinstance(literal_ids, list)
        or len(literal_ids) not in (0, len(LOGIC_AST_VARIABLES))
        or any(
            not isinstance(value, str)
            or not value.isascii()
            or not value.isdigit()
            or str(int(value)) != value
            or int(value) >= 1 << 64
            for value in literal_ids
        )
        or len(set(literal_ids)) != len(literal_ids)
    ):
        raise ModelArtifactError("binding contract is invalid")

    expected_ports = {
        "inputs": [
            {
                "dtype": "uint64",
                "layout": "binding_major_packed64",
                "name": "bindings",
                "shape": [len(LOGIC_AST_VARIABLES)],
            },
            {"dtype": "uint64", "name": "valid_mask", "shape": []},
        ],
        "outputs": [
            {"dtype": "uint64", "name": "values", "shape": []},
            {
                "dtype": "uint32",
                "name": "true_instruction_masks",
                "shape": [64],
            },
            {
                "dtype": "uint32",
                "name": "evaluated_instruction_masks",
                "shape": [64],
            },
        ],
    }
    if manifest.get("ports") != expected_ports:
        raise ModelArtifactError("manifest port contract disagrees with the payload")

    task = manifest.get("task")
    labels = task.get("labels") if isinstance(task, dict) else None
    if (
        not isinstance(task, dict)
        or task.get("kind") != "boolean_function"
        or not isinstance(labels, list)
        or len(labels) != 2
        or any(not isinstance(value, str) or not value for value in labels)
        or labels[0] == labels[1]
    ):
        raise ModelArtifactError("Logic task contract is invalid")

    validation = manifest.get("validation")
    example_count = sum(case.valid_example_mask.bit_count() for case in cases)
    if (
        not isinstance(validation, dict)
        or validation.get("conformance_case_count") != len(cases)
        or validation.get("conformance_example_count") != example_count
        or not isinstance(validation.get("signature"), dict)
    ):
        raise ModelArtifactError("manifest validation contract is invalid")

    producer = manifest.get("producer")
    research = manifest.get("research")
    if (
        not isinstance(manifest.get("title"), str)
        or not manifest["title"]
        or not isinstance(manifest.get("description"), str)
        or not isinstance(producer, dict)
        or any(
            not isinstance(producer.get(key), str) or not producer[key]
            for key in ("name", "version")
        )
        or not isinstance(research, dict)
        or any(
            not isinstance(research.get(key), str)
            for key in ("intended_use", "license", "limitations")
        )
        or any(
            not isinstance(research.get(key), list)
            or any(not isinstance(value, str) for value in research[key])
            for key in ("authors", "citations")
        )
    ):
        raise ModelArtifactError("model artifact metadata contract is invalid")
    if "restoration_reference" in manifest and not isinstance(
        manifest["restoration_reference"], dict
    ):
        raise ModelArtifactError("restoration reference must be an object")
    return tuple(names)


def _validate_masked_threshold_manifest(
    manifest: Mapping[str, Any],
    slot_count: int,
    minimum_true: int,
    selected_count: int,
    selection_words: tuple[int, ...],
    cases: tuple[MaskedThresholdConformanceCase, ...],
) -> None:
    if manifest.get("container_digest") != "sha256-trailer-v1":
        raise ModelArtifactError("unsupported model artifact digest contract")
    model = manifest.get("model")
    source_id = model.get("source_artifact_id") if isinstance(model, dict) else None
    if not isinstance(model, dict) or (
        model.get("slot_count") != slot_count
        or model.get("minimum_true") != minimum_true
        or model.get("selected_count") != selected_count
        or not isinstance(source_id, str)
        or not source_id.startswith("sha256:")
        or len(source_id) != 71
        or any(value not in "0123456789abcdef" for value in source_id[7:])
    ):
        raise ModelArtifactError("manifest and masked-threshold semantics disagree")

    expected_ports = {
        "inputs": [
            {
                "dtype": "uint64",
                "layout": "slot_major_packed64",
                "name": "slots",
                "shape": [slot_count],
            },
            {"dtype": "uint64", "name": "valid_mask", "shape": []},
        ],
        "outputs": [
            {"dtype": "uint64", "name": "values", "shape": []},
            {"dtype": "uint32", "name": "matched_counts", "shape": [64]},
            {
                "dtype": "uint64",
                "layout": "slot_major_packed64",
                "name": "matched_slots",
                "shape": [slot_count],
            },
            {
                "dtype": "uint64",
                "layout": "slot_major_packed64",
                "name": "missing_slots",
                "shape": [slot_count],
            },
        ],
    }
    if manifest.get("ports") != expected_ports:
        raise ModelArtifactError("manifest port contract disagrees with the payload")

    slots = manifest.get("slots")
    bindings = slots.get("bindings") if isinstance(slots, dict) else None
    if (
        not isinstance(slots, dict)
        or slots.get("materialization") != "precomputed"
        or not isinstance(slots.get("mapping_version"), str)
        or not slots["mapping_version"]
        or slots.get("port_semantic")
        not in {"literal_truth", "ta_action", "literal_condition", "clause_output"}
        or not isinstance(bindings, list)
    ):
        raise ModelArtifactError("masked-threshold slot contract is invalid")
    bound_slots: set[int] = set()
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ModelArtifactError("masked-threshold slot binding is invalid")
        slot = binding.get("slot")
        literal_ids = binding.get("provenance_literal_ids")
        if (
            not isinstance(slot, int)
            or not 0 <= slot < slot_count
            or slot in bound_slots
            or binding.get("source_kind")
            not in {"literal", "ta", "literal_condition", "clause", "artifact_output"}
            or not isinstance(binding.get("source_id"), str)
            or not binding["source_id"]
            or not isinstance(literal_ids, list)
            or any(
                not isinstance(value, str)
                or not value.isascii()
                or not value.isdigit()
                or str(int(value)) != value
                or int(value) >= 1 << 64
                for value in literal_ids
            )
        ):
            raise ModelArtifactError("masked-threshold slot binding is invalid")
        bound_slots.add(slot)
    selected_slots = {
        word_index * 64 + bit
        for word_index, word in enumerate(selection_words)
        for bit in range(64)
        if word & (1 << bit)
    }
    if not selected_slots.issubset(bound_slots):
        raise ModelArtifactError("selected PA slot lacks its source binding")

    task = manifest.get("task")
    labels = task.get("labels") if isinstance(task, dict) else None
    if (
        not isinstance(task, dict)
        or task.get("kind") != "boolean_threshold"
        or not isinstance(labels, list)
        or len(labels) != 2
        or any(not isinstance(value, str) or not value for value in labels)
        or labels[0] == labels[1]
    ):
        raise ModelArtifactError("masked-threshold task contract is invalid")
    validation = manifest.get("validation")
    example_count = sum(case.valid_example_mask.bit_count() for case in cases)
    if (
        not isinstance(validation, dict)
        or validation.get("conformance_case_count") != len(cases)
        or validation.get("conformance_example_count") != example_count
        or not isinstance(validation.get("signature"), dict)
    ):
        raise ModelArtifactError("manifest validation contract is invalid")
    producer = manifest.get("producer")
    research = manifest.get("research")
    restoration = manifest.get("restoration_reference")
    if (
        not isinstance(manifest.get("title"), str)
        or not manifest["title"]
        or not isinstance(manifest.get("description"), str)
        or not isinstance(producer, dict)
        or any(
            not isinstance(producer.get(key), str) or not producer[key]
            for key in ("name", "version")
        )
        or not isinstance(research, dict)
        or any(
            not isinstance(research.get(key), str)
            for key in ("intended_use", "license", "limitations")
        )
        or any(
            not isinstance(research.get(key), list)
            or any(not isinstance(value, str) for value in research[key])
            for key in ("authors", "citations")
        )
        or not isinstance(restoration, dict)
    ):
        raise ModelArtifactError("model artifact metadata contract is invalid")


def _validate_snapshot(snapshot: TMSnapshot) -> None:
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported TM snapshot schema version")
    if (
        not 0 < snapshot.number_of_clauses <= _MAX_DIMENSION
        or not 0 < snapshot.number_of_features <= _MAX_DIMENSION
        or not 0 < snapshot.states_per_action <= 0x7FFF
        or not 0 < snapshot.threshold <= 0x7FFFFFFF
    ):
        raise ValueError("TM snapshot configuration is outside artifact bounds")
    expected_width = snapshot.number_of_features * 2
    if len(snapshot.states) != snapshot.number_of_clauses or any(
        len(row) != expected_width for row in snapshot.states
    ):
        raise ValueError("TM snapshot state matrix has the wrong shape")
    maximum_state = snapshot.states_per_action * 2
    if any(
        not 1 <= state <= maximum_state
        for row in snapshot.states
        for state in row
    ):
        raise ValueError("TM snapshot contains an invalid automaton state")


def _default_conformance_rows(feature_count: int) -> tuple[tuple[bool, ...], ...]:
    rows = [
        tuple(False for _ in range(feature_count)),
        tuple(True for _ in range(feature_count)),
        tuple(index % 2 == 0 for index in range(feature_count)),
    ]
    for active in range(min(feature_count, 16)):
        rows.append(tuple(index == active for index in range(feature_count)))
    return tuple(rows)


def _evaluate_snapshot_row(
    snapshot: TMSnapshot, row: Sequence[bool | int]
) -> tuple[int, int]:
    if len(row) != snapshot.number_of_features:
        raise ValueError("conformance row has the wrong feature width")
    score = 0
    for clause, states in enumerate(snapshot.states):
        output = True
        included = False
        for feature, value in enumerate(row):
            truth = bool(value)
            if states[feature * 2] > snapshot.states_per_action:
                included = True
                output = output and truth
            if states[feature * 2 + 1] > snapshot.states_per_action:
                included = True
                output = output and not truth
        if included and output:
            score += 1 if clause % 2 == 0 else -1
    score = max(-snapshot.threshold, min(snapshot.threshold, score))
    return score, int(score > 0)


def _pack_rows(
    rows: Sequence[Sequence[bool | int]], feature_count: int
) -> tuple[tuple[int, ...], int]:
    if len(rows) > 64:
        raise ValueError("a packed artifact page cannot exceed 64 rows")
    words = [0] * feature_count
    for lane, row in enumerate(rows):
        if len(row) != feature_count:
            raise ValueError("conformance row has the wrong feature width")
        for feature, value in enumerate(row):
            if bool(value):
                words[feature] |= 1 << lane
    valid = (1 << len(rows)) - 1 if len(rows) < 64 else (1 << 64) - 1
    return tuple(words), valid


def _evaluate_masks(
    clause_count: int,
    feature_count: int,
    threshold: int,
    positive_masks: Sequence[int],
    negative_masks: Sequence[int],
    feature_words: Sequence[int],
    valid_example_mask: int,
) -> PackedTMArtifactResult64:
    if len(feature_words) != feature_count:
        raise ValueError("packed feature plane has the wrong width")
    if not 0 <= valid_example_mask < 1 << 64:
        raise ValueError("valid_example_mask is outside uint64 range")
    if any(not 0 <= word < 1 << 64 for word in feature_words):
        raise ValueError("packed feature word is outside uint64 range")
    feature_word_count = (feature_count + 63) // 64
    if len(positive_masks) != clause_count * feature_word_count or len(
        negative_masks
    ) != clause_count * feature_word_count:
        raise ValueError("packed Include masks have the wrong shape")

    scores = [0] * 64
    for clause in range(clause_count):
        output = valid_example_mask
        empty = True
        base = clause * feature_word_count
        for mask_word in range(feature_word_count):
            positive = positive_masks[base + mask_word]
            negative = negative_masks[base + mask_word]
            included = positive | negative
            empty = empty and included == 0
            while included:
                offset = (included & -included).bit_length() - 1
                feature = mask_word * 64 + offset
                if feature < feature_count:
                    truth = feature_words[feature]
                    bit = 1 << offset
                    if positive & bit:
                        output &= truth
                    if negative & bit:
                        output &= ~truth
                included &= included - 1
        output &= valid_example_mask
        if empty:
            output = 0
        contribution = 1 if clause % 2 == 0 else -1
        lanes = output
        while lanes:
            lane = (lanes & -lanes).bit_length() - 1
            scores[lane] += contribution
            lanes &= lanes - 1

    prediction = 0
    for lane in range(64):
        bit = 1 << lane
        if not valid_example_mask & bit:
            scores[lane] = 0
            continue
        scores[lane] = max(-threshold, min(threshold, scores[lane]))
        if scores[lane] > 0:
            prediction |= bit
    return PackedTMArtifactResult64(
        valid_example_mask,
        prediction,
        tuple(scores),
    )


def _evaluate_logic_program_packed(
    program: LogicProgram32,
    binding_words: Sequence[int],
    valid_example_mask: int,
) -> LogicProgramArtifactResult64:
    if len(binding_words) != len(LOGIC_AST_VARIABLES):
        raise ValueError("packed Logic bindings have the wrong width")
    if not 0 <= valid_example_mask < 1 << 64:
        raise ValueError("valid_example_mask is outside uint64 range")
    if any(not 0 <= word < 1 << 64 for word in binding_words):
        raise ValueError("packed Logic binding word is outside uint64 range")

    instruction_words: list[int] = []
    for instruction in program.instructions:
        operand_indices = tuple(
            index
            for index in range(len(instruction_words))
            if instruction.operand_mask & (1 << index)
        )
        if instruction.opcode is FixedLogicOpcode.CONSTANT:
            value = valid_example_mask if instruction.argument else 0
        elif instruction.opcode is FixedLogicOpcode.INPUT:
            value = binding_words[instruction.argument] & valid_example_mask
        elif instruction.opcode is FixedLogicOpcode.NOT:
            value = ~instruction_words[operand_indices[0]] & valid_example_mask
        elif instruction.opcode is FixedLogicOpcode.AND:
            value = valid_example_mask
            for operand in operand_indices:
                value &= instruction_words[operand]
        elif instruction.opcode is FixedLogicOpcode.OR:
            value = 0
            for operand in operand_indices:
                value |= instruction_words[operand]
            value &= valid_example_mask
        else:
            value = 0
            for operand in operand_indices:
                value ^= instruction_words[operand]
            value &= valid_example_mask
        instruction_words.append(value)

    true_masks = [0] * 64
    evaluated_masks = [0] * 64
    evaluated = (1 << len(program.instructions)) - 1
    for lane in range(64):
        bit = 1 << lane
        if not valid_example_mask & bit:
            continue
        true_masks[lane] = sum(
            1 << index
            for index, word in enumerate(instruction_words)
            if word & bit
        )
        evaluated_masks[lane] = evaluated
    return LogicProgramArtifactResult64(
        valid_example_mask,
        instruction_words[program.root_instruction] & valid_example_mask,
        tuple(true_masks),
        tuple(evaluated_masks),
    )


def _evaluate_masked_threshold_packed(
    slot_count: int,
    minimum_true: int,
    selection_words: Sequence[int],
    slot_words: Sequence[int],
    valid_example_mask: int,
) -> MaskedThresholdArtifactResult64:
    if slot_count not in (1024, 4096):
        raise ValueError("masked-threshold slot count must be 1024 or 4096")
    if len(selection_words) != slot_count // 64:
        raise ValueError("masked-threshold selection has the wrong width")
    if len(slot_words) != slot_count:
        raise ValueError("packed PA slot plane has the wrong width")
    if not 0 <= valid_example_mask < 1 << 64:
        raise ValueError("valid_example_mask is outside uint64 range")
    if any(not 0 <= word < 1 << 64 for word in slot_words):
        raise ValueError("packed PA slot word is outside uint64 range")
    selected_count = sum(word.bit_count() for word in selection_words)
    if not 0 <= minimum_true <= selected_count:
        raise ValueError("masked-threshold minimum is invalid")

    matched_counts = [0] * 64
    matched_words = [0] * slot_count
    missing_words = [0] * slot_count
    for selection_index, selection_word in enumerate(selection_words):
        selected = selection_word
        while selected:
            bit = (selected & -selected).bit_length() - 1
            slot = selection_index * 64 + bit
            matched = slot_words[slot] & valid_example_mask
            missing = (~slot_words[slot]) & valid_example_mask & ((1 << 64) - 1)
            matched_words[slot] = matched
            missing_words[slot] = missing
            lanes = matched
            while lanes:
                lane = (lanes & -lanes).bit_length() - 1
                matched_counts[lane] += 1
                lanes &= lanes - 1
            selected &= selected - 1
    value_mask = 0
    for lane in range(64):
        bit = 1 << lane
        if not valid_example_mask & bit:
            matched_counts[lane] = 0
        elif matched_counts[lane] >= minimum_true:
            value_mask |= bit
    return MaskedThresholdArtifactResult64(
        valid_example_mask,
        value_mask,
        tuple(matched_counts),
        tuple(matched_words),
        tuple(missing_words),
    )


def _validate_graph_tm_manifest(
    manifest: Mapping[str, Any],
    depth: int,
    clauses: int,
    hv_dim: int,
    edge_type_count: int,
    weights: tuple[tuple[int, int], ...],
    conformance_graphs: tuple[Any, ...],
    expected_labels: tuple[int, ...],
) -> None:
    if manifest.get("container_digest") != "sha256-trailer-v1":
        raise ModelArtifactError("unsupported model artifact digest contract")
    model = manifest.get("model")
    if not isinstance(model, dict) or (
        model.get("depth") != depth
        or model.get("clauses") != clauses
        or model.get("hv_dim") != hv_dim
        or model.get("payload_kind") != GRAPH_TM_PAYLOAD_KIND
        or model.get("payload_version") != GRAPH_TM_PAYLOAD_VERSION
    ):
        raise ModelArtifactError("manifest and graph-TM semantics disagree")
    graph = manifest.get("graph")
    if not isinstance(graph, dict):
        raise ModelArtifactError("model artifact lacks its graph contract")
    if (
        graph.get("depth") != depth
        or graph.get("clauses") != clauses
        or graph.get("hv_dim") != hv_dim
        or graph.get("edge_type_count") != edge_type_count
        or graph.get("edge_type_count") != 16
        or not isinstance(graph.get("components"), list)
        or len(graph["components"]) != clauses
        or not isinstance(graph.get("weights"), list)
        or len(graph["weights"]) != clauses
    ):
        raise ModelArtifactError("graph contract is invalid")
    manifest_weights = graph["weights"]
    # binary weights are authoritative — manifest must exactly equal them
    if manifest_weights != [list(w) for w in weights]:
        raise ModelArtifactError("graph manifest weights disagree with payload")
    for w in manifest_weights:
        if not isinstance(w, list) or len(w) != 2 or any(not isinstance(v, int) for v in w):
            raise ModelArtifactError("graph weight contract is invalid")
        if not -1_000_000 <= w[0] <= 1_000_000 or not -1_000_000 <= w[1] <= 1_000_000:
            raise ModelArtifactError("graph weight value out of range")
    for raw in graph["components"]:
        if not isinstance(raw, dict):
            raise ModelArtifactError("graph component contract is invalid")
        layers = raw.get("components")
        # DeepClause serialises as {"components": [...]}  (not "layers")
        if not isinstance(layers, list) or len(layers) != depth:
            raise ModelArtifactError("graph component depth is invalid")
        # DeepClause.from_dict will do deeper validation on load; here just check shape
    expected_ports = {
        "inputs": [{"dtype": "json", "name": "graph", "shape": []}],
        "outputs": [{"dtype": "uint32", "name": "prediction", "shape": []}],
    }
    if manifest.get("ports") != expected_ports:
        raise ModelArtifactError("manifest port contract disagrees with the payload")
    task = manifest.get("task")
    labels = task.get("labels") if isinstance(task, dict) else None
    if (
        not isinstance(task, dict)
        or task.get("kind") != "graph_classification"
        or not isinstance(labels, list)
        or len(labels) != 2
        or any(not isinstance(v, str) or not v for v in labels)
        or labels[0] == labels[1]
    ):
        raise ModelArtifactError("graph task contract is invalid")
    validation = manifest.get("validation")
    if (
        not isinstance(validation, dict)
        or validation.get("conformance_case_count") != len(conformance_graphs)
        or validation.get("conformance_example_count") != len(conformance_graphs)
        or not isinstance(validation.get("signature"), dict)
    ):
        raise ModelArtifactError("manifest validation contract is invalid")
    producer = manifest.get("producer")
    research = manifest.get("research")
    if (
        not isinstance(manifest.get("title"), str)
        or not manifest["title"]
        or not isinstance(producer, dict)
        or any(not isinstance(producer.get(key), str) or not producer[key] for key in ("name", "version"))
        or not isinstance(research, dict)
        or any(not isinstance(research.get(key), str) for key in ("intended_use", "license", "limitations"))
        or any(not isinstance(research.get(key), list) or any(not isinstance(v, str) for v in research[key]) for key in ("authors", "citations"))
    ):
        raise ModelArtifactError("model artifact metadata contract is invalid")


def _validate_mask_tails(
    masks: Sequence[int], clauses: int, features: int, feature_word_count: int
) -> None:
    tail = features % 64
    if tail == 0:
        return
    invalid = ~((1 << tail) - 1) & ((1 << 64) - 1)
    for clause in range(clauses):
        if masks[clause * feature_word_count + feature_word_count - 1] & invalid:
            raise ModelArtifactError("packed Include mask uses a tail feature bit")
