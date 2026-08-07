"""Dependency-free ctypes access to the versioned PTM C ABI."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Sequence

from .artifact import PAArtifact
from .logic_ast import LOGIC_AST_VARIABLES
from .logic_consolidation import (
    FixedLogicResult,
    LogicProgram32,
)
from .pa import FixedBitBlock, PAResult, PortSemantic
from .reference import TMSnapshot
from .representation import LiteralBatch


PTM_ABI_VERSION = 2


class NativeRuntimeError(RuntimeError):
    pass


class PackedTMBackend(IntEnum):
    AUTOMATIC = 0
    SCALAR = 1
    AVX2 = 2
    AVX512 = 3


@dataclass(frozen=True, slots=True)
class NativeCpuCapabilities:
    brand: str
    x86: bool
    os_xsave: bool
    avx: bool
    avx2: bool
    avx512f: bool
    compiled_avx2: bool
    compiled_avx512: bool
    preferred_backend: PackedTMBackend

    def supports(self, backend: PackedTMBackend) -> bool:
        if backend in (PackedTMBackend.AUTOMATIC, PackedTMBackend.SCALAR):
            return True
        if backend == PackedTMBackend.AVX2:
            return self.avx2 and self.compiled_avx2
        if backend == PackedTMBackend.AVX512:
            return self.avx512f and self.compiled_avx512
        return False


class _BitBlock1024(ctypes.Structure):
    _fields_ = [("words", ctypes.c_uint64 * 16)]


class _BitBlock4096(ctypes.Structure):
    _fields_ = [("words", ctypes.c_uint64 * 64)]


class _ThresholdResult1024(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
        ("matched_count", ctypes.c_uint32),
        ("selected_count", ctypes.c_uint32),
        ("alignment_padding", ctypes.c_uint8 * 52),
        ("matched", _BitBlock1024),
        ("missing", _BitBlock1024),
    ]


class _ThresholdResult4096(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
        ("matched_count", ctypes.c_uint32),
        ("selected_count", ctypes.c_uint32),
        ("alignment_padding", ctypes.c_uint8 * 52),
        ("matched", _BitBlock4096),
        ("missing", _BitBlock4096),
    ]


class _LogicInstruction32(ctypes.Structure):
    _fields_ = [
        ("operand_mask", ctypes.c_uint32),
        ("opcode", ctypes.c_uint8),
        ("argument", ctypes.c_uint8),
        ("reserved", ctypes.c_uint16),
    ]


class _LogicProgram32(ctypes.Structure):
    _fields_ = [
        ("instruction_count", ctypes.c_uint32),
        ("root_instruction", ctypes.c_uint32),
        ("instructions", _LogicInstruction32 * 32),
        ("alignment_padding", ctypes.c_uint8 * 56),
    ]


class _LogicResult32(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
        ("true_instruction_mask", ctypes.c_uint32),
        ("evaluated_instruction_mask", ctypes.c_uint32),
        ("alignment_padding", ctypes.c_uint32),
    ]


class _TMModelConfig(ctypes.Structure):
    _fields_ = [
        ("number_of_clauses", ctypes.c_uint32),
        ("number_of_features", ctypes.c_uint32),
        ("states_per_action", ctypes.c_uint32),
        ("threshold", ctypes.c_int32),
    ]


class _CpuCapabilities(ctypes.Structure):
    _fields_ = [
        ("hardware_flags", ctypes.c_uint64),
        ("compiled_flags", ctypes.c_uint64),
        ("preferred_backend", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("brand", ctypes.c_char * 104),
    ]


class _TMBatch64Result(ctypes.Structure):
    _fields_ = [
        ("valid_example_mask", ctypes.c_uint64),
        ("prediction_mask", ctypes.c_uint64),
        ("scores", ctypes.c_int32 * 64),
        ("alignment_padding", ctypes.c_uint8 * 48),
    ]


if (
    ctypes.sizeof(_BitBlock1024) != 128
    or ctypes.sizeof(_BitBlock4096) != 512
    or _ThresholdResult1024.matched.offset != 64
    or _ThresholdResult1024.missing.offset != 192
    or ctypes.sizeof(_ThresholdResult1024) != 320
    or _ThresholdResult4096.matched.offset != 64
    or _ThresholdResult4096.missing.offset != 576
    or ctypes.sizeof(_ThresholdResult4096) != 1088
    or ctypes.sizeof(_LogicInstruction32) != 8
    or _LogicProgram32.instructions.offset != 8
    or ctypes.sizeof(_LogicProgram32) != 320
    or ctypes.sizeof(_LogicResult32) != 16
    or ctypes.sizeof(_TMModelConfig) != 16
    or ctypes.sizeof(_CpuCapabilities) != 128
    or _TMBatch64Result.scores.offset != 16
    or ctypes.sizeof(_TMBatch64Result) != 320
):
    raise RuntimeError("ctypes could not reproduce the PTM ABI v2 structure layout")


class _AlignedInstance:
    """Own an over-allocated ctypes object aligned to a 64-byte address."""

    def __init__(self, value_type: type[ctypes.Structure]) -> None:
        self._storage = ctypes.create_string_buffer(ctypes.sizeof(value_type) + 63)
        base = ctypes.addressof(self._storage)
        address = (base + 63) & ~63
        self.pointer = ctypes.cast(ctypes.c_void_p(address), ctypes.POINTER(value_type))

    @property
    def value(self) -> ctypes.Structure:
        return self.pointer.contents


class _AlignedArray:
    """Own a 64-byte-aligned contiguous array with an aligned element stride."""

    def __init__(self, value_type: type[ctypes.Structure], count: int) -> None:
        if count <= 0:
            raise ValueError("aligned native array cannot be empty")
        self._storage = ctypes.create_string_buffer(
            ctypes.sizeof(value_type) * count + 63
        )
        base = ctypes.addressof(self._storage)
        address = (base + 63) & ~63
        self.pointer = ctypes.cast(ctypes.c_void_p(address), ctypes.POINTER(value_type))


def _native_library_candidates() -> tuple[Path, ...]:
    configured = os.environ.get("PTM_NATIVE_LIBRARY")
    project_root = Path(__file__).resolve().parents[2]
    names = ("ptm.dll", "libptm.so", "libptm.dylib")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(project_root / "build" / name for name in names)
    discovered = ctypes.util.find_library("ptm")
    if discovered:
        candidates.append(Path(discovered))
    return tuple(candidates)


def find_native_library() -> Path | None:
    return next(
        (candidate.resolve() for candidate in _native_library_candidates() if candidate.is_file()),
        None,
    )


def native_cpu_capabilities(
    library_path: str | os.PathLike[str] | None = None,
) -> NativeCpuCapabilities:
    resolved = Path(library_path).resolve() if library_path else find_native_library()
    if resolved is None or not resolved.is_file():
        raise NativeRuntimeError(
            "PTM native library was not found; set PTM_NATIVE_LIBRARY"
        )
    library = ctypes.CDLL(str(resolved))
    library.ptm_abi_version.argtypes = []
    library.ptm_abi_version.restype = ctypes.c_uint32
    actual_version = int(library.ptm_abi_version())
    if actual_version != PTM_ABI_VERSION:
        raise NativeRuntimeError(
            f"PTM ABI mismatch: Python expects {PTM_ABI_VERSION}, library reports "
            f"{actual_version}"
        )
    library.ptm_cpu_capabilities_query.argtypes = [
        ctypes.POINTER(_CpuCapabilities)
    ]
    library.ptm_cpu_capabilities_query.restype = ctypes.c_int
    native = _CpuCapabilities()
    status = int(library.ptm_cpu_capabilities_query(ctypes.byref(native)))
    if status != 0:
        raise NativeRuntimeError(
            f"native CPU capability query failed with status {status}"
        )
    hardware = int(native.hardware_flags)
    compiled = int(native.compiled_flags)
    return NativeCpuCapabilities(
        brand=bytes(native.brand).split(b"\0", 1)[0].decode(
            "utf-8", errors="replace"
        ),
        x86=bool(hardware & (1 << 0)),
        os_xsave=bool(hardware & (1 << 1)),
        avx=bool(hardware & (1 << 2)),
        avx2=bool(hardware & (1 << 3)),
        avx512f=bool(hardware & (1 << 4)),
        compiled_avx2=bool(compiled & (1 << 3)),
        compiled_avx512=bool(compiled & (1 << 4)),
        preferred_backend=PackedTMBackend(native.preferred_backend),
    )


_SEMANTIC_VALUE = {
    PortSemantic.LITERAL_TRUTH: 0,
    PortSemantic.TA_ACTION: 1,
    PortSemantic.LITERAL_CONDITION: 2,
    PortSemantic.CLAUSE_OUTPUT: 3,
}


class NativePAKernel:
    def __init__(self, library_path: str | os.PathLike[str] | None = None) -> None:
        resolved = Path(library_path).resolve() if library_path else find_native_library()
        if resolved is None or not resolved.is_file():
            raise NativeRuntimeError(
                "PTM native library was not found; set PTM_NATIVE_LIBRARY"
            )
        self.library_path = resolved
        self._library = ctypes.CDLL(str(resolved))
        self._configure_signatures()
        actual_version = int(self._library.ptm_abi_version())
        if actual_version != PTM_ABI_VERSION:
            raise NativeRuntimeError(
                f"PTM ABI mismatch: Python expects {PTM_ABI_VERSION}, library reports "
                f"{actual_version}"
            )

    def _configure_signatures(self) -> None:
        library = self._library
        library.ptm_abi_version.argtypes = []
        library.ptm_abi_version.restype = ctypes.c_uint32
        library.ptm_status_message.argtypes = [ctypes.c_int]
        library.ptm_status_message.restype = ctypes.c_char_p
        library.ptm_threshold_1024_eval.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(_BitBlock1024),
            ctypes.POINTER(_BitBlock1024),
            ctypes.c_uint32,
            ctypes.POINTER(_ThresholdResult1024),
        ]
        library.ptm_threshold_1024_eval.restype = ctypes.c_int
        library.ptm_threshold_4096_eval.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(_BitBlock4096),
            ctypes.POINTER(_BitBlock4096),
            ctypes.c_uint32,
            ctypes.POINTER(_ThresholdResult4096),
        ]
        library.ptm_threshold_4096_eval.restype = ctypes.c_int

    def _raise_status(self, status: int) -> None:
        message = self._library.ptm_status_message(status)
        decoded = message.decode("utf-8", errors="replace") if message else "unknown"
        raise NativeRuntimeError(f"native PA evaluation failed ({status}): {decoded}")

    def evaluate(
        self,
        inputs: FixedBitBlock,
        selection: FixedBitBlock,
        minimum_true: int,
    ) -> PAResult:
        if inputs.bit_count != selection.bit_count:
            raise ValueError("input and selection shapes differ")
        if inputs.semantic is not selection.semantic:
            raise ValueError("input and selection semantics differ")
        if not 0 <= minimum_true <= 0xFFFFFFFF:
            raise ValueError("minimum_true is outside uint32 range")

        if inputs.bit_count == 1024:
            block_type = _BitBlock1024
            result_type = _ThresholdResult1024
            function = self._library.ptm_threshold_1024_eval
        else:
            block_type = _BitBlock4096
            result_type = _ThresholdResult4096
            function = self._library.ptm_threshold_4096_eval

        native_input = _AlignedInstance(block_type)
        native_selection = _AlignedInstance(block_type)
        native_result = _AlignedInstance(result_type)
        for index, word in enumerate(inputs.words):
            native_input.value.words[index] = word
        for index, word in enumerate(selection.words):
            native_selection.value.words[index] = word

        status = function(
            _SEMANTIC_VALUE[inputs.semantic],
            native_input.pointer,
            native_selection.pointer,
            minimum_true,
            native_result.pointer,
        )
        if status != 0:
            self._raise_status(status)
        result = native_result.value
        return PAResult(
            value=bool(result.value),
            matched_count=int(result.matched_count),
            selected_count=int(result.selected_count),
            matched_words=tuple(int(word) for word in result.matched.words),
            missing_words=tuple(int(word) for word in result.missing.words),
        )

    def evaluate_artifact(
        self, artifact: PAArtifact, inputs: FixedBitBlock
    ) -> PAResult:
        if not artifact.verify_artifact_id():
            raise ValueError("artifact content hash is invalid")
        if artifact.input_shape.bit_count != inputs.bit_count:
            raise ValueError("artifact and input shapes differ")
        if artifact.port_semantic is not inputs.semantic:
            raise ValueError("artifact and input semantics differ")
        selection = FixedBitBlock(inputs.bit_count, inputs.semantic)
        for slot in artifact.payload.selected_slots:
            selection.set(slot, True)
        return self.evaluate(inputs, selection, artifact.payload.minimum_true)


def _marshal_logic_program(target: _LogicProgram32, program: LogicProgram32) -> None:
    target.instruction_count = len(program.instructions)
    target.root_instruction = program.root_instruction
    for index, instruction in enumerate(program.instructions):
        target.instructions[index].operand_mask = instruction.operand_mask
        target.instructions[index].opcode = int(instruction.opcode)
        target.instructions[index].argument = instruction.argument
        target.instructions[index].reserved = 0


class NativeLogicProgramBatch:
    """Prepared native state whose program and binding buffers stay synchronized."""

    def __init__(
        self,
        runtime: "NativeLogicKernel",
        programs: Sequence[LogicProgram32],
        bindings: Sequence[Sequence[bool | int]],
    ) -> None:
        if len(programs) != len(bindings):
            raise ValueError("program and binding row counts differ")
        if not programs:
            raise ValueError("native Logic program batch cannot be empty")
        if len(programs) > 0xFFFFFFFF:
            raise ValueError("native Logic program batch exceeds uint32 capacity")
        self._runtime = runtime
        self.count = len(programs)
        self._programs = _AlignedArray(_LogicProgram32, self.count)
        self._bindings = (ctypes.c_uint8 * self.count)()
        self._results = (_LogicResult32 * self.count)()
        self._has_executed = False
        for index, (program, row) in enumerate(zip(programs, bindings)):
            if len(row) != len(LOGIC_AST_VARIABLES):
                raise ValueError("native Logic bindings must provide A through E")
            _marshal_logic_program(self._programs.pointer[index], program)
            self._bindings[index] = sum(
                int(bool(value)) << bit for bit, value in enumerate(row)
            )

    def execute(self) -> None:
        status = self._runtime._library.ptm_logic_program32_eval_batch(
            self._programs.pointer,
            self._bindings,
            self.count,
            self._results,
        )
        if status != 0:
            self._runtime._raise_status(status)
        self._has_executed = True

    def results(self) -> tuple[FixedLogicResult, ...]:
        if not self._has_executed:
            raise NativeRuntimeError("native Logic batch has not been executed")
        return tuple(
            FixedLogicResult(
                value=bool(result.value),
                true_instruction_mask=int(result.true_instruction_mask),
                evaluated_instruction_mask=int(result.evaluated_instruction_mask),
            )
            for result in self._results
        )

    def evaluate(self) -> tuple[FixedLogicResult, ...]:
        self.execute()
        return self.results()


class NativeLogicKernel:
    def __init__(self, library_path: str | os.PathLike[str] | None = None) -> None:
        resolved = Path(library_path).resolve() if library_path else find_native_library()
        if resolved is None or not resolved.is_file():
            raise NativeRuntimeError(
                "PTM native library was not found; set PTM_NATIVE_LIBRARY"
            )
        self.library_path = resolved
        self._library = ctypes.CDLL(str(resolved))
        self._library.ptm_abi_version.argtypes = []
        self._library.ptm_abi_version.restype = ctypes.c_uint32
        self._library.ptm_status_message.argtypes = [ctypes.c_int]
        self._library.ptm_status_message.restype = ctypes.c_char_p
        self._library.ptm_logic_program32_eval_batch.argtypes = [
            ctypes.POINTER(_LogicProgram32),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint32,
            ctypes.POINTER(_LogicResult32),
        ]
        self._library.ptm_logic_program32_eval_batch.restype = ctypes.c_int
        actual_version = int(self._library.ptm_abi_version())
        if actual_version != PTM_ABI_VERSION:
            raise NativeRuntimeError(
                f"PTM ABI mismatch: Python expects {PTM_ABI_VERSION}, library reports "
                f"{actual_version}"
            )

    def _raise_status(self, status: int) -> None:
        message = self._library.ptm_status_message(status)
        decoded = message.decode("utf-8", errors="replace") if message else "unknown"
        raise NativeRuntimeError(
            f"native fixed Logic evaluation failed ({status}): {decoded}"
        )

    def prepare(
        self,
        programs: Sequence[LogicProgram32],
        bindings: Sequence[Sequence[bool | int]],
    ) -> NativeLogicProgramBatch:
        return NativeLogicProgramBatch(self, programs, bindings)


@dataclass(frozen=True, slots=True)
class PackedTMResult64:
    valid_example_mask: int
    prediction_mask: int
    scores: tuple[int, ...]
    clause_outputs: tuple[int, ...]
    feedback_clause_outputs: tuple[int, ...]
    backend: PackedTMBackend

    def predictions(self, lane_count: int) -> tuple[int, ...]:
        if not 0 <= lane_count <= 64:
            raise ValueError("lane_count must be between zero and 64")
        expected_mask = (1 << lane_count) - 1 if lane_count < 64 else (1 << 64) - 1
        if self.valid_example_mask != expected_mask:
            raise ValueError("valid mask is not a contiguous lane prefix")
        return tuple((self.prediction_mask >> lane) & 1 for lane in range(lane_count))


class NativePackedTsetlinMachine:
    """Immutable bit-sliced TA state and 64-example packed inference image."""

    def __init__(
        self,
        snapshot: TMSnapshot,
        library_path: str | os.PathLike[str] | None = None,
    ) -> None:
        resolved = Path(library_path).resolve() if library_path else find_native_library()
        if resolved is None or not resolved.is_file():
            raise NativeRuntimeError(
                "PTM native library was not found; set PTM_NATIVE_LIBRARY"
            )
        if snapshot.schema_version != 1:
            raise ValueError("unsupported TM snapshot schema version")
        if not 0 < snapshot.number_of_clauses <= 0xFFFFFFFF:
            raise ValueError("number_of_clauses is outside uint32 range")
        if not 0 < snapshot.number_of_features <= 0xFFFFFFFF:
            raise ValueError("number_of_features is outside uint32 range")
        if not 0 < snapshot.states_per_action <= 0x7FFF:
            raise ValueError("states_per_action is outside native uint16 range")
        if not 0 < snapshot.threshold <= 0x7FFFFFFF:
            raise ValueError("threshold is outside positive int32 range")
        if len(snapshot.states) != snapshot.number_of_clauses or any(
            len(row) != snapshot.number_of_features * 2 for row in snapshot.states
        ):
            raise ValueError("TM snapshot state matrix has the wrong shape")

        self.library_path = resolved
        self.number_of_clauses = snapshot.number_of_clauses
        self.number_of_features = snapshot.number_of_features
        self._library = ctypes.CDLL(str(resolved))
        self._library.ptm_abi_version.argtypes = []
        self._library.ptm_abi_version.restype = ctypes.c_uint32
        actual_version = int(self._library.ptm_abi_version())
        if actual_version != PTM_ABI_VERSION:
            raise NativeRuntimeError(
                f"PTM ABI mismatch: Python expects {PTM_ABI_VERSION}, library reports "
                f"{actual_version}"
            )
        self._configure_signatures()

        flattened = tuple(value for row in snapshot.states for value in row)
        if any(
            not 1 <= value <= snapshot.states_per_action * 2
            for value in flattened
        ):
            raise ValueError("TA state lies outside its two action regions")
        native_states = (ctypes.c_uint16 * len(flattened))(*flattened)
        config = _TMModelConfig(
            snapshot.number_of_clauses,
            snapshot.number_of_features,
            snapshot.states_per_action,
            snapshot.threshold,
        )
        handle = ctypes.c_void_p()
        status = self._library.ptm_tm_model_create(
            ctypes.byref(config), native_states, len(flattened), ctypes.byref(handle)
        )
        if status != 0:
            self._raise_status(status, "creation")
        if not handle.value:
            raise NativeRuntimeError("native packed TM returned a null model")
        self._handle = handle

    def _configure_signatures(self) -> None:
        library = self._library
        library.ptm_abi_version.argtypes = []
        library.ptm_abi_version.restype = ctypes.c_uint32
        library.ptm_status_message.argtypes = [ctypes.c_int]
        library.ptm_status_message.restype = ctypes.c_char_p
        library.ptm_tm_model_create.argtypes = [
            ctypes.POINTER(_TMModelConfig),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.ptm_tm_model_create.restype = ctypes.c_int
        library.ptm_tm_model_destroy.argtypes = [ctypes.c_void_p]
        library.ptm_tm_model_destroy.restype = None
        library.ptm_tm_model_selected_backend.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        library.ptm_tm_model_selected_backend.restype = ctypes.c_int
        library.ptm_tm_model_eval_packed64.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint32,
            ctypes.POINTER(_TMBatch64Result),
        ]
        library.ptm_tm_model_eval_packed64.restype = ctypes.c_int
        library.ptm_tm_model_eval_packed64_backend.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.POINTER(_TMBatch64Result),
        ]
        library.ptm_tm_model_eval_packed64_backend.restype = ctypes.c_int

    def _raise_status(self, status: int, operation: str) -> None:
        message = self._library.ptm_status_message(status)
        decoded = message.decode("utf-8", errors="replace") if message else "unknown"
        raise NativeRuntimeError(
            f"native packed TM {operation} failed ({status}): {decoded}"
        )

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None and handle.value:
            self._library.ptm_tm_model_destroy(handle)
            handle.value = None

    def __enter__(self) -> "NativePackedTsetlinMachine":
        if not getattr(self, "_handle", None) or not self._handle.value:
            raise NativeRuntimeError("native packed TM is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def evaluate_packed(
        self,
        feature_words: Sequence[int],
        *,
        valid_example_mask: int = (1 << 64) - 1,
        backend: PackedTMBackend | str = PackedTMBackend.AUTOMATIC,
    ) -> PackedTMResult64:
        if not getattr(self, "_handle", None) or not self._handle.value:
            raise NativeRuntimeError("native packed TM is closed")
        if len(feature_words) != self.number_of_features:
            raise ValueError("packed feature plane has the wrong width")
        if not 0 <= valid_example_mask < 1 << 64:
            raise ValueError("valid_example_mask is outside uint64 range")
        if any(not 0 <= word < 1 << 64 for word in feature_words):
            raise ValueError("packed feature word is outside uint64 range")
        if isinstance(backend, str):
            try:
                backend = PackedTMBackend[backend.upper().replace("-", "")]
            except KeyError as error:
                raise ValueError(f"unknown packed TM backend: {backend}") from error
        else:
            try:
                backend = PackedTMBackend(backend)
            except ValueError as error:
                raise ValueError(f"unknown packed TM backend: {backend}") from error

        native_features = (ctypes.c_uint64 * self.number_of_features)(
            *feature_words
        )
        native_clauses = (ctypes.c_uint64 * self.number_of_clauses)()
        native_feedback_clauses = (ctypes.c_uint64 * self.number_of_clauses)()
        native_result = _AlignedInstance(_TMBatch64Result)
        status = self._library.ptm_tm_model_eval_packed64_backend(
            self._handle,
            native_features,
            self.number_of_features,
            valid_example_mask,
            native_clauses,
            native_feedback_clauses,
            self.number_of_clauses,
            int(backend),
            native_result.pointer,
        )
        if status != 0:
            self._raise_status(status, "evaluation")
        result = native_result.value
        actual_backend = (
            self.selected_backend
            if backend == PackedTMBackend.AUTOMATIC
            else backend
        )
        return PackedTMResult64(
            valid_example_mask=int(result.valid_example_mask),
            prediction_mask=int(result.prediction_mask),
            scores=tuple(int(value) for value in result.scores),
            clause_outputs=tuple(int(value) for value in native_clauses),
            feedback_clause_outputs=tuple(
                int(value) for value in native_feedback_clauses
            ),
            backend=actual_backend,
        )

    @property
    def selected_backend(self) -> PackedTMBackend:
        if not getattr(self, "_handle", None) or not self._handle.value:
            raise NativeRuntimeError("native packed TM is closed")
        result = ctypes.c_int()
        status = self._library.ptm_tm_model_selected_backend(
            self._handle, ctypes.byref(result)
        )
        if status != 0:
            self._raise_status(status, "backend selection")
        return PackedTMBackend(result.value)

    def evaluate_rows(
        self,
        rows: Sequence[Sequence[bool | int]],
        *,
        backend: PackedTMBackend | str = PackedTMBackend.AUTOMATIC,
    ) -> PackedTMResult64:
        if not rows:
            raise ValueError("packed TM row batch cannot be empty")
        if len(rows) > 64:
            raise ValueError("packed TM row batch cannot exceed 64 examples")
        words = [0] * self.number_of_features
        for lane, row in enumerate(rows):
            if len(row) != self.number_of_features:
                raise ValueError("packed TM row has the wrong feature width")
            for feature, value in enumerate(row):
                if bool(value):
                    words[feature] |= 1 << lane
        valid = (1 << len(rows)) - 1 if len(rows) < 64 else (1 << 64) - 1
        return self.evaluate_packed(
            words, valid_example_mask=valid, backend=backend
        )

    def evaluate_literal_batch(
        self,
        batch: LiteralBatch,
        *,
        start_row: int = 0,
        backend: PackedTMBackend | str = PackedTMBackend.AUTOMATIC,
    ) -> PackedTMResult64:
        if batch.literal_count != self.number_of_features:
            raise ValueError("literal batch width does not match TM feature count")
        words, valid = batch.feature_major_words64(start_row)
        return self.evaluate_packed(
            words, valid_example_mask=valid, backend=backend
        )
