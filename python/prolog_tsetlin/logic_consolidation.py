"""Fixed-shape Class II compilation for typed Logic programs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence

from .artifact import (
    InputShape,
    RestorationHandle,
    SlotBinding,
    SourceKind,
    ValidationSignature,
)
from .logic_ast import (
    LOGIC_AST_SCHEMA_VERSION,
    LOGIC_AST_VARIABLES,
    PRIMITIVE_LOGIC_SCHEMA_VERSION,
    PrimitiveLogicGraph,
    PrimitiveLogicOp,
)
from .pa import PortSemantic


LOGIC_PROGRAM_SCHEMA_VERSION = 1
LOGIC_EVALUATOR_ARTIFACT_SCHEMA_VERSION = 1
LOGIC_PROGRAM_CAPACITY = 32


class FixedLogicOpcode(IntEnum):
    CONSTANT = 0
    INPUT = 1
    NOT = 2
    AND = 3
    OR = 4
    XOR = 5


_PRIMITIVE_OPCODE = {
    PrimitiveLogicOp.CONSTANT: FixedLogicOpcode.CONSTANT,
    PrimitiveLogicOp.INPUT: FixedLogicOpcode.INPUT,
    PrimitiveLogicOp.NOT: FixedLogicOpcode.NOT,
    PrimitiveLogicOp.AND: FixedLogicOpcode.AND,
    PrimitiveLogicOp.OR: FixedLogicOpcode.OR,
    PrimitiveLogicOp.XOR: FixedLogicOpcode.XOR,
}


@dataclass(frozen=True, slots=True)
class FixedLogicInstruction:
    opcode: FixedLogicOpcode
    operand_mask: int = 0
    argument: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.opcode, FixedLogicOpcode):
            raise ValueError("instruction opcode is not part of schema version 1")
        if not 0 <= self.operand_mask <= 0xFFFFFFFF:
            raise ValueError("instruction operand mask must be unsigned 32-bit")
        if not 0 <= self.argument <= 0xFF:
            raise ValueError("instruction argument must be unsigned 8-bit")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "opcode": self.opcode.name.lower(),
            "opcode_value": int(self.opcode),
            "operand_mask": self.operand_mask,
            "argument": self.argument,
        }


@dataclass(frozen=True, slots=True)
class FixedLogicResult:
    value: bool
    true_instruction_mask: int
    evaluated_instruction_mask: int


@dataclass(frozen=True, slots=True)
class LogicProgram32:
    instructions: tuple[FixedLogicInstruction, ...]
    root_instruction: int
    schema_version: int = LOGIC_PROGRAM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LOGIC_PROGRAM_SCHEMA_VERSION:
            raise ValueError("unsupported fixed Logic program schema")
        count = len(self.instructions)
        if not 1 <= count <= LOGIC_PROGRAM_CAPACITY:
            raise ValueError("fixed Logic program must contain 1 through 32 instructions")
        if self.root_instruction != count - 1:
            raise ValueError("fixed Logic program root must be its final instruction")
        for index, instruction in enumerate(self.instructions):
            if instruction.operand_mask & ~((1 << index) - 1):
                raise ValueError("fixed Logic instruction has a forward reference")
            operand_count = instruction.operand_mask.bit_count()
            if instruction.opcode is FixedLogicOpcode.CONSTANT:
                valid = instruction.operand_mask == 0 and instruction.argument <= 1
            elif instruction.opcode is FixedLogicOpcode.INPUT:
                valid = (
                    instruction.operand_mask == 0
                    and instruction.argument < len(LOGIC_AST_VARIABLES)
                )
            elif instruction.opcode is FixedLogicOpcode.NOT:
                valid = operand_count == 1 and instruction.argument == 0
            else:
                valid = operand_count >= 2 and instruction.argument == 0
            if not valid:
                raise ValueError(f"malformed fixed Logic instruction at index {index}")

    @classmethod
    def compile(cls, graph: PrimitiveLogicGraph) -> "LogicProgram32":
        remapped: dict[int, int] = {}
        instructions: list[FixedLogicInstruction] = []

        def visit(node_id: int) -> int:
            existing = remapped.get(node_id)
            if existing is not None:
                return existing
            node = graph.nodes[node_id]
            operands = tuple(visit(operand) for operand in node.operands)
            if len(instructions) >= LOGIC_PROGRAM_CAPACITY:
                raise ValueError("primitive graph exceeds the 32-instruction kernel")
            opcode = _PRIMITIVE_OPCODE[node.operation]
            operand_mask = sum(1 << operand for operand in operands)
            if node.operation is PrimitiveLogicOp.CONSTANT:
                argument = int(bool(node.constant))
            elif node.operation is PrimitiveLogicOp.INPUT:
                if node.variable not in LOGIC_AST_VARIABLES:
                    raise ValueError("primitive graph references an unknown Logic input")
                argument = LOGIC_AST_VARIABLES.index(node.variable)
            else:
                argument = 0
            compiled_id = len(instructions)
            instructions.append(FixedLogicInstruction(opcode, operand_mask, argument))
            remapped[node_id] = compiled_id
            return compiled_id

        root = visit(graph.root_id)
        return cls(tuple(instructions), root)

    @property
    def program_id(self) -> str:
        return "sha256:" + hashlib.sha256(
            self.canonical_content().encode("utf-8")
        ).hexdigest()

    def canonical_content(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "program_kind": "logic_program_32",
            "instruction_count": len(self.instructions),
            "root_instruction": self.root_instruction,
            "instructions": [instruction.to_dict() for instruction in self.instructions],
        }

    def evaluate(self, bindings: Sequence[bool | int]) -> FixedLogicResult:
        if len(bindings) != len(LOGIC_AST_VARIABLES):
            raise ValueError("fixed Logic program requires bindings for A through E")
        binding_bits = sum(int(bool(value)) << index for index, value in enumerate(bindings))
        values = 0
        for index, instruction in enumerate(self.instructions):
            selected = values & instruction.operand_mask
            if instruction.opcode is FixedLogicOpcode.CONSTANT:
                value = bool(instruction.argument)
            elif instruction.opcode is FixedLogicOpcode.INPUT:
                value = bool((binding_bits >> instruction.argument) & 1)
            elif instruction.opcode is FixedLogicOpcode.NOT:
                value = selected == 0
            elif instruction.opcode is FixedLogicOpcode.AND:
                value = selected == instruction.operand_mask
            elif instruction.opcode is FixedLogicOpcode.OR:
                value = selected != 0
            else:
                value = bool(selected.bit_count() & 1)
            if value:
                values |= 1 << index
        return FixedLogicResult(
            value=bool((values >> self.root_instruction) & 1),
            true_instruction_mask=values,
            evaluated_instruction_mask=(1 << len(self.instructions)) - 1,
        )


@dataclass(frozen=True, slots=True)
class LogicEvaluatorArtifact:
    artifact_id: str
    mapping_version: str
    validation_signature: ValidationSignature
    restoration_handle: RestorationHandle
    schema_version: int = LOGIC_EVALUATOR_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LOGIC_EVALUATOR_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported Logic evaluator artifact schema")
        if not self.mapping_version:
            raise ValueError("mapping_version cannot be empty")

    @classmethod
    def create(
        cls,
        *,
        mapping_version: str,
        validation_signature: ValidationSignature,
        restoration_handle: RestorationHandle,
    ) -> "LogicEvaluatorArtifact":
        provisional = cls(
            artifact_id="sha256:pending",
            mapping_version=mapping_version,
            validation_signature=validation_signature,
            restoration_handle=restoration_handle,
        )
        artifact_id = "sha256:" + hashlib.sha256(
            provisional.canonical_content().encode("utf-8")
        ).hexdigest()
        return cls(
            artifact_id=artifact_id,
            mapping_version=mapping_version,
            validation_signature=validation_signature,
            restoration_handle=restoration_handle,
        )

    def _content_dict(self) -> dict[str, Any]:
        slot_bindings = tuple(
            SlotBinding(
                slot=index,
                source_kind=SourceKind.LITERAL,
                source_id=f"logic_binding:{variable}",
            )
            for index, variable in enumerate(LOGIC_AST_VARIABLES)
        )
        return {
            "schema_version": self.schema_version,
            "artifact_kind": "class_ii_logic_evaluator",
            "mapping_version": self.mapping_version,
            "input_shape": InputShape.PA_32X32.value,
            "port_semantic": PortSemantic.LITERAL_TRUTH.value,
            "slot_bindings": [binding.to_dict() for binding in slot_bindings],
            "kernel": {
                "kernel_kind": "logic_program_32_v1",
                "program_capacity": LOGIC_PROGRAM_CAPACITY,
                "binding_variables": list(LOGIC_AST_VARIABLES),
                "opcodes": {
                    opcode.name.lower(): int(opcode) for opcode in FixedLogicOpcode
                },
                "logic_ast_schema_version": LOGIC_AST_SCHEMA_VERSION,
                "primitive_logic_schema_version": PRIMITIVE_LOGIC_SCHEMA_VERSION,
            },
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
    def from_dict(cls, value: Mapping[str, Any]) -> "LogicEvaluatorArtifact":
        if value.get("artifact_kind") != "class_ii_logic_evaluator":
            raise ValueError("unsupported Logic evaluator artifact kind")
        expected_kernel = cls.create(
            mapping_version=str(value["mapping_version"]),
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
        expected_content = expected_kernel._content_dict()
        for field in ("input_shape", "port_semantic", "slot_bindings", "kernel"):
            if value.get(field) != expected_content[field]:
                raise ValueError(f"unsupported Logic evaluator {field}")
        artifact = cls(
            artifact_id=str(value["artifact_id"]),
            mapping_version=expected_kernel.mapping_version,
            validation_signature=expected_kernel.validation_signature,
            restoration_handle=expected_kernel.restoration_handle,
            schema_version=int(value["schema_version"]),
        )
        if not artifact.verify_artifact_id():
            raise ValueError("Logic evaluator artifact hash does not match its content")
        return artifact
