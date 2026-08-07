"""Bounded, immutable morphology operations for compiled Logic programs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .logic_ast import LOGIC_AST_VARIABLES
from .logic_consolidation import (
    LOGIC_PROGRAM_CAPACITY,
    FixedLogicInstruction,
    FixedLogicOpcode,
    LogicProgram32,
)


LOGIC_MORPHOLOGY_SCHEMA_VERSION = 1
LOGIC_ASSIGNMENT_COUNT = 1 << len(LOGIC_AST_VARIABLES)


class MorphologyCapacityError(ValueError):
    pass


class MorphologyOperation(str, Enum):
    NOOP = "noop"
    SPECIALIZE = "specialize"
    GENERALIZE = "generalize"
    PATCH_FALSE = "patch_false"
    PATCH_TRUE = "patch_true"
    CONDITIONAL_COMPOSE = "conditional_compose"
    EQUIVALENCE_MERGE = "equivalence_merge"


def _binding_tuple(encoded: int) -> tuple[bool, ...]:
    if not 0 <= encoded < LOGIC_ASSIGNMENT_COUNT:
        raise ValueError("Logic assignment lies outside five-bit space")
    return tuple(bool((encoded >> index) & 1) for index in range(5))


def _binding_index(bindings: Sequence[bool | int]) -> int:
    if len(bindings) != len(LOGIC_AST_VARIABLES):
        raise ValueError("Logic morphology requires bindings for A through E")
    return sum(int(bool(value)) << index for index, value in enumerate(bindings))


@dataclass(frozen=True, slots=True)
class LogicBehaviorSignature:
    truth_bits: int

    def __post_init__(self) -> None:
        if not 0 <= self.truth_bits <= 0xFFFFFFFF:
            raise ValueError("Logic behavior signature must be unsigned 32-bit")

    @classmethod
    def from_program(cls, program: LogicProgram32) -> "LogicBehaviorSignature":
        bits = 0
        for assignment in range(LOGIC_ASSIGNMENT_COUNT):
            if program.evaluate(_binding_tuple(assignment)).value:
                bits |= 1 << assignment
        return cls(bits)

    @property
    def hex_value(self) -> str:
        return f"0x{self.truth_bits:08x}"

    @property
    def true_count(self) -> int:
        return self.truth_bits.bit_count()

    def value(self, bindings: Sequence[bool | int]) -> bool:
        return bool((self.truth_bits >> _binding_index(bindings)) & 1)

    def distance(self, other: "LogicBehaviorSignature") -> int:
        return (self.truth_bits ^ other.truth_bits).bit_count()


class _ProgramBuilder:
    def __init__(self) -> None:
        self.instructions: list[FixedLogicInstruction] = []
        self._interned: dict[FixedLogicInstruction, int] = {}

    def _intern(self, instruction: FixedLogicInstruction) -> int:
        existing = self._interned.get(instruction)
        if existing is not None:
            return existing
        if len(self.instructions) >= LOGIC_PROGRAM_CAPACITY:
            raise MorphologyCapacityError(
                "morphology result exceeds the 32-instruction kernel"
            )
        result = len(self.instructions)
        self.instructions.append(instruction)
        self._interned[instruction] = result
        return result

    def constant(self, value: bool) -> int:
        return self._intern(
            FixedLogicInstruction(FixedLogicOpcode.CONSTANT, argument=int(value))
        )

    def input(self, variable: int) -> int:
        if not 0 <= variable < len(LOGIC_AST_VARIABLES):
            raise ValueError("Logic input index lies outside A through E")
        return self._intern(
            FixedLogicInstruction(FixedLogicOpcode.INPUT, argument=variable)
        )

    def negate(self, operand: int) -> int:
        node = self.instructions[operand]
        if node.opcode is FixedLogicOpcode.CONSTANT:
            return self.constant(not bool(node.argument))
        if node.opcode is FixedLogicOpcode.NOT:
            return node.operand_mask.bit_length() - 1
        return self._intern(
            FixedLogicInstruction(FixedLogicOpcode.NOT, 1 << operand)
        )

    def _flatten(
        self, operands: Iterable[int], operation: FixedLogicOpcode
    ) -> set[int]:
        result: set[int] = set()
        pending = list(operands)
        while pending:
            operand = pending.pop()
            node = self.instructions[operand]
            if node.opcode is operation:
                pending.extend(_mask_indices(node.operand_mask))
            else:
                result.add(operand)
        return result

    def all(self, operands: Iterable[int]) -> int:
        reduced = self._flatten(operands, FixedLogicOpcode.AND)
        for operand in tuple(reduced):
            node = self.instructions[operand]
            if node.opcode is FixedLogicOpcode.CONSTANT:
                if not node.argument:
                    return self.constant(False)
                reduced.remove(operand)
        if _contains_complement(self.instructions, reduced):
            return self.constant(False)
        for operand in tuple(reduced):
            node = self.instructions[operand]
            if (
                node.opcode is FixedLogicOpcode.OR
                and any(child in reduced for child in _mask_indices(node.operand_mask))
            ):
                reduced.remove(operand)
        if not reduced:
            return self.constant(True)
        if len(reduced) == 1:
            return next(iter(reduced))
        mask = sum(1 << operand for operand in reduced)
        return self._intern(FixedLogicInstruction(FixedLogicOpcode.AND, mask))

    def any(self, operands: Iterable[int]) -> int:
        reduced = self._flatten(operands, FixedLogicOpcode.OR)
        for operand in tuple(reduced):
            node = self.instructions[operand]
            if node.opcode is FixedLogicOpcode.CONSTANT:
                if node.argument:
                    return self.constant(True)
                reduced.remove(operand)
        if _contains_complement(self.instructions, reduced):
            return self.constant(True)
        for operand in tuple(reduced):
            node = self.instructions[operand]
            if (
                node.opcode is FixedLogicOpcode.AND
                and any(child in reduced for child in _mask_indices(node.operand_mask))
            ):
                reduced.remove(operand)
        if not reduced:
            return self.constant(False)
        if len(reduced) == 1:
            return next(iter(reduced))
        mask = sum(1 << operand for operand in reduced)
        return self._intern(FixedLogicInstruction(FixedLogicOpcode.OR, mask))

    def parity(self, operands: Iterable[int]) -> int:
        pending = list(operands)
        odd: set[int] = set()
        inverted = False
        while pending:
            operand = pending.pop()
            node = self.instructions[operand]
            if node.opcode is FixedLogicOpcode.CONSTANT:
                inverted ^= bool(node.argument)
            elif node.opcode is FixedLogicOpcode.XOR:
                pending.extend(_mask_indices(node.operand_mask))
            elif operand in odd:
                odd.remove(operand)
            else:
                odd.add(operand)
        for operand in tuple(odd):
            node = self.instructions[operand]
            if node.opcode is FixedLogicOpcode.NOT:
                target = node.operand_mask.bit_length() - 1
                if target in odd:
                    odd.remove(operand)
                    odd.remove(target)
                    inverted = not inverted
        if not odd:
            return self.constant(inverted)
        if len(odd) == 1:
            result = next(iter(odd))
        else:
            mask = sum(1 << operand for operand in odd)
            result = self._intern(
                FixedLogicInstruction(FixedLogicOpcode.XOR, mask)
            )
        return self.negate(result) if inverted else result

    def import_program(self, program: LogicProgram32) -> int:
        remapped: list[int] = []
        for instruction in program.instructions:
            operands = tuple(
                remapped[index] for index in _mask_indices(instruction.operand_mask)
            )
            if instruction.opcode is FixedLogicOpcode.CONSTANT:
                result = self.constant(bool(instruction.argument))
            elif instruction.opcode is FixedLogicOpcode.INPUT:
                result = self.input(instruction.argument)
            elif instruction.opcode is FixedLogicOpcode.NOT:
                result = self.negate(operands[0])
            elif instruction.opcode is FixedLogicOpcode.AND:
                result = self.all(operands)
            elif instruction.opcode is FixedLogicOpcode.OR:
                result = self.any(operands)
            else:
                result = self.parity(operands)
            remapped.append(result)
        return remapped[program.root_instruction]

    def exact_cube(self, bindings: Sequence[bool | int]) -> int:
        if len(bindings) != len(LOGIC_AST_VARIABLES):
            raise ValueError("exact cube requires bindings for A through E")
        literals = []
        for index, value in enumerate(bindings):
            source = self.input(index)
            literals.append(source if bool(value) else self.negate(source))
        return self.all(literals)

    def conditional(self, condition: int, when_true: int, when_false: int) -> int:
        if when_true == when_false:
            return when_true
        return self.any(
            (
                self.all((condition, when_true)),
                self.all((self.negate(condition), when_false)),
            )
        )

    def finish(self, root: int) -> LogicProgram32:
        reachable: list[int] = []
        seen: set[int] = set()

        def visit(node_id: int) -> None:
            if node_id in seen:
                return
            seen.add(node_id)
            for operand in _mask_indices(self.instructions[node_id].operand_mask):
                visit(operand)
            reachable.append(node_id)

        visit(root)
        remapped = {old: new for new, old in enumerate(reachable)}
        compacted = []
        for old in reachable:
            instruction = self.instructions[old]
            mask = sum(
                1 << remapped[operand]
                for operand in _mask_indices(instruction.operand_mask)
            )
            compacted.append(
                FixedLogicInstruction(
                    instruction.opcode, mask, instruction.argument
                )
            )
        return LogicProgram32(tuple(compacted), len(compacted) - 1)


def _mask_indices(mask: int) -> tuple[int, ...]:
    return tuple(index for index in range(mask.bit_length()) if (mask >> index) & 1)


def _contains_complement(
    instructions: Sequence[FixedLogicInstruction], operands: set[int]
) -> bool:
    return any(
        instruction.opcode is FixedLogicOpcode.NOT
        and instruction.operand_mask.bit_length() - 1 in operands
        for index in operands
        for instruction in (instructions[index],)
    )


@dataclass(frozen=True, slots=True)
class LogicMorphologyResult:
    operation: MorphologyOperation
    parent_program_ids: tuple[str, ...]
    parent_signatures: tuple[LogicBehaviorSignature, ...]
    program: LogicProgram32
    behavior_signature: LogicBehaviorSignature
    changed_assignments: tuple[int, ...]
    shared_instruction_savings: int = 0
    counterexample_assignment: int | None = None
    counterexample_expected: bool | None = None

    def to_artifact(self, mapping_version: str) -> "LogicMorphologyArtifact":
        return LogicMorphologyArtifact.create(self, mapping_version)


class LogicMorphology:
    @staticmethod
    def input_program(variable: str) -> LogicProgram32:
        if variable not in LOGIC_AST_VARIABLES:
            raise ValueError("unknown Logic input variable")
        builder = _ProgramBuilder()
        return builder.finish(builder.input(LOGIC_AST_VARIABLES.index(variable)))

    @staticmethod
    def specialize(
        program: LogicProgram32, guard: LogicProgram32
    ) -> LogicMorphologyResult:
        return LogicMorphology._combine(
            MorphologyOperation.SPECIALIZE, program, guard
        )

    @staticmethod
    def generalize(
        program: LogicProgram32, extension: LogicProgram32
    ) -> LogicMorphologyResult:
        return LogicMorphology._combine(
            MorphologyOperation.GENERALIZE, program, extension
        )

    @staticmethod
    def _combine(
        operation: MorphologyOperation,
        program: LogicProgram32,
        modifier: LogicProgram32,
    ) -> LogicMorphologyResult:
        builder = _ProgramBuilder()
        parent_root = builder.import_program(program)
        modifier_root = builder.import_program(modifier)
        if operation is MorphologyOperation.SPECIALIZE:
            root = builder.all((parent_root, modifier_root))
        else:
            root = builder.any((parent_root, modifier_root))
        child = builder.finish(root)
        before = LogicBehaviorSignature.from_program(program)
        modifier_signature = LogicBehaviorSignature.from_program(modifier)
        after = LogicBehaviorSignature.from_program(child)
        if operation is MorphologyOperation.SPECIALIZE:
            if after.truth_bits & ~before.truth_bits:
                raise RuntimeError("specialization introduced new true assignments")
        elif before.truth_bits & ~after.truth_bits:
            raise RuntimeError("generalization removed true assignments")
        return LogicMorphologyResult(
            operation,
            (program.program_id, modifier.program_id),
            (before, modifier_signature),
            child,
            after,
            _changed_assignments(before, after),
            len(program.instructions) + len(modifier.instructions) + 1
            - len(child.instructions),
        )

    @staticmethod
    def patch_counterexample(
        program: LogicProgram32,
        bindings: Sequence[bool | int],
        expected: bool,
    ) -> LogicMorphologyResult:
        assignment = _binding_index(bindings)
        before = LogicBehaviorSignature.from_program(program)
        if before.value(bindings) == expected:
            return LogicMorphologyResult(
                MorphologyOperation.NOOP,
                (program.program_id,),
                (before,),
                program,
                before,
                (),
                counterexample_assignment=assignment,
                counterexample_expected=expected,
            )

        builder = _ProgramBuilder()
        parent = builder.import_program(program)
        cube = builder.exact_cube(bindings)
        if expected:
            operation = MorphologyOperation.PATCH_TRUE
            root = builder.any((parent, cube))
        else:
            operation = MorphologyOperation.PATCH_FALSE
            root = builder.all((parent, builder.negate(cube)))
        child = builder.finish(root)
        after = LogicBehaviorSignature.from_program(child)
        expected_change = 1 << assignment
        if before.truth_bits ^ after.truth_bits != expected_change:
            raise RuntimeError("counterexample patch changed more than its target cube")
        if after.value(bindings) != expected:
            raise RuntimeError("counterexample patch did not repair its target")
        return LogicMorphologyResult(
            operation,
            (program.program_id,),
            (before,),
            child,
            after,
            (assignment,),
            0,
            assignment,
            expected,
        )

    @staticmethod
    def compose_conditional(
        condition: LogicProgram32,
        when_true: LogicProgram32,
        when_false: LogicProgram32,
    ) -> LogicMorphologyResult:
        builder = _ProgramBuilder()
        condition_root = builder.import_program(condition)
        true_root = builder.import_program(when_true)
        false_root = builder.import_program(when_false)
        root = builder.conditional(condition_root, true_root, false_root)
        child = builder.finish(root)
        parents = (condition, when_true, when_false)
        signatures = tuple(LogicBehaviorSignature.from_program(item) for item in parents)
        after = LogicBehaviorSignature.from_program(child)
        for assignment in range(LOGIC_ASSIGNMENT_COUNT):
            row = _binding_tuple(assignment)
            expected = (
                signatures[1].value(row)
                if signatures[0].value(row)
                else signatures[2].value(row)
            )
            if after.value(row) != expected:
                raise RuntimeError("conditional composition failed exhaustive validation")
        return LogicMorphologyResult(
            MorphologyOperation.CONDITIONAL_COMPOSE,
            tuple(item.program_id for item in parents),
            signatures,
            child,
            after,
            (),
            sum(len(item.instructions) for item in parents) + 4
            - len(child.instructions),
        )

    @staticmethod
    def merge_equivalent(programs: Sequence[LogicProgram32]) -> LogicMorphologyResult:
        if len(programs) < 2:
            raise ValueError("equivalence merge requires at least two programs")
        signatures = tuple(LogicBehaviorSignature.from_program(item) for item in programs)
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError("cannot merge behaviorally distinct Logic programs")
        child = min(programs, key=lambda item: (len(item.instructions), item.program_id))
        return LogicMorphologyResult(
            MorphologyOperation.EQUIVALENCE_MERGE,
            tuple(item.program_id for item in programs),
            signatures,
            child,
            signatures[0],
            (),
            sum(len(item.instructions) for item in programs)
            - len(child.instructions),
        )


def _changed_assignments(
    before: LogicBehaviorSignature, after: LogicBehaviorSignature
) -> tuple[int, ...]:
    changed = before.truth_bits ^ after.truth_bits
    return tuple(
        assignment
        for assignment in range(LOGIC_ASSIGNMENT_COUNT)
        if (changed >> assignment) & 1
    )


@dataclass(frozen=True, slots=True)
class LogicMorphologyArtifact:
    artifact_id: str
    mapping_version: str
    operation: MorphologyOperation
    parent_program_ids: tuple[str, ...]
    parent_signatures: tuple[str, ...]
    child_program_id: str
    child_signature: str
    changed_assignments: tuple[int, ...]
    child_instruction_count: int
    shared_instruction_savings: int
    counterexample_assignment: int | None = None
    counterexample_expected: bool | None = None
    schema_version: int = LOGIC_MORPHOLOGY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LOGIC_MORPHOLOGY_SCHEMA_VERSION:
            raise ValueError("unsupported Logic morphology artifact schema")
        if not self.mapping_version or not self.parent_program_ids:
            raise ValueError("morphology lineage requires mapping and parent IDs")
        if len(self.parent_program_ids) != len(self.parent_signatures):
            raise ValueError("morphology parent IDs and signatures differ in length")
        if not 1 <= self.child_instruction_count <= LOGIC_PROGRAM_CAPACITY:
            raise ValueError("morphology child instruction count is invalid")
        if tuple(sorted(set(self.changed_assignments))) != self.changed_assignments:
            raise ValueError("changed assignments must be unique and sorted")
        if any(not 0 <= item < LOGIC_ASSIGNMENT_COUNT for item in self.changed_assignments):
            raise ValueError("changed assignment lies outside five-bit space")
        if (self.counterexample_assignment is None) != (
            self.counterexample_expected is None
        ):
            raise ValueError("counterexample assignment and expected value must coexist")
        if (
            self.counterexample_assignment is not None
            and not 0 <= self.counterexample_assignment < LOGIC_ASSIGNMENT_COUNT
        ):
            raise ValueError("counterexample assignment lies outside five-bit space")

    @classmethod
    def create(
        cls,
        result: LogicMorphologyResult,
        mapping_version: str,
    ) -> "LogicMorphologyArtifact":
        provisional = cls(
            "sha256:pending",
            mapping_version,
            result.operation,
            result.parent_program_ids,
            tuple(signature.hex_value for signature in result.parent_signatures),
            result.program.program_id,
            result.behavior_signature.hex_value,
            result.changed_assignments,
            len(result.program.instructions),
            result.shared_instruction_savings,
            result.counterexample_assignment,
            result.counterexample_expected,
        )
        artifact_id = "sha256:" + hashlib.sha256(
            provisional.canonical_content().encode("utf-8")
        ).hexdigest()
        return cls(
            artifact_id,
            provisional.mapping_version,
            provisional.operation,
            provisional.parent_program_ids,
            provisional.parent_signatures,
            provisional.child_program_id,
            provisional.child_signature,
            provisional.changed_assignments,
            provisional.child_instruction_count,
            provisional.shared_instruction_savings,
            provisional.counterexample_assignment,
            provisional.counterexample_expected,
        )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_kind": "class_ii_logic_morphology",
            "mapping_version": self.mapping_version,
            "behavior_domain": {
                "variables": list(LOGIC_AST_VARIABLES),
                "assignment_count": LOGIC_ASSIGNMENT_COUNT,
                "bit_order": "A_lsb_through_E_msb",
            },
            "operation": self.operation.value,
            "parent_program_ids": list(self.parent_program_ids),
            "parent_signatures": list(self.parent_signatures),
            "child_program_id": self.child_program_id,
            "child_signature": self.child_signature,
            "changed_assignments": list(self.changed_assignments),
            "child_instruction_count": self.child_instruction_count,
            "shared_instruction_savings": self.shared_instruction_savings,
            "counterexample": (
                None
                if self.counterexample_assignment is None
                else {
                    "assignment": self.counterexample_assignment,
                    "expected": self.counterexample_expected,
                }
            ),
            "validation": {
                "exhaustive_assignment_count": LOGIC_ASSIGNMENT_COUNT,
                "mismatch_count": 0,
            },
            "restoration_program_ids": list(self.parent_program_ids),
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
        return self.artifact_id == "sha256:" + hashlib.sha256(
            self.canonical_content().encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        result = self._content_dict()
        result["artifact_id"] = self.artifact_id
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LogicMorphologyArtifact":
        if value.get("artifact_kind") != "class_ii_logic_morphology":
            raise ValueError("unsupported Logic morphology artifact kind")
        if value.get("behavior_domain") != {
            "variables": list(LOGIC_AST_VARIABLES),
            "assignment_count": LOGIC_ASSIGNMENT_COUNT,
            "bit_order": "A_lsb_through_E_msb",
        }:
            raise ValueError("unsupported Logic morphology behavior domain")
        validation = value.get("validation")
        if validation != {
            "exhaustive_assignment_count": LOGIC_ASSIGNMENT_COUNT,
            "mismatch_count": 0,
        }:
            raise ValueError("morphology artifact lacks exhaustive validation")
        parent_ids = tuple(str(item) for item in value["parent_program_ids"])
        if tuple(value.get("restoration_program_ids", ())) != parent_ids:
            raise ValueError("morphology restoration lineage does not match parents")
        counterexample = value.get("counterexample")
        if counterexample is not None:
            if not isinstance(counterexample, Mapping):
                raise ValueError("morphology counterexample must be an object")
            if type(counterexample.get("expected")) is not bool:
                raise ValueError(
                    "morphology counterexample expected value must be Boolean"
                )
        artifact = cls(
            artifact_id=str(value["artifact_id"]),
            mapping_version=str(value["mapping_version"]),
            operation=MorphologyOperation(value["operation"]),
            parent_program_ids=parent_ids,
            parent_signatures=tuple(str(item) for item in value["parent_signatures"]),
            child_program_id=str(value["child_program_id"]),
            child_signature=str(value["child_signature"]),
            changed_assignments=tuple(int(item) for item in value["changed_assignments"]),
            child_instruction_count=int(value["child_instruction_count"]),
            shared_instruction_savings=int(value["shared_instruction_savings"]),
            counterexample_assignment=(
                None if counterexample is None else int(counterexample["assignment"])
            ),
            counterexample_expected=(
                None if counterexample is None else bool(counterexample["expected"])
            ),
            schema_version=int(value["schema_version"]),
        )
        if not artifact.verify_artifact_id():
            raise ValueError("Logic morphology artifact hash does not match content")
        return artifact
