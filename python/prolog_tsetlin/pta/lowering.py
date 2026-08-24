"""Exact lowering gate — YES iff proposal has an exact native representation.

Pipeline:
  validate_proposal_schema()
    ↓
  resolve_required_literals()  (preview descriptors, no catalog mutation)
    ↓
  lower_exact()  (attempt construction of NativeCandidate)
    ↓
  LoweredCandidate | NotRepresentable
    ↓
  independent semantic oracle → shadow/audit

Only successful construction means “lowerable”. syntactically_bounded() is the
shallow preliminary check; lower_exact() is the gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable

from ..logic_ast import LOGIC_AST_VARIABLES
from ..representation import LiteralDescriptor
from .executable import ExecutableBinaryClause
from .proposal import (
    MAX_CLAUSES,
    MAX_GRAPH_DEPTH,
    NATIVE_TARGETS,
    NativeTarget,
    PTAEscalationProposal,
)

MAX_LITERALS_PER_CLAUSE = 64
MAX_WEIGHT_ABS = 1_000_000
PATCH_MAX_CELLS = 1 << 20
_LOWERED_CANDIDATE_TOKEN = object()


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class LoweredCandidate:
    """Successful exact lowering containing an executable target object."""

    proposal: PTAEscalationProposal
    native_object: Any
    native_kind: str
    description: str = "ok"
    _verification_token: object = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self._verification_token is not _LOWERED_CANDIDATE_TOKEN:
            raise TypeError("LoweredCandidate can only be created by lower_exact()")
        if self.proposal.native_target == "binary_clause":
            if self.native_kind != "executable_binary_clause" or not isinstance(
                self.native_object, ExecutableBinaryClause
            ):
                raise TypeError(
                    "binary_clause candidates require ExecutableBinaryClause"
                )
            if self.native_object.literal_ids != self.proposal.structure.get(
                "clause"
            ):
                raise ValueError(
                    "binary_clause candidate differs from the declared structure"
                )
            return
        if self.proposal.native_target == "logic_program":
            from ..logic_consolidation import LogicProgram32

            if self.native_kind != "logic_program32" or not isinstance(
                self.native_object, LogicProgram32
            ):
                raise TypeError("logic_program candidates require LogicProgram32")
            if self.native_object.to_dict() != _plain_value(
                self.proposal.structure.get("program")
            ):
                raise ValueError(
                    "logic_program candidate differs from the declared structure"
                )
            return
        raise TypeError(
            f"{self.proposal.native_target} has no executable exact representation"
        )


@dataclass(frozen=True, slots=True)
class NotRepresentable:
    """Proposal cannot be lowered to exact native representation."""

    proposal: PTAEscalationProposal
    reason: str


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and __import__("math").isfinite(float(v))


def validate_proposal_schema(proposal: PTAEscalationProposal) -> tuple[bool, str]:
    if not proposal.proposal_id or any(ord(c) < 0x20 for c in proposal.proposal_id):
        return False, "proposal_id must be nonempty printable"
    if not proposal.source_pta_ids:
        return False, "source_pta_ids empty"
    if proposal.native_target not in NATIVE_TARGETS:
        return False, f"unknown target {proposal.native_target}"
    for k, v in proposal.resource_bounds.items():
        if type(v) is not int or isinstance(v, bool) or v <= 0:
            return False, f"resource_bounds[{k}] must be positive int"
    if proposal.resource_bounds.get("clause_count", 1) > MAX_CLAUSES:
        return False, "clause_count exceeds native bank"
    if proposal.resource_bounds.get("graph_depth", 1) > MAX_GRAPH_DEPTH:
        return False, "graph_depth exceeds MAX_GRAPH_DEPTH"
    if proposal.resource_bounds.get("literal_count", 0) > MAX_LITERALS_PER_CLAUSE:
        return False, "literal_count exceeds native bound"
    return True, "ok"


def syntactically_bounded(proposal: PTAEscalationProposal) -> tuple[bool, str]:
    ok, msg = validate_proposal_schema(proposal)
    if not ok:
        return ok, msg
    rb = proposal.resource_bounds
    struct = proposal.structure
    target = proposal.native_target

    if target in ("binary_clause", "shared_weighted_clause", "regression_clause"):
        if target == "shared_weighted_clause":
            if "weights" in struct:
                # Allow dict or MappingProxyType after freeze
                wdict = struct["weights"]
                if not isinstance(wdict, Mapping) or not len(wdict):
                    return False, "shared weights must be nonempty dict"
                if any(not isinstance(w, int) or isinstance(w, bool) or abs(w) > MAX_WEIGHT_ABS for w in wdict.values()):
                    return False, "weight out of int32 bounded range"
            if proposal.weights is not None:
                if any(not isinstance(w, int) or isinstance(w, bool) or abs(w) > MAX_WEIGHT_ABS for w in proposal.weights):
                    return False, "weight out of int32 bounded range"
            return True, "ok (syntactically bounded)"
        clause = struct.get("clause") or struct.get("literals") or []
        if clause:
            if not isinstance(clause, (list, tuple)) or not all(isinstance(lit, int) and not isinstance(lit, bool) for lit in clause):
                return False, "clause literals must be integer IDs"
            if len(clause) > MAX_LITERALS_PER_CLAUSE:
                return False, "clause exceeds literal ceiling"
        if not clause and not proposal.required_literals:
            return False, "clause literals must be nonempty list or descriptor"
        return True, "ok (syntactically bounded)"

    if target == "graph_clause":
        depth = rb.get("graph_depth", struct.get("depth", 1))
        if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= MAX_GRAPH_DEPTH:
            if struct.get("recursive_unbounded") is True:
                return True, "ok (syntactically bounded, unbounded probe)"
            return False, "graph_depth 1..8 required"
        if struct.get("recursive_unbounded") is True:
            return True, "ok (syntactically bounded, unbounded probe)"
        return True, "ok (syntactically bounded)"

    if target == "patch_clause":
        if "patch_extent" in rb:
            pe = rb["patch_extent"]
            if isinstance(pe, int):
                if pe > PATCH_MAX_CELLS:
                    return False, "patch extent exceeds bounded cells"
            elif isinstance(pe, Mapping):
                cells = pe.get("rows", 1) * pe.get("cols", 1)
                if cells > PATCH_MAX_CELLS:
                    return False, "patch extent exceeds bounded cells"
        extent = struct.get("patch")
        if isinstance(extent, Mapping):
            cells = extent.get("rows", 1) * extent.get("cols", 1)
            if cells > PATCH_MAX_CELLS:
                return False, "patch extent exceeds bounded cells"
        if not isinstance(struct.get("kind", ""), str):
            return False, "patch kind must be string"
        return True, "ok (syntactically bounded)"

    if target in ("logic_program", "threshold", "composite_gate"):
        if not struct:
            return False, "structure must be nonempty"
        if "literal_count" in rb and rb["literal_count"] > MAX_LITERALS_PER_CLAUSE:
            return False, "literal_count exceeds bound"
        return True, "ok (syntactically bounded, delegated)"

    return False, f"unknown target {target}"


def _decode_logic_program32(value: Any) -> Any | None:
    """Decode the canonical mapping form into a validated LogicProgram32."""
    from ..logic_consolidation import (
        FixedLogicInstruction,
        FixedLogicOpcode,
        LogicProgram32,
    )

    if not isinstance(value, Mapping):
        return None
    expected_program_keys = {
        "schema_version",
        "program_kind",
        "instruction_count",
        "root_instruction",
        "instructions",
    }
    if set(value) != expected_program_keys:
        return None
    schema_version = value["schema_version"]
    instruction_count = value["instruction_count"]
    root_instruction = value["root_instruction"]
    raw_instructions = value["instructions"]
    if (
        type(schema_version) is not int
        or value["program_kind"] != "logic_program_32"
        or type(instruction_count) is not int
        or type(root_instruction) is not int
        or isinstance(raw_instructions, (str, bytes))
        or not isinstance(raw_instructions, Sequence)
        or instruction_count != len(raw_instructions)
    ):
        return None

    decoded: list[FixedLogicInstruction] = []
    expected_instruction_keys = {
        "opcode",
        "opcode_value",
        "operand_mask",
        "argument",
    }
    try:
        for raw in raw_instructions:
            if not isinstance(raw, Mapping) or set(raw) != expected_instruction_keys:
                return None
            opcode_value = raw["opcode_value"]
            operand_mask = raw["operand_mask"]
            argument = raw["argument"]
            if (
                type(opcode_value) is not int
                or type(operand_mask) is not int
                or type(argument) is not int
            ):
                return None
            opcode = FixedLogicOpcode(opcode_value)
            if raw["opcode"] != opcode.name.lower():
                return None
            decoded.append(FixedLogicInstruction(opcode, operand_mask, argument))
        return LogicProgram32(
            tuple(decoded),
            root_instruction,
            schema_version=schema_version,
        )
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class _ConstructedCandidate:
    native_object: Any
    native_kind: str


_Constructor = Callable[
    [PTAEscalationProposal, Any | None, Any | None],
    _ConstructedCandidate | NotRepresentable,
]
_SemanticOracle = Callable[
    [PTAEscalationProposal, _ConstructedCandidate, Any | None, Any | None],
    tuple[bool, str],
]


@dataclass(frozen=True, slots=True)
class _TargetLowerer:
    construct: _Constructor
    semantic_oracle: _SemanticOracle


def _required_literal_id(value: LiteralDescriptor | str) -> int | None:
    if isinstance(value, LiteralDescriptor):
        return value.literal_id
    prefix = "literal:"
    if not value.startswith(prefix):
        return None
    raw = value[len(prefix) :]
    if (
        not raw
        or not raw.isascii()
        or not raw.isdecimal()
        or str(int(raw)) != raw
    ):
        return None
    literal_id = int(raw)
    return literal_id if literal_id < (1 << 64) else None


def _lower_binary_clause(
    proposal: PTAEscalationProposal,
    catalog: Any | None,
    context: Any | None,
) -> _ConstructedCandidate | NotRepresentable:
    del context
    clause = proposal.structure.get("clause")
    if not isinstance(clause, tuple) or not clause:
        return NotRepresentable(
            proposal, "binary_clause requires a nonempty canonical clause tuple"
        )
    if any(
        type(literal_id) is not int or not 0 <= literal_id < (1 << 64)
        for literal_id in clause
    ):
        return NotRepresentable(
            proposal, "binary_clause literal IDs must be unsigned 64-bit integers"
        )
    if clause != tuple(sorted(set(clause))):
        return NotRepresentable(
            proposal, "binary_clause literal IDs must be sorted and unique"
        )
    if catalog is None:
        return NotRepresentable(
            proposal, "binary_clause requires a materialized LiteralCatalog"
        )
    registered = {
        descriptor.literal_id: descriptor
        for descriptor in getattr(catalog, "literals", ())
        if isinstance(descriptor, LiteralDescriptor)
    }
    if any(literal_id not in registered for literal_id in clause):
        return NotRepresentable(
            proposal, "binary_clause references a literal absent from the catalog"
        )
    descriptors = tuple(registered[literal_id] for literal_id in clause)
    try:
        for descriptor in descriptors:
            if catalog.validate_descriptor(descriptor) != descriptor:
                raise ValueError("descriptor mismatch")
    except (AttributeError, KeyError, TypeError, ValueError):
        return NotRepresentable(
            proposal, "binary_clause catalog descriptor is not canonical"
        )

    required_ids: list[int] = []
    for required in proposal.required_literals:
        literal_id = _required_literal_id(required)
        if literal_id is None or literal_id not in registered:
            return NotRepresentable(
                proposal, "required literal is not materialized in the catalog"
            )
        if isinstance(required, LiteralDescriptor) and registered[literal_id] != required:
            return NotRepresentable(
                proposal, "required literal descriptor differs from the catalog"
            )
        required_ids.append(literal_id)
    if required_ids != sorted(set(required_ids)):
        return NotRepresentable(
            proposal, "required literal identities must be sorted and unique"
        )
    if not set(required_ids).issubset(clause):
        return NotRepresentable(
            proposal, "required literals are not contained in the declared clause"
        )

    return _ConstructedCandidate(
        ExecutableBinaryClause(descriptors), "executable_binary_clause"
    )


def _oracle_binary_clause(
    proposal: PTAEscalationProposal,
    candidate: _ConstructedCandidate,
    catalog: Any | None,
    context: Any | None,
) -> tuple[bool, str]:
    del catalog, context
    native = candidate.native_object
    clause = proposal.structure["clause"]
    if candidate.native_kind != "executable_binary_clause" or not isinstance(
        native, ExecutableBinaryClause
    ):
        return False, "binary clause constructor returned the wrong executable type"
    if native.literal_ids != clause:
        return False, "binary clause executable differs from declared literal order"
    all_true = {literal_id: True for literal_id in native.literal_ids}
    if not native.evaluate(all_true):
        return False, "binary clause rejected the all-true assignment"
    for literal_id in dict.fromkeys(native.literal_ids):
        assignment = dict(all_true)
        assignment[literal_id] = False
        if native.evaluate(assignment):
            return False, "binary clause accepted an assignment with a false literal"
    return True, "binary conjunction oracle passed"


def _lower_logic_program(
    proposal: PTAEscalationProposal,
    catalog: Any | None,
    context: Any | None,
) -> _ConstructedCandidate | NotRepresentable:
    del catalog, context
    raw_program = proposal.structure.get("program")
    program = _decode_logic_program32(raw_program)
    if program is None:
        if "pattern" in proposal.structure or "window" in proposal.structure:
            reason = "pattern-only logic is not compiled to LogicProgram32"
        else:
            reason = "logic_program requires a canonical LogicProgram32 mapping"
        return NotRepresentable(proposal, reason)
    return _ConstructedCandidate(program, "logic_program32")


def _evaluate_declared_logic_program(
    program: Mapping[str, Any], bindings: tuple[bool, ...]
) -> bool:
    values: list[bool] = []
    for instruction in program["instructions"]:
        operand_mask = instruction["operand_mask"]
        selected = tuple(
            values[index]
            for index in range(len(values))
            if operand_mask & (1 << index)
        )
        opcode = instruction["opcode"]
        if opcode == "constant":
            value = bool(instruction["argument"])
        elif opcode == "input":
            value = bindings[instruction["argument"]]
        elif opcode == "not":
            value = not selected[0]
        elif opcode == "and":
            value = all(selected)
        elif opcode == "or":
            value = any(selected)
        elif opcode == "xor":
            value = bool(sum(selected) & 1)
        else:
            raise AssertionError(f"unvalidated opcode: {opcode}")
        values.append(value)
    return values[program["root_instruction"]]


def _oracle_logic_program(
    proposal: PTAEscalationProposal,
    candidate: _ConstructedCandidate,
    catalog: Any | None,
    context: Any | None,
) -> tuple[bool, str]:
    del catalog, context
    program = candidate.native_object
    from ..logic_consolidation import LogicProgram32

    if candidate.native_kind != "logic_program32" or not isinstance(
        program, LogicProgram32
    ):
        return False, "logic program constructor returned the wrong executable type"
    declared = _plain_value(proposal.structure["program"])
    if program.to_dict() != declared:
        return False, "LogicProgram32 differs from the declared canonical mapping"
    width = len(LOGIC_AST_VARIABLES)
    for assignment in range(1 << width):
        bindings = tuple(bool(assignment & (1 << index)) for index in range(width))
        expected = _evaluate_declared_logic_program(declared, bindings)
        if program.evaluate(bindings).value is not expected:
            return False, "LogicProgram32 behavioral oracle disagreed"
    return True, "LogicProgram32 exhaustive oracle passed"


def _lower_threshold(
    proposal: PTAEscalationProposal,
    catalog: Any | None,
    context: Any | None,
) -> _ConstructedCandidate | NotRepresentable:
    del catalog, context
    return NotRepresentable(
        proposal,
        "threshold has no exact executable target; use binary_clause for a "
        "materialized threshold literal until masked-threshold lowering is defined",
    )


def _oracle_threshold(*args: Any) -> tuple[bool, str]:
    del args
    return False, "threshold has no executable semantic oracle"


def _lower_shared_weighted_clause(
    proposal: PTAEscalationProposal,
    catalog: Any | None,
    context: Any | None,
) -> _ConstructedCandidate | NotRepresentable:
    del catalog, context
    return NotRepresentable(proposal, "native CoTM execution is not implemented")


def _oracle_shared_weighted_clause(*args: Any) -> tuple[bool, str]:
    del args
    return False, "shared weighted clauses have no executable semantic oracle"


def _lower_regression_clause(
    proposal: PTAEscalationProposal,
    catalog: Any | None,
    context: Any | None,
) -> _ConstructedCandidate | NotRepresentable:
    del catalog, context
    return NotRepresentable(
        proposal, "regression_clause has no executable RTM representation"
    )


def _oracle_regression_clause(*args: Any) -> tuple[bool, str]:
    del args
    return False, "regression clauses have no executable semantic oracle"


def _lower_graph_clause(
    proposal: PTAEscalationProposal,
    catalog: Any | None,
    context: Any | None,
) -> _ConstructedCandidate | NotRepresentable:
    del catalog, context
    if proposal.structure.get("recursive_unbounded") is True:
        reason = "unbounded recursion is not lowerable to graph_tm_v1"
    else:
        reason = "Graph execution is unsupported by the exact native gate"
    return NotRepresentable(proposal, reason)


def _oracle_graph_clause(*args: Any) -> tuple[bool, str]:
    del args
    return False, "graph clauses have no executable semantic oracle"


def _lower_patch_clause(
    proposal: PTAEscalationProposal,
    catalog: Any | None,
    context: Any | None,
) -> _ConstructedCandidate | NotRepresentable:
    del catalog, context
    return NotRepresentable(proposal, "native CTM patch execution is not implemented")


def _oracle_patch_clause(*args: Any) -> tuple[bool, str]:
    del args
    return False, "patch clauses have no executable semantic oracle"


def _lower_composite_gate(
    proposal: PTAEscalationProposal,
    catalog: Any | None,
    context: Any | None,
) -> _ConstructedCandidate | NotRepresentable:
    del catalog, context
    return NotRepresentable(proposal, "native composite execution is not implemented")


def _oracle_composite_gate(*args: Any) -> tuple[bool, str]:
    del args
    return False, "composite gates have no executable semantic oracle"


_TARGET_LOWERERS: Mapping[NativeTarget, _TargetLowerer] = MappingProxyType(
    {
        "binary_clause": _TargetLowerer(
            _lower_binary_clause, _oracle_binary_clause
        ),
        "logic_program": _TargetLowerer(_lower_logic_program, _oracle_logic_program),
        "threshold": _TargetLowerer(_lower_threshold, _oracle_threshold),
        "shared_weighted_clause": _TargetLowerer(
            _lower_shared_weighted_clause, _oracle_shared_weighted_clause
        ),
        "regression_clause": _TargetLowerer(
            _lower_regression_clause, _oracle_regression_clause
        ),
        "graph_clause": _TargetLowerer(_lower_graph_clause, _oracle_graph_clause),
        "patch_clause": _TargetLowerer(_lower_patch_clause, _oracle_patch_clause),
        "composite_gate": _TargetLowerer(
            _lower_composite_gate, _oracle_composite_gate
        ),
    }
)

if tuple(_TARGET_LOWERERS) != NATIVE_TARGETS:
    raise RuntimeError("every NativeTarget must have one exact lowerer entry")


def lower_exact(
    proposal: PTAEscalationProposal, *, catalog: Any | None = None, context: Any | None = None
) -> LoweredCandidate | NotRepresentable:
    ok, msg = syntactically_bounded(proposal)
    if not ok:
        return NotRepresentable(proposal, msg)
    lowerer = _TARGET_LOWERERS[proposal.native_target]
    constructed = lowerer.construct(proposal, catalog, context)
    if isinstance(constructed, NotRepresentable):
        return constructed
    oracle_ok, oracle_message = lowerer.semantic_oracle(
        proposal, constructed, catalog, context
    )
    if not oracle_ok:
        return NotRepresentable(proposal, oracle_message)
    return LoweredCandidate(
        proposal,
        constructed.native_object,
        constructed.native_kind,
        oracle_message,
        _LOWERED_CANDIDATE_TOKEN,
    )


# Backward compatibility: lowerable returns (bool,str) for existing callers
def lowerable(proposal: PTAEscalationProposal) -> tuple[bool, str]:
    res = lower_exact(proposal)
    if isinstance(res, LoweredCandidate):
        return True, "ok"
    return False, res.reason


def check_example() -> PTAEscalationProposal:
    """Canonical example from docs/pta-control-plane.md:

    temperature 71–76 ∧ mode=manual ∧ previous=B  → 104∧105∧231∧388

    Note: literal IDs 104 etc. are illustrative magic IDs; exact gate without
    catalog will fail (NotRepresentable). With a catalog containing those IDs,
    lowering succeeds. For syntactically_bounded they pass.
    """
    from .proposal import PTAInsight

    return PTAEscalationProposal(
        proposal_id="pta-temp-manual-B-001",
        source_pta_ids=("input:temperature", "escalation:exception", "de-escalation:prune"),
        supporting_insights=(
            PTAInsight("input:temperature", "interval", "temperature", (71, 76)),
            PTAInsight("escalation:exception", "confusable_when", "mode=manual ∧ prev=B", ()),
        ),
        counterexamples_addressed=(17, 42),
        required_literals=("literal:104", "literal:105", "literal:231", "literal:388"),
        native_target="binary_clause",
        structure={"clause": [104, 105, 231, 388]},
        resource_bounds={"literal_count": 4, "clause_count": 1},
        validation_signature={"acc": "shadow_audit_pending"},
        support_trace=("unify numeric_region", "constraint interval 71..76"),
    )
