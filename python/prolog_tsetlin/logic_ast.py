"""Safe typed AST and primitive Boolean lowering for the Logic dataset."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence


LOGIC_AST_SCHEMA_VERSION = 1
PRIMITIVE_LOGIC_SCHEMA_VERSION = 1
LOGIC_AST_VARIABLES = ("A", "B", "C", "D", "E")


class LogicASTKind(str, Enum):
    VARIABLE = "variable"
    NOT = "not"
    AND = "and"
    OR = "or"
    NOT_EQUAL_CHAIN = "not_equal_chain"
    CONDITIONAL = "conditional"


class LogicChildRole(str, Enum):
    OPERAND = "operand"
    NEGATED = "negated"
    COMPARISON_OPERAND = "comparison_operand"
    CONDITION = "condition"
    TRUE_BRANCH = "true_branch"
    FALSE_BRANCH = "false_branch"


@dataclass(frozen=True, slots=True)
class LogicASTNode:
    node_id: int
    kind: LogicASTKind
    children: tuple[int, ...]
    child_roles: tuple[LogicChildRole, ...]
    parent_id: int | None
    parent_role: LogicChildRole | None
    depth: int
    variable: str | None = None

    def __post_init__(self) -> None:
        if self.node_id < 0 or self.depth < 0:
            raise ValueError("AST node identifier and depth cannot be negative")
        if len(self.children) != len(self.child_roles):
            raise ValueError("AST children and roles differ in length")
        if self.kind is LogicASTKind.VARIABLE:
            if self.variable not in LOGIC_AST_VARIABLES or self.children:
                raise ValueError("variable AST node is malformed")
        elif self.variable is not None:
            raise ValueError("only variable AST nodes may name a variable")


@dataclass(frozen=True, slots=True)
class LogicASTFact:
    predicate: str
    arguments: tuple[str | int | bool, ...]


class PrimitiveLogicOp(str, Enum):
    CONSTANT = "constant"
    INPUT = "input"
    NOT = "not"
    AND = "and"
    OR = "or"
    XOR = "xor"


@dataclass(frozen=True, slots=True)
class PrimitiveLogicNode:
    node_id: int
    operation: PrimitiveLogicOp
    operands: tuple[int, ...] = ()
    variable: str | None = None
    constant: bool | None = None


@dataclass(frozen=True, slots=True)
class PrimitiveLogicGraph:
    nodes: tuple[PrimitiveLogicNode, ...]
    root_id: int
    schema_version: int = PRIMITIVE_LOGIC_SCHEMA_VERSION

    def evaluate(self, bindings: Sequence[bool | int]) -> bool:
        if len(bindings) != len(LOGIC_AST_VARIABLES):
            raise ValueError("primitive graph requires bindings for A through E")
        binding_map = dict(zip(LOGIC_AST_VARIABLES, map(bool, bindings)))
        values: list[bool] = []
        for node in self.nodes:
            if node.operation is PrimitiveLogicOp.CONSTANT:
                value = bool(node.constant)
            elif node.operation is PrimitiveLogicOp.INPUT:
                assert node.variable is not None
                value = binding_map[node.variable]
            elif node.operation is PrimitiveLogicOp.NOT:
                value = not values[node.operands[0]]
            elif node.operation is PrimitiveLogicOp.AND:
                value = all(values[operand] for operand in node.operands)
            elif node.operation is PrimitiveLogicOp.OR:
                value = any(values[operand] for operand in node.operands)
            else:
                value = False
                for operand in node.operands:
                    value ^= values[operand]
            values.append(value)
        return values[self.root_id]


class _PrimitiveBuilder:
    def __init__(self) -> None:
        self.nodes: list[PrimitiveLogicNode] = []
        self._interned: dict[tuple[object, ...], int] = {}

    def _intern(
        self,
        operation: PrimitiveLogicOp,
        operands: Iterable[int] = (),
        *,
        variable: str | None = None,
        constant: bool | None = None,
    ) -> int:
        operand_tuple = tuple(operands)
        key = (operation.value, operand_tuple, variable, constant)
        existing = self._interned.get(key)
        if existing is not None:
            return existing
        node_id = len(self.nodes)
        self.nodes.append(
            PrimitiveLogicNode(
                node_id,
                operation,
                operand_tuple,
                variable=variable,
                constant=constant,
            )
        )
        self._interned[key] = node_id
        return node_id

    def constant(self, value: bool) -> int:
        return self._intern(PrimitiveLogicOp.CONSTANT, constant=value)

    def input(self, variable: str) -> int:
        return self._intern(PrimitiveLogicOp.INPUT, variable=variable)

    def negate(self, operand: int) -> int:
        node = self.nodes[operand]
        if node.operation is PrimitiveLogicOp.CONSTANT:
            return self.constant(not node.constant)
        if node.operation is PrimitiveLogicOp.NOT:
            return node.operands[0]
        return self._intern(PrimitiveLogicOp.NOT, (operand,))

    def all(self, source: Iterable[int]) -> int:
        operands: list[int] = []
        for operand in source:
            node = self.nodes[operand]
            if node.operation is PrimitiveLogicOp.CONSTANT:
                if not node.constant:
                    return self.constant(False)
                continue
            if node.operation is PrimitiveLogicOp.AND:
                operands.extend(node.operands)
            else:
                operands.append(operand)
        reduced = tuple(sorted(set(operands)))
        if not reduced:
            return self.constant(True)
        if len(reduced) == 1:
            return reduced[0]
        for operand in reduced:
            node = self.nodes[operand]
            if node.operation is PrimitiveLogicOp.NOT and node.operands[0] in reduced:
                return self.constant(False)
        return self._intern(PrimitiveLogicOp.AND, reduced)

    def any(self, source: Iterable[int]) -> int:
        operands: list[int] = []
        for operand in source:
            node = self.nodes[operand]
            if node.operation is PrimitiveLogicOp.CONSTANT:
                if node.constant:
                    return self.constant(True)
                continue
            if node.operation is PrimitiveLogicOp.OR:
                operands.extend(node.operands)
            else:
                operands.append(operand)
        reduced = tuple(sorted(set(operands)))
        if not reduced:
            return self.constant(False)
        if len(reduced) == 1:
            return reduced[0]
        for operand in reduced:
            node = self.nodes[operand]
            if node.operation is PrimitiveLogicOp.NOT and node.operands[0] in reduced:
                return self.constant(True)
        return self._intern(PrimitiveLogicOp.OR, reduced)

    def parity(self, source: Iterable[int]) -> int:
        odd: dict[int, bool] = {}
        pending = list(source)
        inverted = False
        while pending:
            operand = pending.pop()
            node = self.nodes[operand]
            if node.operation is PrimitiveLogicOp.CONSTANT:
                inverted ^= bool(node.constant)
            elif node.operation is PrimitiveLogicOp.XOR:
                pending.extend(node.operands)
            else:
                odd[operand] = not odd.get(operand, False)
        operands = tuple(sorted(operand for operand, present in odd.items() if present))
        if not operands:
            return self.constant(inverted)
        if len(operands) == 1:
            return self.negate(operands[0]) if inverted else operands[0]
        result = self._intern(PrimitiveLogicOp.XOR, operands)
        return self.negate(result) if inverted else result


@dataclass(frozen=True, slots=True)
class LogicASTProgram:
    nodes: tuple[LogicASTNode, ...]
    root_id: int
    source: str
    schema_version: int = LOGIC_AST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LOGIC_AST_SCHEMA_VERSION:
            raise ValueError("unsupported logic AST schema")
        if not self.nodes or not 0 <= self.root_id < len(self.nodes):
            raise ValueError("logic AST root is invalid")
        if tuple(node.node_id for node in self.nodes) != tuple(range(len(self.nodes))):
            raise ValueError("logic AST node identifiers must be dense")

    @property
    def maximum_depth(self) -> int:
        return max(node.depth for node in self.nodes)

    def evaluate(self, bindings: Sequence[bool | int]) -> bool:
        if len(bindings) != len(LOGIC_AST_VARIABLES):
            raise ValueError("logic AST requires bindings for A through E")
        binding_map = dict(zip(LOGIC_AST_VARIABLES, map(bool, bindings)))

        def visit(node_id: int) -> bool:
            node = self.nodes[node_id]
            if node.kind is LogicASTKind.VARIABLE:
                assert node.variable is not None
                return binding_map[node.variable]
            children = tuple(visit(child) for child in node.children)
            if node.kind is LogicASTKind.NOT:
                return not children[0]
            if node.kind is LogicASTKind.AND:
                return all(children)
            if node.kind is LogicASTKind.OR:
                return any(children)
            if node.kind is LogicASTKind.NOT_EQUAL_CHAIN:
                return all(left != right for left, right in zip(children, children[1:]))
            condition, true_branch, false_branch = children
            return true_branch if condition else false_branch

        return visit(self.root_id)

    def facts(self, bindings: Sequence[bool | int]) -> tuple[LogicASTFact, ...]:
        if len(bindings) != len(LOGIC_AST_VARIABLES):
            raise ValueError("logic AST facts require bindings for A through E")
        result = [LogicASTFact("root", (self.root_id,))]
        for node in self.nodes:
            result.append(LogicASTFact("operator", (node.node_id, node.kind.value)))
            result.append(LogicASTFact("depth", (node.node_id, node.depth)))
            if node.variable is not None:
                result.append(LogicASTFact("references", (node.node_id, node.variable)))
            for child, role in zip(node.children, node.child_roles):
                result.append(
                    LogicASTFact("child", (node.node_id, role.value, child))
                )
        for variable, value in zip(LOGIC_AST_VARIABLES, bindings):
            result.append(LogicASTFact("bound_value", (variable, bool(value))))
        return tuple(result)

    def lower(self) -> PrimitiveLogicGraph:
        builder = _PrimitiveBuilder()

        def visit(node_id: int) -> int:
            node = self.nodes[node_id]
            if node.kind is LogicASTKind.VARIABLE:
                assert node.variable is not None
                return builder.input(node.variable)
            children = tuple(visit(child) for child in node.children)
            if node.kind is LogicASTKind.NOT:
                return builder.negate(children[0])
            if node.kind is LogicASTKind.AND:
                return builder.all(children)
            if node.kind is LogicASTKind.OR:
                return builder.any(children)
            if node.kind is LogicASTKind.NOT_EQUAL_CHAIN:
                comparisons = (
                    builder.parity((left, right))
                    for left, right in zip(children, children[1:])
                )
                return builder.all(comparisons)
            condition, true_branch, false_branch = children
            return builder.any(
                (
                    builder.all((condition, true_branch)),
                    builder.all((builder.negate(condition), false_branch)),
                )
            )

        root = visit(self.root_id)
        return PrimitiveLogicGraph(tuple(builder.nodes), root)


@dataclass(slots=True)
class _TreeNode:
    kind: LogicASTKind
    children: list[tuple[LogicChildRole, "_TreeNode"]]
    variable: str | None = None


def _convert_python_ast(node: ast.expr) -> _TreeNode:
    if isinstance(node, ast.Name):
        if node.id not in LOGIC_AST_VARIABLES:
            raise ValueError(f"unknown logic variable: {node.id}")
        return _TreeNode(LogicASTKind.VARIABLE, [], variable=node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _TreeNode(
            LogicASTKind.NOT,
            [(LogicChildRole.NEGATED, _convert_python_ast(node.operand))],
        )
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        kind = LogicASTKind.AND if isinstance(node.op, ast.And) else LogicASTKind.OR
        return _TreeNode(
            kind,
            [
                (LogicChildRole.OPERAND, _convert_python_ast(child))
                for child in node.values
            ],
        )
    if isinstance(node, ast.Compare):
        if not node.ops or not all(isinstance(op, ast.NotEq) for op in node.ops):
            raise ValueError("only != comparisons are allowed in Logic expressions")
        operands = (node.left, *node.comparators)
        return _TreeNode(
            LogicASTKind.NOT_EQUAL_CHAIN,
            [
                (LogicChildRole.COMPARISON_OPERAND, _convert_python_ast(child))
                for child in operands
            ],
        )
    if isinstance(node, ast.IfExp):
        return _TreeNode(
            LogicASTKind.CONDITIONAL,
            [
                (LogicChildRole.CONDITION, _convert_python_ast(node.test)),
                (LogicChildRole.TRUE_BRANCH, _convert_python_ast(node.body)),
                (LogicChildRole.FALSE_BRANCH, _convert_python_ast(node.orelse)),
            ],
        )
    raise ValueError(f"unsupported Logic syntax node: {type(node).__name__}")


def parse_logic_tokens(tokens: Sequence[str]) -> LogicASTProgram:
    if not tokens:
        raise ValueError("cannot parse an empty Logic expression")
    translations = {"&": "and", "x": "or", "-": "not", "$": "else"}
    source = " ".join(translations.get(token, token) for token in tokens)
    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"invalid Logic expression: {source}") from error
    tree = _convert_python_ast(parsed.body)

    nodes: list[LogicASTNode | None] = []

    def flatten(
        current: _TreeNode,
        parent_id: int | None,
        parent_role: LogicChildRole | None,
        depth: int,
    ) -> int:
        node_id = len(nodes)
        nodes.append(None)
        children: list[int] = []
        roles: list[LogicChildRole] = []
        for role, child in current.children:
            roles.append(role)
            children.append(flatten(child, node_id, role, depth + 1))
        nodes[node_id] = LogicASTNode(
            node_id=node_id,
            kind=current.kind,
            children=tuple(children),
            child_roles=tuple(roles),
            parent_id=parent_id,
            parent_role=parent_role,
            depth=depth,
            variable=current.variable,
        )
        return node_id

    root = flatten(tree, None, None, 0)
    if any(node is None for node in nodes):
        raise RuntimeError("logic AST flattening left an uninitialized node")
    return LogicASTProgram(tuple(node for node in nodes if node is not None), root, source)
