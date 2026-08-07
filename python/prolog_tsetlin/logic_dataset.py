"""Class I connector and bounded encodings for evaluated logic programs."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

from .logic_ast import LogicASTKind, LogicASTProgram, parse_logic_tokens


LOGIC_DATASET_SCHEMA_VERSION = 1
LOGIC_TOKEN_CATALOG_VERSION = 1
LOGIC_VARIABLES = ("A", "B", "C", "D", "E")
LOGIC_TOKENS = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "&",
    "x",
    "-",
    "!=",
    "if",
    "$",
    "(",
    ")",
)

_TOKEN_PATTERN = re.compile(r"!=|if|[A-E&x$()\-]")
_BINDING_PATTERN = re.compile(r"'([A-E])':([01])")


class LogicEncoding(str, Enum):
    TOKEN_PRESENCE = "token_presence"
    TOKEN_COUNT_THRESHOLD = "token_count_threshold"
    POSITION_ONE_HOT = "position_one_hot"
    AST_RELATIONAL = "ast_relational"


class LogicLiteralKind(str, Enum):
    TOKEN_PRESENT = "token_present"
    TOKEN_COUNT_AT_LEAST = "token_count_at_least"
    TOKEN_AT_POSITION = "token_at_position"
    BINDING_VALUE = "binding_value"
    AST_KIND_COUNT_AT_LEAST = "ast_kind_count_at_least"
    AST_KIND_AT_DEPTH = "ast_kind_at_depth"
    AST_EDGE = "ast_edge"
    AST_PATH_TWO = "ast_path_two"


@dataclass(frozen=True, slots=True)
class LogicProblem:
    row_id: int
    natural_problem: str
    symbolic_source: str
    expression_tokens: tuple[str, ...]
    bindings: tuple[bool, ...]
    target: int
    syntax_tree: LogicASTProgram | None = None

    def __post_init__(self) -> None:
        if self.row_id < 0:
            raise ValueError("logic row identifier cannot be negative")
        if not self.expression_tokens:
            raise ValueError("logic expression cannot be empty")
        if any(token not in LOGIC_TOKENS for token in self.expression_tokens):
            raise ValueError("logic expression contains a token outside the catalog")
        if len(self.bindings) != len(LOGIC_VARIABLES):
            raise ValueError("logic problem must bind A through E")
        if self.target not in (0, 1):
            raise ValueError("logic target must be zero or one")
        if self.syntax_tree is None:
            object.__setattr__(
                self, "syntax_tree", parse_logic_tokens(self.expression_tokens)
            )


@dataclass(frozen=True, slots=True)
class LogicDataset:
    problems: tuple[LogicProblem, ...]
    source_digest: str
    schema_version: int = LOGIC_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LOGIC_DATASET_SCHEMA_VERSION:
            raise ValueError("unsupported logic dataset schema")
        if not self.problems:
            raise ValueError("logic dataset cannot be empty")
        if not self.source_digest.startswith("sha256:"):
            raise ValueError("logic dataset digest must use SHA-256")


@dataclass(frozen=True, slots=True)
class LogicSplit:
    train_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]
    seed: int
    evaluation_fraction: float


@dataclass(frozen=True, slots=True)
class LogicLiteralDescriptor:
    feature_index: int
    kind: LogicLiteralKind
    token: str | None = None
    count_at_least: int | None = None
    position: int | None = None
    depth: int | None = None
    variable: str | None = None
    relation: tuple[str, ...] | None = None

    @property
    def literal_id(self) -> int:
        payload = json.dumps(
            self.canonical_payload(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "catalog_version": LOGIC_TOKEN_CATALOG_VERSION,
            "kind": self.kind.value,
            "token": self.token,
            "count_at_least": self.count_at_least,
            "position": self.position,
            "depth": self.depth,
            "variable": self.variable,
            "relation": self.relation,
        }

    @property
    def name(self) -> str:
        if self.kind is LogicLiteralKind.TOKEN_PRESENT:
            return f"token_present:{self.token}"
        if self.kind is LogicLiteralKind.TOKEN_COUNT_AT_LEAST:
            return f"token_count:{self.token}>={self.count_at_least}"
        if self.kind is LogicLiteralKind.TOKEN_AT_POSITION:
            return f"token_at:{self.position}:{self.token}"
        if self.kind is LogicLiteralKind.AST_KIND_COUNT_AT_LEAST:
            return f"ast_count:{self.relation[0]}>={self.count_at_least}"
        if self.kind is LogicLiteralKind.AST_KIND_AT_DEPTH:
            return f"ast_depth:{self.depth}:{self.relation[0]}"
        if self.kind is LogicLiteralKind.AST_EDGE:
            return "ast_edge:" + ":".join(self.relation or ())
        if self.kind is LogicLiteralKind.AST_PATH_TWO:
            return "ast_path_two:" + ":".join(self.relation or ())
        return f"binding:{self.variable}=1"


@dataclass(frozen=True, slots=True)
class EncodedLogicRows:
    row_ids: tuple[int, ...]
    rows: tuple[tuple[int, ...], ...]
    targets: tuple[int, ...]
    truncated_rows: int = 0

    def __post_init__(self) -> None:
        if not (len(self.row_ids) == len(self.rows) == len(self.targets)):
            raise ValueError("encoded logic row arrays differ in length")
        widths = {len(row) for row in self.rows}
        if len(widths) > 1:
            raise ValueError("encoded logic rows differ in width")


@dataclass(frozen=True, slots=True)
class EncodedLogicSplit:
    encoding: LogicEncoding
    literals: tuple[LogicLiteralDescriptor, ...]
    train: EncodedLogicRows
    evaluation: EncodedLogicRows
    token_catalog_version: int = LOGIC_TOKEN_CATALOG_VERSION

    @property
    def feature_count(self) -> int:
        return len(self.literals)


@dataclass(frozen=True, slots=True)
class CollisionReport:
    row_count: int
    unique_signatures: int
    conflicting_signatures: int
    rows_in_conflicting_signatures: int
    optimistic_correct: int

    @property
    def optimistic_ceiling(self) -> float:
        return self.optimistic_correct / self.row_count


@dataclass(frozen=True, slots=True)
class EvaluationSignatureReport:
    evaluation_rows: int
    seen_rows: int
    unseen_rows: int
    conflicting_seen_rows: int
    majority_disagreement_rows: int
    lookup_correct_with_majority_fallback: int

    @property
    def seen_fraction(self) -> float:
        return self.seen_rows / self.evaluation_rows

    @property
    def lookup_accuracy(self) -> float:
        return self.lookup_correct_with_majority_fallback / self.evaluation_rows


def _parse_symbolic_source(source: str) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    expression_source, separator, binding_source = source.partition("/")
    if not separator:
        raise ValueError("symbolic logic row is missing its binding separator")
    if len(expression_source) < 2 or not (
        expression_source.startswith("'") and expression_source.endswith("'")
    ):
        raise ValueError("symbolic expression must be wrapped in single quotes")
    expression = expression_source[1:-1]
    tokens = tuple(_TOKEN_PATTERN.findall(expression))
    if "".join(tokens) != expression:
        raise ValueError(f"unrecognized symbolic expression fragment: {expression!r}")

    bindings_found = _BINDING_PATTERN.findall(binding_source)
    if tuple(variable for variable, _ in bindings_found) != LOGIC_VARIABLES:
        raise ValueError("symbolic bindings must contain ordered A through E")
    bindings = tuple(value == "1" for _, value in bindings_found)
    return tokens, bindings


def load_logic_dataset(
    natural_path: str | Path,
    symbolic_path: str | Path,
) -> LogicDataset:
    natural_file = Path(natural_path)
    symbolic_file = Path(symbolic_path)
    with natural_file.open(encoding="utf-8-sig", newline="") as stream:
        natural_rows = list(csv.DictReader(stream))
    if not natural_rows or set(natural_rows[0]) != {"Problem", "Solution"}:
        raise ValueError("natural logic CSV must contain Problem and Solution columns")

    symbolic_rows: list[tuple[str, int]] = []
    with symbolic_file.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if len(row) != 2 or row[1] not in ("0", "1"):
                raise ValueError("symbolic logic CSV rows must contain source and bit")
            symbolic_rows.append((row[0], int(row[1])))
    if len(natural_rows) != len(symbolic_rows):
        raise ValueError("natural and symbolic logic CSV row counts differ")

    problems: list[LogicProblem] = []
    for row_id, (natural, symbolic) in enumerate(zip(natural_rows, symbolic_rows)):
        natural_target_text = natural["Solution"].strip().lower()
        if natural_target_text not in ("true", "false"):
            raise ValueError("natural logic solution must be True or False")
        natural_target = int(natural_target_text == "true")
        symbolic_source, symbolic_target = symbolic
        if natural_target != symbolic_target:
            raise ValueError(f"paired logic labels differ at row {row_id}")
        tokens, bindings = _parse_symbolic_source(symbolic_source)
        problem = LogicProblem(
            row_id=row_id,
            natural_problem=natural["Problem"],
            symbolic_source=symbolic_source,
            expression_tokens=tokens,
            bindings=bindings,
            target=symbolic_target,
        )
        assert problem.syntax_tree is not None
        ast_value = problem.syntax_tree.evaluate(problem.bindings)
        primitive_value = problem.syntax_tree.lower().evaluate(problem.bindings)
        if ast_value != bool(problem.target) or primitive_value != bool(problem.target):
            raise ValueError(f"logic evaluator disagrees with label at row {row_id}")
        problems.append(problem)

    digest = hashlib.sha256()
    for path in (natural_file, symbolic_file):
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return LogicDataset(tuple(problems), f"sha256:{digest.hexdigest()}")


def stratified_logic_split(
    dataset: LogicDataset,
    *,
    evaluation_fraction: float = 0.2,
    seed: int = 20260806,
) -> LogicSplit:
    if not 0.0 < evaluation_fraction < 1.0:
        raise ValueError("evaluation fraction must lie strictly between zero and one")
    by_target: dict[int, list[int]] = {0: [], 1: []}
    for index, problem in enumerate(dataset.problems):
        by_target[problem.target].append(index)
    if any(len(indices) < 2 for indices in by_target.values()):
        raise ValueError("stratified split requires at least two rows per class")

    rng = random.Random(seed)
    train: list[int] = []
    evaluation: list[int] = []
    for indices in by_target.values():
        rng.shuffle(indices)
        evaluation_count = round(len(indices) * evaluation_fraction)
        evaluation_count = max(1, min(len(indices) - 1, evaluation_count))
        evaluation.extend(indices[:evaluation_count])
        train.extend(indices[evaluation_count:])
    rng.shuffle(train)
    rng.shuffle(evaluation)
    return LogicSplit(tuple(train), tuple(evaluation), seed, evaluation_fraction)


def _ast_relations(
    problem: LogicProblem,
) -> tuple[
    Counter[str],
    set[tuple[str, int]],
    set[tuple[str, str, str]],
    set[tuple[str, str, str, str, str]],
]:
    tree = problem.syntax_tree
    assert tree is not None
    counts: Counter[str] = Counter()
    depths: set[tuple[str, int]] = set()
    edges: set[tuple[str, str, str]] = set()
    paths: set[tuple[str, str, str, str, str]] = set()

    def signature(node_id: int) -> str:
        node = tree.nodes[node_id]
        return node.variable if node.variable is not None else node.kind.value

    for node in tree.nodes:
        counts[node.kind.value] += 1
        depths.add((node.kind.value, node.depth))
        for child_id, role in zip(node.children, node.child_roles):
            child = tree.nodes[child_id]
            edges.add((node.kind.value, role.value, signature(child_id)))
            for grandchild_id, child_role in zip(
                child.children, child.child_roles
            ):
                paths.add(
                    (
                        node.kind.value,
                        role.value,
                        child.kind.value,
                        child_role.value,
                        signature(grandchild_id),
                    )
                )
    return counts, depths, edges, paths


class LogicEncoder:
    def __init__(
        self,
        encoding: LogicEncoding,
        literals: Sequence[LogicLiteralDescriptor],
        *,
        maximum_position: int = 0,
        maximum_counts: Sequence[int] = (),
        ast_maximum_counts: Sequence[tuple[str, int]] = (),
        ast_depths: Iterable[tuple[str, int]] = (),
        ast_edges: Iterable[tuple[str, str, str]] = (),
        ast_paths: Iterable[tuple[str, str, str, str, str]] = (),
    ) -> None:
        self.encoding = encoding
        self.literals = tuple(literals)
        self.maximum_position = maximum_position
        self.maximum_counts = tuple(maximum_counts)
        self.ast_maximum_counts = dict(ast_maximum_counts)
        self.ast_depths = frozenset(ast_depths)
        self.ast_edges = frozenset(ast_edges)
        self.ast_paths = frozenset(ast_paths)
        if not self.literals:
            raise ValueError("logic encoder must allocate at least one literal")

    @classmethod
    def fit(
        cls,
        encoding: LogicEncoding,
        problems: Iterable[LogicProblem],
    ) -> "LogicEncoder":
        training = tuple(problems)
        if not training:
            raise ValueError("cannot fit logic encoder without training rows")
        literals: list[LogicLiteralDescriptor] = []

        def append(kind: LogicLiteralKind, **parameters: object) -> None:
            literals.append(
                LogicLiteralDescriptor(
                    feature_index=len(literals), kind=kind, **parameters
                )
            )

        maximum_position = 0
        fitted_maximum_counts: tuple[int, ...] = ()
        fitted_ast_maximum_counts: tuple[tuple[str, int], ...] = ()
        fitted_ast_depths: set[tuple[str, int]] = set()
        fitted_ast_edges: set[tuple[str, str, str]] = set()
        fitted_ast_paths: set[tuple[str, str, str, str, str]] = set()
        if encoding is LogicEncoding.TOKEN_PRESENCE:
            for token in LOGIC_TOKENS:
                append(LogicLiteralKind.TOKEN_PRESENT, token=token)
        elif encoding is LogicEncoding.TOKEN_COUNT_THRESHOLD:
            maximum_counts = Counter()
            for problem in training:
                counts = Counter(problem.expression_tokens)
                for token in LOGIC_TOKENS:
                    maximum_counts[token] = max(maximum_counts[token], counts[token])
            for token in LOGIC_TOKENS:
                for threshold in range(1, maximum_counts[token] + 1):
                    append(
                        LogicLiteralKind.TOKEN_COUNT_AT_LEAST,
                        token=token,
                        count_at_least=threshold,
                    )
            fitted_maximum_counts = tuple(
                maximum_counts[token] for token in LOGIC_TOKENS
            )
        elif encoding is LogicEncoding.POSITION_ONE_HOT:
            maximum_position = max(len(problem.expression_tokens) for problem in training)
            for position in range(maximum_position):
                for token in LOGIC_TOKENS:
                    append(
                        LogicLiteralKind.TOKEN_AT_POSITION,
                        token=token,
                        position=position,
                    )
        elif encoding is LogicEncoding.AST_RELATIONAL:
            ast_maximum_counts: Counter[str] = Counter()
            for problem in training:
                counts, depths, edges, paths = _ast_relations(problem)
                for kind in LogicASTKind:
                    ast_maximum_counts[kind.value] = max(
                        ast_maximum_counts[kind.value], counts[kind.value]
                    )
                fitted_ast_depths.update(depths)
                fitted_ast_edges.update(edges)
                fitted_ast_paths.update(paths)
            for kind in LogicASTKind:
                for threshold in range(1, ast_maximum_counts[kind.value] + 1):
                    append(
                        LogicLiteralKind.AST_KIND_COUNT_AT_LEAST,
                        relation=(kind.value,),
                        count_at_least=threshold,
                    )
            for kind, depth in sorted(fitted_ast_depths):
                append(
                    LogicLiteralKind.AST_KIND_AT_DEPTH,
                    relation=(kind,),
                    depth=depth,
                )
            for edge in sorted(fitted_ast_edges):
                append(LogicLiteralKind.AST_EDGE, relation=edge)
            for path in sorted(fitted_ast_paths):
                append(LogicLiteralKind.AST_PATH_TWO, relation=path)
            fitted_ast_maximum_counts = tuple(
                (kind.value, ast_maximum_counts[kind.value])
                for kind in LogicASTKind
            )
        else:
            raise ValueError(f"unsupported logic encoding: {encoding}")

        for variable in LOGIC_VARIABLES:
            append(LogicLiteralKind.BINDING_VALUE, variable=variable)
        return cls(
            encoding,
            literals,
            maximum_position=maximum_position,
            maximum_counts=fitted_maximum_counts,
            ast_maximum_counts=fitted_ast_maximum_counts,
            ast_depths=fitted_ast_depths,
            ast_edges=fitted_ast_edges,
            ast_paths=fitted_ast_paths,
        )

    def encode_problem(self, problem: LogicProblem) -> tuple[tuple[int, ...], bool]:
        counts = Counter(problem.expression_tokens)
        present = set(problem.expression_tokens)
        values: list[int] = []
        was_truncated = False
        if self.encoding is LogicEncoding.TOKEN_PRESENCE:
            values.extend(int(token in present) for token in LOGIC_TOKENS)
        elif self.encoding is LogicEncoding.TOKEN_COUNT_THRESHOLD:
            for literal in self.literals:
                if literal.kind is not LogicLiteralKind.TOKEN_COUNT_AT_LEAST:
                    break
                values.append(int(counts[literal.token] >= literal.count_at_least))
            was_truncated = any(
                counts[token] > maximum
                for token, maximum in zip(LOGIC_TOKENS, self.maximum_counts)
            )
        elif self.encoding is LogicEncoding.POSITION_ONE_HOT:
            for position in range(self.maximum_position):
                actual = (
                    problem.expression_tokens[position]
                    if position < len(problem.expression_tokens)
                    else None
                )
                values.extend(int(actual == token) for token in LOGIC_TOKENS)
            was_truncated = len(problem.expression_tokens) > self.maximum_position
        else:
            ast_counts, ast_depths, ast_edges, ast_paths = _ast_relations(problem)
            for literal in self.literals:
                if literal.kind is LogicLiteralKind.BINDING_VALUE:
                    break
                if literal.kind is LogicLiteralKind.AST_KIND_COUNT_AT_LEAST:
                    values.append(
                        int(
                            ast_counts[literal.relation[0]]
                            >= literal.count_at_least
                        )
                    )
                elif literal.kind is LogicLiteralKind.AST_KIND_AT_DEPTH:
                    values.append(
                        int((literal.relation[0], literal.depth) in ast_depths)
                    )
                elif literal.kind is LogicLiteralKind.AST_EDGE:
                    values.append(int(tuple(literal.relation) in ast_edges))
                else:
                    values.append(int(tuple(literal.relation) in ast_paths))
            was_truncated = (
                any(
                    ast_counts[kind] > maximum
                    for kind, maximum in self.ast_maximum_counts.items()
                )
                or not ast_depths <= self.ast_depths
                or not ast_edges <= self.ast_edges
                or not ast_paths <= self.ast_paths
            )
        values.extend(int(value) for value in problem.bindings)
        if len(values) != len(self.literals):
            raise RuntimeError("logic encoder produced the wrong feature width")
        return tuple(values), was_truncated

    def encode(
        self,
        dataset: LogicDataset,
        indices: Sequence[int],
    ) -> EncodedLogicRows:
        rows: list[tuple[int, ...]] = []
        targets: list[int] = []
        truncated = 0
        for index in indices:
            problem = dataset.problems[index]
            row, was_truncated = self.encode_problem(problem)
            rows.append(row)
            targets.append(problem.target)
            truncated += was_truncated
        return EncodedLogicRows(
            tuple(dataset.problems[index].row_id for index in indices),
            tuple(rows),
            tuple(targets),
            truncated,
        )


def encode_logic_split(
    dataset: LogicDataset,
    split: LogicSplit,
    encoding: LogicEncoding,
) -> EncodedLogicSplit:
    encoder = LogicEncoder.fit(
        encoding, (dataset.problems[index] for index in split.train_indices)
    )
    return EncodedLogicSplit(
        encoding=encoding,
        literals=encoder.literals,
        train=encoder.encode(dataset, split.train_indices),
        evaluation=encoder.encode(dataset, split.evaluation_indices),
    )


def collision_report(
    rows: Sequence[Sequence[int]],
    targets: Sequence[int],
) -> CollisionReport:
    if len(rows) != len(targets) or not rows:
        raise ValueError("collision rows and targets must be non-empty and aligned")
    groups: dict[tuple[int, ...], list[int]] = defaultdict(lambda: [0, 0])
    for row, target in zip(rows, targets):
        groups[tuple(row)][int(target)] += 1
    conflicting = [counts for counts in groups.values() if min(counts) != 0]
    return CollisionReport(
        row_count=len(rows),
        unique_signatures=len(groups),
        conflicting_signatures=len(conflicting),
        rows_in_conflicting_signatures=sum(sum(counts) for counts in conflicting),
        optimistic_correct=sum(max(counts) for counts in groups.values()),
    )


def evaluation_signature_report(
    train_rows: Sequence[Sequence[int]],
    train_targets: Sequence[int],
    evaluation_rows: Sequence[Sequence[int]],
    evaluation_targets: Sequence[int],
) -> EvaluationSignatureReport:
    if len(train_rows) != len(train_targets) or not train_rows:
        raise ValueError("training signatures must be non-empty and aligned")
    if len(evaluation_rows) != len(evaluation_targets) or not evaluation_rows:
        raise ValueError("evaluation signatures must be non-empty and aligned")
    groups: dict[tuple[int, ...], list[int]] = defaultdict(lambda: [0, 0])
    global_counts = [0, 0]
    for row, target in zip(train_rows, train_targets):
        groups[tuple(row)][int(target)] += 1
        global_counts[int(target)] += 1
    fallback = int(global_counts[1] > global_counts[0])

    seen = 0
    conflicting = 0
    disagreement = 0
    correct = 0
    for row, target in zip(evaluation_rows, evaluation_targets):
        counts = groups.get(tuple(row))
        if counts is None:
            prediction = fallback
        else:
            seen += 1
            conflicting += min(counts) != 0
            prediction = int(counts[1] > counts[0])
            disagreement += prediction != int(target)
        correct += prediction == int(target)
    return EvaluationSignatureReport(
        evaluation_rows=len(evaluation_rows),
        seen_rows=seen,
        unseen_rows=len(evaluation_rows) - seen,
        conflicting_seen_rows=conflicting,
        majority_disagreement_rows=disagreement,
        lookup_correct_with_majority_fallback=correct,
    )
