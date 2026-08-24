"""Bounded shared knowledge-base state for the PTA collective.

Python owns validation and safe fact serialization. GNU Prolog owns symbolic
derivation over those facts through :mod:`prolog_tsetlin.pta.collective`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

from .proposal import PTAEscalationProposal, PTAInsight

if TYPE_CHECKING:
    from .collective import PTACollectiveResult


MAX_RELATION_ITEMS = 100_000
MAX_TEXT_CHARS = 1_024
MAX_TERM_DEPTH = 8
MAX_TERM_NODES = 1_024
MAX_UNSIGNED_ID = (1 << 64) - 1
GPROLOG_MAX_INTEGER = (1 << 60) - 1
GPROLOG_MIN_INTEGER = -(1 << 60)
EXACT_NUMERIC_MAGNITUDE = 1 << 52
MAX_COTM_SCORE_MAGNITUDE = 250_000


def _printable(value: Any, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or len(value) > MAX_TEXT_CHARS or any(ord(char) < 0x20 for char in value):
        raise ValueError(
            f"{name} must be a nonempty printable string of at most "
            f"{MAX_TEXT_CHARS} characters"
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if value > MAX_UNSIGNED_ID:
        raise ValueError(f"{name} exceeds the unsigned 64-bit boundary")
    return value


def _binary(value: Any, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")
    return value


def _portable_nonnegative_int(value: Any, name: str) -> int:
    _nonnegative_int(value, name)
    if value > GPROLOG_MAX_INTEGER:
        raise ValueError(f"{name} exceeds GNU Prolog's portable integer range")
    return value


def _finite_number(value: Any, name: str) -> int | float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an integer or float")
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not GPROLOG_MIN_INTEGER <= value <= GPROLOG_MAX_INTEGER:
        raise ValueError(f"{name} exceeds GNU Prolog's portable numeric range")
    return value


def _prolog_atom(value: str) -> str:
    """Encode a validated Python string as a single-quoted Prolog atom."""

    if type(value) is not str:
        raise TypeError("atom must be a string")
    parts = ["'"]
    for char in value:
        if char == "'":
            parts.append("''")
        elif char == "\\":
            parts.append("\\\\")
        elif char == "\n":
            parts.append("\\n")
        elif char == "\r":
            parts.append("\\r")
        elif char == "\t":
            parts.append("\\t")
        elif ord(char) < 0x20:
            parts.append(f"\\x{ord(char):02x}\\")
        else:
            parts.append(char)
    parts.append("'")
    return "".join(parts)


def _prolog_term(value: Any) -> str:
    """Encode a value in PTM's bounded data-only Prolog term grammar."""

    nodes = 0

    def encode(item: Any, depth: int) -> str:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_TERM_NODES:
            raise ValueError("Prolog term node budget exceeded")
        if depth > MAX_TERM_DEPTH:
            raise ValueError("Prolog term depth budget exceeded")
        if item is None:
            return "null"
        if type(item) is bool:
            return "true" if item else "false"
        if type(item) is int:
            if 0 <= item <= MAX_UNSIGNED_ID and item > GPROLOG_MAX_INTEGER:
                # Preserve semantic 64-bit IDs as tagged data. Arithmetic PTA
                # inputs are independently range-checked before execution.
                return f"uint64({_prolog_atom(str(item))})"
            if not GPROLOG_MIN_INTEGER <= item <= GPROLOG_MAX_INTEGER:
                raise ValueError("integer exceeds GNU Prolog's portable range")
            return str(item)
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("non-finite float not allowed in Prolog term")
            if abs(item) > EXACT_NUMERIC_MAGNITUDE:
                raise ValueError("float exceeds PTA's exact numeric magnitude")
            return repr(item)
        if type(item) is str:
            if len(item) > MAX_TEXT_CHARS:
                raise ValueError("Prolog term string budget exceeded")
            return _prolog_atom(item)
        if isinstance(item, (list, tuple)):
            return "[" + ",".join(encode(child, depth + 1) for child in item) + "]"
        if isinstance(item, Mapping):
            keys = list(item)
            if any(type(key) is not str for key in keys):
                raise TypeError("Prolog term mapping keys must be strings")
            if any(len(key) > MAX_TEXT_CHARS for key in keys):
                raise ValueError("Prolog term mapping key budget exceeded")
            pairs: list[str] = []
            for key in sorted(keys):
                pairs.append(f"{_prolog_atom(key)}-{encode(item[key], depth + 1)}")
            return "[" + ",".join(pairs) + "]"
        raise TypeError(f"unsupported Prolog term type: {type(item).__name__}")

    return encode(value, 0)


def _validate_budget(value: Any, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= MAX_RELATION_ITEMS:
        raise ValueError(f"{name} must be in 1..{MAX_RELATION_ITEMS}")
    return value


@dataclass
class PTAReasoningSession:
    """A bounded, serializable snapshot of facts shared by the three PTAs."""

    dataset_id: str
    generation: int = 0
    max_observations: int = 1_024
    max_example_labels: int = 1_024
    max_example_domains: int = 1_024
    max_feature_supports: int = 4_096
    max_feature_relations: int = 4_096
    max_literal_truths: int = 65_536
    max_clause_truths: int = 65_536
    max_clause_literals: int = 65_536
    max_class_supports: int = 4_096
    max_clause_class_scores: int = 65_536
    max_clause_supports: int = 65_536
    max_clause_conflicts: int = 65_536
    max_counterexamples: int = 1_024
    max_insights: int = 256
    max_proposals: int = 128
    observations: list[tuple[str, int, str, Any]] = field(default_factory=list)
    example_labels: list[tuple[int, int]] = field(default_factory=list)
    example_domains: set[int] = field(default_factory=set)
    feature_supports: list[tuple[int, int, int]] = field(default_factory=list)
    feature_relations: list[tuple[int, str, int]] = field(default_factory=list)
    literal_truths: list[tuple[int, int, int]] = field(default_factory=list)
    clause_truths: list[tuple[int, int, int]] = field(default_factory=list)
    clause_literals: list[tuple[int, int]] = field(default_factory=list)
    class_supports: list[tuple[int, int, int]] = field(default_factory=list)
    clause_class_scores: list[tuple[int, int, int | float]] = field(default_factory=list)
    clause_supports: list[tuple[int, int]] = field(default_factory=list)
    clause_conflicts: list[tuple[int, int]] = field(default_factory=list)
    counterexamples: list[tuple[str, int, int, int]] = field(default_factory=list)
    insights: list[PTAInsight] = field(default_factory=list)
    proposals: list[PTAEscalationProposal] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    @staticmethod
    def _budget_names() -> tuple[str, ...]:
        return (
            "max_observations",
            "max_example_labels",
            "max_example_domains",
            "max_feature_supports",
            "max_feature_relations",
            "max_literal_truths",
            "max_clause_truths",
            "max_clause_literals",
            "max_class_supports",
            "max_clause_class_scores",
            "max_clause_supports",
            "max_clause_conflicts",
            "max_counterexamples",
            "max_insights",
            "max_proposals",
        )

    @staticmethod
    def _require_room(collection: Sequence[Any] | set[Any], limit: int, name: str) -> None:
        if len(collection) >= limit:
            raise ValueError(f"{name} budget exceeded")

    def add_observation(self, pta: str, example: int, field: str, raw_value: Any) -> None:
        _printable(pta, "pta")
        _nonnegative_int(example, "example")
        _printable(field, "field")
        _prolog_term(raw_value)
        self._require_room(self.observations, self.max_observations, "observation")
        self.observations.append((pta, example, field, raw_value))

    def add_example_label(self, example: int, label: int) -> None:
        _nonnegative_int(example, "example")
        _binary(label, "label")
        self._require_room(self.example_labels, self.max_example_labels, "example_label")
        if example not in self.example_domains:
            self._require_room(
                self.example_domains, self.max_example_domains, "example_domain"
            )
        self.example_labels.append((example, label))
        self.example_domains.add(example)

    def add_example_domain(self, example: int) -> None:
        _nonnegative_int(example, "example")
        if example not in self.example_domains:
            self._require_room(
                self.example_domains, self.max_example_domains, "example_domain"
            )
        self.example_domains.add(example)

    def add_feature_support(self, literal: int, positive: int, negative: int) -> None:
        _nonnegative_int(literal, "literal")
        _portable_nonnegative_int(positive, "positive")
        _portable_nonnegative_int(negative, "negative")
        self._require_room(
            self.feature_supports, self.max_feature_supports, "feature_support"
        )
        self.feature_supports.append((literal, positive, negative))

    def add_feature_relation(self, left: int, relation: str, right: int) -> None:
        _nonnegative_int(left, "left literal")
        _printable(relation, "relation")
        _nonnegative_int(right, "right literal")
        self._require_room(
            self.feature_relations, self.max_feature_relations, "feature_relation"
        )
        self.feature_relations.append((left, relation, right))

    def add_literal_truth(self, literal: int, example: int, truth: int) -> None:
        _nonnegative_int(literal, "literal")
        _nonnegative_int(example, "example")
        _binary(truth, "truth")
        self._require_room(self.literal_truths, self.max_literal_truths, "literal_truth")
        if example not in self.example_domains:
            self._require_room(
                self.example_domains, self.max_example_domains, "example_domain"
            )
        self.literal_truths.append((literal, example, truth))
        self.example_domains.add(example)

    def add_clause_truth(self, clause: int, example: int, truth: int) -> None:
        _nonnegative_int(clause, "clause")
        _nonnegative_int(example, "example")
        _binary(truth, "truth")
        self._require_room(self.clause_truths, self.max_clause_truths, "clause_truth")
        if example not in self.example_domains:
            self._require_room(
                self.example_domains, self.max_example_domains, "example_domain"
            )
        self.clause_truths.append((clause, example, truth))
        self.example_domains.add(example)

    def add_clause_literal(self, clause: int, literal: int) -> None:
        _nonnegative_int(clause, "clause")
        _nonnegative_int(literal, "literal")
        self._require_room(
            self.clause_literals, self.max_clause_literals, "clause_literal"
        )
        self.clause_literals.append((clause, literal))

    def add_class_support(self, target_class: int, positive: int, negative: int) -> None:
        _nonnegative_int(target_class, "target_class")
        _portable_nonnegative_int(positive, "positive")
        _portable_nonnegative_int(negative, "negative")
        self._require_room(self.class_supports, self.max_class_supports, "class_support")
        self.class_supports.append((target_class, positive, negative))

    def add_clause_class_score(
        self, clause: int, target_class: int, score: int | float
    ) -> None:
        _nonnegative_int(clause, "clause")
        _nonnegative_int(target_class, "target_class")
        numeric = _finite_number(score, "score")
        if abs(numeric) > MAX_COTM_SCORE_MAGNITUDE:
            raise ValueError("score exceeds the bounded CoTM weight range")
        self._require_room(
            self.clause_class_scores,
            self.max_clause_class_scores,
            "clause_class_score",
        )
        self.clause_class_scores.append((clause, target_class, numeric))

    def add_clause_support(self, clause: int, example: int) -> None:
        _nonnegative_int(clause, "clause")
        _nonnegative_int(example, "example")
        self._require_room(
            self.clause_supports, self.max_clause_supports, "clause_support"
        )
        self.clause_supports.append((clause, example))

    def add_clause_conflict(self, clause: int, example: int) -> None:
        _nonnegative_int(clause, "clause")
        _nonnegative_int(example, "example")
        self._require_room(
            self.clause_conflicts, self.max_clause_conflicts, "clause_conflict"
        )
        self.clause_conflicts.append((clause, example))

    def add_counterexample(
        self, model: str, example: int, expected: int, actual: int
    ) -> None:
        _printable(model, "model")
        _nonnegative_int(example, "example")
        _binary(expected, "expected")
        _binary(actual, "actual")
        self._require_room(
            self.counterexamples, self.max_counterexamples, "counterexample"
        )
        self.counterexamples.append((model, example, expected, actual))

    def add_insight(self, insight: PTAInsight) -> None:
        if not isinstance(insight, PTAInsight):
            raise TypeError("insight must be PTAInsight")
        self._require_room(self.insights, self.max_insights, "insight")
        self.insights.append(insight)

    def add_proposal(self, proposal: PTAEscalationProposal) -> None:
        if not isinstance(proposal, PTAEscalationProposal):
            raise TypeError("proposal must be PTAEscalationProposal")
        self._require_room(self.proposals, self.max_proposals, "proposal")
        self.proposals.append(proposal)

    def validate(self) -> None:
        """Validate bounds and values, including direct collection mutations."""

        _printable(self.dataset_id, "dataset_id")
        _nonnegative_int(self.generation, "generation")
        for name in self._budget_names():
            _validate_budget(getattr(self, name), name)
        for name in (
            "observations",
            "example_labels",
            "feature_supports",
            "feature_relations",
            "literal_truths",
            "clause_truths",
            "clause_literals",
            "class_supports",
            "clause_class_scores",
            "clause_supports",
            "clause_conflicts",
            "counterexamples",
            "insights",
            "proposals",
        ):
            if type(getattr(self, name)) is not list:
                raise TypeError(f"{name} must be a list")
        if type(self.example_domains) is not set:
            raise TypeError("example_domains must be a set")

        sized = (
            (self.observations, self.max_observations, "observation"),
            (self.example_labels, self.max_example_labels, "example_label"),
            (self.example_domains, self.max_example_domains, "example_domain"),
            (self.feature_supports, self.max_feature_supports, "feature_support"),
            (self.feature_relations, self.max_feature_relations, "feature_relation"),
            (self.literal_truths, self.max_literal_truths, "literal_truth"),
            (self.clause_truths, self.max_clause_truths, "clause_truth"),
            (self.clause_literals, self.max_clause_literals, "clause_literal"),
            (self.class_supports, self.max_class_supports, "class_support"),
            (
                self.clause_class_scores,
                self.max_clause_class_scores,
                "clause_class_score",
            ),
            (self.clause_supports, self.max_clause_supports, "clause_support"),
            (self.clause_conflicts, self.max_clause_conflicts, "clause_conflict"),
            (self.counterexamples, self.max_counterexamples, "counterexample"),
            (self.insights, self.max_insights, "insight"),
            (self.proposals, self.max_proposals, "proposal"),
        )
        for values, limit, name in sized:
            if len(values) > limit:
                raise ValueError(f"{name} budget exceeded")

        for pta, example, field_name, raw_value in self.observations:
            _printable(pta, "pta")
            _nonnegative_int(example, "example")
            _printable(field_name, "field")
            _prolog_term(raw_value)
        for example, label in self.example_labels:
            _nonnegative_int(example, "example")
            _binary(label, "label")
        for example in self.example_domains:
            _nonnegative_int(example, "example")
        for literal, positive, negative in self.feature_supports:
            _nonnegative_int(literal, "literal")
            _portable_nonnegative_int(positive, "positive")
            _portable_nonnegative_int(negative, "negative")
        for left, relation, right in self.feature_relations:
            _nonnegative_int(left, "left literal")
            _printable(relation, "relation")
            _nonnegative_int(right, "right literal")
        for literal, example, truth in self.literal_truths:
            _nonnegative_int(literal, "literal")
            _nonnegative_int(example, "example")
            _binary(truth, "truth")
        for clause, example, truth in self.clause_truths:
            _nonnegative_int(clause, "clause")
            _nonnegative_int(example, "example")
            _binary(truth, "truth")
        truth_examples = {
            example for _, example, _ in self.literal_truths
        } | {example for _, example, _ in self.clause_truths}
        if not truth_examples.issubset(self.example_domains):
            raise ValueError("truth facts contain examples outside example_domains")
        for clause, literal in self.clause_literals:
            _nonnegative_int(clause, "clause")
            _nonnegative_int(literal, "literal")
        for target_class, positive, negative in self.class_supports:
            _nonnegative_int(target_class, "target_class")
            _portable_nonnegative_int(positive, "positive")
            _portable_nonnegative_int(negative, "negative")
        for clause, target_class, score in self.clause_class_scores:
            _nonnegative_int(clause, "clause")
            _nonnegative_int(target_class, "target_class")
            numeric = _finite_number(score, "score")
            if abs(numeric) > MAX_COTM_SCORE_MAGNITUDE:
                raise ValueError("score exceeds the bounded CoTM weight range")
        for clause, example in self.clause_supports:
            _nonnegative_int(clause, "clause")
            _nonnegative_int(example, "example")
        for clause, example in self.clause_conflicts:
            _nonnegative_int(clause, "clause")
            _nonnegative_int(example, "example")
        for model, example, expected, actual in self.counterexamples:
            _printable(model, "model")
            _nonnegative_int(example, "example")
            _binary(expected, "expected")
            _binary(actual, "actual")
        if any(not isinstance(item, PTAInsight) for item in self.insights):
            raise TypeError("insights must contain PTAInsight objects")
        if any(not isinstance(item, PTAEscalationProposal) for item in self.proposals):
            raise TypeError("proposals must contain PTAEscalationProposal objects")

    def iter_prolog_fact_lines(
        self,
        *,
        observation_field_ids: Mapping[str, int] | None = None,
        example_ids: Mapping[int, int] | None = None,
        literal_ids: Mapping[int, int] | None = None,
        clause_ids: Mapping[int, int] | None = None,
        class_ids: Mapping[int, int] | None = None,
    ) -> Iterator[str]:
        """Yield bounded data-only Prolog facts one line at a time.

        The optional maps replace semantic identifiers with small opaque
        integers that fit GNU Prolog's portable integer range. The collective
        decoder restores the original identifiers.
        """

        self.validate()
        if observation_field_ids is not None:
            if any(
                type(name) is not str or type(identifier) is not int
                for name, identifier in observation_field_ids.items()
            ):
                raise TypeError("observation_field_ids must map strings to integers")
            if any(identifier < 0 for identifier in observation_field_ids.values()):
                raise ValueError("observation field identifiers must be nonnegative")
            if len(set(observation_field_ids.values())) != len(observation_field_ids):
                raise ValueError("observation field identifiers must be unique")

        identifier_maps = {
            "example_ids": example_ids,
            "literal_ids": literal_ids,
            "clause_ids": clause_ids,
            "class_ids": class_ids,
        }
        for name, identifiers in identifier_maps.items():
            if identifiers is None:
                continue
            if any(
                type(source) is not int or type(target) is not int
                for source, target in identifiers.items()
            ):
                raise TypeError(f"{name} must map integers to integers")
            if any(
                target < 0 or target > GPROLOG_MAX_INTEGER
                for target in identifiers.values()
            ):
                raise ValueError(f"{name} target is outside GNU Prolog's range")
            if len(set(identifiers.values())) != len(identifiers):
                raise ValueError(f"{name} targets must be unique")

        def mapped(value: int, identifiers: Mapping[int, int] | None, name: str) -> int:
            if identifiers is None:
                return value
            if value not in identifiers:
                raise ValueError(f"missing opaque identifier for {name} {value}")
            return identifiers[value]

        yield "% PTAReasoningSession facts -- generated, bounded, data-only"
        for pta, example, field_name, raw_value in self.observations:
            if observation_field_ids is None:
                field_term = _prolog_atom(field_name)
            else:
                if field_name not in observation_field_ids:
                    raise ValueError(f"missing opaque identifier for field {field_name!r}")
                field_term = str(observation_field_ids[field_name])
            yield (
                f"observation({_prolog_atom(pta)},"
                f"{mapped(example, example_ids, 'example')},{field_term},"
                f"{_prolog_term(raw_value)})."
            )
        for example in sorted(self.example_domains):
            yield f"example_domain({mapped(example, example_ids, 'example')})."
        for example, label in self.example_labels:
            yield (
                f"example_label({mapped(example, example_ids, 'example')},{label})."
            )
        for literal, positive, negative in self.feature_supports:
            yield (
                f"feature_support({mapped(literal, literal_ids, 'literal')},"
                f"{positive},{negative})."
            )
        for left, relation, right in self.feature_relations:
            yield (
                f"feature_relation({mapped(left, literal_ids, 'literal')},"
                f"{_prolog_atom(relation)},"
                f"{mapped(right, literal_ids, 'literal')})."
            )
        for literal, example, truth in self.literal_truths:
            yield (
                f"literal_truth({mapped(literal, literal_ids, 'literal')},"
                f"{mapped(example, example_ids, 'example')},{truth})."
            )
        for clause, example, truth in self.clause_truths:
            yield (
                f"clause_truth({mapped(clause, clause_ids, 'clause')},"
                f"{mapped(example, example_ids, 'example')},{truth})."
            )
        for clause, literal in self.clause_literals:
            yield (
                f"clause_literal({mapped(clause, clause_ids, 'clause')},"
                f"{mapped(literal, literal_ids, 'literal')})."
            )
        for target_class, positive, negative in self.class_supports:
            yield (
                f"class_support({mapped(target_class, class_ids, 'class')},"
                f"{positive},{negative})."
            )
        for clause, target_class, score in self.clause_class_scores:
            yield (
                f"clause_class_score({mapped(clause, clause_ids, 'clause')},"
                f"{mapped(target_class, class_ids, 'class')},{score!r})."
            )
        for clause, example in self.clause_supports:
            yield (
                f"clause_support({mapped(clause, clause_ids, 'clause')},"
                f"{mapped(example, example_ids, 'example')})."
            )
        for clause, example in self.clause_conflicts:
            yield (
                f"clause_conflict({mapped(clause, clause_ids, 'clause')},"
                f"{mapped(example, example_ids, 'example')})."
            )
        for insight in self.insights:
            yield (
                f"insight({_prolog_atom(insight.source_pta)},"
                f"{_prolog_atom(insight.kind)},{_prolog_atom(insight.subject)},"
                f"{_prolog_term(list(insight.evidence))})."
            )
        for model, example, expected, actual in self.counterexamples:
            yield (
                f"counterexample({_prolog_atom(model)},"
                f"{mapped(example, example_ids, 'example')},{expected},{actual})."
            )
        for proposal in self.proposals:
            yield (
                f"proposal({_prolog_atom(proposal.proposal_id)},"
                f"{_prolog_atom(proposal.native_target)},"
                f"{_prolog_atom(proposal.proposal_hash())})."
            )

    def to_prolog_facts(
        self,
        *,
        observation_field_ids: Mapping[str, int] | None = None,
        example_ids: Mapping[int, int] | None = None,
        literal_ids: Mapping[int, int] | None = None,
        clause_ids: Mapping[int, int] | None = None,
        class_ids: Mapping[int, int] | None = None,
    ) -> str:
        """Materialize fact lines for callers that explicitly need a string."""

        return "\n".join(
            self.iter_prolog_fact_lines(
                observation_field_ids=observation_field_ids,
                example_ids=example_ids,
                literal_ids=literal_ids,
                clause_ids=clause_ids,
                class_ids=class_ids,
            )
        ) + "\n"

    def consult_via_gprolog(self, *, timeout: float = 10.0) -> PTACollectiveResult:
        """Run the typed PTA collective (compatibility entry point)."""

        from .collective import PTACollectiveBudget, PTACollectiveService

        return PTACollectiveService().run(
            self, budget=PTACollectiveBudget(timeout_seconds=timeout)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "generation": self.generation,
            "observations": list(self.observations),
            "example_labels": list(self.example_labels),
            "example_domains": sorted(self.example_domains),
            "feature_supports": list(self.feature_supports),
            "feature_relations": list(self.feature_relations),
            "literal_truths": list(self.literal_truths),
            "clause_truths": list(self.clause_truths),
            "clause_literals": list(self.clause_literals),
            "class_supports": list(self.class_supports),
            "clause_class_scores": list(self.clause_class_scores),
            "clause_supports": list(self.clause_supports),
            "clause_conflicts": list(self.clause_conflicts),
            "counterexamples": list(self.counterexamples),
            "insights": [
                {
                    "source_pta": insight.source_pta,
                    "kind": insight.kind,
                    "subject": insight.subject,
                    "evidence": list(insight.evidence),
                }
                for insight in self.insights
            ],
            "proposals": [proposal.to_dict() for proposal in self.proposals],
        }
