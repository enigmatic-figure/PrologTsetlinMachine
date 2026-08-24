"""Stable request/result services for bounded GNU Prolog searches."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any

from ..feature_templates import DataType
from ..logic_consolidation import LogicProgram32
from ..model_artifact import export_logic_program
from ..prolog_bridge import (
    BooleanDecisionTree,
    DecisionTreeSearchProblem,
    FeatureTemplateCandidate,
    FeatureTemplateSearchProblem,
    GNUPrologSearch,
    PrologBridgeError,
    TAClauseSearchProblem,
    ThresholdSearchProblem,
)
from ._atomic import publish_bytes


SEARCH_REQUEST_SCHEMA = "ptm.search.request.v1"
SEARCH_RESULT_SCHEMA = "ptm.search.result.v1"


class SearchKind(str, Enum):
    THRESHOLD = "threshold"
    FEATURE_TEMPLATE = "feature-template"
    TA_CLAUSE = "ta-clause"
    DECISION_TREE = "decision-tree"
    REPAIR = "repair"


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


@dataclass(frozen=True, slots=True)
class BoundedSearchRequest:
    kind: SearchKind
    problem: Mapping[str, Any]
    timeout_seconds: float = 30.0
    max_iterations: int = 32

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        expected_kind: SearchKind | str | None = None,
        timeout_seconds: float | None = None,
    ) -> "BoundedSearchRequest":
        if value.get("schema") != SEARCH_REQUEST_SCHEMA:
            raise ValueError(f"search request schema must be {SEARCH_REQUEST_SCHEMA}")
        try:
            kind = SearchKind(str(value["kind"]))
        except (KeyError, ValueError) as error:
            raise ValueError("search request kind is invalid") from error
        if expected_kind is not None and kind is not SearchKind(expected_kind):
            raise ValueError(
                f"search request kind {kind.value} does not match {SearchKind(expected_kind).value}"
            )
        problem = value.get("problem")
        if not isinstance(problem, Mapping):
            raise ValueError("search request problem must be an object")
        timeout = _number(
            timeout_seconds
            if timeout_seconds is not None
            else value.get("timeout_seconds", 30.0),
            "timeout_seconds",
        )
        if not 0.1 <= timeout <= 300.0:
            raise ValueError("timeout_seconds must be between 0.1 and 300")
        iterations = _integer(value.get("max_iterations", 32), "max_iterations")
        if not 1 <= iterations <= 256:
            raise ValueError("max_iterations must be between 1 and 256")
        if kind is not SearchKind.REPAIR and "max_iterations" in value:
            raise ValueError("max_iterations is valid only for repair requests")
        allowed = {"schema", "kind", "problem", "timeout_seconds", "max_iterations"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"search request contains unknown fields: {', '.join(unknown)}")
        return cls(kind, dict(problem), timeout, iterations)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": SEARCH_REQUEST_SCHEMA,
            "kind": self.kind.value,
            "problem": dict(self.problem),
            "timeout_seconds": self.timeout_seconds,
        }
        if self.kind is SearchKind.REPAIR:
            result["max_iterations"] = self.max_iterations
        return result


@dataclass(frozen=True, slots=True)
class BoundedSearchResult:
    kind: SearchKind
    report: Mapping[str, Any]
    elapsed_seconds: float
    logic_program: LogicProgram32 | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def exportable(self) -> bool:
        return self.logic_program is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SEARCH_RESULT_SCHEMA,
            "kind": self.kind.value,
            "status": "solved",
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "exportable": self.exportable,
            "result": dict(self.report),
        }


def _tree_from_dict(value: object, *, nesting: int = 0) -> BooleanDecisionTree:
    if nesting > 32 or not isinstance(value, Mapping):
        raise ValueError("repair parent tree is invalid or too deeply nested")
    if set(value) == {"leaf"} and type(value["leaf"]) is bool:
        return BooleanDecisionTree.leaf(bool(value["leaf"]))
    if set(value) != {"feature", "false", "true"}:
        raise ValueError("tree nodes require feature, false, and true fields")
    feature = value["feature"]
    if type(feature) is not int or feature < 0:
        raise ValueError("tree feature indices must be nonnegative integers")
    return BooleanDecisionTree.node(
        feature,
        _tree_from_dict(value["false"], nesting=nesting + 1),
        _tree_from_dict(value["true"], nesting=nesting + 1),
    )


def _threshold_candidate_bound(problem: ThresholdSearchProblem) -> int:
    return sum(
        math.comb(problem.slot_count, width) * width
        for width in range(1, problem.max_selected + 1)
    )


def _logic_program_if_portable(
    tree: BooleanDecisionTree, slot_count: int
) -> LogicProgram32 | None:
    if slot_count > 5 or any(feature >= 5 for feature in tree.feature_indices):
        return None
    return tree.to_logic_program()


def _template_candidates(value: object) -> tuple[FeatureTemplateCandidate, ...]:
    if not isinstance(value, list):
        raise ValueError("feature-template candidates must be an array")
    candidates = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"feature-template candidate {index} must be an object")
        parameters = item.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError(f"feature-template candidate {index} parameters must be an object")
        candidates.append(
            FeatureTemplateCandidate.create(
                field_name=str(item["field_name"]),
                template_id=str(item["template_id"]),
                data_type=DataType(str(item["data_type"])),
                parameters=parameters,
            )
        )
    return tuple(candidates)


def search_request_budget(request: BoundedSearchRequest) -> dict[str, Any]:
    """Validate a request and report its finite bounds without launching Prolog."""

    value = request.problem
    if request.kind is SearchKind.THRESHOLD:
        problem = ThresholdSearchProblem.create(
            slot_count=_integer(value["slot_count"], "slot_count"),
            max_selected=_integer(value["max_selected"], "max_selected"),
            positive_examples=value["positive_examples"],
            negative_examples=value["negative_examples"],
        )
        candidates = _threshold_candidate_bound(problem)
        examples = len(problem.positive_examples) + len(problem.negative_examples)
    elif request.kind is SearchKind.FEATURE_TEMPLATE:
        candidates_value = _template_candidates(value["candidates"])
        problem = FeatureTemplateSearchProblem.create(
            candidates=candidates_value,
            labels=value["labels"],
            coverage=value["coverage"],
        )
        candidates = len(problem.candidates)
        examples = len(problem.labels)
    elif request.kind is SearchKind.TA_CLAUSE:
        problem = TAClauseSearchProblem.create(
            feature_count=_integer(value["feature_count"], "feature_count"),
            max_literals=_integer(value["max_literals"], "max_literals"),
            examples=value["examples"],
            labels=value["labels"],
        )
        candidates = problem.candidate_upper_bound
        examples = len(problem.examples)
    else:
        problem = DecisionTreeSearchProblem.create(
            slot_count=_integer(value["slot_count"], "slot_count"),
            max_depth=_integer(value["max_depth"], "max_depth"),
            examples=value["examples"],
            labels=value["labels"],
        )
        if request.kind is SearchKind.REPAIR:
            parent = _tree_from_dict(value["parent"])
            if not parent.is_read_once() or any(
                feature >= problem.slot_count for feature in parent.feature_indices
            ):
                raise ValueError("repair parent is incompatible with the slot domain")
        candidates = problem.candidate_upper_bound
        examples = len(problem.examples)
    return {
        "kind": request.kind.value,
        "candidate_upper_bound": candidates,
        "example_count": examples,
        "timeout_seconds": request.timeout_seconds,
        "max_iterations": (
            request.max_iterations if request.kind is SearchKind.REPAIR else None
        ),
    }


def run_bounded_search(
    request: BoundedSearchRequest,
    *,
    cancel: Callable[[], bool] | None = None,
    executable: str | Path | None = None,
) -> BoundedSearchResult:
    """Validate all bounds, run one search, and return a typed JSON report."""

    started = perf_counter()
    deadline = started + request.timeout_seconds

    def remaining_time() -> float:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise PrologBridgeError(
                f"bounded Prolog {request.kind.value} request timed out after "
                f"{request.timeout_seconds:g}s"
            )
        return remaining

    search_request_budget(request)
    search = GNUPrologSearch(executable)
    problem_value = request.problem
    program: LogicProgram32 | None = None

    if request.kind is SearchKind.THRESHOLD:
        problem = ThresholdSearchProblem.create(
            slot_count=_integer(problem_value["slot_count"], "slot_count"),
            max_selected=_integer(problem_value["max_selected"], "max_selected"),
            positive_examples=problem_value["positive_examples"],
            negative_examples=problem_value["negative_examples"],
        )
        result = search.search(
            problem, timeout_seconds=remaining_time(), cancel=cancel
        )
        report = {
            "selected_slots": list(result.selected_slots),
            "minimum_true": result.minimum_true,
            "mismatch_count": result.mismatch_count,
            "dataset_digest": problem.dataset_digest(),
            "candidate_upper_bound": _threshold_candidate_bound(problem),
        }
    elif request.kind is SearchKind.FEATURE_TEMPLATE:
        candidates = _template_candidates(problem_value["candidates"])
        problem = FeatureTemplateSearchProblem.create(
            candidates=candidates,
            labels=problem_value["labels"],
            coverage=problem_value["coverage"],
        )
        result = search.search_feature_template(
            problem, timeout_seconds=remaining_time(), cancel=cancel
        )
        report = {
            "candidate_index": result.candidate_index,
            "candidate": {
                "field_name": result.candidate.field_name,
                "template_id": result.candidate.template_id,
                "data_type": result.candidate.data_type.value,
                "parameters": result.candidate.parameters,
            },
            "mismatch_count": result.mismatch_count,
            "dataset_digest": result.dataset_digest,
            "candidate_upper_bound": len(candidates),
        }
    elif request.kind is SearchKind.TA_CLAUSE:
        problem = TAClauseSearchProblem.create(
            feature_count=_integer(problem_value["feature_count"], "feature_count"),
            max_literals=_integer(problem_value["max_literals"], "max_literals"),
            examples=problem_value["examples"],
            labels=problem_value["labels"],
        )
        result = search.search_ta_clause(
            problem, timeout_seconds=remaining_time(), cancel=cancel
        )
        configuration_value = problem_value.get("configuration", {})
        if not isinstance(configuration_value, Mapping):
            raise ValueError("TA configuration must be an object")
        configuration = result.to_ta_configuration(
            problem,
            states_per_action=_integer(
                configuration_value.get("states_per_action", 100),
                "states_per_action",
            ),
            specificity=_number(
                configuration_value.get("specificity", 3.0), "specificity"
            ),
            threshold=_integer(
                configuration_value.get("threshold", 10), "threshold"
            ),
        )
        report = {
            "included_literals": list(result.included_literals),
            "signed_literals": [
                {
                    "feature": literal // 2,
                    "negated": literal % 2 == 1,
                }
                for literal in result.included_literals
            ],
            "mismatch_count": result.mismatch_count,
            "dataset_digest": problem.dataset_digest(),
            "candidate_upper_bound": problem.candidate_upper_bound,
            "ta_configuration": json.loads(configuration.to_json()),
        }
    else:
        problem = DecisionTreeSearchProblem.create(
            slot_count=_integer(problem_value["slot_count"], "slot_count"),
            max_depth=_integer(problem_value["max_depth"], "max_depth"),
            examples=problem_value["examples"],
            labels=problem_value["labels"],
        )
        if request.kind is SearchKind.DECISION_TREE:
            result = search.search_decision_tree(
                problem, timeout_seconds=remaining_time(), cancel=cancel
            )
            tree = result.tree
            program = _logic_program_if_portable(tree, problem.slot_count)
            report = {
                "tree": tree.to_dict(),
                "prefix_encoding": list(tree.prefix_encoding()),
                "node_count": tree.node_count,
                "depth": tree.depth,
                "mismatch_count": result.mismatch_count,
                "dataset_digest": problem.dataset_digest(),
                "candidate_upper_bound": problem.candidate_upper_bound,
            }
        else:
            parent = _tree_from_dict(problem_value["parent"])
            result = search.repair_decision_tree(
                parent,
                problem,
                max_iterations=request.max_iterations,
                timeout_seconds=remaining_time(),
                cancel=cancel,
            )
            program = (
                result.to_logic_program()
                if problem.slot_count <= 5
                and all(feature < 5 for feature in parent.feature_indices)
                and all(feature < 5 for feature in result.guard.feature_indices)
                else None
            )
            report = {
                "parent": result.parent.to_dict(),
                "guard": result.guard.to_dict(),
                "mismatches_before": result.mismatches_before,
                "mismatch_count": result.mismatch_count,
                "counterexamples": [
                    {
                        "example": list(item.example),
                        "expected": item.expected,
                        "parent_prediction": item.parent_prediction,
                        "required_flip": item.required_flip,
                    }
                    for item in result.counterexamples
                ],
                "dataset_digest": problem.dataset_digest(),
                "candidate_upper_bound": problem.candidate_upper_bound,
            }
    remaining_time()
    return BoundedSearchResult(request.kind, report, perf_counter() - started, program)


def export_search_artifact(
    result: BoundedSearchResult,
    path: str | Path,
    *,
    name: str = "bounded-prolog-search",
) -> dict[str, Any]:
    """Export tree/repair behavior as a verified fixed-Logic artifact."""

    if result.logic_program is None:
        raise ValueError("this search result cannot be lowered to fixed five-binding Logic")
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".ptm":
        raise ValueError("search artifact path must end in .ptm")
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact = export_logic_program(
        result.logic_program,
        name=name.strip() or "bounded-prolog-search",
        description=f"Bounded GNU Prolog {result.kind.value} result",
        license="Apache-2.0",
        intended_use="bounded symbolic search exploration",
        limitations="five Boolean bindings; validate against the source search problem",
        validation_signature={
            "search_kind": result.kind.value,
            "dataset_digest": result.report.get("dataset_digest"),
            "mismatch_count": result.report.get("mismatch_count"),
        },
    )
    publish_bytes(destination, artifact.serialized, overwrite=False)
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_kind": artifact.manifest["artifact_kind"],
        "output": str(destination),
        "size_bytes": len(artifact.serialized),
    }


_DEMO_REQUESTS: dict[SearchKind, dict[str, Any]] = {
    SearchKind.THRESHOLD: {
        "schema": SEARCH_REQUEST_SCHEMA,
        "kind": "threshold",
        "timeout_seconds": 30,
        "problem": {
            "slot_count": 3,
            "max_selected": 3,
            "positive_examples": [[0], [1], [0, 1], [0, 2]],
            "negative_examples": [[], [2]],
        },
    },
    SearchKind.FEATURE_TEMPLATE: {
        "schema": SEARCH_REQUEST_SCHEMA,
        "kind": "feature-template",
        "timeout_seconds": 30,
        "problem": {
            "candidates": [
                {
                    "field_name": "status",
                    "template_id": "categorical_v1",
                    "data_type": "categorical",
                    "parameters": {"categories": ["cold"]},
                },
                {
                    "field_name": "status",
                    "template_id": "categorical_v1",
                    "data_type": "categorical",
                    "parameters": {"categories": ["hot"]},
                },
            ],
            "labels": [0, 1, 1, 0],
            "coverage": [[0, 0, 1, 0], [0, 1, 1, 0]],
        },
    },
    SearchKind.TA_CLAUSE: {
        "schema": SEARCH_REQUEST_SCHEMA,
        "kind": "ta-clause",
        "timeout_seconds": 30,
        "problem": {
            "feature_count": 2,
            "max_literals": 2,
            "examples": [[], [0], [1], [0, 1]],
            "labels": [0, 1, 0, 0],
            "configuration": {
                "states_per_action": 100,
                "specificity": 3.0,
                "threshold": 10,
            },
        },
    },
    SearchKind.DECISION_TREE: {
        "schema": SEARCH_REQUEST_SCHEMA,
        "kind": "decision-tree",
        "timeout_seconds": 30,
        "problem": {
            "slot_count": 2,
            "max_depth": 2,
            "examples": [[], [0], [1], [0, 1]],
            "labels": [0, 1, 1, 0],
        },
    },
    SearchKind.REPAIR: {
        "schema": SEARCH_REQUEST_SCHEMA,
        "kind": "repair",
        "timeout_seconds": 30,
        "max_iterations": 4,
        "problem": {
            "slot_count": 2,
            "max_depth": 2,
            "examples": [[], [0], [1], [0, 1]],
            "labels": [0, 1, 1, 0],
            "parent": {"leaf": False},
        },
    },
}


def demo_search_document(kind: SearchKind | str) -> dict[str, Any]:
    return copy.deepcopy(_DEMO_REQUESTS[SearchKind(kind)])
