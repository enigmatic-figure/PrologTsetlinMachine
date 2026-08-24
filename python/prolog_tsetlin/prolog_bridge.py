"""Bounded GNU Prolog search and lowering to native PA artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from ._bounded_process import (
    BoundedProcessCancelled,
    BoundedProcessDrainError,
    BoundedProcessLaunchError,
    BoundedProcessOutputLimit,
    BoundedProcessTimeout,
    run_bounded_process,
)
from .artifact import (
    InputShape,
    PAArtifact,
    RestorationHandle,
    SlotBinding,
    ValidationSignature,
)
from .feature_templates import (
    ClauseConfiguration,
    DataType,
    TAClauseConfiguration,
    TemplateRegistry,
    create_feature_template_catalog,
)
from .logic_ast import (
    LOGIC_AST_VARIABLES,
    PrimitiveLogicGraph,
    PrimitiveLogicNode,
    PrimitiveLogicOp,
)
from .logic_consolidation import LogicProgram32
from .pa import PortSemantic
from .prolog_resources import PrologResourceError, resolve_prolog_module


_RESULT_PATTERN = re.compile(
    r"^PTM_RESULT v1 masked_threshold selected=(?P<slots>[0-9,]+) "
    r"minimum=(?P<minimum>[0-9]+) mismatches=(?P<mismatches>[0-9]+)$",
    re.MULTILINE,
)
_NO_SOLUTION_PATTERN = re.compile(r"^PTM_RESULT v1 no_solution$", re.MULTILINE)
MAX_SEARCH_CANDIDATES = 1_000_000
MAX_PROLOG_OUTPUT_BYTES = 262_144


class PrologBridgeError(RuntimeError):
    pass


class NoThresholdSolution(PrologBridgeError):
    pass


class NoFeatureTemplateSolution(PrologBridgeError):
    pass


class NoTAClauseSolution(PrologBridgeError):
    pass


class NoDecisionTreeSolution(PrologBridgeError):
    pass


class RepairDidNotConverge(PrologBridgeError):
    pass


class PrologSearchCancelled(PrologBridgeError):
    pass


def _require_binary(value: object) -> bool:
    if value is True or value is False:
        return bool(value)
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError("binary value must be bool or integer 0/1")


def _normalized_example(example: Iterable[int], slot_count: int) -> tuple[int, ...]:
    result = tuple(sorted(set(example)))
    if any(slot < 0 or slot >= slot_count for slot in result):
        raise ValueError("example contains a slot outside the declared shape")
    return result


@dataclass(frozen=True, slots=True)
class ThresholdSearchProblem:
    slot_count: int
    max_selected: int
    positive_examples: tuple[tuple[int, ...], ...]
    negative_examples: tuple[tuple[int, ...], ...]

    @classmethod
    def create(
        cls,
        *,
        slot_count: int,
        max_selected: int,
        positive_examples: Iterable[Iterable[int]],
        negative_examples: Iterable[Iterable[int]],
    ) -> "ThresholdSearchProblem":
        if not 1 <= slot_count <= 4096:
            raise ValueError("slot_count must be between 1 and 4096")
        if not 1 <= max_selected <= min(slot_count, 16):
            raise ValueError("max_selected must be between 1 and min(slot_count, 16)")
        candidate_rules = sum(
            math.comb(slot_count, width) * width
            for width in range(1, max_selected + 1)
        )
        if candidate_rules > MAX_SEARCH_CANDIDATES:
            raise ValueError(
                f"search bound admits {candidate_rules} candidates; maximum is "
                f"{MAX_SEARCH_CANDIDATES}"
            )
        positives = tuple(
            _normalized_example(example, slot_count) for example in positive_examples
        )
        negatives = tuple(
            _normalized_example(example, slot_count) for example in negative_examples
        )
        if not positives or not negatives:
            raise ValueError("exact search requires both positive and negative examples")
        if len(positives) + len(negatives) > 4096:
            raise ValueError("bounded search accepts at most 4096 examples")
        return cls(slot_count, max_selected, positives, negatives)

    def dataset_digest(self) -> str:
        payload = {
            "schema": "ptm-threshold-problem-v1",
            "slot_count": self.slot_count,
            "positive_examples": self.positive_examples,
            "negative_examples": self.negative_examples,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_threshold_result(
    problem: "ThresholdSearchProblem",
    result: "ThresholdSearchResult",
) -> None:
    """Pure-Python trust-boundary check for GNU Prolog threshold output."""

    selected = result.selected_slots
    minimum = result.minimum_true
    mismatches = result.mismatch_count
    if mismatches != 0:
        raise PrologBridgeError("Prolog threshold result reports nonzero mismatches")
    if not selected:
        raise PrologBridgeError("Prolog threshold result has no selected slots")
    if len(selected) != len(set(selected)):
        raise PrologBridgeError("Prolog threshold result has duplicate slots")
    if len(selected) > problem.max_selected:
        raise PrologBridgeError(
            "Prolog threshold result exceeds the declared max_selected bound"
        )
    if any(not 0 <= slot < problem.slot_count for slot in selected):
        raise PrologBridgeError("Prolog threshold result has a slot outside its domain")
    if not 1 <= minimum <= len(selected):
        raise PrologBridgeError("Prolog threshold result has an invalid minimum")
    selected_set = frozenset(selected)
    for example in problem.positive_examples:
        if len(selected_set.intersection(example)) < minimum:
            raise PrologBridgeError("Prolog threshold result failed Python validation")
    for example in problem.negative_examples:
        if len(selected_set.intersection(example)) >= minimum:
            raise PrologBridgeError("Prolog threshold result failed Python validation")


@dataclass(frozen=True, slots=True)
class ThresholdSearchResult:
    selected_slots: tuple[int, ...]
    minimum_true: int
    mismatch_count: int


def _prolog_list(values: Sequence[int]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _prolog_examples(examples: Sequence[Sequence[int]]) -> str:
    return "[" + ",".join(_prolog_list(example) for example in examples) + "]"


def _has_monotonicity_contradiction(problem: ThresholdSearchProblem) -> bool:
    """Prove no monotone threshold can separate a positive subset of a negative."""
    negatives = tuple(frozenset(example) for example in problem.negative_examples)
    return any(
        frozenset(positive) <= negative
        for positive in problem.positive_examples
        for negative in negatives
    )


def _default_gprolog_path() -> Path | None:
    configured = os.environ.get("PTM_GPROLOG")
    candidates = [
        Path(configured) if configured else None,
        Path(shutil.which("gprolog")) if shutil.which("gprolog") else None,
    ]
    return next((path for path in candidates if path is not None and path.is_file()), None)


def _default_prolog_source(filename: str) -> Path:
    try:
        return resolve_prolog_module(filename)
    except PrologResourceError as exc:
        raise PrologBridgeError(str(exc)) from exc


def _prolog_process_environment(
    environment: Mapping[str, str] | None = None,
    *,
    windows: bool | None = None,
) -> dict[str, str]:
    """Return an isolated environment for a noninteractive GNU Prolog child.

    Windows GNU Prolog distributions can be linked with the linedit GUI
    console.  ``CREATE_NO_WINDOW`` does not suppress a GUI window opened by
    linedit itself, so explicitly force GNU Prolog's documented text-console
    mode.  Do not mutate the caller's environment or disable linedit for an
    interactive GNU Prolog process launched outside PTM.
    """

    child_environment = dict(os.environ if environment is None else environment)
    is_windows = os.name == "nt" if windows is None else windows
    if is_windows:
        child_environment["LINEDIT"] = "gui=no"
    return child_environment


def _run_prolog_process(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    cancel: Callable[[], bool] | None,
) -> subprocess.CompletedProcess[str]:
    """Run GNU Prolog through PTM's shared process-tree boundary."""

    def diagnostic(error: object) -> str:
        stdout = getattr(error, "stdout", b"")
        stderr = getattr(error, "stderr", b"")
        return (stdout + b"\n" + stderr)[-2_000:].decode(
            "utf-8", errors="replace"
        )

    try:
        completed = run_bounded_process(
            command,
            timeout_seconds=timeout_seconds,
            max_output_bytes=MAX_PROLOG_OUTPUT_BYTES,
            cancel=cancel,
            env=_prolog_process_environment(),
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            ),
        )
    except BoundedProcessCancelled as exc:
        raise PrologSearchCancelled(
            "bounded Prolog search cancelled\n" + diagnostic(exc)
        ) from exc
    except BoundedProcessTimeout as exc:
        raise PrologBridgeError(
            f"bounded Prolog search timed out after {timeout_seconds:g}s\n"
            + diagnostic(exc)
        ) from exc
    except BoundedProcessOutputLimit as exc:
        raise PrologBridgeError(
            "bounded Prolog search output exceeded its byte budget\n"
            + diagnostic(exc)
        ) from exc
    except (BoundedProcessLaunchError, BoundedProcessDrainError) as exc:
        raise PrologBridgeError(
            f"bounded Prolog process boundary failed: {exc}\n" + diagnostic(exc)
        ) from exc

    def decoded(stream: bytes) -> str:
        return stream.decode("utf-8", errors="replace").replace(
            "\r\n", "\n"
        ).replace("\r", "\n")

    return subprocess.CompletedProcess(
        list(command),
        completed.returncode,
        decoded(completed.stdout),
        decoded(completed.stderr),
    )


class GNUPrologThresholdSearch:
    def __init__(
        self,
        executable: str | os.PathLike[str] | None = None,
        source_file: str | os.PathLike[str] | None = None,
    ) -> None:
        resolved = Path(executable) if executable is not None else _default_gprolog_path()
        if resolved is None or not resolved.is_file():
            raise PrologBridgeError(
                "GNU Prolog was not found; set PTM_GPROLOG to the executable path"
            )
        self.executable = resolved.resolve()
        default_source = _default_prolog_source("bounded_threshold_search.pl")
        self.source_file = (
            Path(source_file).resolve() if source_file is not None else default_source
        )
        if not self.source_file.is_file():
            raise PrologBridgeError(f"Prolog search source not found: {self.source_file}")

    def _driver_source(self, problem: ThresholdSearchProblem) -> str:
        source = self.source_file.as_posix().replace("'", "''")
        return (
            f":- include('{source}').\n"
            f"problem({problem.slot_count},{problem.max_selected},"
            f"{_prolog_examples(problem.positive_examples)},"
            f"{_prolog_examples(problem.negative_examples)}).\n"
        )

    def search(
        self,
        problem: ThresholdSearchProblem,
        *,
        timeout_seconds: float = 30.0,
        cancel: Callable[[], bool] | None = None,
    ) -> ThresholdSearchResult:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if _has_monotonicity_contradiction(problem):
            raise NoThresholdSolution(
                "no monotone threshold can classify a positive subset of a negative"
            )
        with tempfile.TemporaryDirectory(prefix="ptm-prolog-") as directory:
            driver = Path(directory) / "problem_driver.pl"
            driver.write_text(self._driver_source(problem), encoding="utf-8")
            completed = _run_prolog_process(
                [
                    str(self.executable),
                    "--consult-file",
                    str(driver),
                    "--query-goal",
                    "ptm_run_problem",
                ],
                timeout_seconds=timeout_seconds,
                cancel=cancel,
            )
        output = completed.stdout + "\n" + completed.stderr
        if completed.returncode != 0:
            raise PrologBridgeError(
                f"GNU Prolog exited with status {completed.returncode}\n"
                f"{output[-2000:]}"
            )
        match = _RESULT_PATTERN.search(output)
        if match is not None:
            selected = tuple(
                int(slot) for slot in match.group("slots").split(",") if slot
            )
            result = ThresholdSearchResult(
                selected_slots=selected,
                minimum_true=int(match.group("minimum")),
                mismatch_count=int(match.group("mismatches")),
            )
            _validate_threshold_result(problem, result)
            return result
        if _NO_SOLUTION_PATTERN.search(output):
            raise NoThresholdSolution("no exact bounded threshold rule exists")
        raise PrologBridgeError(
            f"GNU Prolog did not return the PTM result protocol (exit {completed.returncode})\n"
            f"{output[-2000:]}"
        )

    def search_artifact(
        self,
        problem: ThresholdSearchProblem,
        *,
        input_shape: InputShape,
        port_semantic: PortSemantic,
        mapping_version: str,
        slot_bindings: Sequence[SlotBinding],
        restoration_handle: RestorationHandle,
        timeout_seconds: float = 30.0,
        cancel: Callable[[], bool] | None = None,
    ) -> PAArtifact:
        if problem.slot_count > input_shape.bit_count:
            raise ValueError("search problem exceeds the target PA input shape")
        result = self.search(
            problem, timeout_seconds=timeout_seconds, cancel=cancel
        )
        return PAArtifact.create_masked_threshold(
            input_shape=input_shape,
            port_semantic=port_semantic,
            mapping_version=mapping_version,
            slot_bindings=slot_bindings,
            selected_slots=result.selected_slots,
            minimum_true=result.minimum_true,
            validation_signature=ValidationSignature(
                dataset_digest=problem.dataset_digest(),
                example_count=(
                    len(problem.positive_examples) + len(problem.negative_examples)
                ),
                mismatch_count=result.mismatch_count,
            ),
            restoration_handle=restoration_handle,
        )


_FEATURE_TEMPLATE_RESULT_PATTERN = re.compile(
    r"^PTM_RESULT v2 feature_template candidate=(?P<candidate>[0-9]+) "
    r"mismatches=0$",
    re.MULTILINE,
)
_TA_CLAUSE_RESULT_PATTERN = re.compile(
    r"^PTM_RESULT v2 ta_clause literals=(?P<literals>[0-9,]+) "
    r"mismatches=0$",
    re.MULTILINE,
)
_DECISION_TREE_RESULT_PATTERN = re.compile(
    r"^PTM_RESULT v2 decision_tree nodes=(?P<nodes>[0-9]+) "
    r"depth=(?P<depth>[0-9]+) tree=(?P<tree>[0-9,]+) mismatches=0$",
    re.MULTILINE,
)


def _structure_no_solution_pattern(kind: str) -> re.Pattern[str]:
    return re.compile(
        rf"^PTM_RESULT v2 no_solution kind={re.escape(kind)}$", re.MULTILINE
    )


def _normalized_parameters(parameters: Mapping[str, object]) -> str:
    try:
        return json.dumps(
            dict(parameters),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("template parameters must be finite JSON values") from error


@dataclass(frozen=True, slots=True)
class FeatureTemplateCandidate:
    """One typed, registry-backed template proposal owned by Python."""

    field_name: str
    template_id: str
    data_type: DataType
    parameters_json: str = "{}"

    @classmethod
    def create(
        cls,
        *,
        field_name: str,
        template_id: str,
        data_type: DataType | str,
        parameters: Mapping[str, object] | None = None,
        registry: TemplateRegistry | None = None,
    ) -> "FeatureTemplateCandidate":
        if not field_name.strip():
            raise ValueError("feature-template field name cannot be empty")
        resolved_type = DataType(data_type)
        template = (registry or TemplateRegistry()).get(template_id)
        if template.data_type is not resolved_type:
            raise ValueError(
                f"template {template_id} has type {template.data_type.value}, not "
                f"{resolved_type.value}"
            )
        return cls(
            field_name.strip(),
            template_id,
            resolved_type,
            _normalized_parameters(parameters or {}),
        )

    @property
    def parameters(self) -> dict[str, object]:
        value = json.loads(self.parameters_json)
        if not isinstance(value, dict):
            raise ValueError("feature-template parameters are not an object")
        return value

    def template_spec(self) -> dict[str, object]:
        return {"template_id": self.template_id, **self.parameters}


@dataclass(frozen=True, slots=True)
class FeatureTemplateSearchProblem:
    """Finite exact selection over typed template candidates and their coverage."""

    candidates: tuple[FeatureTemplateCandidate, ...]
    labels: tuple[bool, ...]
    coverage: tuple[tuple[bool, ...], ...]

    @classmethod
    def create(
        cls,
        *,
        candidates: Iterable[FeatureTemplateCandidate],
        labels: Iterable[bool | int],
        coverage: Iterable[Iterable[bool | int]],
    ) -> "FeatureTemplateSearchProblem":
        candidate_tuple = tuple(candidates)
        label_tuple = tuple(_require_binary(value) for value in labels)
        coverage_tuple = tuple(
            tuple(_require_binary(value) for value in candidate) for candidate in coverage
        )
        if not 1 <= len(candidate_tuple) <= 4096:
            raise ValueError("typed template search accepts 1 through 4096 candidates")
        if not 2 <= len(label_tuple) <= 4096 or len(set(label_tuple)) != 2:
            raise ValueError("typed template search requires both output classes")
        if len(coverage_tuple) != len(candidate_tuple):
            raise ValueError("template candidates and coverage rows differ in length")
        if any(len(row) != len(label_tuple) for row in coverage_tuple):
            raise ValueError("each template coverage row must match the example count")
        for candidate in candidate_tuple:
            FeatureTemplateCandidate.create(
                field_name=candidate.field_name,
                template_id=candidate.template_id,
                data_type=candidate.data_type,
                parameters=candidate.parameters,
            )
        return cls(candidate_tuple, label_tuple, coverage_tuple)

    def dataset_digest(self) -> str:
        payload = {
            "schema": "ptm-feature-template-problem-v1",
            "candidates": [
                {
                    "field_name": item.field_name,
                    "template_id": item.template_id,
                    "data_type": item.data_type.value,
                    "parameters": item.parameters,
                }
                for item in self.candidates
            ],
            "labels": self.labels,
            "coverage": self.coverage,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureTemplateSearchResult:
    candidate_index: int
    candidate: FeatureTemplateCandidate
    dataset_digest: str
    mismatch_count: int = 0

    def create_catalog(self, schema):
        """Instantiate the selected typed template through the existing registry."""

        return create_feature_template_catalog(
            schema,
            {self.candidate.field_name: self.candidate.template_spec()},
        )


def _labeled_examples(
    slot_count: int,
    examples: Iterable[Iterable[int]],
    labels: Iterable[bool | int],
    *,
    require_both_classes: bool,
) -> tuple[tuple[tuple[int, ...], ...], tuple[bool, ...]]:
    rows = tuple(_normalized_example(example, slot_count) for example in examples)
    outputs = tuple(_require_binary(value) for value in labels)
    if not rows or len(rows) != len(outputs):
        raise ValueError("examples and labels must be nonempty and equal in length")
    if len(rows) > 4096:
        raise ValueError("bounded structure search accepts at most 4096 examples")
    if require_both_classes and len(set(outputs)) != 2:
        raise ValueError("exact TA-clause search requires both output classes")
    seen: dict[tuple[int, ...], bool] = {}
    for row, output in zip(rows, outputs):
        previous = seen.get(row)
        if previous is not None and previous is not output:
            raise ValueError("the same Boolean row has contradictory labels")
        seen[row] = output
    return rows, outputs


@dataclass(frozen=True, slots=True)
class TAClauseSearchProblem:
    """Bounded exact conjunction over positive and negated feature literals."""

    feature_count: int
    max_literals: int
    examples: tuple[tuple[int, ...], ...]
    labels: tuple[bool, ...]
    candidate_upper_bound: int

    @classmethod
    def create(
        cls,
        *,
        feature_count: int,
        max_literals: int,
        examples: Iterable[Iterable[int]],
        labels: Iterable[bool | int],
    ) -> "TAClauseSearchProblem":
        if not 1 <= feature_count <= 2048:
            raise ValueError("feature_count must be between 1 and 2048")
        if not 1 <= max_literals <= min(feature_count, 16):
            raise ValueError(
                "max_literals must be between 1 and min(feature_count, 16)"
            )
        candidate_upper_bound = sum(
            math.comb(feature_count * 2, width)
            for width in range(1, max_literals + 1)
        )
        if candidate_upper_bound > MAX_SEARCH_CANDIDATES:
            raise ValueError(
                f"TA-clause bound admits at most {candidate_upper_bound} candidates; "
                f"maximum is {MAX_SEARCH_CANDIDATES}"
            )
        rows, outputs = _labeled_examples(
            feature_count,
            examples,
            labels,
            require_both_classes=True,
        )
        return cls(
            feature_count,
            max_literals,
            rows,
            outputs,
            candidate_upper_bound,
        )

    @property
    def positive_examples(self) -> tuple[tuple[int, ...], ...]:
        return tuple(row for row, label in zip(self.examples, self.labels) if label)

    @property
    def negative_examples(self) -> tuple[tuple[int, ...], ...]:
        return tuple(row for row, label in zip(self.examples, self.labels) if not label)

    def dataset_digest(self) -> str:
        payload = {
            "schema": "ptm-ta-clause-problem-v1",
            "feature_count": self.feature_count,
            "examples": self.examples,
            "labels": self.labels,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TAClauseSearchResult:
    included_literals: tuple[int, ...]
    mismatch_count: int = 0

    def matches(self, example: Iterable[int]) -> bool:
        active = frozenset(example)
        return all(
            ((literal // 2) in active) == (literal % 2 == 0)
            for literal in self.included_literals
        )

    def to_clause_configuration(
        self,
        problem: TAClauseSearchProblem,
        *,
        clause_index: int = 0,
        polarity: int = 1,
    ) -> ClauseConfiguration:
        if polarity not in (-1, 1):
            raise ValueError("clause polarity must be +1 or -1")
        included = frozenset(self.included_literals)
        return ClauseConfiguration(
            clause_index=clause_index,
            included_literals=self.included_literals,
            excluded_literals=tuple(
                literal
                for literal in range(problem.feature_count * 2)
                if literal not in included
            ),
            polarity=polarity,
            activation_count=sum(self.matches(row) for row in problem.examples),
            avg_activation_rate=(
                sum(self.matches(row) for row in problem.examples)
                / len(problem.examples)
            ),
            contribution_score=0.0,
        )

    def to_ta_configuration(
        self,
        problem: TAClauseSearchProblem,
        *,
        states_per_action: int,
        specificity: float,
        threshold: int,
        clause_index: int = 0,
        polarity: int = 1,
    ) -> TAClauseConfiguration:
        if states_per_action <= 0 or specificity <= 1 or threshold <= 0:
            raise ValueError("TA configuration hyperparameters are invalid")
        if clause_index != 0:
            raise ValueError("a standalone TA configuration must use clause index zero")
        return TAClauseConfiguration(
            number_of_clauses=1,
            number_of_features=problem.feature_count,
            states_per_action=states_per_action,
            specificity=specificity,
            threshold=threshold,
            clause_configs=(
                self.to_clause_configuration(
                    problem,
                    clause_index=clause_index,
                    polarity=polarity,
                ),
            ),
            final_accuracy=1.0,
            metadata={
                "analysis_type": "prolog_ta_clause_configuration",
                "version": 1,
                "dataset_digest": problem.dataset_digest(),
                "candidate_upper_bound": problem.candidate_upper_bound,
                "mismatch_count": self.mismatch_count,
            },
        )


def _decision_tree_candidate_bound(slot_count: int, max_depth: int) -> int:
    def count(remaining_slots: int, depth: int) -> int:
        if depth == 0 or remaining_slots == 0:
            return 2
        child = count(remaining_slots - 1, depth - 1)
        total = 2 + remaining_slots * child * child
        return min(total, MAX_SEARCH_CANDIDATES + 1)

    return count(slot_count, max_depth)


@dataclass(frozen=True, slots=True)
class DecisionTreeSearchProblem:
    slot_count: int
    max_depth: int
    examples: tuple[tuple[int, ...], ...]
    labels: tuple[bool, ...]
    candidate_upper_bound: int

    @classmethod
    def create(
        cls,
        *,
        slot_count: int,
        max_depth: int,
        examples: Iterable[Iterable[int]],
        labels: Iterable[bool | int],
    ) -> "DecisionTreeSearchProblem":
        if not 1 <= slot_count <= 4096:
            raise ValueError("slot_count must be between 1 and 4096")
        if not 0 <= max_depth <= min(slot_count, 8):
            raise ValueError("max_depth must be between 0 and min(slot_count, 8)")
        candidate_upper_bound = _decision_tree_candidate_bound(slot_count, max_depth)
        if candidate_upper_bound > MAX_SEARCH_CANDIDATES:
            raise ValueError(
                f"decision-tree bound admits more than {MAX_SEARCH_CANDIDATES} "
                "candidates"
            )
        rows, outputs = _labeled_examples(
            slot_count,
            examples,
            labels,
            require_both_classes=False,
        )
        return cls(slot_count, max_depth, rows, outputs, candidate_upper_bound)

    def dataset_digest(self) -> str:
        payload = {
            "schema": "ptm-decision-tree-problem-v1",
            "slot_count": self.slot_count,
            "max_depth": self.max_depth,
            "examples": self.examples,
            "labels": self.labels,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BooleanDecisionTree:
    """Immutable Boolean tree; false and true branches are explicit."""

    feature: int | None = None
    value: bool | None = None
    false_branch: "BooleanDecisionTree | None" = None
    true_branch: "BooleanDecisionTree | None" = None

    def __post_init__(self) -> None:
        is_leaf = self.value is not None
        if is_leaf:
            if self.feature is not None or self.false_branch or self.true_branch:
                raise ValueError("decision-tree leaf cannot contain branches")
        elif (
            self.feature is None
            or self.feature < 0
            or self.false_branch is None
            or self.true_branch is None
        ):
            raise ValueError("decision-tree node requires a feature and two branches")

    @classmethod
    def leaf(cls, value: bool | int) -> "BooleanDecisionTree":
        return cls(value=_require_binary(value))

    @classmethod
    def node(
        cls,
        feature: int,
        false_branch: "BooleanDecisionTree",
        true_branch: "BooleanDecisionTree",
    ) -> "BooleanDecisionTree":
        return cls(feature, None, false_branch, true_branch)

    @property
    def is_leaf(self) -> bool:
        return self.value is not None

    @property
    def node_count(self) -> int:
        if self.is_leaf:
            return 1
        assert self.false_branch is not None and self.true_branch is not None
        return 1 + self.false_branch.node_count + self.true_branch.node_count

    @property
    def depth(self) -> int:
        if self.is_leaf:
            return 0
        assert self.false_branch is not None and self.true_branch is not None
        return 1 + max(self.false_branch.depth, self.true_branch.depth)

    @property
    def feature_indices(self) -> tuple[int, ...]:
        if self.is_leaf:
            return ()
        assert self.feature is not None
        assert self.false_branch is not None and self.true_branch is not None
        return (
            self.feature,
            *self.false_branch.feature_indices,
            *self.true_branch.feature_indices,
        )

    def is_read_once(self, used: frozenset[int] = frozenset()) -> bool:
        if self.is_leaf:
            return True
        assert self.feature is not None
        assert self.false_branch is not None and self.true_branch is not None
        if self.feature in used:
            return False
        child_used = used | {self.feature}
        return self.false_branch.is_read_once(
            child_used
        ) and self.true_branch.is_read_once(child_used)

    def evaluate(self, example: Iterable[int]) -> bool:
        active = example if isinstance(example, (set, frozenset)) else frozenset(example)
        current = self
        while not current.is_leaf:
            assert current.feature is not None
            assert current.false_branch is not None and current.true_branch is not None
            current = (
                current.true_branch
                if current.feature in active
                else current.false_branch
            )
        return bool(current.value)

    def prefix_encoding(self) -> tuple[int, ...]:
        if self.is_leaf:
            return (int(bool(self.value)),)
        assert self.feature is not None
        assert self.false_branch is not None and self.true_branch is not None
        return (
            2,
            self.feature,
            *self.false_branch.prefix_encoding(),
            *self.true_branch.prefix_encoding(),
        )

    def to_dict(self) -> dict[str, object]:
        if self.is_leaf:
            return {"leaf": bool(self.value)}
        assert self.feature is not None
        assert self.false_branch is not None and self.true_branch is not None
        return {
            "feature": self.feature,
            "false": self.false_branch.to_dict(),
            "true": self.true_branch.to_dict(),
        }

    def to_logic_program(self) -> LogicProgram32:
        builder = _DecisionLogicBuilder()
        return builder.finish(builder.add_tree(self))


def _decode_decision_tree(values: Sequence[int]) -> BooleanDecisionTree:
    def read(offset: int) -> tuple[BooleanDecisionTree, int]:
        if offset >= len(values):
            raise PrologBridgeError("decision-tree result is truncated")
        tag = values[offset]
        if tag in (0, 1):
            return BooleanDecisionTree.leaf(bool(tag)), offset + 1
        if tag != 2 or offset + 1 >= len(values):
            raise PrologBridgeError("decision-tree result has an invalid prefix tag")
        feature = values[offset + 1]
        false_branch, next_offset = read(offset + 2)
        true_branch, final_offset = read(next_offset)
        return BooleanDecisionTree.node(feature, false_branch, true_branch), final_offset

    tree, consumed = read(0)
    if consumed != len(values):
        raise PrologBridgeError("decision-tree result has trailing prefix values")
    return tree


class _DecisionLogicBuilder:
    def __init__(self) -> None:
        self.nodes: list[PrimitiveLogicNode] = []
        self._interned: dict[tuple[object, ...], int] = {}

    def _intern(
        self,
        operation: PrimitiveLogicOp,
        operands: Sequence[int] = (),
        *,
        variable: str | None = None,
        constant: bool | None = None,
    ) -> int:
        operand_tuple = tuple(operands)
        key = (operation, operand_tuple, variable, constant)
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
        return self._intern(PrimitiveLogicOp.CONSTANT, constant=bool(value))

    def input(self, feature: int) -> int:
        if not 0 <= feature < len(LOGIC_AST_VARIABLES):
            raise ValueError("fixed Logic lowering supports only feature slots 0 through 4")
        return self._intern(PrimitiveLogicOp.INPUT, variable=LOGIC_AST_VARIABLES[feature])

    def negate(self, operand: int) -> int:
        node = self.nodes[operand]
        if node.operation is PrimitiveLogicOp.CONSTANT:
            return self.constant(not bool(node.constant))
        if node.operation is PrimitiveLogicOp.NOT:
            return node.operands[0]
        return self._intern(PrimitiveLogicOp.NOT, (operand,))

    def all(self, operands: Sequence[int]) -> int:
        reduced = []
        for operand in dict.fromkeys(operands):
            node = self.nodes[operand]
            if node.operation is PrimitiveLogicOp.CONSTANT:
                if not node.constant:
                    return self.constant(False)
            else:
                reduced.append(operand)
        if not reduced:
            return self.constant(True)
        if len(reduced) == 1:
            return reduced[0]
        return self._intern(PrimitiveLogicOp.AND, tuple(sorted(reduced)))

    def any(self, operands: Sequence[int]) -> int:
        reduced = []
        for operand in dict.fromkeys(operands):
            node = self.nodes[operand]
            if node.operation is PrimitiveLogicOp.CONSTANT:
                if node.constant:
                    return self.constant(True)
            else:
                reduced.append(operand)
        if not reduced:
            return self.constant(False)
        if len(reduced) == 1:
            return reduced[0]
        return self._intern(PrimitiveLogicOp.OR, tuple(sorted(reduced)))

    def parity(self, operands: Sequence[int]) -> int:
        if len(operands) == 1:
            return operands[0]
        return self._intern(PrimitiveLogicOp.XOR, tuple(sorted(operands)))

    def conditional(self, condition: int, when_false: int, when_true: int) -> int:
        if when_false == when_true:
            return when_true
        false_node = self.nodes[when_false]
        true_node = self.nodes[when_true]
        if (
            false_node.operation is PrimitiveLogicOp.CONSTANT
            and true_node.operation is PrimitiveLogicOp.CONSTANT
        ):
            if not false_node.constant and true_node.constant:
                return condition
            if false_node.constant and not true_node.constant:
                return self.negate(condition)
        return self.any(
            (
                self.all((self.negate(condition), when_false)),
                self.all((condition, when_true)),
            )
        )

    def add_tree(self, tree: BooleanDecisionTree) -> int:
        if tree.is_leaf:
            return self.constant(bool(tree.value))
        assert tree.feature is not None
        assert tree.false_branch is not None and tree.true_branch is not None
        return self.conditional(
            self.input(tree.feature),
            self.add_tree(tree.false_branch),
            self.add_tree(tree.true_branch),
        )

    def finish(self, root: int) -> LogicProgram32:
        return LogicProgram32.compile(PrimitiveLogicGraph(tuple(self.nodes), root))


@dataclass(frozen=True, slots=True)
class DecisionTreeSearchResult:
    tree: BooleanDecisionTree
    mismatch_count: int = 0


@dataclass(frozen=True, slots=True)
class RepairCounterexample:
    example: tuple[int, ...]
    expected: bool
    parent_prediction: bool
    required_flip: bool


@dataclass(frozen=True, slots=True)
class DecisionTreeRepairResult:
    parent: BooleanDecisionTree
    guard: BooleanDecisionTree
    counterexamples: tuple[RepairCounterexample, ...]
    mismatches_before: int
    mismatch_count: int

    def evaluate(self, example: Iterable[int]) -> bool:
        active = frozenset(example)
        return self.parent.evaluate(active) ^ self.guard.evaluate(active)

    def to_logic_program(self) -> LogicProgram32:
        builder = _DecisionLogicBuilder()
        parent = builder.add_tree(self.parent)
        guard = builder.add_tree(self.guard)
        return builder.finish(builder.parity((parent, guard)))


class GNUPrologSearch(GNUPrologThresholdSearch):
    """All bounded Class III templates, including the v1 threshold search."""

    def __init__(
        self,
        executable: str | os.PathLike[str] | None = None,
        source_file: str | os.PathLike[str] | None = None,
        structure_source_file: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(executable=executable, source_file=source_file)
        default_structure = _default_prolog_source("bounded_structure_search.pl")
        self.structure_source_file = (
            Path(structure_source_file).resolve()
            if structure_source_file is not None
            else default_structure
        )
        if not self.structure_source_file.is_file():
            raise PrologBridgeError(
                f"Prolog structure-search source not found: {self.structure_source_file}"
            )

    def _run_structure(
        self,
        problem_fact: str,
        goal: str,
        *,
        timeout_seconds: float,
        cancel: Callable[[], bool] | None,
    ) -> str:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if goal not in {
            "ptm_run_feature_template_problem",
            "ptm_run_ta_clause_problem",
            "ptm_run_decision_tree_problem",
        }:
            raise ValueError("unsupported bounded structure-search goal")
        source = self.structure_source_file.as_posix().replace("'", "''")
        driver_source = f":- include('{source}').\n{problem_fact}\n"
        with tempfile.TemporaryDirectory(prefix="ptm-prolog-structure-") as directory:
            driver = Path(directory) / "problem_driver.pl"
            driver.write_text(driver_source, encoding="utf-8")
            completed = _run_prolog_process(
                [
                    str(self.executable),
                    "--consult-file",
                    str(driver),
                    "--query-goal",
                    goal,
                ],
                timeout_seconds=timeout_seconds,
                cancel=cancel,
            )
        return completed.stdout + "\n" + completed.stderr

    def search_feature_template(
        self,
        problem: FeatureTemplateSearchProblem,
        *,
        timeout_seconds: float = 30.0,
        cancel: Callable[[], bool] | None = None,
    ) -> FeatureTemplateSearchResult:
        positives = tuple(index for index, label in enumerate(problem.labels) if label)
        negatives = tuple(index for index, label in enumerate(problem.labels) if not label)
        coverages = tuple(
            tuple(index for index, matched in enumerate(row) if matched)
            for row in problem.coverage
        )
        fact = (
            f"feature_template_problem({_prolog_list(positives)},"
            f"{_prolog_list(negatives)},{_prolog_examples(coverages)})."
        )
        output = self._run_structure(
            fact,
            "ptm_run_feature_template_problem",
            timeout_seconds=timeout_seconds,
            cancel=cancel,
        )
        match = _FEATURE_TEMPLATE_RESULT_PATTERN.search(output)
        if match is not None:
            candidate_index = int(match.group("candidate"))
            if not 0 <= candidate_index < len(problem.candidates):
                raise PrologBridgeError("Prolog returned an out-of-range template index")
            if tuple(problem.coverage[candidate_index]) != problem.labels:
                raise PrologBridgeError("Prolog template result failed Python validation")
            return FeatureTemplateSearchResult(
                candidate_index,
                problem.candidates[candidate_index],
                problem.dataset_digest(),
            )
        if _structure_no_solution_pattern("feature_template").search(output):
            raise NoFeatureTemplateSolution("no exact typed feature template exists")
        raise PrologBridgeError(
            "GNU Prolog did not return the feature-template protocol\n"
            f"{output[-2000:]}"
        )

    def search_ta_clause(
        self,
        problem: TAClauseSearchProblem,
        *,
        timeout_seconds: float = 30.0,
        cancel: Callable[[], bool] | None = None,
    ) -> TAClauseSearchResult:
        fact = (
            f"ta_clause_problem({problem.feature_count},{problem.max_literals},"
            f"{_prolog_examples(problem.positive_examples)},"
            f"{_prolog_examples(problem.negative_examples)})."
        )
        output = self._run_structure(
            fact,
            "ptm_run_ta_clause_problem",
            timeout_seconds=timeout_seconds,
            cancel=cancel,
        )
        match = _TA_CLAUSE_RESULT_PATTERN.search(output)
        if match is not None:
            literals = tuple(int(value) for value in match.group("literals").split(","))
            result = TAClauseSearchResult(literals)
            if (
                len(literals) > problem.max_literals
                or len(set(literals)) != len(literals)
                or any(not 0 <= literal < problem.feature_count * 2 for literal in literals)
                or any(
                    result.matches(row) != label
                    for row, label in zip(problem.examples, problem.labels)
                )
            ):
                raise PrologBridgeError("Prolog TA-clause result failed Python validation")
            return result
        if _structure_no_solution_pattern("ta_clause").search(output):
            raise NoTAClauseSolution("no exact bounded TA clause exists")
        raise PrologBridgeError(
            "GNU Prolog did not return the TA-clause protocol\n" f"{output[-2000:]}"
        )

    def search_decision_tree(
        self,
        problem: DecisionTreeSearchProblem,
        *,
        timeout_seconds: float = 30.0,
        cancel: Callable[[], bool] | None = None,
    ) -> DecisionTreeSearchResult:
        labels = tuple(int(value) for value in problem.labels)
        fact = (
            f"decision_tree_problem({problem.slot_count},{problem.max_depth},"
            f"{_prolog_examples(problem.examples)},{_prolog_list(labels)})."
        )
        output = self._run_structure(
            fact,
            "ptm_run_decision_tree_problem",
            timeout_seconds=timeout_seconds,
            cancel=cancel,
        )
        match = _DECISION_TREE_RESULT_PATTERN.search(output)
        if match is not None:
            tree = _decode_decision_tree(
                tuple(int(value) for value in match.group("tree").split(","))
            )
            reported_nodes = int(match.group("nodes"))
            reported_depth = int(match.group("depth"))
            if (
                tree.node_count != reported_nodes
                or tree.depth != reported_depth
                or tree.depth > problem.max_depth
                or not tree.is_read_once()
                or any(feature >= problem.slot_count for feature in tree.feature_indices)
                or any(
                    tree.evaluate(row) != label
                    for row, label in zip(problem.examples, problem.labels)
                )
            ):
                raise PrologBridgeError(
                    "Prolog decision-tree result failed Python validation"
                )
            return DecisionTreeSearchResult(tree)
        if _structure_no_solution_pattern("decision_tree").search(output):
            raise NoDecisionTreeSolution("no exact bounded decision tree exists")
        raise PrologBridgeError(
            "GNU Prolog did not return the decision-tree protocol\n"
            f"{output[-2000:]}"
        )

    def repair_decision_tree(
        self,
        parent: BooleanDecisionTree,
        problem: DecisionTreeSearchProblem,
        *,
        max_iterations: int = 32,
        timeout_seconds: float = 30.0,
        cancel: Callable[[], bool] | None = None,
    ) -> DecisionTreeRepairResult:
        if not 1 <= max_iterations <= 256:
            raise ValueError("max_iterations must be between 1 and 256")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        if not parent.is_read_once() or any(
            feature >= problem.slot_count for feature in parent.feature_indices
        ):
            raise ValueError("repair parent is incompatible with the problem slot domain")
        deadline = time.monotonic() + float(timeout_seconds)

        def remaining_time() -> float:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PrologBridgeError(
                    f"bounded Prolog repair timed out after {timeout_seconds:g}s"
                )
            return remaining

        iteration_limit = min(max_iterations, len(problem.examples))
        parent_predictions = tuple(parent.evaluate(row) for row in problem.examples)
        mismatches_before = sum(
            prediction != label
            for prediction, label in zip(parent_predictions, problem.labels)
        )
        guard = BooleanDecisionTree.leaf(False)
        counterexamples: list[RepairCounterexample] = []
        constrained_indices: list[int] = []
        for _ in range(iteration_limit):
            if cancel is not None and cancel():
                raise PrologSearchCancelled("bounded Prolog repair cancelled")
            remaining_time()
            mismatch = next(
                (
                    index
                    for index, (row, expected) in enumerate(
                        zip(problem.examples, problem.labels)
                    )
                    if (parent.evaluate(row) ^ guard.evaluate(row)) != expected
                ),
                None,
            )
            if mismatch is None:
                remaining_time()
                return DecisionTreeRepairResult(
                    parent,
                    guard,
                    tuple(counterexamples),
                    mismatches_before,
                    0,
                )
            if mismatch not in constrained_indices:
                constrained_indices.append(mismatch)
            row = problem.examples[mismatch]
            expected = problem.labels[mismatch]
            parent_prediction = parent_predictions[mismatch]
            counterexamples.append(
                RepairCounterexample(
                    row,
                    expected,
                    parent_prediction,
                    parent_prediction != expected,
                )
            )
            repair_problem = DecisionTreeSearchProblem.create(
                slot_count=problem.slot_count,
                max_depth=problem.max_depth,
                examples=(problem.examples[index] for index in constrained_indices),
                labels=(
                    parent_predictions[index] != problem.labels[index]
                    for index in constrained_indices
                ),
            )
            remaining_seconds = remaining_time()
            try:
                guard = self.search_decision_tree(
                    repair_problem,
                    timeout_seconds=remaining_seconds,
                    cancel=cancel,
                ).tree
            except NoDecisionTreeSolution as error:
                raise RepairDidNotConverge(
                    "no bounded repair guard satisfies the accumulated counterexamples"
                ) from error
        remaining = sum(
            (parent.evaluate(row) ^ guard.evaluate(row)) != expected
            for row, expected in zip(problem.examples, problem.labels)
        )
        if remaining == 0:
            remaining_time()
            return DecisionTreeRepairResult(
                parent,
                guard,
                tuple(counterexamples),
                mismatches_before,
                0,
            )
        remaining_time()
        raise RepairDidNotConverge(
            f"counterexample repair stopped after {iteration_limit} iterations with "
            f"{remaining} mismatches"
        )
