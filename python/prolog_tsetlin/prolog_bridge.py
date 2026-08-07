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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .artifact import (
    InputShape,
    PAArtifact,
    RestorationHandle,
    SlotBinding,
    ValidationSignature,
)
from .pa import PortSemantic


_RESULT_PATTERN = re.compile(
    r"^PTM_RESULT v1 masked_threshold selected=(?P<slots>[0-9,]+) "
    r"minimum=(?P<minimum>[0-9]+) mismatches=(?P<mismatches>[0-9]+)$",
    re.MULTILINE,
)
_NO_SOLUTION_PATTERN = re.compile(r"^PTM_RESULT v1 no_solution$", re.MULTILINE)
MAX_SEARCH_CANDIDATES = 1_000_000


class PrologBridgeError(RuntimeError):
    pass


class NoThresholdSolution(PrologBridgeError):
    pass


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
        Path(r"C:\GNU-Prolog\bin\gprolog.exe") if os.name == "nt" else None,
    ]
    return next((path for path in candidates if path is not None and path.is_file()), None)


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
        default_source = Path(__file__).resolve().parents[2] / "prolog" / "bounded_threshold_search.pl"
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
            try:
                completed = subprocess.run(
                    [
                        str(self.executable),
                        "--consult-file",
                        str(driver),
                        "--query-goal",
                        "ptm_run_problem",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_seconds,
                    creationflags=(
                        subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                    ),
                )
            except subprocess.TimeoutExpired as exc:
                partial_stdout = exc.stdout or ""
                partial_stderr = exc.stderr or ""
                if isinstance(partial_stdout, bytes):
                    partial_stdout = partial_stdout.decode("utf-8", errors="replace")
                if isinstance(partial_stderr, bytes):
                    partial_stderr = partial_stderr.decode("utf-8", errors="replace")
                partial = (partial_stdout + "\n" + partial_stderr)[-2000:]
                raise PrologBridgeError(
                    f"bounded Prolog search timed out after {timeout_seconds:g}s\n"
                    f"{partial}"
                ) from exc
        output = completed.stdout + "\n" + completed.stderr
        match = _RESULT_PATTERN.search(output)
        if match is not None:
            selected = tuple(int(slot) for slot in match.group("slots").split(","))
            return ThresholdSearchResult(
                selected_slots=selected,
                minimum_true=int(match.group("minimum")),
                mismatch_count=int(match.group("mismatches")),
            )
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
    ) -> PAArtifact:
        if problem.slot_count > input_shape.bit_count:
            raise ValueError("search problem exceeds the target PA input shape")
        result = self.search(problem, timeout_seconds=timeout_seconds)
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
