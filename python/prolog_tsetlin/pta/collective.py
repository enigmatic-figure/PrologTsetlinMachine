"""Typed, bounded execution service for the GNU Prolog PTA collective."""

from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

from .._bounded_process import (
    BoundedProcessDrainError,
    BoundedProcessLaunchError,
    BoundedProcessOutputLimit,
    BoundedProcessTimeout,
    run_bounded_process,
)
from ..prolog_resources import (
    PrologResourceError,
    prolog_process_environment,
    resolve_gprolog,
    resolve_prolog_module_set,
)
from .proposal import PTAEscalationProposal, PTAInsight
from .session import (
    EXACT_NUMERIC_MAGNITUDE,
    MAX_RELATION_ITEMS,
    PTAReasoningSession,
)


PROTOCOL_BEGIN = "PTM_PTA_COLLECTIVE_V1_BEGIN"
PROTOCOL_END = "PTM_PTA_COLLECTIVE_V1_END"
MAX_PROTOCOL_BYTES = 1_048_576
MAX_FACT_BYTES = 16_777_216
_NUMBER = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
)
_INTEGER = re.compile(r"^[0-9]+$")


class PTACollectiveError(RuntimeError):
    """Base class for collective execution failures."""


class PTACollectiveUnavailable(PTACollectiveError):
    """GNU Prolog or a packaged collective module is unavailable."""


class PTACollectiveTimeout(PTACollectiveError):
    """The collective exceeded its wall-clock budget."""


class PTACollectiveExecutionError(PTACollectiveError):
    """GNU Prolog failed before producing a valid result."""


class PTACollectiveProtocolError(PTACollectiveError):
    """The child output did not satisfy the narrow interchange grammar."""


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _field_name(value: object) -> str:
    if type(value) is not str:
        raise TypeError("numeric_fields items must be strings")
    if not value or len(value) > 1_024 or any(ord(char) < 0x20 for char in value):
        raise ValueError("numeric_fields items must be nonempty printable strings")
    return value


@dataclass(frozen=True, slots=True)
class PTACollectiveQuery:
    """Select which bounded derivations the collective should perform."""

    numeric_fields: tuple[str, ...] | None = None
    discover_thresholds: bool = True
    discover_intervals: bool = True
    derive_deescalation: bool = True
    derive_escalation: bool = True

    def __post_init__(self) -> None:
        if self.numeric_fields is not None:
            if type(self.numeric_fields) is not tuple:
                raise TypeError("numeric_fields must be a tuple or None")
            checked = tuple(_field_name(field) for field in self.numeric_fields)
            if len(set(checked)) != len(checked):
                raise ValueError("numeric_fields must not contain duplicates")
            object.__setattr__(self, "numeric_fields", checked)
        for name in (
            "discover_thresholds",
            "discover_intervals",
            "derive_deescalation",
            "derive_escalation",
        ):
            _strict_bool(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class PTACollectiveBudget:
    """Wall-clock, output, result, and relation-cardinality budgets."""

    timeout_seconds: float | int = 10.0
    max_output_bytes: int = 262_144
    max_input_bytes: int = 1_048_576
    max_results_per_product: int = 64
    max_observations: int = 2_048
    max_examples: int = 2_048
    max_literals: int = 256
    max_clauses: int = 256
    max_classes: int = 256
    max_pair_candidates: int = 65_536
    max_facts: int = MAX_RELATION_ITEMS

    def __post_init__(self) -> None:
        if type(self.timeout_seconds) not in (int, float):
            raise TypeError("timeout_seconds must be an integer or float")
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 60:
            raise ValueError("timeout_seconds must be finite and in (0, 60]")
        limits = {
            "max_output_bytes": (self.max_output_bytes, 1_024, MAX_PROTOCOL_BYTES),
            "max_input_bytes": (self.max_input_bytes, 1_024, MAX_FACT_BYTES),
            "max_results_per_product": (
                self.max_results_per_product,
                1,
                1_024,
            ),
            "max_observations": (self.max_observations, 1, MAX_RELATION_ITEMS),
            "max_examples": (self.max_examples, 1, MAX_RELATION_ITEMS),
            "max_literals": (self.max_literals, 1, 4_096),
            "max_clauses": (self.max_clauses, 1, 4_096),
            "max_classes": (self.max_classes, 1, 4_096),
            "max_pair_candidates": (
                self.max_pair_candidates,
                1,
                1_000_000,
            ),
            "max_facts": (self.max_facts, 1, MAX_RELATION_ITEMS),
        }
        for name, (value, lower, upper) in limits.items():
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if not lower <= value <= upper:
                raise ValueError(f"{name} must be in {lower}..{upper}")


@dataclass(frozen=True, slots=True)
class PTACollectiveProductCount:
    """Completeness metadata for one typed result product."""

    emitted: int
    available: int

    def __post_init__(self) -> None:
        if type(self.emitted) is not int or type(self.available) is not int:
            raise TypeError("product counts must be integers")
        if self.emitted < 0 or self.available < self.emitted:
            raise ValueError("product counts must satisfy 0 <= emitted <= available")

    @property
    def truncated(self) -> bool:
        return self.emitted < self.available


_PRODUCT_KEYS = (
    "threshold_insights",
    "interval_insights",
    "literal_redundancies",
    "literal_subsumptions",
    "clause_subsumptions",
    "threshold_proposals",
    "weight_proposals",
)


@dataclass(frozen=True, slots=True)
class PTACollectiveResult:
    """Decoded collective products; raw child output is intentionally absent."""

    insights: tuple[PTAInsight, ...]
    proposals: tuple[PTAEscalationProposal, ...]
    field_ids: Mapping[str, int]
    product_counts: Mapping[str, PTACollectiveProductCount]
    elapsed_seconds: float
    protocol_version: int = 1

    def __post_init__(self) -> None:
        if type(self.insights) is not tuple:
            raise TypeError("insights must be a tuple")
        if type(self.proposals) is not tuple:
            raise TypeError("proposals must be a tuple")
        if any(not isinstance(item, PTAInsight) for item in self.insights):
            raise TypeError("insights must contain PTAInsight values")
        if any(not isinstance(item, PTAEscalationProposal) for item in self.proposals):
            raise TypeError("proposals must contain PTAEscalationProposal values")
        field_ids = dict(self.field_ids)
        if any(
            type(field) is not str or type(identifier) is not int
            for field, identifier in field_ids.items()
        ):
            raise TypeError("field_ids must map strings to integers")
        if any(identifier < 0 for identifier in field_ids.values()):
            raise ValueError("field_ids values must be nonnegative")
        if len(set(field_ids.values())) != len(field_ids):
            raise ValueError("field_ids values must be unique")
        object.__setattr__(self, "field_ids", MappingProxyType(field_ids))
        product_counts = dict(self.product_counts)
        if tuple(product_counts) != _PRODUCT_KEYS:
            raise ValueError("product_counts must contain every product in protocol order")
        if any(
            not isinstance(value, PTACollectiveProductCount)
            for value in product_counts.values()
        ):
            raise TypeError("product_counts values must be PTACollectiveProductCount")
        object.__setattr__(
            self, "product_counts", MappingProxyType(product_counts)
        )
        if (
            type(self.elapsed_seconds) is not float
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be a nonnegative float")
        if type(self.protocol_version) is not int or self.protocol_version != 1:
            raise ValueError("unsupported PTA collective protocol version")

    @property
    def truncated(self) -> bool:
        return any(count.truncated for count in self.product_counts.values())


_MODULE_NAMES = (
    "pta_ontology.pl",
    "pta_input.pl",
    "pta_deescalation.pl",
    "pta_escalation.pl",
)


def _quoted_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _driver_source(
    module_paths: Mapping[str, Path],
    facts_path: Path,
    selected_field_ids: tuple[int, ...],
    literal_ids: tuple[int, ...],
    clause_ids: tuple[int, ...],
    query: PTACollectiveQuery,
    budget: PTACollectiveBudget,
) -> str:
    includes = "".join(
        f":- include('{_quoted_path(module_paths[name])}').\n"
        for name in _MODULE_NAMES
    )
    includes += f":- include('{_quoted_path(facts_path)}').\n"
    fields = "[" + ",".join(str(value) for value in selected_field_ids) + "]"
    literals = "[" + ",".join(str(value) for value in literal_ids) + "]"
    clauses = "[" + ",".join(str(value) for value in clause_ids) + "]"
    threshold_goal = (
        f"findall(t(F,T),(member(F,{fields}),invent_threshold(F,T)),ThresholdRows)"
        if query.discover_thresholds
        else "ThresholdRows=[]"
    )
    interval_goal = (
        f"findall(i(F,L,H),(member(F,{fields}),invent_interval(F,L,H)),IntervalRows)"
        if query.discover_intervals
        else "IntervalRows=[]"
    )
    deescalation_goals = (
        f"findall(d(le,A,B),(member(A,{literals}),member(B,{literals}),"
        "A<B,literals_equivalent(A,B)),EquivalentRows),\n"
        f"    findall(d(ls,A,B),(member(A,{literals}),member(B,{literals}),"
        "A\\=B,literal_subsumes(A,B)),SubsumedRows),\n"
        f"    findall(d(cs,A,B),(member(A,{clauses}),member(B,{clauses}),"
        "A\\=B,clause_subsumes(A,B)),ClauseRows)"
        if query.derive_deescalation
        else "EquivalentRows=[], SubsumedRows=[], ClauseRows=[]"
    )
    escalation_goals = (
        f"findall(e(F,T),(member(F,{fields}),exception_clause(F,T,_)),ExceptionRows),\n"
        "    findall(w(C,K,W),cotm_weight(C,K,W),WeightRows)"
        if query.derive_escalation
        else "ExceptionRows=[], WeightRows=[]"
    )
    return (
        includes
        + "\n"
        + "ptm_take(0,_,[]) :- !.\n"
        + "ptm_take(_,[],[]) :- !.\n"
        + "ptm_take(N,[H|T],[H|R]) :- N>0, N1 is N-1, ptm_take(N1,T,R).\n"
        + "ptm_cap(Raw,N,Rows,Available) :- sort(Raw,Sorted), "
        + "length(Sorted,Available), ptm_take(N,Sorted,Rows).\n"
        + "ptm_emit(t(F,T)) :- write('T|'),write(F),write('|'),write(T),nl.\n"
        + "ptm_emit(i(F,L,H)) :- write('I|'),write(F),write('|'),write(L),"
        + "write('|'),write(H),nl.\n"
        + "ptm_emit(e(F,T)) :- write('E|'),write(F),write('|'),write(T),nl.\n"
        + "ptm_emit(d(le,A,B)) :- write('D|LE|'),write(A),write('|'),write(B),nl.\n"
        + "ptm_emit(d(ls,A,B)) :- write('D|LS|'),write(A),write('|'),write(B),nl.\n"
        + "ptm_emit(d(cs,A,B)) :- write('D|CS|'),write(A),write('|'),write(B),nl.\n"
        + "ptm_emit(w(C,K,W)) :- write('W|'),write(C),write('|'),write(K),"
        + "write('|'),write(W),nl.\n"
        + "ptm_emit_all([]).\n"
        + "ptm_emit_all([H|T]) :- ptm_emit(H), ptm_emit_all(T).\n"
        + "ptm_main :-\n"
        + f"    {threshold_goal},\n"
        + f"    {interval_goal},\n"
        + f"    {deescalation_goals},\n"
        + f"    {escalation_goals},\n"
        + f"    ptm_cap(ThresholdRows,{budget.max_results_per_product},T0,TC),\n"
        + f"    ptm_cap(IntervalRows,{budget.max_results_per_product},I0,IC),\n"
        + f"    ptm_cap(EquivalentRows,{budget.max_results_per_product},D0,LEC),\n"
        + f"    ptm_cap(SubsumedRows,{budget.max_results_per_product},D1,LSC),\n"
        + f"    ptm_cap(ClauseRows,{budget.max_results_per_product},D2,CSC),\n"
        + f"    ptm_cap(ExceptionRows,{budget.max_results_per_product},E0,EC),\n"
        + f"    ptm_cap(WeightRows,{budget.max_results_per_product},W0,WC),\n"
        + "    append(T0,I0,R1), append(R1,D0,R2), append(R2,D1,R3),\n"
        + "    append(R3,D2,R4), append(R4,E0,R5), append(R5,W0,All),\n"
        + "    length(All,Count), Available is TC+IC+LEC+LSC+CSC+EC+WC,\n"
        + f"    write('{PROTOCOL_BEGIN}'),nl,\n"
        + "    ptm_emit_all(All),\n"
        + f"    write('{PROTOCOL_END}|'),write(Count),write('|'),"
        + "write(Available),write('|'),write(TC),write('|'),write(IC),write('|'),"
        + "write(LEC),write('|'),write(LSC),write('|'),write(CSC),write('|'),"
        + "write(EC),write('|'),write(WC),nl,halt.\n"
    )


def _parse_number(token: str) -> float:
    if not _NUMBER.fullmatch(token):
        raise PTACollectiveProtocolError(f"invalid numeric token: {token!r}")
    value = float(token)
    if not math.isfinite(value):
        raise PTACollectiveProtocolError("collective returned a non-finite number")
    return value


def _parse_integer(token: str) -> int:
    if not _INTEGER.fullmatch(token):
        raise PTACollectiveProtocolError(f"invalid integer token: {token!r}")
    return int(token)


def _threshold_proposal(field: str, threshold: float) -> PTAEscalationProposal:
    insight = PTAInsight("pta:input", "threshold", field, (threshold,))
    threshold_text = format(threshold, ".17g")
    field_identity = hashlib.sha256(field.encode("utf-8")).hexdigest()
    return PTAEscalationProposal(
        proposal_id=f"pta:escalation:threshold:{field_identity}:{threshold_text}",
        source_pta_ids=("pta:input", "pta:escalation"),
        supporting_insights=(insight,),
        counterexamples_addressed=(),
        required_literals=(f"numeric_ge:{field}:{threshold_text}",),
        native_target="threshold",
        structure={"field": field, "operator": "ge", "threshold": threshold},
        resource_bounds={"literal_count": 1},
        support_trace=("derived by GNU Prolog exception_clause/3",),
    )


def _weight_proposal(clause: int, target_class: int, weight: int) -> PTAEscalationProposal:
    insight = PTAInsight(
        "pta:escalation", "cotm_weight", f"{clause}->{target_class}", (weight,)
    )
    return PTAEscalationProposal(
        proposal_id=f"pta:escalation:weight:{clause}:{target_class}:{weight}",
        source_pta_ids=("pta:escalation",),
        supporting_insights=(insight,),
        counterexamples_addressed=(),
        required_literals=(),
        native_target="shared_weighted_clause",
        structure={"clause": clause, "class": target_class, "weight": weight},
        weights=(weight,),
        output_assignments=((clause, target_class),),
        resource_bounds={"clause_count": 1},
        support_trace=("derived by GNU Prolog cotm_weight/3",),
    )


def _decode_protocol(
    stdout: bytes,
    *,
    id_to_field: Mapping[int, str],
    id_to_literal: Mapping[int, int],
    id_to_clause: Mapping[int, int],
    id_to_class: Mapping[int, int],
    max_results_per_product: int,
) -> tuple[
    tuple[PTAInsight, ...],
    tuple[PTAEscalationProposal, ...],
    Mapping[str, PTACollectiveProductCount],
]:
    try:
        text = stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PTACollectiveProtocolError("collective output was not UTF-8") from exc
    lines = [line.rstrip("\r") for line in text.splitlines()]
    starts = [index for index, line in enumerate(lines) if line == PROTOCOL_BEGIN]
    ends = [
        index
        for index, line in enumerate(lines)
        if line.startswith(PROTOCOL_END + "|")
    ]
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        raise PTACollectiveProtocolError("missing or ambiguous collective protocol frame")
    payload = lines[starts[0] + 1 : ends[0]]
    end_parts = lines[ends[0]].split("|")
    if len(end_parts) != 10:
        raise PTACollectiveProtocolError("invalid collective completeness record")
    declared = _parse_integer(end_parts[1])
    available_total = _parse_integer(end_parts[2])
    available_by_product = tuple(_parse_integer(token) for token in end_parts[3:])
    if declared != len(payload) or available_total != sum(available_by_product):
        raise PTACollectiveProtocolError("collective result count violated its budget")

    insights: list[PTAInsight] = []
    proposals: list[PTAEscalationProposal] = []
    emitted = {key: 0 for key in _PRODUCT_KEYS}
    for line in payload:
        parts = line.split("|")
        kind = parts[0] if parts else ""
        if kind in ("T", "E") and len(parts) == 3:
            field_id = _parse_integer(parts[1])
            if field_id not in id_to_field:
                raise PTACollectiveProtocolError("unknown observation field identifier")
            threshold = _parse_number(parts[2])
            field = id_to_field[field_id]
            if kind == "T":
                insights.append(PTAInsight("pta:input", "threshold", field, (threshold,)))
                emitted["threshold_insights"] += 1
            else:
                proposals.append(_threshold_proposal(field, threshold))
                emitted["threshold_proposals"] += 1
        elif kind == "I" and len(parts) == 4:
            field_id = _parse_integer(parts[1])
            if field_id not in id_to_field:
                raise PTACollectiveProtocolError("unknown observation field identifier")
            lower = _parse_number(parts[2])
            upper = _parse_number(parts[3])
            if lower > upper:
                raise PTACollectiveProtocolError("collective returned an inverted interval")
            insights.append(
                PTAInsight(
                    "pta:input",
                    "interval",
                    id_to_field[field_id],
                    (lower, upper),
                )
            )
            emitted["interval_insights"] += 1
        elif kind == "D" and len(parts) == 4 and parts[1] in ("LE", "LS", "CS"):
            left_opaque = _parse_integer(parts[2])
            right_opaque = _parse_integer(parts[3])
            identifiers = id_to_clause if parts[1] == "CS" else id_to_literal
            if left_opaque not in identifiers or right_opaque not in identifiers:
                raise PTACollectiveProtocolError("unknown de-escalation identifier")
            left = identifiers[left_opaque]
            right = identifiers[right_opaque]
            names = {
                "LE": "literal_redundant",
                "LS": "literal_subsumes",
                "CS": "clause_subsumes",
            }
            insights.append(
                PTAInsight(
                    "pta:deescalation",
                    names[parts[1]],
                    f"{left}->{right}",
                    (left, right),
                )
            )
            product = {
                "LE": "literal_redundancies",
                "LS": "literal_subsumptions",
                "CS": "clause_subsumptions",
            }[parts[1]]
            emitted[product] += 1
        elif kind == "W" and len(parts) == 4:
            clause_opaque = _parse_integer(parts[1])
            class_opaque = _parse_integer(parts[2])
            if clause_opaque not in id_to_clause or class_opaque not in id_to_class:
                raise PTACollectiveProtocolError("unknown CoTM identifier")
            clause = id_to_clause[clause_opaque]
            target_class = id_to_class[class_opaque]
            weight_token = parts[3]
            if not re.fullmatch(r"-?[0-9]+", weight_token):
                raise PTACollectiveProtocolError("CoTM weight must be an integer")
            proposals.append(_weight_proposal(clause, target_class, int(weight_token)))
            emitted["weight_proposals"] += 1
        else:
            raise PTACollectiveProtocolError(f"unknown collective record: {line!r}")
    if any(
        available_by_product[index] < emitted[key]
        for index, key in enumerate(_PRODUCT_KEYS)
    ):
        raise PTACollectiveProtocolError("collective completeness counts are invalid")
    product_counts = {
        key: PTACollectiveProductCount(
            emitted=emitted[key], available=available_by_product[index]
        )
        for index, key in enumerate(_PRODUCT_KEYS)
    }
    if any(count.emitted > max_results_per_product for count in product_counts.values()):
        raise PTACollectiveProtocolError("collective product exceeded its result budget")
    if sum(count.emitted for count in product_counts.values()) != declared:
        raise PTACollectiveProtocolError("collective product counts do not match payload")
    return tuple(insights), tuple(proposals), product_counts


def _run_bounded_process(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float | int,
    max_output_bytes: int,
) -> subprocess.CompletedProcess[bytes]:
    """Run the collective through PTM's shared process-tree boundary."""

    try:
        return run_bounded_process(
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            env=prolog_process_environment(),
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            ),
        )
    except BoundedProcessTimeout as exc:
        raise PTACollectiveTimeout(
            f"PTA collective timed out after {timeout_seconds:g}s"
        ) from exc
    except BoundedProcessOutputLimit as exc:
        raise PTACollectiveProtocolError(
            "collective output exceeded its byte budget"
        ) from exc
    except (BoundedProcessLaunchError, BoundedProcessDrainError) as exc:
        raise PTACollectiveExecutionError(
            f"GNU Prolog process boundary failed: {exc}"
        ) from exc


def _write_bounded_fact_lines(
    path: Path,
    lines: Iterable[str],
    *,
    max_bytes: int,
) -> int:
    """Write UTF-8 fact lines without materializing an unbounded document."""

    written = 0
    with path.open("wb") as destination:
        for line in lines:
            if type(line) is not str:
                raise TypeError("serialized PTA fact lines must be strings")
            encoded = (line + "\n").encode("utf-8")
            if written + len(encoded) > max_bytes:
                raise ValueError(
                    "serialized PTA facts exceed collective input byte budget"
                )
            destination.write(encoded)
            written += len(encoded)
    return written


class PTACollectiveService:
    """Execute all three PTA classes against one bounded reasoning session."""

    def __init__(
        self,
        executable: str | os.PathLike[str] | None = None,
        *,
        module_paths: Mapping[str, str | os.PathLike[str]] | None = None,
    ) -> None:
        try:
            self.executable = resolve_gprolog(executable)
            if module_paths is None:
                resolved = dict(resolve_prolog_module_set(_MODULE_NAMES))
            else:
                if set(module_paths) != set(_MODULE_NAMES):
                    raise ValueError(
                        "module_paths must contain ontology, input, de-escalation, "
                        "and escalation modules"
                    )
                resolved = {
                    name: Path(module_paths[name]).resolve() for name in _MODULE_NAMES
                }
                missing = [str(path) for path in resolved.values() if not path.is_file()]
                if missing:
                    raise PrologResourceError(
                        "PTA collective modules were not found: " + ", ".join(missing)
                    )
                roots = {path.parent for path in resolved.values()}
                if len(roots) != 1:
                    raise ValueError("module_paths must come from one coherent directory")
        except PrologResourceError as exc:
            raise PTACollectiveUnavailable(str(exc)) from exc
        self.module_paths = MappingProxyType(resolved)

    @staticmethod
    def _identifier_sets(
        session: PTAReasoningSession,
    ) -> tuple[set[int], set[int], set[int], set[int]]:
        examples = set(session.example_domains)
        examples.update(example for example, _ in session.example_labels)
        examples.update(example for _, example, _, _ in session.observations)
        examples.update(example for _, example, _ in session.literal_truths)
        examples.update(example for _, example, _ in session.clause_truths)
        examples.update(example for _, example in session.clause_supports)
        examples.update(example for _, example in session.clause_conflicts)
        examples.update(example for _, example, _, _ in session.counterexamples)
        literals = (
            {literal for literal, _, _ in session.literal_truths}
            | {literal for _, literal in session.clause_literals}
            | {literal for literal, _, _ in session.feature_supports}
            | {
                literal
                for left, _, right in session.feature_relations
                for literal in (left, right)
            }
        )
        clauses = (
            {clause for clause, _, _ in session.clause_truths}
            | {clause for clause, _ in session.clause_literals}
            | {clause for clause, _, _ in session.clause_class_scores}
            | {clause for clause, _ in session.clause_supports}
            | {clause for clause, _ in session.clause_conflicts}
        )
        classes = {target_class for target_class, _, _ in session.class_supports} | {
            target_class for _, target_class, _ in session.clause_class_scores
        }
        return examples, literals, clauses, classes

    @staticmethod
    def _validate_resource_budget(
        session: PTAReasoningSession, budget: PTACollectiveBudget
    ) -> None:
        session.validate()
        if len(session.observations) > budget.max_observations:
            raise ValueError("session observations exceed collective budget")
        example_ids, literal_ids, clause_ids, class_ids = (
            PTACollectiveService._identifier_sets(session)
        )
        if len(example_ids) > budget.max_examples:
            raise ValueError("session examples exceed collective budget")
        if len(literal_ids) > budget.max_literals:
            raise ValueError("session literals exceed collective budget")
        if len(clause_ids) > budget.max_clauses:
            raise ValueError("session clauses exceed collective budget")
        if len(class_ids) > budget.max_classes:
            raise ValueError("session classes exceed collective budget")
        fact_count = sum(
            len(values)
            for values in (
                session.observations,
                session.example_labels,
                session.example_domains,
                session.feature_supports,
                session.feature_relations,
                session.literal_truths,
                session.clause_truths,
                session.clause_literals,
                session.class_supports,
                session.clause_class_scores,
                session.clause_supports,
                session.clause_conflicts,
                session.counterexamples,
                session.insights,
                session.proposals,
            )
        )
        if fact_count > budget.max_facts:
            raise ValueError("session facts exceed collective budget")

    @staticmethod
    def _field_maps(
        session: PTAReasoningSession, query: PTACollectiveQuery
    ) -> tuple[dict[str, int], tuple[int, ...]]:
        field_values: dict[str, list[object]] = {}
        for _, _, field, value in session.observations:
            field_values.setdefault(field, []).append(value)
        all_fields = sorted(field_values)
        field_ids = {name: index for index, name in enumerate(all_fields)}
        if query.numeric_fields is None:
            selected = tuple(
                name
                for name in all_fields
                if field_values[name]
                and all(type(value) in (int, float) for value in field_values[name])
                and all(
                    type(value) is int or math.isfinite(value)
                    for value in field_values[name]
                )
            )
        else:
            selected = query.numeric_fields
            unknown = [name for name in selected if name not in field_values]
            if unknown:
                raise ValueError(f"numeric_fields are not observed: {unknown!r}")
            for name in selected:
                if any(type(value) not in (int, float) for value in field_values[name]):
                    raise ValueError(f"numeric field {name!r} contains a nonnumeric value")
                if any(
                    type(value) is float and not math.isfinite(value)
                    for value in field_values[name]
                ):
                    raise ValueError(f"numeric field {name!r} contains a non-finite value")
        for name in selected:
            if any(
                abs(value) > EXACT_NUMERIC_MAGNITUDE
                for value in field_values[name]
            ):
                raise ValueError(
                    f"numeric field {name!r} exceeds PTA's exact arithmetic range"
                )

        labels: dict[int, int] = {}
        for example, label in session.example_labels:
            if example in labels and labels[example] != label:
                raise ValueError(f"example {example} has conflicting labels")
            labels[example] = label
        for name in selected:
            observed: dict[int, int | float] = {}
            for _, example, field, value in session.observations:
                if field != name:
                    continue
                if example in observed and observed[example] != value:
                    raise ValueError(
                        f"example {example} has conflicting observations for {name!r}"
                    )
                observed[example] = value
        return field_ids, tuple(field_ids[name] for name in selected)

    @staticmethod
    def _validate_deescalation_truths(session: PTAReasoningSession) -> None:
        domain = set(session.example_domains)

        def validate_vectors(
            facts: list[tuple[int, int, int]], relation: str
        ) -> None:
            vectors: dict[int, dict[int, int]] = {}
            for subject, example, truth in facts:
                values = vectors.setdefault(subject, {})
                if example in values:
                    raise ValueError(
                        f"{relation} contains duplicate truth for subject "
                        f"{subject}, example {example}"
                    )
                values[example] = truth
            if vectors and not domain:
                raise ValueError(f"{relation} requires a nonempty example domain")
            for subject, values in vectors.items():
                if set(values) != domain:
                    raise ValueError(
                        f"{relation} subject {subject} lacks one exact truth value "
                        "for every domain example"
                    )

        validate_vectors(session.literal_truths, "literal_truth")
        validate_vectors(session.clause_truths, "clause_truth")

    @staticmethod
    def _validate_input_products(
        session: PTAReasoningSession,
        insights: tuple[PTAInsight, ...],
        proposals: tuple[PTAEscalationProposal, ...],
    ) -> None:
        labels = dict(session.example_labels)
        by_field: dict[str, set[tuple[int | float, int]]] = {}
        for _, example, field, value in session.observations:
            if example in labels and type(value) in (int, float):
                by_field.setdefault(field, set()).add((value, labels[example]))

        def separating_boundary(field: str, boundary: float) -> bool:
            pairs = sorted(by_field.get(field, ()))
            return any(
                left_label != right_label
                and left_value < boundary < right_value
                for (left_value, left_label), (right_value, right_label) in zip(
                    pairs, pairs[1:]
                )
            )

        for insight in insights:
            if insight.kind == "threshold":
                boundary = insight.evidence[0]
                if not separating_boundary(insight.subject, boundary):
                    raise PTACollectiveProtocolError(
                        "collective threshold does not strictly separate a label flip"
                    )
            elif insight.kind == "interval":
                lower, upper = insight.evidence
                pairs = by_field.get(insight.subject, set())
                if (
                    lower >= upper
                    or not separating_boundary(insight.subject, lower)
                    or not separating_boundary(insight.subject, upper)
                    or not any(
                        label == 1 and lower <= value < upper
                        for value, label in pairs
                    )
                    or any(
                        label == 0 and lower <= value < upper
                        for value, label in pairs
                    )
                ):
                    raise PTACollectiveProtocolError(
                        "collective interval failed independent behavioral validation"
                    )
        for proposal in proposals:
            if proposal.native_target == "threshold":
                field = proposal.structure["field"]
                boundary = proposal.structure["threshold"]
                if not separating_boundary(field, boundary):
                    raise PTACollectiveProtocolError(
                        "collective threshold proposal failed behavioral validation"
                    )

    def run(
        self,
        session: PTAReasoningSession,
        *,
        query: PTACollectiveQuery | None = None,
        budget: PTACollectiveBudget | None = None,
    ) -> PTACollectiveResult:
        if not isinstance(session, PTAReasoningSession):
            raise TypeError("session must be PTAReasoningSession")
        resolved_query = PTACollectiveQuery() if query is None else query
        resolved_budget = PTACollectiveBudget() if budget is None else budget
        if not isinstance(resolved_query, PTACollectiveQuery):
            raise TypeError("query must be PTACollectiveQuery")
        if not isinstance(resolved_budget, PTACollectiveBudget):
            raise TypeError("budget must be PTACollectiveBudget")
        self._validate_resource_budget(session, resolved_budget)
        if resolved_query.derive_deescalation:
            self._validate_deescalation_truths(session)
        field_ids, selected_ids = self._field_maps(session, resolved_query)
        id_to_field = {identifier: field for field, identifier in field_ids.items()}
        semantic_examples, semantic_literals, semantic_clauses, semantic_classes = (
            self._identifier_sets(session)
        )
        if resolved_query.derive_deescalation:
            for name, count in (
                ("literal", len({item[0] for item in session.literal_truths})),
                ("clause", len({item[0] for item in session.clause_truths})),
            ):
                if count * max(0, count - 1) > resolved_budget.max_pair_candidates:
                    raise ValueError(
                        f"{name} pair candidates exceed collective budget"
                    )

        def opaque(values: set[int]) -> dict[int, int]:
            return {value: index for index, value in enumerate(sorted(values))}

        example_ids = opaque(semantic_examples)
        literal_ids = opaque(semantic_literals)
        clause_ids = opaque(semantic_clauses)
        class_ids = opaque(semantic_classes)
        id_to_literal = {value: key for key, value in literal_ids.items()}
        id_to_clause = {value: key for key, value in clause_ids.items()}
        id_to_class = {value: key for key, value in class_ids.items()}
        truth_literal_ids = tuple(
            literal_ids[value]
            for value in sorted({literal for literal, _, _ in session.literal_truths})
        )
        truth_clauses = {clause for clause, _, _ in session.clause_truths}
        literal_clauses = {clause for clause, _ in session.clause_literals}
        truth_clause_ids = tuple(
            clause_ids[value]
            for value in sorted(truth_clauses & literal_clauses)
        )
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="ptm-pta-") as temporary:
            temporary_path = Path(temporary)
            facts_path = temporary_path / "facts.pl"
            driver_path = temporary_path / "driver.pl"
            _write_bounded_fact_lines(
                facts_path,
                session.iter_prolog_fact_lines(
                    observation_field_ids=field_ids,
                    example_ids=example_ids,
                    literal_ids=literal_ids,
                    clause_ids=clause_ids,
                    class_ids=class_ids,
                ),
                max_bytes=resolved_budget.max_input_bytes,
            )
            driver_path.write_text(
                _driver_source(
                    self.module_paths,
                    facts_path,
                    selected_ids,
                    truth_literal_ids,
                    truth_clause_ids,
                    resolved_query,
                    resolved_budget,
                ),
                encoding="utf-8",
                newline="\n",
            )
            command = [
                str(self.executable),
                "--consult-file",
                str(driver_path),
                "--query-goal",
                "ptm_main",
            ]
            completed = _run_bounded_process(
                command,
                cwd=temporary_path,
                timeout_seconds=resolved_budget.timeout_seconds,
                max_output_bytes=resolved_budget.max_output_bytes,
            )
        if completed.returncode != 0:
            diagnostic = (completed.stdout + b"\n" + completed.stderr)[-2_000:].decode(
                "utf-8", errors="replace"
            )
            raise PTACollectiveExecutionError(
                f"GNU Prolog exited with code {completed.returncode}: {diagnostic}"
            )
        insights, proposals, product_counts = _decode_protocol(
            completed.stdout,
            id_to_field=id_to_field,
            id_to_literal=id_to_literal,
            id_to_clause=id_to_clause,
            id_to_class=id_to_class,
            max_results_per_product=resolved_budget.max_results_per_product,
        )
        self._validate_input_products(session, insights, proposals)
        return PTACollectiveResult(
            insights=insights,
            proposals=proposals,
            field_ids=field_ids,
            product_counts=product_counts,
            elapsed_seconds=float(time.monotonic() - started),
        )


__all__ = [
    "PTACollectiveBudget",
    "PTACollectiveError",
    "PTACollectiveExecutionError",
    "PTACollectiveProtocolError",
    "PTACollectiveProductCount",
    "PTACollectiveQuery",
    "PTACollectiveResult",
    "PTACollectiveService",
    "PTACollectiveTimeout",
    "PTACollectiveUnavailable",
]
