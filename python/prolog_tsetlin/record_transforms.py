"""Versioned deterministic record transforms for connector-side pipelines."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Protocol


RECORD_PIPELINE_SCHEMA = "ptm.record_pipeline.v1"
MAX_PIPELINE_TRANSFORMS = 256
MAX_SEQUENCE_ITEMS = 65_536
MAX_REGEX_PATTERN_CHARS = 512
MAX_REGEX_INPUT_CHARS = 65_536
MAX_REGEX_MATCHES = 4_096


class TransformError(ValueError):
    """Raised when a record cannot satisfy a declared transform contract."""


class RegexOperation(str, Enum):
    SEARCH = "search"
    FULLMATCH = "fullmatch"
    COUNT = "count"
    EXTRACT = "extract"


class AggregateOperation(str, Enum):
    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    ANY = "any"
    ALL = "all"


class RelationalOperation(str, Enum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class SequenceOperation(str, Enum):
    LENGTH = "length"
    UNIQUE_COUNT = "unique_count"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    ITEM = "item"


class TemporalOperation(str, Enum):
    EPOCH_SECONDS = "epoch_seconds"
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    HOUR = "hour"
    WEEKDAY = "weekday"
    SECONDS_SINCE = "seconds_since"
    WITHIN_SECONDS = "within_seconds"


class RecordTransform(Protocol):
    output_field: str

    def evaluate(self, record: Mapping[str, object]) -> object: ...

    def to_dict(self) -> dict[str, Any]: ...


def _field_name(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    if any(ord(character) < 0x20 for character in value):
        raise ValueError(f"{label} cannot contain control characters")
    return value


def _required(record: Mapping[str, object], field_name: str) -> object:
    if field_name not in record or record[field_name] is None:
        raise TransformError(f"required field {field_name!r} is missing or null")
    return record[field_name]


def _typed_equal(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


def _sequence(value: object, field_name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TransformError(f"field {field_name!r} must be a sequence")
    if len(value) > MAX_SEQUENCE_ITEMS:
        raise TransformError(
            f"field {field_name!r} exceeds {MAX_SEQUENCE_ITEMS} sequence items"
        )
    return tuple(value)


def _number(value: object, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TransformError(f"{label} must be numeric")
    if isinstance(value, float) and not math.isfinite(value):
        raise TransformError(f"{label} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class RegexTransform:
    source_field: str
    output_field: str
    pattern: str
    operation: RegexOperation = RegexOperation.SEARCH
    ignore_case: bool = False
    group: int = 0
    max_input_chars: int = MAX_REGEX_INPUT_CHARS
    _compiled: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.operation, RegexOperation):
            raise ValueError("regex operation is invalid")
        _field_name(self.source_field, "regex source field")
        _field_name(self.output_field, "regex output field")
        if not isinstance(self.pattern, str) or not self.pattern:
            raise ValueError("regex pattern must be a nonempty string")
        if len(self.pattern) > MAX_REGEX_PATTERN_CHARS:
            raise ValueError(
                f"regex pattern exceeds {MAX_REGEX_PATTERN_CHARS} characters"
            )
        if type(self.ignore_case) is not bool:
            raise ValueError("regex ignore_case must be Boolean")
        if not isinstance(self.group, int) or isinstance(self.group, bool) or self.group < 0:
            raise ValueError("regex group must be a nonnegative integer")
        if not 1 <= self.max_input_chars <= MAX_REGEX_INPUT_CHARS:
            raise ValueError(
                f"regex input ceiling must be between 1 and {MAX_REGEX_INPUT_CHARS}"
            )
        try:
            compiled = re.compile(self.pattern, re.IGNORECASE if self.ignore_case else 0)
        except re.error as error:
            raise ValueError(f"regex pattern is invalid: {error}") from error
        if self.group > compiled.groups:
            raise ValueError("regex extraction group does not exist")
        object.__setattr__(self, "_compiled", compiled)

    def evaluate(self, record: Mapping[str, object]) -> object:
        raw = _required(record, self.source_field)
        if not isinstance(raw, str):
            raise TransformError(f"field {self.source_field!r} must be text")
        if len(raw) > self.max_input_chars:
            raise TransformError(
                f"field {self.source_field!r} exceeds the regex input ceiling"
            )
        if self.operation is RegexOperation.SEARCH:
            return self._compiled.search(raw) is not None
        if self.operation is RegexOperation.FULLMATCH:
            return self._compiled.fullmatch(raw) is not None
        if self.operation is RegexOperation.EXTRACT:
            match = self._compiled.search(raw)
            return None if match is None else match.group(self.group)
        count = 0
        for _ in self._compiled.finditer(raw):
            count += 1
            if count > MAX_REGEX_MATCHES:
                raise TransformError(
                    f"regex match count exceeds {MAX_REGEX_MATCHES}"
                )
        return count

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "ignore_case": self.ignore_case,
            "kind": "regex",
            "max_input_chars": self.max_input_chars,
            "operation": self.operation.value,
            "output_field": self.output_field,
            "pattern": self.pattern,
            "source_field": self.source_field,
        }


@dataclass(frozen=True, slots=True)
class AggregateTransform:
    source_fields: tuple[str, ...]
    output_field: str
    operation: AggregateOperation
    skip_nulls: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.operation, AggregateOperation):
            raise ValueError("aggregate operation is invalid")
        if not self.source_fields or len(self.source_fields) > MAX_SEQUENCE_ITEMS:
            raise ValueError("aggregate requires a bounded nonempty source list")
        for name in self.source_fields:
            _field_name(name, "aggregate source field")
        _field_name(self.output_field, "aggregate output field")
        if type(self.skip_nulls) is not bool:
            raise ValueError("aggregate skip_nulls must be Boolean")

    def _values(self, record: Mapping[str, object]) -> tuple[object, ...]:
        if len(self.source_fields) == 1:
            raw = _required(record, self.source_fields[0])
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
                values = _sequence(raw, self.source_fields[0])
            else:
                values = (raw,)
        else:
            values = tuple(record.get(name) for name in self.source_fields)
        if self.skip_nulls:
            values = tuple(value for value in values if value is not None)
        elif any(value is None for value in values):
            raise TransformError("aggregate input contains a null value")
        return values

    def evaluate(self, record: Mapping[str, object]) -> object:
        values = self._values(record)
        if self.operation is AggregateOperation.COUNT:
            return len(values)
        if not values:
            raise TransformError("aggregate has no values after null handling")
        if self.operation in (AggregateOperation.ANY, AggregateOperation.ALL):
            if any(type(value) is not bool for value in values):
                raise TransformError("any/all aggregate inputs must be Boolean")
            return any(values) if self.operation is AggregateOperation.ANY else all(values)
        numbers = tuple(_number(value, "aggregate input") for value in values)
        if self.operation is AggregateOperation.SUM:
            result: int | float = sum(numbers)
        elif self.operation is AggregateOperation.MEAN:
            result = sum(numbers) / len(numbers)
        elif self.operation is AggregateOperation.MIN:
            result = min(numbers)
        elif self.operation is AggregateOperation.MAX:
            result = max(numbers)
        else:
            raise AssertionError("unhandled aggregate operation")
        if isinstance(result, float) and not math.isfinite(result):
            raise TransformError("aggregate result is not finite")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "aggregate",
            "operation": self.operation.value,
            "output_field": self.output_field,
            "skip_nulls": self.skip_nulls,
            "source_fields": list(self.source_fields),
        }


@dataclass(frozen=True, slots=True)
class RelationalTransform:
    left_field: str
    right_field: str
    output_field: str
    operation: RelationalOperation = RelationalOperation.EQ
    missing: str = "error"

    def __post_init__(self) -> None:
        if not isinstance(self.operation, RelationalOperation):
            raise ValueError("relational operation is invalid")
        _field_name(self.left_field, "relational left field")
        _field_name(self.right_field, "relational right field")
        _field_name(self.output_field, "relational output field")
        if self.missing not in ("error", "false", "true"):
            raise ValueError("relational missing policy must be error, false, or true")

    def evaluate(self, record: Mapping[str, object]) -> bool:
        if (
            self.left_field not in record
            or self.right_field not in record
            or record[self.left_field] is None
            or record[self.right_field] is None
        ):
            if self.missing == "error":
                raise TransformError("relational input is missing or null")
            return self.missing == "true"
        left = record[self.left_field]
        right = record[self.right_field]
        if self.operation is RelationalOperation.EQ:
            return _typed_equal(left, right)
        if self.operation is RelationalOperation.NE:
            return not _typed_equal(left, right)
        if type(left) is not type(right) or isinstance(left, bool):
            raise TransformError("ordered relational inputs must have the same type")
        try:
            if self.operation is RelationalOperation.LT:
                return bool(left < right)  # type: ignore[operator]
            if self.operation is RelationalOperation.LE:
                return bool(left <= right)  # type: ignore[operator]
            if self.operation is RelationalOperation.GT:
                return bool(left > right)  # type: ignore[operator]
            return bool(left >= right)  # type: ignore[operator]
        except TypeError as error:
            raise TransformError("relational inputs are not orderable") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "relational",
            "left_field": self.left_field,
            "missing": self.missing,
            "operation": self.operation.value,
            "output_field": self.output_field,
            "right_field": self.right_field,
        }


@dataclass(frozen=True, slots=True)
class SequenceTransform:
    source_field: str
    output_field: str
    operation: SequenceOperation
    argument: object = None
    index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.operation, SequenceOperation):
            raise ValueError("sequence operation is invalid")
        _field_name(self.source_field, "sequence source field")
        _field_name(self.output_field, "sequence output field")
        if not isinstance(self.index, int) or isinstance(self.index, bool):
            raise ValueError("sequence index must be an integer")
        if self.operation in (SequenceOperation.STARTS_WITH, SequenceOperation.ENDS_WITH):
            argument = _sequence(self.argument, "sequence argument")
            if any(
                value is not None
                and not isinstance(value, (bool, int, float, str))
                for value in argument
            ):
                raise ValueError("sequence prefix/suffix values must be JSON scalars")
            if any(isinstance(value, float) and not math.isfinite(value) for value in argument):
                raise ValueError("sequence prefix/suffix numbers must be finite")
            object.__setattr__(self, "argument", argument)
        elif self.operation is SequenceOperation.CONTAINS:
            if self.argument is not None and not isinstance(
                self.argument, (bool, int, float, str)
            ):
                raise ValueError("sequence contains argument must be a JSON scalar")
            if isinstance(self.argument, float) and not math.isfinite(self.argument):
                raise ValueError("sequence contains argument must be finite")
        elif self.argument is not None:
            raise ValueError("sequence argument is not valid for this operation")
        if self.operation is not SequenceOperation.ITEM and self.index != 0:
            raise ValueError("sequence index is only valid for the item operation")

    def evaluate(self, record: Mapping[str, object]) -> object:
        values = _sequence(_required(record, self.source_field), self.source_field)
        if self.operation is SequenceOperation.LENGTH:
            return len(values)
        if self.operation is SequenceOperation.UNIQUE_COUNT:
            unique: list[object] = []
            for value in values:
                if not any(_typed_equal(value, existing) for existing in unique):
                    unique.append(value)
            return len(unique)
        if self.operation is SequenceOperation.CONTAINS:
            return any(_typed_equal(value, self.argument) for value in values)
        if self.operation is SequenceOperation.ITEM:
            try:
                return values[self.index]
            except IndexError as error:
                raise TransformError("sequence item index is out of range") from error
        expected = _sequence(self.argument, "sequence argument")
        if not expected:
            return True
        if len(expected) > len(values):
            return False
        actual = values[: len(expected)] if self.operation is SequenceOperation.STARTS_WITH else values[-len(expected) :]
        return len(actual) == len(expected) and all(
            _typed_equal(left, right) for left, right in zip(actual, expected)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "argument": (
                list(self.argument)
                if isinstance(self.argument, tuple)
                else self.argument
            ),
            "index": self.index,
            "kind": "sequence",
            "operation": self.operation.value,
            "output_field": self.output_field,
            "source_field": self.source_field,
        }


def _instant(value: object, field_name: str) -> datetime:
    if isinstance(value, bool):
        raise TransformError(f"field {field_name!r} is not a timestamp")
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        source = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
        try:
            parsed = datetime.fromisoformat(source)
        except ValueError as error:
            raise TransformError(f"field {field_name!r} is not ISO-8601") from error
    elif isinstance(value, (int, float)):
        numeric = _number(value, f"field {field_name!r}")
        try:
            parsed = datetime.fromtimestamp(numeric, tz=timezone.utc)
        except (OverflowError, OSError, ValueError) as error:
            raise TransformError(f"field {field_name!r} timestamp is out of range") from error
    else:
        raise TransformError(f"field {field_name!r} is not a timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TransformError(f"field {field_name!r} must include a timezone")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class TemporalTransform:
    source_field: str
    output_field: str
    operation: TemporalOperation
    reference_field: str | None = None
    window_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, TemporalOperation):
            raise ValueError("temporal operation is invalid")
        _field_name(self.source_field, "temporal source field")
        _field_name(self.output_field, "temporal output field")
        if self.reference_field is not None:
            _field_name(self.reference_field, "temporal reference field")
        needs_reference = self.operation in (
            TemporalOperation.SECONDS_SINCE,
            TemporalOperation.WITHIN_SECONDS,
        )
        if needs_reference != (self.reference_field is not None):
            raise ValueError("temporal reference_field does not match the operation")
        if self.operation is TemporalOperation.WITHIN_SECONDS:
            if (
                self.window_seconds is None
                or isinstance(self.window_seconds, bool)
                or not isinstance(self.window_seconds, (int, float))
                or not math.isfinite(float(self.window_seconds))
                or self.window_seconds < 0
            ):
                raise ValueError("temporal window_seconds must be finite and nonnegative")
        elif self.window_seconds is not None:
            raise ValueError("window_seconds is only valid for within_seconds")

    def evaluate(self, record: Mapping[str, object]) -> object:
        instant = _instant(_required(record, self.source_field), self.source_field)
        if self.operation is TemporalOperation.EPOCH_SECONDS:
            return instant.timestamp()
        if self.operation is TemporalOperation.YEAR:
            return instant.year
        if self.operation is TemporalOperation.MONTH:
            return instant.month
        if self.operation is TemporalOperation.DAY:
            return instant.day
        if self.operation is TemporalOperation.HOUR:
            return instant.hour
        if self.operation is TemporalOperation.WEEKDAY:
            return instant.weekday()
        assert self.reference_field is not None
        reference = _instant(
            _required(record, self.reference_field), self.reference_field
        )
        difference = (instant - reference).total_seconds()
        if self.operation is TemporalOperation.SECONDS_SINCE:
            return difference
        assert self.window_seconds is not None
        return abs(difference) <= self.window_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "temporal",
            "operation": self.operation.value,
            "output_field": self.output_field,
            "reference_field": self.reference_field,
            "source_field": self.source_field,
            "window_seconds": self.window_seconds,
        }


Transform = (
    RegexTransform
    | AggregateTransform
    | RelationalTransform
    | SequenceTransform
    | TemporalTransform
)


def _transform_from_dict(value: Mapping[str, Any]) -> Transform:
    kind = value.get("kind")
    try:
        if kind == "regex":
            result: Transform = RegexTransform(
                source_field=value["source_field"],
                output_field=value["output_field"],
                pattern=value["pattern"],
                operation=RegexOperation(value["operation"]),
                ignore_case=value["ignore_case"],
                group=value["group"],
                max_input_chars=value["max_input_chars"],
            )
        elif kind == "aggregate":
            sources = value["source_fields"]
            if not isinstance(sources, list):
                raise ValueError("aggregate source_fields must be an array")
            result = AggregateTransform(
                source_fields=tuple(sources),
                output_field=value["output_field"],
                operation=AggregateOperation(value["operation"]),
                skip_nulls=value["skip_nulls"],
            )
        elif kind == "relational":
            result = RelationalTransform(
                left_field=value["left_field"],
                right_field=value["right_field"],
                output_field=value["output_field"],
                operation=RelationalOperation(value["operation"]),
                missing=value["missing"],
            )
        elif kind == "sequence":
            result = SequenceTransform(
                source_field=value["source_field"],
                output_field=value["output_field"],
                operation=SequenceOperation(value["operation"]),
                argument=value["argument"],
                index=value["index"],
            )
        elif kind == "temporal":
            result = TemporalTransform(
                source_field=value["source_field"],
                output_field=value["output_field"],
                operation=TemporalOperation(value["operation"]),
                reference_field=value["reference_field"],
                window_seconds=value["window_seconds"],
            )
        else:
            raise ValueError("unknown record transform kind")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("record transform descriptor is invalid") from error
    if result.to_dict() != dict(value):
        raise ValueError("record transform descriptor is not canonical")
    return result


@dataclass(frozen=True, slots=True)
class RecordTransformPipeline:
    transforms: tuple[Transform, ...]
    schema: str = RECORD_PIPELINE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RECORD_PIPELINE_SCHEMA:
            raise ValueError("unsupported record pipeline schema")
        if not self.transforms:
            raise ValueError("record pipeline requires at least one transform")
        if len(self.transforms) > MAX_PIPELINE_TRANSFORMS:
            raise ValueError(
                f"record pipeline exceeds {MAX_PIPELINE_TRANSFORMS} transforms"
            )
        allowed = (
            RegexTransform,
            AggregateTransform,
            RelationalTransform,
            SequenceTransform,
            TemporalTransform,
        )
        if any(not isinstance(transform, allowed) for transform in self.transforms):
            raise ValueError("record pipeline contains an unsupported transform")
        outputs = tuple(transform.output_field for transform in self.transforms)
        if len(set(outputs)) != len(outputs):
            raise ValueError("record pipeline output fields must be unique")

    @property
    def pipeline_id(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def transform(self, record: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(record, Mapping):
            raise TransformError("record pipeline input must be a mapping")
        result = dict(record)
        for transform in self.transforms:
            if transform.output_field in result:
                raise TransformError(
                    f"record already contains pipeline output {transform.output_field!r}"
                )
            result[transform.output_field] = transform.evaluate(result)
        return result

    def iter_transform(
        self, records: Iterable[Mapping[str, object]]
    ) -> Iterable[dict[str, object]]:
        for record in records:
            yield self.transform(record)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "transforms": [transform.to_dict() for transform in self.transforms],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecordTransformPipeline":
        if not isinstance(value, Mapping) or value.get("schema") != RECORD_PIPELINE_SCHEMA:
            raise ValueError("unsupported record pipeline schema")
        raw = value.get("transforms")
        if not isinstance(raw, list) or not raw:
            raise ValueError("record pipeline transforms must be a nonempty array")
        if len(raw) > MAX_PIPELINE_TRANSFORMS:
            raise ValueError(
                f"record pipeline exceeds {MAX_PIPELINE_TRANSFORMS} transforms"
            )
        parsed: list[Transform] = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("record transform descriptor must be an object")
            parsed.append(_transform_from_dict(item))
        transforms = tuple(parsed)
        result = cls(transforms)
        if result.to_dict() != dict(value):
            raise ValueError("record pipeline is not canonical")
        return result
