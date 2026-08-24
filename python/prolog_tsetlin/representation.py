"""Class I representation with synchronized Boolean and typed views."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


TRANSFORM_CATALOG_VERSION = 1
IDENTITY_SCHEMA_VERSION = 1


class FieldKind(str, Enum):
    NUMBER = "number"
    CATEGORY = "category"
    TEXT = "text"
    BOOLEAN = "boolean"


class TransformKind(str, Enum):
    NUMERIC_GE = "numeric_ge"
    NUMERIC_BETWEEN = "numeric_between"
    CATEGORY_EQ = "category_eq"
    CATEGORY_IN = "category_in"
    IS_MISSING = "is_missing"
    TOKEN_CONTAINS = "token_contains"


class NullPolicy(str, Enum):
    FALSE = "false"
    TRUE = "true"
    ERROR = "error"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_u64(value: object) -> int:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _typed_equal(left: object, right: object) -> bool:
    """Compare scalar category values without Python's bool/int aliasing."""

    return type(left) is type(right) and left == right


def _validate_typed_field_value(
    kind: FieldKind, value: object, field: str
) -> object:
    """Validate one non-null value against the Class-I field schema."""

    if kind is FieldKind.NUMBER:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"field {field!r} must be a number")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"field {field!r} must be finite")
        return value
    if kind is FieldKind.CATEGORY:
        if not isinstance(value, (str, int, bool)):
            raise ValueError(
                f"field {field!r} must be a string, integer, or Boolean"
            )
        return value
    if kind is FieldKind.BOOLEAN:
        if type(value) is not bool:
            raise ValueError(f"field {field!r} must be Boolean")
        return value
    if kind is FieldKind.TEXT:
        if type(value) is not str:
            raise ValueError(f"field {field!r} must be text")
        return value
    raise AssertionError(f"unhandled field kind: {kind}")


def _freeze_parameter(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_parameter(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_parameter(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("literal parameters must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported literal parameter type: {type(value).__name__}")


def _thaw_parameter(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_thaw_parameter(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class FieldDefinition:
    source_field_id: int
    name: str
    kind: FieldKind

    @classmethod
    def create(cls, name: str, kind: FieldKind) -> "FieldDefinition":
        if not name:
            raise ValueError("field name cannot be empty")
        payload = {
            "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "name": name,
            "kind": kind.value,
        }
        return cls(_stable_u64(payload), name, kind)


class FeatureSchema:
    def __init__(self, fields: Iterable[FieldDefinition]) -> None:
        by_name: dict[str, FieldDefinition] = {}
        by_id: dict[int, FieldDefinition] = {}
        for field in fields:
            if field.name in by_name:
                raise ValueError(f"duplicate field name: {field.name}")
            existing = by_id.get(field.source_field_id)
            if existing is not None and existing != field:
                raise ValueError("source-field ID collision")
            by_name[field.name] = field
            by_id[field.source_field_id] = field
        if not by_name:
            raise ValueError("feature schema cannot be empty")
        self._by_name = MappingProxyType(by_name)
        self._by_id = MappingProxyType(by_id)

    @classmethod
    def from_fields(cls, **fields: FieldKind) -> "FeatureSchema":
        return cls(FieldDefinition.create(name, kind) for name, kind in fields.items())

    @property
    def fields(self) -> tuple[FieldDefinition, ...]:
        return tuple(self._by_name.values())

    def field(self, name: str) -> FieldDefinition:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"unknown source field: {name}") from exc

    def field_by_id(self, source_field_id: int) -> FieldDefinition:
        return self._by_id[source_field_id]


@dataclass(frozen=True, slots=True)
class LiteralDescriptor:
    literal_id: int
    source_field_id: int
    source_field: str
    transform: TransformKind
    parameters: tuple[tuple[str, Any], ...]
    null_policy: NullPolicy
    catalog_version: int = TRANSFORM_CATALOG_VERSION

    def parameter(self, name: str) -> Any:
        for key, value in self.parameters:
            if key == name:
                return value
        raise KeyError(name)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "catalog_version": self.catalog_version,
            "source_field_id": self.source_field_id,
            "transform": self.transform.value,
            "parameters": {
                key: _thaw_parameter(value) for key, value in self.parameters
            },
            "null_policy": self.null_policy.value,
        }


@dataclass(frozen=True, slots=True)
class EvaluationTrace:
    row_id: str | int
    literal_id: int
    source_field_id: int
    raw_value: Any
    result: bool


@dataclass(frozen=True, slots=True)
class TypedFact:
    predicate: str
    row_id: str | int
    value: Any
    field_kind: FieldKind
    source_field_id: int


@dataclass(frozen=True, slots=True)
class LiteralBatch:
    row_ids: tuple[str | int, ...]
    literal_ids: tuple[int, ...]
    words: tuple[tuple[int, ...], ...]

    @property
    def row_count(self) -> int:
        return len(self.row_ids)

    @property
    def literal_count(self) -> int:
        return len(self.literal_ids)

    def bit(self, row_index: int, literal_position: int) -> bool:
        if not 0 <= literal_position < self.literal_count:
            raise IndexError(literal_position)
        word = self.words[row_index][literal_position // 64]
        return bool((word >> (literal_position % 64)) & 1)

    def row_values(self, row_index: int) -> tuple[bool, ...]:
        return tuple(self.bit(row_index, position) for position in range(self.literal_count))

    def feature_major_words64(
        self,
        start_row: int = 0,
    ) -> tuple[tuple[int, ...], int]:
        """Transpose up to 64 rows into the native feature-major batch layout."""
        if not 0 <= start_row < self.row_count:
            raise IndexError(start_row)
        lane_count = min(64, self.row_count - start_row)
        packed = [0] * self.literal_count
        for lane in range(lane_count):
            source_words = self.words[start_row + lane]
            for feature in range(self.literal_count):
                if (source_words[feature // 64] >> (feature % 64)) & 1:
                    packed[feature] |= 1 << lane
        valid = (1 << lane_count) - 1 if lane_count < 64 else (1 << 64) - 1
        return tuple(packed), valid


@dataclass(frozen=True, slots=True)
class RepresentationBatch:
    raw_records: tuple[Mapping[str, Any], ...]
    ta: LiteralBatch
    symbolic: tuple[tuple[TypedFact, ...], ...]
    traces: tuple[tuple[EvaluationTrace, ...], ...]


class LiteralCatalog:
    """A bounded, insertion-ordered literal catalog with deterministic IDs."""

    def __init__(self, schema: FeatureSchema) -> None:
        self.schema = schema
        self._literals: list[LiteralDescriptor] = []
        self._by_id: dict[int, LiteralDescriptor] = {}

    @property
    def literals(self) -> tuple[LiteralDescriptor, ...]:
        return tuple(self._literals)

    @classmethod
    def from_descriptors(
        cls,
        schema: FeatureSchema,
        descriptors: Iterable[LiteralDescriptor],
    ) -> "LiteralCatalog":
        """Rebuild one catalog in the declared positional order.

        Stable literal IDs do not make feature positions interchangeable.  This
        constructor validates every externally supplied descriptor and retains
        its sequence so snapshots can be restored against an exact feature
        manifest.
        """

        catalog = cls(schema)
        for descriptor in descriptors:
            catalog.register_descriptor(descriptor)
        return catalog

    def clone(self) -> "LiteralCatalog":
        """Return an independently mutable catalog with identical positions."""

        return type(self).from_descriptors(self.schema, self._literals)

    def register_descriptor(
        self, descriptor: LiteralDescriptor
    ) -> LiteralDescriptor:
        """Append one canonical external descriptor without reordering it."""

        canonical = self.validate_descriptor(descriptor)
        existing = self._by_id.get(canonical.literal_id)
        if existing is not None:
            if existing != canonical:
                raise ValueError("literal ID collision")
            return existing
        self._by_id[canonical.literal_id] = canonical
        self._literals.append(canonical)
        return canonical

    def _describe(
        self,
        field_name: str,
        transform: TransformKind,
        parameters: Mapping[str, Any],
        null_policy: NullPolicy,
    ) -> LiteralDescriptor:
        field = self.schema.field(field_name)
        frozen_parameters = tuple(
            (name, _freeze_parameter(value))
            for name, value in sorted(parameters.items())
        )
        payload = {
            "catalog_version": TRANSFORM_CATALOG_VERSION,
            "source_field_id": field.source_field_id,
            "transform": transform.value,
            "parameters": {
                name: _thaw_parameter(value) for name, value in frozen_parameters
            },
            "null_policy": null_policy.value,
        }
        return LiteralDescriptor(
            literal_id=_stable_u64(payload),
            source_field_id=field.source_field_id,
            source_field=field.name,
            transform=transform,
            parameters=frozen_parameters,
            null_policy=null_policy,
        )

    def _register(
        self,
        field_name: str,
        transform: TransformKind,
        parameters: Mapping[str, Any],
        null_policy: NullPolicy,
    ) -> LiteralDescriptor:
        descriptor = self.preview(field_name, transform, parameters, null_policy)
        existing = self._by_id.get(descriptor.literal_id)
        if existing is not None:
            if existing != descriptor:
                raise ValueError("literal ID collision")
            return existing
        self._by_id[descriptor.literal_id] = descriptor
        self._literals.append(descriptor)
        return descriptor

    def preview(
        self,
        field_name: str,
        transform: TransformKind,
        parameters: Mapping[str, Any],
        null_policy: NullPolicy = NullPolicy.FALSE,
    ) -> LiteralDescriptor:
        """Validate and describe a literal without mutating the catalog."""
        if not isinstance(transform, TransformKind):
            raise ValueError("transform is not part of the catalog schema")
        if not isinstance(null_policy, NullPolicy):
            raise ValueError("null_policy is not part of the catalog schema")
        if not isinstance(parameters, Mapping):
            raise TypeError("literal parameters must be a mapping")
        if any(not isinstance(key, str) for key in parameters):
            raise TypeError("literal parameter names must be strings")
        keys = set(parameters)
        if transform is TransformKind.NUMERIC_GE:
            if keys != {"threshold"}:
                raise ValueError("numeric_ge requires exactly threshold")
            return self.preview_numeric_ge(
                field_name, parameters["threshold"], null_policy=null_policy
            )
        if transform is TransformKind.NUMERIC_BETWEEN:
            expected = {"lower", "upper", "inclusive_lower", "inclusive_upper"}
            if keys != expected:
                raise ValueError(
                    "numeric_between requires lower, upper, inclusive_lower, and inclusive_upper"
                )
            return self.preview_numeric_between(
                field_name,
                parameters["lower"],
                parameters["upper"],
                inclusive_lower=parameters["inclusive_lower"],
                inclusive_upper=parameters["inclusive_upper"],
                null_policy=null_policy,
            )
        if transform is TransformKind.CATEGORY_EQ:
            if keys != {"value"}:
                raise ValueError("category_eq requires exactly value")
            return self.preview_category_eq(
                field_name, parameters["value"], null_policy=null_policy
            )
        if transform is TransformKind.CATEGORY_IN:
            if keys != {"values"}:
                raise ValueError("category_in requires exactly values")
            return self.preview_category_in(
                field_name, parameters["values"], null_policy=null_policy
            )
        if transform is TransformKind.IS_MISSING:
            if keys:
                raise ValueError("is_missing does not accept parameters")
            if null_policy is not NullPolicy.FALSE:
                raise ValueError("is_missing requires the false null policy")
            return self.preview_is_missing(field_name)
        if transform is TransformKind.TOKEN_CONTAINS:
            if keys != {"token", "case_sensitive"}:
                raise ValueError(
                    "token_contains requires exactly token and case_sensitive"
                )
            return self.preview_token_contains(
                field_name,
                parameters["token"],
                case_sensitive=parameters["case_sensitive"],
                null_policy=null_policy,
            )
        raise AssertionError(f"unhandled transform: {transform}")

    def validate_descriptor(self, descriptor: LiteralDescriptor) -> LiteralDescriptor:
        """Return the canonical descriptor or reject an invalid external value."""
        if not isinstance(descriptor, LiteralDescriptor):
            raise TypeError("descriptor must be LiteralDescriptor")
        if type(descriptor.literal_id) is not int:
            raise ValueError("literal_id must be a strict integer")
        if descriptor.catalog_version != TRANSFORM_CATALOG_VERSION:
            raise ValueError("catalog_version mismatch")
        try:
            field = self.schema.field(descriptor.source_field)
        except (KeyError, TypeError) as exc:
            raise ValueError("descriptor field not in schema") from exc
        if descriptor.source_field_id != field.source_field_id:
            raise ValueError("descriptor source_field_id mismatch")
        if not isinstance(descriptor.parameters, tuple):
            raise ValueError("descriptor parameters must be canonical tuple pairs")
        try:
            names = [item[0] for item in descriptor.parameters]
            if any(not isinstance(name, str) for name in names):
                raise ValueError("descriptor parameter names must be strings")
            if len(set(names)) != len(names):
                raise ValueError("descriptor parameter names must be unique")
            parameters = {
                name: _thaw_parameter(value)
                for name, value in descriptor.parameters
            }
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("descriptor parameters are malformed") from exc
        canonical = self.preview(
            descriptor.source_field,
            descriptor.transform,
            parameters,
            descriptor.null_policy,
        )
        if canonical != descriptor:
            raise ValueError("descriptor is not canonical")
        return canonical

    def preview_numeric_ge(
        self,
        field_name: str,
        threshold: int | float,
        *,
        null_policy: NullPolicy = NullPolicy.FALSE,
    ) -> LiteralDescriptor:
        self._require_kind(field_name, FieldKind.NUMBER)
        self._require_finite_number(threshold, "threshold")
        return self._describe(field_name, TransformKind.NUMERIC_GE, {"threshold": threshold}, null_policy)

    def preview_numeric_between(
        self,
        field_name: str,
        lower: int | float,
        upper: int | float,
        *,
        inclusive_lower: bool = True,
        inclusive_upper: bool = True,
        null_policy: NullPolicy = NullPolicy.FALSE,
    ) -> LiteralDescriptor:
        self._require_kind(field_name, FieldKind.NUMBER)
        self._require_finite_number(lower, "lower")
        self._require_finite_number(upper, "upper")
        if type(inclusive_lower) is not bool or type(inclusive_upper) is not bool:
            raise TypeError("numeric interval inclusivity flags must be booleans")
        if lower > upper:
            raise ValueError("lower bound cannot exceed upper bound")
        return self._describe(
            field_name,
            TransformKind.NUMERIC_BETWEEN,
            {"lower": lower, "upper": upper, "inclusive_lower": inclusive_lower, "inclusive_upper": inclusive_upper},
            null_policy,
        )

    def preview_category_eq(
        self,
        field_name: str,
        value: str | int | bool,
        *,
        null_policy: NullPolicy = NullPolicy.FALSE,
    ) -> LiteralDescriptor:
        self._require_kind(field_name, FieldKind.CATEGORY, FieldKind.BOOLEAN)
        self._require_category_value(value)
        return self._describe(field_name, TransformKind.CATEGORY_EQ, {"value": value}, null_policy)

    def preview_category_in(
        self,
        field_name: str,
        values: Sequence[str | int | bool],
        *,
        null_policy: NullPolicy = NullPolicy.FALSE,
    ) -> LiteralDescriptor:
        self._require_kind(field_name, FieldKind.CATEGORY, FieldKind.BOOLEAN)
        canonical_values = self._canonical_category_values(values)
        return self._describe(
            field_name,
            TransformKind.CATEGORY_IN,
            {"values": canonical_values},
            null_policy,
        )

    def preview_is_missing(self, field_name: str) -> LiteralDescriptor:
        self.schema.field(field_name)
        return self._describe(
            field_name, TransformKind.IS_MISSING, {}, NullPolicy.FALSE
        )

    def preview_token_contains(
        self,
        field_name: str,
        token: str,
        *,
        case_sensitive: bool = False,
        null_policy: NullPolicy = NullPolicy.FALSE,
    ) -> LiteralDescriptor:
        self._require_kind(field_name, FieldKind.TEXT)
        if not isinstance(token, str) or not token or any(character.isspace() for character in token):
            raise ValueError("token must be one non-whitespace token")
        if type(case_sensitive) is not bool:
            raise TypeError("case_sensitive must be a boolean")
        return self._describe(
            field_name,
            TransformKind.TOKEN_CONTAINS,
            {"token": token, "case_sensitive": case_sensitive},
            null_policy,
        )

    def numeric_ge(
        self,
        field_name: str,
        threshold: int | float,
        *,
        null_policy: NullPolicy = NullPolicy.FALSE,
    ) -> LiteralDescriptor:
        return self._register(
            field_name, TransformKind.NUMERIC_GE, {"threshold": threshold}, null_policy
        )

    def numeric_between(
        self,
        field_name: str,
        lower: int | float,
        upper: int | float,
        *,
        inclusive_lower: bool = True,
        inclusive_upper: bool = True,
        null_policy: NullPolicy = NullPolicy.FALSE,
    ) -> LiteralDescriptor:
        return self._register(
            field_name,
            TransformKind.NUMERIC_BETWEEN,
            {
                "lower": lower,
                "upper": upper,
                "inclusive_lower": inclusive_lower,
                "inclusive_upper": inclusive_upper,
            },
            null_policy,
        )

    def category_eq(
        self,
        field_name: str,
        value: str | int | bool,
        *,
        null_policy: NullPolicy = NullPolicy.FALSE,
    ) -> LiteralDescriptor:
        return self._register(
            field_name, TransformKind.CATEGORY_EQ, {"value": value}, null_policy
        )

    def category_in(
        self,
        field_name: str,
        values: Sequence[str | int | bool],
        *,
        null_policy: NullPolicy = NullPolicy.FALSE,
    ) -> LiteralDescriptor:
        return self._register(
            field_name,
            TransformKind.CATEGORY_IN,
            {"values": values},
            null_policy,
        )

    def is_missing(self, field_name: str) -> LiteralDescriptor:
        return self._register(
            field_name, TransformKind.IS_MISSING, {}, NullPolicy.FALSE
        )

    def token_contains(
        self,
        field_name: str,
        token: str,
        *,
        case_sensitive: bool = False,
        null_policy: NullPolicy = NullPolicy.FALSE,
    ) -> LiteralDescriptor:
        return self._register(
            field_name,
            TransformKind.TOKEN_CONTAINS,
            {"token": token, "case_sensitive": case_sensitive},
            null_policy,
        )

    @staticmethod
    def _require_finite_number(value: Any, name: str) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise TypeError(f"{name} must be a finite number")

    @staticmethod
    def _require_category_value(value: Any) -> None:
        if not isinstance(value, (str, int, bool)):
            raise TypeError("category values must be strings, integers, or booleans")

    @classmethod
    def _canonical_category_values(
        cls, values: Sequence[str | int | bool]
    ) -> tuple[str | int | bool, ...]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError("category membership must be a sequence of values")
        if not values:
            raise ValueError("category membership cannot be empty")
        for value in values:
            cls._require_category_value(value)
        # Python considers ``True == 1``. Literal equality is deliberately
        # typed, so preserve both while still removing exact typed duplicates.
        typed_values = {(type(value), value): value for value in values}
        return tuple(
            sorted(
                typed_values.values(),
                key=lambda value: (type(value).__name__, str(value)),
            )
        )

    def _require_kind(self, field_name: str, *allowed: FieldKind) -> None:
        field = self.schema.field(field_name)
        if field.kind not in allowed:
            expected = ", ".join(kind.value for kind in allowed)
            raise TypeError(f"field {field_name!r} must be one of: {expected}")

    @staticmethod
    def _null_result(descriptor: LiteralDescriptor) -> bool:
        if descriptor.null_policy is NullPolicy.ERROR:
            raise ValueError(f"null value for literal {descriptor.literal_id}")
        return descriptor.null_policy is NullPolicy.TRUE

    def evaluate(self, descriptor: LiteralDescriptor, raw_value: Any) -> bool:
        transform = descriptor.transform
        if raw_value is None:
            if transform is TransformKind.IS_MISSING:
                return True
            return self._null_result(descriptor)

        field = self.schema.field(descriptor.source_field)
        normalized = _validate_typed_field_value(
            field.kind, raw_value, field.name
        )
        if transform is TransformKind.IS_MISSING:
            return False

        if transform is TransformKind.NUMERIC_GE:
            return normalized >= descriptor.parameter("threshold")
        if transform is TransformKind.NUMERIC_BETWEEN:
            lower = descriptor.parameter("lower")
            upper = descriptor.parameter("upper")
            lower_ok = normalized >= lower if descriptor.parameter("inclusive_lower") else normalized > lower
            upper_ok = normalized <= upper if descriptor.parameter("inclusive_upper") else normalized < upper
            return lower_ok and upper_ok
        if transform is TransformKind.CATEGORY_EQ:
            return _typed_equal(normalized, descriptor.parameter("value"))
        if transform is TransformKind.CATEGORY_IN:
            return any(
                _typed_equal(normalized, candidate)
                for candidate in descriptor.parameter("values")
            )
        if transform is TransformKind.TOKEN_CONTAINS:
            token = descriptor.parameter("token")
            case_sensitive = descriptor.parameter("case_sensitive")
            assert isinstance(normalized, str)
            tokens = normalized.split()
            if not case_sensitive:
                token = token.casefold()
                tokens = [item.casefold() if isinstance(item, str) else item for item in tokens]
            return token in tokens
        raise AssertionError(f"unhandled transform: {transform}")

    def encode(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        row_ids: Iterable[str | int] | None = None,
        include_traces: bool = True,
    ) -> RepresentationBatch:
        frozen_records = tuple(MappingProxyType(dict(record)) for record in records)
        if row_ids is None:
            ids: tuple[str | int, ...] = tuple(range(len(frozen_records)))
        else:
            ids = tuple(row_ids)
            if len(ids) != len(frozen_records):
                raise ValueError("row_ids and records must have equal length")
        if len(set(ids)) != len(ids):
            raise ValueError("row IDs must be unique within a batch")

        word_count = (len(self._literals) + 63) // 64
        packed_rows: list[tuple[int, ...]] = []
        symbolic_rows: list[tuple[TypedFact, ...]] = []
        trace_rows: list[tuple[EvaluationTrace, ...]] = []

        for row_id, record in zip(ids, frozen_records):
            for field in self.schema.fields:
                raw_value = record.get(field.name)
                if raw_value is not None:
                    _validate_typed_field_value(
                        field.kind, raw_value, field.name
                    )
            words = [0] * word_count
            traces: list[EvaluationTrace] = []
            for position, descriptor in enumerate(self._literals):
                raw_value = record.get(descriptor.source_field)
                result = self.evaluate(descriptor, raw_value)
                if result:
                    words[position // 64] |= 1 << (position % 64)
                if include_traces:
                    traces.append(
                        EvaluationTrace(
                            row_id=row_id,
                            literal_id=descriptor.literal_id,
                            source_field_id=descriptor.source_field_id,
                            raw_value=raw_value,
                            result=result,
                        )
                    )
            facts = tuple(
                TypedFact(
                    predicate=field.name,
                    row_id=row_id,
                    value=record.get(field.name),
                    field_kind=field.kind,
                    source_field_id=field.source_field_id,
                )
                for field in self.schema.fields
            )
            packed_rows.append(tuple(words))
            symbolic_rows.append(facts)
            trace_rows.append(tuple(traces))

        return RepresentationBatch(
            raw_records=frozen_records,
            ta=LiteralBatch(
                row_ids=ids,
                literal_ids=tuple(item.literal_id for item in self._literals),
                words=tuple(packed_rows),
            ),
            symbolic=tuple(symbolic_rows),
            traces=tuple(trace_rows),
        )
