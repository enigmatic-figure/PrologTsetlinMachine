"""Deterministic raw-record preprocessing contract for portable artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

from .representation import (
    FeatureSchema,
    FieldDefinition,
    FieldKind,
    LiteralCatalog,
    LiteralDescriptor,
    NullPolicy,
    TransformKind,
    _typed_equal,
    _validate_typed_field_value,
)


PREPROCESSING_SCHEMA = "ptm.preprocessing.v1"
MAX_PREPROCESSING_OUTPUTS = 4096
MAX_PREPROCESSING_DEPTH = 8
MAX_PREPROCESSING_NODES = 100_000
_PORTABLE_TRANSFORMS = frozenset(
    {
        TransformKind.NUMERIC_GE,
        TransformKind.NUMERIC_BETWEEN,
        TransformKind.CATEGORY_EQ,
        TransformKind.CATEGORY_IN,
        TransformKind.IS_MISSING,
    }
)


def _portable_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    if isinstance(value, int) and not -(1 << 53) <= value <= 1 << 53:
        raise ValueError(f"{label} lies outside the exact binary64 integer range")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _portable_category(value: object, label: str) -> str | int | bool:
    if not isinstance(value, (str, int, bool)):
        raise ValueError(f"{label} must be a string, signed integer, or Boolean")
    if isinstance(value, int) and not isinstance(value, bool):
        if not -(1 << 63) <= value < 1 << 63:
            raise ValueError(f"{label} lies outside the signed 64-bit range")
    if isinstance(value, str) and any(ord(character) < 0x20 for character in value):
        raise ValueError(f"{label} cannot contain control characters")
    return value


def _validate_contract_complexity(value: object) -> None:
    """Reject cyclic, deeply nested, or oversized JSON-like contracts early."""

    nodes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_PREPROCESSING_NODES:
            raise ValueError(
                f"preprocessing contract exceeds {MAX_PREPROCESSING_NODES} nodes"
            )
        if depth > MAX_PREPROCESSING_DEPTH:
            raise ValueError(
                f"preprocessing contract exceeds depth {MAX_PREPROCESSING_DEPTH}"
            )
        if isinstance(current, Mapping):
            if any(not isinstance(key, str) for key in current):
                raise ValueError("preprocessing object keys must be strings")
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            stack.extend((child, depth + 1) for child in current)
        elif current is not None and not isinstance(
            current, (bool, int, float, str)
        ):
            raise ValueError("preprocessing contract must contain JSON values")


@dataclass(frozen=True, slots=True)
class PortableLiteral:
    field: FieldDefinition
    descriptor: LiteralDescriptor

    def __post_init__(self) -> None:
        if self.field.kind not in (
            FieldKind.NUMBER,
            FieldKind.CATEGORY,
            FieldKind.BOOLEAN,
        ):
            raise ValueError(
                f"field kind {self.field.kind.value} is not portable in v1"
            )
        if self.descriptor.transform not in _PORTABLE_TRANSFORMS:
            raise ValueError(
                f"transform {self.descriptor.transform.value} is not portable in v1"
            )
        if self.field.name != self.descriptor.source_field or (
            self.field.source_field_id != self.descriptor.source_field_id
        ):
            raise ValueError("portable literal and source field identities disagree")
        if self.descriptor.transform in (
            TransformKind.NUMERIC_GE,
            TransformKind.NUMERIC_BETWEEN,
        ) and self.field.kind is not FieldKind.NUMBER:
            raise ValueError("numeric transforms require a numeric field")
        if self.descriptor.transform in (
            TransformKind.CATEGORY_EQ,
            TransformKind.CATEGORY_IN,
        ) and self.field.kind not in (FieldKind.CATEGORY, FieldKind.BOOLEAN):
            raise ValueError("category transforms require a categorical field")
        if (
            self.descriptor.transform is TransformKind.IS_MISSING
            and self.descriptor.null_policy is not NullPolicy.FALSE
        ):
            raise ValueError("missingness transform requires the false null policy")

    def to_dict(self) -> dict[str, Any]:
        payload = self.descriptor.canonical_payload()
        return {
            "field": self.field.name,
            "field_id": str(self.field.source_field_id),
            "field_kind": self.field.kind.value,
            "literal_id": str(self.descriptor.literal_id),
            "null_policy": self.descriptor.null_policy.value,
            "parameters": payload["parameters"],
            "transform": self.descriptor.transform.value,
        }


@dataclass(frozen=True, slots=True)
class PreprocessingContract:
    """Ordered raw-record transforms producing one model input vector."""

    outputs: tuple[PortableLiteral, ...]
    schema: str = PREPROCESSING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PREPROCESSING_SCHEMA:
            raise ValueError("unsupported preprocessing schema")
        if not self.outputs:
            raise ValueError("preprocessing contract requires at least one output")
        if len(self.outputs) > MAX_PREPROCESSING_OUTPUTS:
            raise ValueError(
                f"preprocessing contract exceeds {MAX_PREPROCESSING_OUTPUTS} outputs"
            )
        literal_ids = tuple(item.descriptor.literal_id for item in self.outputs)
        if len(set(literal_ids)) != len(literal_ids):
            raise ValueError("preprocessing output literal IDs must be unique")
        for item in self.outputs:
            self._validate_parameters(item)

    @classmethod
    def from_catalog(cls, catalog: LiteralCatalog) -> "PreprocessingContract":
        return cls(
            tuple(
                PortableLiteral(
                    catalog.schema.field(descriptor.source_field), descriptor
                )
                for descriptor in catalog.literals
            )
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PreprocessingContract":
        if not isinstance(value, Mapping):
            raise ValueError("preprocessing contract must be an object")
        if value.get("schema") != PREPROCESSING_SCHEMA:
            raise ValueError("unsupported preprocessing schema")
        raw_outputs = value.get("outputs")
        if not isinstance(raw_outputs, list) or not raw_outputs:
            raise ValueError("preprocessing outputs must be a nonempty array")
        if len(raw_outputs) > MAX_PREPROCESSING_OUTPUTS:
            raise ValueError(
                f"preprocessing contract exceeds {MAX_PREPROCESSING_OUTPUTS} outputs"
            )
        _validate_contract_complexity(value)
        fields: dict[str, FieldDefinition] = {}
        for raw in raw_outputs:
            if not isinstance(raw, dict):
                raise ValueError("preprocessing output must be an object")
            try:
                field = FieldDefinition.create(
                    str(raw["field"]), FieldKind(str(raw["field_kind"]))
                )
            except (KeyError, ValueError) as error:
                raise ValueError("preprocessing field descriptor is invalid") from error
            if str(field.source_field_id) != raw.get("field_id"):
                raise ValueError("preprocessing source-field ID is invalid")
            existing = fields.get(field.name)
            if existing is not None and existing != field:
                raise ValueError("preprocessing field has conflicting definitions")
            fields[field.name] = field
        catalog = LiteralCatalog(FeatureSchema(fields.values()))
        for raw in raw_outputs:
            assert isinstance(raw, dict)
            field_name = str(raw["field"])
            try:
                transform = TransformKind(str(raw["transform"]))
                null_policy = NullPolicy(str(raw["null_policy"]))
            except (KeyError, ValueError) as error:
                raise ValueError("preprocessing transform descriptor is invalid") from error
            parameters = raw.get("parameters")
            if not isinstance(parameters, dict):
                raise ValueError("preprocessing parameters must be an object")
            try:
                descriptor = _register_descriptor(
                    catalog, field_name, transform, parameters, null_policy
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "preprocessing transform parameters are invalid"
                ) from error
            if str(descriptor.literal_id) != raw.get("literal_id"):
                raise ValueError("preprocessing literal ID is invalid")
        result = cls.from_catalog(catalog)
        if result.to_dict() != dict(value):
            raise ValueError("preprocessing contract is not canonical")
        return result

    @property
    def literal_ids(self) -> tuple[int, ...]:
        return tuple(item.descriptor.literal_id for item in self.outputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputs": [item.to_dict() for item in self.outputs],
            "schema": self.schema,
        }

    def materialize(self, record: Mapping[str, object]) -> tuple[bool, ...]:
        if not isinstance(record, Mapping):
            raise ValueError("raw record must be a mapping")
        result = []
        for output in self.outputs:
            present = output.field.name in record
            raw = record.get(output.field.name)
            result.append(self._evaluate(output, raw, present=present))
        return tuple(result)

    def materialize_many(
        self, records: Iterable[Mapping[str, object]]
    ) -> tuple[tuple[bool, ...], ...]:
        return tuple(self.materialize(record) for record in records)

    @staticmethod
    def _validate_parameters(output: PortableLiteral) -> None:
        if any(ord(character) < 0x20 for character in output.field.name):
            raise ValueError("portable field names cannot contain control characters")
        descriptor = output.descriptor
        transform = descriptor.transform
        if transform is TransformKind.NUMERIC_GE:
            _portable_number(descriptor.parameter("threshold"), "numeric threshold")
        elif transform is TransformKind.NUMERIC_BETWEEN:
            lower = _portable_number(descriptor.parameter("lower"), "lower bound")
            upper = _portable_number(descriptor.parameter("upper"), "upper bound")
            if lower > upper:
                raise ValueError("portable numeric lower bound exceeds upper bound")
            if type(descriptor.parameter("inclusive_lower")) is not bool or type(
                descriptor.parameter("inclusive_upper")
            ) is not bool:
                raise ValueError("numeric bound inclusion flags must be Boolean")
        elif transform is TransformKind.CATEGORY_EQ:
            value = _portable_category(
                descriptor.parameter("value"), "category value"
            )
            _validate_category_kind(output.field.kind, value)
        elif transform is TransformKind.CATEGORY_IN:
            values = descriptor.parameter("values")
            if not isinstance(values, tuple) or not values:
                raise ValueError("category membership must be a nonempty tuple")
            for value in values:
                normalized = _portable_category(value, "category value")
                _validate_category_kind(output.field.kind, normalized)

    @staticmethod
    def _evaluate(
        output: PortableLiteral, raw: object, *, present: bool
    ) -> bool:
        descriptor = output.descriptor
        if descriptor.transform is TransformKind.IS_MISSING:
            if present and raw is not None:
                _validate_field_value(output.field.kind, raw, output.field.name)
            return not present or raw is None
        if not present or raw is None:
            if descriptor.null_policy is NullPolicy.ERROR:
                raise ValueError(
                    f"missing or null field {output.field.name!r} is not allowed"
                )
            return descriptor.null_policy is NullPolicy.TRUE
        normalized = _validate_field_value(
            output.field.kind, raw, output.field.name
        )
        transform = descriptor.transform
        if transform is TransformKind.NUMERIC_GE:
            return normalized >= _portable_number(
                descriptor.parameter("threshold"), "numeric threshold"
            )
        if transform is TransformKind.NUMERIC_BETWEEN:
            lower = _portable_number(descriptor.parameter("lower"), "lower bound")
            upper = _portable_number(descriptor.parameter("upper"), "upper bound")
            lower_ok = (
                normalized >= lower
                if descriptor.parameter("inclusive_lower")
                else normalized > lower
            )
            upper_ok = (
                normalized <= upper
                if descriptor.parameter("inclusive_upper")
                else normalized < upper
            )
            return lower_ok and upper_ok
        if transform is TransformKind.CATEGORY_EQ:
            return _typed_equal(normalized, descriptor.parameter("value"))
        if transform is TransformKind.CATEGORY_IN:
            return any(
                _typed_equal(normalized, candidate)
                for candidate in descriptor.parameter("values")
            )
        raise AssertionError("unhandled portable transform")


def _validate_field_value(kind: FieldKind, value: object, field: str) -> object:
    normalized = _validate_typed_field_value(kind, value, field)
    if kind is FieldKind.NUMBER:
        return _portable_number(normalized, f"field {field!r}")
    if kind is FieldKind.CATEGORY:
        return _portable_category(normalized, f"field {field!r}")
    if kind is FieldKind.BOOLEAN:
        return normalized
    raise ValueError(f"field kind {kind.value} is not portable in preprocessing v1")


def _validate_category_kind(kind: FieldKind, value: object) -> None:
    if kind is FieldKind.BOOLEAN and type(value) is not bool:
        raise ValueError("Boolean fields require Boolean category values")


def _register_descriptor(
    catalog: LiteralCatalog,
    field_name: str,
    transform: TransformKind,
    parameters: Mapping[str, Any],
    null_policy: NullPolicy,
) -> LiteralDescriptor:
    if transform is TransformKind.NUMERIC_GE:
        return catalog.numeric_ge(
            field_name, parameters["threshold"], null_policy=null_policy
        )
    if transform is TransformKind.NUMERIC_BETWEEN:
        return catalog.numeric_between(
            field_name,
            parameters["lower"],
            parameters["upper"],
            inclusive_lower=parameters["inclusive_lower"],
            inclusive_upper=parameters["inclusive_upper"],
            null_policy=null_policy,
        )
    if transform is TransformKind.CATEGORY_EQ:
        return catalog.category_eq(
            field_name, parameters["value"], null_policy=null_policy
        )
    if transform is TransformKind.CATEGORY_IN:
        values = parameters["values"]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError("category membership values must be an array")
        return catalog.category_in(field_name, values, null_policy=null_policy)
    if transform is TransformKind.IS_MISSING:
        if parameters:
            raise ValueError("missingness transform cannot have parameters")
        return catalog.is_missing(field_name)
    raise ValueError(f"transform {transform.value} is not portable in v1")
