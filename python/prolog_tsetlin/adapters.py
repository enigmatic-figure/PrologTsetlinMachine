"""Task adapters for multi-class, convolutional, regression, and graph (Milestone 4).

All adapters are host-side, bounded, deterministic, JSON-round-trippable.
They materialize scalar fields upstream of ptm.preprocessing.v1 and never
pretend to be embedded preprocessing. No mandatory third-party deps.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .representation import FieldKind

ADAPTER_SCHEMA = "ptm.adapters.v1"
MAX_CLASSES = 256
MAX_PATCHES = 4096
MAX_BANDS = 256
MAX_PATCH_CELLS = 1 << 20  # ~1M — prevents billion×billion allocation before bounds catch


def _canon(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _aid(d: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canon(d).encode("utf-8")).hexdigest()


def _strict_01(value: object) -> int:
    if type(value) is not int or value not in (0, 1):
        raise ValueError("binary value must be strict int 0 or 1")
    return int(value)


def _typed_equal(a: object, b: object) -> bool:
    return type(a) is type(b) and a == b


# ---- Multi-class ----


@dataclass(frozen=True, slots=True)
class MultiClassAdapter:
    """One-vs-rest label expansion / vote aggregation for binary TMs.

    source_field: categorical label field
    classes: ordered unique class values (2..256)
    output_prefix: prefix for binary indicator fields  e.g. label__mc_0 ...
    schema: version tag
    """

    source_field: str
    classes: tuple[Any, ...]
    output_prefix: str | None = None
    schema: str = ADAPTER_SCHEMA

    def __post_init__(self) -> None:
        if not self.source_field or any(ord(c) < 0x20 for c in self.source_field):
            raise ValueError("source_field must be nonempty")
        if not isinstance(self.classes, tuple):
            raise ValueError("classes must be tuple")
        if not 2 <= len(self.classes) <= MAX_CLASSES:
            raise ValueError(f"classes must be 2..{MAX_CLASSES}")
        if len(set((type(v), v) for v in self.classes)) != len(self.classes):
            raise ValueError("classes must be unique typed")
        if self.output_prefix is not None and (not self.output_prefix or any(ord(c) < 0x20 for c in self.output_prefix)):
            raise ValueError("output_prefix invalid")
        if self.schema != ADAPTER_SCHEMA:
            raise ValueError("unsupported schema")

    @property
    def adapter_id(self) -> str:
        return _aid(self.to_dict())

    def adapt(self, record: Mapping[str, Any]) -> dict[str, Any]:
        value = record.get(self.source_field)
        prefix = self.output_prefix or f"{self.source_field}__mc"
        out = dict(record)
        # check all output fields for collision before writing
        for idx in range(len(self.classes)):
            for field in (f"{prefix}_{idx}", f"{prefix}_is_{idx}"):
                if field in out:
                    raise ValueError(f"output collision: {field}")
        for meta in (f"{prefix}__label", f"{prefix}__class_count"):
            if meta in out:
                raise ValueError(f"output collision: {meta}")
        for idx, cls_val in enumerate(self.classes):
            out[f"{prefix}_{idx}"] = 1 if _typed_equal(value, cls_val) else 0
            out[f"{prefix}_is_{idx}"] = _typed_equal(value, cls_val)
        out[f"{prefix}__label"] = value
        out[f"{prefix}__class_count"] = len(self.classes)
        return out

    def inverse(self, binary_predictions: Mapping[str, int]) -> Any:
        """Recover class from one-vs-rest votes (ties → lowest index)."""
        prefix = self.output_prefix or f"{self.source_field}__mc"
        best_idx = 0
        best = -1
        for idx in range(len(self.classes)):
            raw = binary_predictions.get(f"{prefix}_{idx}", 0)
            v = _strict_01(raw)
            if v > best:
                best = v
                best_idx = idx
        return self.classes[best_idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": "multiclass_v1",
            "source_field": self.source_field,
            "classes": list(self.classes),
            "output_prefix": self.output_prefix,
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MultiClassAdapter":
        try:
            obj = cls(
                source_field=str(d["source_field"]),
                classes=tuple(d["classes"]),  # type: ignore[arg-type]
                output_prefix=d.get("output_prefix"),
                schema=str(d.get("schema", ADAPTER_SCHEMA)),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"multiclass descriptor invalid: {e}") from e
        if obj.to_dict() != dict(d):
            raise ValueError("multiclass descriptor not canonical")
        return obj

    def iter_adapt(self, records: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
        for r in records:
            yield self.adapt(r)


# ---- Convolutional / patch ----


@dataclass(frozen=True, slots=True)
class PatchAdapter:
    """Sliding-window patch extractor over scalar fields.

    Produces one record per patch (bounded), each with patch-local fields.
    Deterministic, no image dep.
    """

    field_prefix: str  # prefix of fields that form a matrix, e.g. pixel_
    rows: int
    cols: int
    kernel_rows: int
    kernel_cols: int
    stride_rows: int = 1
    stride_cols: int = 1
    output_prefix: str = "patch"
    schema: str = ADAPTER_SCHEMA

    def __post_init__(self) -> None:
        for name in (self.field_prefix, self.output_prefix):
            if not name or any(ord(c) < 0x20 for c in name):
                raise ValueError("prefix must be nonempty")
        for v, n in [
            (self.rows, "rows"),
            (self.cols, "cols"),
            (self.kernel_rows, "kernel_rows"),
            (self.kernel_cols, "kernel_cols"),
            (self.stride_rows, "stride_rows"),
            (self.stride_cols, "stride_cols"),
        ]:
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"{n} must be positive int")
        if self.kernel_rows > self.rows or self.kernel_cols > self.cols:
            raise ValueError("kernel larger than matrix")
        if self.schema != ADAPTER_SCHEMA:
            raise ValueError("unsupported schema")
        # Bound total input cells before any allocation — billion×billion with kernel billion×billion has patch_count 1 but would OOM
        input_cells = self.rows * self.cols
        if input_cells > MAX_PATCH_CELLS:
            raise ValueError(f"rows*cols={input_cells} exceeds {MAX_PATCH_CELLS}")
        kernel_cells = self.kernel_rows * self.kernel_cols
        if kernel_cells > MAX_PATCH_CELLS:
            raise ValueError(f"kernel cells {kernel_cells} exceeds {MAX_PATCH_CELLS}")
        if self.patch_count() > MAX_PATCHES:
            raise ValueError(f"patch count exceeds {MAX_PATCHES}")

    def patch_count(self) -> int:
        return ((self.rows - self.kernel_rows) // self.stride_rows + 1) * (
            (self.cols - self.kernel_cols) // self.stride_cols + 1
        )

    @property
    def adapter_id(self) -> str:
        return _aid(self.to_dict())

    def _field_name(self, r: int, c: int) -> str:
        # supports both _pixel_0000 and prefix_r_c styles
        # try flat index first, then r_c
        flat = r * self.cols + c
        # width 4 as in ImageAdapter
        digits = max(4, len(str(self.rows * self.cols - 1)))
        return f"{self.field_prefix}_{flat:0{digits}d}"

    def iter_patches(self, record: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
        # collect matrix
        matrix: list[list[Any]] = [[None] * self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                fname = self._field_name(r, c)
                # fallback to prefix_r_c if flat not found
                if fname not in record:
                    alt = f"{self.field_prefix}_{r}_{c}"
                    if alt in record:
                        fname = alt
                    else:
                        raise KeyError(f"missing field {fname} for patch adapter")
                matrix[r][c] = record[fname]
        pid = 0
        for pr in range(0, self.rows - self.kernel_rows + 1, self.stride_rows):
            for pc in range(0, self.cols - self.kernel_cols + 1, self.stride_cols):
                # check for silent overwrite — require patch output fields not already in record
                for kr in range(self.kernel_rows):
                    for kc in range(self.kernel_cols):
                        fname = f"{self.output_prefix}_{kr}_{kc}"
                        if fname in record:
                            raise ValueError(f"output collision: {fname}")
                for meta in (f"{self.output_prefix}__id", f"{self.output_prefix}__row", f"{self.output_prefix}__col"):
                    if meta in record:
                        raise ValueError(f"output collision: {meta}")
                patch: dict[str, Any] = dict(record)
                for kr in range(self.kernel_rows):
                    for kc in range(self.kernel_cols):
                        patch[f"{self.output_prefix}_{kr}_{kc}"] = matrix[pr + kr][pc + kc]
                patch[f"{self.output_prefix}__id"] = pid
                patch[f"{self.output_prefix}__row"] = pr
                patch[f"{self.output_prefix}__col"] = pc
                pid += 1
                yield patch

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": "patch_v1",
            "field_prefix": self.field_prefix,
            "rows": self.rows,
            "cols": self.cols,
            "kernel_rows": self.kernel_rows,
            "kernel_cols": self.kernel_cols,
            "stride_rows": self.stride_rows,
            "stride_cols": self.stride_cols,
            "output_prefix": self.output_prefix,
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PatchAdapter":
        try:
            obj = cls(
                field_prefix=str(d["field_prefix"]),
                rows=int(d["rows"]),
                cols=int(d["cols"]),
                kernel_rows=int(d["kernel_rows"]),
                kernel_cols=int(d["kernel_cols"]),
                stride_rows=int(d.get("stride_rows", 1)),
                stride_cols=int(d.get("stride_cols", 1)),
                output_prefix=str(d.get("output_prefix", "patch")),
                schema=str(d.get("schema", ADAPTER_SCHEMA)),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"patch descriptor invalid: {e}") from e
        if obj.to_dict() != dict(d):
            raise ValueError("patch descriptor not canonical")
        return obj


# ---- Regression (thermometer / banding) ----


@dataclass(frozen=True, slots=True)
class RegressionAdapter:
    """Thermometer encodes continuous target into Boolean bands and back.

    thresholds: sorted strictly increasing finite numbers, 1..256
    source_field: numeric field to encode
    output_prefix: prefix for band fields  e.g. y__band_0 ...
    """

    source_field: str
    thresholds: tuple[float, ...]
    output_prefix: str | None = None
    schema: str = ADAPTER_SCHEMA

    def __post_init__(self) -> None:
        if not self.source_field or any(ord(c) < 0x20 for c in self.source_field):
            raise ValueError("source_field nonempty")
        if not isinstance(self.thresholds, tuple):
            raise ValueError("thresholds must be tuple")
        if not 1 <= len(self.thresholds) <= MAX_BANDS:
            raise ValueError(f"thresholds 1..{MAX_BANDS}")
        prev = None
        for t in self.thresholds:
            if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(float(t)):
                raise ValueError("thresholds must be finite numbers")
            if prev is not None and not float(t) > float(prev):
                raise ValueError("thresholds must be strictly increasing")
            prev = t
        if self.output_prefix is not None and (not self.output_prefix or any(ord(c) < 0x20 for c in self.output_prefix)):
            raise ValueError("output_prefix invalid")
        if self.schema != ADAPTER_SCHEMA:
            raise ValueError("unsupported schema")

    @property
    def adapter_id(self) -> str:
        return _aid(self.to_dict())

    def adapt(self, record: Mapping[str, Any]) -> dict[str, Any]:
        val = record.get(self.source_field)
        if val is None:
            raise ValueError("regression source is missing")
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise TypeError("regression source must be numeric")
        fv = float(val)
        if not math.isfinite(fv):
            raise ValueError("regression source must be finite")
        prefix = self.output_prefix or f"{self.source_field}__band"
        out = dict(record)
        # Collision-check band fields AND metadata fields before any write
        for idx in range(len(self.thresholds)):
            field = f"{prefix}_{idx}"
            if field in out:
                raise ValueError(f"output collision {field}")
        for meta in (f"{prefix}__count", f"{prefix}__value"):
            if meta in out:
                raise ValueError(f"output collision {meta}")
        for idx, thr in enumerate(self.thresholds):
            field = f"{prefix}_{idx}"
            out[field] = 1 if fv >= float(thr) else 0
        out[f"{prefix}__count"] = len(self.thresholds)
        out[f"{prefix}__value"] = fv
        return out

    def inverse(self, band_predictions: Mapping[str, int]) -> float:
        """Decode thermometer votes to scalar: midpoints between thresholds."""
        prefix = self.output_prefix or f"{self.source_field}__band"
        # Validate every band is strict 0/1 and thermometer is monotone 1*0* — reject 1,0,garbage or 1,0,1
        values: list[int] = []
        for idx in range(len(self.thresholds)):
            raw = band_predictions.get(f"{prefix}_{idx}", 0)
            v = _strict_01(raw)
            values.append(v)
        # Must be leading ones then zeros
        try:
            first_zero = values.index(0)
        except ValueError:
            first_zero = len(values)
        # After first zero, no further 1 allowed
        if any(values[i] == 1 for i in range(first_zero + 1, len(values))):
            raise ValueError("thermometer predictions must be monotone 1*0* (not 1,0,1)")
        ones = first_zero if 0 in values else len(values)
        if ones == 0:
            return float(self.thresholds[0]) - 1.0  # below lowest
        if ones == len(self.thresholds):
            return float(self.thresholds[-1]) + 1.0
        # between thresholds[ones-1] and thresholds[ones]
        lo = float(self.thresholds[ones - 1])
        hi = float(self.thresholds[ones])
        return (lo + hi) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": "regression_v1",
            "source_field": self.source_field,
            "thresholds": list(self.thresholds),
            "output_prefix": self.output_prefix,
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RegressionAdapter":
        try:
            obj = cls(
                source_field=str(d["source_field"]),
                thresholds=tuple(float(x) for x in d["thresholds"]),  # type: ignore[arg-type]
                output_prefix=d.get("output_prefix"),
                schema=str(d.get("schema", ADAPTER_SCHEMA)),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"regression descriptor invalid: {e}") from e
        # canonical check: allow int vs float repr via canon
        if _canon(obj.to_dict()) != _canon(dict(d)):
            raise ValueError("regression descriptor not canonical")
        return obj

    def iter_adapt(self, records: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
        for r in records:
            yield self.adapt(r)


# Graph adapter is already in prolog_tsetlin.graph.connectors.GraphConnector
# Re-export for convenience
try:
    from .graph.connectors import GraphConnector  # noqa: F401
    from .graph.types import GraphInput  # noqa: F401
except Exception:  # pragma: no cover
    GraphConnector = None  # type: ignore
    GraphInput = None  # type: ignore

__all__ = ["MultiClassAdapter", "PatchAdapter", "RegressionAdapter", "GraphConnector", "GraphInput", "ADAPTER_SCHEMA"]
