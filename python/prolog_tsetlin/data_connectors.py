"""Bounded streaming connectors and deterministic image/token adapters."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any


RECORD_CONNECTOR_SCHEMA = "ptm.records.v1"
TOKEN_ADAPTER_SCHEMA = "ptm.token_adapter.v1"
IMAGE_ADAPTER_SCHEMA = "ptm.image_adapter.v1"
MAX_CONNECTOR_BATCH_SIZE = 65_536
MAX_TOKEN_INPUT_CHARS = 1_048_576
MAX_TOKEN_PATTERN_CHARS = 512
MAX_TOKENS = 65_536
MAX_TOKEN_CHARS = 1_024
MAX_IMAGE_SOURCE_PIXELS = 16_777_216
MAX_IMAGE_OUTPUT_VALUES = 65_536
DEFAULT_TOKEN_PATTERN = r"[^\W_]+(?:['’][^\W_]+)*"


class ConnectorError(ValueError):
    """Raised when a connector cannot produce the declared typed record stream."""


def _batch_size(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("connector batch size must be an integer")
    if not 1 <= value <= MAX_CONNECTOR_BATCH_SIZE:
        raise ValueError(
            f"connector batch size must be between 1 and {MAX_CONNECTOR_BATCH_SIZE}"
        )
    return value


def _pyarrow() -> Any:
    try:
        import pyarrow as pa
    except ModuleNotFoundError as error:
        raise ConnectorError(
            "Arrow support is not installed; run "
            "`python -m pip install 'prolog-tsetlin-machine[data]'`"
        ) from error
    return pa


def _selected_columns(columns: Sequence[str] | None) -> tuple[str, ...] | None:
    if columns is None:
        return None
    result = tuple(columns)
    if not result or any(not isinstance(name, str) or not name for name in result):
        raise ValueError("connector columns must be nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError("connector columns must be unique")
    return result


def iter_arrow_records(
    source: object,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 1024,
) -> Iterator[dict[str, object]]:
    """Yield Python mappings from Arrow tables, batches, readers, or batches."""

    pa = _pyarrow()
    limit = _batch_size(batch_size)
    selected = _selected_columns(columns)
    if isinstance(source, pa.Table):
        batches: Iterable[object] = source.to_batches(max_chunksize=limit)
    elif isinstance(source, pa.RecordBatch):
        batches = (source,)
    elif isinstance(source, pa.RecordBatchReader):
        batches = source
    elif isinstance(source, Iterable) and not isinstance(source, (str, bytes, Mapping)):
        batches = source
    else:
        raise ConnectorError(
            "Arrow source must be a Table, RecordBatch, RecordBatchReader, "
            "or iterable of RecordBatch values"
        )

    for raw_batch in batches:
        if not isinstance(raw_batch, pa.RecordBatch):
            raise ConnectorError("Arrow batch stream yielded a non-RecordBatch value")
        names = tuple(raw_batch.schema.names)
        if len(set(names)) != len(names):
            raise ConnectorError("Arrow records require unique top-level column names")
        if selected is not None:
            missing = tuple(name for name in selected if name not in names)
            if missing:
                raise ConnectorError(
                    "Arrow source is missing selected columns: " + ", ".join(missing)
                )
            raw_batch = raw_batch.select(selected)
        for first in range(0, raw_batch.num_rows, limit):
            page = raw_batch.slice(first, min(limit, raw_batch.num_rows - first))
            rows = page.to_pylist()
            if len(rows) > limit:
                raise AssertionError("Arrow page exceeded its requested bound")
            for row in rows:
                if not isinstance(row, dict) or any(
                    not isinstance(key, str) for key in row
                ):
                    raise ConnectorError("Arrow row did not materialize as a string-keyed object")
                yield row


def iter_parquet_records(
    source: str | Path | Sequence[str | Path],
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 1024,
    filter_expression: object | None = None,
    use_threads: bool = True,
) -> Iterator[dict[str, object]]:
    """Stream one file, a file list, or a directory through Arrow Dataset."""

    _pyarrow()
    try:
        import pyarrow.dataset as ds
    except ModuleNotFoundError as error:
        raise ConnectorError("installed PyArrow does not include Dataset support") from error
    limit = _batch_size(batch_size)
    selected = _selected_columns(columns)
    if isinstance(source, (str, Path)):
        normalized: str | list[str] = str(source)
    elif isinstance(source, Sequence) and source:
        normalized = [str(path) for path in source]
    else:
        raise ValueError("Parquet source must be a path or nonempty path sequence")
    arrow_exception = getattr(_pyarrow(), "ArrowException", Exception)
    try:
        dataset = ds.dataset(normalized, format="parquet")
        scanner = dataset.scanner(
            columns=None if selected is None else list(selected),
            filter=filter_expression,
            batch_size=limit,
            use_threads=use_threads,
        )
    except (arrow_exception, OSError, ValueError) as error:
        raise ConnectorError(f"could not open Parquet dataset: {error}") from error
    yield from iter_arrow_records(scanner.to_batches(), batch_size=limit)


@dataclass(frozen=True, slots=True)
class TokenAdapter:
    source_field: str
    output_field: str = "tokens"
    normalization: str = "NFKC"
    casefold: bool = True
    pattern: str = DEFAULT_TOKEN_PATTERN
    max_tokens: int = 4096
    overflow: str = "error"
    schema: str = TOKEN_ADAPTER_SCHEMA
    unicode_version: str = unicodedata.unidata_version
    _compiled: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema != TOKEN_ADAPTER_SCHEMA:
            raise ValueError("unsupported token adapter schema")
        if (
            not self.source_field
            or not self.output_field
            or any(ord(character) < 0x20 for character in self.source_field)
            or any(ord(character) < 0x20 for character in self.output_field)
        ):
            raise ValueError("token adapter fields must be nonempty")
        if self.normalization not in ("NFC", "NFKC", "none"):
            raise ValueError("token normalization must be NFC, NFKC, or none")
        if type(self.casefold) is not bool:
            raise ValueError("token casefold must be Boolean")
        if not self.pattern or len(self.pattern) > MAX_TOKEN_PATTERN_CHARS:
            raise ValueError("token pattern is empty or exceeds its size ceiling")
        if not 1 <= self.max_tokens <= MAX_TOKENS:
            raise ValueError(f"max_tokens must be between 1 and {MAX_TOKENS}")
        if self.overflow not in ("error", "truncate"):
            raise ValueError("token overflow policy must be error or truncate")
        if self.unicode_version != unicodedata.unidata_version:
            raise ValueError(
                "token adapter Unicode database version differs from this runtime"
            )
        try:
            compiled = re.compile(self.pattern, re.UNICODE)
        except re.error as error:
            raise ValueError(f"token pattern is invalid: {error}") from error
        object.__setattr__(self, "_compiled", compiled)

    @property
    def adapter_id(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def tokenize(self, text: str) -> tuple[str, ...]:
        if not isinstance(text, str):
            raise ConnectorError("token adapter input must be text")
        if len(text) > MAX_TOKEN_INPUT_CHARS:
            raise ConnectorError("token adapter input exceeds its character ceiling")
        normalized = (
            text
            if self.normalization == "none"
            else unicodedata.normalize(self.normalization, text)
        )
        if self.casefold:
            normalized = normalized.casefold()
        tokens: list[str] = []
        for match in self._compiled.finditer(normalized):
            token = match.group(0)
            if len(token) > MAX_TOKEN_CHARS:
                raise ConnectorError("token exceeds its character ceiling")
            if len(tokens) == self.max_tokens:
                if self.overflow == "truncate":
                    break
                raise ConnectorError("token count exceeds the adapter ceiling")
            tokens.append(token)
        return tuple(tokens)

    def adapt(self, record: Mapping[str, object]) -> dict[str, object]:
        if self.output_field in record or f"{self.output_field}__count" in record:
            raise ConnectorError("token adapter output collides with an input field")
        raw = record.get(self.source_field)
        tokens = () if raw is None else self.tokenize(raw)  # type: ignore[arg-type]
        result = dict(record)
        result[self.output_field] = tokens
        result[f"{self.output_field}__count"] = len(tokens)
        return result

    def iter_adapt(
        self, records: Iterable[Mapping[str, object]]
    ) -> Iterator[dict[str, object]]:
        for record in records:
            yield self.adapt(record)

    def to_dict(self) -> dict[str, Any]:
        return {
            "casefold": self.casefold,
            "max_tokens": self.max_tokens,
            "normalization": self.normalization,
            "output_field": self.output_field,
            "overflow": self.overflow,
            "pattern": self.pattern,
            "schema": self.schema,
            "source_field": self.source_field,
            "unicode_version": self.unicode_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TokenAdapter":
        try:
            result = cls(
                source_field=value["source_field"],
                output_field=value["output_field"],
                normalization=value["normalization"],
                casefold=value["casefold"],
                pattern=value["pattern"],
                max_tokens=value["max_tokens"],
                overflow=value["overflow"],
                schema=value["schema"],
                unicode_version=value["unicode_version"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("token adapter descriptor is invalid") from error
        if result.to_dict() != dict(value):
            raise ValueError("token adapter descriptor is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ImageAdapter:
    width: int
    height: int
    mode: str = "L"
    output_prefix: str = "image"
    resample: str = "nearest"
    max_source_pixels: int = MAX_IMAGE_SOURCE_PIXELS
    schema: str = IMAGE_ADAPTER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != IMAGE_ADAPTER_SCHEMA:
            raise ValueError("unsupported image adapter schema")
        if (
            not isinstance(self.width, int)
            or isinstance(self.width, bool)
            or not isinstance(self.height, int)
            or isinstance(self.height, bool)
            or self.width <= 0
            or self.height <= 0
        ):
            raise ValueError("image target dimensions must be positive integers")
        if self.mode not in ("L", "RGB"):
            raise ValueError("image mode must be L or RGB")
        if not self.output_prefix or any(
            ord(character) < 0x20 for character in self.output_prefix
        ):
            raise ValueError("image output prefix must be nonempty")
        if self.resample != "nearest":
            raise ValueError("image adapter v1 supports only deterministic nearest resize")
        channels = 1 if self.mode == "L" else 3
        if self.width * self.height * channels > MAX_IMAGE_OUTPUT_VALUES:
            raise ValueError(
                f"image adapter exceeds {MAX_IMAGE_OUTPUT_VALUES} output values"
            )
        if not 1 <= self.max_source_pixels <= MAX_IMAGE_SOURCE_PIXELS:
            raise ValueError("image source-pixel ceiling is invalid")

    @property
    def channel_count(self) -> int:
        return 1 if self.mode == "L" else 3

    @property
    def adapter_id(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def _materialize(
        self,
        values: Sequence[object],
        record: Mapping[str, object] | None,
    ) -> dict[str, object]:
        expected = self.width * self.height
        if len(values) != expected:
            raise ConnectorError(
                f"image adapter expected {expected} pixels, received {len(values)}"
            )
        result = {} if record is None else dict(record)
        metadata = {
            f"{self.output_prefix}__width": self.width,
            f"{self.output_prefix}__height": self.height,
            f"{self.output_prefix}__mode": self.mode,
        }
        if any(name in result for name in metadata):
            raise ConnectorError("image adapter metadata collides with an input field")
        result.update(metadata)
        digits = max(4, len(str(max(0, expected - 1))))
        suffixes = ("r", "g", "b")
        for index, raw in enumerate(values):
            if self.mode == "L":
                components = (raw,)
                names = (f"{self.output_prefix}_pixel_{index:0{digits}d}",)
            else:
                if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                    raise ConnectorError("RGB image pixel must have three channels")
                components = tuple(raw)
                if len(components) != 3:
                    raise ConnectorError("RGB image pixel must have three channels")
                names = tuple(
                    f"{self.output_prefix}_pixel_{index:0{digits}d}_{suffix}"
                    for suffix in suffixes
                )
            for name, component in zip(names, components):
                if name in result:
                    raise ConnectorError("image adapter pixel collides with an input field")
                if (
                    not isinstance(component, int)
                    or isinstance(component, bool)
                    or not 0 <= component <= 255
                ):
                    raise ConnectorError("image channel values must be integers from 0 to 255")
                result[name] = component
        return result

    def adapt_pixels(
        self,
        values: Sequence[object],
        *,
        record: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        return self._materialize(values, record)

    def adapt_image(
        self,
        source: str | Path | object,
        *,
        record: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        try:
            from PIL import Image
        except ModuleNotFoundError as error:
            raise ConnectorError(
                "image support is not installed; run "
                "`python -m pip install 'prolog-tsetlin-machine[data]'`"
            ) from error

        opened = None
        image = source
        if isinstance(source, (str, Path)):
            try:
                opened = Image.open(source)
                image = opened
            except (OSError, ValueError) as error:
                raise ConnectorError(f"could not open image {source!s}: {error}") from error
        try:
            if not isinstance(image, Image.Image):
                raise ConnectorError("image source must be a path or Pillow Image")
            if image.width * image.height > self.max_source_pixels:
                raise ConnectorError("image exceeds the source-pixel ceiling")
            converted = image.convert(self.mode)
            resized = converted.resize(
                (self.width, self.height), resample=Image.Resampling.NEAREST
            )
            getter = getattr(resized, "get_flattened_data", resized.getdata)
            values = tuple(getter())
            return self._materialize(values, record)
        finally:
            if opened is not None:
                opened.close()

    def iter_paths(
        self, paths: Iterable[str | Path]
    ) -> Iterator[dict[str, object]]:
        for path in paths:
            yield self.adapt_image(path, record={"image_path": str(path)})

    def to_dict(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "max_source_pixels": self.max_source_pixels,
            "mode": self.mode,
            "output_prefix": self.output_prefix,
            "resample": self.resample,
            "schema": self.schema,
            "width": self.width,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ImageAdapter":
        try:
            result = cls(
                width=value["width"],
                height=value["height"],
                mode=value["mode"],
                output_prefix=value["output_prefix"],
                resample=value["resample"],
                max_source_pixels=value["max_source_pixels"],
                schema=value["schema"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("image adapter descriptor is invalid") from error
        if result.to_dict() != dict(value):
            raise ValueError("image adapter descriptor is not canonical")
        return result
