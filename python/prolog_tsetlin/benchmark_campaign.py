"""Small, deterministic substrate for PTM's internal benchmark campaigns.

The campaign layer deliberately keeps logical benchmark material separate from
wrapper-native layouts.  Dataset manifests identify the exact Boolean rows and
labels; subprocess wrappers may materialize those rows differently, but must
return predictions for independent verification by the runner.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Iterable, Mapping, Sequence

from ._bounded_process import (
    BoundedProcessError,
    BoundedProcessOutputLimit,
    BoundedProcessTimeout,
    run_bounded_process,
)
from .model_generation import canonical_json_bytes, content_digest
from .reference import ScalarBinaryTsetlinMachine
from .services._atomic import publish_bytes


CAMPAIGN_DATASET_SCHEMA = "ptm.campaign-dataset.v1"
CAMPAIGN_REQUEST_SCHEMA = "ptm.campaign-run-request.v1"
CAMPAIGN_WRAPPER_RESULT_SCHEMA = "ptm.campaign-wrapper-result.v1"
CAMPAIGN_RUN_SCHEMA = "ptm.campaign-run.v1"
DENSE_BIT_TEXT_FORMAT = "ptm.dense-bit-text.v1"
_SHA256_PREFIX = "sha256:"
_MASK64 = (1 << 64) - 1
_MAX_MANIFEST_BYTES = 4 << 20
_MAX_RESULT_BYTES = 4 << 20
_TIMING_BOUNDARIES = (
    "preprocessing_materialization_s",
    "adaptive_training_s",
    "diagnostic_collection_s",
    "resident_inference_samples_s",
    "pta_lifecycle_episode_s",
)


class BenchmarkCampaignError(ValueError):
    """Raised when campaign material or wrapper output violates its contract."""


def _require_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise BenchmarkCampaignError(f"{label} must be a nonempty string")
    return value


def _require_identifier(value: object, label: str) -> str:
    result = _require_string(value, label)
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in result):
        raise BenchmarkCampaignError(f"{label} contains unsupported characters")
    return result


def _require_digest(value: object, label: str) -> str:
    result = _require_string(value, label)
    if (
        not result.startswith(_SHA256_PREFIX)
        or len(result) != len(_SHA256_PREFIX) + 64
        or any(character not in "0123456789abcdef" for character in result[7:])
    ):
        raise BenchmarkCampaignError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _bytes_digest(value: bytes) -> str:
    return _SHA256_PREFIX + hashlib.sha256(value).hexdigest()


def _read_json_object(path: Path, *, maximum_bytes: int) -> dict[str, object]:
    data = path.read_bytes()
    if len(data) > maximum_bytes:
        raise BenchmarkCampaignError(f"JSON object exceeds {maximum_bytes} bytes")
    try:
        decoded = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkCampaignError("JSON object is malformed") from error
    if not isinstance(decoded, dict):
        raise BenchmarkCampaignError("JSON value must be an object")
    return decoded


def _safe_relative_path(value: object, label: str) -> Path:
    text = _require_string(value, label)
    path = Path(text)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise BenchmarkCampaignError(f"{label} must be a contained relative path")
    return path


def _resolve_contained(root: Path, relative: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise BenchmarkCampaignError(f"{label} escapes its declared root") from error
    return resolved


def _strict_bit(value: object, label: str) -> int:
    if value is True or (type(value) is int and value == 1):
        return 1
    if value is False or (type(value) is int and value == 0):
        return 0
    raise BenchmarkCampaignError(f"{label} must be Boolean")


def _canonical_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise BenchmarkCampaignError(f"{label} must be a string-keyed mapping")
    result = dict(value)
    try:
        canonical_json_bytes(result)
    except (TypeError, ValueError) as error:
        raise BenchmarkCampaignError(f"{label} is not finite canonical JSON") from error
    return result


@dataclass(frozen=True, slots=True)
class DenseBitSplitReceipt:
    path: str
    row_count: int
    file_digest: str
    feature_digest: str
    label_digest: str
    positive_count: int
    feature_ones: int
    feature_cells: int

    def __post_init__(self) -> None:
        _safe_relative_path(self.path, "split path")
        if type(self.row_count) is not int or self.row_count <= 0:
            raise BenchmarkCampaignError("split row count must be positive")
        for value, label in (
            (self.file_digest, "split file digest"),
            (self.feature_digest, "split feature digest"),
            (self.label_digest, "split label digest"),
        ):
            _require_digest(value, label)
        if (
            type(self.positive_count) is not int
            or not 0 <= self.positive_count <= self.row_count
        ):
            raise BenchmarkCampaignError("split positive count is invalid")
        if (
            type(self.feature_ones) is not int
            or type(self.feature_cells) is not int
            or not 0 <= self.feature_ones <= self.feature_cells
            or self.feature_cells < self.row_count
        ):
            raise BenchmarkCampaignError("split feature density counts are invalid")

    @property
    def density(self) -> float:
        return self.feature_ones / self.feature_cells

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "row_count": self.row_count,
            "file_digest": self.file_digest,
            "feature_digest": self.feature_digest,
            "label_digest": self.label_digest,
            "positive_count": self.positive_count,
            "feature_ones": self.feature_ones,
            "feature_cells": self.feature_cells,
        }

    @classmethod
    def from_dict(cls, value: object) -> "DenseBitSplitReceipt":
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "row_count",
            "file_digest",
            "feature_digest",
            "label_digest",
            "positive_count",
            "feature_ones",
            "feature_cells",
        }:
            raise BenchmarkCampaignError("split receipt fields are not canonical")
        return cls(
            path=value["path"],  # type: ignore[arg-type]
            row_count=value["row_count"],  # type: ignore[arg-type]
            file_digest=value["file_digest"],  # type: ignore[arg-type]
            feature_digest=value["feature_digest"],  # type: ignore[arg-type]
            label_digest=value["label_digest"],  # type: ignore[arg-type]
            positive_count=value["positive_count"],  # type: ignore[arg-type]
            feature_ones=value["feature_ones"],  # type: ignore[arg-type]
            feature_cells=value["feature_cells"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CampaignDatasetManifest:
    dataset_id: str
    variant_id: str
    source: Mapping[str, object]
    representation_id: str
    feature_count: int
    splits: tuple[tuple[str, DenseBitSplitReceipt], ...]
    schema: str = CAMPAIGN_DATASET_SCHEMA
    format: str = DENSE_BIT_TEXT_FORMAT

    def __post_init__(self) -> None:
        if self.schema != CAMPAIGN_DATASET_SCHEMA:
            raise BenchmarkCampaignError("campaign dataset schema is unsupported")
        if self.format != DENSE_BIT_TEXT_FORMAT:
            raise BenchmarkCampaignError("campaign dataset format is unsupported")
        _require_identifier(self.dataset_id, "dataset ID")
        _require_identifier(self.variant_id, "variant ID")
        _require_identifier(self.representation_id, "representation ID")
        if type(self.feature_count) is not int or self.feature_count <= 0:
            raise BenchmarkCampaignError("feature count must be positive")
        if not isinstance(self.source, Mapping) or not self.source:
            raise BenchmarkCampaignError("source receipt must be a nonempty mapping")
        try:
            canonical_json_bytes(dict(self.source))
        except (TypeError, ValueError) as error:
            raise BenchmarkCampaignError("source receipt is not canonical JSON") from error
        names = []
        for name, receipt in self.splits:
            names.append(_require_identifier(name, "split name"))
            if not isinstance(receipt, DenseBitSplitReceipt):
                raise BenchmarkCampaignError("split receipt has the wrong type")
            if receipt.feature_cells != receipt.row_count * self.feature_count:
                raise BenchmarkCampaignError("split feature-cell count disagrees with shape")
        if not names or len(names) != len(set(names)) or tuple(sorted(names)) != tuple(names):
            raise BenchmarkCampaignError("split names must be unique and sorted")

    @property
    def split_map(self) -> dict[str, DenseBitSplitReceipt]:
        return dict(self.splits)

    @property
    def representation_digest(self) -> str:
        return content_digest(
            {
                "format": self.format,
                "feature_count": self.feature_count,
                "representation_id": self.representation_id,
                "features": {
                    name: receipt.feature_digest for name, receipt in self.splits
                },
            }
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dataset_id": self.dataset_id,
            "variant_id": self.variant_id,
            "source": dict(self.source),
            "representation": {
                "id": self.representation_id,
                "format": self.format,
                "feature_count": self.feature_count,
                "digest": self.representation_digest,
            },
            "splits": {name: receipt.to_dict() for name, receipt in self.splits},
        }

    @property
    def manifest_digest(self) -> str:
        return content_digest(self._identity_payload())

    def to_dict(self) -> dict[str, object]:
        return self._identity_payload() | {"manifest_digest": self.manifest_digest}

    @classmethod
    def from_dict(cls, value: object) -> "CampaignDatasetManifest":
        if not isinstance(value, Mapping) or set(value) != {
            "schema",
            "dataset_id",
            "variant_id",
            "source",
            "representation",
            "splits",
            "manifest_digest",
        }:
            raise BenchmarkCampaignError("dataset manifest fields are not canonical")
        representation = value["representation"]
        raw_splits = value["splits"]
        if not isinstance(representation, Mapping) or set(representation) != {
            "id",
            "format",
            "feature_count",
            "digest",
        }:
            raise BenchmarkCampaignError("representation receipt fields are not canonical")
        if not isinstance(raw_splits, Mapping):
            raise BenchmarkCampaignError("dataset splits must be a mapping")
        result = cls(
            schema=value["schema"],  # type: ignore[arg-type]
            dataset_id=value["dataset_id"],  # type: ignore[arg-type]
            variant_id=value["variant_id"],  # type: ignore[arg-type]
            source=value["source"],  # type: ignore[arg-type]
            representation_id=representation["id"],  # type: ignore[arg-type]
            format=representation["format"],  # type: ignore[arg-type]
            feature_count=representation["feature_count"],  # type: ignore[arg-type]
            splits=tuple(
                (str(name), DenseBitSplitReceipt.from_dict(receipt))
                for name, receipt in sorted(raw_splits.items())
            ),
        )
        if representation["digest"] != result.representation_digest:
            raise BenchmarkCampaignError("representation digest mismatch")
        if value["manifest_digest"] != result.manifest_digest:
            raise BenchmarkCampaignError("dataset manifest digest mismatch")
        return result

    @classmethod
    def load(cls, path: str | Path, *, verify_files: bool = True) -> "CampaignDatasetManifest":
        manifest_path = Path(path)
        result = cls.from_dict(
            _read_json_object(manifest_path, maximum_bytes=_MAX_MANIFEST_BYTES)
        )
        if verify_files:
            for name in result.split_map:
                load_dense_bit_split(manifest_path, result, name)
        return result

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        return publish_bytes(
            target,
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n",
            overwrite=True,
        )


class _SplitMix64:
    """Small PTM-owned deterministic generator used only for campaign material."""

    def __init__(self, seed: int) -> None:
        if type(seed) is not int:
            raise TypeError("generator seed must be an integer")
        self._state = seed & _MASK64

    def next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & _MASK64
        value = self._state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
        return (value ^ (value >> 31)) & _MASK64

    def randbelow(self, bound: int) -> int:
        if type(bound) is not int or not 0 < bound <= 1 << 63:
            raise ValueError("random bound is invalid")
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % bound


def _derived_seed(seed: int, label: str) -> int:
    payload = f"ptm.campaign.seed.v1\0{seed}\0{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _permutation(length: int, seed: int) -> list[int]:
    values = list(range(length))
    generator = _SplitMix64(seed)
    for index in range(length - 1, 0, -1):
        selected = generator.randbelow(index + 1)
        values[index], values[selected] = values[selected], values[index]
    return values


def _dense_split_bytes(
    rows: Sequence[Sequence[int | bool]], labels: Sequence[int | bool]
) -> tuple[bytes, DenseBitSplitReceipt]:
    if not rows or len(rows) != len(labels):
        raise BenchmarkCampaignError("dense split rows and labels must be nonempty and aligned")
    width = len(rows[0])
    if width <= 0 or any(len(row) != width for row in rows):
        raise BenchmarkCampaignError("dense split rows must have one stable positive width")
    output = bytearray()
    feature_hash = hashlib.sha256()
    label_hash = hashlib.sha256()
    positive_count = 0
    feature_ones = 0
    for row, raw_label in zip(rows, labels):
        features: list[str] = []
        for raw_feature in row:
            feature = _strict_bit(raw_feature, "dense split feature")
            features.append(str(feature))
            feature_ones += feature
        label = _strict_bit(raw_label, "dense split label")
        feature_line = " ".join(features).encode("ascii") + b"\n"
        label_line = str(label).encode("ascii") + b"\n"
        feature_hash.update(feature_line)
        label_hash.update(label_line)
        output.extend(feature_line[:-1])
        output.extend(b" ")
        output.extend(label_line)
        positive_count += label
    data = bytes(output)
    receipt = DenseBitSplitReceipt(
        path="pending",
        row_count=len(rows),
        file_digest=_bytes_digest(data),
        feature_digest=_SHA256_PREFIX + feature_hash.hexdigest(),
        label_digest=_SHA256_PREFIX + label_hash.hexdigest(),
        positive_count=positive_count,
        feature_ones=feature_ones,
        feature_cells=len(rows) * width,
    )
    return data, receipt


def _write_dense_split(
    directory: Path,
    name: str,
    rows: Sequence[Sequence[int | bool]],
    labels: Sequence[int | bool],
) -> DenseBitSplitReceipt:
    data, pending = _dense_split_bytes(rows, labels)
    filename = f"{name}.txt"
    publish_bytes(directory / filename, data, overwrite=True)
    return DenseBitSplitReceipt(
        path=filename,
        row_count=pending.row_count,
        file_digest=pending.file_digest,
        feature_digest=pending.feature_digest,
        label_digest=pending.label_digest,
        positive_count=pending.positive_count,
        feature_ones=pending.feature_ones,
        feature_cells=pending.feature_cells,
    )


def _inspect_dense_source(data: bytes, feature_count: int) -> DenseBitSplitReceipt:
    rows: list[tuple[int, ...]] = []
    labels: list[int] = []
    for line_number, raw_line in enumerate(data.splitlines(), 1):
        fields = raw_line.split()
        if len(fields) != feature_count + 1 or any(
            field not in (b"0", b"1") for field in fields
        ):
            raise BenchmarkCampaignError(
                f"dense source row {line_number} has invalid width or values"
            )
        rows.append(tuple(int(field) for field in fields[:-1]))
        labels.append(int(fields[-1]))
    _, normalized = _dense_split_bytes(rows, labels)
    return DenseBitSplitReceipt(
        path="pending",
        row_count=normalized.row_count,
        file_digest=_bytes_digest(data),
        feature_digest=normalized.feature_digest,
        label_digest=normalized.label_digest,
        positive_count=normalized.positive_count,
        feature_ones=normalized.feature_ones,
        feature_cells=normalized.feature_cells,
    )


def import_dense_bit_dataset(
    output_directory: str | Path,
    *,
    dataset_id: str,
    variant_id: str,
    representation_id: str,
    feature_count: int,
    split_paths: Mapping[str, str | Path],
    source: Mapping[str, object],
) -> Path:
    """Copy exact dense-bit sources into one content-identified campaign unit."""

    if type(feature_count) is not int or feature_count <= 0:
        raise BenchmarkCampaignError("import feature count must be positive")
    if not isinstance(split_paths, Mapping) or not split_paths:
        raise BenchmarkCampaignError("import split paths must be a nonempty mapping")
    source_receipt = _canonical_mapping(source, "import source receipt")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    receipts: dict[str, DenseBitSplitReceipt] = {}
    source_files: dict[str, str] = {}
    for split_name, raw_path in sorted(split_paths.items()):
        name = _require_identifier(split_name, "import split name")
        if not isinstance(raw_path, (str, os.PathLike)):
            raise BenchmarkCampaignError("import split path has the wrong type")
        source_path = Path(raw_path).resolve()
        data = source_path.read_bytes()
        pending = _inspect_dense_source(data, feature_count)
        filename = f"{name}.txt"
        publish_bytes(output / filename, data, overwrite=True)
        receipts[name] = DenseBitSplitReceipt(
            path=filename,
            row_count=pending.row_count,
            file_digest=pending.file_digest,
            feature_digest=pending.feature_digest,
            label_digest=pending.label_digest,
            positive_count=pending.positive_count,
            feature_ones=pending.feature_ones,
            feature_cells=pending.feature_cells,
        )
        source_files[name] = pending.file_digest
    source_receipt["files"] = source_files
    source_receipt["digest"] = content_digest(source_receipt)
    manifest = CampaignDatasetManifest(
        dataset_id=dataset_id,
        variant_id=variant_id,
        source=source_receipt,
        representation_id=representation_id,
        feature_count=feature_count,
        splits=tuple(sorted(receipts.items())),
    )
    return manifest.write(output / "manifest.json")


def prepare_local_baselines(
    project_root: str | Path,
    logic_material_directory: str | Path,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Receipt the restored NoisyXOR and four prepared Logic representations."""

    root = Path(project_root).resolve()
    logic_root = Path(logic_material_directory).resolve()
    output = Path(output_directory)
    noisy_root = root / "data" / "NoisyXOR"
    manifests = [
        import_dense_bit_dataset(
            output / "noisy-xor" / "archived",
            dataset_id="ptm.noisy-xor-archive.v1",
            variant_id="archived",
            representation_id="boolean-12",
            feature_count=12,
            split_paths={
                "evaluation": noisy_root / "NoisyXORTestData.txt",
                "train": noisy_root / "NoisyXORTrainingData.txt",
            },
            source={
                "kind": "local-restored-archive",
                "collection": "PrologTsetlinMachineArchive_1",
                "role": "wiring-and-historical-reproduction-anchor",
            },
        )
    ]

    preparation_path = logic_root / "logic_baseline_manifest.json"
    preparation_bytes = preparation_path.read_bytes()
    preparation = _read_json_object(
        preparation_path, maximum_bytes=_MAX_MANIFEST_BYTES
    )
    encodings = preparation.get("encodings")
    split = preparation.get("split")
    source_digest = preparation.get("source_digest")
    if not isinstance(encodings, Mapping) or not isinstance(split, Mapping):
        raise BenchmarkCampaignError("Logic preparation manifest is malformed")
    _require_digest(source_digest, "Logic source digest")
    expected_encodings = (
        "ast_relational",
        "position_one_hot",
        "token_count_threshold",
        "token_presence",
    )
    if tuple(sorted(encodings)) != expected_encodings:
        raise BenchmarkCampaignError(
            "Logic preparation manifest must contain all four campaign encodings"
        )
    for encoding_name in expected_encodings:
        details = encodings[encoding_name]
        if not isinstance(details, Mapping):
            raise BenchmarkCampaignError("Logic encoding receipt is malformed")
        feature_count = details.get("feature_count")
        train_file = _safe_relative_path(
            details.get("train_file"), "Logic training file"
        )
        evaluation_file = _safe_relative_path(
            details.get("evaluation_file"), "Logic evaluation file"
        )
        train_digest = _require_digest(
            details.get("train_digest"), "Logic training digest"
        )
        evaluation_digest = _require_digest(
            details.get("evaluation_digest"), "Logic evaluation digest"
        )
        if type(feature_count) is not int or feature_count <= 0:
            raise BenchmarkCampaignError("Logic feature count is invalid")
        train_path = _resolve_contained(
            logic_root, train_file, "Logic training file"
        )
        evaluation_path = _resolve_contained(
            logic_root, evaluation_file, "Logic evaluation file"
        )
        if _bytes_digest(train_path.read_bytes()) != train_digest:
            raise BenchmarkCampaignError("Logic training file digest mismatch")
        if _bytes_digest(evaluation_path.read_bytes()) != evaluation_digest:
            raise BenchmarkCampaignError("Logic evaluation file digest mismatch")
        manifests.append(
            import_dense_bit_dataset(
                output / "logic-pairs" / encoding_name,
                dataset_id="ptm.logic-pairs.v1",
                variant_id=encoding_name.replace("_", "-"),
                representation_id=encoding_name.replace("_", "-"),
                feature_count=feature_count,
                split_paths={
                    "evaluation": evaluation_path,
                    "train": train_path,
                },
                source={
                    "kind": "ptm-logic-preparation",
                    "paired_source_digest": source_digest,
                    "preparation_manifest_digest": _bytes_digest(preparation_bytes),
                    "split": dict(split),
                    "encoding": encoding_name,
                },
            )
        )
    return tuple(manifests)


def load_dense_bit_split(
    manifest_path: str | Path,
    manifest: CampaignDatasetManifest,
    split_name: str,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    receipt = manifest.split_map.get(split_name)
    if receipt is None:
        raise BenchmarkCampaignError(f"dataset split is absent: {split_name}")
    path = _resolve_contained(
        Path(manifest_path).resolve().parent,
        _safe_relative_path(receipt.path, "split path"),
        "split path",
    )
    data = path.read_bytes()
    if _bytes_digest(data) != receipt.file_digest:
        raise BenchmarkCampaignError(f"dataset split file digest mismatch: {split_name}")
    rows: list[tuple[int, ...]] = []
    labels: list[int] = []
    feature_hash = hashlib.sha256()
    label_hash = hashlib.sha256()
    feature_ones = 0
    for line_number, raw_line in enumerate(data.splitlines(), 1):
        fields = raw_line.split()
        if len(fields) != manifest.feature_count + 1 or any(
            field not in (b"0", b"1") for field in fields
        ):
            raise BenchmarkCampaignError(
                f"dense split row {line_number} has invalid width or values"
            )
        row = tuple(int(field) for field in fields[:-1])
        label = int(fields[-1])
        rows.append(row)
        labels.append(label)
        feature_hash.update(b" ".join(fields[:-1]) + b"\n")
        label_hash.update(fields[-1] + b"\n")
        feature_ones += sum(row)
    if len(rows) != receipt.row_count:
        raise BenchmarkCampaignError(f"dataset split row count mismatch: {split_name}")
    if sum(labels) != receipt.positive_count or feature_ones != receipt.feature_ones:
        raise BenchmarkCampaignError(f"dataset split counts mismatch: {split_name}")
    if _SHA256_PREFIX + feature_hash.hexdigest() != receipt.feature_digest:
        raise BenchmarkCampaignError(f"dataset split feature digest mismatch: {split_name}")
    if _SHA256_PREFIX + label_hash.hexdigest() != receipt.label_digest:
        raise BenchmarkCampaignError(f"dataset split label digest mismatch: {split_name}")
    return tuple(rows), tuple(labels)


def prepare_xor_noise(
    output_directory: str | Path,
    *,
    seed: int = 2026082501,
    feature_count: int = 20,
    train_rows: int = 5_000,
    validation_rows: int = 5_000,
    evaluation_rows: int = 10_000,
    noise_basis_points: Sequence[int] = (0, 1_000, 2_000, 3_000, 4_000),
) -> tuple[Path, ...]:
    """Prepare deterministic XOR rows with exact-count training-label noise."""

    if type(feature_count) is not int or feature_count < 2:
        raise BenchmarkCampaignError("XOR feature count must be at least two")
    counts = {"train": train_rows, "validation": validation_rows, "evaluation": evaluation_rows}
    if any(type(count) is not int or count <= 0 for count in counts.values()):
        raise BenchmarkCampaignError("XOR split sizes must be positive integers")
    if sum(counts.values()) > 1 << feature_count:
        raise BenchmarkCampaignError(
            "XOR split sizes exceed the unique Boolean feature domain"
        )
    rates = tuple(noise_basis_points)
    if not rates or len(rates) != len(set(rates)) or any(
        type(rate) is not int or not 0 <= rate <= 10_000 for rate in rates
    ):
        raise BenchmarkCampaignError("XOR noise basis points are invalid")

    rows_by_split: dict[str, tuple[tuple[int, ...], ...]] = {}
    labels_by_split: dict[str, tuple[int, ...]] = {}
    seen_rows: set[tuple[int, ...]] = set()
    for split_name, count in counts.items():
        generator = _SplitMix64(_derived_seed(seed, f"xor:{split_name}"))
        unique_rows: list[tuple[int, ...]] = []
        while len(unique_rows) < count:
            row = tuple(generator.next_u64() & 1 for _ in range(feature_count))
            if row in seen_rows:
                continue
            seen_rows.add(row)
            unique_rows.append(row)
        rows = tuple(unique_rows)
        rows_by_split[split_name] = rows
        labels_by_split[split_name] = tuple(row[0] ^ row[1] for row in rows)

    root = Path(output_directory)
    manifests: list[Path] = []
    for rate in sorted(rates):
        variant_id = f"noise-{rate:05d}bp"
        directory = root / variant_id
        directory.mkdir(parents=True, exist_ok=True)
        train_labels = list(labels_by_split["train"])
        flip_count = (train_rows * rate + 5_000) // 10_000
        flip_order = _permutation(
            train_rows, _derived_seed(seed, f"xor:noise:{rate}")
        )
        for index in flip_order[:flip_count]:
            train_labels[index] ^= 1
        receipts = {
            "evaluation": _write_dense_split(
                directory,
                "evaluation",
                rows_by_split["evaluation"],
                labels_by_split["evaluation"],
            ),
            "train": _write_dense_split(
                directory, "train", rows_by_split["train"], train_labels
            ),
            "validation": _write_dense_split(
                directory,
                "validation",
                rows_by_split["validation"],
                labels_by_split["validation"],
            ),
        }
        source = {
            "kind": "generated",
            "revision": "ptm.splitmix64-xor20.v2",
            "seed": seed,
            "feature_count": feature_count,
            "split_rows": counts,
            "unique_rows_across_splits": True,
            "noise_basis_points": rate,
            "training_label_flips": flip_count,
        }
        manifest = CampaignDatasetManifest(
            dataset_id="synthetic.xor20-noise.v1",
            variant_id=variant_id,
            source=source | {"digest": content_digest(source)},
            representation_id=f"boolean-{feature_count}",
            feature_count=feature_count,
            splits=tuple(sorted(receipts.items())),
        )
        manifests.append(manifest.write(directory / "manifest.json"))
    return tuple(manifests)


def prepare_parity_ladder(
    output_directory: str | Path,
    *,
    seed: int = 2026082502,
    widths: Sequence[int] = (6, 8, 10, 12, 14, 16),
    train_basis_points: int = 6_000,
    validation_basis_points: int = 2_000,
) -> tuple[Path, ...]:
    """Prepare exhaustive parity domains with deterministic held-out splits."""

    widths_tuple = tuple(widths)
    if not widths_tuple or len(widths_tuple) != len(set(widths_tuple)) or any(
        type(width) is not int or not 2 <= width <= 20 for width in widths_tuple
    ):
        raise BenchmarkCampaignError("parity widths are invalid")
    if (
        type(train_basis_points) is not int
        or type(validation_basis_points) is not int
        or train_basis_points <= 0
        or validation_basis_points <= 0
        or train_basis_points + validation_basis_points >= 10_000
    ):
        raise BenchmarkCampaignError("parity split proportions are invalid")

    root = Path(output_directory)
    manifests: list[Path] = []
    for width in sorted(widths_tuple):
        variant_id = f"n-{width:02d}"
        directory = root / variant_id
        directory.mkdir(parents=True, exist_ok=True)
        domain = tuple(
            tuple((value >> bit) & 1 for bit in range(width))
            for value in range(1 << width)
        )
        labels = tuple(sum(row) & 1 for row in domain)
        order = _permutation(len(domain), _derived_seed(seed, f"parity:{width}"))
        train_count = len(domain) * train_basis_points // 10_000
        validation_count = len(domain) * validation_basis_points // 10_000
        split_indices = {
            "train": order[:train_count],
            "validation": order[train_count : train_count + validation_count],
            "evaluation": order[train_count + validation_count :],
        }
        receipts: dict[str, DenseBitSplitReceipt] = {}
        for name, indices in sorted(split_indices.items()):
            receipts[name] = _write_dense_split(
                directory,
                name,
                tuple(domain[index] for index in indices),
                tuple(labels[index] for index in indices),
            )
        source = {
            "kind": "generated",
            "revision": "ptm.splitmix64-exhaustive-parity.v1",
            "seed": seed,
            "width": width,
            "domain_rows": len(domain),
            "split_basis_points": {
                "train": train_basis_points,
                "validation": validation_basis_points,
                "evaluation": 10_000 - train_basis_points - validation_basis_points,
            },
        }
        manifest = CampaignDatasetManifest(
            dataset_id="synthetic.parity-ladder.v1",
            variant_id=variant_id,
            source=source | {"digest": content_digest(source)},
            representation_id=f"boolean-{width}",
            feature_count=width,
            splits=tuple(sorted(receipts.items())),
        )
        manifests.append(manifest.write(directory / "manifest.json"))
    return tuple(manifests)


@dataclass(frozen=True, slots=True)
class CampaignRunRequest:
    campaign_id: str
    run_id: str
    pass_name: str
    track: str
    dataset_manifest: str
    dataset_manifest_digest: str
    train_split: str
    score_splits: tuple[str, ...]
    model: Mapping[str, object]
    output_directory: str
    schema: str = CAMPAIGN_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CAMPAIGN_REQUEST_SCHEMA:
            raise BenchmarkCampaignError("campaign request schema is unsupported")
        for value, label in (
            (self.campaign_id, "campaign ID"),
            (self.run_id, "run ID"),
            (self.pass_name, "pass name"),
            (self.track, "track"),
            (self.train_split, "train split"),
        ):
            _require_identifier(value, label)
        if self.track not in ("shared", "native"):
            raise BenchmarkCampaignError("campaign track must be shared or native")
        _require_digest(self.dataset_manifest_digest, "dataset manifest digest")
        if type(self.dataset_manifest) is not str or not Path(self.dataset_manifest).is_absolute():
            raise BenchmarkCampaignError("dataset manifest path must be absolute")
        if type(self.output_directory) is not str or not Path(self.output_directory).is_absolute():
            raise BenchmarkCampaignError("output directory path must be absolute")
        if (
            not self.score_splits
            or len(self.score_splits) != len(set(self.score_splits))
            or tuple(sorted(self.score_splits)) != self.score_splits
        ):
            raise BenchmarkCampaignError("score splits must be unique and sorted")
        for name in self.score_splits:
            _require_identifier(name, "score split")
        if not isinstance(self.model, Mapping) or not self.model:
            raise BenchmarkCampaignError("model request must be a nonempty mapping")
        for name in ("implementation", "backend", "commit"):
            _require_string(self.model.get(name), f"model {name}")
        if not isinstance(self.model.get("config"), Mapping):
            raise BenchmarkCampaignError("model config must be a mapping")
        try:
            canonical_json_bytes(dict(self.model))
        except (TypeError, ValueError) as error:
            raise BenchmarkCampaignError("model request is not canonical JSON") from error

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "pass": self.pass_name,
            "track": self.track,
            "dataset_manifest": self.dataset_manifest,
            "dataset_manifest_digest": self.dataset_manifest_digest,
            "train_split": self.train_split,
            "score_splits": list(self.score_splits),
            "model": dict(self.model),
            "output_directory": self.output_directory,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CampaignRunRequest":
        if not isinstance(value, Mapping) or set(value) != {
            "schema",
            "campaign_id",
            "run_id",
            "pass",
            "track",
            "dataset_manifest",
            "dataset_manifest_digest",
            "train_split",
            "score_splits",
            "model",
            "output_directory",
        }:
            raise BenchmarkCampaignError("campaign request fields are not canonical")
        score_splits = value["score_splits"]
        if not isinstance(score_splits, list) or any(type(item) is not str for item in score_splits):
            raise BenchmarkCampaignError("campaign score splits are malformed")
        return cls(
            schema=value["schema"],  # type: ignore[arg-type]
            campaign_id=value["campaign_id"],  # type: ignore[arg-type]
            run_id=value["run_id"],  # type: ignore[arg-type]
            pass_name=value["pass"],  # type: ignore[arg-type]
            track=value["track"],  # type: ignore[arg-type]
            dataset_manifest=value["dataset_manifest"],  # type: ignore[arg-type]
            dataset_manifest_digest=value["dataset_manifest_digest"],  # type: ignore[arg-type]
            train_split=value["train_split"],  # type: ignore[arg-type]
            score_splits=tuple(score_splits),
            model=value["model"],  # type: ignore[arg-type]
            output_directory=value["output_directory"],  # type: ignore[arg-type]
        )

    @classmethod
    def load(cls, path: str | Path) -> "CampaignRunRequest":
        return cls.from_dict(_read_json_object(Path(path), maximum_bytes=_MAX_MANIFEST_BYTES))

    def write(self, path: str | Path) -> Path:
        return publish_bytes(
            path,
            canonical_json_bytes(self.to_dict()) + b"\n",
            overwrite=True,
        )


def _prediction_metrics(expected: Sequence[int], actual: Sequence[int]) -> dict[str, object]:
    if len(expected) != len(actual) or not expected:
        raise BenchmarkCampaignError("prediction count does not match nonempty labels")
    if any(value not in (0, 1) for value in actual):
        raise BenchmarkCampaignError("predictions are outside the binary label domain")
    correct = sum(left == right for left, right in zip(expected, actual))
    recalls: list[float] = []
    for label in (0, 1):
        count = sum(value == label for value in expected)
        if count:
            recalls.append(
                sum(left == right == label for left, right in zip(expected, actual))
                / count
            )
    payload = b"".join(f"{value}\n".encode("ascii") for value in actual)
    return {
        "rows": len(expected),
        "accuracy": correct / len(expected),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "prediction_digest": _bytes_digest(payload),
    }


def _read_predictions(path: Path) -> tuple[int, ...]:
    try:
        values = tuple(int(line) for line in path.read_text(encoding="ascii").splitlines())
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise BenchmarkCampaignError("prediction file is malformed") from error
    if any(value not in (0, 1) for value in values):
        raise BenchmarkCampaignError("prediction file contains non-binary values")
    return values


def _read_vote_scores(path: Path) -> tuple[int, ...]:
    try:
        values = tuple(
            int(line) for line in path.read_text(encoding="ascii").splitlines()
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise BenchmarkCampaignError("vote-score file is malformed") from error
    if not values:
        raise BenchmarkCampaignError("vote-score file is empty")
    return values


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    data = canonical_json_bytes(dict(value)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as stream:
        stream.write(data)
        os.fsync(stream.fileno())


def run_campaign_attempt(
    request: CampaignRunRequest,
    wrapper_command: Sequence[str],
    *,
    raw_jsonl: str | Path,
    timeout_seconds: float = 300.0,
    max_output_bytes: int = _MAX_RESULT_BYTES,
) -> dict[str, object]:
    """Run one contained wrapper and retain exactly one attempted-run record."""

    if (
        isinstance(wrapper_command, (str, bytes))
        or not wrapper_command
        or any(type(item) is not str or not item for item in wrapper_command)
    ):
        raise TypeError("wrapper command must be a nonempty string sequence")
    manifest_path = Path(request.dataset_manifest)
    manifest = CampaignDatasetManifest.load(manifest_path)
    if manifest.manifest_digest != request.dataset_manifest_digest:
        raise BenchmarkCampaignError("request dataset manifest identity is stale")
    if request.train_split not in manifest.split_map or any(
        name not in manifest.split_map for name in request.score_splits
    ):
        raise BenchmarkCampaignError("request names a missing dataset split")

    output_root = Path(request.output_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    request_path = output_root / "request.json"
    request.write(request_path)
    stdout = b""
    stderr = b""
    return_code: int | None = None
    failure: dict[str, object] | None = None
    wrapper_result: dict[str, object] | None = None
    started = time.perf_counter()
    try:
        completed = run_bounded_process(
            [*wrapper_command, str(request_path)],
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        return_code = completed.returncode
        if completed.returncode != 0:
            failure = {
                "class": "nonzero_exit",
                "message": "wrapper exited with a nonzero status",
            }
        else:
            try:
                decoded = json.loads(stdout.decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise BenchmarkCampaignError("wrapper result must be one JSON object")
                wrapper_result = decoded
            except (UnicodeDecodeError, json.JSONDecodeError, BenchmarkCampaignError) as error:
                failure = {"class": "invalid_output", "message": str(error)}
    except BoundedProcessError as error:
        stdout = error.stdout
        stderr = error.stderr
        if isinstance(error, BoundedProcessTimeout):
            failure_class = "timeout"
        elif isinstance(error, BoundedProcessOutputLimit):
            failure_class = "output_limit"
        else:
            failure_class = "launch_or_containment"
        failure = {"class": failure_class, "message": str(error)}
    attempt_elapsed = time.perf_counter() - started

    publish_bytes(output_root / "stdout.bin", stdout, overwrite=True)
    publish_bytes(output_root / "stderr.bin", stderr, overwrite=True)
    status = "failed"
    metrics: dict[str, object] = {}
    timing: dict[str, object] = {
        "attempt_wall_s": attempt_elapsed,
        **{boundary: "n/a" for boundary in _TIMING_BOUNDARIES},
    }
    diagnostics: dict[str, object] = {}
    environment: dict[str, object] = {
        "runner_os": platform.platform(),
        "runner_python": platform.python_version(),
    }
    artifacts: dict[str, object] = {
        "request": "request.json",
        "request_digest": _bytes_digest(request_path.read_bytes()),
        "stdout": "stdout.bin",
        "stdout_digest": _bytes_digest(stdout),
        "stderr": "stderr.bin",
        "stderr_digest": _bytes_digest(stderr),
    }

    if wrapper_result is not None and failure is None:
        required = {"schema", "run_id", "status", "predictions", "timing", "diagnostics", "environment", "artifacts", "failure"}
        if set(wrapper_result) != required:
            failure = {
                "class": "invalid_output",
                "message": "wrapper result fields are not canonical",
            }
        elif wrapper_result.get("schema") != CAMPAIGN_WRAPPER_RESULT_SCHEMA:
            failure = {"class": "invalid_output", "message": "wrapper result schema is unsupported"}
        elif wrapper_result.get("run_id") != request.run_id:
            failure = {"class": "invalid_output", "message": "wrapper result run ID mismatch"}
        elif wrapper_result.get("status") not in ("ok", "failed", "unsupported"):
            failure = {"class": "invalid_output", "message": "wrapper status is invalid"}
        else:
            status = str(wrapper_result["status"])
            try:
                wrapper_timing = _canonical_mapping(
                    wrapper_result["timing"], "wrapper timing"
                )
                if any(key not in _TIMING_BOUNDARIES for key in wrapper_timing):
                    raise BenchmarkCampaignError(
                        "wrapper timing contains an undeclared boundary"
                    )
                diagnostics = _canonical_mapping(
                    wrapper_result["diagnostics"], "wrapper diagnostics"
                )
                wrapper_environment = _canonical_mapping(
                    wrapper_result["environment"], "wrapper environment"
                )
                environment.update(wrapper_environment)
                artifacts["wrapper"] = _canonical_mapping(
                    wrapper_result["artifacts"], "wrapper artifacts"
                )
                timing.update(wrapper_timing)
            except BenchmarkCampaignError as error:
                failure = {"class": "invalid_output", "message": str(error)}
            if failure is None and status == "ok":
                if (
                    wrapper_environment.get("source_commit")
                    != request.model["commit"]
                    or wrapper_environment.get("backend_requested")
                    != request.model["backend"]
                    or type(wrapper_environment.get("backend_actual")) is not str
                    or not wrapper_environment["backend_actual"]
                ):
                    failure = {
                        "class": "invalid_output",
                        "message": "wrapper execution identity contradicts the request",
                    }
                elif wrapper_result["failure"] is not None:
                    failure = {
                        "class": "invalid_output",
                        "message": "successful wrapper result carries a failure",
                    }
                predictions = wrapper_result["predictions"]
                if failure is None and (
                    not isinstance(predictions, Mapping)
                    or set(predictions) != set(request.score_splits)
                ):
                    failure = {"class": "invalid_output", "message": "wrapper prediction splits mismatch"}
                elif failure is None:
                    try:
                        for split_name in request.score_splits:
                            relative = _safe_relative_path(predictions[split_name], "prediction path")
                            prediction_path = _resolve_contained(output_root, relative, "prediction path")
                            if not prediction_path.is_file():
                                raise BenchmarkCampaignError("prediction file is absent")
                            _, expected = load_dense_bit_split(manifest_path, manifest, split_name)
                            actual = _read_predictions(prediction_path)
                            metrics[split_name] = _prediction_metrics(expected, actual)
                    except BenchmarkCampaignError as error:
                        failure = {"class": "invalid_predictions", "message": str(error)}
            elif failure is None:
                wrapper_failure = wrapper_result["failure"]
                if type(wrapper_failure) is not str or not wrapper_failure:
                    failure = {
                        "class": "invalid_output",
                        "message": "failed or unsupported wrapper result lacks a failure",
                    }
                elif wrapper_result["predictions"] not in ({}, None):
                    failure = {
                        "class": "invalid_output",
                        "message": "failed or unsupported wrapper result carries predictions",
                    }
                else:
                    failure = {"class": status, "message": wrapper_failure}

    if failure is not None:
        status = "unsupported" if failure.get("class") == "unsupported" else "failed"
    record = {
        "schema": CAMPAIGN_RUN_SCHEMA,
        "campaign_id": request.campaign_id,
        "run_id": request.run_id,
        "attempted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pass": request.pass_name,
        "track": request.track,
        "status": status,
        "dataset": {
            "id": manifest.dataset_id,
            "variant": manifest.variant_id,
            "manifest_digest": manifest.manifest_digest,
            "representation_digest": manifest.representation_digest,
            "feature_count": manifest.feature_count,
            "density": {
                name: manifest.split_map[name].density
                for name in (request.train_split, *request.score_splits)
            },
        },
        "model": dict(request.model),
        "environment": environment,
        "metrics": metrics,
        "timing": timing,
        "diagnostics": diagnostics,
        "artifacts": artifacts,
        "return_code": return_code,
        "failure": failure,
    }
    _append_jsonl(Path(raw_jsonl), record)
    return record


def _require_model_int(config: Mapping[str, object], name: str, minimum: int) -> int:
    value = config.get(name)
    if type(value) is not int or value < minimum:
        raise BenchmarkCampaignError(f"model {name} must be an integer >= {minimum}")
    return value


def _ptm_scalar_wrapper(request_path: Path) -> dict[str, object]:
    request = CampaignRunRequest.load(request_path)
    manifest_path = Path(request.dataset_manifest)
    manifest = CampaignDatasetManifest.load(manifest_path)
    if manifest.manifest_digest != request.dataset_manifest_digest:
        raise BenchmarkCampaignError("wrapper received a stale dataset manifest")
    model = dict(request.model)
    if model.get("implementation") != "ptm.scalar-reference" or model.get("backend") != "python-scalar-reference":
        return {
            "schema": CAMPAIGN_WRAPPER_RESULT_SCHEMA,
            "run_id": request.run_id,
            "status": "unsupported",
            "predictions": {},
            "timing": {},
            "diagnostics": {},
            "environment": {},
            "artifacts": {},
            "failure": "the smoke wrapper supports only the PTM Python scalar reference",
        }
    config = model.get("config")
    if not isinstance(config, Mapping):
        raise BenchmarkCampaignError("PTM model config must be a mapping")
    clauses = _require_model_int(config, "clauses", 2)
    states_per_action = _require_model_int(config, "states_per_action", 1)
    threshold = _require_model_int(config, "threshold", 1)
    epochs = _require_model_int(config, "epochs", 1)
    seed = _require_model_int(config, "seed", 0)
    inference_repeats = _require_model_int(config, "inference_repeats", 1)
    inference_warmup_repeats = _require_model_int(
        config, "inference_warmup_repeats", 0
    )
    specificity = config.get("specificity")
    if (
        type(specificity) not in (int, float)
        or not math.isfinite(float(specificity))
        or float(specificity) <= 1.0
    ):
        raise BenchmarkCampaignError("model specificity must be finite and greater than one")
    preprocessing_started = time.perf_counter()
    train_rows, train_labels = load_dense_bit_split(
        manifest_path, manifest, request.train_split
    )
    score_rows = {
        split_name: load_dense_bit_split(manifest_path, manifest, split_name)[0]
        for split_name in request.score_splits
    }
    machine = ScalarBinaryTsetlinMachine(
        clauses,
        manifest.feature_count,
        states_per_action=states_per_action,
        specificity=float(specificity),
        threshold=threshold,
        seed=seed,
    )
    preprocessing_elapsed = time.perf_counter() - preprocessing_started
    training_started = time.perf_counter()
    machine.fit(train_rows, train_labels, epochs=epochs)
    training_elapsed = time.perf_counter() - training_started
    output_root = Path(request.output_directory).resolve()
    predictions: dict[str, str] = {}
    inference_samples: dict[str, list[float]] = {}
    for split_name in request.score_splits:
        rows = score_rows[split_name]
        for _ in range(inference_warmup_repeats):
            machine.predict(rows)
        samples: list[float] = []
        selected: list[int] | None = None
        for _ in range(inference_repeats):
            started = time.perf_counter()
            current = machine.predict(rows)
            samples.append(time.perf_counter() - started)
            if selected is None:
                selected = current
            elif current != selected:
                raise BenchmarkCampaignError("scalar reference predictions changed without adaptation")
        assert selected is not None
        filename = f"predictions-{split_name}.txt"
        publish_bytes(
            output_root / filename,
            b"".join(f"{value}\n".encode("ascii") for value in selected),
            overwrite=True,
        )
        predictions[split_name] = filename
        inference_samples[split_name] = samples

    snapshot = machine.snapshot()
    included = [
        sum(state > states_per_action for state in clause)
        for clause in snapshot.states
    ]
    state_count = clauses * manifest.feature_count * 2
    low = sum(state == 1 for clause in snapshot.states for state in clause)
    high = sum(
        state == 2 * states_per_action for clause in snapshot.states for state in clause
    )
    return {
        "schema": CAMPAIGN_WRAPPER_RESULT_SCHEMA,
        "run_id": request.run_id,
        "status": "ok",
        "predictions": predictions,
        "timing": {
            "preprocessing_materialization_s": preprocessing_elapsed,
            "adaptive_training_s": training_elapsed,
            "resident_inference_samples_s": inference_samples,
        },
        "diagnostics": {
            "mean_included_literals_per_clause": sum(included) / len(included),
            "empty_clauses": sum(value == 0 for value in included),
            "ta_saturated_low_fraction": low / state_count,
            "ta_saturated_high_fraction": high / state_count,
        },
        "environment": {
            "wrapper_python": platform.python_version(),
            "source_commit": model["commit"],
            "backend_requested": model["backend"],
            "backend_actual": "python-scalar-reference",
        },
        "artifacts": {},
        "failure": None,
    }


def _ptm_native_wrapper(
    request_path: Path,
    executable_path: Path,
) -> dict[str, object]:
    request = CampaignRunRequest.load(request_path)
    manifest_path = Path(request.dataset_manifest)
    manifest = CampaignDatasetManifest.load(manifest_path)
    if manifest.manifest_digest != request.dataset_manifest_digest:
        raise BenchmarkCampaignError("native wrapper received a stale dataset manifest")
    model = dict(request.model)
    if (
        model.get("implementation") != "ptm.native-binary"
        or model.get("backend") != "cpp-scalar-train+packed-cpu-inference"
    ):
        return {
            "schema": CAMPAIGN_WRAPPER_RESULT_SCHEMA,
            "run_id": request.run_id,
            "status": "unsupported",
            "predictions": {},
            "timing": {},
            "diagnostics": {},
            "environment": {},
            "artifacts": {},
            "failure": "native wrapper supports only the PTM C++ binary CPU path",
        }
    if request.track != "shared":
        raise BenchmarkCampaignError("native binary wrapper requires the shared track")
    config = model.get("config")
    if not isinstance(config, Mapping):
        raise BenchmarkCampaignError("PTM native model config must be a mapping")
    clauses = _require_model_int(config, "clauses", 2)
    states_per_action = _require_model_int(config, "states_per_action", 1)
    threshold = _require_model_int(config, "threshold", 1)
    epochs = _require_model_int(config, "epochs", 1)
    seed = _require_model_int(config, "seed", 0)
    inference_repeats = _require_model_int(config, "inference_repeats", 1)
    inference_warmup_repeats = _require_model_int(
        config, "inference_warmup_repeats", 0
    )
    specificity = config.get("specificity")
    if (
        type(specificity) not in (int, float)
        or not math.isfinite(float(specificity))
        or float(specificity) <= 1.0
    ):
        raise BenchmarkCampaignError(
            "native model specificity must be finite and greater than one"
        )
    executable = executable_path.resolve()
    if not executable.is_file():
        raise BenchmarkCampaignError("PTM native campaign executable is absent")
    expected_executable_digest = _require_digest(
        model.get("executable_digest"), "PTM native executable digest"
    )
    executable_digest = _bytes_digest(executable.read_bytes())
    if executable_digest != expected_executable_digest:
        raise BenchmarkCampaignError(
            "PTM native campaign executable disagrees with the request"
        )
    output_root = Path(request.output_directory).resolve()
    split_map = manifest.split_map
    train_receipt = split_map[request.train_split]
    train_path = _resolve_contained(
        manifest_path.resolve().parent,
        _safe_relative_path(train_receipt.path, "training path"),
        "training path",
    )
    score_arguments: list[str] = []
    predictions: dict[str, str] = {}
    vote_score_paths: dict[str, Path] = {}
    for split_name in request.score_splits:
        receipt = split_map[split_name]
        input_path = _resolve_contained(
            manifest_path.resolve().parent,
            _safe_relative_path(receipt.path, "scoring path"),
            "scoring path",
        )
        filename = f"predictions-{split_name}.txt"
        predictions[split_name] = filename
        prediction_path = output_root / filename
        vote_score_paths[split_name] = Path(str(prediction_path) + ".scores")
        score_arguments.extend((str(input_path), str(prediction_path)))
    command = [
        str(executable),
        str(train_path),
        str(clauses),
        str(states_per_action),
        repr(float(specificity)),
        str(threshold),
        str(epochs),
        str(seed),
        str(inference_repeats),
        str(inference_warmup_repeats),
        str(len(request.score_splits)),
        *score_arguments,
    ]
    try:
        completed = run_bounded_process(
            command,
            timeout_seconds=600,
            max_output_bytes=_MAX_RESULT_BYTES,
            isolate_process_tree=False,
        )
    except BoundedProcessError as error:
        raise BenchmarkCampaignError(
            f"native campaign process failed: {error}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise BenchmarkCampaignError(
            "native campaign process exited unsuccessfully"
            + (f": {detail}" if detail else "")
        )
    if _bytes_digest(executable.read_bytes()) != expected_executable_digest:
        raise BenchmarkCampaignError(
            "PTM native campaign executable changed during execution"
        )
    try:
        native = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkCampaignError("native campaign result is malformed") from error
    if not isinstance(native, Mapping) or set(native) != {
        "schema",
        "preprocessing_materialization_s",
        "adaptive_training_s",
        "diagnostic_collection_s",
        "resident_inference_samples_s",
        "backend",
        "diagnostics",
    }:
        raise BenchmarkCampaignError("native campaign result fields are not canonical")
    if native["schema"] != "ptm.native-campaign-runner.v2":
        raise BenchmarkCampaignError("native campaign result schema is unsupported")
    for name in (
        "preprocessing_materialization_s",
        "adaptive_training_s",
        "diagnostic_collection_s",
    ):
        value = native[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise BenchmarkCampaignError(f"native campaign {name} is invalid")
    raw_samples = native["resident_inference_samples_s"]
    if not isinstance(raw_samples, list) or len(raw_samples) != len(
        request.score_splits
    ):
        raise BenchmarkCampaignError("native inference samples are malformed")
    inference_samples: dict[str, list[float]] = {}
    for split_name, samples in zip(request.score_splits, raw_samples):
        if (
            not isinstance(samples, list)
            or len(samples) != inference_repeats
            or any(
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                for value in samples
            )
        ):
            raise BenchmarkCampaignError("native inference sample is invalid")
        inference_samples[split_name] = [float(value) for value in samples]
    backend = _require_identifier(native["backend"], "native backend")
    diagnostics = _canonical_mapping(
        native["diagnostics"], "native campaign diagnostics"
    )
    vote_score_artifacts: dict[str, object] = {}
    vote_score_summaries: dict[str, object] = {}
    for split_name in request.score_splits:
        prediction_path = output_root / predictions[split_name]
        vote_score_path = vote_score_paths[split_name]
        if not vote_score_path.is_file():
            raise BenchmarkCampaignError("native vote-score file is absent")
        retained_predictions = _read_predictions(prediction_path)
        vote_scores = _read_vote_scores(vote_score_path)
        expected_rows = split_map[split_name].row_count
        if len(vote_scores) != expected_rows or len(retained_predictions) != expected_rows:
            raise BenchmarkCampaignError("native vote-score row count is inconsistent")
        if any(value < -threshold or value > threshold for value in vote_scores):
            raise BenchmarkCampaignError("native vote score exceeds the clipped range")
        if any(
            prediction != int(score > 0)
            for prediction, score in zip(retained_predictions, vote_scores)
        ):
            raise BenchmarkCampaignError("native vote score contradicts its prediction")
        vote_bytes = vote_score_path.read_bytes()
        vote_score_artifacts[split_name] = {
            "path": vote_score_path.name,
            "digest": _bytes_digest(vote_bytes),
        }
        vote_score_summaries[split_name] = {
            "minimum": min(vote_scores),
            "maximum": max(vote_scores),
            "mean": sum(vote_scores) / len(vote_scores),
            "mean_absolute": sum(abs(value) for value in vote_scores)
            / len(vote_scores),
            "ties": sum(value == 0 for value in vote_scores),
        }
    diagnostics = diagnostics | {"vote_score_summaries": vote_score_summaries}
    return {
        "schema": CAMPAIGN_WRAPPER_RESULT_SCHEMA,
        "run_id": request.run_id,
        "status": "ok",
        "predictions": predictions,
        "timing": {
            "preprocessing_materialization_s": float(
                native["preprocessing_materialization_s"]
            ),
            "adaptive_training_s": float(native["adaptive_training_s"]),
            "diagnostic_collection_s": float(native["diagnostic_collection_s"]),
            "resident_inference_samples_s": inference_samples,
        },
        "diagnostics": diagnostics,
        "environment": {
            "wrapper_python": platform.python_version(),
            "backend_actual": f"cpp-scalar-train+packed-{backend}",
            "backend_requested": model["backend"],
            "source_commit": model.get("commit", "unknown"),
        },
        "artifacts": {
            "native_executable": executable.name,
            "native_executable_digest": executable_digest,
            "vote_score_files": vote_score_artifacts,
        },
        "failure": None,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PTM internal benchmark campaign tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-synthetic")
    prepare.add_argument("output_directory", type=Path)
    local = subparsers.add_parser("prepare-local-baselines")
    local.add_argument("project_root", type=Path)
    local.add_argument("logic_material_directory", type=Path)
    local.add_argument("output_directory", type=Path)
    wrapper = subparsers.add_parser("wrapper-ptm-scalar")
    wrapper.add_argument("request", type=Path)
    native_wrapper = subparsers.add_parser("wrapper-ptm-native")
    native_wrapper.add_argument("executable", type=Path)
    native_wrapper.add_argument("request", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "prepare-synthetic":
        xor = prepare_xor_noise(arguments.output_directory / "xor20-noise")
        parity = prepare_parity_ladder(arguments.output_directory / "parity-ladder")
        print(
            json.dumps(
                {"xor_manifests": [str(path) for path in xor], "parity_manifests": [str(path) for path in parity]},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "prepare-local-baselines":
        manifests = prepare_local_baselines(
            arguments.project_root,
            arguments.logic_material_directory,
            arguments.output_directory,
        )
        print(
            json.dumps(
                {"manifests": [str(path) for path in manifests]},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "wrapper-ptm-scalar":
        try:
            result = _ptm_scalar_wrapper(arguments.request)
        except (BenchmarkCampaignError, OSError) as error:
            print(str(error), file=sys.stderr)
            return 2
        sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
        return 0
    if arguments.command == "wrapper-ptm-native":
        try:
            result = _ptm_native_wrapper(
                arguments.request,
                arguments.executable,
            )
        except (BenchmarkCampaignError, OSError) as error:
            print(str(error), file=sys.stderr)
            return 2
        sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
        return 0
    raise AssertionError(arguments.command)


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "BenchmarkCampaignError",
    "CAMPAIGN_DATASET_SCHEMA",
    "CAMPAIGN_REQUEST_SCHEMA",
    "CAMPAIGN_RUN_SCHEMA",
    "CAMPAIGN_WRAPPER_RESULT_SCHEMA",
    "CampaignDatasetManifest",
    "CampaignRunRequest",
    "DENSE_BIT_TEXT_FORMAT",
    "DenseBitSplitReceipt",
    "import_dense_bit_dataset",
    "load_dense_bit_split",
    "prepare_local_baselines",
    "prepare_parity_ladder",
    "prepare_xor_noise",
    "run_campaign_attempt",
]
