"""Native MNIST preparation and multiclass training for interactive frontends."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
from threading import Event, Thread
from typing import Callable, Sequence

from .training import (
    MulticlassTrainingRun,
    TrainingCancelled,
    TrainingProgress,
    TrainingRequest,
    TrainingWorkload,
)


MNIST_MAGIC = b"PTMMNIST"
MNIST_MATERIAL_VERSION = 1
MNIST_MATERIAL_SCHEMA = "ptm.mnist-bits.v1"
MNIST_RUNNER_SCHEMA = "ptm.mnist-ovr-epoch.v1"
MNIST_PIXEL_THRESHOLD = 0.3


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_split(path: Path, features: object, labels: object) -> dict[str, object]:
    import numpy as np

    bits = np.ascontiguousarray(features, dtype=np.uint8)
    targets = np.ascontiguousarray(labels, dtype=np.uint8)
    with path.open("wb") as stream:
        stream.write(
            struct.pack(
                "<8sIII",
                MNIST_MAGIC,
                MNIST_MATERIAL_VERSION,
                bits.shape[0],
                bits.shape[1],
            )
        )
        stream.write(bits.tobytes(order="C"))
        stream.write(targets.tobytes(order="C"))
    return {
        "path": path.name,
        "rows": int(bits.shape[0]),
        "features": int(bits.shape[1]),
        "positive_cells": int(bits.sum()),
        "digest": _digest_file(path),
        "class_counts": {
            str(value): int((targets == value).sum()) for value in range(10)
        },
    }


def prepare_mnist_material(
    source: str | Path,
    output: str | Path,
    *,
    threshold: float = MNIST_PIXEL_THRESHOLD,
) -> Path:
    """Convert the classic Python-2 MNIST pickle to deterministic bit material."""

    import numpy as np

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not 0.0 < threshold < 1.0:
        raise ValueError("MNIST threshold must lie between zero and one")
    source_bytes = source_path.read_bytes()
    payload = pickle.loads(source_bytes, encoding="latin1")
    if not isinstance(payload, tuple) or len(payload) != 3:
        raise ValueError("MNIST pickle must contain train, validation, and test splits")
    output_path.mkdir(parents=True, exist_ok=True)
    splits: dict[str, object] = {}
    for name, split in zip(("train", "validation", "test"), payload):
        if not isinstance(split, tuple) or len(split) != 2:
            raise ValueError(f"MNIST {name} split is malformed")
        features = np.asarray(split[0])
        labels = np.asarray(split[1])
        if (
            features.ndim != 2
            or features.shape[1] != 784
            or labels.shape != (features.shape[0],)
        ):
            raise ValueError(f"MNIST {name} split has the wrong shape")
        if labels.size == 0 or labels.min() != 0 or labels.max() != 9:
            raise ValueError(f"MNIST {name} labels are outside 0..9")
        splits[name] = _write_split(
            output_path / f"{name}.ptmb", features > threshold, labels
        )
    manifest = {
        "schema": MNIST_MATERIAL_SCHEMA,
        "source_digest": _digest_bytes(source_bytes),
        "threshold": threshold,
        "threshold_rule": "pixel > threshold",
        "splits": splits,
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _validated_material(
    source: Path, output: Path, threshold: float
) -> Path | None:
    manifest_path = output / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != MNIST_MATERIAL_SCHEMA
        or manifest.get("threshold") != threshold
        or manifest.get("threshold_rule") != "pixel > threshold"
        or manifest.get("source_digest") != _digest_file(source)
    ):
        return None
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        return None
    for name, rows in (("train", 50_000), ("validation", 10_000), ("test", 10_000)):
        receipt = splits.get(name)
        if not isinstance(receipt, dict) or receipt.get("rows") != rows:
            return None
        path = output / str(receipt.get("path", ""))
        if not path.is_file() or receipt.get("digest") != _digest_file(path):
            return None
    return manifest_path


def ensure_mnist_material(
    workspace: str | Path,
    *,
    threshold: float = MNIST_PIXEL_THRESHOLD,
    output: str | Path | None = None,
) -> Path:
    root = Path(workspace).expanduser().resolve()
    source = root / "data" / "mnist.pkl"
    if not source.is_file():
        raise FileNotFoundError(
            f"MNIST workload requires the classic pickle at {source}"
        )
    destination = (
        Path(output).expanduser().resolve()
        if output is not None
        else root / "out" / "workbench" / "mnist-bits-030"
    )
    validated = _validated_material(source, destination, threshold)
    if validated is not None:
        return validated
    return prepare_mnist_material(source, destination, threshold=threshold)


def _windows_to_wsl(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    if not drive:
        raise RuntimeError("cannot translate a non-drive Windows path to WSL")
    suffix = path.as_posix()[2:].lstrip("/")
    return f"/mnt/{drive}/{suffix}"


def _runner_invocation(
    workspace: Path,
    runner_command: Sequence[str] | None,
) -> tuple[tuple[str, ...], Callable[[Path], str]]:
    if runner_command is not None:
        if not runner_command or any(not item for item in runner_command):
            raise ValueError("MNIST runner command cannot be empty")
        return tuple(runner_command), lambda path: str(path)
    configured = os.environ.get("PTM_MNIST_RUNNER")
    candidates = [
        Path(configured).expanduser() if configured else None,
        workspace / "out" / "build" / "Release" / "ptm_mnist_ovr_benchmark.exe",
        workspace / "out" / "build" / "ptm_mnist_ovr_benchmark",
        workspace
        / "out"
        / "benchmark-campaign"
        / "mnist-native-wsl"
        / "ptm_mnist_ovr_benchmark",
    ]
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if os.name == "nt" and resolved.suffix.lower() != ".exe":
            return ("wsl.exe", _windows_to_wsl(resolved)), _windows_to_wsl
        return (str(resolved),), lambda path: str(path)
    raise RuntimeError(
        "native MNIST runner is unavailable; build the "
        "ptm_mnist_ovr_benchmark target or set PTM_MNIST_RUNNER"
    )


def _confusion(value: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or len(value) != 10:
        raise RuntimeError("MNIST runner confusion matrix has the wrong shape")
    rows: list[tuple[int, ...]] = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 10
            or any(type(item) is not int or item < 0 for item in row)
        ):
            raise RuntimeError("MNIST runner confusion matrix is invalid")
        rows.append(tuple(row))
    return tuple(rows)


def train_mnist_native(
    request: TrainingRequest,
    *,
    workspace: str | Path,
    progress: Callable[[TrainingProgress], None] | None = None,
    cancel: Event | None = None,
    runner_command: Sequence[str] | None = None,
    material_directory: str | Path | None = None,
) -> MulticlassTrainingRun:
    """Run the native ten-bank MNIST learner and validate every epoch record."""

    request.validate()
    if request.workload is not TrainingWorkload.MNIST:
        raise ValueError("native MNIST training requires the MNIST workload")
    if cancel is not None and cancel.is_set():
        raise TrainingCancelled("MNIST training cancelled before launch")
    root = Path(workspace).expanduser().resolve()
    manifest_path = ensure_mnist_material(
        root, output=material_directory, threshold=MNIST_PIXEL_THRESHOLD
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_map = manifest["splits"]
    train_path = manifest_path.parent / split_map["train"]["path"]
    validation_path = manifest_path.parent / split_map["validation"]["path"]
    prefix, convert_path = _runner_invocation(root, runner_command)
    command = (
        *prefix,
        convert_path(train_path),
        convert_path(validation_path),
        str(request.number_of_clauses),
        str(request.states_per_action),
        str(request.specificity),
        str(request.threshold),
        str(request.epochs),
        str(request.seed),
        "paired",
        "boost" if request.boost_true_positive_feedback else "standard",
    )
    process = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    done = Event()

    def watch_cancel() -> None:
        if cancel is None:
            return
        while not done.wait(0.1):
            if cancel.is_set():
                process.terminate()
                return

    watcher = Thread(target=watch_cancel, daemon=True)
    watcher.start()
    records: list[dict[str, object]] = []
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError("native MNIST runner emitted invalid JSON") from error
            expected_epoch = len(records) + 1
            if (
                not isinstance(record, dict)
                or record.get("schema") != MNIST_RUNNER_SCHEMA
                or record.get("epoch") != expected_epoch
                or record.get("clauses_per_class") != request.number_of_clauses
                or record.get("threshold") != request.threshold
                or record.get("specificity") != request.specificity
                or record.get("training_policy") != "paired"
                or record.get("epoch_shuffle") is not True
                or record.get("parallel_class_training") is not True
                or record.get("parallel_validation") is not True
                or record.get("boost_true_positive_feedback")
                is not request.boost_true_positive_feedback
            ):
                raise RuntimeError("native MNIST runner record violates the request")
            accuracy = record.get("validation_accuracy")
            elapsed = record.get("training_seconds")
            cumulative = record.get("cumulative_training_seconds")
            if (
                type(accuracy) not in (int, float)
                or not 0.0 <= float(accuracy) <= 1.0
                or type(elapsed) not in (int, float)
                or float(elapsed) < 0.0
                or type(cumulative) not in (int, float)
                or float(cumulative) < 0.0
            ):
                raise RuntimeError("native MNIST runner metrics are invalid")
            matrix = _confusion(record.get("confusion_matrix"))
            validation_rows = int(split_map["validation"]["rows"])
            if sum(sum(row) for row in matrix) != validation_rows:
                raise RuntimeError("native MNIST runner did not score validation exactly")
            correct = sum(matrix[index][index] for index in range(10))
            if abs(correct / validation_rows - float(accuracy)) > 1e-12:
                raise RuntimeError(
                    "native MNIST runner accuracy disagrees with confusion counts"
                )
            records.append(record)
            if progress is not None:
                progress(
                    TrainingProgress(
                        expected_epoch,
                        request.epochs,
                        float(accuracy),
                        float(elapsed),
                        float(cumulative),
                    )
                )
        return_code = process.wait()
        stderr = process.stderr.read() if process.stderr is not None else ""
    finally:
        done.set()
        watcher.join(timeout=1.0)
        if process.poll() is None:
            process.kill()
            process.wait()
    if cancel is not None and cancel.is_set():
        raise TrainingCancelled("MNIST training cancelled")
    if return_code != 0:
        detail = stderr.strip() or f"exit status {return_code}"
        raise RuntimeError(f"native MNIST runner failed: {detail}")
    if len(records) != request.epochs:
        raise RuntimeError("native MNIST runner omitted one or more epochs")
    final = records[-1]
    run = MulticlassTrainingRun(
        request=request,
        class_labels=tuple(range(10)),
        validation_rows=int(split_map["validation"]["rows"]),
        confusion_matrix=_confusion(final["confusion_matrix"]),
        accuracy=float(final["validation_accuracy"]),
        training_seconds=float(final["cumulative_training_seconds"]),
        backend="cpp-scalar-ten-bank-native",
        material_manifest=str(manifest_path),
    )
    run.validate()
    return run


__all__ = [
    "MNIST_PIXEL_THRESHOLD",
    "ensure_mnist_material",
    "prepare_mnist_material",
    "train_mnist_native",
]
