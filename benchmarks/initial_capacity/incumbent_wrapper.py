#!/usr/bin/env python3
"""Standalone shared-input wrapper for the two pinned incumbent TM projects."""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Mapping


REQUEST_SCHEMA = "ptm.campaign-run-request.v1"
DATASET_SCHEMA = "ptm.campaign-dataset.v1"
RESULT_SCHEMA = "ptm.campaign-wrapper-result.v1"
PINS = {
    "pytsetlinmachine": "d6c1cf0e4aaa4a8ae2f2818ba27878fb89d31dc5",
    "tmu": "5605ff070a18549328028c907a9acf68e063346e",
}


class WrapperError(ValueError):
    pass


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WrapperError(f"JSON document is not an object: {path}")
    return value


def _contained(root: Path, raw_relative: object) -> Path:
    if type(raw_relative) is not str or not raw_relative:
        raise WrapperError("dataset split path is invalid")
    relative = Path(raw_relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise WrapperError("dataset split path is not contained")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise WrapperError("dataset split path escapes its manifest") from error
    return resolved


def _load_split(manifest_path: Path, manifest: Mapping[str, object], name: str):
    import numpy as np

    representation = manifest.get("representation")
    splits = manifest.get("splits")
    if not isinstance(representation, Mapping) or not isinstance(splits, Mapping):
        raise WrapperError("dataset manifest representation is malformed")
    width = representation.get("feature_count")
    receipt = splits.get(name)
    if type(width) is not int or width <= 0 or not isinstance(receipt, Mapping):
        raise WrapperError(f"dataset split is malformed: {name}")
    path = _contained(manifest_path.parent, receipt.get("path"))
    data = path.read_bytes()
    expected_digest = receipt.get("file_digest")
    actual_digest = "sha256:" + hashlib.sha256(data).hexdigest()
    if actual_digest != expected_digest:
        raise WrapperError(f"dataset split digest mismatch: {name}")
    rows: list[list[int]] = []
    labels: list[int] = []
    for line_number, line in enumerate(data.splitlines(), 1):
        fields = line.split()
        if len(fields) != width + 1 or any(field not in (b"0", b"1") for field in fields):
            raise WrapperError(f"dataset row is malformed: {name}:{line_number}")
        rows.append([int(field) for field in fields[:-1]])
        labels.append(int(fields[-1]))
    if not rows or len(rows) != receipt.get("row_count"):
        raise WrapperError(f"dataset row count mismatch: {name}")
    return np.asarray(rows, dtype=np.uint32), np.asarray(labels, dtype=np.uint32)


def _integer(config: Mapping[str, object], name: str, minimum: int) -> int:
    value = config.get(name)
    if type(value) is not int or value < minimum:
        raise WrapperError(f"model {name} must be an integer >= {minimum}")
    return value


def _boolean(config: Mapping[str, object], name: str, default: bool) -> bool:
    value = config.get(name, default)
    if type(value) is not bool:
        raise WrapperError(f"model {name} must be Boolean")
    return value


def _specificity(config: Mapping[str, object]) -> float:
    value = config.get("specificity")
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 1:
        raise WrapperError("model specificity must be finite and greater than one")
    return float(value)


def _write_predictions(output: Path, name: str, values) -> str:
    filename = f"predictions-{name}.txt"
    data = b"".join(f"{int(value)}\n".encode("ascii") for value in values)
    (output / filename).write_bytes(data)
    return filename


def _unsupported(run_id: str, message: str) -> dict[str, object]:
    return {
        "schema": RESULT_SCHEMA,
        "run_id": run_id,
        "status": "unsupported",
        "predictions": {},
        "timing": {},
        "diagnostics": {},
        "environment": {},
        "artifacts": {},
        "failure": message,
    }


def _run(backend: str, request_path: Path) -> dict[str, object]:
    request = _load_object(request_path)
    if request.get("schema") != REQUEST_SCHEMA:
        raise WrapperError("campaign request schema is unsupported")
    run_id = request.get("run_id")
    if type(run_id) is not str or not run_id:
        raise WrapperError("campaign run ID is invalid")
    model = request.get("model")
    if not isinstance(model, Mapping):
        raise WrapperError("campaign model request is malformed")
    expected_implementation = {
        "pytsetlinmachine": "pytsetlinmachine.multiclass",
        "tmu": "tmu.vanilla-classifier",
    }[backend]
    if model.get("implementation") != expected_implementation:
        return _unsupported(
            run_id, f"wrapper supports only {expected_implementation}"
        )
    if model.get("commit") != PINS[backend]:
        raise WrapperError("requested incumbent commit does not match the wrapper pin")
    requested_backend = model.get("backend")
    if type(requested_backend) is not str:
        raise WrapperError("requested incumbent backend is invalid")
    config = model.get("config")
    if not isinstance(config, Mapping):
        raise WrapperError("campaign model config is malformed")

    manifest_path = Path(str(request.get("dataset_manifest"))).resolve()
    manifest = _load_object(manifest_path)
    if manifest.get("schema") != DATASET_SCHEMA:
        raise WrapperError("campaign dataset schema is unsupported")
    if manifest.get("manifest_digest") != request.get("dataset_manifest_digest"):
        raise WrapperError("campaign dataset identity is stale")
    train_name = request.get("train_split")
    score_names = request.get("score_splits")
    output = Path(str(request.get("output_directory"))).resolve()
    if type(train_name) is not str or not isinstance(score_names, list) or not score_names:
        raise WrapperError("campaign split request is malformed")
    if any(type(name) is not str for name in score_names):
        raise WrapperError("campaign score split is malformed")
    output.mkdir(parents=True, exist_ok=True)

    preprocessing_started = time.perf_counter()
    import numpy as np

    train_rows, train_labels = _load_split(manifest_path, manifest, train_name)
    if set(int(value) for value in train_labels) != {0, 1}:
        raise WrapperError("incumbent binary classifier requires both training classes")
    score_rows = {
        name: _load_split(manifest_path, manifest, name)[0] for name in score_names
    }
    clauses = _integer(config, "clauses", 2)
    if clauses % 2:
        raise WrapperError("model clauses must be even")
    threshold = _integer(config, "threshold", 1)
    epochs = _integer(config, "epochs", 1)
    repeats = _integer(config, "inference_repeats", 1)
    warmup_repeats = _integer(config, "inference_warmup_repeats", 0)
    seed = _integer(config, "seed", 0)
    specificity = _specificity(config)
    weighted = _boolean(config, "weighted_clauses", False)
    feature_negation = _boolean(config, "feature_negation", True)
    boost = _integer(config, "boost_true_positive_feedback", 0)
    state_bits = _integer(config, "number_of_state_bits", 1)
    max_literals = config.get("max_included_literals")
    if max_literals is not None and (type(max_literals) is not int or max_literals < 1):
        raise WrapperError("model max_included_literals must be null or positive")

    if backend == "pytsetlinmachine":
        if requested_backend != "cpu":
            raise WrapperError("pyTsetlinMachine wrapper supports only the CPU backend")
        from pyTsetlinMachine.tm import MultiClassTsetlinMachine

        indexed = _boolean(config, "indexed", True)
        np.random.seed(seed)
        machine = MultiClassTsetlinMachine(
            clauses,
            threshold,
            specificity,
            boost_true_positive_feedback=boost,
            number_of_state_bits=state_bits,
            indexed=indexed,
            append_negated=feature_negation,
            weighted_clauses=weighted,
            max_included_literals=max_literals,
        )
        package_name = "pyTsetlinMachine"
        backend_actual = "pytsetlinmachine-cpu"
        seed_control = "fixed-library-stream-no-public-seed"
    else:
        platform_name = config.get("platform", "CPU")
        if platform_name not in ("CPU", "CUDA"):
            return _unsupported(run_id, "TMU platform must be CPU or CUDA")
        if requested_backend != platform_name.lower():
            raise WrapperError("requested TMU backend disagrees with model platform")
        from tmu.models.classification.vanilla_classifier import TMClassifier

        shuffle = _boolean(config, "shuffle", True)
        machine = TMClassifier(
            number_of_clauses=clauses,
            T=threshold,
            s=specificity,
            platform=platform_name,
            feature_negation=feature_negation,
            boost_true_positive_feedback=boost,
            max_included_literals=max_literals,
            number_of_state_bits_ta=state_bits,
            weighted_clauses=weighted,
            incremental=True,
            seed=seed,
        )
        package_name = "tmu"
        backend_actual = f"tmu-{platform_name.lower()}"
        seed_control = "public-seed"
    preprocessing_elapsed = time.perf_counter() - preprocessing_started

    training_started = time.perf_counter()
    if backend == "pytsetlinmachine":
        machine.fit(train_rows, train_labels, epochs=epochs)
    else:
        for _ in range(epochs):
            machine.fit(train_rows, train_labels, shuffle=shuffle)
    training_elapsed = time.perf_counter() - training_started

    prediction_paths: dict[str, str] = {}
    inference_samples: dict[str, list[float]] = {}
    for name in score_names:
        selected = None
        samples: list[float] = []
        for _ in range(warmup_repeats):
            machine.predict(score_rows[name])
        for _ in range(repeats):
            started = time.perf_counter()
            current = machine.predict(score_rows[name])
            samples.append(time.perf_counter() - started)
            if selected is None:
                selected = current.copy()
            elif not np.array_equal(current, selected):
                raise WrapperError("predictions changed without an adaptive update")
        prediction_paths[name] = _write_predictions(output, name, selected)
        inference_samples[name] = samples

    try:
        package_version = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        package_version = "unknown"
    return {
        "schema": RESULT_SCHEMA,
        "run_id": run_id,
        "status": "ok",
        "predictions": prediction_paths,
        "timing": {
            "preprocessing_materialization_s": preprocessing_elapsed,
            "adaptive_training_s": training_elapsed,
            "resident_inference_samples_s": inference_samples,
        },
        "diagnostics": {
            "clause_count_scope": "per-class",
            "number_of_classes": 2,
            "seed_control": seed_control,
        },
        "environment": {
            "wrapper_python": platform.python_version(),
            "numpy": np.__version__,
            "package": package_name,
            "package_version": package_version,
            "source_commit": PINS[backend],
            "backend_requested": requested_backend,
            "backend_actual": backend_actual,
        },
        "artifacts": {},
        "failure": None,
    }


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in PINS:
        print("usage: incumbent_wrapper.py {pytsetlinmachine|tmu} REQUEST", file=sys.stderr)
        return 2
    backend = sys.argv[1]
    request_path = Path(sys.argv[2])
    try:
        with redirect_stdout(sys.stderr):
            result = _run(backend, request_path)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        try:
            request = _load_object(request_path)
            run_id = str(request.get("run_id", "unknown"))
        except Exception:
            run_id = "unknown"
        result = {
            "schema": RESULT_SCHEMA,
            "run_id": run_id,
            "status": "failed",
            "predictions": {},
            "timing": {},
            "diagnostics": {},
            "environment": {
                "wrapper_python": platform.python_version(),
                "source_commit": PINS[backend],
            },
            "artifacts": {},
            "failure": f"{type(error).__name__}: {error}",
        }
    sys.stdout.write(json.dumps(result, allow_nan=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
