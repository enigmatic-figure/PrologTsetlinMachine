"""Application-facing operations for immutable portable model artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from ..model_artifact import (
    InferenceArtifact,
    PackedTMInferenceArtifact,
    load_model_artifact,
)
from ..representation import FieldKind, NullPolicy


MAX_INTERACTIVE_RECORDS = 10_000


def _transform_expression(output: Any) -> str:
    parameters = output.descriptor.canonical_payload()["parameters"]
    arguments = ", ".join(
        f"{key}={json.dumps(value, ensure_ascii=False, sort_keys=True)}"
        for key, value in parameters.items()
    )
    return f"{output.descriptor.transform.value}({arguments})"


@dataclass(frozen=True, slots=True)
class ArtifactInputField:
    """One unique raw field required by an embedded preprocessing contract."""

    name: str
    kind: FieldKind
    required: bool
    transforms: tuple[str, ...]
    accepted_values: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactInferenceSession:
    """One immutable artifact instance pinned for inspection and inference."""

    source: Path
    artifact: InferenceArtifact
    inspection: Mapping[str, Any]
    verification: Mapping[str, Any]
    fields: tuple[ArtifactInputField, ...]


def _artifact_input_fields(
    artifact: InferenceArtifact,
) -> tuple[ArtifactInputField, ...]:
    """Describe raw input fields from one already-loaded artifact."""

    if not isinstance(artifact, PackedTMInferenceArtifact):
        return ()
    preprocessing = artifact.preprocessing
    if preprocessing is None:
        return ()

    order: list[str] = []
    grouped: dict[str, dict[str, Any]] = {}
    for output in preprocessing.outputs:
        name = output.field.name
        if name not in grouped:
            order.append(name)
            grouped[name] = {
                "kind": output.field.kind,
                "required": False,
                "transforms": [],
                "accepted_values": [],
            }
        item = grouped[name]
        item["required"] = item["required"] or (
            output.descriptor.null_policy is NullPolicy.ERROR
        )
        transform = _transform_expression(output)
        if transform not in item["transforms"]:
            item["transforms"].append(transform)
        parameters = output.descriptor.parameters
        for key in ("value", "values"):
            candidate = dict(parameters).get(key)
            candidates = candidate if key == "values" else (candidate,)
            if candidate is None:
                continue
            for value in candidates:
                if not any(
                    type(value) is type(existing) and value == existing
                    for existing in item["accepted_values"]
                ):
                    item["accepted_values"].append(value)

    return tuple(
        ArtifactInputField(
            name,
            grouped[name]["kind"],
            bool(grouped[name]["required"]),
            tuple(grouped[name]["transforms"]),
            tuple(grouped[name]["accepted_values"]),
        )
        for name in order
    )


def artifact_input_fields(path: str | Path) -> tuple[ArtifactInputField, ...]:
    """Describe raw input fields in stable first-use order."""

    _, artifact = _load(path)
    return _artifact_input_fields(artifact)


def parse_typed_record(
    fields: Sequence[ArtifactInputField], values: Mapping[str, str]
) -> dict[str, object]:
    """Convert TUI-style strings into the strict portable record types."""

    record: dict[str, object] = {}
    for field in fields:
        raw = values.get(field.name, "").strip()
        if not raw:
            if field.required:
                raise ValueError(f"{field.name} is required")
            continue
        if raw == "null":
            record[field.name] = None
            continue
        if field.kind is FieldKind.BOOLEAN:
            normalized = raw.casefold()
            if normalized in ("true", "1"):
                record[field.name] = True
            elif normalized in ("false", "0"):
                record[field.name] = False
            else:
                raise ValueError(f"{field.name} must be true or false")
        elif field.kind is FieldKind.NUMBER:
            try:
                number = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"{field.name} must be a JSON number") from error
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError(f"{field.name} must be a JSON number")
            record[field.name] = number
        elif field.kind is FieldKind.CATEGORY:
            try:
                category = json.loads(raw)
            except json.JSONDecodeError:
                category = raw
            if not isinstance(category, (str, int, bool)):
                raise ValueError(
                    f"{field.name} must be a string, integer, or Boolean category"
                )
            record[field.name] = category
        else:
            raise ValueError(f"{field.name} has unsupported kind {field.kind.value}")
    return record


def _load(path: str | Path) -> tuple[Path, InferenceArtifact]:
    source = Path(path)
    return source, load_model_artifact(source)


def inspect_artifact(
    path: str | Path, *, include_manifest: bool = False
) -> dict[str, Any]:
    """Load, validate, and return a stable human/tool-facing description."""

    source, artifact = _load(path)
    return _inspect_artifact(source, artifact, include_manifest=include_manifest)


def _inspect_artifact(
    source: Path,
    artifact: InferenceArtifact,
    *,
    include_manifest: bool = False,
) -> dict[str, Any]:
    """Inspect one already-loaded immutable artifact."""

    manifest = artifact.manifest
    ports = manifest.get("ports", {})
    validation = manifest.get("validation", {})
    preprocessing = manifest.get("preprocessing")
    result: dict[str, Any] = {
        "artifact_id": artifact.artifact_id,
        "artifact_kind": manifest.get("artifact_kind"),
        "artifact_schema": manifest.get("artifact_schema"),
        "file": str(source),
        "size_bytes": len(artifact.serialized),
        "title": manifest.get("title"),
        "task": manifest.get("task"),
        "producer": manifest.get("producer"),
        "model": manifest.get("model"),
        "inputs": ports.get("inputs") if isinstance(ports, Mapping) else None,
        "outputs": ports.get("outputs") if isinstance(ports, Mapping) else None,
        "has_preprocessing": preprocessing is not None,
        "preprocessing_schema": (
            preprocessing.get("schema")
            if isinstance(preprocessing, Mapping)
            else None
        ),
        "conformance_case_count": (
            validation.get("conformance_case_count")
            if isinstance(validation, Mapping)
            else None
        ),
    }
    if include_manifest:
        result["manifest"] = manifest
    return result


def verify_artifact(path: str | Path) -> dict[str, Any]:
    """Validate container integrity, contracts, and embedded conformance cases."""

    source, artifact = _load(path)
    return _verify_artifact(source, artifact)


def _verify_artifact(
    source: Path, artifact: InferenceArtifact
) -> dict[str, Any]:
    """Verify one already-loaded immutable artifact."""

    verified = artifact.verify_conformance()
    validation = artifact.manifest.get("validation")
    case_count = (
        validation.get("conformance_case_count", 0)
        if isinstance(validation, Mapping)
        else 0
    )
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_kind": artifact.manifest.get("artifact_kind"),
        "conformance_case_count": int(case_count),
        "file": str(source),
        "size_bytes": len(artifact.serialized),
        "verified": verified,
    }


def open_artifact_session(path: str | Path) -> ArtifactInferenceSession:
    """Load once and bind all later UI operations to the verified bytes."""

    source = Path(path).expanduser().resolve()
    artifact = load_model_artifact(source)
    inspection = _inspect_artifact(source, artifact)
    verification = _verify_artifact(source, artifact)
    fields = _artifact_input_fields(artifact)
    return ArtifactInferenceSession(
        source=source,
        artifact=artifact,
        inspection=inspection,
        verification=verification,
        fields=fields,
    )


def run_artifact_records(
    path: str | Path, records: Sequence[Mapping[str, object]]
) -> dict[str, Any]:
    """Materialize typed records and run a raw-record-capable packed TM."""

    _, artifact = _load(path)
    return _run_artifact_records(artifact, records)


def run_session_artifact_records(
    session: ArtifactInferenceSession,
    records: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Run records against the exact immutable artifact opened by the session."""

    return _run_artifact_records(session.artifact, records)


def _run_artifact_records(
    artifact: InferenceArtifact,
    records: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    if not isinstance(artifact, PackedTMInferenceArtifact):
        raise ValueError("raw-record inference currently requires a packed-TM artifact")
    preprocessing = artifact.preprocessing
    if preprocessing is None:
        raise ValueError("artifact requires precomputed Boolean features")
    if not records:
        raise ValueError("at least one raw record is required")
    if len(records) > MAX_INTERACTIVE_RECORDS:
        raise ValueError(
            f"raw-record command is limited to {MAX_INTERACTIVE_RECORDS} records"
        )

    feature_rows = tuple(preprocessing.materialize(record) for record in records)
    predictions = artifact.predict_rows(feature_rows)
    task = artifact.manifest.get("task")
    labels = task.get("labels") if isinstance(task, Mapping) else None
    results = []
    for index, (features, prediction) in enumerate(zip(feature_rows, predictions)):
        feature_trace = [
            {
                "expression": _transform_expression(output),
                "field": output.field.name,
                "literal_id": str(output.descriptor.literal_id),
                "parameters": output.descriptor.canonical_payload()["parameters"],
                "transform": output.descriptor.transform.value,
                "value": int(value),
            }
            for output, value in zip(preprocessing.outputs, features)
        ]
        result: dict[str, Any] = {
            "feature_trace": feature_trace,
            "index": index,
            "features": [int(value) for value in features],
            "prediction": prediction,
        }
        if isinstance(labels, list) and 0 <= prediction < len(labels):
            result["label"] = labels[prediction]
        results.append(result)
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_kind": artifact.manifest.get("artifact_kind"),
        "preprocessing_schema": preprocessing.schema,
        "record_count": len(results),
        "results": results,
    }
