from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from hypothesis import given, settings, strategies as st
import pytest

from prolog_tsetlin import (
    FeatureSchema,
    FieldKind,
    LiteralCatalog,
    ModelArtifactError,
    NullPolicy,
    PreprocessingContract,
    load_model_artifact_from_bytes,
)
from prolog_tsetlin.preprocessing import (
    MAX_PREPROCESSING_OUTPUTS,
    PREPROCESSING_SCHEMA,
)


DATA = Path(__file__).resolve().parents[1] / "data"
ARTIFACTS = tuple(
    bytes.fromhex((DATA / filename).read_text(encoding="ascii"))
    for filename in (
        "xor_packed_tm_v1.hex",
        "raw_xor_packed_tm_v1.hex",
        "preprocessing_demo_v1.hex",
    )
)


def _replace_manifest(serialized: bytes, manifest_bytes: bytes) -> bytes:
    manifest_size = int.from_bytes(serialized[24:32], "little")
    payload_size = int.from_bytes(serialized[32:40], "little")
    payload_first = 64 + manifest_size
    header = bytearray(serialized[:64])
    header[24:32] = len(manifest_bytes).to_bytes(8, "little")
    content = bytes(header) + manifest_bytes + serialized[
        payload_first : payload_first + payload_size
    ]
    return content + hashlib.sha256(content).digest()


def _test_preprocessing() -> PreprocessingContract:
    catalog = LiteralCatalog(
        FeatureSchema.from_fields(
            age=FieldKind.NUMBER,
            state=FieldKind.CATEGORY,
            active=FieldKind.BOOLEAN,
        )
    )
    catalog.numeric_ge("age", 18, null_policy=NullPolicy.ERROR)
    catalog.category_in("state", ["ready", "running", 7])
    catalog.category_eq("active", True)
    catalog.is_missing("active")
    return PreprocessingContract.from_catalog(catalog)


JSON_SCALAR = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=32),
)
JSON_VALUE = st.recursive(
    JSON_SCALAR,
    lambda children: st.one_of(
        st.lists(children, max_size=8),
        st.dictionaries(st.text(max_size=16), children, max_size=8),
    ),
    max_leaves=32,
)


@settings(max_examples=300, deadline=None, derandomize=True)
@given(st.binary(max_size=4096))
def test_arbitrary_artifact_bytes_are_rejected_with_a_typed_error(data: bytes) -> None:
    try:
        artifact = load_model_artifact_from_bytes(data)
    except ModelArtifactError:
        return
    assert artifact.serialized == data
    assert artifact.verify_conformance()


@settings(max_examples=200, deadline=None, derandomize=True)
@given(st.sampled_from(ARTIFACTS), st.data())
def test_every_single_byte_artifact_mutation_fails_closed(
    serialized: bytes, data: st.DataObject
) -> None:
    index = data.draw(st.integers(min_value=0, max_value=len(serialized) - 1))
    mask = data.draw(st.integers(min_value=1, max_value=255))
    mutated = bytearray(serialized)
    mutated[index] ^= mask
    with pytest.raises(ModelArtifactError):
        load_model_artifact_from_bytes(mutated)


@settings(max_examples=150, deadline=None, derandomize=True)
@given(st.sampled_from(ARTIFACTS), st.data())
def test_every_artifact_truncation_fails_closed(
    serialized: bytes, data: st.DataObject
) -> None:
    length = data.draw(st.integers(min_value=0, max_value=len(serialized) - 1))
    with pytest.raises(ModelArtifactError):
        load_model_artifact_from_bytes(serialized[:length])


@settings(max_examples=250, deadline=None, derandomize=True)
@given(
    st.dictionaries(
        st.sampled_from(("age", "state", "active", "unknown")),
        JSON_SCALAR,
        max_size=4,
    )
)
def test_raw_record_materialization_has_a_total_typed_boundary(
    record: dict[str, Any],
) -> None:
    contract = _test_preprocessing()
    try:
        result = contract.materialize(record)
    except ValueError:
        return
    assert len(result) == len(contract.outputs)
    assert all(type(value) is bool for value in result)


@settings(max_examples=300, deadline=None, derandomize=True)
@given(JSON_VALUE)
def test_arbitrary_json_values_do_not_escape_preprocessing_validation(
    value: object,
) -> None:
    try:
        contract = PreprocessingContract.from_dict(value)  # type: ignore[arg-type]
    except ValueError:
        return
    assert contract.to_dict() == value


def test_preprocessing_output_limit_is_checked_before_output_descriptors() -> None:
    oversized = {
        "schema": PREPROCESSING_SCHEMA,
        "outputs": [{}] * (MAX_PREPROCESSING_OUTPUTS + 1),
    }
    with pytest.raises(ValueError, match="4096 outputs"):
        PreprocessingContract.from_dict(oversized)


def test_preprocessing_rejects_cycles_and_excessive_nesting() -> None:
    cyclic: dict[str, object] = {
        "schema": PREPROCESSING_SCHEMA,
        "outputs": [{}],
    }
    cyclic["cycle"] = cyclic
    with pytest.raises(ValueError, match="depth 8"):
        PreprocessingContract.from_dict(cyclic)

    nested = _test_preprocessing().to_dict()
    cursor: dict[str, object] = nested
    for _ in range(12):
        child: dict[str, object] = {}
        cursor["extra"] = child
        cursor = child
    with pytest.raises(ValueError, match="depth 8"):
        PreprocessingContract.from_dict(nested)


def test_valid_digest_manifest_depth_and_constants_fail_closed() -> None:
    serialized = ARTIFACTS[0]
    manifest_size = int.from_bytes(serialized[24:32], "little")
    manifest = json.loads(serialized[64 : 64 + manifest_size])
    cursor = manifest
    for _ in range(18):
        cursor["hostile"] = {}
        cursor = cursor["hostile"]
    deeply_nested = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with pytest.raises(ModelArtifactError, match="nested too deeply"):
        load_model_artifact_from_bytes(_replace_manifest(serialized, deeply_nested))

    original = serialized[64 : 64 + manifest_size]
    nonfinite = original[:-1] + b',"hostile":NaN}'
    with pytest.raises(ModelArtifactError, match="invalid JSON"):
        load_model_artifact_from_bytes(_replace_manifest(serialized, nonfinite))


def test_manifest_size_ceiling_is_checked_before_section_walk() -> None:
    serialized = ARTIFACTS[0]
    content = bytearray(serialized[:-32])
    content[24:32] = ((16 << 20) + 1).to_bytes(8, "little")
    malformed = bytes(content) + hashlib.sha256(content).digest()
    with pytest.raises(ModelArtifactError, match="manifest exceeds"):
        load_model_artifact_from_bytes(malformed)


def test_preprocessing_round_trip_remains_canonical_under_copy() -> None:
    document = copy.deepcopy(_test_preprocessing().to_dict())
    assert PreprocessingContract.from_dict(document).to_dict() == document
