from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Callable
from unittest import TestCase

from prolog_tsetlin import (
    FeatureSchema,
    FieldKind,
    LiteralCatalog,
    NullPolicy,
    PreprocessingContract,
    ScalarBinaryTsetlinMachine,
    ModelArtifactError,
    export_packed_tm,
    load_model_artifact_from_bytes,
)


ASSERTIONS = TestCase()


def _rewrite_manifest(
    serialized: bytes, mutate: Callable[[dict[str, Any]], None]
) -> bytes:
    manifest_size = int.from_bytes(serialized[24:32], "little")
    payload_size = int.from_bytes(serialized[32:40], "little")
    first = 64
    last = first + manifest_size
    manifest = json.loads(serialized[first:last])
    mutate(manifest)
    manifest_bytes = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    header = bytearray(serialized[:64])
    header[24:32] = len(manifest_bytes).to_bytes(8, "little")
    content = bytes(header) + manifest_bytes + serialized[last : last + payload_size]
    return content + hashlib.sha256(content).digest()


def xor_machine() -> ScalarBinaryTsetlinMachine:
    machine = ScalarBinaryTsetlinMachine(
        4, 2, states_per_action=3, specificity=3.0, threshold=8, seed=41
    )
    for clause in range(4):
        for literal in range(4):
            machine.set_state(clause, literal, 3)
    machine.set_state(0, 0, 4)
    machine.set_state(0, 3, 4)
    machine.set_state(1, 0, 4)
    machine.set_state(1, 2, 4)
    machine.set_state(2, 1, 4)
    machine.set_state(2, 2, 4)
    machine.set_state(3, 1, 4)
    machine.set_state(3, 3, 4)
    return machine


def test_preprocessing_round_trips_and_materializes_typed_records() -> None:
    schema = FeatureSchema.from_fields(
        age=FieldKind.NUMBER,
        status=FieldKind.CATEGORY,
        active=FieldKind.BOOLEAN,
    )
    catalog = LiteralCatalog(schema)
    catalog.numeric_ge("age", 18, null_policy=NullPolicy.ERROR)
    catalog.numeric_between("age", 21, 65)
    catalog.category_eq("status", "ready", null_policy=NullPolicy.TRUE)
    catalog.category_in("status", ["ready", "running"])
    catalog.category_eq("active", True)
    catalog.is_missing("active")
    contract = PreprocessingContract.from_catalog(catalog)

    assert contract.materialize(
        {"age": 30, "status": "ready", "active": True}
    ) == (True, True, True, True, True, False)
    assert contract.materialize({"age": 19, "status": None}) == (
        True,
        False,
        True,
        False,
        False,
        True,
    )
    assert PreprocessingContract.from_dict(contract.to_dict()) == contract


def test_preprocessing_rejects_ambiguous_types_and_tampered_ids() -> None:
    schema = FeatureSchema.from_fields(age=FieldKind.NUMBER)
    catalog = LiteralCatalog(schema)
    catalog.numeric_ge("age", 18)
    contract = PreprocessingContract.from_catalog(catalog)
    with ASSERTIONS.assertRaisesRegex(ValueError, "must be a number"):
        contract.materialize({"age": True})
    with ASSERTIONS.assertRaisesRegex(ValueError, "binary64"):
        contract.materialize({"age": 1 << 54})
    with ASSERTIONS.assertRaisesRegex(ValueError, "finite"):
        contract.materialize({"age": float("nan")})

    tampered = copy.deepcopy(contract.to_dict())
    tampered["outputs"][0]["literal_id"] = "1"
    with ASSERTIONS.assertRaisesRegex(ValueError, "literal ID"):
        PreprocessingContract.from_dict(tampered)


def test_literal_catalog_and_portable_contract_share_typed_semantics() -> None:
    schema = FeatureSchema.from_fields(
        age=FieldKind.NUMBER,
        status=FieldKind.CATEGORY,
        active=FieldKind.BOOLEAN,
    )
    catalog = LiteralCatalog(schema)
    catalog.numeric_ge("age", 18)
    catalog.category_in("status", [True, 1, "1"])
    catalog.category_eq("active", True)
    contract = PreprocessingContract.from_catalog(catalog)

    for record in (
        {"age": 18, "status": True, "active": True},
        {"age": 17.5, "status": 1, "active": False},
        {"age": 20, "status": "1", "active": True},
        {"age": None, "status": None, "active": None},
    ):
        assert catalog.encode([record]).ta.row_values(0) == contract.materialize(
            record
        )

    for malformed in (
        {"age": True, "status": 1, "active": True},
        {"age": float("nan"), "status": 1, "active": True},
        {"age": 18, "status": 1.0, "active": True},
        {"age": 18, "status": 1, "active": 1},
    ):
        with ASSERTIONS.assertRaises(ValueError):
            catalog.encode([malformed])
        with ASSERTIONS.assertRaises(ValueError):
            contract.materialize(malformed)


def test_artifact_rejects_rehashed_content_identity_forgery() -> None:
    schema = FeatureSchema.from_fields(
        left=FieldKind.BOOLEAN,
        right=FieldKind.BOOLEAN,
    )
    catalog = LiteralCatalog(schema)
    catalog.category_eq("left", True)
    catalog.category_eq("right", True)
    preprocessing = PreprocessingContract.from_catalog(catalog)
    artifact = export_packed_tm(
        xor_machine().snapshot(),
        name="Raw-record identity forgery test",
        preprocessing=preprocessing,
        validation_records=({"left": False, "right": False},),
    )

    def forge_field(manifest: dict[str, Any]) -> None:
        manifest["preprocessing"]["outputs"][0]["field_id"] = "1"

    with ASSERTIONS.assertRaises(ModelArtifactError):
        load_model_artifact_from_bytes(
            _rewrite_manifest(artifact.serialized, forge_field)
        )

    def forge_literal(manifest: dict[str, Any]) -> None:
        manifest["preprocessing"]["outputs"][0]["literal_id"] = "1"
        manifest["features"]["literal_ids"][0] = "1"

    with ASSERTIONS.assertRaises(ModelArtifactError):
        load_model_artifact_from_bytes(
            _rewrite_manifest(artifact.serialized, forge_literal)
        )


def test_preprocessing_rejects_nonportable_token_semantics() -> None:
    schema = FeatureSchema.from_fields(text=FieldKind.TEXT)
    catalog = LiteralCatalog(schema)
    catalog.token_contains("text", "hello", case_sensitive=False)
    with ASSERTIONS.assertRaisesRegex(ValueError, "not portable"):
        PreprocessingContract.from_catalog(catalog)


def test_category_membership_preserves_typed_values() -> None:
    schema = FeatureSchema.from_fields(value=FieldKind.CATEGORY)
    catalog = LiteralCatalog(schema)
    descriptor = catalog.category_in("value", [True, 1, "1", True, 1])
    preprocessing = PreprocessingContract.from_catalog(catalog)
    assert descriptor.parameter("values") == (True, 1, "1")
    assert preprocessing.materialize({"value": True}) == (True,)
    assert preprocessing.materialize({"value": 1}) == (True,)
    assert preprocessing.materialize({"value": "1"}) == (True,)

    boolean_catalog = LiteralCatalog(
        FeatureSchema.from_fields(flag=FieldKind.BOOLEAN)
    )
    boolean_catalog.category_eq("flag", 1)
    with ASSERTIONS.assertRaisesRegex(ValueError, "Boolean category"):
        PreprocessingContract.from_catalog(boolean_catalog)


def test_preprocessing_rejects_control_characters() -> None:
    catalog = LiteralCatalog(
        FeatureSchema.from_fields(category=FieldKind.CATEGORY)
    )
    catalog.category_eq("category", "bad\nvalue")
    with ASSERTIONS.assertRaisesRegex(ValueError, "control characters"):
        PreprocessingContract.from_catalog(catalog)


def test_packed_artifact_predicts_raw_records_deterministically() -> None:
    schema = FeatureSchema.from_fields(
        left=FieldKind.BOOLEAN,
        right=FieldKind.BOOLEAN,
    )
    catalog = LiteralCatalog(schema)
    catalog.category_eq("left", True)
    catalog.category_eq("right", True)
    preprocessing = PreprocessingContract.from_catalog(catalog)
    records = (
        {"left": False, "right": False},
        {"left": False, "right": True},
        {"left": True, "right": False},
        {"left": True, "right": True},
    )
    artifact = export_packed_tm(
        xor_machine().snapshot(),
        name="Raw-record XOR",
        preprocessing=preprocessing,
        validation_records=records,
        feature_names=("left", "right"),
        feature_catalog_version="raw-xor-v1",
    )
    loaded = load_model_artifact_from_bytes(artifact.serialized)
    assert loaded.manifest["features"]["materialization"] == (
        "precomputed_or_raw_record_v1"
    )
    assert loaded.predict_records(records) == (0, 1, 1, 0)
    assert tuple(loaded.iter_predict_records(iter(records))) == (0, 1, 1, 0)
    assert loaded.preprocessing == preprocessing


def test_artifact_rejects_disagreement_between_rows_and_records() -> None:
    schema = FeatureSchema.from_fields(
        left=FieldKind.BOOLEAN,
        right=FieldKind.BOOLEAN,
    )
    catalog = LiteralCatalog(schema)
    catalog.category_eq("left", True)
    catalog.category_eq("right", True)
    preprocessing = PreprocessingContract.from_catalog(catalog)
    with ASSERTIONS.assertRaisesRegex(ValueError, "disagree"):
        export_packed_tm(
            xor_machine().snapshot(),
            name="Bad raw-record XOR",
            preprocessing=preprocessing,
            validation_records=({"left": False, "right": False},),
            validation_rows=((1, 1),),
        )
