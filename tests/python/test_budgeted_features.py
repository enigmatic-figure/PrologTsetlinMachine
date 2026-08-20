"""Tests for budgeted feature persistence and retirement."""

import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from prolog_tsetlin.representation import (
    TRANSFORM_CATALOG_VERSION,
    FeatureSchema,
    FieldKind,
    LiteralDescriptor,
    NullPolicy,
    TransformKind,
)
from prolog_tsetlin.budgeted_features import (
    MAX_STORE_BYTES,
    BudgetedFeatureStore,
    MAX_BUDGET,
)


def _schema() -> FeatureSchema:
    return FeatureSchema.from_fields(score=FieldKind.NUMBER, city=FieldKind.CATEGORY, text=FieldKind.TEXT)


def _stable_literal_id(source_field_id: int, transform: TransformKind, parameters: dict[str, object]) -> int:
    payload = {
        "catalog_version": TRANSFORM_CATALOG_VERSION,
        "source_field_id": source_field_id,
        "transform": transform.value,
        "parameters": parameters,
        "null_policy": NullPolicy.FALSE.value,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return int.from_bytes(hashlib.sha256(canonical.encode("utf-8")).digest()[:8], "big")


def _rehash_store(document: dict[str, object]) -> None:
    payload = {key: value for key, value in document.items() if key != "store_id"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    document["store_id"] = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_budgeted_basic_add_and_enforce() -> None:
    store = BudgetedFeatureStore(_schema(), budget=2, policy="oldest")
    d1 = store.catalog.numeric_ge("score", 10)
    d2 = store.catalog.numeric_ge("score", 20)
    assert store.size == 2
    d3 = store.catalog.numeric_ge("score", 30)
    assert store.size == 2  # oldest evicted
    assert d1.literal_id not in {d.literal_id for d in store.literals}
    assert d3.literal_id in {d.literal_id for d in store.literals}


def test_budgeted_least_used_retires_least() -> None:
    store = BudgetedFeatureStore(_schema(), budget=2, policy="least_used")
    d1 = store.catalog.numeric_ge("score", 10)
    store.record_use(d1.literal_id, 5)
    d2 = store.catalog.numeric_ge("score", 20)
    store.record_use(d2.literal_id, 1)
    d3 = store.catalog.category_eq("city", "OSLO")
    # d1 has 5, d2 has 1, d3 has 0+1 after record -> will retire d2 (least among old)
    store.record_use(d3.literal_id, 1)
    ids = {d.literal_id for d in store.literals}
    assert d1.literal_id in ids
    assert d2.literal_id not in ids
    assert d3.literal_id in ids


def test_budgeted_lowest_utility() -> None:
    schema = FeatureSchema.from_fields(x=FieldKind.NUMBER)
    store = BudgetedFeatureStore(schema, budget=2, policy="lowest_utility")
    a = store.catalog.numeric_ge("x", 1)
    store.set_utility(a.literal_id, 10)
    b = store.catalog.numeric_ge("x", 2)
    store.set_utility(b.literal_id, 1)
    c = store.catalog.numeric_ge("x", 3)
    store.set_utility(c.literal_id, 5)
    ids = {d.literal_id for d in store.literals}
    # lowest utility b should be evicted, keep a and c
    assert a.literal_id in ids
    assert b.literal_id not in ids
    assert c.literal_id in ids


def test_budgeted_persist_and_load_canonical(tmp_path: Path) -> None:
    store = BudgetedFeatureStore(_schema(), budget=3, policy="least_used")
    d1 = store.catalog.numeric_ge("score", 5)
    store.record_use(d1.literal_id, 2)
    store.catalog.token_contains("text", "hello")
    path = tmp_path / "store.json"
    sid = store.persist(path)
    assert sid.startswith("sha256:")
    restored = BudgetedFeatureStore.load(path)
    assert restored.to_dict() == store.to_dict()
    assert restored.size == store.size


def test_budgeted_persist_without_directory_fsync_support(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "O_DIRECTORY", raising=False)
    path = tmp_path / "store.json"
    store = BudgetedFeatureStore(_schema(), budget=3)
    store.catalog.numeric_ge("score", 5)

    store_id = store.persist(path)

    assert store_id.startswith("sha256:")
    assert BudgetedFeatureStore.load(path).to_dict() == store.to_dict()


def test_budgeted_load_uses_bounded_read(monkeypatch: pytest.MonkeyPatch) -> None:
    read_sizes: list[int] = []

    class OversizedSource(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return b"x" * size

    monkeypatch.setattr(
        Path,
        "open",
        lambda self, mode="r", *args, **kwargs: OversizedSource(),
    )

    with pytest.raises(ValueError, match="exceeds ceiling"):
        BudgetedFeatureStore.load("hostile-store.json")

    assert read_sizes == [MAX_STORE_BYTES + 1]


def test_budgeted_persist_cleans_temp_after_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "store.json"
    path.write_bytes(b"original")
    store = BudgetedFeatureStore(_schema(), budget=3)
    store.catalog.numeric_ge("score", 5)

    def fail_replace(source: str, target: str) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        store.persist(path)

    assert path.read_bytes() == b"original"
    assert list(tmp_path.glob(".store.json.tmp.*")) == []


def test_budgeted_persist_ignores_directory_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_open = os.open
    monkeypatch.setattr(os, "O_DIRECTORY", 0, raising=False)

    def fail_directory_open(path: str, flags: int, *args: object, **kwargs: object) -> int:
        if Path(path) == tmp_path:
            raise OSError("directory fsync unsupported")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", fail_directory_open)
    path = tmp_path / "store.json"
    store = BudgetedFeatureStore(_schema(), budget=3)
    store.catalog.numeric_ge("score", 5)

    store.persist(path)

    assert path.is_file()
    assert list(tmp_path.glob(".store.json.tmp.*")) == []


def test_budgeted_hostile_budget() -> None:
    with pytest.raises(ValueError):
        BudgetedFeatureStore(_schema(), budget=0)
    with pytest.raises(ValueError):
        BudgetedFeatureStore(_schema(), budget=MAX_BUDGET + 1)
    with pytest.raises(ValueError):
        BudgetedFeatureStore(_schema(), budget=2, policy="bad")  # type: ignore[arg-type]


def test_budgeted_hostile_tampered_store_id() -> None:
    store = BudgetedFeatureStore(_schema(), budget=2)
    store.catalog.numeric_ge("score", 1)
    d = store.to_dict()
    d["store_id"] = "sha256:" + "00" * 32
    with pytest.raises(ValueError):
        BudgetedFeatureStore.from_dict(d)


def test_budgeted_rejects_invalid_external_descriptor() -> None:
    schema = _schema()
    field = schema.field("score")
    parameters = {"token": "hello", "case_sensitive": False}
    descriptor = LiteralDescriptor(
        literal_id=_stable_literal_id(field.source_field_id, TransformKind.TOKEN_CONTAINS, parameters),
        source_field_id=field.source_field_id,
        source_field=field.name,
        transform=TransformKind.TOKEN_CONTAINS,
        parameters=(("case_sensitive", False), ("token", "hello")),
        null_policy=NullPolicy.FALSE,
    )

    with pytest.raises(TypeError, match="must be one of: text"):
        BudgetedFeatureStore(schema, budget=2).add_literal(descriptor)


def test_budgeted_rejects_rehashed_semantically_invalid_store() -> None:
    store = BudgetedFeatureStore(_schema(), budget=2)
    store.catalog.numeric_ge("score", 1)
    document = store.to_dict()
    entry = document["literals"][0]
    parameters = {"token": "hello", "case_sensitive": False}
    entry["transform"] = TransformKind.TOKEN_CONTAINS.value
    entry["parameters"] = parameters
    entry["literal_id"] = _stable_literal_id(
        entry["source_field_id"], TransformKind.TOKEN_CONTAINS, parameters
    )
    _rehash_store(document)

    with pytest.raises(ValueError, match="literal entry invalid"):
        BudgetedFeatureStore.from_dict(document)


def test_budgeted_retire_least_useful() -> None:
    schema = FeatureSchema.from_fields(x=FieldKind.NUMBER)
    store = BudgetedFeatureStore(schema, budget=4, policy="least_used")
    ids = []
    for i in range(4):
        d = store.catalog.numeric_ge("x", i)
        ids.append(d.literal_id)
        store.record_use(d.literal_id, i + 1)
    removed = store.retire_least_useful(1)
    assert len(removed) == 1
    assert removed[0] not in {d.literal_id for d in store.literals}


def test_budgeted_invalid_budget_type_rejected() -> None:
    store = BudgetedFeatureStore(_schema(), budget=2)
    store.catalog.numeric_ge("score", 1)
    bad = store.to_dict()
    bad["budget"] = "2"  # wrong type
    with pytest.raises(ValueError):
        BudgetedFeatureStore.from_dict(bad)  # type: ignore[arg-type]


def test_budgeted_infinite_utility_rejected_and_oversized_literals():
    store = BudgetedFeatureStore(_schema(), budget=5)
    d = store.catalog.numeric_ge("score", 1)
    with pytest.raises(ValueError):
        store.set_utility(d.literal_id, float("inf"))
    with pytest.raises(ValueError):
        store.set_utility(d.literal_id, float("nan"))
    # oversized literal via from_dict: literal entry oversized should be rejected
    bad = store.to_dict()
    # inject an entry with huge parameters (oversized literal)
    huge_str = "x" * 9000
    bad["literals"][0]["parameters"] = {"threshold": huge_str}
    # store_id must be recomputed to pass that check, but oversize check is after
    import hashlib, json as _json
    canonical = _json.dumps({k: v for k, v in bad.items() if k != "store_id"}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    bad["store_id"] = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    with pytest.raises(ValueError):
        BudgetedFeatureStore.from_dict(bad)
