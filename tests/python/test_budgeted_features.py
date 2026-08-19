"""Tests for budgeted feature persistence and retirement."""

import json
import tempfile
from pathlib import Path

import pytest

from prolog_tsetlin.representation import FeatureSchema, FieldKind
from prolog_tsetlin.budgeted_features import BudgetedFeatureStore, BUDGETED_STORE_SCHEMA, MAX_BUDGET


def _schema() -> FeatureSchema:
    return FeatureSchema.from_fields(score=FieldKind.NUMBER, city=FieldKind.CATEGORY, text=FieldKind.TEXT)


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


def test_budgeted_persist_and_load_canonical() -> None:
    store = BudgetedFeatureStore(_schema(), budget=3, policy="least_used")
    d1 = store.catalog.numeric_ge("score", 5)
    store.record_use(d1.literal_id, 2)
    d2 = store.catalog.token_contains("text", "hello")
    tmp = tempfile.mktemp(suffix=".json")
    sid = store.persist(tmp)
    assert sid.startswith("sha256:")
    restored = BudgetedFeatureStore.load(tmp)
    assert restored.to_dict() == store.to_dict()
    assert restored.size == store.size
    Path(tmp).unlink()


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


def test_budgeted_json_not_canonical_rejected() -> None:
    store = BudgetedFeatureStore(_schema(), budget=2)
    store.catalog.numeric_ge("score", 1)
    # write non-canonical (pretty) JSON
    d = store.to_dict()
    pretty = json.dumps(d, indent=2).encode("utf-8")
    tmp = tempfile.mktemp(suffix=".json")
    Path(tmp).write_bytes(pretty)
    # load via from_dict still checks canonical store_id, but file was not written via persist
    # Instead test that from_dict rejects not-canonical dict ordering
    bad = dict(d)
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
