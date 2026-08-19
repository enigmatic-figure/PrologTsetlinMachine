"""Budgeted feature persistence and retirement (Milestone 4).

Host-side, deterministic store that bounds the LiteralCatalog.
Persists a canonical JSON envelope with SHA-256 content addressing,
atomic file replacement, and deterministic retirement policies.

Not part of ptm.preprocessing.v1 portability — host preprocessing.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .representation import (
    TRANSFORM_CATALOG_VERSION,
    FeatureSchema,
    FieldDefinition,
    FieldKind,
    LiteralCatalog,
    LiteralDescriptor,
    NullPolicy,
    TransformKind,
    _freeze_parameter,
    _thaw_parameter,
)

BUDGETED_STORE_SCHEMA = "ptm.budgeted_features.v1"
MAX_BUDGET = 4096
MAX_STORE_BYTES = 2 * 1024 * 1024  # 2 MiB — keeps host checkpoint bounded
MAX_LITERAL_BYTES = 8 * 1024

RetirementPolicy = str  # "oldest" | "least_used" | "lowest_utility"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stable_id(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass(slots=True)
class BudgetedLiteralRecord:
    descriptor: LiteralDescriptor
    use_count: int = 0
    utility: float = 0.0
    creation_order: int = 0


class BudgetedFeatureStore:
    """Bounded, deterministic literal store with persistence and retirement.

    Example:
        schema = FeatureSchema.from_fields(score=FieldKind.NUMBER)
        store = BudgetedFeatureStore(schema, budget=128, policy="least_used")
        d = store.catalog.numeric_ge("score", 10)
        store.record_use(d.literal_id)
        store.persist("out/store.json")
        restored = BudgetedFeatureStore.load("out/store.json")
    """

    def __init__(
        self,
        schema: FeatureSchema,
        *,
        budget: int = 256,
        policy: RetirementPolicy = "least_used",
    ) -> None:
        if not isinstance(budget, int) or isinstance(budget, bool):
            raise ValueError("budget must be integer")
        if not 1 <= budget <= MAX_BUDGET:
            raise ValueError(f"budget must be 1..{MAX_BUDGET}")
        if policy not in ("oldest", "least_used", "lowest_utility"):
            raise ValueError("policy must be oldest, least_used, or lowest_utility")
        self.schema = schema
        self.budget = budget
        self.policy = policy
        self._catalog = LiteralCatalog(schema)
        self._records: dict[int, BudgetedLiteralRecord] = {}
        self._order_counter = 0
        self._schema_version = 1
        # Wrap catalog._register to auto-track and enforce budget deterministically
        orig_register = self._catalog._register  # type: ignore[attr-defined]

        def _tracking_register(field_name: str, transform, parameters, null_policy):  # type: ignore[no-untyped-def]
            desc = orig_register(field_name, transform, parameters, null_policy)
            # _track will enforce budget if needed
            self._track(desc)
            return desc

        self._catalog._register = _tracking_register  # type: ignore[attr-defined]

    @property
    def catalog(self) -> LiteralCatalog:
        return self._catalog

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def literals(self) -> tuple[LiteralDescriptor, ...]:
        return tuple(r.descriptor for r in sorted(self._records.values(), key=lambda r: r.creation_order))

    def _track(self, descriptor: LiteralDescriptor) -> BudgetedLiteralRecord:
        rec = self._records.get(descriptor.literal_id)
        if rec is not None:
            return rec
        rec = BudgetedLiteralRecord(descriptor=descriptor, creation_order=self._order_counter)
        self._order_counter += 1
        self._records[descriptor.literal_id] = rec
        # mirror into catalog order
        # LiteralCatalog already holds descriptor; ensure store and catalog stay in sync
        if descriptor not in self._catalog.literals:
            # already registered via catalog.numeric_ge etc; the catalog already contains it
            pass
        self._enforce(exclude_id=rec.descriptor.literal_id)
        return rec

    def add_literal(self, descriptor: LiteralDescriptor) -> BudgetedLiteralRecord:
        """Add an existing descriptor under budget (idempotent)."""
        if not isinstance(descriptor, LiteralDescriptor):
            raise TypeError("descriptor must be LiteralDescriptor")
        if descriptor.source_field not in {f.name for f in self.schema.fields}:
            raise ValueError("descriptor field not in schema")
        # ensure catalog knows it (re-register via private path is not needed;
        # just ensure store tracks it)
        rec = self._track(descriptor)
        # Also ensure catalog contains it — if descriptor was built from another catalog
        # with same schema, re-register via its transform
        if descriptor.literal_id not in {d.literal_id for d in self._catalog.literals}:
            # Re-create via public API to get canonical ID (should match)
            # We cannot directly inject; use _register via known transform
            # Fallback: inject via internal map for determinism
            self._catalog._by_id[descriptor.literal_id] = descriptor  # type: ignore[attr-defined]
            self._catalog._literals.append(descriptor)  # type: ignore[attr-defined]
        return rec

    def record_use(self, literal_id: int, count: int = 1) -> None:
        rec = self._records.get(literal_id)
        if rec is None:
            raise KeyError(f"unknown literal_id {literal_id}")
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be positive int")
        object.__setattr__(rec, "use_count", rec.use_count + count)  # type: ignore[attr-defined]

    def set_utility(self, literal_id: int, utility: float) -> None:
        rec = self._records.get(literal_id)
        if rec is None:
            raise KeyError(f"unknown literal_id {literal_id}")
        if not isinstance(utility, (int, float)) or not (utility == utility):  # NaN check
            raise ValueError("utility must be finite number")
        object.__setattr__(rec, "utility", float(utility))  # type: ignore[attr-defined]

    def _enforce(self, exclude_id: int | None = None) -> list[int]:
        if len(self._records) <= self.budget:
            return []
        # deterministic ordering
        def key(rec: BudgetedLiteralRecord) -> tuple[Any, int]:
            if self.policy == "oldest":
                return (rec.creation_order, rec.descriptor.literal_id)
            if self.policy == "least_used":
                return (rec.use_count, rec.creation_order)
            # lowest_utility
            return (rec.utility, rec.creation_order)

        # optionally protect newly added item for this round
        candidates = [r for r in self._records.values() if r.descriptor.literal_id != exclude_id]
        sorted_recs = sorted(candidates, key=key)  # type: ignore[arg-type]
        to_remove = len(self._records) - self.budget
        # if exclude_id protected us from evicting enough, we need to evict from remaining
        # but we already excluded new item, so we evict oldest among old items
        removed: list[int] = []
        for rec in sorted_recs[:to_remove]:
            removed.append(rec.descriptor.literal_id)
            del self._records[rec.descriptor.literal_id]
            # also remove from catalog list
            self._catalog._literals = [d for d in self._catalog._literals if d.literal_id != rec.descriptor.literal_id]  # type: ignore[attr-defined]
            self._catalog._by_id.pop(rec.descriptor.literal_id, None)  # type: ignore[attr-defined]
        return removed

    def retire_least_useful(self, count: int = 1) -> list[int]:
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be positive int")
        # Temporarily enforce tighter budget
        old = self.budget
        self.budget = max(1, len(self._records) - count)
        removed = self._enforce()
        self.budget = old
        return removed

    # --- serialization ---

    def to_dict(self) -> dict[str, Any]:
        fields = [{"name": f.name, "kind": f.kind.value, "source_field_id": f.source_field_id} for f in self.schema.fields]
        literals = []
        for rec in sorted(self._records.values(), key=lambda r: r.creation_order):
            d = rec.descriptor
            literals.append(
                {
                    "literal_id": d.literal_id,
                    "source_field": d.source_field,
                    "source_field_id": d.source_field_id,
                    "transform": d.transform.value,
                    "parameters": {k: _thaw_parameter(v) for k, v in d.parameters},
                    "null_policy": d.null_policy.value,
                    "catalog_version": d.catalog_version,
                    "use_count": rec.use_count,
                    "utility": rec.utility,
                    "creation_order": rec.creation_order,
                }
            )
        payload: dict[str, Any] = {
            "schema": BUDGETED_STORE_SCHEMA,
            "schema_version": self._schema_version,
            "budget": self.budget,
            "policy": self.policy,
            "fields": fields,
            "literals": literals,
        }
        payload["store_id"] = _stable_id({k: v for k, v in payload.items() if k != "store_id"})
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BudgetedFeatureStore":
        if not isinstance(value, Mapping):
            raise ValueError("budgeted store must be object")
        if value.get("schema") != BUDGETED_STORE_SCHEMA:
            raise ValueError("unsupported budgeted store schema")
        if value.get("schema_version") != 1:
            raise ValueError("unsupported schema_version")
        budget = value.get("budget")
        policy = value.get("policy")
        fields_raw = value.get("fields")
        literals_raw = value.get("literals")
        if not isinstance(budget, int) or not 1 <= budget <= MAX_BUDGET:
            raise ValueError("budget invalid")
        if policy not in ("oldest", "least_used", "lowest_utility"):
            raise ValueError("policy invalid")
        if not isinstance(fields_raw, list) or not fields_raw:
            raise ValueError("fields invalid")
        if not isinstance(literals_raw, list):
            raise ValueError("literals invalid")
        if len(literals_raw) > budget and len(literals_raw) > MAX_BUDGET:
            raise ValueError("store exceeds budget ceiling")
        # rebuild schema
        fds = []
        for f in fields_raw:
            if not isinstance(f, Mapping):
                raise ValueError("field entry invalid")
            name = f.get("name")
            kind = f.get("kind")
            sid = f.get("source_field_id")
            if not isinstance(name, str) or not name or not isinstance(sid, int):
                raise ValueError("field name/id invalid")
            try:
                k = FieldKind(kind)
            except Exception as e:
                raise ValueError(f"field kind invalid: {kind}") from e
            fds.append(FieldDefinition(source_field_id=int(sid), name=str(name), kind=k))
        schema = FeatureSchema(fds)
        store = cls(schema, budget=budget, policy=policy)
        # verify canonical store_id
        expected_id = _stable_id({k: v for k, v in value.items() if k != "store_id"})
        if value.get("store_id") != expected_id:
            raise ValueError("store_id mismatch — not canonical or corrupted")
        # rebuild literals
        for entry in literals_raw:
            if not isinstance(entry, Mapping):
                raise ValueError("literal entry invalid")
            try:
                source_field = str(entry["source_field"])
                transform = TransformKind(str(entry["transform"]))
                null_policy = NullPolicy(str(entry["null_policy"]))
                params = dict(entry.get("parameters") or {})
                frozen = tuple(sorted((str(k), _freeze_parameter(v)) for k, v in params.items()))
                # stable ID check
                field = schema.field(source_field)
                payload = {
                    "catalog_version": TRANSFORM_CATALOG_VERSION,
                    "source_field_id": field.source_field_id,
                    "transform": transform.value,
                    "parameters": {k: _thaw_parameter(v) for k, v in frozen},
                    "null_policy": null_policy.value,
                }
                # ID must match stored literal_id
                lid = int(entry["literal_id"])
                import hashlib, json as _json

                calc = int.from_bytes(
                    hashlib.sha256(_json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).digest()[:8], "big"
                )
                if calc != lid:
                    raise ValueError("literal_id does not match descriptor")
                if entry.get("catalog_version") != TRANSFORM_CATALOG_VERSION:
                    raise ValueError("catalog_version mismatch")
                desc = LiteralDescriptor(
                    literal_id=lid,
                    source_field_id=field.source_field_id,
                    source_field=field.name,
                    transform=transform,
                    parameters=frozen,
                    null_policy=null_policy,
                    catalog_version=TRANSFORM_CATALOG_VERSION,
                )
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"literal entry invalid: {e}") from e
            rec = BudgetedLiteralRecord(
                descriptor=desc,
                use_count=int(entry.get("use_count", 0)),
                utility=float(entry.get("utility", 0.0)),
                creation_order=int(entry.get("creation_order", 0)),
            )
            if rec.use_count < 0 or rec.utility != rec.utility:
                raise ValueError("use_count/utility invalid")
            if len(_canonical(entry).encode("utf-8")) > MAX_LITERAL_BYTES:
                raise ValueError("literal entry oversized")
            store._records[desc.literal_id] = rec
            store._catalog._by_id[desc.literal_id] = desc  # type: ignore[attr-defined]
            store._catalog._literals.append(desc)  # type: ignore[attr-defined]
            store._order_counter = max(store._order_counter, rec.creation_order + 1)
        # final budget enforcement (should be within budget already)
        if len(store._records) > store.budget:
            raise ValueError("store exceeds declared budget")
        # canonical round-trip check
        if store.to_dict() != dict(value):
            raise ValueError("store dict is not canonical")
        json_bytes = _canonical(value).encode("utf-8")
        if len(json_bytes) > MAX_STORE_BYTES:
            raise ValueError("store exceeds byte ceiling")
        return store

    def persist(self, path: str | Path) -> str:
        data = self.to_dict()
        text = _canonical(data)
        encoded = text.encode("utf-8")
        if len(encoded) > MAX_STORE_BYTES:
            raise ValueError("store exceeds persistence ceiling")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # atomic write
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix="." + p.name + ".tmp.")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(encoded)
                f.flush()
                os.fsync(f.fileno())
            # fsync directory on POSIX
            try:
                dfd = os.open(str(p.parent), os.O_DIRECTORY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except OSError:
                pass
            os.replace(tmp, str(p))
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
        return data["store_id"]

    @classmethod
    def load(cls, path: str | Path) -> "BudgetedFeatureStore":
        p = Path(path)
        data = p.read_bytes()
        if len(data) > MAX_STORE_BYTES:
            raise ValueError("store file exceeds ceiling")
        if not data:
            raise ValueError("store file empty")
        try:
            obj = json.loads(data.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"store JSON invalid: {e}") from e
        return cls.from_dict(obj)
