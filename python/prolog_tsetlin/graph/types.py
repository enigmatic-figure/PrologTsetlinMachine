"""Graph input contracts — bounded, deterministic, content-hashable."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

MAX_NODES = 4096
MAX_EDGES = 16384
MAX_DISTINCT_PROPERTIES = 4096
MAX_EDGE_TYPES = 16
MAX_PROPERTY_STRING_CHARS = 256

GRAPH_SCHEMA = "ptm.graph.v1"


class GraphValidationError(ValueError):
    """Raised when a supplied graph violates its bounded contract."""


def _stable_property_id(value: object) -> str:
    if isinstance(value, str):
        if not value or len(value) > MAX_PROPERTY_STRING_CHARS:
            raise GraphValidationError("property string must be 1..256 chars")
        if any(ord(c) < 0x20 for c in value):
            raise GraphValidationError("property contains control character")
        canonical = f"str:{value}"
    elif isinstance(value, int) and not isinstance(value, bool):
        if not -(1 << 53) <= value <= (1 << 53):
            raise GraphValidationError("integer property out of 53-bit range")
        canonical = f"int:{value}"
    elif isinstance(value, bool):
        canonical = f"bool:{value}"
    elif isinstance(value, float):
        if not (value == value and abs(value) != float("inf")):
            raise GraphValidationError("property float must be finite")
        canonical = f"float:{repr(value)}"
    else:
        raise GraphValidationError("property must be str, int, bool, or finite float")
    return "prop:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class GraphInput:
    """Directed labeled multigraph with typed edges and node property sets.

    - Nodes are contiguous ``0..n-1`` (validated).
    - Each edge is ``(src, dst, edge_type)`` where ``edge_type`` is a small int
      in ``0..MAX_EDGE_TYPES-1`` or a short string.
    - Each node maps to a frozenset of stable property-ids (first hashed).
    - Raw property values are kept for provenance but not used for identity.
    """

    node_count: int
    edges: tuple[tuple[int, int, object], ...]
    node_properties_raw: tuple[frozenset[object], ...]
    node_properties: tuple[frozenset[str], ...]
    edge_types: frozenset[object]
    schema: str = GRAPH_SCHEMA

    @classmethod
    def create(
        cls,
        *,
        node_count: int,
        edges: list[tuple[int, int, object]] | tuple[tuple[int, int, object], ...] | None = None,
        node_properties: dict[int, list[object] | tuple[object, ...] | set[object]] | None = None,
        validate: bool = True,
    ) -> "GraphInput":
        if not isinstance(node_count, int) or isinstance(node_count, bool):
            raise GraphValidationError("node_count must be int")
        if not 1 <= node_count <= MAX_NODES:
            raise GraphValidationError(f"node_count must be 1..{MAX_NODES}")
        raw_edges = tuple(edges or ())
        raw_props: dict[int, frozenset[object]] = {}
        if node_properties:
            for node, props in node_properties.items():
                if not isinstance(node, int) or isinstance(node, bool):
                    raise GraphValidationError("node index must be int")
                if not 0 <= node < node_count:
                    raise GraphValidationError("node property for unknown node")
                prop_set = frozenset(props) if props is not None else frozenset()
                # validate each property
                for p in prop_set:
                    _stable_property_id(p)
                if len(prop_set) > 256:
                    raise GraphValidationError("too many properties on one node")
                raw_props[node] = prop_set
        # fill missing nodes with empty set
        node_props_raw: list[frozenset[object]] = []
        for n in range(node_count):
            node_props_raw.append(raw_props.get(n, frozenset()))

        if len(raw_edges) > MAX_EDGES:
            raise GraphValidationError(f"edge count exceeds {MAX_EDGES}")

        edge_type_set: set[object] = set()
        normalized_edges: list[tuple[int, int, object]] = []
        for src, dst, etype in raw_edges:
            if not isinstance(src, int) or isinstance(src, bool):
                raise GraphValidationError("edge src must be int")
            if not isinstance(dst, int) or isinstance(dst, bool):
                raise GraphValidationError("edge dst must be int")
            if not 0 <= src < node_count or not 0 <= dst < node_count:
                raise GraphValidationError("edge endpoint out of range")
            if isinstance(etype, int) and not isinstance(etype, bool):
                if not 0 <= etype < MAX_EDGE_TYPES:
                    raise GraphValidationError("int edge_type must be 0..15")
            elif isinstance(etype, str):
                if not etype or len(etype) > 64 or any(ord(c) < 0x20 for c in etype):
                    raise GraphValidationError("str edge_type must be 1..64 printable chars")
            else:
                raise GraphValidationError("edge_type must be int 0..15 or short str")
            edge_type_set.add(etype)
            normalized_edges.append((src, dst, etype))

        if len(edge_type_set) > MAX_EDGE_TYPES:
            raise GraphValidationError(f"too many distinct edge types (max {MAX_EDGE_TYPES})")

        distinct_props: set[str] = set()
        node_props_hashed: list[frozenset[str]] = []
        for s in node_props_raw:
            hashed = frozenset(_stable_property_id(p) for p in s)
            node_props_hashed.append(hashed)
            distinct_props.update(hashed)
        if len(distinct_props) > MAX_DISTINCT_PROPERTIES:
            raise GraphValidationError(f"too many distinct properties (max {MAX_DISTINCT_PROPERTIES})")

        return cls(
            node_count=node_count,
            edges=tuple(normalized_edges),
            node_properties_raw=tuple(node_props_raw),
            node_properties=tuple(node_props_hashed),
            edge_types=frozenset(edge_type_set),
            schema=GRAPH_SCHEMA,
        )

    @classmethod
    def from_chain(cls, properties: list[object], *, edge_type: object = 0) -> "GraphInput":
        """Helper: linear chain graph for sequences (left/right edges)."""
        n = len(properties)
        edges = [(i, i + 1, edge_type) for i in range(n - 1)]
        # also add reverse edges with distinct type for bidirection
        if n > 1 and edge_type == 0:
            # use 1 for reverse to mimic paper's l/r
            rev = [(i + 1, i, 1) for i in range(n - 1)]
            edges = edges + rev  # type: ignore[assignment]
        node_props = {i: [properties[i]] for i in range(n)}  # type: ignore[dict-item]
        return cls.create(node_count=n, edges=edges, node_properties=node_props)  # type: ignore[arg-type]

    @classmethod
    def from_grid(
        cls, rows: int, cols: int, cell_properties: list[list[object]] | None = None
    ) -> "GraphInput":
        """Helper: grid graph for images (CIFAR-style)."""
        n = rows * cols
        edges: list[tuple[int, int, object]] = []
        # edge types: 0=right,1=left,2=down,3=up
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if c + 1 < cols:
                    edges.append((idx, idx + 1, 0))
                    edges.append((idx + 1, idx, 1))
                if r + 1 < rows:
                    edges.append((idx, idx + cols, 2))
                    edges.append((idx + cols, idx, 3))
        node_props: dict[int, list[object]] = {}
        if cell_properties:
            for r in range(rows):
                for c in range(cols):
                    idx = r * cols + c
                    prop = cell_properties[r][c]
                    node_props[idx] = [prop] if not isinstance(prop, (list, tuple, set)) else list(prop)  # type: ignore[arg-type]
        return cls.create(node_count=n, edges=edges, node_properties=node_props)

    def digest(self) -> str:
        payload = {
            "schema": self.schema,
            "node_count": self.node_count,
            "edges": sorted(self.edges),
            "node_properties": [sorted(p) for p in self.node_properties],
            "edge_types": sorted(str(t) for t in self.edge_types),
        }
        enc = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(enc).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "node_count": self.node_count,
            "edges": [list(e) for e in self.edges],
            "node_properties": [sorted(p) for p in self.node_properties_raw],
            "edge_types": sorted(self.edge_types, key=lambda x: str(x)),
        }


@dataclass(frozen=True, slots=True)
class GraphDataset:
    graphs: tuple[GraphInput, ...]
    labels: tuple[int, ...]

    @classmethod
    def create(cls, graphs: list[GraphInput], labels: list[int]) -> "GraphDataset":
        if len(graphs) != len(labels):
            raise GraphValidationError("graphs and labels must be equal length")
        if not graphs:
            raise GraphValidationError("dataset cannot be empty")
        if len(graphs) > 100000:
            raise GraphValidationError("dataset too large")
        for l in labels:
            if type(l) is not int or l not in (0, 1):
                raise GraphValidationError("labels must be 0/1 ints")
        return cls(tuple(graphs), tuple(labels))
