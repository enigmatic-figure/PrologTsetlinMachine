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
MAX_GRAPH_CLAUSES = 1024
MAX_GRAPH_DEPTH = 8

GRAPH_SCHEMA = "ptm.graph.v1"


class GraphValidationError(ValueError):
    """Raised when a supplied graph violates its bounded contract."""


def _typed_canonical_key(value: object) -> tuple[str, str]:
    """Canonical (type, repr) for typed equality, ordering, and dedup."""
    if isinstance(value, str):
        if not value or len(value) > MAX_PROPERTY_STRING_CHARS:
            raise GraphValidationError("property string must be 1..256 chars")
        if any(ord(c) < 0x20 for c in value):
            raise GraphValidationError("property contains control character")
        return ("str", value)
    if isinstance(value, bool):
        return ("bool", str(value))
    if isinstance(value, int):
        if not -(1 << 53) <= value <= (1 << 53):
            raise GraphValidationError("integer property out of 53-bit range")
        return ("int", str(value))
    if isinstance(value, float):
        if not (value == value and abs(value) != float("inf")):
            raise GraphValidationError("property float must be finite")
        return ("float", repr(value))
    raise GraphValidationError("property must be str, int, bool, or finite float")


def _stable_property_id(value: object) -> str:
    type_name, repr_str = _typed_canonical_key(value)
    canonical = f"{type_name}:{repr_str}"
    return "prop:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _canonical_property_string(value: object) -> str:
    type_name, repr_str = _typed_canonical_key(value)
    return f"{type_name}:{repr_str}"


def canonical_graph_scalar(value: object) -> str:
    """One canonical identity for graph scalars: int:1 ≠ str:1 ≠ bool:True ≠ float:1.0."""
    type_name, repr_str = _typed_canonical_key(value)
    return f"{type_name}:{repr_str}"


def canonical_edge_type(value: object) -> str:
    """Typed canonical edge type — int:1 versus str:1 are distinct."""
    # Edge types are restricted to int 0..15 or short str; reuse typed key for unambiguity
    if isinstance(value, bool):
        raise GraphValidationError("edge_type bool not allowed")
    if isinstance(value, int):
        if not 0 <= value < MAX_EDGE_TYPES:
            raise GraphValidationError("int edge_type must be 0..15")
        return f"int:{value}"
    if isinstance(value, str):
        if not value or len(value) > 64 or any(ord(c) < 0x20 for c in value):
            raise GraphValidationError("str edge_type must be 1..64 printable chars")
        return f"str:{value}"
    raise GraphValidationError("edge_type must be int 0..15 or short str")


def _decode_canonical_string(s: str) -> object:
    if ":" not in s:
        return s
    type_name, repr_str = s.split(":", 1)
    if type_name == "str":
        return repr_str
    if type_name == "int":
        return int(repr_str)
    if type_name == "bool":
        return repr_str == "True"
    if type_name == "float":
        return float(repr_str)
    return s


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
    node_properties_raw: tuple[frozenset[str], ...]  # typed canonical strings "type:repr"
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
        raw_props: dict[int, frozenset[str]] = {}
        if node_properties:
            for node, props in node_properties.items():
                if not isinstance(node, int) or isinstance(node, bool):
                    raise GraphValidationError("node index must be int")
                if not 0 <= node < node_count:
                    raise GraphValidationError("node property for unknown node")
                # canonicalize before dedup to keep typed distinction (1 vs True vs 1.0)
                dedup: dict[tuple[str, str], str] = {}
                for p in (props or []):
                    key = _typed_canonical_key(p)
                    if key not in dedup:
                        dedup[key] = _canonical_property_string(p)
                prop_set = frozenset(dedup.values())
                if len(prop_set) > 256:
                    raise GraphValidationError("too many properties on one node")
                raw_props[node] = prop_set
        # fill missing nodes with empty set
        node_props_raw: list[frozenset[str]] = []
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
            # s already canonical strings like "int:1"; hash directly via same canonical
            hashed = frozenset("prop:" + hashlib.sha256(p.encode("utf-8")).hexdigest()[:16] for p in s)
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
        # Validate dimensions before any allocation — prevents billion×billion DoS
        if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
            raise GraphValidationError("rows must be positive int")
        if not isinstance(cols, int) or isinstance(cols, bool) or cols <= 0:
            raise GraphValidationError("cols must be positive int")
        if rows > MAX_NODES or cols > MAX_NODES:
            raise GraphValidationError(f"rows/cols must be 1..{MAX_NODES}")
        n = rows * cols
        if n < 1 or n > MAX_NODES:
            raise GraphValidationError(f"rows*cols must be 1..{MAX_NODES}")
        # Pre-compute edge count without loops: horizontal 2*rows*(cols-1) + vertical 2*(rows-1)*cols
        horiz = 2 * rows * (cols - 1) if cols > 1 else 0
        vert = 2 * (rows - 1) * cols if rows > 1 else 0
        edge_count = horiz + vert
        if edge_count > MAX_EDGES:
            raise GraphValidationError(f"grid edge count {edge_count} exceeds {MAX_EDGES}")
        if cell_properties is not None:
            if len(cell_properties) != rows or any(len(r) != cols for r in cell_properties):
                raise GraphValidationError("cell_properties shape mismatch")
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

    def _edge_sort_key(self, e: tuple[int, int, object]) -> tuple[int, int, str]:
        return (e[0], e[1], canonical_edge_type(e[2]))

    def digest(self) -> str:
        payload = {
            "schema": self.schema,
            "node_count": self.node_count,
            "edges": sorted(self.edges, key=self._edge_sort_key),
            "node_properties": [sorted(p) for p in self.node_properties],
            "edge_types": sorted(self.edge_types, key=lambda x: canonical_edge_type(x)),
        }
        enc = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(enc).hexdigest()

    def to_dict(self) -> dict[str, object]:
        # Decode canonical raw for external representation, sorted via typed key
        def _decode_sorted(s: frozenset[str]) -> list[object]:
            decoded = [_decode_canonical_string(x) for x in s]
            return sorted(decoded, key=lambda v: (type(v).__name__, str(v)))

        return {
            "schema": self.schema,
            "node_count": self.node_count,
            "edges": [list(e) for e in self.edges],
            "node_properties": [_decode_sorted(p) for p in self.node_properties_raw],
            "edge_types": sorted(self.edge_types, key=lambda x: canonical_edge_type(x)),
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
