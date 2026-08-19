"""Graph connectors — bridge raw records / grids / sequences to GraphInput."""

from __future__ import annotations

from typing import Iterable, Mapping

from .types import GraphInput, GraphValidationError


def sequence_to_graph(seq: str | list[object], *, edge_type_left: object = "left", edge_type_right: object = "right") -> GraphInput:
    props = list(seq) if isinstance(seq, str) else list(seq)
    n = len(props)
    edges: list[tuple[int, int, object]] = []
    for i in range(n - 1):
        edges.append((i, i + 1, edge_type_right))
        edges.append((i + 1, i, edge_type_left))
    node_props = {i: [props[i]] for i in range(n)}
    return GraphInput.create(node_count=n, edges=edges, node_properties=node_props)


def grid_to_graph(rows: int, cols: int, cell_values: list[list[object]] | None = None) -> GraphInput:
    if cell_values is not None and (len(cell_values) != rows or any(len(r) != cols for r in cell_values)):
        raise GraphValidationError("cell_values shape mismatch")
    return GraphInput.from_grid(rows, cols, cell_properties=cell_values)


class GraphConnector:
    """Iterate over records containing graph-structured fields."""

    def __init__(self, node_field: str = "nodes", edge_field: str = "edges", prop_field: str = "props") -> None:
        self.node_field = node_field
        self.edge_field = edge_field
        self.prop_field = prop_field

    def adapt(self, record: Mapping[str, object]) -> GraphInput:
        # record expected to have {"nodes": [...], "edges": [[src,dst,type],...], "props": {node: [props]}}
        # For simplicity, if record is already a GraphInput dict, convert
        try:
            nodes = record.get(self.node_field)  # type: ignore[assignment]
            edges = record.get(self.edge_field, [])  # type: ignore[assignment]
            props = record.get(self.prop_field, {})  # type: ignore[assignment]
            if isinstance(nodes, int):
                n = int(nodes)
                return GraphInput.create(node_count=n, edges=edges, node_properties=props)  # type: ignore[arg-type]
            if isinstance(nodes, list):
                n = len(nodes)
                # nodes may be list of property lists
                node_props = {i: nodes[i] for i in range(n)} if nodes and isinstance(nodes[0], (list, tuple, set)) else {}
                return GraphInput.create(node_count=n, edges=edges, node_properties=node_props)  # type: ignore[arg-type]
        except Exception as e:
            raise GraphValidationError(f"cannot adapt record to graph: {e}") from e
        raise GraphValidationError("record lacks graph fields")

    def iter_adapt(self, records: Iterable[Mapping[str, object]]) -> Iterable[GraphInput]:
        for r in records:
            yield self.adapt(r)
