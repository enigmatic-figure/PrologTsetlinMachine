"""GraphTM example: train on chain motif and export graph_tm_v1 artifact.

This mirrors the paper's G=(V,E,P,T) with deep clauses Cj=∧ C^i_j.
Graphs are classified by whether they contain a length-2 chain a→b (positive)
vs random. Demonstrates HypervectorEncoder + DeepClause layered inbox.
"""

from prolog_tsetlin.graph.connectors import grid_to_graph, sequence_to_graph
from prolog_tsetlin.graph.types import GraphInput
from prolog_tsetlin.graph.graph_tm import GraphTsetlinMachine
from prolog_tsetlin.model_artifact import export_graph_tm, load_model_artifact_from_bytes

# Build toy graphs: chain of property 'a' -> 'b' is positive.
def make_graph(has_chain: bool) -> GraphInput:
    if has_chain:
        # 3 nodes: a -> b -> c  (chain present)
        return GraphInput.create(
            node_count=3,
            edges=[(0, 1, 0), (1, 2, 0)],
            node_properties={0: ["a"], 1: ["b"], 2: ["c"]},
        )
    else:
        # 3 nodes no edge or disconnected
        return GraphInput.create(
            node_count=3,
            edges=[(0, 1, 0)],
            node_properties={0: ["a"], 1: ["a"], 2: ["a"]},
        )


graphs = [make_graph(True), make_graph(False), make_graph(True), make_graph(False)]
labels = [1, 0, 1, 0]

gtm = GraphTsetlinMachine(depth=2, clauses=8, hv_dim=512, seed=1)
gtm.fit(graphs, labels, epochs=3)

for g, y in zip(graphs, labels):
    print(f"label {y} -> pred {gtm.predict(g)}")

artifact = export_graph_tm(gtm, graphs, name="graph-chain-demo", description="Toy chain motif")
print(f"exported {artifact.artifact_id} depth={artifact.graph_depth} clauses={artifact.graph_clauses} hv_dim={artifact.graph_hv_dim}")

# Round-trip verify via Python and native ptmrt (if available)
rt = load_model_artifact_from_bytes(artifact.serialized)
print(f"round-trip verify_conformance={rt.verify_conformance()} pred={rt.predict(graphs[0])}")

# Also show connector on sequence
seq_graph = sequence_to_graph(["a", "b", "a"])
print(f"sequence graph nodes={seq_graph.node_count} edges={len(seq_graph.edges)} pred={gtm.predict(seq_graph)}")

# Grid connector
grid_g = grid_to_graph(2, 2, cell_values=[[1, 0], [0, 1]])
print(f"grid graph nodes={grid_g.node_count} edges={len(grid_g.edges)}")
