"""Graph Tsetlin Machine — structural prototype for layered message passing + CoTM voting.

**Status: experimental structural prototype, not yet a full Graph Tsetlin Machine.**

This module provides a deterministic, bounded Python reference that mirrors the
paper's Algorithm §2 (steps 1–5) *structure* (per-node per-clause per-layer
evaluation, message inboxes, CoTM voting), but does **not** yet implement the
full Tsetlin-automata feedback, hypervector bundle/⊗ participation in
inference, or the shared CoTM clause pool dynamics. HypervectorEncoder is
constructed (hv_dim, bundle/bind) but _evaluate_graph() currently operates on
Python sets of hashed property strings and strings such as "M:0:3⊗edge" for
clarity and testability. fit() is a heuristic weight + random literal
insertion scaffold (see its docstring) — not TA/CoTM learning.

Use as: deterministic oracle for exact message-routing, edge-type, and
negated-literal semantics; serialize→reload exact-prediction checks; and for
native load/inspect of graph_tm_v1 artifacts (native run/verify currently
returns UNSUPPORTED_MODEL). The eventual full Graph TM will integrate TA
states, sparse binary hypervectors in inference, and coalesced voting.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

from .types import GraphInput, GraphValidationError
from .deep_clause import DeepClause, DeepClauseComponent
from .hypervector import bundle, bind
from .encoding import HypervectorEncoder

# For now, DeepClauseComponent evaluation is set-based (present symbols).
# GraphTM evaluation mirrors paper: per-node per-clause per-layer.

MAX_DEPTH = 8
MAX_CLAUSES = 1024


@dataclass
class GraphTsetlinMachine:
    """Experimental structural prototype — see module docstring for status."""

    depth: int
    clauses: int
    specificity: float = 3.9
    threshold: int = 15
    hv_dim: int = 4096
    edge_type_count: int = 4
    seed: int = 1
    # internal TA states per component literal (simplified: prototype scaffold)
    # Full TM would store per-literal TA states and CoTM shared pool; here we
    # store literal sets and per-class weights heuristically.
    _components: list[DeepClause] = field(default_factory=list, init=False, repr=False)
    _weights: list[list[int]] = field(default_factory=list, init=False, repr=False)
    _rng: random.Random = field(default_factory=lambda: random.Random(1), init=False, repr=False)

    def __post_init__(self) -> None:
        if not 1 <= self.depth <= MAX_DEPTH:
            raise ValueError("depth must be 1..8")
        if not 1 <= self.clauses <= MAX_CLAUSES:
            raise ValueError("clauses must be 1..1024")
        if self.hv_dim % 64 != 0:
            raise ValueError("hv_dim must be multiple of 64")
        self._rng = random.Random(self.seed)
        self._encoder = HypervectorEncoder(dim=self.hv_dim, sparsity=0.01)
        # initialise random deep clauses: each component is empty (matches everything) initially
        self._components = []
        for j in range(self.clauses):
            comps = []
            for d in range(self.depth):
                # start with empty conjunction (true)
                comps.append(DeepClauseComponent(layer=d, literals=frozenset(), negated=frozenset()))
            self._components.append(DeepClause(tuple(comps)))
        self._weights = [[1, 1] for _ in range(self.clauses)]
        # TA states: per clause per layer per symbol -> include flag (simplified)
        # We store literal pools as sets; learning will add/remove literals heuristically

    # ---- evaluation ----

    def _node_properties_set(self, g: GraphInput, node: int) -> frozenset[str]:
        return g.node_properties[node]

    def _evaluate_graph(self, g: GraphInput) -> list[bool]:
        """Return per-clause truth for graph G (OR over nodes)."""
        n = g.node_count
        # build adjacency list
        adj: list[list[tuple[int, object]]] = [[] for _ in range(n)]
        for src, dst, etype in g.edges:
            adj[src].append((dst, etype))

        # inboxes per node per layer: list of sets of message symbols present
        # inbox[d][q] = set of bound message strings received at node q before layer d evaluation
        inboxes: list[list[set[str]]] = [[set() for _ in range(n)] for _ in range(self.depth)]

        # layer 0 evaluates property sets; subsequent layers evaluate inbox sets
        # we also need to simulate message submission per layer
        # For simplicity, deep clause component literals are string ids for properties or messages.
        # We treat inbox symbols as "M:d:clause" bound with edge_type as "M:d:clause⊗t"
        # For evaluation, we check if required literals ⊆ present set.

        # First compute per-node per-clause per-layer truth
        # layer_truth[d][j][q] bool
        layer_truth: list[list[list[bool]]] = [
            [[False for _ in range(n)] for _ in range(self.clauses)] for _ in range(self.depth)
        ]

        for d in range(self.depth):
            for j, clause in enumerate(self._components):
                comp = clause.components[d]
                for q in range(n):
                    if d == 0:
                        present = self._node_properties_set(g, q)
                    else:
                        # inbox before this layer: all messages from previous layer received at q
                        # present is set of bound message strings that would be in inbox
                        # For simplicity we treat present as the raw message ids (without bind) plus edge info
                        # Our component literals are expected to be message ids like "M:0:3"
                        present = frozenset(inboxes[d][q])  # type: ignore[assignment]
                    layer_truth[d][j][q] = comp.evaluate(present)  # type: ignore[arg-type]

            # submit messages for this layer to next layer's inboxes
            if d + 1 < self.depth:
                for j in range(self.clauses):
                    for q in range(n):
                        if layer_truth[d][j][q]:
                            # send M^d_j bound with edge type to each neighbor
                            for dst, etype in adj[q]:
                                # bound message string representation
                                msg = f"M:{d}:{j}⊗{etype}"
                                # also plain message for matching without edge type
                                inboxes[d + 1][dst].add(f"M:{d}:{j}")
                                inboxes[d + 1][dst].add(msg)

        # full clause truth per node: ∧ over depth
        clause_per_node: list[list[bool]] = [[True for _ in range(n)] for _ in range(self.clauses)]
        for j in range(self.clauses):
            for q in range(n):
                truth = True
                for d in range(self.depth):
                    truth = truth and layer_truth[d][j][q]
                clause_per_node[j][q] = truth

        # graph-level clause output: OR over nodes (exists node where full clause true)
        graph_clause: list[bool] = [any(clause_per_node[j][q] for q in range(n)) for j in range(self.clauses)]
        return graph_clause

    def predict(self, g: GraphInput) -> int:
        per_clause = self._evaluate_graph(g)
        scores = [0, 0]
        for j, out in enumerate(per_clause):
            if out:
                # CoTM weighted vote: even clauses traditionally vote for class 0, odd for 1
                # but we use stored weights; init 1,1 so both classes get 1 per true clause
                # To make binary, we bias: j%2==0 → class0, odd→class1
                if j % 2 == 0:
                    scores[0] += self._weights[j][0]
                else:
                    scores[1] += self._weights[j][1]
        # threshold clamp and voting margin similar to TM
        # simplified: argmax
        return 0 if scores[0] >= scores[1] else 1

    def fit(self, graphs: list[GraphInput], labels: list[int], *, epochs: int = 1) -> "GraphTsetlinMachine":
        """Heuristic scaffold — not TA/CoTM learning.

        Updates per-class weights and occasionally inserts a randomly selected
        property literal. Exists only to provide a deterministic training path
        for serialization and smoke tests until the true TA feedback is
        implemented. Do not use to claim learning accuracy.
        """
        if len(graphs) != len(labels):
            raise ValueError("graphs/labels length mismatch")
        for _ in range(epochs):
            for g, y in zip(graphs, labels):
                pred = self.predict(g)
                # Heuristic scaffold: not full TA feedback
                if pred != y:
                    for j in range(self.clauses):
                        self._weights[j][y] += 1
                        if g.node_count > 0 and self._rng.random() < 0.1:
                            node = self._rng.randrange(g.node_count)
                            props = list(g.node_properties[node])
                            if props:
                                prop = self._rng.choice(props)
                                comp = self._components[j].components[0]
                                new_lits = set(comp.literals)
                                new_lits.add(prop)
                                if len(new_lits) < 8:
                                    new_comp = DeepClauseComponent(layer=0, literals=frozenset(new_lits), negated=comp.negated)
                                    comps = list(self._components[j].components)
                                    comps[0] = new_comp
                                    self._components[j] = DeepClause(tuple(comps))
                else:
                    pass
        return self

    def get_clause(self, idx: int) -> DeepClause:
        return self._components[idx]

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "clauses": self.clauses,
            "hv_dim": self.hv_dim,
            "components": [c.to_dict() for c in self._components],
            "weights": self._weights,
        }
