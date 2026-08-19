"""Graph Tsetlin Machine — optional add-on (zero required deps).

This package is additive: it does not change existing
packed_tm / logic / PA / preprocessing contracts.
Import it explicitly: ``from prolog_tsetlin.graph import GraphInput``.
"""

from .types import GraphInput, GraphDataset, GraphValidationError
from .hypervector import Hypervector, HypervectorSpace, bundle, bind
from .encoding import HypervectorEncoder
from .deep_clause import DeepClause, DeepClauseComponent
from .graph_tm import GraphTsetlinMachine

__all__ = [
    "GraphInput",
    "GraphDataset",
    "GraphValidationError",
    "Hypervector",
    "HypervectorSpace",
    "bundle",
    "bind",
    "HypervectorEncoder",
    "DeepClause",
    "DeepClauseComponent",
    "GraphTsetlinMachine",
]
