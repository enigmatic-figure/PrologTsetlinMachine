from .proposal import (
    NATIVE_TARGETS,
    NativeTarget,
    PTAEscalationProposal,
    PTAInsight,
    PTAMorphologyProposal,
)
from .executable import ExecutableBinaryClause
from .lowering import (
    LoweredCandidate,
    NotRepresentable,
    check_example,
    lower_exact,
    lowerable,
    syntactically_bounded,
)
from .ontology import PROLOG_ONTOLOGY
from .input import InputPTA, LiteralProposal
from .deescalation import DeescalationPTA
from .escalation import EscalationPTA
from .sparse import SparseClauseBank, lower_to_sparse, to_sparse_exact, propose_sparse_morphology
from .sampling import fns_mask_from_confusable, fns_mask_from_counterexamples, multigranularity_schedule
from .regression import find_residual_regions, residual_to_proposal
from .convolutional import invent_spatial_templates, template_to_proposal as spatial_to_proposal
from .graph_pta import hypothesize_graph_relation, hypothesis_to_proposal as graph_hypothesis_to_proposal
from .sequence import discover_sequence_patterns, pattern_to_proposal as seq_to_proposal
from .composite import discover_specialist_gates, gate_to_proposal, smallest_specialist_subset
from .session import PTAReasoningSession

__all__ = [
    "PTAEscalationProposal",
    "PTAInsight",
    "PTAMorphologyProposal",
    "NativeTarget",
    "NATIVE_TARGETS",
    "ExecutableBinaryClause",
    "LoweredCandidate",
    "NotRepresentable",
    "lowerable",
    "lower_exact",
    "syntactically_bounded",
    "check_example",
    "PROLOG_ONTOLOGY",
    "InputPTA",
    "LiteralProposal",
    "DeescalationPTA",
    "EscalationPTA",
    "SparseClauseBank",
    "lower_to_sparse",
    "to_sparse_exact",
    "propose_sparse_morphology",
    "fns_mask_from_confusable",
    "fns_mask_from_counterexamples",
    "multigranularity_schedule",
    "find_residual_regions",
    "residual_to_proposal",
    "invent_spatial_templates",
    "spatial_to_proposal",
    "hypothesize_graph_relation",
    "graph_hypothesis_to_proposal",
    "discover_sequence_patterns",
    "seq_to_proposal",
    "discover_specialist_gates",
    "gate_to_proposal",
    "smallest_specialist_subset",
    "PTAReasoningSession",
]
