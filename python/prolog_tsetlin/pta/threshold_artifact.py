"""Reviewed promotion of an Input-PTA threshold into a native artifact.

The threshold proposal itself intentionally remains ``NotRepresentable``.  This
module implements the narrower, explicit transition from a validated symbolic
threshold to a materialized numeric literal, an exact binary-clause semantic
candidate, and finally a packed-TM inference artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from ..model_artifact import PackedTMInferenceArtifact, export_packed_tm
from ..preprocessing import PreprocessingContract
from ..reference import SNAPSHOT_SCHEMA_VERSION, TMSnapshot
from ..representation import FieldKind, LiteralCatalog, LiteralDescriptor
from .executable import ExecutableBinaryClause
from .lowering import LoweredCandidate, lower_exact
from .proposal import PTAEscalationProposal
from .session import EXACT_NUMERIC_MAGNITUDE, PTAReasoningSession


_MATERIALIZATION_VERSION = "pta.threshold-materialization.v1"
_ARTIFACT_COMPILER_VERSION = "pta.threshold-artifact.v1"
_STATES_PER_ACTION = 100


class PTAThresholdPromotionError(ValueError):
    """A threshold proposal failed review, materialization, or compilation."""


@dataclass(frozen=True, slots=True)
class ThresholdBoundaryEvidence:
    """Observed adjacent label flip that independently justifies a threshold."""

    dataset_id: str
    field: str
    threshold: int | float
    lower_value: int | float
    lower_label: int
    upper_value: int | float
    upper_label: int
    observations_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "field": self.field,
            "lower_label": self.lower_label,
            "lower_value": self.lower_value,
            "observations_digest": self.observations_digest,
            "threshold": self.threshold,
            "upper_label": self.upper_label,
            "upper_value": self.upper_value,
        }


@dataclass(frozen=True, slots=True)
class ReviewedThresholdProposal:
    """Non-mutating review result for one exact Input-PTA threshold product."""

    proposal: PTAEscalationProposal
    descriptor: LiteralDescriptor
    evidence: ThresholdBoundaryEvidence


@dataclass(frozen=True, slots=True)
class MaterializedThresholdClause:
    """A threshold literal admitted to a catalog and the exact clause it forms."""

    reviewed: ReviewedThresholdProposal
    descriptor: LiteralDescriptor
    proposal: PTAEscalationProposal
    lowered: LoweredCandidate


@dataclass(frozen=True, slots=True)
class ThresholdArtifactCompilation:
    """Portable packed-TM compilation of a materialized one-literal clause."""

    materialized: MaterializedThresholdClause
    snapshot: TMSnapshot
    artifact: PackedTMInferenceArtifact


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _threshold_structure(
    proposal: PTAEscalationProposal,
) -> tuple[str, int | float]:
    if not isinstance(proposal, PTAEscalationProposal):
        raise TypeError("proposal must be PTAEscalationProposal")
    if proposal.native_target != "threshold":
        raise PTAThresholdPromotionError("proposal target must be threshold")
    structure = dict(proposal.structure)
    if set(structure) != {"field", "operator", "threshold"}:
        raise PTAThresholdPromotionError(
            "threshold proposal structure must contain only field, operator, and threshold"
        )
    field = structure["field"]
    operator = structure["operator"]
    threshold = structure["threshold"]
    if type(field) is not str or not field:
        raise PTAThresholdPromotionError("threshold field must be a nonempty string")
    if operator != "ge":
        raise PTAThresholdPromotionError("only numeric_ge threshold proposals are supported")
    if (
        type(threshold) not in (int, float)
        or (type(threshold) is float and not math.isfinite(threshold))
        or abs(threshold) > EXACT_NUMERIC_MAGNITUDE
    ):
        raise PTAThresholdPromotionError("threshold must be a finite strict number")
    if proposal.weights is not None or proposal.output_assignments is not None:
        raise PTAThresholdPromotionError("threshold proposal cannot carry weights")
    if proposal.resource_bounds.get("literal_count") != 1:
        raise PTAThresholdPromotionError("threshold proposal must declare one literal")

    threshold_text = format(threshold, ".17g")
    expected_required = (f"numeric_ge:{field}:{threshold_text}",)
    if proposal.required_literals != expected_required:
        raise PTAThresholdPromotionError(
            "threshold proposal required literal does not match its declared structure"
        )
    if not any(
        insight.source_pta == "pta:input"
        and insight.kind == "threshold"
        and insight.subject == field
        and len(insight.evidence) == 1
        and type(insight.evidence[0]) is type(threshold)
        and insight.evidence[0] == threshold
        for insight in proposal.supporting_insights
    ):
        raise PTAThresholdPromotionError(
            "threshold proposal lacks matching Input-PTA threshold evidence"
        )
    return field, threshold


def _boundary_evidence(
    session: PTAReasoningSession, field: str, threshold: int | float
) -> ThresholdBoundaryEvidence:
    session.validate()
    labels: dict[int, int] = {}
    for example, label in session.example_labels:
        existing = labels.get(example)
        if existing is not None and existing != label:
            raise PTAThresholdPromotionError(
                f"example {example} has conflicting labels"
            )
        labels[example] = label

    observations: dict[int, int | float] = {}
    for _, example, observed_field, value in session.observations:
        if observed_field != field:
            continue
        if (
            type(value) not in (int, float)
            or (type(value) is float and not math.isfinite(value))
            or abs(value) > EXACT_NUMERIC_MAGNITUDE
        ):
            raise PTAThresholdPromotionError(
                f"field {field!r} contains a non-finite or nonnumeric observation"
            )
        existing = observations.get(example)
        if existing is not None and existing != value:
            raise PTAThresholdPromotionError(
                f"example {example} has conflicting observations for {field!r}"
            )
        observations[example] = value

    labeled = sorted(
        (value, labels[example], example)
        for example, value in observations.items()
        if example in labels
    )
    if not labeled:
        raise PTAThresholdPromotionError(
            "threshold review requires labeled observations for its field"
        )
    states: dict[int | float, set[int]] = {}
    for value, label, _ in labeled:
        states.setdefault(value, set()).add(label)
    ordered_states = sorted(states.items())

    boundary: tuple[int | float, int, int | float, int] | None = None
    for (lower_value, lower_labels), (upper_value, upper_labels) in zip(
        ordered_states, ordered_states[1:]
    ):
        if len(lower_labels) != 1 or len(upper_labels) != 1:
            continue
        lower_label = next(iter(lower_labels))
        upper_label = next(iter(upper_labels))
        if (
            lower_label != upper_label
            and lower_value < threshold < upper_value
        ):
            boundary = (lower_value, lower_label, upper_value, upper_label)
            break
    if boundary is None:
        raise PTAThresholdPromotionError(
            "threshold does not strictly separate an observed adjacent label flip"
        )

    digest_payload = [
        {"example": example, "label": label, "value": value}
        for value, label, example in labeled
    ]
    lower_value, lower_label, upper_value, upper_label = boundary
    return ThresholdBoundaryEvidence(
        dataset_id=session.dataset_id,
        field=field,
        threshold=threshold,
        lower_value=lower_value,
        lower_label=lower_label,
        upper_value=upper_value,
        upper_label=upper_label,
        observations_digest=_canonical_digest(digest_payload),
    )


def review_threshold_proposal(
    proposal: PTAEscalationProposal,
    *,
    session: PTAReasoningSession,
    catalog: LiteralCatalog,
) -> ReviewedThresholdProposal:
    """Review a threshold without mutating the destination literal catalog."""

    if not isinstance(session, PTAReasoningSession):
        raise TypeError("session must be PTAReasoningSession")
    if not isinstance(catalog, LiteralCatalog):
        raise TypeError("catalog must be LiteralCatalog")
    field, threshold = _threshold_structure(proposal)
    try:
        field_definition = catalog.schema.field(field)
    except KeyError as error:
        raise PTAThresholdPromotionError(
            "threshold cannot be represented by the destination catalog"
        ) from error
    if field_definition.kind is not FieldKind.NUMBER:
        raise PTAThresholdPromotionError(
            f"threshold field {field!r} is not numeric"
        )
    try:
        descriptor = catalog.preview_numeric_ge(field, threshold)
    except (TypeError, ValueError) as error:
        raise PTAThresholdPromotionError(
            "threshold cannot be represented by the destination catalog"
        ) from error
    evidence = _boundary_evidence(session, field, threshold)
    return ReviewedThresholdProposal(proposal, descriptor, evidence)


def _derived_clause_proposal(
    reviewed: ReviewedThresholdProposal,
    descriptor: LiteralDescriptor,
) -> PTAEscalationProposal:
    origin_semantic_id = reviewed.proposal.semantic_id()
    origin_provenance_id = reviewed.proposal.provenance_id()
    return PTAEscalationProposal(
        proposal_id=(
            "pta:materialized-threshold:"
            + origin_provenance_id.removeprefix("sha256:")
        ),
        source_pta_ids=reviewed.proposal.source_pta_ids,
        supporting_insights=reviewed.proposal.supporting_insights,
        counterexamples_addressed=reviewed.proposal.counterexamples_addressed,
        required_literals=(descriptor,),
        native_target="binary_clause",
        structure={"clause": (descriptor.literal_id,)},
        resource_bounds={"clause_count": 1, "literal_count": 1},
        validation_signature={
            "boundary_evidence": reviewed.evidence.to_dict(),
            "materialization_version": _MATERIALIZATION_VERSION,
            "origin_proposal_provenance_id": origin_provenance_id,
            "origin_proposal_semantic_id": origin_semantic_id,
        },
        support_trace=reviewed.proposal.support_trace
        + (
            "independently validated against an adjacent labeled boundary",
            f"materialized canonical literal {descriptor.literal_id}",
        ),
    )


def materialize_threshold_clause(
    reviewed: ReviewedThresholdProposal,
    *,
    session: PTAReasoningSession,
    catalog: LiteralCatalog,
) -> MaterializedThresholdClause:
    """Approve a reviewed threshold and cross the exact binary-clause gate."""

    if not isinstance(reviewed, ReviewedThresholdProposal):
        raise TypeError("reviewed must be ReviewedThresholdProposal")
    if not isinstance(catalog, LiteralCatalog):
        raise TypeError("catalog must be LiteralCatalog")
    expected_review = review_threshold_proposal(
        reviewed.proposal, session=session, catalog=catalog
    )
    if expected_review != reviewed:
        raise PTAThresholdPromotionError(
            "reviewed threshold no longer matches its reasoning session"
        )
    field, threshold = _threshold_structure(reviewed.proposal)
    try:
        preview = catalog.preview_numeric_ge(field, threshold)
    except (KeyError, TypeError, ValueError) as error:
        raise PTAThresholdPromotionError(
            "reviewed threshold no longer matches the destination catalog"
        ) from error
    if preview != reviewed.descriptor:
        raise PTAThresholdPromotionError(
            "reviewed threshold descriptor differs from the destination catalog"
        )

    derived = _derived_clause_proposal(reviewed, preview)

    descriptor = catalog.numeric_ge(field, threshold)
    if descriptor != preview:
        raise PTAThresholdPromotionError(
            "catalog materialization changed the reviewed literal identity"
        )
    lowered = lower_exact(derived, catalog=catalog)
    if not isinstance(lowered, LoweredCandidate):
        raise PTAThresholdPromotionError(
            f"materialized threshold clause failed exact lowering: {lowered.reason}"
        )
    if not isinstance(lowered.native_object, ExecutableBinaryClause):
        raise PTAThresholdPromotionError(
            "exact gate returned the wrong executable candidate type"
        )
    return MaterializedThresholdClause(reviewed, descriptor, derived, lowered)


def compile_threshold_artifact(
    materialized: MaterializedThresholdClause,
    *,
    session: PTAReasoningSession,
    catalog: LiteralCatalog,
    name: str = "PTA materialized threshold clause",
) -> ThresholdArtifactCompilation:
    """Compile one exact threshold clause into a portable packed-TM artifact.

    The artifact predicts clause activation, not the source dataset's class
    label.  Its output labels are therefore ``inactive`` and ``active``.
    """

    if not isinstance(materialized, MaterializedThresholdClause):
        raise TypeError("materialized must be MaterializedThresholdClause")
    if not isinstance(catalog, LiteralCatalog):
        raise TypeError("catalog must be LiteralCatalog")
    expected_review = review_threshold_proposal(
        materialized.reviewed.proposal,
        session=session,
        catalog=catalog,
    )
    if expected_review != materialized.reviewed:
        raise PTAThresholdPromotionError(
            "materialized threshold no longer matches its reasoning session"
        )
    expected_proposal = _derived_clause_proposal(
        expected_review, expected_review.descriptor
    )
    if (
        materialized.descriptor != expected_review.descriptor
        or materialized.proposal != expected_proposal
    ):
        raise PTAThresholdPromotionError(
            "materialized threshold differs from the approved exact clause"
        )
    descriptor = materialized.descriptor
    try:
        if catalog.validate_descriptor(descriptor) != descriptor:
            raise ValueError("descriptor mismatch")
    except (KeyError, TypeError, ValueError) as error:
        raise PTAThresholdPromotionError(
            "materialized literal is not canonical for the destination catalog"
        ) from error
    registered = {item.literal_id: item for item in catalog.literals}
    if registered.get(descriptor.literal_id) != descriptor:
        raise PTAThresholdPromotionError(
            "materialized literal is absent from the destination catalog"
        )
    if (
        materialized.lowered.proposal != materialized.proposal
        or materialized.lowered.native_kind != "executable_binary_clause"
        or materialized.lowered.native_object.literal_ids
        != (descriptor.literal_id,)
    ):
        raise PTAThresholdPromotionError(
            "materialized exact candidate differs from the one-literal clause"
        )

    # Compile through a dedicated one-literal preprocessing contract so the
    # artifact is deterministic and unaffected by unrelated catalog entries.
    export_catalog = LiteralCatalog(catalog.schema)
    exported_descriptor = export_catalog.numeric_ge(
        descriptor.source_field, descriptor.parameter("threshold")
    )
    if exported_descriptor != descriptor:
        raise PTAThresholdPromotionError(
            "artifact preprocessing changed the materialized literal identity"
        )
    preprocessing = PreprocessingContract.from_catalog(export_catalog)

    snapshot = TMSnapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        number_of_clauses=1,
        number_of_features=1,
        states_per_action=_STATES_PER_ACTION,
        specificity=3.9,
        threshold=1,
        states=((_STATES_PER_ACTION + 1, _STATES_PER_ACTION),),
        rng_state=None,
    )
    evidence = materialized.reviewed.evidence
    validation_records = (
        {evidence.field: evidence.lower_value},
        {evidence.field: evidence.threshold},
        {evidence.field: evidence.upper_value},
    )
    materialized_rows = preprocessing.materialize_many(validation_records)
    if materialized_rows != ((False,), (True,), (True,)):
        raise PTAThresholdPromotionError(
            "numeric_ge preprocessing disagrees with threshold boundary semantics"
        )
    semantic = materialized.lowered.native_object
    expected = tuple(
        int(semantic.evaluate({descriptor.literal_id: row[0]}))
        for row in materialized_rows
    )
    if expected != (0, 1, 1):
        raise PTAThresholdPromotionError(
            "exact binary clause disagrees with threshold activation semantics"
        )

    origin = materialized.reviewed.proposal
    artifact = export_packed_tm(
        snapshot,
        name=name,
        description=(
            "Exact packed-TM carrier for one independently reviewed "
            "Input-PTA numeric threshold proposal."
        ),
        intended_use="PTA threshold materialization and runtime conformance",
        limitations=(
            "Predicts clause activation only; it is not a trained source-label classifier."
        ),
        feature_names=(
            f"{evidence.field} >= {format(evidence.threshold, '.17g')}",
        ),
        feature_literal_ids=(descriptor.literal_id,),
        feature_catalog_version=_ARTIFACT_COMPILER_VERSION,
        output_labels=("inactive", "active"),
        preprocessing=preprocessing,
        validation_records=validation_records,
        validation_signature={
            "artifact_compiler_version": _ARTIFACT_COMPILER_VERSION,
            "boundary_evidence": evidence.to_dict(),
            "materialized_proposal_provenance_id": materialized.proposal.provenance_id(),
            "materialized_proposal_semantic_id": materialized.proposal.semantic_id(),
            "origin_proposal_provenance_id": origin.provenance_id(),
            "origin_proposal_semantic_id": origin.semantic_id(),
            "semantic_oracle": {
                "assignments": 2,
                "mismatch_count": 0,
                "target": "executable_binary_clause",
            },
        },
    )
    if not artifact.verify_conformance():
        raise PTAThresholdPromotionError(
            "packed threshold artifact failed embedded conformance"
        )
    actual = artifact.predict_records(validation_records)
    if actual != expected:
        raise PTAThresholdPromotionError(
            "packed artifact disagrees with the exact semantic candidate"
        )
    return ThresholdArtifactCompilation(materialized, snapshot, artifact)


__all__ = [
    "MaterializedThresholdClause",
    "PTAThresholdPromotionError",
    "ReviewedThresholdProposal",
    "ThresholdArtifactCompilation",
    "ThresholdBoundaryEvidence",
    "compile_threshold_artifact",
    "materialize_threshold_clause",
    "review_threshold_proposal",
]
