from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from prolog_tsetlin.prolog_resources import PrologResourceError, resolve_gprolog
from prolog_tsetlin.pta import (
    LoweredCandidate,
    NotRepresentable,
    PTACollectiveQuery,
    PTACollectiveService,
    PTAEscalationProposal,
    PTAInsight,
    PTAReasoningSession,
    PTAThresholdPromotionError,
    compile_threshold_artifact,
    lower_exact,
    materialize_threshold_clause,
    review_threshold_proposal,
)
from prolog_tsetlin.representation import FeatureSchema, FieldKind, LiteralCatalog
from prolog_tsetlin.services import publish_packed_inference_artifact


ROOT = Path(__file__).resolve().parents[2]
GOLDEN_HEX = ROOT / "tests" / "data" / "pta_threshold_clause_v1.hex"


def _session(
    *,
    values: tuple[int | float, ...] = (60, 70, 80, 90),
    labels: tuple[int, ...] = (0, 0, 1, 1),
) -> PTAReasoningSession:
    session = PTAReasoningSession("thermostat-threshold-v1")
    for example, value in enumerate(values):
        session.add_observation("pta:input", example, "temperature", value)
    for example, label in enumerate(labels):
        session.add_example_label(example, label)
    return session


def _proposal(
    threshold: int | float = 75.0,
    *,
    operator: str = "ge",
    required_literal: str | None = None,
    insights: tuple[PTAInsight, ...] | None = None,
) -> PTAEscalationProposal:
    try:
        threshold_text = (
            format(threshold, ".17g")
            if type(threshold) in (int, float)
            else str(threshold)
        )
    except OverflowError:
        threshold_text = str(threshold)
    resolved_insights = (
        (PTAInsight("pta:input", "threshold", "temperature", (threshold,)),)
        if insights is None
        else insights
    )
    return PTAEscalationProposal(
        proposal_id=f"pta:escalation:threshold:test:{threshold_text}",
        source_pta_ids=("pta:input", "pta:escalation"),
        supporting_insights=resolved_insights,
        counterexamples_addressed=(),
        required_literals=(
            required_literal
            if required_literal is not None
            else f"numeric_ge:temperature:{threshold_text}",
        ),
        native_target="threshold",
        structure={
            "field": "temperature",
            "operator": operator,
            "threshold": threshold,
        },
        resource_bounds={"literal_count": 1},
        support_trace=("derived by GNU Prolog exception_clause/3",),
    )


def _catalog(kind: FieldKind = FieldKind.NUMBER) -> LiteralCatalog:
    return LiteralCatalog(FeatureSchema.from_fields(temperature=kind))


def _compile():
    session = _session()
    proposal = _proposal()
    catalog = _catalog()
    reviewed = review_threshold_proposal(proposal, session=session, catalog=catalog)
    materialized = materialize_threshold_clause(
        reviewed, session=session, catalog=catalog
    )
    return proposal, catalog, reviewed, materialized, compile_threshold_artifact(
        materialized, session=session, catalog=catalog
    )


def test_review_is_non_mutating_and_threshold_target_remains_fail_closed() -> None:
    proposal = _proposal()
    catalog = _catalog()

    reviewed = review_threshold_proposal(
        proposal, session=_session(), catalog=catalog
    )

    assert catalog.literals == ()
    assert reviewed.descriptor.source_field == "temperature"
    assert reviewed.descriptor.parameter("threshold") == 75.0
    rejected = lower_exact(proposal, catalog=catalog)
    assert isinstance(rejected, NotRepresentable)
    assert "materialized threshold literal" in rejected.reason


def test_approved_threshold_materializes_and_crosses_only_binary_exact_gate() -> None:
    proposal = _proposal()
    catalog = _catalog()
    reviewed = review_threshold_proposal(
        proposal, session=_session(), catalog=catalog
    )

    result = materialize_threshold_clause(
        reviewed, session=_session(), catalog=catalog
    )

    assert catalog.literals == (reviewed.descriptor,)
    assert result.descriptor == reviewed.descriptor
    assert result.proposal.native_target == "binary_clause"
    assert result.proposal.structure["clause"] == (result.descriptor.literal_id,)
    assert result.proposal.required_literals == (result.descriptor,)
    assert isinstance(result.lowered, LoweredCandidate)
    assert result.lowered.native_kind == "executable_binary_clause"
    assert result.lowered.native_object.evaluate(
        {result.descriptor.literal_id: False}
    ) is False
    assert result.lowered.native_object.evaluate(
        {result.descriptor.literal_id: True}
    ) is True
    signature = result.proposal.validation_signature
    assert signature["origin_proposal_semantic_id"] == proposal.semantic_id()
    assert signature["origin_proposal_provenance_id"] == proposal.provenance_id()
    assert signature["boundary_evidence"]["lower_value"] == 70
    assert signature["boundary_evidence"]["upper_value"] == 80


def test_compilation_is_deterministic_and_preserves_full_pta_provenance() -> None:
    proposal, _, reviewed, materialized, compilation = _compile()
    _, _, _, _, repeated = _compile()

    assert compilation.artifact.serialized == repeated.artifact.serialized
    assert compilation.artifact.verify_conformance()
    assert compilation.artifact.number_of_clauses == 1
    assert compilation.artifact.number_of_features == 1
    assert compilation.artifact.positive_include_masks == (1,)
    assert compilation.artifact.negative_include_masks == (0,)
    assert compilation.artifact.predict_records(
        (
            {"temperature": 60},
            {"temperature": 74.999},
            {"temperature": 75},
            {"temperature": 90},
        )
    ) == (0, 0, 1, 1)
    signature = compilation.artifact.manifest["validation"]["signature"]
    assert signature["origin_proposal_semantic_id"] == proposal.semantic_id()
    assert signature["origin_proposal_provenance_id"] == proposal.provenance_id()
    assert (
        signature["materialized_proposal_semantic_id"]
        == materialized.proposal.semantic_id()
    )
    assert (
        signature["materialized_proposal_provenance_id"]
        == materialized.proposal.provenance_id()
    )
    assert signature["boundary_evidence"] == reviewed.evidence.to_dict()
    assert signature["semantic_oracle"] == {
        "assignments": 2,
        "mismatch_count": 0,
        "target": "executable_binary_clause",
    }
    assert compilation.artifact.manifest["task"]["labels"] == [
        "inactive",
        "active",
    ]
    assert "not a trained" in compilation.artifact.manifest["research"][
        "limitations"
    ]


def test_compilation_matches_cross_language_golden_bytes() -> None:
    _, _, _, _, compilation = _compile()
    assert compilation.artifact.serialized == bytes.fromhex(
        GOLDEN_HEX.read_text(encoding="ascii")
    )


def test_compiled_artifact_is_atomically_publishable(tmp_path: Path) -> None:
    _, _, _, _, compilation = _compile()
    destination = tmp_path / "threshold.ptm"

    summary = publish_packed_inference_artifact(
        compilation.artifact, destination
    )

    assert summary.path == destination.resolve()
    assert summary.artifact_id == compilation.artifact.artifact_id
    assert summary.conformance_examples == 3
    assert destination.read_bytes() == compilation.artifact.serialized
    with pytest.raises(FileExistsError):
        publish_packed_inference_artifact(compilation.artifact, destination)
    publish_packed_inference_artifact(
        compilation.artifact, destination, overwrite=True
    )
    assert destination.read_bytes() == compilation.artifact.serialized


@pytest.mark.parametrize(
    "proposal",
    (
        _proposal(operator="gt"),
        _proposal(required_literal="numeric_ge:temperature:74"),
        _proposal(insights=()),
        _proposal(
            insights=(
                PTAInsight("pta:input", "threshold", "temperature", (75,)),
            )
        ),
        _proposal(85.0),
        _proposal(True),
        _proposal(10**1000),
    ),
)
def test_review_rejects_semantically_false_or_malformed_threshold_products(
    proposal: PTAEscalationProposal,
) -> None:
    catalog = _catalog()
    with pytest.raises(PTAThresholdPromotionError):
        review_threshold_proposal(proposal, session=_session(), catalog=catalog)
    assert catalog.literals == ()


def test_review_rejects_mixed_value_barrier_and_nonnumeric_schema() -> None:
    mixed = _session(values=(60, 70, 70, 80), labels=(0, 0, 1, 1))
    catalog = _catalog()
    with pytest.raises(PTAThresholdPromotionError, match="adjacent label flip"):
        review_threshold_proposal(
            _proposal(75.0), session=mixed, catalog=catalog
        )
    with pytest.raises(PTAThresholdPromotionError, match="not numeric"):
        review_threshold_proposal(
            _proposal(), session=_session(), catalog=_catalog(FieldKind.CATEGORY)
        )


def test_compiler_rejects_a_candidate_not_registered_in_destination_catalog() -> None:
    _, _, _, materialized, _ = _compile()
    empty_catalog = _catalog()
    with pytest.raises(PTAThresholdPromotionError, match="absent"):
        compile_threshold_artifact(
            materialized, session=_session(), catalog=empty_catalog
        )
    assert empty_catalog.literals == ()


def _has_gprolog() -> bool:
    try:
        resolve_gprolog()
    except PrologResourceError:
        return False
    return True


@pytest.mark.skipif(not _has_gprolog(), reason="GNU Prolog is not installed")
def test_live_collective_invents_threshold_and_same_product_reaches_ptm_artifact(
    tmp_path: Path,
) -> None:
    session = _session()
    collective = PTACollectiveService().run(
        session,
        query=PTACollectiveQuery(
            numeric_fields=("temperature",),
            discover_intervals=False,
            derive_deescalation=False,
            derive_escalation=True,
        ),
    )
    proposal = next(
        proposal
        for proposal in collective.proposals
        if proposal.native_target == "threshold"
    )
    assert proposal.structure["threshold"] == 75.0
    assert isinstance(lower_exact(proposal), NotRepresentable)

    catalog = _catalog()
    reviewed = review_threshold_proposal(
        proposal, session=session, catalog=catalog
    )
    materialized = materialize_threshold_clause(
        reviewed, session=session, catalog=catalog
    )
    compilation = compile_threshold_artifact(
        materialized, session=session, catalog=catalog
    )
    destination = tmp_path / "collective-threshold.ptm"
    publish_packed_inference_artifact(compilation.artifact, destination)

    assert destination.is_file()
    assert compilation.artifact.predict_records(
        ({"temperature": value} for value in (60, 70, 80, 90))
    ) == (0, 0, 1, 1)


def test_reviewed_object_cannot_be_reused_after_descriptor_tampering() -> None:
    catalog = _catalog()
    reviewed = review_threshold_proposal(
        _proposal(), session=_session(), catalog=catalog
    )
    tampered = replace(
        reviewed,
        descriptor=replace(reviewed.descriptor, literal_id=reviewed.descriptor.literal_id + 1),
    )
    with pytest.raises(PTAThresholdPromotionError, match="no longer matches"):
        materialize_threshold_clause(
            tampered, session=_session(), catalog=catalog
        )
    assert catalog.literals == ()


def test_forged_review_evidence_and_post_approval_session_changes_fail_closed() -> None:
    session = _session()
    catalog = _catalog()
    reviewed = review_threshold_proposal(
        _proposal(), session=session, catalog=catalog
    )
    forged = replace(
        reviewed,
        evidence=replace(reviewed.evidence, observations_digest="sha256:" + "0" * 64),
    )
    with pytest.raises(PTAThresholdPromotionError, match="no longer matches"):
        materialize_threshold_clause(
            forged, session=session, catalog=catalog
        )
    assert catalog.literals == ()

    materialized = materialize_threshold_clause(
        reviewed, session=session, catalog=catalog
    )
    forged_proposal = replace(
        materialized.proposal,
        proposal_id="pta:forged-materialization-provenance",
    )
    forged_lowered = lower_exact(forged_proposal, catalog=catalog)
    assert isinstance(forged_lowered, LoweredCandidate)
    forged_materialized = replace(
        materialized,
        proposal=forged_proposal,
        lowered=forged_lowered,
    )
    with pytest.raises(PTAThresholdPromotionError, match="approved exact clause"):
        compile_threshold_artifact(
            forged_materialized, session=session, catalog=catalog
        )

    session.add_observation("pta:input", 4, "temperature", 100)
    session.add_example_label(4, 1)
    with pytest.raises(PTAThresholdPromotionError, match="no longer matches"):
        compile_threshold_artifact(
            materialized, session=session, catalog=catalog
        )


def test_review_provenance_is_field_scoped_and_numeric_types_are_identity_bearing() -> None:
    session = _session()
    catalog = _catalog()
    reviewed = review_threshold_proposal(
        _proposal(), session=session, catalog=catalog
    )
    materialized = materialize_threshold_clause(
        reviewed, session=session, catalog=catalog
    )

    session.add_observation("pta:input", 0, "humidity", 50)
    compilation = compile_threshold_artifact(
        materialized, session=session, catalog=catalog
    )
    assert compilation.artifact.verify_conformance()

    integer_review = review_threshold_proposal(
        _proposal(75), session=_session(), catalog=_catalog()
    )
    float_review = review_threshold_proposal(
        _proposal(75.0), session=_session(), catalog=_catalog()
    )
    assert integer_review.descriptor.literal_id != float_review.descriptor.literal_id
