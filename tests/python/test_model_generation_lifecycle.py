from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil
from itertools import product

import pytest
import prolog_tsetlin.services.model_generation as model_generation_service

from prolog_tsetlin.model_generation import (
    AdaptiveBehaviorIdentity,
    AdaptiveSnapshotEnvelope,
    CorpusExample,
    CorpusRole,
    DeescalationCorpora,
    DriftAuditPolicy,
    EvidenceUsage,
    EvidenceUsagePurpose,
    GenerationKind,
    LabeledCorpus,
    LiveRuntimeConformanceEvidence,
    LifecycleCorpora,
    LiteralContractionLineage,
    ModelGeneration,
    ModelGenerationError,
    ModelGenerationLineage,
    OrderedLiteralManifest,
    PromotionAuditPolicy,
    PromotionAuditSnapshot,
    PromotionRuntimeConformanceEvidence,
    PrologInventionEvidence,
    PrologDeescalationEvidence,
    RuntimeConformanceReport,
    ThresholdCandidateBudget,
    ThresholdCandidateSelection,
    ThresholdCandidateSelectionPolicy,
    adapt_extended_parent,
    audit_parent_child,
    audit_runtime_conformance,
    drift_requires_reopen,
    extend_parent_with_threshold,
)
from prolog_tsetlin.model_artifact import PackedTMInferenceArtifact
from prolog_tsetlin.pta import (
    PTACollectiveProductCount,
    PTAEscalationProposal,
    PTAInsight,
    PTAReasoningSession,
    review_threshold_proposal,
)
from prolog_tsetlin.prolog_resources import PrologResourceError, resolve_gprolog
from prolog_tsetlin.reference import ScalarBinaryTsetlinMachine, extend_snapshot_features
from prolog_tsetlin.representation import FeatureSchema, FieldKind, LiteralCatalog
from prolog_tsetlin.services.model_generation import (
    LifecycleEventKind,
    ModelGenerationController,
    ModelGenerationStore,
    compile_generation_artifact,
    execute_trained_parent_lifecycle,
    execute_trained_parent_lifecycle_with_candidates,
    execute_literal_deescalation_lifecycle,
    invent_literal_contraction_for_corpus,
    invent_threshold_candidates_for_corpus,
    reopen_and_restore_for_drift,
    verify_artifact_with_ptmrt,
)
from prolog_tsetlin.services.telemetry import TelemetrySession


ROOT = Path(__file__).resolve().parents[2]
GOLDEN_CHILD_HEX = ROOT / "tests" / "data" / "trained_parent_child_v1.hex"


def _assert_same_portable_artifact_contract(
    actual: PackedTMInferenceArtifact,
    expected: PackedTMInferenceArtifact,
) -> None:
    """Compare executable semantics while retaining environment attestations.

    Candidate-set identity binds the measured GNU Prolog executable, version,
    and packaged modules. Independently replaying the same episode with a
    different attested GNU Prolog installation therefore produces a different
    candidate-set ID, selection ID, and outer artifact digest even when the
    portable model is otherwise byte-for-byte equivalent.
    """

    assert actual.number_of_clauses == expected.number_of_clauses
    assert actual.number_of_features == expected.number_of_features
    assert actual.threshold == expected.threshold
    assert actual.positive_include_masks == expected.positive_include_masks
    assert actual.negative_include_masks == expected.negative_include_masks
    assert actual.conformance_cases == expected.conformance_cases

    actual_manifest = deepcopy(dict(actual.manifest))
    expected_manifest = deepcopy(dict(expected.manifest))
    environment_bound_keys = (
        "threshold_candidate_set_id",
        "threshold_candidate_selection_id",
    )
    for manifest in (actual_manifest, expected_manifest):
        signature = manifest["validation"]["signature"]
        for key in environment_bound_keys:
            del signature[key]
    assert actual_manifest == expected_manifest


def test_portable_artifact_contract_excludes_only_environment_attestations() -> None:
    golden = PackedTMInferenceArtifact.from_bytes(
        bytes.fromhex(GOLDEN_CHILD_HEX.read_text(encoding="ascii"))
    )
    environment_manifest = deepcopy(dict(golden.manifest))
    environment_signature = environment_manifest["validation"]["signature"]
    environment_signature["threshold_candidate_set_id"] = "sha256:" + "1" * 64
    environment_signature["threshold_candidate_selection_id"] = (
        "sha256:" + "2" * 64
    )
    environment_artifact = replace(golden, manifest=environment_manifest)

    _assert_same_portable_artifact_contract(environment_artifact, golden)

    semantic_manifest = deepcopy(environment_manifest)
    semantic_manifest["validation"]["signature"]["adaptive_snapshot_id"] = (
        "sha256:" + "3" * 64
    )
    with pytest.raises(AssertionError):
        _assert_same_portable_artifact_contract(
            replace(golden, manifest=semantic_manifest), golden
        )


def _ptmrt_path() -> Path | None:
    located = shutil.which("ptmrt")
    candidates = (
        Path(located) if located else None,
        ROOT / "out" / "build" / "Release" / "ptmrt.exe",
        ROOT / "out" / "build" / "ptmrt.exe",
        ROOT / "out" / "build" / "ptmrt",
    )
    return next((path for path in candidates if path is not None and path.is_file()), None)


def _has_gprolog() -> bool:
    try:
        resolve_gprolog()
    except PrologResourceError:
        return False
    return True


STRICT_DRIFT_POLICY = DriftAuditPolicy(
    minimum_observations=8,
    minimum_regressions=4,
    minimum_regression_rate=0.5,
    minimum_error_increase=4,
    minimum_observations_per_class=4,
)


def _corpus(
    role: CorpusRole,
    first_id: int,
    values: tuple[int, ...],
    labels: tuple[int, ...],
) -> LabeledCorpus:
    return LabeledCorpus(
        "thermostat-generation-v1",
        role,
        tuple(
            CorpusExample(
                first_id + index,
                {
                    "temperature": value,
                    "mode": "heat",
                    "previous_state": False,
                },
                label,
            )
            for index, (value, label) in enumerate(zip(values, labels))
        ),
    )


def _fixture_corpora() -> tuple[LabeledCorpus, LifecycleCorpora, LabeledCorpus]:
    parent = _corpus(
        CorpusRole.PARENT_TRAINING,
        0,
        (50, 55, 60, 65, 85, 90, 95, 100),
        (0, 0, 0, 0, 1, 1, 1, 1),
    )
    invention = _corpus(
        CorpusRole.INVENTION, 100, (62, 72, 78, 88), (0, 0, 1, 1)
    )
    adaptation = _corpus(
        CorpusRole.ADAPTATION,
        200,
        (58, 64, 68, 71, 79, 82, 87, 92),
        (0, 0, 0, 0, 1, 1, 1, 1),
    )
    promotion = _corpus(
        CorpusRole.PROMOTION,
        300,
        (61, 66, 73, 74, 76, 81, 86, 89),
        (0, 0, 0, 0, 1, 1, 1, 1),
    )
    live = _corpus(
        CorpusRole.LIVE,
        400,
        (59, 63, 69, 72, 78, 83, 88, 94),
        (1, 1, 1, 1, 0, 0, 0, 0),
    )
    return parent, LifecycleCorpora(invention, adaptation, promotion), live


def _multi_field_corpora() -> tuple[LabeledCorpus, LifecycleCorpora]:
    def make(
        role: CorpusRole,
        first_id: int,
        temperatures: tuple[int, ...],
        pressures: tuple[int, ...],
        labels: tuple[int, ...],
    ) -> LabeledCorpus:
        return LabeledCorpus(
            "thermostat-candidate-set-v1",
            role,
            tuple(
                CorpusExample(
                    first_id + index,
                    {
                        "temperature": temperature,
                        "pressure": pressure,
                        "mode": "heat",
                        "previous_state": False,
                    },
                    label,
                )
                for index, (temperature, pressure, label) in enumerate(
                    zip(temperatures, pressures, labels)
                )
            ),
        )

    parent = make(
        CorpusRole.PARENT_TRAINING,
        10_000,
        (50, 55, 60, 65, 85, 90, 95, 100),
        (20,) * 8,
        (0, 0, 0, 0, 1, 1, 1, 1),
    )
    invention = make(
        CorpusRole.INVENTION,
        10_100,
        (62, 72, 78, 88),
        (10, 20, 30, 40),
        (0, 0, 1, 1),
    )
    adaptation = make(
        CorpusRole.ADAPTATION,
        10_200,
        (58, 64, 68, 71, 79, 82, 87, 92),
        (20,) * 8,
        (0, 0, 0, 0, 1, 1, 1, 1),
    )
    promotion = make(
        CorpusRole.PROMOTION,
        10_300,
        (61, 66, 73, 74, 76, 81, 86, 89),
        (20,) * 8,
        (0, 0, 0, 0, 1, 1, 1, 1),
    )
    return parent, LifecycleCorpora(invention, adaptation, promotion)


def _parent() -> tuple[ScalarBinaryTsetlinMachine, OrderedLiteralManifest]:
    parent_training, _, _ = _fixture_corpora()
    schema = FeatureSchema.from_fields(
        temperature=FieldKind.NUMBER,
        mode=FieldKind.CATEGORY,
        previous_state=FieldKind.BOOLEAN,
    )
    catalog = LiteralCatalog(schema)
    catalog.category_eq("mode", "heat")
    catalog.category_eq("previous_state", True)
    machine = ScalarBinaryTsetlinMachine(
        20,
        2,
        states_per_action=20,
        specificity=3.0,
        threshold=10,
        seed=7,
    )
    machine.fit_literal_batch(
        catalog.encode(parent_training.records).ta,
        parent_training.labels,
        epochs=150,
    )
    return machine, OrderedLiteralManifest.from_catalog(catalog)


def _multi_field_parent() -> tuple[ScalarBinaryTsetlinMachine, OrderedLiteralManifest]:
    parent_training, _ = _multi_field_corpora()
    schema = FeatureSchema.from_fields(
        temperature=FieldKind.NUMBER,
        pressure=FieldKind.NUMBER,
        mode=FieldKind.CATEGORY,
        previous_state=FieldKind.BOOLEAN,
    )
    catalog = LiteralCatalog(schema)
    catalog.category_eq("mode", "heat")
    catalog.category_eq("previous_state", True)
    machine = ScalarBinaryTsetlinMachine(
        20,
        2,
        states_per_action=20,
        specificity=3.0,
        threshold=10,
        seed=7,
    )
    machine.fit_literal_batch(
        catalog.encode(parent_training.records).ta,
        parent_training.labels,
        epochs=150,
    )
    return machine, OrderedLiteralManifest.from_catalog(catalog)


def _deescalation_fixture() -> tuple[
    LabeledCorpus,
    DeescalationCorpora,
    LabeledCorpus,
    ScalarBinaryTsetlinMachine,
    OrderedLiteralManifest,
]:
    def make(
        role: CorpusRole,
        first_id: int,
        values: tuple[float, ...],
        labels: tuple[int, ...],
    ) -> LabeledCorpus:
        return LabeledCorpus(
            "thermostat-deescalation-v1",
            role,
            tuple(
                CorpusExample(
                    first_id + index,
                    {"temperature": value},
                    label,
                )
                for index, (value, label) in enumerate(zip(values, labels))
            ),
        )

    parent_training = make(
        CorpusRole.PARENT_TRAINING,
        20_000,
        (50.0, 60.0, 70.0, 74.0, 76.0, 80.0, 90.0, 100.0),
        (0, 0, 0, 0, 1, 1, 1, 1),
    )
    proof = make(
        CorpusRole.DEESCALATION_PROOF,
        20_100,
        (51.0, 69.0, 77.0, 91.0),
        (0, 0, 1, 1),
    )
    confirmation = make(
        CorpusRole.DEESCALATION_CONFIRMATION,
        20_200,
        (52.0, 68.0, 78.0, 92.0),
        (0, 0, 1, 1),
    )
    promotion = make(
        CorpusRole.PROMOTION,
        20_300,
        (53.0, 67.0, 79.0, 93.0),
        (0, 0, 1, 1),
    )
    live = make(
        CorpusRole.LIVE,
        20_400,
        (54.0, 66.0, 75.25, 75.75),
        (0, 0, 1, 1),
    )
    schema = FeatureSchema.from_fields(temperature=FieldKind.NUMBER)
    catalog = LiteralCatalog(schema)
    catalog.numeric_ge("temperature", 76.0)
    catalog.numeric_ge("temperature", 75.0)
    machine = ScalarBinaryTsetlinMachine(
        1,
        2,
        states_per_action=20,
        specificity=3.0,
        threshold=10,
        seed=23,
    )
    parent_snapshot = replace(
        machine.snapshot(),
        states=((20, 20, 21, 20),),
    )
    machine.restore(parent_snapshot)
    return (
        parent_training,
        DeescalationCorpora(proof, confirmation, promotion),
        live,
        machine,
        OrderedLiteralManifest.from_catalog(catalog),
    )


def _session(corpus: LabeledCorpus) -> PTAReasoningSession:
    session = PTAReasoningSession(corpus.dataset_id)
    for example in corpus.examples:
        for field, value in example.record.items():
            session.add_observation("pta:input", example.example_id, field, value)
        session.add_example_label(example.example_id, example.label)
    return session


def _reviewed(corpus: LabeledCorpus, manifest: OrderedLiteralManifest):
    proposal = PTAEscalationProposal(
        proposal_id="pta:escalation:threshold:test:75",
        source_pta_ids=("pta:input", "pta:escalation"),
        supporting_insights=(
            PTAInsight("pta:input", "threshold", "temperature", (75.0,)),
        ),
        counterexamples_addressed=(),
        required_literals=("numeric_ge:temperature:75",),
        native_target="threshold",
        structure={"field": "temperature", "operator": "ge", "threshold": 75.0},
        resource_bounds={"literal_count": 1},
        support_trace=("derived by GNU Prolog exception_clause/3",),
    )
    session = _session(corpus)
    return session, review_threshold_proposal(
        proposal, session=session, catalog=manifest.build_catalog()
    )


def _candidate():
    parent_training, corpora, _ = _fixture_corpora()
    parent, manifest = _parent()
    session, reviewed = _reviewed(corpora.invention, manifest)
    extended = extend_parent_with_threshold(
        parent.snapshot(),
        manifest,
        reviewed,
        session=session,
        equivalence_records=tuple(
            record
            for corpus in (parent_training, corpora.invention, corpora.promotion)
            for record in corpus.records
        ),
    )
    child = adapt_extended_parent(extended, corpora.adaptation, epochs=5)
    _, artifact = compile_generation_artifact(
        child.snapshot.snapshot,
        child.manifest,
        name="test adapted child",
        validation_records=corpora.promotion.records,
        validation_signature={"test": "model-generation"},
    )
    return parent_training, corpora, parent, manifest, session, reviewed, extended, child, artifact


def test_literal_manifest_round_trip_preserves_exact_feature_positions() -> None:
    _, manifest = _parent()
    rebuilt = OrderedLiteralManifest.from_dict(manifest.to_dict())

    assert rebuilt == manifest
    assert rebuilt.build_catalog().literals == manifest.literals
    assert rebuilt.manifest_id == manifest.manifest_id

    forged = manifest.to_dict()
    forged["literals"] = list(reversed(forged["literals"]))
    with pytest.raises(ModelGenerationError, match="digest"):
        OrderedLiteralManifest.from_dict(forged)

    wrong_type = manifest.to_dict()
    wrong_type["fields"][0]["name"] = 7
    with pytest.raises(ModelGenerationError, match="types"):
        OrderedLiteralManifest.from_dict(wrong_type)


@pytest.mark.skipif(not _has_gprolog(), reason="GNU Prolog is required")
def test_input_pta_returns_a_complete_reviewed_threshold_candidate_set() -> None:
    _, corpora = _multi_field_corpora()
    _, manifest = _multi_field_parent()

    invention = invent_threshold_candidates_for_corpus(
        corpora.invention,
        manifest,
        numeric_fields=("temperature", "pressure"),
        budget=ThresholdCandidateBudget(maximum_fields=2, maximum_candidates=4),
    )

    assert invention.evidence.numeric_fields == ("pressure", "temperature")
    assert invention.evidence.available_candidates == 2
    assert {
        (candidate.field, candidate.threshold)
        for candidate in invention.evidence.candidates
    } == {("pressure", 25.0), ("temperature", 75.0)}
    assert tuple(
        proposal.semantic_id() for proposal in invention.proposals
    ) == tuple(
        candidate.proposal_semantic_id
        for candidate in invention.evidence.candidates
    )
    assert all(
        reviewed.descriptor.literal_id == candidate.invented_literal_id
        for reviewed, candidate in zip(
            invention.reviewed, invention.evidence.candidates
        )
    )


@pytest.mark.skipif(not _has_gprolog(), reason="GNU Prolog is required")
def test_input_pta_rejects_a_truncated_candidate_set() -> None:
    _, manifest = _parent()
    corpus = _corpus(
        CorpusRole.INVENTION,
        11_000,
        (10, 20, 30, 40),
        (0, 1, 0, 1),
    )

    with pytest.raises(ModelGenerationError, match="exceeds its explicit budget"):
        invent_threshold_candidates_for_corpus(
            corpus,
            manifest,
            numeric_fields=("temperature",),
            budget=ThresholdCandidateBudget(
                maximum_fields=1, maximum_candidates=2
            ),
        )


@pytest.mark.skipif(not _has_gprolog(), reason="GNU Prolog is required")
def test_deescalation_pta_attests_complete_literal_equivalence() -> None:
    _, corpora, _, parent, manifest = _deescalation_fixture()

    invention = invent_literal_contraction_for_corpus(
        corpora.proof,
        parent.snapshot(),
        manifest,
        maximum_candidates=4,
    )

    assert isinstance(invention.evidence, PrologDeescalationEvidence)
    assert invention.evidence.equivalent_pairs == (
        tuple(sorted(manifest.literal_ids)),
    )
    assert invention.evidence.surviving_literal_id == manifest.literal_ids[0]
    assert invention.evidence.removed_literal_id == manifest.literal_ids[1]
    assert invention.evidence.session_digest == (
        model_generation_service.content_digest(invention.session.to_dict())
    )
    assert PrologDeescalationEvidence.from_dict(
        invention.evidence.to_dict()
    ) == invention.evidence


@pytest.mark.skipif(not _has_gprolog(), reason="GNU Prolog is required")
def test_deescalation_pta_rejects_truncated_equivalence_set() -> None:
    schema = FeatureSchema.from_fields(temperature=FieldKind.NUMBER)
    catalog = LiteralCatalog(schema)
    catalog.numeric_ge("temperature", 76.0)
    catalog.numeric_ge("temperature", 75.0)
    catalog.numeric_ge("temperature", 74.0)
    manifest = OrderedLiteralManifest.from_catalog(catalog)
    machine = ScalarBinaryTsetlinMachine(1, 3, seed=5)
    proof = LabeledCorpus(
        "thermostat-deescalation-budget-v1",
        CorpusRole.DEESCALATION_PROOF,
        (
            CorpusExample(0, {"temperature": 60.0}, 0),
            CorpusExample(1, {"temperature": 80.0}, 1),
        ),
    )

    with pytest.raises(ModelGenerationError, match="exceeds its explicit budget"):
        invent_literal_contraction_for_corpus(
            proof,
            machine.snapshot(),
            manifest,
            maximum_candidates=2,
        )


@pytest.mark.skipif(not _has_gprolog(), reason="GNU Prolog is required")
def test_deescalation_pta_rejects_self_attested_incomplete_valid_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = FeatureSchema.from_fields(temperature=FieldKind.NUMBER)
    catalog = LiteralCatalog(schema)
    for threshold in (76.0, 75.0, 74.0):
        catalog.numeric_ge("temperature", threshold)
    manifest = OrderedLiteralManifest.from_catalog(catalog)
    machine = ScalarBinaryTsetlinMachine(1, 3, seed=5)
    proof = LabeledCorpus(
        "thermostat-deescalation-completeness-v1",
        CorpusRole.DEESCALATION_PROOF,
        (
            CorpusExample(0, {"temperature": 60.0}, 0),
            CorpusExample(1, {"temperature": 80.0}, 1),
        ),
    )
    real_run = model_generation_service.PTACollectiveService.run

    def omit_one_valid_pair(self, *args, **kwargs):
        result = real_run(self, *args, **kwargs)
        redundancies = tuple(
            item for item in result.insights if item.kind == "literal_redundant"
        )
        assert len(redundancies) == 3
        omitted = redundancies[0]
        retained = tuple(item for item in result.insights if item != omitted)
        counts = dict(result.product_counts)
        counts["literal_redundancies"] = PTACollectiveProductCount(2, 2)
        return replace(result, insights=retained, product_counts=counts)

    monkeypatch.setattr(
        model_generation_service.PTACollectiveService,
        "run",
        omit_one_valid_pair,
    )
    with pytest.raises(ModelGenerationError, match="independent Python"):
        invent_literal_contraction_for_corpus(
            proof,
            machine.snapshot(),
            manifest,
            maximum_candidates=4,
        )


@pytest.mark.skipif(
    not _has_gprolog() or _ptmrt_path() is None,
    reason="live GNU Prolog and a built ptmrt are required",
)
def test_literal_deescalation_lifecycle_contracts_promotes_and_restores(
    tmp_path: Path,
) -> None:
    parent_training, corpora, live, parent, manifest = _deescalation_fixture()
    original_snapshot = parent.snapshot()
    store = ModelGenerationStore(tmp_path / "deescalation-store")

    result = execute_literal_deescalation_lifecycle(
        parent_snapshot=original_snapshot,
        parent_manifest=manifest,
        parent_training_corpus=parent_training,
        corpora=corpora,
        maximum_candidates=4,
        promotion_policy=PromotionAuditPolicy(
            4,
            require_strict_improvement=False,
            maximum_regressions=0,
        ),
        store=store,
        ptmrt_executable=_ptmrt_path(),
    )

    assert isinstance(result.lineage, LiteralContractionLineage)
    assert LiteralContractionLineage.from_dict(
        result.lineage.to_dict()
    ) == result.lineage
    assert result.contracted_parent.snapshot.snapshot.number_of_features == 1
    assert result.contracted_parent.manifest.literals == manifest.literals[:1]
    assert result.contracted_parent.snapshot.snapshot.states == ((21, 20),)
    assert result.contracted_parent.snapshot.snapshot.rng_state == (
        original_snapshot.rng_state
    )
    assert result.promotion_audit.accepted
    assert result.promotion_audit.parent_errors == (
        result.promotion_audit.child_errors
    )
    assert result.promotion_audit.regressions == 0
    assert result.conformance.exact
    assert store.load_deescalation_evidence(
        result.deescalation_evidence.evidence_id
    ) == result.deescalation_evidence
    assert PromotionRuntimeConformanceEvidence.from_dict(
        result.promotion_conformance_evidence.to_dict()
    ) == result.promotion_conformance_evidence
    assert ModelGenerationController(
        store, ptmrt_executable=_ptmrt_path()
    ).active_generation_id == result.child_generation.generation_id

    drift, restored = reopen_and_restore_for_drift(
        result,
        live,
        DriftAuditPolicy(
            minimum_observations=4,
            minimum_regressions=2,
            minimum_regression_rate=0.5,
            minimum_error_increase=2,
            minimum_observations_per_class=2,
        ),
    )
    assert drift.parent_errors == 0
    assert drift.child_errors == 2
    assert drift.regressions == 2
    assert restored.snapshot.snapshot == original_snapshot
    assert restored.manifest == manifest
    assert result.controller.active_generation_id == (
        result.parent_generation.generation_id
    )

    original_next = ScalarBinaryTsetlinMachine(
        original_snapshot.number_of_clauses,
        original_snapshot.number_of_features,
        states_per_action=original_snapshot.states_per_action,
        specificity=original_snapshot.specificity,
        threshold=original_snapshot.threshold,
        seed=0,
    )
    original_next.restore(original_snapshot)
    update = manifest.build_catalog().encode((parent_training.records[0],)).ta
    original_next.fit_literal_batch(update, (parent_training.labels[0],), epochs=1)
    restored.machine.fit_literal_batch(
        update, (parent_training.labels[0],), epochs=1
    )
    assert restored.machine.snapshot() == original_next.snapshot()


@pytest.mark.skipif(
    not _has_gprolog() or _ptmrt_path() is None,
    reason="live GNU Prolog and a built ptmrt are required",
)
def test_literal_deescalation_rejects_confirmation_divergence(
    tmp_path: Path,
) -> None:
    parent_training, corpora, _, parent, manifest = _deescalation_fixture()
    divergent_confirmation = LabeledCorpus(
        corpora.confirmation.dataset_id,
        CorpusRole.DEESCALATION_CONFIRMATION,
        (
            CorpusExample(21_000, {"temperature": 55.5}, 0),
            CorpusExample(21_001, {"temperature": 65.5}, 0),
            CorpusExample(21_002, {"temperature": 75.5}, 1),
            CorpusExample(21_003, {"temperature": 85.5}, 1),
        ),
    )
    divergent = DeescalationCorpora(
        corpora.proof,
        divergent_confirmation,
        corpora.promotion,
    )
    store = ModelGenerationStore(tmp_path / "deescalation-divergence-store")

    with pytest.raises(ModelGenerationError, match="confirmation corpus"):
        execute_literal_deescalation_lifecycle(
            parent_snapshot=parent.snapshot(),
            parent_manifest=manifest,
            parent_training_corpus=parent_training,
            corpora=divergent,
            maximum_candidates=4,
            promotion_policy=PromotionAuditPolicy(
                4,
                require_strict_improvement=False,
                maximum_regressions=0,
            ),
            store=store,
            ptmrt_executable=_ptmrt_path(),
        )
    assert store.read_events()[-1].kind is LifecycleEventKind.EVIDENCE_ABANDONED


@pytest.mark.skipif(
    not _has_gprolog() or _ptmrt_path() is None,
    reason="live GNU Prolog and a built ptmrt are required",
)
def test_literal_deescalation_admission_reconstructs_exact_contraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_training, corpora, _, parent, manifest = _deescalation_fixture()
    real_contract = model_generation_service.contract_parent_with_equivalent_literal
    calls = 0

    def inject_unattested_contraction(*args, **kwargs):
        nonlocal calls
        calls += 1
        contracted = real_contract(*args, **kwargs)
        if calls != 1:
            return contracted
        snapshot = contracted.snapshot.snapshot
        states = [list(row) for row in snapshot.states]
        states[0][0] += 1
        forged = replace(snapshot, states=tuple(tuple(row) for row in states))
        return replace(
            contracted,
            snapshot=AdaptiveSnapshotEnvelope(forged),
        )

    monkeypatch.setattr(
        model_generation_service,
        "contract_parent_with_equivalent_literal",
        inject_unattested_contraction,
    )
    store = ModelGenerationStore(tmp_path / "unattested-contraction-store")
    with pytest.raises(ModelGenerationError, match="cannot be reconstructed"):
        execute_literal_deescalation_lifecycle(
            parent_snapshot=parent.snapshot(),
            parent_manifest=manifest,
            parent_training_corpus=parent_training,
            corpora=corpora,
            maximum_candidates=4,
            promotion_policy=PromotionAuditPolicy(
                4,
                require_strict_improvement=False,
                maximum_regressions=0,
            ),
            store=store,
            ptmrt_executable=_ptmrt_path(),
        )
    assert calls >= 2


@pytest.mark.skipif(
    not _has_gprolog() or _ptmrt_path() is None,
    reason="live GNU Prolog and a built ptmrt are required",
)
def test_multi_candidate_lifecycle_selects_before_promotion_and_recovers(
    tmp_path: Path,
) -> None:
    parent_training, corpora = _multi_field_corpora()
    parent, manifest = _multi_field_parent()
    store = ModelGenerationStore(tmp_path / "multi-candidate-store")

    result = execute_trained_parent_lifecycle_with_candidates(
        parent_snapshot=parent.snapshot(),
        parent_manifest=manifest,
        parent_training_corpus=parent_training,
        corpora=corpora,
        numeric_fields=("temperature", "pressure"),
        candidate_budget=ThresholdCandidateBudget(
            maximum_fields=2, maximum_candidates=4
        ),
        selection_policy=ThresholdCandidateSelectionPolicy(
            minimum_observations=8
        ),
        adaptation_epochs=5,
        promotion_policy=PromotionAuditPolicy(8),
        store=store,
        ptmrt_executable=_ptmrt_path(),
    )

    assert len(result.candidate_selection.outcomes) == 2
    selected = result.candidate_selection.selected_outcome
    selected_candidate = next(
        candidate
        for candidate in result.invention_evidence.candidates
        if candidate.proposal_semantic_id == selected.proposal_semantic_id
    )
    assert selected_candidate.field == "temperature"
    assert selected_candidate.threshold == 75.0
    assert selected.child_errors < selected.parent_errors
    assert result.lineage.candidate_selection_id == (
        result.candidate_selection.selection_id
    )
    assert result.promotion_audit.accepted
    assert store.load_threshold_candidate_set(
        result.invention_evidence.candidate_set_id
    ) == result.invention_evidence
    assert store.load_threshold_candidate_selection(
        result.candidate_selection.selection_id
    ) == result.candidate_selection
    assert ModelGenerationController(
        store, ptmrt_executable=_ptmrt_path()
    ).active_generation_id == result.child_generation.generation_id

    forged = result.candidate_selection.to_dict()
    loser = next(
        outcome
        for outcome in result.candidate_selection.outcomes
        if outcome.proposal_semantic_id != selected.proposal_semantic_id
    )
    forged["selected_proposal_semantic_id"] = loser.proposal_semantic_id
    forged["selected_proposal_provenance_id"] = loser.proposal_provenance_id
    with pytest.raises(ModelGenerationError, match="deterministic"):
        ThresholdCandidateSelection.from_dict(forged)

    legacy_lineage = replace(
        result.lineage,
        candidate_selection_id=None,
        promotion_conformance_evidence_id=None,
        schema="ptm.model-generation-lineage.v4",
    )
    with pytest.raises(ModelGenerationError, match="current lineage schema"):
        result.controller.record_candidate(legacy_lineage)


@pytest.mark.skipif(
    not _has_gprolog() or _ptmrt_path() is None,
    reason="live GNU Prolog and a built ptmrt are required",
)
def test_v5_candidate_admission_replays_prolog_and_independent_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_training, corpora, _ = _fixture_corpora()
    parent, manifest = _parent()
    real_invent = model_generation_service.invent_threshold_candidates_for_corpus

    def self_authored_candidate_set(*args, **kwargs):
        invention = real_invent(*args, **kwargs)
        proposal = invention.proposals[0]
        forged_proposal = replace(
            proposal, proposal_id=proposal.proposal_id + ":self-authored"
        )
        forged_review = review_threshold_proposal(
            forged_proposal,
            session=invention.session,
            catalog=manifest.build_catalog(),
        )
        candidate = invention.evidence.candidates[0]
        forged_candidate = replace(
            candidate,
            proposal_semantic_id=forged_proposal.semantic_id(),
            proposal_provenance_id=forged_proposal.provenance_id(),
            proposal_payload=forged_proposal.to_dict(),
        )
        forged_evidence = replace(
            invention.evidence, candidates=(forged_candidate,)
        )
        return model_generation_service.ThresholdCandidateInvention(
            invention.session,
            (forged_proposal,),
            (forged_review,),
            forged_evidence,
        )

    monkeypatch.setattr(
        model_generation_service,
        "invent_threshold_candidates_for_corpus",
        self_authored_candidate_set,
    )
    store = ModelGenerationStore(tmp_path / "self-authored-candidate-store")
    with pytest.raises(ModelGenerationError, match="GNU Prolog replay"):
        execute_trained_parent_lifecycle_with_candidates(
            parent_snapshot=parent.snapshot(),
            parent_manifest=manifest,
            parent_training_corpus=parent_training,
            corpora=corpora,
            numeric_fields=("temperature",),
            candidate_budget=ThresholdCandidateBudget(
                maximum_fields=1, maximum_candidates=1
            ),
            selection_policy=ThresholdCandidateSelectionPolicy(),
            adaptation_epochs=5,
            promotion_policy=PromotionAuditPolicy(8),
            store=store,
            ptmrt_executable=_ptmrt_path(),
        )
    assert store.read_events()[-1].kind is LifecycleEventKind.EVIDENCE_ABANDONED


@pytest.mark.skipif(
    not _has_gprolog() or _ptmrt_path() is None,
    reason="live GNU Prolog and a built ptmrt are required",
)
def test_v5_candidate_admission_replays_exact_adaptation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_training, corpora, _ = _fixture_corpora()
    parent, manifest = _parent()
    real_adapt = model_generation_service.adapt_extended_parent
    adaptation_calls = 0

    def inject_unattested_child(extended, corpus, *, epochs):
        nonlocal adaptation_calls
        adaptation_calls += 1
        child = real_adapt(extended, corpus, epochs=epochs)
        if adaptation_calls != 1:
            return child
        snapshot = child.snapshot.snapshot
        states = [list(row) for row in snapshot.states]
        states[0][0] = states[0][0] - 1 if states[0][0] > 1 else 2
        forged_snapshot = replace(
            snapshot, states=tuple(tuple(row) for row in states)
        )
        return replace(child, snapshot=AdaptiveSnapshotEnvelope(forged_snapshot))

    monkeypatch.setattr(
        model_generation_service,
        "adapt_extended_parent",
        inject_unattested_child,
    )
    store = ModelGenerationStore(tmp_path / "unattested-adaptation-store")
    with pytest.raises(ModelGenerationError, match="alternative graph"):
        execute_trained_parent_lifecycle_with_candidates(
            parent_snapshot=parent.snapshot(),
            parent_manifest=manifest,
            parent_training_corpus=parent_training,
            corpora=corpora,
            numeric_fields=("temperature",),
            candidate_budget=ThresholdCandidateBudget(
                maximum_fields=1, maximum_candidates=1
            ),
            selection_policy=ThresholdCandidateSelectionPolicy(),
            adaptation_epochs=5,
            promotion_policy=PromotionAuditPolicy(8),
            store=store,
            ptmrt_executable=_ptmrt_path(),
        )
    assert adaptation_calls >= 2
    assert store.read_events()[-1].kind is LifecycleEventKind.EVIDENCE_ABANDONED


def test_representation_extension_appends_excluded_ta_pairs_without_rng_change() -> None:
    _, _, parent, _, _, _, extended, _, _ = _candidate()
    before = parent.snapshot()
    after = extended.snapshot.snapshot

    assert after.number_of_features == before.number_of_features + 1
    assert after.rng_state == before.rng_state
    assert extended.manifest.literals[:-1] == extended.parent_manifest.literals
    for old_states, new_states in zip(before.states, after.states):
        assert new_states[:-2] == old_states
        assert new_states[-2:] == (
            before.states_per_action,
            before.states_per_action,
        )
    assert extended.equivalence_case_count > 0

    parent_oracle = ScalarBinaryTsetlinMachine(
        before.number_of_clauses,
        before.number_of_features,
        states_per_action=before.states_per_action,
        specificity=before.specificity,
        threshold=before.threshold,
        seed=0,
    )
    parent_oracle.restore(before)
    extended_oracle = ScalarBinaryTsetlinMachine(
        after.number_of_clauses,
        after.number_of_features,
        states_per_action=after.states_per_action,
        specificity=after.specificity,
        threshold=after.threshold,
        seed=0,
    )
    extended_oracle.restore(after)
    for old_row in product((False, True), repeat=before.number_of_features):
        for new_truth in (False, True):
            new_row = old_row + (new_truth,)
            assert parent_oracle.score(old_row) == extended_oracle.score(new_row)
            assert parent_oracle.predict_one(old_row) == extended_oracle.predict_one(
                new_row
            )
            for clause in range(before.number_of_clauses):
                for prediction in (False, True):
                    assert parent_oracle.clause_output(
                        clause, old_row, prediction=prediction
                    ) == extended_oracle.clause_output(
                        clause, new_row, prediction=prediction
                    )


def test_snapshot_extension_rejects_invalid_width_and_does_not_consume_rng() -> None:
    parent, _ = _parent()
    snapshot = parent.snapshot()

    first = extend_snapshot_features(snapshot, 2)
    second = extend_snapshot_features(snapshot, 2)

    assert first == second
    assert first.rng_state == snapshot.rng_state
    with pytest.raises(ValueError, match="positive"):
        extend_snapshot_features(snapshot, 0)


def test_snapshot_envelope_round_trip_preserves_integer_specificity(
    tmp_path: Path,
) -> None:
    parent, _ = _parent()
    envelope = AdaptiveSnapshotEnvelope(replace(parent.snapshot(), specificity=3))
    store = ModelGenerationStore(tmp_path / "store")

    store.put_snapshot(envelope)
    restored = store.load_snapshot(envelope.snapshot_id)

    assert restored == envelope
    assert type(restored.snapshot.specificity) is int
    assert restored.snapshot_id == envelope.snapshot_id


def test_snapshot_envelope_rejects_huge_declared_dimensions_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, _ = _parent()
    forged = AdaptiveSnapshotEnvelope(parent.snapshot()).to_dict()
    raw = forged["snapshot"]
    raw["number_of_clauses"] = 1_000_000_000
    raw["number_of_features"] = 1_000_000_000
    raw["states"] = [[1, 1]]
    forged["snapshot_id"] = model_generation_service.content_digest(
        {
            "schema": forged["schema"],
            "rng_algorithm": forged["rng_algorithm"],
            "snapshot": raw,
        }
    )

    def allocation_must_not_run(*args, **kwargs):
        raise AssertionError("hostile snapshot reached TM allocation")

    monkeypatch.setattr(
        model_generation_service.ScalarBinaryTsetlinMachine,
        "__init__",
        allocation_must_not_run,
    )
    with pytest.raises(ModelGenerationError, match="outside its bounds"):
        AdaptiveSnapshotEnvelope.from_dict(forged)


def test_promotion_is_paired_and_requires_exact_runtime_conformance() -> None:
    _, corpora, parent, manifest, _, _, _, child, artifact = _candidate()
    unverified = audit_runtime_conformance(
        child,
        artifact,
        corpora.promotion.records,
        ptmrt_verified=False,
        ptmrt_artifact_id=None,
    )
    rejected = audit_parent_child(
        parent.snapshot(),
        manifest,
        child,
        corpora.promotion,
        unverified,
        PromotionAuditPolicy(8),
    )
    assert rejected.parent_errors == 4
    assert rejected.child_errors == 0
    assert rejected.improvements == 4
    assert rejected.regressions == 0
    assert not rejected.accepted

    exact = replace(
        unverified,
        ptmrt_verified=True,
        ptmrt_artifact_id=artifact.artifact_id,
    )
    accepted = audit_parent_child(
        parent.snapshot(),
        manifest,
        child,
        corpora.promotion,
        exact,
        PromotionAuditPolicy(8),
    )
    assert accepted.accepted
    assert accepted.disagreements == 4
    assert sum(item.observed for item in accepted.class_counts) == 8
    assert accepted.from_dict(accepted.to_dict()) == accepted

    signature = artifact.manifest["validation"]["signature"]
    assert signature["adaptive_snapshot_id"] == child.snapshot.snapshot_id
    assert signature["ordered_literal_manifest_id"] == child.manifest.manifest_id

    unchanged_examples = tuple(
        example
        for example, parent_score, child_score in zip(
            corpora.promotion.examples,
            accepted.parent_scores,
            accepted.child_scores,
        )
        if (parent_score > 0) == (child_score > 0)
    )
    unchanged = LabeledCorpus(
        corpora.promotion.dataset_id,
        CorpusRole.PROMOTION,
        unchanged_examples,
    )
    unchanged_conformance = audit_runtime_conformance(
        child,
        artifact,
        unchanged.records,
        ptmrt_verified=True,
        ptmrt_artifact_id=artifact.artifact_id,
    )
    equal = audit_parent_child(
        parent.snapshot(),
        manifest,
        child,
        unchanged,
        unchanged_conformance,
        PromotionAuditPolicy(
            len(unchanged_examples), require_strict_improvement=False
        ),
    )
    strict_equal = audit_parent_child(
        parent.snapshot(),
        manifest,
        child,
        unchanged,
        unchanged_conformance,
        PromotionAuditPolicy(len(unchanged_examples)),
    )
    assert equal.parent_errors == equal.child_errors
    assert equal.accepted
    assert not strict_equal.accepted


def test_live_native_conformance_rejects_runtime_semantic_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, corpora, _, _, _, _, _, child, artifact = _candidate()

    def forged_run(*args, **kwargs):
        del kwargs
        return model_generation_service.subprocess.CompletedProcess(
            args[0],
            0,
            (
                '{"artifact_id":"'
                + artifact.artifact_id
                + '","features":[],"prediction":0,"score":0}\n'
            ),
            "",
        )

    monkeypatch.setattr(model_generation_service.subprocess, "run", forged_run)
    with pytest.raises(ModelGenerationError, match="disagrees"):
        model_generation_service.verify_records_with_ptmrt(
            "ptmrt",
            "child.ptm",
            child,
            artifact,
            (corpora.promotion.records[0],),
        )


def test_disagreement_alone_is_not_a_drift_reopen_decision() -> None:
    _, corpora, parent, manifest, _, _, _, child, artifact = _candidate()
    _, _, live_corpus = _fixture_corpora()
    conformance = RuntimeConformanceReport(
        artifact.artifact_id, 8, 0, True, artifact.artifact_id
    )
    promotion = audit_parent_child(
        parent.snapshot(),
        manifest,
        child,
        corpora.promotion,
        conformance,
        PromotionAuditPolicy(8),
    )
    assert promotion.disagreements > 0
    with pytest.raises(ModelGenerationError, match="live/drift"):
        drift_requires_reopen(promotion, STRICT_DRIFT_POLICY)

    live = audit_parent_child(
        parent.snapshot(),
        manifest,
        child,
        live_corpus,
        conformance,
        PromotionAuditPolicy(1),
    )
    assert live.parent_errors == 4
    assert live.child_errors == 8
    assert drift_requires_reopen(live, STRICT_DRIFT_POLICY)

    one_regression = replace(
        live,
        observations=1,
        parent_errors=0,
        child_errors=1,
        disagreements=1,
        improvements=0,
        regressions=1,
        both_correct=0,
        both_wrong=0,
        parent_scores=(1,),
        child_scores=(-1,),
        conformance=replace(live.conformance, case_count=1),
        class_counts=(
            replace(
                live.class_counts[0],
                observed=1,
                both_correct=0,
                both_wrong=0,
                improvements=0,
                regressions=1,
            ),
            replace(
                live.class_counts[1],
                observed=0,
                both_correct=0,
                both_wrong=0,
                improvements=0,
                regressions=0,
            ),
        ),
    )
    assert not drift_requires_reopen(one_regression, STRICT_DRIFT_POLICY)


def test_lifecycle_corpora_reject_holdout_reuse() -> None:
    _, corpora, _ = _fixture_corpora()
    assert not hasattr(corpora, "live")
    reused = LabeledCorpus(
        corpora.promotion.dataset_id,
        CorpusRole.PROMOTION,
        tuple(
            CorpusExample(
                900 + index,
                example.record,
                example.label,
            )
            for index, example in enumerate(corpora.adaptation.examples)
        ),
    )
    with pytest.raises(ModelGenerationError, match="independent"):
        LifecycleCorpora(
            corpora.invention,
            corpora.adaptation,
            reused,
        )


def test_evidence_usage_round_trip_preserves_exact_spent_rows(
    tmp_path: Path,
) -> None:
    _, corpora, live = _fixture_corpora()
    subject = "sha256:" + "a" * 64
    usage = EvidenceUsage(
        EvidenceUsagePurpose.CANDIDATE_EPISODE,
        subject,
        (corpora.invention, corpora.adaptation, corpora.promotion),
    )
    assert EvidenceUsage.from_dict(usage.to_dict()) == usage
    store = ModelGenerationStore(tmp_path / "store")
    store.put_evidence_usage(usage)
    assert store.load_evidence_usage(usage.usage_id) == usage

    repeated = LabeledCorpus(
        live.dataset_id,
        CorpusRole.LIVE,
        (
            live.examples[0],
            CorpusExample(
                "same-row-new-observation",
                live.examples[0].record,
                live.examples[0].label,
            ),
        ),
    )
    with pytest.raises(ModelGenerationError, match="repeated observations"):
        EvidenceUsage(
            EvidenceUsagePurpose.LIVE_DRIFT,
            subject,
            (repeated,),
        )


def test_content_addressed_loaders_reject_valid_objects_at_wrong_addresses(
    tmp_path: Path,
) -> None:
    parent, manifest = _parent()
    first_snapshot = AdaptiveSnapshotEnvelope(parent.snapshot())
    second_snapshot = AdaptiveSnapshotEnvelope(
        extend_snapshot_features(parent.snapshot(), 1)
    )
    extended_catalog = manifest.build_catalog()
    extended_catalog.numeric_ge("temperature", 75)
    second_manifest = OrderedLiteralManifest.from_catalog(extended_catalog)
    store = ModelGenerationStore(tmp_path / "store")
    for value in (first_snapshot, second_snapshot):
        store.put_snapshot(value)
    for value in (manifest, second_manifest):
        store.put_manifest(value)

    snapshot_path = (
        store.root
        / "objects"
        / "snapshots"
        / f"{first_snapshot.snapshot_id[7:]}.json"
    )
    other_snapshot_path = (
        store.root
        / "objects"
        / "snapshots"
        / f"{second_snapshot.snapshot_id[7:]}.json"
    )
    snapshot_path.write_bytes(other_snapshot_path.read_bytes())
    with pytest.raises(ModelGenerationError, match="address"):
        store.load_snapshot(first_snapshot.snapshot_id)

    manifest_path = (
        store.root
        / "objects"
        / "literal-manifests"
        / f"{manifest.manifest_id[7:]}.json"
    )
    other_manifest_path = (
        store.root
        / "objects"
        / "literal-manifests"
        / f"{second_manifest.manifest_id[7:]}.json"
    )
    manifest_path.write_bytes(other_manifest_path.read_bytes())
    with pytest.raises(ModelGenerationError, match="address"):
        store.load_manifest(manifest.manifest_id)


def test_initial_parent_recovery_validates_the_complete_deployable_graph(
    tmp_path: Path,
) -> None:
    parent_training, _, _ = _fixture_corpora()
    parent, manifest = _parent()
    envelope = AdaptiveSnapshotEnvelope(parent.snapshot())
    preprocessing, artifact = compile_generation_artifact(
        parent.snapshot(),
        manifest,
        name="misbound initial parent",
        validation_records=parent_training.records,
        validation_signature={
            "generation_stage": "adapted_child",
            "training_corpus_digest": parent_training.digest,
        },
    )
    store = ModelGenerationStore(tmp_path / "store")
    preprocessing_id, _ = store.put_preprocessing(preprocessing)
    store.put_snapshot(envelope)
    store.put_manifest(manifest)
    store.put_artifact(artifact)
    generation = ModelGeneration(
        GenerationKind.TRAINED_PARENT,
        envelope.snapshot_id,
        manifest.manifest_id,
        preprocessing_id,
        artifact.artifact_id,
        None,
        None,
        ((CorpusRole.PARENT_TRAINING.value, parent_training.digest),),
    )
    store.put_generation(generation)
    parent_usage = EvidenceUsage(
        EvidenceUsagePurpose.PARENT_REGISTRATION,
        generation.generation_id,
        (parent_training,),
    )
    store.put_evidence_usage(parent_usage)
    # Simulate a legacy or externally assembled durable route that bypassed
    # current registration. Recovery must still validate the full graph.
    store.append_event(
        LifecycleEventKind.PARENT_REGISTERED,
        generation.generation_id,
        evidence_usage_id=parent_usage.usage_id,
    )

    with pytest.raises(ModelGenerationError, match="deployable generation"):
        ModelGenerationController(store)


def test_store_instances_share_one_process_local_event_lock(tmp_path: Path) -> None:
    first = ModelGenerationStore(tmp_path / "store")
    second = ModelGenerationStore(tmp_path / "store")

    assert first._event_lock is second._event_lock


@pytest.mark.skipif(
    not _has_gprolog() or _ptmrt_path() is None,
    reason="live GNU Prolog and a built ptmrt are required",
)
def test_live_trained_parent_loop_restores_bit_exact_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_training, corpora, live = _fixture_corpora()
    parent, manifest = _parent()
    original_snapshot = parent.snapshot()
    store = ModelGenerationStore(tmp_path / "generation-store")

    def fail_activated_telemetry(event) -> None:
        if event.kind == "activated":
            raise RuntimeError("injected telemetry sink failure")

    result = execute_trained_parent_lifecycle(
        parent_snapshot=original_snapshot,
        parent_manifest=manifest,
        parent_training_corpus=parent_training,
        corpora=corpora,
        numeric_field="temperature",
        adaptation_epochs=5,
        promotion_policy=PromotionAuditPolicy(8),
        store=store,
        ptmrt_executable=_ptmrt_path(),
        telemetry=TelemetrySession(),
        event_sink=fail_activated_telemetry,
    )

    assert result.conformance.exact
    assert result.extended_parent.materialized.reviewed.proposal.structure[
        "threshold"
    ] == 75.0
    assert result.invention_evidence.invention_corpus_digest == (
        corpora.invention.digest
    )
    assert isinstance(result.invention_evidence, PrologInventionEvidence)
    assert len(result.candidate_set.candidates) == 1
    assert result.promotion_audit.accepted
    assert result.promotion_audit.improvements == 4
    assert result.promotion_audit.regressions == 0
    assert result.extended_parent.equivalence_case_count == len(
        parent_training.examples
    )
    assert result.invention_evidence == result.controller.store.load_invention_evidence(
        result.invention_evidence.evidence_id
    )
    golden_child = PackedTMInferenceArtifact.from_bytes(
        bytes.fromhex(GOLDEN_CHILD_HEX.read_text(encoding="ascii"))
    )
    _assert_same_portable_artifact_contract(result.child_artifact, golden_child)
    validation_signature = result.child_artifact.manifest["validation"]["signature"]
    assert validation_signature["threshold_candidate_set_id"] == (
        result.candidate_set.candidate_set_id
    )
    assert validation_signature["threshold_candidate_selection_id"] == (
        result.candidate_selection.selection_id
    )
    assert result.controller.active_generation_id == result.child_generation.generation_id
    assert store.load_promotion_conformance(
        result.promotion_conformance_evidence.evidence_id
    ) == result.promotion_conformance_evidence
    assert isinstance(result.controller.last_telemetry_error, RuntimeError)
    assert str(result.controller.last_telemetry_error) == (
        "injected telemetry sink failure"
    )
    # A process restart derives routing from the durable event chain.
    assert ModelGenerationController(
        store, ptmrt_executable=_ptmrt_path()
    ).active_generation_id == (
        result.child_generation.generation_id
    )

    forged_evidence_root = tmp_path / "forged-evidence-reuse-store"
    shutil.copytree(store.root, forged_evidence_root)
    forged_evidence_store = ModelGenerationStore(forged_evidence_root)
    reused_as_live = LabeledCorpus(
        corpora.invention.dataset_id,
        CorpusRole.LIVE,
        tuple(
            CorpusExample(1700 + index, example.record, example.label)
            for index, example in enumerate(corpora.invention.examples)
        ),
    )
    forged_usage = EvidenceUsage(
        EvidenceUsagePurpose.LIVE_DRIFT,
        result.child_generation.generation_id,
        (reused_as_live,),
    )
    forged_evidence_store.put_evidence_usage(forged_usage)
    forged_evidence_store.append_event(
        LifecycleEventKind.EVIDENCE_RESERVED,
        result.child_generation.generation_id,
        evidence_usage_id=forged_usage.usage_id,
        purpose=forged_usage.purpose.value,
        dataset_id=forged_usage.dataset_id,
    )
    with pytest.raises(ModelGenerationError, match="fingerprint"):
        ModelGenerationController(
            forged_evidence_store, ptmrt_executable=_ptmrt_path()
        )

    with pytest.raises(ModelGenerationError, match="reopen request"):
        result.controller.restore_parent(result.restoration_bundle)

    def prove_promotion_replay_runs_native(*args, **kwargs):
        del args, kwargs
        raise ModelGenerationError("injected promotion native replay")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            model_generation_service,
            "_verify_snapshot_records_with_ptmrt",
            prove_promotion_replay_runs_native,
        )
        with pytest.raises(ModelGenerationError, match="promotion native replay"):
            result.controller.record_candidate(result.lineage)

    self_attested_lineage = replace(
        result.lineage,
        promotion_conformance_evidence_id="sha256:" + "f" * 64,
    )
    store.put_lineage(self_attested_lineage)
    with pytest.raises(
        ModelGenerationError, match="promotion conformance evidence is unavailable"
    ):
        result.controller.record_candidate(self_attested_lineage)

    with pytest.raises(ModelGenerationError, match="active generation"):
        result.controller.record_candidate(result.lineage)
    with pytest.raises(ModelGenerationError, match="current state"):
        result.controller.approve_promotion(
            result.lineage, result.promotion_audit
        )
    assert store.read_events()[-1].kind is LifecycleEventKind.ACTIVATED

    overlapping_live = LabeledCorpus(
        live.dataset_id,
        CorpusRole.LIVE,
        (
            CorpusExample(
                parent_training.examples[0].example_id,
                live.examples[0].record,
                live.examples[0].label,
            ),
        ),
    )
    with pytest.raises(ModelGenerationError, match="overlap"):
        reopen_and_restore_for_drift(
            result, overlapping_live, STRICT_DRIFT_POLICY
        )

    forged_lineage = replace(
        result.lineage,
        invented_literal_id=result.lineage.invented_literal_id ^ 1,
    )
    result.controller.store.put_lineage(forged_lineage)
    with pytest.raises(
        ModelGenerationError, match="object graph|representation extension"
    ):
        result.controller.record_candidate(forged_lineage)

    alternate_states = [list(row) for row in result.child.snapshot.snapshot.states]
    original_state = alternate_states[0][0]
    action_boundary = result.child.snapshot.snapshot.states_per_action
    if original_state < action_boundary or (
        original_state == action_boundary + 1
    ):
        alternate_states[0][0] = original_state + 1
    else:
        alternate_states[0][0] = original_state - 1
    alternate_snapshot = replace(
        result.child.snapshot.snapshot,
        states=tuple(tuple(row) for row in alternate_states),
    )
    _, alternate_artifact = compile_generation_artifact(
        alternate_snapshot,
        result.child.manifest,
        name="same-mask alternate adaptive child",
        validation_records=corpora.promotion.records,
        validation_signature={
            "generation_stage": "adapted_child",
            "invention_corpus_digest": corpora.invention.digest,
            "adaptation_corpus_digest": corpora.adaptation.digest,
            "promotion_corpus_digest": corpora.promotion.digest,
            "origin_proposal_semantic_id": (
                result.lineage.origin_proposal_semantic_id
            ),
            "origin_proposal_provenance_id": (
                result.lineage.origin_proposal_provenance_id
            ),
        },
        restoration_reference=result.restoration_bundle.to_dict(),
    )
    alternate_path = result.controller.store.put_artifact(alternate_artifact)
    verified_alternate = verify_artifact_with_ptmrt(
        _ptmrt_path(), alternate_path, alternate_artifact.artifact_id
    )
    alternate_conformance = audit_runtime_conformance(
        result.child,
        alternate_artifact,
        corpora.promotion.records,
        ptmrt_verified=True,
        ptmrt_artifact_id=verified_alternate,
    )
    assert alternate_conformance.exact
    alternate_audit = audit_parent_child(
        original_snapshot,
        manifest,
        result.child,
        corpora.promotion,
        alternate_conformance,
        PromotionAuditPolicy(8),
    )
    result.controller.store.put_audit(alternate_audit)
    alternate_generation = replace(
        result.child_generation,
        inference_artifact_id=alternate_artifact.artifact_id,
    )
    result.controller.store.put_generation(alternate_generation)
    alternate_lineage = replace(
        result.lineage,
        child_generation_id=alternate_generation.generation_id,
        promotion_audit_id=alternate_audit.audit_id,
    )
    result.controller.store.put_lineage(alternate_lineage)
    with pytest.raises(ModelGenerationError, match="object graph"):
        result.controller.record_candidate(alternate_lineage)

    unauthorized_restore_root = tmp_path / "unauthorized-restore-store"
    shutil.copytree(store.root, unauthorized_restore_root)
    unauthorized_restore_store = ModelGenerationStore(unauthorized_restore_root)
    unauthorized_restore_store.append_event(
        LifecycleEventKind.PARENT_RESTORED,
        result.parent_generation.generation_id,
        restoration_bundle_id=result.restoration_bundle.bundle_id,
        previous_generation_id=result.child_generation.generation_id,
        lineage_id=result.lineage.lineage_id,
        drift_audit_id=result.promotion_audit.audit_id,
    )
    with pytest.raises(ModelGenerationError, match="reopen request"):
        ModelGenerationController(
            unauthorized_restore_store, ptmrt_executable=_ptmrt_path()
        )

    forged_reopen_root = tmp_path / "forged-reopen-store"
    shutil.copytree(store.root, forged_reopen_root)
    forged_reopen_store = ModelGenerationStore(forged_reopen_root)
    forged_conformance = RuntimeConformanceReport(
        result.child_artifact.artifact_id,
        len(live.examples),
        0,
        True,
        result.child_artifact.artifact_id,
    )
    forged_drift = audit_parent_child(
        original_snapshot,
        manifest,
        result.child,
        live,
        forged_conformance,
        PromotionAuditPolicy(1),
    )
    assert drift_requires_reopen(forged_drift, STRICT_DRIFT_POLICY)
    forged_live_usage = EvidenceUsage(
        EvidenceUsagePurpose.LIVE_DRIFT,
        result.child_generation.generation_id,
        (live,),
    )
    forged_reopen_store.put_evidence_usage(forged_live_usage)
    forged_reopen_store.put_audit(forged_drift)
    forged_reopen_store.append_event(
        LifecycleEventKind.EVIDENCE_RESERVED,
        result.child_generation.generation_id,
        evidence_usage_id=forged_live_usage.usage_id,
        purpose=forged_live_usage.purpose.value,
        dataset_id=forged_live_usage.dataset_id,
    )
    forged_reopen_store.append_event(
        LifecycleEventKind.REOPEN_REQUESTED,
        result.child_generation.generation_id,
        drift_audit_id=forged_drift.audit_id,
        lineage_id=result.lineage.lineage_id,
        restoration_bundle_id=result.restoration_bundle.bundle_id,
        parent_errors=forged_drift.parent_errors,
        child_errors=forged_drift.child_errors,
        drift_policy=STRICT_DRIFT_POLICY.to_dict(),
        live_conformance_evidence_id="sha256:" + "f" * 64,
        evidence_usage_id=forged_live_usage.usage_id,
    )
    with pytest.raises(ModelGenerationError, match="conformance evidence"):
        ModelGenerationController(
            forged_reopen_store, ptmrt_executable=_ptmrt_path()
        )

    forged_receipt_root = tmp_path / "forged-receipt-store"
    shutil.copytree(store.root, forged_receipt_root)
    forged_receipt_store = ModelGenerationStore(forged_receipt_root)
    child_batch = result.child.manifest.build_catalog().encode(live.records).ta
    child_rows = tuple(
        child_batch.row_values(index) for index in range(child_batch.row_count)
    )
    forged_machine = ScalarBinaryTsetlinMachine(
        result.child.snapshot.snapshot.number_of_clauses,
        result.child.snapshot.snapshot.number_of_features,
        states_per_action=result.child.snapshot.snapshot.states_per_action,
        specificity=result.child.snapshot.snapshot.specificity,
        threshold=result.child.snapshot.snapshot.threshold,
        seed=0,
    )
    forged_machine.restore(result.child.snapshot.snapshot)
    forged_scores = tuple(forged_machine.score(row) for row in child_rows)
    forged_predictions = tuple(int(score > 0) for score in forged_scores)
    forged_receipt = LiveRuntimeConformanceEvidence(
        result.child_generation.generation_id,
        result.child_artifact.artifact_id,
        result.child.snapshot.snapshot_id,
        result.child.manifest.manifest_id,
        live,
        child_rows,
        forged_scores,
        forged_predictions,
        result.child_artifact.predict_records(live.records),
        tuple(tuple(int(value) for value in row) for row in child_rows),
        forged_scores,
        forged_predictions,
        "sha256:" + "0" * 64,
    )
    forged_receipt_store.put_live_conformance(forged_receipt)
    forged_receipt_store.put_evidence_usage(forged_live_usage)
    forged_receipt_store.put_audit(forged_drift)
    forged_receipt_store.append_event(
        LifecycleEventKind.EVIDENCE_RESERVED,
        result.child_generation.generation_id,
        evidence_usage_id=forged_live_usage.usage_id,
        purpose=forged_live_usage.purpose.value,
        dataset_id=forged_live_usage.dataset_id,
    )
    forged_receipt_store.append_event(
        LifecycleEventKind.REOPEN_REQUESTED,
        result.child_generation.generation_id,
        drift_audit_id=forged_drift.audit_id,
        lineage_id=result.lineage.lineage_id,
        restoration_bundle_id=result.restoration_bundle.bundle_id,
        parent_errors=forged_drift.parent_errors,
        child_errors=forged_drift.child_errors,
        drift_policy=STRICT_DRIFT_POLICY.to_dict(),
        live_conformance_evidence_id=forged_receipt.evidence_id,
        evidence_usage_id=forged_live_usage.usage_id,
    )
    with pytest.raises(ModelGenerationError, match="executable digest"):
        ModelGenerationController(
            forged_receipt_store, ptmrt_executable=_ptmrt_path()
        )

    failed_live = _corpus(
        CorpusRole.LIVE,
        450,
        (45, 46, 47, 48, 102, 103, 104, 105),
        (1, 1, 1, 1, 0, 0, 0, 0),
    )
    real_native_verification = (
        model_generation_service._verify_snapshot_records_with_ptmrt
    )

    def block_live_native_verification(*args, **kwargs):
        records = args[5] if len(args) > 5 else kwargs["records"]
        if tuple(records) == failed_live.records:
            raise ModelGenerationError(
                "injected native verification requirement"
            )
        return real_native_verification(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            model_generation_service,
            "_verify_snapshot_records_with_ptmrt",
            block_live_native_verification,
        )
        with pytest.raises(ModelGenerationError, match="native verification"):
            result.controller.request_reopen(
                result.child_generation.generation_id,
                failed_live,
                STRICT_DRIFT_POLICY,
                _ptmrt_path(),
            )
    failed_events = store.read_events()
    assert failed_events[-2].kind is LifecycleEventKind.EVIDENCE_RESERVED
    assert failed_events[-1].kind is LifecycleEventKind.EVIDENCE_ABANDONED
    assert failed_events[-1].details["evidence_usage_id"] == (
        failed_events[-2].details["evidence_usage_id"]
    )

    drift, restored = reopen_and_restore_for_drift(
        result, live, STRICT_DRIFT_POLICY
    )
    assert drift.child_errors > drift.parent_errors
    assert drift.conformance.exact
    assert drift.conformance.case_count == len(live.examples)
    reopen_event = store.read_events()[-2]
    live_evidence = store.load_live_conformance(
        reopen_event.details["live_conformance_evidence_id"]
    )
    assert live_evidence.corpus == live
    assert live_evidence.corpus.digest == drift.corpus_digest
    assert live_evidence.scalar_scores == live_evidence.native_scores
    assert restored.snapshot.snapshot == original_snapshot
    assert restored.manifest == manifest
    with pytest.raises(ModelGenerationError, match="already been activated"):
        result.controller.record_candidate(result.lineage)
    assert result.controller.active_generation_id == result.parent_generation.generation_id
    assert store.read_events()[-1].kind is LifecycleEventKind.PARENT_RESTORED
    original_rows = manifest.build_catalog().encode(parent_training.records).ta
    rows = tuple(
        original_rows.row_values(index) for index in range(original_rows.row_count)
    )
    assert restored.machine.predict(rows) == parent.predict(rows)
    assert restored.artifact.predict_records(parent_training.records) == tuple(
        parent.predict(rows)
    )

    replay = ScalarBinaryTsetlinMachine(
        original_snapshot.number_of_clauses,
        original_snapshot.number_of_features,
        states_per_action=original_snapshot.states_per_action,
        specificity=original_snapshot.specificity,
        threshold=original_snapshot.threshold,
        seed=0,
    )
    replay.restore(original_snapshot)
    replay.update(rows[0], parent_training.labels[0])
    restored.machine.update(rows[0], parent_training.labels[0])
    assert restored.machine.snapshot() == replay.snapshot()
    assert ModelGenerationController(
        store, ptmrt_executable=_ptmrt_path()
    ).active_generation_id == (
        result.parent_generation.generation_id
    )

    orphan_root = tmp_path / "orphaned-candidate-evidence-store"
    shutil.copytree(store.root, orphan_root)
    orphan_store = ModelGenerationStore(orphan_root)
    orphan_controller = ModelGenerationController(
        orphan_store, ptmrt_executable=_ptmrt_path()
    )
    orphan_corpora = LifecycleCorpora(
        _corpus(CorpusRole.INVENTION, 1800, (10, 20, 30, 40), (0, 0, 1, 1)),
        _corpus(
            CorpusRole.ADAPTATION,
            1900,
            (11, 12, 13, 14, 31, 32, 33, 34),
            (0, 0, 0, 0, 1, 1, 1, 1),
        ),
        _corpus(
            CorpusRole.PROMOTION,
            2000,
            (15, 16, 17, 18, 35, 36, 37, 38),
            (0, 0, 0, 0, 1, 1, 1, 1),
        ),
    )
    orphan_usage, _, _ = orphan_controller.reserve_candidate_evidence(
        orphan_corpora
    )
    assert orphan_store.read_events()[-1].kind is (
        LifecycleEventKind.EVIDENCE_RESERVED
    )
    recovered_orphan = ModelGenerationController(
        orphan_store, ptmrt_executable=_ptmrt_path()
    )
    assert recovered_orphan.active_generation_id == (
        result.parent_generation.generation_id
    )
    orphan_events = orphan_store.read_events()
    assert orphan_events[-1].kind is LifecycleEventKind.EVIDENCE_ABANDONED
    assert orphan_events[-1].details["evidence_usage_id"] == orphan_usage.usage_id

    def package_candidate(
        adapted_child,
        promotion_corpus: LabeledCorpus,
        *,
        name: str,
        claimed_adaptation_corpus: LabeledCorpus | None = None,
        require_accepted: bool | None = True,
    ) -> tuple[
        ModelGeneration,
        ModelGenerationLineage,
        PromotionAuditSnapshot,
    ]:
        preprocessing_id = result.child_generation.preprocessing_contract_id
        adaptation_corpus = (
            corpora.adaptation
            if claimed_adaptation_corpus is None
            else claimed_adaptation_corpus
        )
        adaptation_digest = adaptation_corpus.digest
        behavior = AdaptiveBehaviorIdentity.from_child(
            adapted_child,
            preprocessing_contract_id=preprocessing_id,
        )
        selected_candidate = next(
            candidate
            for candidate in result.candidate_set.candidates
            if candidate.proposal_semantic_id
            == result.lineage.origin_proposal_semantic_id
        )
        outcome = model_generation_service._threshold_candidate_outcome(
            original_snapshot,
            manifest,
            result.extended_parent.materialized.reviewed.proposal,
            result.extended_parent.materialized.reviewed,
            result.extended_parent,
            adapted_child,
            preprocessing_id,
            adaptation_corpus,
        )
        selection = ThresholdCandidateSelection(
            candidate_set_id=result.candidate_set.candidate_set_id,
            parent_generation_id=result.parent_generation.generation_id,
            parent_snapshot_id=result.parent_generation.snapshot_id,
            parent_manifest_id=result.parent_generation.literal_manifest_id,
            adaptation_corpus_digest=adaptation_digest,
            adaptation_epochs=adapted_child.epochs,
            policy=ThresholdCandidateSelectionPolicy(
                minimum_observations=len(adaptation_corpus.examples),
                require_strict_improvement=False,
            ),
            outcomes=(outcome,),
            selected_proposal_semantic_id=(
                selected_candidate.proposal_semantic_id
            ),
            selected_proposal_provenance_id=(
                selected_candidate.proposal_provenance_id
            ),
        )
        store.put_threshold_candidate_selection(selection)
        preprocessing, artifact = compile_generation_artifact(
            adapted_child.snapshot.snapshot,
            adapted_child.manifest,
            name=name,
            validation_records=promotion_corpus.records,
            validation_signature={
                "generation_stage": "adapted_child",
                "adaptive_behavior_id": behavior.behavior_id,
                "invention_corpus_digest": corpora.invention.digest,
                "adaptation_corpus_digest": (
                    adaptation_digest
                ),
                "promotion_corpus_digest": promotion_corpus.digest,
                "origin_proposal_semantic_id": (
                    result.lineage.origin_proposal_semantic_id
                ),
                "origin_proposal_provenance_id": (
                    result.lineage.origin_proposal_provenance_id
                ),
                "threshold_candidate_set_id": (
                    result.candidate_set.candidate_set_id
                ),
                "threshold_candidate_selection_id": selection.selection_id,
            },
            restoration_reference=result.restoration_bundle.to_dict(),
        )
        stored_preprocessing_id, _ = store.put_preprocessing(preprocessing)
        assert stored_preprocessing_id == preprocessing_id
        store.put_snapshot(adapted_child.snapshot)
        store.put_manifest(adapted_child.manifest)
        artifact_path = store.put_artifact(artifact)
        generation = ModelGeneration(
            GenerationKind.ADAPTED_CHILD,
            adapted_child.snapshot.snapshot_id,
            adapted_child.manifest.manifest_id,
            preprocessing_id,
            artifact.artifact_id,
            result.extended_generation.generation_id,
            result.restoration_bundle.bundle_id,
            (
                (CorpusRole.INVENTION.value, corpora.invention.digest),
                (
                    CorpusRole.ADAPTATION.value,
                    adaptation_digest,
                ),
                (CorpusRole.PROMOTION.value, promotion_corpus.digest),
            ),
            result.lineage.origin_proposal_semantic_id,
            result.lineage.origin_proposal_provenance_id,
        )
        store.put_generation(generation)
        verified_id = verify_artifact_with_ptmrt(
            _ptmrt_path(), artifact_path, artifact.artifact_id
        )
        promotion_vectors = (
            model_generation_service._verify_snapshot_records_with_ptmrt(
                _ptmrt_path(),
                artifact_path,
                adapted_child.snapshot,
                adapted_child.manifest,
                artifact,
                promotion_corpus.records,
            )
        )
        promotion_evidence = (
            model_generation_service._promotion_conformance_evidence(
                generation,
                promotion_corpus,
                promotion_vectors,
                _ptmrt_path(),
            )
        )
        store.put_promotion_conformance(promotion_evidence)
        conformance = audit_runtime_conformance(
            adapted_child,
            artifact,
            promotion_corpus.records,
            ptmrt_verified=True,
            ptmrt_artifact_id=verified_id,
        )
        audit = audit_parent_child(
            original_snapshot,
            manifest,
            adapted_child,
            promotion_corpus,
            conformance,
            PromotionAuditPolicy(len(promotion_corpus.examples)),
        )
        if require_accepted is not None:
            assert audit.accepted is require_accepted
        store.put_audit(audit)
        usage = EvidenceUsage(
            EvidenceUsagePurpose.CANDIDATE_EPISODE,
            result.parent_generation.generation_id,
            (corpora.invention, adaptation_corpus, promotion_corpus),
        )
        store.put_evidence_usage(usage)
        lineage = ModelGenerationLineage(
            parent_generation_id=result.parent_generation.generation_id,
            extended_generation_id=result.extended_generation.generation_id,
            child_generation_id=generation.generation_id,
            adaptive_behavior_id=behavior.behavior_id,
            restoration_bundle_id=result.restoration_bundle.bundle_id,
            promotion_audit_id=audit.audit_id,
            promotion_conformance_evidence_id=promotion_evidence.evidence_id,
            invention_evidence_id=result.candidate_set.candidate_set_id,
            evidence_usage_id=usage.usage_id,
            activation_sequence=2,
            previous_activated_lineage_id=result.lineage.lineage_id,
            invented_literal_id=result.lineage.invented_literal_id,
            invention_corpus_digest=corpora.invention.digest,
            adaptation_corpus_digest=adaptation_digest,
            promotion_corpus_digest=promotion_corpus.digest,
            origin_proposal_semantic_id=(
                result.lineage.origin_proposal_semantic_id
            ),
            origin_proposal_provenance_id=(
                result.lineage.origin_proposal_provenance_id
            ),
            candidate_selection_id=selection.selection_id,
            schema=model_generation_service.LINEAGE_SCHEMA,
        )
        store.put_lineage(lineage)
        return generation, lineage, audit

    repackaged_promotion = _corpus(
        CorpusRole.PROMOTION,
        500,
        (60, 67, 70, 74, 77, 80, 90, 96),
        (0, 0, 0, 0, 1, 1, 1, 1),
    )
    forged_adaptation_corpus = _corpus(
        CorpusRole.ADAPTATION,
        600,
        (55, 57, 65, 68, 82, 85, 93, 101),
        (0, 0, 0, 0, 1, 1, 1, 1),
    )
    assert forged_adaptation_corpus.digest != result.child.adaptation_corpus_digest
    repackaged_generation, repackaged_lineage, repackaged_audit = package_candidate(
        result.child,
        repackaged_promotion,
        name="same adaptive behavior with fresh promotion evidence",
        claimed_adaptation_corpus=forged_adaptation_corpus,
    )
    assert repackaged_generation.generation_id != result.child_generation.generation_id
    assert repackaged_lineage.adaptive_behavior_id == result.lineage.adaptive_behavior_id
    with pytest.raises(ModelGenerationError, match="adaptive behavior"):
        result.controller.record_candidate(repackaged_lineage)

    raw_approval_root = tmp_path / "raw-approval-live-store"
    shutil.copytree(store.root, raw_approval_root)
    raw_approval_store = ModelGenerationStore(raw_approval_root)
    raw_approval_controller = ModelGenerationController(
        raw_approval_store, ptmrt_executable=_ptmrt_path()
    )
    raw_approval_store.append_event(
        LifecycleEventKind.PROMOTION_APPROVED,
        repackaged_lineage.child_generation_id,
        lineage_id=repackaged_lineage.lineage_id,
        audit_id=repackaged_audit.audit_id,
    )
    with pytest.raises(ModelGenerationError, match="promotion approval"):
        raw_approval_controller.activate_child(
            repackaged_lineage, repackaged_audit
        )
    with pytest.raises(ModelGenerationError, match="promotion approval"):
        ModelGenerationController(
            raw_approval_store, ptmrt_executable=_ptmrt_path()
        )

    reused_evidence_child = adapt_extended_parent(
        result.extended_parent,
        forged_adaptation_corpus,
        epochs=6,
    )
    _, reused_evidence_lineage, _ = package_candidate(
        reused_evidence_child,
        repackaged_promotion,
        name="new behavior packaged with already-spent invention evidence",
        claimed_adaptation_corpus=forged_adaptation_corpus,
        require_accepted=None,
    )
    assert reused_evidence_lineage.adaptive_behavior_id != (
        result.lineage.adaptive_behavior_id
    )
    reused_evidence_usage = store.load_evidence_usage(
        reused_evidence_lineage.evidence_usage_id
    )

    raw_reuse_root = tmp_path / "raw-reuse-live-store"
    shutil.copytree(store.root, raw_reuse_root)
    raw_reuse_store = ModelGenerationStore(raw_reuse_root)
    raw_reuse_controller = ModelGenerationController(
        raw_reuse_store, ptmrt_executable=_ptmrt_path()
    )
    raw_reuse_store.append_event(
        LifecycleEventKind.EVIDENCE_RESERVED,
        result.parent_generation.generation_id,
        evidence_usage_id=reused_evidence_usage.usage_id,
        purpose=reused_evidence_usage.purpose.value,
        dataset_id=reused_evidence_usage.dataset_id,
    )
    with pytest.raises(
        ModelGenerationError, match="observation identity|fingerprint"
    ):
        raw_reuse_controller.record_candidate(reused_evidence_lineage)
    with pytest.raises(
        ModelGenerationError, match="observation identity|fingerprint"
    ):
        ModelGenerationController(
            raw_reuse_store, ptmrt_executable=_ptmrt_path()
        )

    raw_misbound_root = tmp_path / "raw-misbound-live-store"
    shutil.copytree(store.root, raw_misbound_root)
    raw_misbound_store = ModelGenerationStore(raw_misbound_root)
    raw_misbound_controller = ModelGenerationController(
        raw_misbound_store, ptmrt_executable=_ptmrt_path()
    )
    raw_misbound_store.append_event(
        LifecycleEventKind.EVIDENCE_RESERVED,
        reused_evidence_lineage.child_generation_id,
        evidence_usage_id=reused_evidence_usage.usage_id,
        purpose=reused_evidence_usage.purpose.value,
        dataset_id=reused_evidence_usage.dataset_id,
    )
    with pytest.raises(ModelGenerationError, match="evidence-reserved"):
        raw_misbound_controller.record_candidate(reused_evidence_lineage)
    with pytest.raises(ModelGenerationError, match="evidence-reserved"):
        ModelGenerationController(
            raw_misbound_store, ptmrt_executable=_ptmrt_path()
        )

    rejected_corpora = LifecycleCorpora(
        _corpus(
            CorpusRole.INVENTION,
            800,
            (110, 120, 130, 140),
            (0, 0, 1, 1),
        ),
        _corpus(
            CorpusRole.ADAPTATION,
            900,
            (111, 112, 113, 114, 131, 132, 133, 134),
            (0, 0, 0, 0, 1, 1, 1, 1),
        ),
        _corpus(
            CorpusRole.PROMOTION,
            1000,
            (115, 116, 117, 118, 135, 136, 137, 138),
            (1, 1, 1, 1, 0, 0, 0, 0),
        ),
    )
    with pytest.raises(ModelGenerationError, match="promotion policy rejected"):
        execute_trained_parent_lifecycle(
            parent_snapshot=original_snapshot,
            parent_manifest=manifest,
            parent_training_corpus=parent_training,
            corpora=rejected_corpora,
            numeric_field="temperature",
            adaptation_epochs=6,
            promotion_policy=PromotionAuditPolicy(8),
            store=store,
            ptmrt_executable=_ptmrt_path(),
        )
    rejected_event = store.read_events()[-1]
    assert rejected_event.kind is LifecycleEventKind.CANDIDATE_REJECTED
    rejected_lineage = store.load_lineage(rejected_event.details["lineage_id"])
    rejected_audit = store.load_audit(rejected_event.details["audit_id"])
    rejected_recovery_root = tmp_path / "rejected-activation-store"
    shutil.copytree(store.root, rejected_recovery_root)
    rejected_recovery_store = ModelGenerationStore(rejected_recovery_root)
    rejected_recovery_store.append_event(
        LifecycleEventKind.ACTIVATED,
        rejected_lineage.child_generation_id,
        previous_generation_id=result.parent_generation.generation_id,
        lineage_id=rejected_lineage.lineage_id,
        adaptive_behavior_id=rejected_lineage.adaptive_behavior_id,
        audit_id=rejected_audit.audit_id,
    )
    with pytest.raises(ModelGenerationError, match="promotion approval"):
        ModelGenerationController(
            rejected_recovery_store, ptmrt_executable=_ptmrt_path()
        )

    reused_invention = LifecycleCorpora(
        corpora.invention,
        _corpus(
            CorpusRole.ADAPTATION,
            1100,
            (141, 142, 143, 144, 151, 152, 153, 154),
            (0, 0, 0, 0, 1, 1, 1, 1),
        ),
        _corpus(
            CorpusRole.PROMOTION,
            1200,
            (145, 146, 147, 148, 155, 156, 157, 158),
            (0, 0, 0, 0, 1, 1, 1, 1),
        ),
    )
    with pytest.raises(ModelGenerationError, match="already been used"):
        execute_trained_parent_lifecycle(
            parent_snapshot=original_snapshot,
            parent_manifest=manifest,
            parent_training_corpus=parent_training,
            corpora=reused_invention,
            numeric_field="temperature",
            adaptation_epochs=6,
            promotion_policy=PromotionAuditPolicy(8),
            store=store,
            ptmrt_executable=_ptmrt_path(),
        )

    cloned_invention = LabeledCorpus(
        corpora.invention.dataset_id,
        CorpusRole.INVENTION,
        tuple(
            CorpusExample(1300 + index, example.record, example.label)
            for index, example in enumerate(corpora.invention.examples)
        ),
    )
    cloned_rows = LifecycleCorpora(
        cloned_invention,
        reused_invention.adaptation,
        reused_invention.promotion,
    )
    with pytest.raises(ModelGenerationError, match="fingerprint"):
        execute_trained_parent_lifecycle(
            parent_snapshot=original_snapshot,
            parent_manifest=manifest,
            parent_training_corpus=parent_training,
            corpora=cloned_rows,
            numeric_field="temperature",
            adaptation_epochs=6,
            promotion_policy=PromotionAuditPolicy(8),
            store=store,
            ptmrt_executable=_ptmrt_path(),
        )

    second_corpora = LifecycleCorpora(
        _corpus(
            CorpusRole.INVENTION,
            1400,
            (53, 70, 80, 97),
            (0, 0, 1, 1),
        ),
        _corpus(
            CorpusRole.ADAPTATION,
            1500,
            (51, 52, 54, 57, 75, 84, 91, 96),
            (0, 0, 0, 0, 1, 1, 1, 1),
        ),
        _corpus(
            CorpusRole.PROMOTION,
            1600,
            (39, 40, 41, 42, 106, 107, 108, 109),
            (0, 0, 0, 0, 1, 1, 1, 1),
        ),
    )
    second = execute_trained_parent_lifecycle(
        parent_snapshot=original_snapshot,
        parent_manifest=manifest,
        parent_training_corpus=parent_training,
        corpora=second_corpora,
        numeric_field="temperature",
        adaptation_epochs=6,
        promotion_policy=PromotionAuditPolicy(8),
        store=store,
        ptmrt_executable=_ptmrt_path(),
    )
    assert second.lineage.activation_sequence == 2
    assert second.lineage.previous_activated_lineage_id == result.lineage.lineage_id
    assert second.lineage.adaptive_behavior_id != result.lineage.adaptive_behavior_id
    assert second.controller.active_generation_id == second.child_generation.generation_id
    assert [event.kind.value for event in store.read_events()] == [
        "parent_registered",
        "evidence_reserved",
        "candidate_created",
        "promotion_approved",
        "activated",
        "evidence_reserved",
        "evidence_abandoned",
        "evidence_reserved",
        "reopen_requested",
        "parent_restored",
        "evidence_reserved",
        "candidate_created",
        "candidate_rejected",
        "evidence_reserved",
        "candidate_created",
        "promotion_approved",
        "activated",
    ]
    assert ModelGenerationController(
        store, ptmrt_executable=_ptmrt_path()
    ).active_generation_id == (
        second.child_generation.generation_id
    )


def test_event_log_tampering_fails_closed(tmp_path: Path) -> None:
    store = ModelGenerationStore(tmp_path / "store")
    identifier = "sha256:" + "1" * 64
    store.append_event(
        # No object is required to test the independent event-chain envelope.
        LifecycleEventKind.PARENT_REGISTERED,
        identifier,
    )
    data = store.event_log_path.read_bytes()
    store.event_log_path.write_bytes(data.replace(b"parent_registered", b"parent_tampered__"))
    with pytest.raises(ModelGenerationError):
        store.read_events()


def test_event_head_detects_complete_tail_truncation(tmp_path: Path) -> None:
    store = ModelGenerationStore(tmp_path / "store")
    parent_id = "sha256:" + "1" * 64
    child_id = "sha256:" + "2" * 64
    store.append_event(LifecycleEventKind.PARENT_REGISTERED, parent_id)
    store.append_event(LifecycleEventKind.CANDIDATE_CREATED, child_id)
    complete = store.event_log_path.read_bytes()

    store.event_log_path.write_bytes(complete.splitlines(keepends=True)[0])
    with pytest.raises(ModelGenerationError, match="head"):
        store.read_events()

    store.event_log_path.write_bytes(b"")
    with pytest.raises(ModelGenerationError, match="head"):
        store.read_events()


def test_event_head_publication_failure_restores_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ModelGenerationStore(tmp_path / "store")
    parent_id = "sha256:" + "1" * 64
    child_id = "sha256:" + "2" * 64
    store.append_event(LifecycleEventKind.PARENT_REGISTERED, parent_id)
    previous = store.read_events()
    publish = model_generation_service.publish_bytes
    fail_head_once = True

    def injected_publish(path, data, *, overwrite):
        nonlocal fail_head_once
        if Path(path) == store.event_head_path and fail_head_once:
            fail_head_once = False
            raise OSError("injected event-head failure")
        return publish(path, data, overwrite=overwrite)

    monkeypatch.setattr(model_generation_service, "publish_bytes", injected_publish)
    with pytest.raises(OSError, match="injected event-head failure"):
        store.append_event(LifecycleEventKind.CANDIDATE_CREATED, child_id)

    assert store.read_events() == previous


@pytest.mark.skipif(
    not _has_gprolog() or _ptmrt_path() is None,
    reason="live GNU Prolog and a built ptmrt are required",
)
def test_failed_activation_is_completed_idempotently_during_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_training, corpora, _ = _fixture_corpora()
    parent, manifest = _parent()
    store = ModelGenerationStore(tmp_path / "generation-store")
    append_event = store.append_event

    def fail_activation(kind, generation_id, **details):
        if kind is LifecycleEventKind.ACTIVATED:
            raise OSError("injected durable activation failure")
        return append_event(kind, generation_id, **details)

    monkeypatch.setattr(store, "append_event", fail_activation)
    with pytest.raises(OSError, match="injected"):
        execute_trained_parent_lifecycle(
            parent_snapshot=parent.snapshot(),
            parent_manifest=manifest,
            parent_training_corpus=parent_training,
            corpora=corpora,
            numeric_field="temperature",
            adaptation_epochs=5,
            promotion_policy=PromotionAuditPolicy(8),
            store=store,
            ptmrt_executable=_ptmrt_path(),
        )

    events = store.read_events()
    assert events[-1].kind is LifecycleEventKind.PROMOTION_APPROVED
    child_id = events[-1].generation_id
    recovered_store = ModelGenerationStore(store.root)
    controller = ModelGenerationController(
        recovered_store, ptmrt_executable=_ptmrt_path()
    )
    assert controller.active_generation_id == child_id
    assert recovered_store.read_events()[-1].kind is LifecycleEventKind.ACTIVATED


@pytest.mark.skipif(
    not _has_gprolog() or _ptmrt_path() is None,
    reason="live GNU Prolog and a built ptmrt are required",
)
def test_candidate_created_is_completed_idempotently_during_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_training, corpora, _ = _fixture_corpora()
    parent, manifest = _parent()
    store = ModelGenerationStore(tmp_path / "candidate-recovery-store")
    append_event = store.append_event

    def fail_approval(kind, generation_id, **details):
        if kind is LifecycleEventKind.PROMOTION_APPROVED:
            raise OSError("injected durable approval failure")
        return append_event(kind, generation_id, **details)

    monkeypatch.setattr(store, "append_event", fail_approval)
    with pytest.raises(OSError, match="approval failure"):
        execute_trained_parent_lifecycle(
            parent_snapshot=parent.snapshot(),
            parent_manifest=manifest,
            parent_training_corpus=parent_training,
            corpora=corpora,
            numeric_field="temperature",
            adaptation_epochs=5,
            promotion_policy=PromotionAuditPolicy(8),
            store=store,
            ptmrt_executable=_ptmrt_path(),
        )

    events = store.read_events()
    assert events[-1].kind is LifecycleEventKind.CANDIDATE_CREATED
    child_id = events[-1].generation_id
    recovered_store = ModelGenerationStore(store.root)
    controller = ModelGenerationController(
        recovered_store, ptmrt_executable=_ptmrt_path()
    )
    assert controller.active_generation_id == child_id
    assert [event.kind for event in recovered_store.read_events()[-2:]] == [
        LifecycleEventKind.PROMOTION_APPROVED,
        LifecycleEventKind.ACTIVATED,
    ]


@pytest.mark.skipif(
    not _has_gprolog() or _ptmrt_path() is None,
    reason="live GNU Prolog and a built ptmrt are required",
)
def test_reopen_requested_is_completed_idempotently_during_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_training, corpora, live = _fixture_corpora()
    parent, manifest = _parent()
    store = ModelGenerationStore(tmp_path / "reopen-recovery-store")
    result = execute_trained_parent_lifecycle(
        parent_snapshot=parent.snapshot(),
        parent_manifest=manifest,
        parent_training_corpus=parent_training,
        corpora=corpora,
        numeric_field="temperature",
        adaptation_epochs=5,
        promotion_policy=PromotionAuditPolicy(8),
        store=store,
        ptmrt_executable=_ptmrt_path(),
    )
    append_event = store.append_event

    def fail_restoration(kind, generation_id, **details):
        if kind is LifecycleEventKind.PARENT_RESTORED:
            raise OSError("injected durable restoration failure")
        return append_event(kind, generation_id, **details)

    monkeypatch.setattr(store, "append_event", fail_restoration)
    with pytest.raises(OSError, match="restoration failure"):
        reopen_and_restore_for_drift(result, live, STRICT_DRIFT_POLICY)
    assert store.read_events()[-1].kind is LifecycleEventKind.REOPEN_REQUESTED

    recovered_store = ModelGenerationStore(store.root)
    controller = ModelGenerationController(
        recovered_store, ptmrt_executable=_ptmrt_path()
    )
    assert controller.active_generation_id == result.parent_generation.generation_id
    assert recovered_store.read_events()[-1].kind is LifecycleEventKind.PARENT_RESTORED
