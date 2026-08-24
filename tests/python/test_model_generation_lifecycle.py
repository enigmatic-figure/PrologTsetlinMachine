from __future__ import annotations

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
    DriftAuditPolicy,
    GenerationKind,
    LabeledCorpus,
    LifecycleCorpora,
    ModelGeneration,
    ModelGenerationError,
    ModelGenerationLineage,
    OrderedLiteralManifest,
    PromotionAuditPolicy,
    PromotionAuditSnapshot,
    RuntimeConformanceReport,
    adapt_extended_parent,
    audit_parent_child,
    audit_runtime_conformance,
    drift_requires_reopen,
    extend_parent_with_threshold,
)
from prolog_tsetlin.pta import (
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
    reopen_and_restore_for_drift,
    verify_artifact_with_ptmrt,
)
from prolog_tsetlin.services.telemetry import TelemetrySession


ROOT = Path(__file__).resolve().parents[2]
GOLDEN_CHILD_HEX = ROOT / "tests" / "data" / "trained_parent_child_v1.hex"


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
    # Simulate a legacy or externally assembled durable route that bypassed
    # current registration. Recovery must still validate the full graph.
    store.append_event(LifecycleEventKind.PARENT_REGISTERED, generation.generation_id)

    with pytest.raises(ModelGenerationError, match="deployable generation"):
        ModelGenerationController(store)


def test_store_instances_share_one_process_local_event_lock(tmp_path: Path) -> None:
    first = ModelGenerationStore(tmp_path / "store")
    second = ModelGenerationStore(tmp_path / "store")

    assert first._event_lock is second._event_lock


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
    assert result.promotion_audit.accepted
    assert result.promotion_audit.improvements == 4
    assert result.promotion_audit.regressions == 0
    assert result.extended_parent.equivalence_case_count == len(
        parent_training.examples
    )
    assert result.invention_evidence == result.controller.store.load_invention_evidence(
        result.invention_evidence.evidence_id
    )
    assert result.child_artifact.serialized == bytes.fromhex(
        GOLDEN_CHILD_HEX.read_text(encoding="ascii")
    )
    assert result.controller.active_generation_id == result.child_generation.generation_id
    assert isinstance(result.controller.last_telemetry_error, RuntimeError)
    assert str(result.controller.last_telemetry_error) == (
        "injected telemetry sink failure"
    )
    # A process restart derives routing from the durable event chain.
    assert ModelGenerationController(store).active_generation_id == (
        result.child_generation.generation_id
    )

    with pytest.raises(ModelGenerationError, match="reopen request"):
        result.controller.restore_parent(result.restoration_bundle)

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
    with pytest.raises(ModelGenerationError, match="representation extension"):
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
        ModelGenerationController(unauthorized_restore_store)

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
    forged_reopen_store.put_audit(forged_drift)
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
    )
    with pytest.raises(ModelGenerationError, match="conformance evidence"):
        ModelGenerationController(forged_reopen_store)

    def block_native_verification(*args, **kwargs):
        del args, kwargs
        raise ModelGenerationError("injected native verification requirement")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            model_generation_service,
            "_verify_snapshot_records_with_ptmrt",
            block_native_verification,
        )
        with pytest.raises(ModelGenerationError, match="native verification"):
            result.controller.request_reopen(
                result.child_generation.generation_id,
                live,
                STRICT_DRIFT_POLICY,
                _ptmrt_path(),
            )
    assert store.read_events()[-1].kind is LifecycleEventKind.ACTIVATED

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
    assert ModelGenerationController(store).active_generation_id == (
        result.parent_generation.generation_id
    )

    def package_candidate(
        adapted_child,
        promotion_corpus: LabeledCorpus,
        *,
        name: str,
        claimed_adaptation_digest: str | None = None,
        require_accepted: bool = True,
    ) -> tuple[
        ModelGeneration,
        ModelGenerationLineage,
        PromotionAuditSnapshot,
    ]:
        preprocessing_id = result.child_generation.preprocessing_contract_id
        adaptation_digest = (
            adapted_child.adaptation_corpus_digest
            if claimed_adaptation_digest is None
            else claimed_adaptation_digest
        )
        behavior = AdaptiveBehaviorIdentity.from_child(
            adapted_child,
            preprocessing_contract_id=preprocessing_id,
        )
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
        assert audit.accepted is require_accepted
        store.put_audit(audit)
        lineage = ModelGenerationLineage(
            result.parent_generation.generation_id,
            result.extended_generation.generation_id,
            generation.generation_id,
            behavior.behavior_id,
            result.restoration_bundle.bundle_id,
            audit.audit_id,
            result.invention_evidence.evidence_id,
            result.lineage.invented_literal_id,
            corpora.invention.digest,
            adaptation_digest,
            promotion_corpus.digest,
            result.lineage.origin_proposal_semantic_id,
            result.lineage.origin_proposal_provenance_id,
        )
        store.put_lineage(lineage)
        return generation, lineage, audit

    repackaged_promotion = _corpus(
        CorpusRole.PROMOTION,
        500,
        (60, 67, 70, 74, 77, 80, 90, 96),
        (0, 0, 0, 0, 1, 1, 1, 1),
    )
    forged_adaptation_digest = _corpus(
        CorpusRole.ADAPTATION,
        600,
        (56, 63, 70, 72, 78, 85, 93, 101),
        (0, 0, 0, 0, 1, 1, 1, 1),
    ).digest
    assert forged_adaptation_digest != result.child.adaptation_corpus_digest
    repackaged_generation, repackaged_lineage, _ = package_candidate(
        result.child,
        repackaged_promotion,
        name="same adaptive behavior with fresh promotion evidence",
        claimed_adaptation_digest=forged_adaptation_digest,
    )
    assert repackaged_generation.generation_id != result.child_generation.generation_id
    assert repackaged_lineage.adaptive_behavior_id == result.lineage.adaptive_behavior_id
    with pytest.raises(ModelGenerationError, match="adaptive behavior"):
        result.controller.record_candidate(repackaged_lineage)

    genuinely_readapted = adapt_extended_parent(
        result.extended_parent, corpora.adaptation, epochs=6
    )
    rejected_promotion = _corpus(
        CorpusRole.PROMOTION,
        800,
        (59, 63, 69, 72, 78, 83, 88, 94),
        (1, 1, 1, 1, 0, 0, 0, 0),
    )
    rejected_generation, rejected_lineage, rejected_audit = package_candidate(
        genuinely_readapted,
        rejected_promotion,
        name="readapted child with rejected promotion evidence",
        require_accepted=False,
    )
    result.controller.record_candidate(rejected_lineage)
    result.controller.reject_candidate(
        rejected_generation.generation_id, rejected_audit
    )
    rejected_recovery_root = tmp_path / "rejected-activation-store"
    shutil.copytree(store.root, rejected_recovery_root)
    rejected_recovery_store = ModelGenerationStore(rejected_recovery_root)
    rejected_recovery_store.append_event(
        LifecycleEventKind.ACTIVATED,
        rejected_generation.generation_id,
        previous_generation_id=result.parent_generation.generation_id,
        lineage_id=rejected_lineage.lineage_id,
        adaptive_behavior_id=rejected_lineage.adaptive_behavior_id,
        audit_id=rejected_audit.audit_id,
    )
    with pytest.raises(ModelGenerationError, match="promotion approval"):
        ModelGenerationController(rejected_recovery_store)

    fresh_promotion = _corpus(
        CorpusRole.PROMOTION,
        700,
        (57, 65, 69, 72, 78, 84, 91, 99),
        (0, 0, 0, 0, 1, 1, 1, 1),
    )
    _, readapted_lineage, _ = package_candidate(
        genuinely_readapted,
        fresh_promotion,
        name="genuinely readapted child",
    )
    assert readapted_lineage.adaptive_behavior_id != result.lineage.adaptive_behavior_id
    result.controller.record_candidate(readapted_lineage)
    assert [event.kind.value for event in store.read_events()] == [
        "parent_registered",
        "candidate_created",
        "promotion_approved",
        "activated",
        "reopen_requested",
        "parent_restored",
        "candidate_created",
        "candidate_rejected",
        "candidate_created",
    ]
    assert ModelGenerationController(store).active_generation_id == (
        result.parent_generation.generation_id
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
def test_failed_activation_recovers_the_last_durable_parent_route(
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
    parent_id = events[0].generation_id
    controller = ModelGenerationController(store)
    assert controller.active_generation_id == parent_id
    lineage = store.load_lineage(events[-1].details["lineage_id"])
    audit = store.load_audit(events[-1].details["audit_id"])
    with pytest.raises(OSError, match="injected"):
        controller.activate_child(lineage, audit)
    assert controller.active_generation_id == parent_id
