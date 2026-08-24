from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
from itertools import product

import pytest

from prolog_tsetlin.model_generation import (
    AdaptiveSnapshotEnvelope,
    CorpusExample,
    CorpusRole,
    LabeledCorpus,
    LifecycleCorpora,
    ModelGenerationError,
    OrderedLiteralManifest,
    PromotionAuditPolicy,
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
)


ROOT = Path(__file__).resolve().parents[2]
GOLDEN_CHILD_HEX = ROOT / "tests" / "data" / "trained_parent_child_v1.hex"


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
        drift_requires_reopen(promotion)

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
    assert drift_requires_reopen(live)


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
def test_live_trained_parent_loop_restores_bit_exact_parent(tmp_path: Path) -> None:
    parent_training, corpora, live = _fixture_corpora()
    parent, manifest = _parent()
    original_snapshot = parent.snapshot()
    store = ModelGenerationStore(tmp_path / "generation-store")
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
    # A process restart derives routing from the durable event chain.
    assert ModelGenerationController(store).active_generation_id == (
        result.child_generation.generation_id
    )

    with pytest.raises(ModelGenerationError, match="reopen request"):
        result.controller.restore_parent(result.restoration_bundle)

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
        reopen_and_restore_for_drift(result, overlapping_live)

    forged_lineage = replace(
        result.lineage,
        invented_literal_id=result.lineage.invented_literal_id ^ 1,
    )
    result.controller.store.put_lineage(forged_lineage)
    with pytest.raises(ModelGenerationError, match="representation extension"):
        result.controller.record_candidate(forged_lineage)

    drift, restored = reopen_and_restore_for_drift(result, live)
    assert drift.child_errors > drift.parent_errors
    assert restored.snapshot.snapshot == original_snapshot
    assert restored.manifest == manifest
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
    assert [event.kind.value for event in store.read_events()] == [
        "parent_registered",
        "candidate_created",
        "promotion_approved",
        "activated",
        "reopen_requested",
        "parent_restored",
    ]


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
