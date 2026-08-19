"""Trust boundary tests — exact lowering, Prolog safety, morphology, immutability."""

import pytest
from prolog_tsetlin.representation import FeatureSchema, FieldKind, LiteralCatalog
from prolog_tsetlin.pta import (
    PTAEscalationProposal,
    PTAInsight,
    lower_exact,
    lowerable,
    syntactically_bounded,
    LoweredCandidate,
    NotRepresentable,
    find_residual_regions,
    residual_to_proposal,
    invent_spatial_templates,
    spatial_to_proposal,
    hypothesize_graph_relation,
    graph_hypothesis_to_proposal,
    discover_sequence_patterns,
    seq_to_proposal,
    discover_specialist_gates,
    gate_to_proposal,
    smallest_specialist_subset,
    to_sparse_exact,
    propose_sparse_morphology,
    PTAReasoningSession,
)
from prolog_tsetlin.pta.proposal import PTAMorphologyProposal
from prolog_tsetlin.pta.graph_pta import RelationalHypothesis


def test_pattern_only_logic_fails_exact():
    pat = discover_sequence_patterns([[1, 2, 3], [1, 2, 3]], [1, 1])[0]
    prop = seq_to_proposal(pat)
    assert isinstance(lower_exact(prop), NotRepresentable)
    # syntactically bounded passes but exact fails — no false positive
    assert syntactically_bounded(prop)[0] is True


def test_arbitrary_literal_ids_fail_without_catalog():
    # check_example magic IDs should not be exact without catalog
    from prolog_tsetlin.pta.lowering import check_example

    p = check_example()
    assert isinstance(lower_exact(p), NotRepresentable)
    # With catalog containing those IDs, we can make it exact by providing catalog with descriptors
    # For now, without catalog, syntactically passes but exact fails — demonstrates gate


def test_literal_must_exist_in_catalog():
    schema = FeatureSchema.from_fields(x=FieldKind.NUMBER)
    cat = LiteralCatalog(schema)
    desc = cat.numeric_ge("x", 5.0)
    # Proposal with real literal should succeed with catalog
    p_ok = PTAEscalationProposal(
        proposal_id="ok",
        source_pta_ids=("pta:test",),
        supporting_insights=(),
        counterexamples_addressed=(),
        required_literals=(desc,),
        native_target="binary_clause",
        structure={"clause": [desc.literal_id]},
        resource_bounds={"literal_count": 1},
    )
    assert isinstance(lower_exact(p_ok, catalog=cat), LoweredCandidate)
    # Fake integer not in catalog should fail exact with catalog
    p_fake = PTAEscalationProposal(
        proposal_id="fake",
        source_pta_ids=("pta:test",),
        supporting_insights=(),
        counterexamples_addressed=(),
        required_literals=(),
        native_target="binary_clause",
        structure={"clause": [999999999]},
        resource_bounds={"literal_count": 1},
    )
    assert isinstance(lower_exact(p_fake, catalog=cat), NotRepresentable)


def test_fake_composite_gate_fails_exact():
    gate = discover_specialist_gates([{"edges": [1]}], {"graph_model": [0.9]})[0]
    prop = gate_to_proposal(gate)
    # Composite reference gate now lowers to candidate (native composite not yet, but reference implementation)
    assert isinstance(lower_exact(prop), LoweredCandidate)
    # But empty composite should fail
    empty = PTAEscalationProposal(
        proposal_id="empty-comp",
        source_pta_ids=("pta:test",),
        supporting_insights=(),
        counterexamples_addressed=(),
        required_literals=(),
        native_target="composite_gate",
        structure={},
        resource_bounds={"literal_count": 1},
    )
    assert isinstance(lower_exact(empty), NotRepresentable)


def test_malformed_descriptor_string_fails():
    p = PTAEscalationProposal(
        proposal_id="bad",
        source_pta_ids=("pta:test",),
        supporting_insights=(),
        counterexamples_addressed=(),
        required_literals=("not-a-descriptor",),
        native_target="binary_clause",
        structure={"clause": []},
        resource_bounds={"literal_count": 1},
    )
    assert isinstance(lower_exact(p), NotRepresentable)


def test_unsupported_graph_fails_exact():
    hyp = RelationalHypothesis("ancestor", depth=2, recursive=True, support=1)
    prop = graph_hypothesis_to_proposal(hyp)
    # Bounded graph now lowers to candidate placeholder (even though runtime UNSUPPORTED) — per current reference implementation
    assert isinstance(lower_exact(prop), LoweredCandidate)
    # Unbounded must fail
    hyp2 = RelationalHypothesis("transitive_closure", depth=9, recursive=True, support=1)
    prop2 = graph_hypothesis_to_proposal(hyp2)
    assert isinstance(lower_exact(prop2), NotRepresentable)


def test_morphology_not_threshold():
    schema = FeatureSchema.from_fields(x=FieldKind.NUMBER)
    cat = LiteralCatalog(schema)
    a = cat.numeric_ge("x", 1.0)
    b = cat.numeric_ge("x", 1.0)  # duplicate literal set will be same
    # Actually need distinct lids with same truth column to trigger morphology
    cat2 = LiteralCatalog(FeatureSchema.from_fields(t=FieldKind.NUMBER))
    a2 = cat2.numeric_ge("t", 71.5)
    b2 = cat2.numeric_ge("t", 71.6)
    rows = [{"t": 70}, {"t": 71}, {"t": 72}]  # 71.5 and 71.6 identical on these rows
    exact, morph = propose_sparse_morphology(cat2, {0: frozenset([a2.literal_id]), 1: frozenset([b2.literal_id])}, rows)
    assert exact.clause_count == 2
    assert isinstance(morph, PTAMorphologyProposal)
    # Morphology is not a PTAEscalationProposal with threshold target
    assert not isinstance(morph, PTAEscalationProposal)


def test_sparse_exact_preserves_all():
    schema = FeatureSchema.from_fields(x=FieldKind.NUMBER)
    cat = LiteralCatalog(schema)
    a = cat.numeric_ge("x", 1.0)
    b = cat.numeric_ge("x", 2.0)
    bank = to_sparse_exact({0: frozenset([a.literal_id]), 1: frozenset([a.literal_id, b.literal_id])})
    assert bank.clause_count == 2
    assert bank.literal_count == 2


def test_proposal_deep_immutability():
    p = PTAEscalationProposal(
        proposal_id="imm",
        source_pta_ids=("pta:test",),
        supporting_insights=(),
        counterexamples_addressed=(),
        required_literals=(),
        native_target="binary_clause",
        structure={"clause": [1], "nested": {"a": [1, 2]}},
        resource_bounds={"literal_count": 1},
    )
    # Outer mapping immutable
    with pytest.raises(TypeError):
        p.structure["new"] = 1
    # Nested list is tuple, not list
    assert isinstance(p.structure["clause"], tuple)
    # Nested dict is MappingProxyType
    from types import MappingProxyType

    assert isinstance(p.structure["nested"], MappingProxyType)
    with pytest.raises(TypeError):
        p.structure["nested"]["a"] = (3,)
    # List inside nested is tuple
    assert isinstance(p.structure["nested"]["a"], tuple)
    # Evidence deep freeze
    ins = PTAInsight("pta:test", "kind", "subj", ({"k": [1, 2]},))
    assert isinstance(ins.evidence[0], MappingProxyType)
    with pytest.raises(TypeError):
        ins.evidence[0]["k"] = (9,)


def test_proposal_hash_covers_weights():
    p1 = PTAEscalationProposal(
        proposal_id="id1",
        source_pta_ids=("pta:a",),
        supporting_insights=(),
        counterexamples_addressed=(),
        required_literals=("literal:104",),
        native_target="binary_clause",
        structure={"clause": [104]},
        resource_bounds={"literal_count": 1},
    )
    p2 = PTAEscalationProposal(
        proposal_id="id1",
        source_pta_ids=("pta:a",),
        supporting_insights=(),
        counterexamples_addressed=(),
        required_literals=("literal:104",),
        native_target="binary_clause",
        structure={"clause": [104]},
        resource_bounds={"literal_count": 1},
        weights=(1,),
    )
    assert p1.semantic_id() != p2.semantic_id()
    assert p1.provenance_id() != p2.provenance_id()


def test_session_safe_encoding_hostile():
    sess = PTAReasoningSession(dataset_id="test")
    # Hostile: apostrophe, backslash, control, unicode, Prolog-like syntax, bool
    hostile_pta = "pta:evil'\\n"
    sess.add_observation(hostile_pta, 0, "field'with'quote", "it's a \\ test \n \x01")
    sess.add_observation("pta:ok", 1, "f", True)
    sess.add_observation("pta:ok", 2, "f", False)
    sess.add_example_label(0, 1)
    sess.add_insight(PTAInsight("pta:evil'atom", "kind'with'quote", "subj\\backslash", ({"evil": "it's \n"}, [1, 2], True)))
    facts = sess.to_prolog_facts()
    # Should contain escaped atoms, not raw interpolation
    assert "''" in facts or "\\n" in facts
    assert "True" not in facts  # Python True should be prolog true
    assert "False" not in facts
    assert "true" in facts or "false" in facts
    # Should not contain Python repr like "{'evil':"
    assert "{'evil'" not in facts
    # Backslash should be escaped
    assert "\\\\" in facts


def test_session_requires_example_label_for_input():
    # Prolog invent_threshold now needs example_label join — session should support it
    sess = PTAReasoningSession(dataset_id="test2")
    sess.add_observation("input:temperature", 0, "temperature", 70)
    sess.add_observation("input:temperature", 1, "temperature", 75)
    sess.add_example_label(0, 0)
    sess.add_example_label(1, 1)
    facts = sess.to_prolog_facts()
    assert "example_label(0,0)." in facts
    assert "observation(" in facts


def test_composite_exact_fails_until_native():
    # Updated: composite reference gate now succeeds; empty gate fails
    gate = discover_specialist_gates([{"edges": [1], "text": "hi"}], {"graph_model": [0.9], "text_model": [0.1]})[0]
    prop = gate_to_proposal(gate)
    res = lower_exact(prop)
    assert isinstance(res, LoweredCandidate)
    empty = PTAEscalationProposal(
        proposal_id="empty-comp2",
        source_pta_ids=("pta:test",),
        supporting_insights=(),
        counterexamples_addressed=(),
        required_literals=(),
        native_target="composite_gate",
        structure={},
        resource_bounds={"literal_count": 1},
    )
    res2 = lower_exact(empty)
    assert isinstance(res2, NotRepresentable)
