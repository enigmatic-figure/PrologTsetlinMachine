"""Trust boundary tests — exact lowering, Prolog safety, morphology, immutability."""

import json

import pytest
from prolog_tsetlin.representation import FeatureSchema, FieldKind, LiteralCatalog
from prolog_tsetlin.pta import (
    NATIVE_TARGETS,
    ExecutableBinaryClause,
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


def _minimal_proposal(**overrides) -> PTAEscalationProposal:
    values = {
        "proposal_id": "strict-types",
        "source_pta_ids": ("pta:test",),
        "supporting_insights": (),
        "counterexamples_addressed": (),
        "required_literals": (),
        "native_target": "composite_gate",
        "structure": {"specialists": ["binary"]},
        "resource_bounds": {},
    }
    values.update(overrides)
    return PTAEscalationProposal(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("proposal_id", True),
        ("source_pta_ids", (1.9,)),
        ("source_pta_ids", (True,)),
        ("counterexamples_addressed", (1.9,)),
        ("counterexamples_addressed", (True,)),
        ("required_literals", (1.9,)),
        ("required_literals", (True,)),
        ("weights", (1.9,)),
        ("weights", (True,)),
        ("output_assignments", ((0, 1.9),)),
        ("output_assignments", ((True, 0),)),
        ("support_trace", (1.9,)),
        ("support_trace", (True,)),
        ("native_target", True),
        ("lowering_version", True),
    ],
)
def test_escalation_proposal_rejects_semantic_type_coercion(field, value):
    with pytest.raises(TypeError):
        _minimal_proposal(**{field: value})


def test_insight_rejects_non_string_identity_fields():
    with pytest.raises(TypeError):
        PTAInsight(True, "kind", "subject")


def _proposal_for_native_target(target, catalog, descriptor):
    from prolog_tsetlin.logic_consolidation import (
        FixedLogicInstruction,
        FixedLogicOpcode,
        LogicProgram32,
    )

    program = LogicProgram32(
        (FixedLogicInstruction(FixedLogicOpcode.INPUT, argument=0),),
        root_instruction=0,
    )
    structures = {
        "binary_clause": {"clause": [descriptor.literal_id]},
        "logic_program": {"program": program.to_dict()},
        "threshold": {"clause": [descriptor.literal_id]},
        "shared_weighted_clause": {"weights": {"0:0": 1}},
        "regression_clause": {"clause": [descriptor.literal_id]},
        "graph_clause": {"depth": 1},
        "patch_clause": {
            "kind": "region",
            "patch": {"rows": 1, "cols": 1},
        },
        "composite_gate": {"specialists": ["binary"]},
    }
    bounds = {
        "binary_clause": {"literal_count": 1},
        "logic_program": {"literal_count": 1},
        "threshold": {"literal_count": 1},
        "shared_weighted_clause": {"clause_count": 1},
        "regression_clause": {"literal_count": 1},
        "graph_clause": {"graph_depth": 1},
        "patch_clause": {"patch_extent": 1},
        "composite_gate": {"literal_count": 1},
    }
    required = (
        (descriptor,)
        if target in {"binary_clause", "threshold", "regression_clause"}
        else ()
    )
    return PTAEscalationProposal(
        proposal_id=f"target:{target}",
        source_pta_ids=("pta:test",),
        supporting_insights=(),
        counterexamples_addressed=(),
        required_literals=required,
        native_target=target,
        structure=structures[target],
        resource_bounds=bounds[target],
    )


@pytest.mark.parametrize("target", NATIVE_TARGETS)
def test_every_native_target_has_exact_fail_closed_semantics(target):
    schema = FeatureSchema.from_fields(x=FieldKind.NUMBER)
    catalog = LiteralCatalog(schema)
    descriptor = catalog.numeric_ge("x", 5.0)
    proposal = _proposal_for_native_target(target, catalog, descriptor)

    result = lower_exact(proposal, catalog=catalog)

    if target == "binary_clause":
        assert isinstance(result, LoweredCandidate)
        assert result.native_kind == "executable_binary_clause"
        assert isinstance(result.native_object, ExecutableBinaryClause)
    elif target == "logic_program":
        assert isinstance(result, LoweredCandidate)
        assert result.native_kind == "logic_program32"
    else:
        assert isinstance(result, NotRepresentable)


@pytest.mark.parametrize(
    ("patch", "patch_extent"),
    [
        ({"rows": "2", "cols": 3}, 6),
        ({"rows": 2.0, "cols": 3}, 6),
        ({"rows": True, "cols": 3}, 3),
        ({"rows": 0, "cols": 3}, 3),
        ({"rows": -1, "cols": 3}, 3),
        ({"cols": 3}, 3),
        ({"rows": 3}, 3),
        ({"rows": 1 << 21, "cols": 1}, 1 << 21),
        ({"rows": 2, "cols": 3}, 5),
    ],
)
def test_malformed_patch_dimensions_fail_closed(patch, patch_extent):
    proposal = PTAEscalationProposal(
        proposal_id="malformed-patch",
        source_pta_ids=("pta:test",),
        supporting_insights=(),
        counterexamples_addressed=(),
        required_literals=(),
        native_target="patch_clause",
        structure={"kind": "region", "patch": patch},
        resource_bounds={"patch_extent": patch_extent},
    )

    bounded, reason = syntactically_bounded(proposal)
    assert bounded is False
    assert "patch" in reason
    assert isinstance(lower_exact(proposal), NotRepresentable)


def test_lowered_candidate_cannot_claim_an_unsupported_target():
    proposal = _minimal_proposal()
    with pytest.raises(TypeError, match="only be created by lower_exact"):
        LoweredCandidate(proposal, object(), "composite_gate")


def test_lowered_candidate_factory_cannot_be_bypassed_with_different_semantics():
    schema = FeatureSchema.from_fields(x=FieldKind.NUMBER)
    catalog = LiteralCatalog(schema)
    declared = catalog.numeric_ge("x", 1.0)
    different = catalog.numeric_ge("x", 2.0)
    proposal = PTAEscalationProposal(
        proposal_id="candidate-mismatch",
        source_pta_ids=("pta:test",),
        supporting_insights=(),
        counterexamples_addressed=(),
        required_literals=(declared,),
        native_target="binary_clause",
        structure={"clause": (declared.literal_id,)},
        resource_bounds={"literal_count": 1},
    )

    with pytest.raises(TypeError, match="only be created by lower_exact"):
        LoweredCandidate(
            proposal,
            ExecutableBinaryClause((different,)),
            "executable_binary_clause",
        )


def test_pattern_only_logic_fails_exact():
    pat = discover_sequence_patterns([[1, 2, 3], [1, 2, 3]], [1, 1])[0]
    prop = seq_to_proposal(pat)
    assert isinstance(lower_exact(prop), NotRepresentable)
    # syntactically bounded passes but exact fails — no false positive
    assert syntactically_bounded(prop)[0] is True


def _logic_program_proposal(program: dict) -> PTAEscalationProposal:
    return PTAEscalationProposal(
        proposal_id="logic-program",
        source_pta_ids=("pta:test",),
        supporting_insights=(),
        counterexamples_addressed=(),
        required_literals=(),
        native_target="logic_program",
        structure={"program": program},
        resource_bounds={"literal_count": 1},
    )


def test_canonical_logic_program_lowers_to_validated_native_object():
    from prolog_tsetlin.logic_consolidation import (
        FixedLogicInstruction,
        FixedLogicOpcode,
        LogicProgram32,
    )

    program = LogicProgram32(
        (FixedLogicInstruction(FixedLogicOpcode.INPUT, argument=0),),
        root_instruction=0,
    )

    result = lower_exact(_logic_program_proposal(program.to_dict()))

    assert isinstance(result, LoweredCandidate)
    assert result.native_kind == "logic_program32"
    assert result.native_object == program


def test_nested_proposal_to_dict_is_json_serializable():
    from prolog_tsetlin.logic_consolidation import (
        FixedLogicInstruction,
        FixedLogicOpcode,
        LogicProgram32,
    )

    program = LogicProgram32(
        (FixedLogicInstruction(FixedLogicOpcode.INPUT, argument=0),),
        root_instruction=0,
    )
    proposal = PTAEscalationProposal(
        proposal_id="nested-json",
        source_pta_ids=("pta:test",),
        supporting_insights=(
            PTAInsight(
                "pta:test",
                "nested",
                "logic",
                ({"levels": {"values": [1, 2]}},),
            ),
        ),
        counterexamples_addressed=(),
        required_literals=(),
        native_target="logic_program",
        structure={"program": program.to_dict()},
        resource_bounds={"literal_count": 1},
        validation_signature={"oracle": {"cases": [False, True]}},
    )

    decoded = json.loads(json.dumps(proposal.to_dict(), allow_nan=False))
    assert decoded["structure"]["program"] == program.to_dict()
    assert decoded["supporting_insights"][0]["evidence"][0]["levels"] == {
        "values": [1, 2]
    }
    assert decoded["validation_signature"]["oracle"]["cases"] == [False, True]


@pytest.mark.parametrize(
    "program",
    [
        {"instructions": [{}]},
        {
            "schema_version": 1,
            "program_kind": "logic_program_32",
            "instruction_count": 1,
            "root_instruction": 0,
            "instructions": [
                {"opcode": "input", "opcode_value": 99, "operand_mask": 0, "argument": 0}
            ],
        },
        {
            "schema_version": 1,
            "program_kind": "logic_program_32",
            "instruction_count": 1,
            "root_instruction": 1,
            "instructions": [
                {"opcode": "input", "opcode_value": 1, "operand_mask": 0, "argument": 0}
            ],
        },
        {
            "schema_version": 1,
            "program_kind": "logic_program_32",
            "instruction_count": 1,
            "root_instruction": 0,
            "instructions": [
                {"opcode": "input", "opcode_value": 1, "operand_mask": 1, "argument": 0}
            ],
        },
        {
            "schema_version": 1,
            "program_kind": "logic_program_32",
            "instruction_count": 1,
            "root_instruction": 0,
            "instructions": [
                {"opcode": "input", "opcode_value": 1, "operand_mask": 0, "argument": 256}
            ],
        },
    ],
)
def test_malformed_logic_program_never_crosses_exact_gate(program: dict):
    assert isinstance(lower_exact(_logic_program_proposal(program)), NotRepresentable)


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
    lowered = lower_exact(p_ok, catalog=cat)
    assert isinstance(lowered, LoweredCandidate)
    assert isinstance(lowered.native_object, ExecutableBinaryClause)
    assert lowered.native_object.evaluate({desc.literal_id: True}) is True
    assert lowered.native_object.evaluate({desc.literal_id: False}) is False
    with pytest.raises(KeyError, match="missing truth value"):
        lowered.native_object.evaluate({})
    with pytest.raises(TypeError, match="must be bool"):
        lowered.native_object.evaluate({desc.literal_id: 1})
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


def test_binary_clause_exact_lowering_requires_canonical_literal_identity():
    schema = FeatureSchema.from_fields(x=FieldKind.NUMBER)
    catalog = LiteralCatalog(schema)
    descriptors = (
        catalog.numeric_ge("x", 1.0),
        catalog.numeric_ge("x", 2.0),
    )
    literal_ids = tuple(sorted(descriptor.literal_id for descriptor in descriptors))

    for clause in (tuple(reversed(literal_ids)), (literal_ids[0], literal_ids[0])):
        proposal = PTAEscalationProposal(
            proposal_id="noncanonical-clause",
            source_pta_ids=("pta:test",),
            supporting_insights=(),
            counterexamples_addressed=(),
            required_literals=(),
            native_target="binary_clause",
            structure={"clause": clause},
            resource_bounds={"literal_count": len(clause)},
        )
        assert isinstance(lower_exact(proposal, catalog=catalog), NotRepresentable)

    noncanonical_reference = PTAEscalationProposal(
        proposal_id="noncanonical-reference",
        source_pta_ids=("pta:test",),
        supporting_insights=(),
        counterexamples_addressed=(),
        required_literals=(f"literal:0{literal_ids[0]}",),
        native_target="binary_clause",
        structure={"clause": (literal_ids[0],)},
        resource_bounds={"literal_count": 1},
    )
    assert isinstance(
        lower_exact(noncanonical_reference, catalog=catalog), NotRepresentable
    )


def test_descriptor_and_canonical_literal_id_share_semantic_identity():
    schema = FeatureSchema.from_fields(x=FieldKind.NUMBER)
    catalog = LiteralCatalog(schema)
    descriptor = catalog.numeric_ge("x", 1.0)
    common = {
        "proposal_id": "semantic-literal-identity",
        "source_pta_ids": ("pta:test",),
        "supporting_insights": (),
        "counterexamples_addressed": (),
        "native_target": "binary_clause",
        "structure": {"clause": (descriptor.literal_id,)},
        "resource_bounds": {"literal_count": 1},
    }
    by_descriptor = PTAEscalationProposal(
        required_literals=(descriptor,), **common
    )
    by_id = PTAEscalationProposal(
        required_literals=(f"literal:{descriptor.literal_id}",), **common
    )

    assert by_descriptor.semantic_id() == by_id.semantic_id()


def test_fake_composite_gate_fails_exact():
    gate = discover_specialist_gates([{"edges": [1]}], {"graph_model": [0.9]})[0]
    prop = gate_to_proposal(gate)
    # Composite not yet native — exact is NotRepresentable
    assert isinstance(lower_exact(prop), NotRepresentable)
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
    # Graph not yet native — exact is NotRepresentable
    assert isinstance(lower_exact(prop), NotRepresentable)
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
    gate = discover_specialist_gates([{"edges": [1], "text": "hi"}], {"graph_model": [0.9], "text_model": [0.1]})[0]
    prop = gate_to_proposal(gate)
    res = lower_exact(prop)
    assert isinstance(res, NotRepresentable)
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
