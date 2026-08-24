"""Contract tests for the bounded, typed GNU Prolog PTA collective."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

import prolog_tsetlin.prolog_resources as prolog_resources
from prolog_tsetlin.prolog_resources import (
    PrologResourceError,
    prolog_module_candidates,
    resolve_gprolog,
    resolve_prolog_module,
    resolve_prolog_module_set,
)
from prolog_tsetlin.pta import (
    NotRepresentable,
    PTACollectiveBudget,
    PTACollectiveProtocolError,
    PTACollectiveQuery,
    PTACollectiveService,
    PTACollectiveUnavailable,
    PTAEscalationProposal,
    PTAInsight,
    PTAReasoningSession,
    lower_exact,
)
from prolog_tsetlin.pta.collective import (
    PROTOCOL_BEGIN,
    PROTOCOL_END,
    _decode_protocol,
    _run_bounded_process,
    _write_bounded_fact_lines,
)


def _has_gprolog() -> bool:
    try:
        resolve_gprolog()
    except PrologResourceError:
        return False
    return True


HAS_GPROLOG = _has_gprolog()


def test_shared_module_resolver_finds_installed_wheel_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_file = tmp_path / "site-packages" / "prolog_tsetlin" / "resources.py"
    installed = (
        tmp_path
        / "prefix"
        / "share"
        / "prolog-tsetlin-machine"
        / "prolog"
        / "pta_input.pl"
    )
    installed.parent.mkdir(parents=True)
    installed.write_text("% installed wheel resource\n", encoding="utf-8")
    monkeypatch.setattr(prolog_resources, "__file__", str(package_file))

    resolved = resolve_prolog_module("pta_input.pl", prefix=tmp_path / "prefix")

    assert resolved == installed.resolve()
    assert prolog_module_candidates(
        "pta_input.pl", prefix=tmp_path / "prefix"
    )[1] == installed


def test_collective_module_resolver_never_mixes_resource_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_file = (
        tmp_path / "site" / "python" / "prolog_tsetlin" / "resources.py"
    )
    checkout = tmp_path / "site" / "prolog"
    checkout.mkdir(parents=True)
    (checkout / "pta_ontology.pl").write_text("% partial checkout\n", encoding="utf-8")
    installed = (
        tmp_path
        / "prefix"
        / "share"
        / "prolog-tsetlin-machine"
        / "prolog"
    )
    installed.mkdir(parents=True)
    names = (
        "pta_ontology.pl",
        "pta_input.pl",
        "pta_deescalation.pl",
        "pta_escalation.pl",
    )
    for name in names:
        (installed / name).write_text(f"% installed {name}\n", encoding="utf-8")
    monkeypatch.setattr(prolog_resources, "__file__", str(package_file))

    resolved = resolve_prolog_module_set(names, prefix=tmp_path / "prefix")

    assert {path.parent for path in resolved.values()} == {installed.resolve()}


def test_module_resolver_rejects_unregistered_path() -> None:
    with pytest.raises(ValueError, match="unknown PTM Prolog module"):
        resolve_prolog_module("../hostile.pl")


def test_explicit_missing_gprolog_does_not_fall_back_to_path(tmp_path: Path) -> None:
    with pytest.raises(PTACollectiveUnavailable, match="was not found"):
        PTACollectiveService(executable=tmp_path / "missing-gprolog")


def test_explicit_non_executable_gprolog_is_rejected(tmp_path: Path) -> None:
    not_an_executable = tmp_path / "gprolog-data.pl"
    not_an_executable.write_text("% not an executable\n", encoding="utf-8")

    with pytest.raises(PTACollectiveUnavailable, match="was not found"):
        PTACollectiveService(executable=not_an_executable)


@pytest.mark.parametrize(
    ("collection_name", "limit_name", "values"),
    [
        ("observations", "max_observations", [("pta:x", 0, "x", 1), ("pta:x", 1, "x", 2)]),
        ("example_labels", "max_example_labels", [(0, 0), (1, 1)]),
        ("example_domains", "max_example_domains", {0, 1}),
        ("feature_supports", "max_feature_supports", [(0, 1, 0), (1, 1, 0)]),
        ("feature_relations", "max_feature_relations", [(0, "subsumes", 1), (1, "subsumes", 2)]),
        ("literal_truths", "max_literal_truths", [(0, 0, 1), (0, 1, 0)]),
        ("clause_truths", "max_clause_truths", [(0, 0, 1), (0, 1, 0)]),
        ("clause_literals", "max_clause_literals", [(0, 0), (0, 1)]),
        ("class_supports", "max_class_supports", [(0, 1, 0), (1, 1, 0)]),
        ("clause_class_scores", "max_clause_class_scores", [(0, 0, 0.5), (1, 0, 0.5)]),
        ("clause_supports", "max_clause_supports", [(0, 0), (0, 1)]),
        ("clause_conflicts", "max_clause_conflicts", [(0, 0), (0, 1)]),
        ("counterexamples", "max_counterexamples", [("m", 0, 0, 1), ("m", 1, 1, 0)]),
    ],
)
def test_every_fact_relation_is_bounded_even_after_direct_mutation(
    collection_name: str, limit_name: str, values: object
) -> None:
    session = PTAReasoningSession("bounded", **{limit_name: 1})
    setattr(session, collection_name, values)

    with pytest.raises(ValueError, match="budget exceeded"):
        session.validate()


def test_derived_product_collections_are_bounded() -> None:
    insight = PTAInsight("pta:test", "kind", "subject")
    proposal = PTAEscalationProposal(
        proposal_id="bounded-proposal",
        source_pta_ids=("pta:test",),
        supporting_insights=(),
        counterexamples_addressed=(),
        required_literals=(),
        native_target="threshold",
        structure={"field": "x", "operator": "ge", "threshold": 1.0},
        resource_bounds={"literal_count": 1},
    )
    session = PTAReasoningSession("products", max_insights=1, max_proposals=1)
    session.add_insight(insight)
    session.add_proposal(proposal)

    with pytest.raises(ValueError, match="insight budget exceeded"):
        session.add_insight(insight)
    with pytest.raises(ValueError, match="proposal budget exceeded"):
        session.add_proposal(proposal)


def test_session_rejects_unbounded_or_semantically_coerced_terms() -> None:
    session = PTAReasoningSession("strict")
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        session.add_observation("pta:input", 0, "x", {1: "coerced"})
    with pytest.raises(ValueError, match="depth budget"):
        session.add_observation(
            "pta:input",
            0,
            "x",
            [[[[[[[[[1]]]]]]]]],
        )
    with pytest.raises(ValueError, match="mapping key budget"):
        session.add_observation("pta:input", 0, "x", {"k" * 1_025: 1})
    session.add_observation("pta:input", 0, "large-id", 1 << 60)
    assert f"uint64('{1 << 60}')" in session.to_prolog_facts()


def test_query_and_budget_reject_coercive_types() -> None:
    with pytest.raises(TypeError, match="tuple"):
        PTACollectiveQuery(numeric_fields=["temperature"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="boolean"):
        PTACollectiveQuery(discover_thresholds=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="timeout_seconds"):
        PTACollectiveBudget(timeout_seconds=True)


def test_protocol_decoder_rejects_unknown_records_and_count_mismatch() -> None:
    unknown = (
        f"{PROTOCOL_BEGIN}\nX|0|1\n"
        f"{PROTOCOL_END}|1|1|1|0|0|0|0|0|0\n"
    ).encode()
    with pytest.raises(PTACollectiveProtocolError, match="unknown collective record"):
        _decode_protocol(
            unknown,
            id_to_field={0: "x"},
            id_to_literal={},
            id_to_clause={},
            id_to_class={},
            max_results_per_product=4,
        )

    mismatch = (
        f"{PROTOCOL_BEGIN}\nT|0|1.5\n"
        f"{PROTOCOL_END}|0|0|0|0|0|0|0|0|0\n"
    ).encode()
    with pytest.raises(PTACollectiveProtocolError, match="count"):
        _decode_protocol(
            mismatch,
            id_to_field={0: "x"},
            id_to_literal={},
            id_to_clause={},
            id_to_class={},
            max_results_per_product=4,
        )


def test_collective_rejects_conflicting_labels_before_prolog() -> None:
    session = PTAReasoningSession("conflict")
    session.add_observation("pta:input", 0, "temperature", 70)
    session.add_example_label(0, 0)
    session.add_example_label(0, 1)

    with pytest.raises(ValueError, match="conflicting labels"):
        PTACollectiveService._field_maps(session, PTACollectiveQuery())


def test_truth_facts_automatically_extend_the_evaluation_domain() -> None:
    session = PTAReasoningSession("domain")
    session.add_literal_truth(1, 7, 1)
    session.add_clause_truth(2, 9, 0)

    assert session.example_domains == {7, 9}


def test_direct_truth_fact_outside_domain_is_rejected() -> None:
    session = PTAReasoningSession("domain-mutation")
    session.literal_truths.append((1, 0, 1))

    with pytest.raises(ValueError, match="outside example_domains"):
        session.validate()


def test_partial_truth_vectors_are_rejected_before_prolog() -> None:
    session = PTAReasoningSession("partial-vectors")
    session.add_example_domain(0)
    session.add_example_domain(1)
    session.add_literal_truth(1, 0, 0)
    session.add_literal_truth(2, 0, 1)
    session.add_literal_truth(2, 1, 0)

    with pytest.raises(ValueError, match="lacks one exact truth value"):
        PTACollectiveService._validate_deescalation_truths(session)


def test_duplicate_truth_values_are_rejected_before_prolog() -> None:
    session = PTAReasoningSession("duplicate-vectors")
    session.add_literal_truth(1, 0, 0)
    session.add_literal_truth(1, 0, 0)

    with pytest.raises(ValueError, match="duplicate truth"):
        PTACollectiveService._validate_deescalation_truths(session)


def test_selected_numeric_field_requires_exact_arithmetic_range() -> None:
    session = PTAReasoningSession("large-numeric")
    session.add_observation("pta:input", 0, "x", 1 << 53)
    session.add_observation("pta:input", 1, "x", (1 << 53) + 2)
    session.add_example_label(0, 0)
    session.add_example_label(1, 1)

    with pytest.raises(ValueError, match="exact arithmetic range"):
        PTACollectiveService._field_maps(session, PTACollectiveQuery())


def test_subprocess_output_is_bounded_while_reading(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "print('x' * 5000)"]

    with pytest.raises(PTACollectiveProtocolError, match="output exceeded"):
        _run_bounded_process(
            command,
            cwd=tmp_path,
            timeout_seconds=5,
            max_output_bytes=1_024,
        )


@pytest.mark.skipif(not HAS_GPROLOG, reason="GNU Prolog is not installed")
def test_gprolog_input_pta_derives_threshold_and_interval_from_raw_facts() -> None:
    session = PTAReasoningSession("thermostat")
    for example, temperature in enumerate((60, 70, 80, 90)):
        session.add_observation("pta:input", example, "temperature", temperature)
    for example, label in enumerate((0, 1, 1, 0)):
        session.add_example_label(example, label)

    # The serialized input contains observations and labels, not candidates.
    facts = session.to_prolog_facts()
    assert "65.0" not in facts
    assert "85.0" not in facts
    result = PTACollectiveService().run(session)

    thresholds = {
        insight.evidence[0]
        for insight in result.insights
        if insight.kind == "threshold" and insight.subject == "temperature"
    }
    intervals = {
        insight.evidence
        for insight in result.insights
        if insight.kind == "interval" and insight.subject == "temperature"
    }
    assert thresholds == {65.0, 85.0}
    assert intervals == {(65.0, 85.0)}
    assert {proposal.structure["threshold"] for proposal in result.proposals} == {
        65.0,
        85.0,
    }
    assert all(isinstance(lower_exact(proposal), NotRepresentable) for proposal in result.proposals)
    assert not result.truncated


@pytest.mark.skipif(not HAS_GPROLOG, reason="GNU Prolog is not installed")
def test_collective_reports_per_product_truncation_without_starvation() -> None:
    session = PTAReasoningSession("truncation", max_observations=512)
    for example in range(130):
        session.add_observation("pta:input", example, "x", example)
        session.add_example_label(example, example % 2)

    result = PTACollectiveService().run(
        session,
        budget=PTACollectiveBudget(max_results_per_product=8),
    )

    thresholds = result.product_counts["threshold_insights"]
    proposals = result.product_counts["threshold_proposals"]
    assert thresholds.emitted == 8
    assert thresholds.available == 129
    assert proposals.emitted == 8
    assert proposals.available == 129
    assert thresholds.truncated and proposals.truncated and result.truncated
    assert len([item for item in result.insights if item.kind == "threshold"]) == 8
    assert len([item for item in result.proposals if item.native_target == "threshold"]) == 8


@pytest.mark.skipif(not HAS_GPROLOG, reason="GNU Prolog is not installed")
def test_collective_rejects_a_rounded_nonseparating_midpoint() -> None:
    lower = 1.0
    upper = math.nextafter(lower, 2.0)
    session = PTAReasoningSession("rounded-midpoint")
    session.add_observation("pta:input", 0, "x", lower)
    session.add_observation("pta:input", 1, "x", upper)
    session.add_example_label(0, 0)
    session.add_example_label(1, 1)

    with pytest.raises(PTACollectiveProtocolError, match="strictly separate"):
        PTACollectiveService().run(session)


@pytest.mark.skipif(not HAS_GPROLOG, reason="GNU Prolog is not installed")
def test_collective_maps_unsigned_64_bit_semantic_ids_to_portable_ids() -> None:
    first = (1 << 63) + 17
    second = (1 << 63) + 31
    session = PTAReasoningSession("opaque-identifiers")
    for offset, values in enumerate(((0, 0), (1, 1))):
        example = (1 << 63) + 100 + offset
        session.add_literal_truth(first, example, values[0])
        session.add_literal_truth(second, example, values[1])

    result = PTACollectiveService().run(
        session,
        query=PTACollectiveQuery(
            numeric_fields=(),
            discover_thresholds=False,
            discover_intervals=False,
            derive_escalation=False,
        ),
    )

    redundancy = next(
        item for item in result.insights if item.kind == "literal_redundant"
    )
    assert redundancy.evidence == (first, second)
    session.add_insight(redundancy)
    facts = session.to_prolog_facts()
    assert f"uint64('{first}')" in facts


@pytest.mark.skipif(not HAS_GPROLOG, reason="GNU Prolog is not installed")
def test_collective_rejects_serialized_fact_bytes_before_launch() -> None:
    session = PTAReasoningSession("input-bytes")
    session.add_observation("pta:input", 0, "x", "a" * 1_024)
    session.add_observation("pta:input", 1, "x", "b" * 1_024)

    with pytest.raises(ValueError, match="input byte budget"):
        PTACollectiveService().run(
            session,
            query=PTACollectiveQuery(numeric_fields=()),
            budget=PTACollectiveBudget(max_input_bytes=1_024),
        )


def test_bounded_fact_writer_stops_at_the_first_oversized_line(
    tmp_path: Path,
) -> None:
    def lines():
        yield "x" * 1_024
        raise AssertionError("writer consumed facts after the byte limit was exceeded")

    with pytest.raises(ValueError, match="input byte budget"):
        _write_bounded_fact_lines(
            tmp_path / "facts.pl",
            lines(),
            max_bytes=1_024,
        )


@pytest.mark.skipif(not HAS_GPROLOG, reason="GNU Prolog is not installed")
def test_unique_id_deescalation_avoids_truth_row_cross_product() -> None:
    session = PTAReasoningSession("moderate-deescalation")
    for literal in range(10):
        for example in range(30):
            session.add_literal_truth(literal, example, (example >> (literal % 5)) & 1)

    result = PTACollectiveService().run(
        session,
        query=PTACollectiveQuery(
            numeric_fields=(),
            discover_thresholds=False,
            discover_intervals=False,
            derive_escalation=False,
        ),
        budget=PTACollectiveBudget(timeout_seconds=2),
    )

    assert result.elapsed_seconds < 2


@pytest.mark.skipif(not HAS_GPROLOG, reason="GNU Prolog is not installed")
def test_collective_derives_deescalation_and_weight_products() -> None:
    session = PTAReasoningSession("collective-products")
    for example in range(3):
        session.add_example_domain(example)
    for literal, vector in ((1, (0, 1, 1)), (2, (0, 1, 1)), (3, (1, 1, 1))):
        for example, truth in enumerate(vector):
            session.add_literal_truth(literal, example, truth)
    session.add_clause_literal(10, 1)
    session.add_clause_literal(20, 1)
    session.add_clause_literal(20, 3)
    session.add_clause_literal(30, 1)
    for clause, vector in (
        (10, (0, 1, 1)),
        (20, (0, 1, 0)),
        (30, (0, 1, 1)),
    ):
        for example, truth in enumerate(vector):
            session.add_clause_truth(clause, example, truth)
    session.add_class_support(0, 2, 1)
    session.add_clause_class_score(10, 0, 0.75)

    result = PTACollectiveService().run(
        session,
        query=PTACollectiveQuery(
            numeric_fields=(),
            discover_thresholds=False,
            discover_intervals=False,
        ),
    )

    kinds = {insight.kind for insight in result.insights}
    assert "literal_redundant" in kinds
    assert "literal_subsumes" in kinds
    assert "clause_subsumes" in kinds
    clause_subjects = {
        insight.subject
        for insight in result.insights
        if insight.kind == "clause_subsumes"
    }
    assert "10->30" not in clause_subjects
    assert "30->10" not in clause_subjects
    assert len(result.proposals) == 1
    weight = result.proposals[0]
    assert weight.native_target == "shared_weighted_clause"
    assert weight.weights == (3,)
    assert weight.output_assignments == ((10, 0),)


@pytest.mark.skipif(not HAS_GPROLOG, reason="GNU Prolog is not installed")
def test_compatibility_entry_point_returns_typed_result() -> None:
    session = PTAReasoningSession("compatibility")
    session.add_observation("pta:input", 0, "x", 0)
    session.add_observation("pta:input", 1, "x", 2)
    session.add_example_label(0, 0)
    session.add_example_label(1, 1)

    result = session.consult_via_gprolog()

    assert result.insights[0].evidence == (1.0,)
    assert not isinstance(result, str)
