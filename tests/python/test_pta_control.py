"""Tests for PTA control plane — Input invention, Type III pruning, CoTM."""

import pytest
from prolog_tsetlin.representation import FeatureSchema, FieldKind, LiteralCatalog
from prolog_tsetlin.budgeted_features import BudgetedFeatureStore
from prolog_tsetlin.pta import InputPTA, DeescalationPTA, EscalationPTA, lowerable, check_example


def _schema():
    return FeatureSchema.from_fields(temperature=FieldKind.NUMBER, mode=FieldKind.CATEGORY, score=FieldKind.NUMBER)


def test_input_pta_numeric_proposal_and_lowering():
    schema = _schema()
    catalog = LiteralCatalog(schema)
    pta = InputPTA(catalog, budget=8, pta_id="input:temperature")
    vals = [70, 71, 72, 73, 74, 75, 76, 77, 78, 80]
    labels = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    props = pta.propose_for_numeric("temperature", vals, labels, max_proposals=2)
    assert len(props) == 2
    # First threshold should be 72.5 (between 72→73 where label flips)
    assert props[0].parameters["threshold"] == pytest.approx(72.5)
    # Every proposal must be syntactically bounded; exact requires catalog and ClauseConfiguration
    from prolog_tsetlin.pta import syntactically_bounded, lower_exact, LoweredCandidate
    for p in props:
        prop = pta.to_proposal(p)
        assert syntactically_bounded(prop)[0] is True
        # Exact lowering requires catalog materialization — with catalog it should produce LoweredCandidate
        # Materialize descriptor in catalog first
        desc = catalog.preview_numeric_ge("temperature", p.parameters["threshold"])
        cat2 = catalog  # preview already computed stable ID
        # Register it to make exact succeed (simulating publish step)
        catalog.numeric_ge("temperature", p.parameters["threshold"])
        assert isinstance(lower_exact(prop, catalog=catalog), LoweredCandidate)
    # Intervals
    intervals = pta.propose_interval("temperature", vals, labels, max_proposals=1)
    assert intervals
    # Interval proposal syntactically bounded, but exact without clause is NotRepresentable until clause materialized
    assert syntactically_bounded(pta.to_proposal(intervals[0]))[0] is True
    # Budget enforced
    pta2 = InputPTA(catalog, budget=1, pta_id="input:score")
    many = pta2.propose_for_numeric("score", vals, labels, max_proposals=10)
    assert len(many) == 1
    assert pta2.proposed_count == 1


def test_input_pta_strict_and_dedup():
    schema = _schema()
    catalog = LiteralCatalog(schema)
    pta = InputPTA(catalog, budget=4, pta_id="input:temperature")
    with pytest.raises(ValueError):
        pta.propose_for_numeric("unknown", [1, 2], [0, 1])
    with pytest.raises(ValueError):
        pta.propose_for_numeric("temperature", [1, 2], [0, 2])  # bad label
    with pytest.raises(ValueError):
        pta.propose_for_numeric("temperature", [1, float("inf")], [0, 1])
    # Dedup: same threshold not proposed twice
    vals = [70, 72, 74, 76]
    labels = [0, 1, 0, 1]
    p1 = pta.propose_for_numeric("temperature", vals, labels, max_proposals=10)
    n1 = pta.proposed_count
    p2 = pta.propose_for_numeric("temperature", vals, labels, max_proposals=10)
    assert len(p2) == 0
    assert pta.proposed_count == n1


def test_deescalation_type_iii_pruning():
    schema = _schema()
    catalog = LiteralCatalog(schema)
    a = catalog.numeric_ge("temperature", 71.5)
    b = catalog.numeric_ge("temperature", 71.6)
    # Rows where thresholds are identical → thresholds_equivalent
    rows_eq = [{"temperature": v} for v in [70, 71, 72]]
    de = DeescalationPTA()
    insights = de.find_redundant_literals(catalog, [a.literal_id, b.literal_id], rows_eq)
    # With rows [70,71,72], 71.5 → [F,F,T], 71.6 → [F,F,T] identical → thresholds_equivalent
    assert any(i.kind == "thresholds_equivalent" for i in insights)
    # Rows with a value between thresholds → subsumption: 71.6 → 71.5
    rows_sub = [{"temperature": v} for v in [70, 71.55, 71.7]]
    sub = de.find_subsumed_literals(catalog, [a.literal_id, b.literal_id], rows_sub)
    assert any("subsumed" in i.kind for i in sub)


def test_deescalation_stable_absorption():
    schema = _schema()
    store = BudgetedFeatureStore(schema, budget=4)
    d = store.catalog.numeric_ge("score", 10)
    store.record_use(d.literal_id, 12)
    store.set_utility(d.literal_id, 0.95)
    de = DeescalationPTA()
    insights = de.propose_stable_absorption(store, utility_threshold=0.9, min_uses=10)
    assert len(insights) == 1
    assert insights[0].kind == "stable_inclusion"


def test_escalation_cotm_and_graph_depth():
    from prolog_tsetlin.pta import LoweredCandidate, NotRepresentable

    esc = EscalationPTA()
    weights = esc.allocate_cotm_weights(4, 2, {(0, 0): 0.9, (1, 0): 0.4, (2, 1): 0.85, (3, 1): 0.2})
    assert len(weights) > 0
    prop = esc.weights_to_proposal(weights)
    # CoTM not yet native — exact should be NotRepresentable
    from prolog_tsetlin.pta import lower_exact

    assert isinstance(lower_exact(prop), NotRepresentable)
    # Graph depth increase — not yet native, exact is NotRepresentable
    gprop = esc.propose_graph_depth_increase(3, [{"edges": []}], max_depth=8)
    assert gprop is not None and isinstance(lower_exact(gprop), NotRepresentable)
    # At max depth, no proposal
    assert esc.propose_graph_depth_increase(8, [], max_depth=8) is None
    # Unbounded recursion not lowerable (checked via lowerable directly)
    from prolog_tsetlin.pta import PTAEscalationProposal

    bad = PTAEscalationProposal(
        proposal_id="bad",
        source_pta_ids=("pta:test",),
        supporting_insights=(),
        counterexamples_addressed=(),
        required_literals=(),
        native_target="graph_clause",
        structure={"requires_recursion": True, "recursive_unbounded": True, "depth": 3},
        resource_bounds={"graph_depth": 3},
    )
    assert lowerable(bad)[0] is False


def test_check_example_lowerable():
    from prolog_tsetlin.pta import LoweredCandidate, NotRepresentable, lower_exact, syntactically_bounded

    p = check_example()
    # check_example uses magic IDs 104 etc — syntactically bounded True, but exact without catalog is NotRepresentable
    synt_ok, _ = syntactically_bounded(p)
    assert synt_ok is True
    assert isinstance(lower_exact(p), NotRepresentable)
    # With a catalog containing those IDs, it would be lowerable via lower_exact with catalog
    # But lowerable (alias to lower_exact without catalog) should be False per exact gate
    ok, _ = lowerable(p)
    assert ok is False


def test_input_categorical_group():
    schema = _schema()
    catalog = LiteralCatalog(schema)
    pta = InputPTA(catalog, budget=8, pta_id="input:mode")
    vals = ["manual", "manual", "auto", "manual", "auto", "manual"]
    labels = [1, 1, 0, 1, 0, 1]
    props = pta.propose_categorical_group("mode", vals, labels)
    assert any(p.parameters["value"] == "manual" for p in props)
