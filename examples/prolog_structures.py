"""Bounded typed-template, TA-clause, tree, and repair searches."""

from prolog_tsetlin import (
    BooleanDecisionTree,
    DataType,
    DecisionTreeSearchProblem,
    FeatureTemplateCandidate,
    FeatureTemplateSearchProblem,
    GNUPrologSearch,
    TAClauseSearchProblem,
)


search = GNUPrologSearch()

template_candidates = (
    FeatureTemplateCandidate.create(
        field_name="status",
        template_id="categorical_v1",
        data_type=DataType.CATEGORICAL,
        parameters={"categories": ["cold"]},
    ),
    FeatureTemplateCandidate.create(
        field_name="status",
        template_id="categorical_v1",
        data_type=DataType.CATEGORICAL,
        parameters={"categories": ["hot"]},
    ),
)
template_problem = FeatureTemplateSearchProblem.create(
    candidates=template_candidates,
    labels=[0, 1, 1, 0],
    coverage=[
        [0, 0, 1, 0],
        [0, 1, 1, 0],
    ],
)
template = search.search_feature_template(template_problem)
print("typed template:", template.candidate.template_spec())

rows = [set(), {0}, {1}, {0, 1}]
clause_problem = TAClauseSearchProblem.create(
    feature_count=2,
    max_literals=2,
    examples=rows,
    labels=[0, 1, 0, 0],
)
clause = search.search_ta_clause(clause_problem)
configuration = clause.to_ta_configuration(
    clause_problem,
    states_per_action=100,
    specificity=3.0,
    threshold=10,
)
print("signed TA literals:", clause.included_literals)
print("TA configuration:", configuration.to_json(indent=2))

tree_problem = DecisionTreeSearchProblem.create(
    slot_count=2,
    max_depth=2,
    examples=rows,
    labels=[0, 1, 1, 0],
)
tree = search.search_decision_tree(tree_problem).tree
print("XOR tree:", tree.to_dict())
print("fixed Logic instructions:", len(tree.to_logic_program().instructions))

repair = search.repair_decision_tree(
    BooleanDecisionTree.leaf(False),
    tree_problem,
    max_iterations=4,
)
print(
    "repair:",
    repair.mismatches_before,
    "->",
    repair.mismatch_count,
    "mismatches from",
    len(repair.counterexamples),
    "counterexamples",
)
