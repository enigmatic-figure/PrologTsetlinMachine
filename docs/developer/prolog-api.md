# Prolog predicate reference

> Transitional authored reference. Prolog files own predicate modes, determinism,
> bounds, and effects; Python owns request validation and the versioned result
> protocol. A structured-comment renderer is planned. See [GNU Prolog integration](prolog.md).

## Protocol overview

Every bounded search uses a **numeric-only driver fact** plus a single versioned `PTM_RESULT` line. Python validates candidates before and after the subprocess:

| Protocol | Driver fact | Query goal | Result line | Python validator |
|---|---|---|---|---|
| `v1` masked threshold | `problem(SlotCount, MaxSelected, Positives, Negatives)` | `ptm_run_problem` | `PTM_RESULT v1 masked_threshold selected=... minimum=... mismatches=0` | `ThresholdSearchProblem.create` + `_validate_threshold_result` |
| `v2` feature template | `feature_template_problem(Positives, Negatives, Coverages)` | `ptm_run_feature_template_problem` | `PTM_RESULT v2 feature_template candidate=... mismatches=0` | `FeatureTemplateSearchProblem.create` + coverage equality |
| `v2` TA clause | `ta_clause_problem(FeatureCount, MaxLiterals, Positives, Negatives)` | `ptm_run_ta_clause_problem` | `PTM_RESULT v2 ta_clause literals=... mismatches=0` | `TAClauseSearchProblem.create` + `TAClauseSearchResult.matches` |
| `v2` decision tree | `decision_tree_problem(SlotCount, MaxDepth, Examples, Labels)` | `ptm_run_decision_tree_problem` | `PTM_RESULT v2 decision_tree nodes=... depth=... tree=... mismatches=0` | `DecisionTreeSearchProblem.create` + read-once + depth checks |
| `v2` no_solution | — | — | `PTM_RESULT v2 no_solution kind=...` | `No*Solution` typed error |

Monotonicity pre-check: `ThresholdSearchProblem` rejects a positive subset of a negative before launching GNU Prolog (no subprocess). All other candidate ceilings are computed in `prolog_bridge.py` before launch; exceeding `MAX_SEARCH_CANDIDATES = 1_000_000` raises `ValueError` without invoking Prolog.

Subprocess: GNU Prolog via `gprolog --consult-file driver.pl --query-goal <goal>` with `LINEDIT=gui=no` on Windows, wall-clock deadline (default 30s), cooperative `cancel` poll, and `CREATE_NO_WINDOW`. Timeouts and cancellation raise `PrologBridgeError`/`PrologSearchCancelled`. Only `PTM_RESULT` lines are parsed; other stdout/stderr is truncated to 2000 chars in the error.

## bounded_threshold_search.pl

File: `prolog/bounded_threshold_search.pl` — smallest exact monotone `popcount(input AND selected) >= minimum`.

| Predicate | Mode | Determinism | Bounds | Effects |
|---|---|---|---|---|
| `ptm_integer_between(+Lower:int, +Upper:int, -Value:int)` | `+ + -` | nondet (enumerates `Lower..Upper`) | `Lower =< Upper` | none |
| `ptm_slots(+Count:int, -Slots:list(int))` | `+ -` | det | `1 =< Count =< 4096`, produces `0..Count-1` | none |
| `ptm_slots_from(+Current, +Last, -Rest)` | `+ + -` | det | internal helper for `ptm_slots` | none |
| `ptm_choose_exact(+Count, +List, -Chosen:list(int))` | `+ + -` | nondet | `Count =< len(List)`; enumerates `comb(List, Count)` | none |
| `ptm_contains(+Value, +List)` | `+ +` | semidet | first arg ground | cut on first match |
| `ptm_match_count(+Selected:list(int), +Example:list(int), -Count:int)` | `+ + -` | det | `|Selected| =< 16` (Python side: `max_selected`) | none |
| `ptm_all_positive(+Positives:list(list(int)), +Selected, +Minimum)` | `+ + +` | semidet | `Minimum in 1..|Selected|` | none |
| `ptm_all_negative(+Negatives, +Selected, +Minimum)` | `+ + +` | semidet | same | none |
| `ptm_search_threshold(+SlotCount, +MaxSelected, +Positives, +Negatives, -Selected, -Minimum)` | `+ + + + - -` | nondet (cut after first solution) | `SlotCount 1..4096`, `MaxSelected 1..min(SlotCount,16)`, `candidate_rules = sum C(SlotCount,w)*w` bounded by 1M, examples `<=4096` | none; fails if monotonicity contradiction |
| `ptm_write_csv(+Values:list(int))` | `+` | det | writes `a,b,c` | side-effect: `write/1` |
| `ptm_run_problem` | `0` | det | consults `problem/4` fact, calls search, writes `PTM_RESULT v1 ...`, `halt` | writes exactly one `PTM_RESULT` line, halts |
| `problem/4` (driver fact) | fact | — | asserted by Python driver: `problem(SlotCount, MaxSelected, Positives, Negatives)` where each example is sorted, deduplicated `list(int)` of true slot indices | none |

**Search order:** enumerates width `1..EffectiveMax`, then `Selected` combinations, then `Minimum 1..Width`; first candidate satisfying both `ptm_all_positive` and `ptm_all_negative` is returned (red cut in `ptm_search_threshold`).

## bounded_structure_search.pl

File: `prolog/bounded_structure_search.pl` — three independent v2 templates.

### Shared helpers

| Predicate | Mode | Determinism | Bounds |
|---|---|---|---|
| `ptm_contains/2`, `ptm_all_members/2`, `ptm_no_members/2`, `ptm_nth0/3` | as above | semidet/det | lists bounded by 4096 |
| `ptm_integer_between/3`, `ptm_slots/2`, `ptm_choose_exact/3` | as above | — | same as v1 |
| `ptm_complement(+Literal:int, -Complement:int)` | `+ -` | det | `Literal in 0..2*FeatureCount-1`; `Comp = Lit ^ 1` |
| `ptm_consistent_literals(+Literals:list(int))` | `+` | semidet | `|Literals| =< 16`; no complementary pair |
| `ptm_consistent_pair(+A:list(int), +B:list(int))` | `+ +` | semidet | checks `A` vs complement of `B` |

### Feature-template selection

Typed candidates are owned by Python (`TemplateRegistry`); Prolog sees only bounded numeric indices and coverage.

| Predicate | Mode | Determinism | Bounds | Effects |
|---|---|---|---|---|
| `ptm_search_feature_template(+Positives:list(int), +Negatives:list(int), +Coverages:list(list(int)), -Candidate:int)` | `+ + + -` | nondet → cut after first | `|Candidates| 1..4096`, `examples 2..4096` with both classes | none |
| `feature_template_problem/3` (driver fact) | fact | — | `feature_template_problem(Positives, Negatives, Coverages)` where `Coverage[k]` is sorted example ids where candidate `k` fires | none |
| `ptm_run_feature_template_problem` | `0` | det | writes `PTM_RESULT v2 feature_template candidate=...` or `no_solution kind=feature_template`, `halt` | single result line |

Python candidate ceiling: `len(candidates)` itself; no combinatorial explosion.

### Signed TA-clause conjunction

Literal `2*N` is feature `N` positive, `2*N+1` is its negation. Conjunction is exact, not threshold.

| Predicate | Mode | Determinism | Bounds |
|---|---|---|---|
| `ptm_search_ta_clause(+FeatureCount, +MaxLiterals, +Positives, +Negatives, -Literals:list(int))` | `+ + + + -` | nondet → cut | `FeatureCount 1..2048`, `MaxLiterals 1..min(FeatureCount,16)`, candidate bound `sum C(2*FeatureCount, w)` ≤ 1M, examples ≤4096, contradictory rows rejected |
| `ta_clause_problem/4` (driver fact) | fact | — | `ta_clause_problem(FeatureCount, MaxLiterals, Positives, Negatives)` |
| `ptm_run_ta_clause_problem` | `0` | det | writes `PTM_RESULT v2 ta_clause literals=...` or no_solution |

Consistency: `ptm_consistent_literals(Literals)` enforced before `ptm_all_positive/negative` checks.

### Read-once Boolean decision tree

Depth-limited, no repeated feature on a root-to-leaf path; prefix encoding is `0`=leaf false, `1`=leaf true, `2, Feature, FalseSubtree, TrueSubtree`.

| Predicate | Mode | Determinism | Bounds |
|---|---|---|---|
| `ptm_search_decision_tree(+SlotCount, +MaxDepth, +Examples, +Labels, -Tree:list(int))` | `+ + + + -` | nondet → cut | `SlotCount 1..4096`, `MaxDepth 0..min(SlotCount,8)`, candidate bound conservative recurrence `T(s,d)=2+s*T(s-1,d-1)^2` ≤ 1M |
| `decision_tree_problem/4` (driver fact) | fact | — | `decision_tree_problem(SlotCount, MaxDepth, Examples, Labels)` where `Labels` are `0/1` ints |
| `ptm_run_decision_tree_problem` | `0` | det | writes `PTM_RESULT v2 decision_tree nodes=... depth=... tree=...` or no_solution; validates `is_read_once` before reporting |

### Repair (Python-only, uses tree search iteratively)

`GNUPrologSearch.repair_decision_tree(parent, problem, max_iterations=32, timeout_seconds=30)` treats `parent` immutable and synthesizes guard `G` with `parent XOR guard == label`. Each iteration adds the first Python-side mismatch as a constrained example and re-runs bounded tree search on the accumulated flip constraints. Completion requires zero mismatches; otherwise `RepairDidNotConverge`. Guard and `parent XOR guard` both lower via `BooleanDecisionTree.to_logic_program()` when the combined expression fits 32 instructions and bindings `A..E`.

## PTA ontology and I/O

Files: `prolog/pta_ontology.pl`, `prolog/pta_input.pl`, `prolog/pta_deescalation.pl`, `prolog/pta_escalation.pl` (Class III control-plane proposals).

| File | Exported predicates (examples) | Mode | Bounds |
|---|---|---|---|
| `pta_ontology.pl` | bounded dynamic relations including `observation/4`, `example_label/2`, `literal_truth/3`, `clause_truth/3`, `insight/4`, `proposal/3` | facts | each relation is bounded by `PTAReasoningSession`; collective execution applies a second resource budget |
| `pta_input.pl` | `invent_threshold/2`, `invent_interval/3` | `+ -` / `+ - -`, nondet | numeric fields and observations are bounded; finite intervals require observed negative regions on both sides |
| `pta_deescalation.pl` | `literals_equivalent/2`, `literal_subsumes/2`, `clause_subsumes/2`, `stable_inclusion/1` | mixed, nondet | complete truth vectors are required; equivalence is reported generically as literal redundancy unless transform metadata justifies a stronger term; clause subsumption requires strict literal-set containment |
| `pta_escalation.pl` | `exception_clause/3`, `cotm_weight/3`, `graph_depth_increase/2`, `specialist_gate/2` | mixed, nondet | collective queries threshold and weight proposals only; target lowerers still fail closed when exact native semantics do not exist |

`PTACollectiveService` is the execution boundary. It resolves all four files
from one coherent checkout or installed-wheel data directory, maps field names
and 64-bit semantic example/literal/clause/class IDs to small opaque Prolog
integers, writes data-only facts, executes a bounded query, and decodes a small
framed record grammar into `PTAInsight` and `PTAEscalationProposal` objects.
Raw GNU Prolog stdout is not returned as an application API. Missing or
non-executable interpreters/modules, launch failure, timeout, nonzero exit,
streaming output overflow, and malformed protocol records are distinct typed
failures.

The execution budget independently caps encoded input bytes, captured output
bytes, observations, examples, literals, clauses, classes, total facts, and
results per product. Every product reports emitted/available counts, so a
researcher can distinguish a complete result from bounded truncation without
one high-cardinality category starving the others. De-escalation additionally
requires exactly one truth value per participating literal/clause for every
example in a nonempty evaluation domain.

GNU Prolog's portable integer ceiling and floating midpoint behavior are part
of the trust boundary. Semantic IDs are opaque-mapped; data terms are range
checked; threshold inputs use the exact arithmetic magnitude; and Python
independently verifies that returned boundaries lie strictly between an
observed label flip. Duplicate numeric values are consolidated into positive,
negative, or mixed states; mixed values are barriers rather than zero-width
thresholds or members of an exact positive interval. Rounded non-separating
midpoints fail as protocol errors.

Python validates, serializes, decodes, and later audits candidates. The
threshold/interval midpoint is computed by GNU Prolog, not duplicated in the
collective service. A typed proposal is still only a proposal: it must pass the
target-specific exact-lowering and behavioral-oracle gates before publication.

For threshold products, `review_threshold_proposal()` performs a non-mutating
review against the originating `PTAReasoningSession`.
`materialize_threshold_clause()` is the explicit approval boundary: it creates
the canonical numeric literal and derives a `binary_clause` proposal.
`compile_threshold_artifact()` then produces a deterministic
`packed_tm_binary_v1` clause-activation artifact whose origin and boundary
evidence remain in the manifest. The original `threshold` target continues to
fail closed in `lower_exact()`.

## Validation trust boundary

Python re-evaluates every Prolog candidate rather than trusting syntax:

* threshold: threshold semantics checked row-wise;
* template: coverage row `==` label row;
* TA clause: `TAClauseSearchResult.matches` checked;
* tree: `tree.evaluate(row) == label` plus `is_read_once`, `depth <= max_depth`, `feature < slot_count`.

CI compiles the bounded-search templates and all four PTA files, runs live GNU
Prolog service tests, and builds a wheel whose installed Prolog resources are
exercised from outside the source checkout.

See `python/prolog_tsetlin/prolog_bridge.py` docstrings for `ThresholdSearchProblem`, `FeatureTemplateSearchProblem`, `TAClauseSearchProblem`, `DecisionTreeSearchProblem`, and `GNUPrologSearch` class-level limits.
