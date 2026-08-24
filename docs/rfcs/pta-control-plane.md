# PTA Control Plane — Prolog as Reasoning/Control around the Native Tsetlin Substrate

> **Status:** design — Milestone 7 proposal. Native TM remains the fast executable representation; the PTA collective reasons, invents, and prunes, and every deployable result crosses a strict lowering gate.

See [Architecture and contracts](../architecture/index.md) for existing
contracts. The broader ecosystem survey is historical context and is not a
repository-local source.

## Two planes

```text
                   PTA REASONING / CONTROL PLANE
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   Input PTAs       De-escalation PTAs      Escalation PTAs   │
│       │                    ↕                       ↕          │
│ raw values ─────── shared symbolic knowledge ─── search      │
│ context             consolidated insights       invention    │
│ provenance          redundancy/pruning          exceptions   │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                    LOWERABILITY GATE
                           │
                 typed native representation
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                    NATIVE TSETLIN PLANE                      │
│ clause banks • TA states • weights • bit planes • SIMD/CUDA │
│ sparse clauses • patch evaluation • graph messages • artifacts│
└─────────────────────────────────────────────────────────────┘
```

*Prolog is more expressive during deliberation than the final model is during execution.* That asymmetry is the design.

## Lowerability gate (exact, no approximation)

Every deployable result is a `PTAEscalationProposal`:

```text
proposal_id, source_pta_ids, supporting_insights, counterexamples_addressed
required_literals { existing IDs, proposed raw-data transforms }
native_target ∈ { binary_clause, shared_weighted_clause, regression_clause,
                  patch_clause, graph_clause, logic_program, threshold, composite_gate }
structure, weights, output_assignments
resource_bounds { literal_count, clause_count, graph_depth, patch_extent, int_ranges }
lowering_version, validation_signature, support_trace
```

The Python `lower_exact(proposal)` service is the sole authority for exact
representability. Prolog may propose a declared target, but does not assert
that the target has an executable representation. Pipeline:

`PTA reasoning → typed proposal → lowerability checker → native candidate → oracle validation → shadow audit → publish` and, on drift, `reopen → restore`.

The current `pta.lowering.v2` contract uses `LoweredCandidate` only when an
executable object for the proposal's exact declared target has been constructed
and its target-specific semantic oracle passes. Proposal identity fields are
strictly typed at the trust boundary; floats, booleans, and arbitrary objects
are not coerced into integers or strings.

| Native target | Exact executable representation | Semantic oracle |
| --- | --- | --- |
| `binary_clause` | `ExecutableBinaryClause` over canonical, materialized literal descriptors | conjunction activation over the Booleanized substrate |
| `logic_program` | validated `LogicProgram32` | exhaustive comparison over all 32 A–E assignments |
| `threshold` | none; the target remains fail closed | fail closed |
| `shared_weighted_clause` | none until CoTM execution exists | fail closed |
| `regression_clause` | none until executable RTM semantics exist | fail closed |
| `graph_clause` | none | fail closed |
| `patch_clause` | none | fail closed |
| `composite_gate` | none | fail closed |

`ClauseConfiguration` remains an analysis/configuration record and is not an
exact lowering result. Adding a target to the executable set requires both a
constructor and an independent behavioral oracle.

One deliberately explicit promotion path is now implemented. A GNU
Prolog-invented threshold can be independently reviewed against its reasoning
session, previewed without catalog mutation, approved as a canonical
`numeric_ge` literal, and used to derive an exact `binary_clause` proposal.
That derived clause—not the original threshold target—can cross `lower_exact()`
and compile to a one-clause packed-TM artifact. See the current
[PTA threshold materialization contract](../architecture/pta-threshold-materialization.md).

## What each PTA class owns

**Input PTAs — literal-invention layer.** See raw values (`temperature=73.4`, `category`, `timestamp`, `position`, graph relations, prior observations) while native TAs see only instantiated Booleans. Propose `x ≥ 7.3`, `7.3 ≤ x < 9.8`, categorical groups, intervals. Escalation tests coverage; de-escalation collapses `x≥7.1..7.4` to the surviving boundary — *learned Booleanization with provenance and retirement*.

**De-escalation PTAs — Type-III-inspired reasoning + reversible absorption.** Reason over `literal_redundant/2`, `literal_subsumed/2`, `clauses_equivalent/2`, `clause_subsumes/2`, `thresholds_equivalent/2`, `stable_inclusion/2` etc., then propose simplifications. With shadow auditing/maturity/restoration, PTM does `ordinary TA → candidate frozen → shadow audit → consolidated` and can `reopen` on drift — reversible absorbing states. These equivalence/subsumption rules are a PTM control-plane analogue, not an implementation of the published Type III feedback algorithm.

**Escalation PTAs — structure inventors.** Question: *“current Boolean clause substrate fails here — what richer but lowerable structure fixes it?”* Answers: mutate clause, invent literal, join features, interval vs. threshold, share clause across outputs (CoTM), assign integer weight, add bounded graph-message condition, patch pattern, exception rule, specialist gate, simplification enabled by another PTA. Queries de-escalated knowledge — cumulative symbolic learning.

## Prolog strengths to exploit

* **Unification/relational matching:** `similar_failure/2` across class/category/region without enumerating Booleans. Richer than finite candidate-index interface.
* **Constraint solving (CLP(FD)):** `Clause×Output→Weight`, literal budget, threshold choice, clause subset covering counterexamples — all bounded integer assignments suited to constraints, not gradients.
* **Set reasoning:** de-escalation computes `unused/permanently-excluded/duplicate/subsumed` literals/clauses and zero-weight outputs → lowers to `SparseClauseBank`/`ClauseIndex`. Native executes without knowing Prolog derived it.
* **DCG/sequence reasoning:** `A followed by B`, `A … B within 3`, `A before B unless C` → fixed positional literals.

## Ecosystem additions mapped to PTAs

* **CoTM / weighted:** shared clause bank `C0..C5` + integer matrix `classes×clauses`. PTAs propose sharing, consolidate `C7≡C19`, Input PTA literal merges clauses. Native stays `TA states + int weights`.
* **FNS:** PTAs derive `confusable/2` / `confusable_when/3` → native `negative_candidates[ class] = bitmask`.
* **Multigranularity:** de-escalation flags `clause 17 covers 2 positives (over-specific)`, escalation `clause 22 covers 6 failure regions (over-broad)` → per-clause `s` schedule.
* **Regression (RTM):** native does continuous summation + error-dependent feedback; Input PTAs classify monotone/piecewise/outliers, escalation proposes clauses for residual error.
* **CTM:** C++/CUDA scans patches; Input PTAs expose `pixel/relative position/adjacency`, escalation invents `A above B` / `X within 2 of Y` compiled to patch-relative literals.
* **Graph TM:** escalation reasons with `edge/3`, `property/2`, recursive `reachable` during discovery but lowers via bounded unrolling `≤3 hops → depth 3`; arbitrary recursion is `not lowerable`.
* **Sequences/text:** escalation needs `R` → Input materializes bounded feature → de-escalation deduplicates against `Q`.
* **Composites:** escalation discovers `use(graph_model, E) :- has_relation_structure(E)`, lowers gate to bounded Logic program or smallest specialist subset covering validation.

Every item follows *Prolog determines structure; native repeatedly executes it* — millions of executions stay native, decisions about them are PTA-driven.

## Communication ontology (no free-form drift)

```prolog
observation(pta, example, field, raw_value).
feature_support(literal, pos, neg).
feature_relation(l1, subsumes, l2).
clause_support(clause, example).
clause_conflict(clause, example).
insight(source_pta, kind, subject, evidence).
counterexample(model, example, expected, actual).
proposal(pta, kind, candidate).
```

Shared by Input/de-escalation/escalation PTAs; graph/regression/composite PTAs consume cross-domain insights. Exact representability is deliberately absent from this ontology and remains owned by Python's target-specific `lower_exact()` gate.

## Implementation order (extends roadmap M5–M6)

1. **PTA proposal/message ontology + exact lowering contract + checker** — foundation.
2. Input-PTA adaptive numeric thresholds/intervals + literal budget.
3. Type-III-inspired de-escalation/subsumption + reversible absorption.
4. Shared weighted clause bank / CoTM with PTA allocation.
5. Multiclass/multilabel heads over that bank.
6. Sparse/indexed lowering from de-escalation knowledge.
7. Regression TM (PTAs on residuals).
8. CTM with Prolog-assisted spatial templates.
9. Graph TM learning (rich relational search → bounded lowering).
10. FNS + multigranularity schedules via PTA masks/policies.
11. Composite specialists + symbolic gates.
12. PTA-assisted hyperparameter/resource allocation.
13. Native parallel-training optimizations after semantics stabilize.

Partial reference implementation: `python/prolog_tsetlin/pta/` contains the
typed proposal model, target-specific fail-closed lowerers, bounded reasoning
session, and an executable GNU Prolog collective. The Input PTA can derive
threshold/interval insights from raw observations and labels and the collective
can decode de-escalation insights and escalation proposals through a
range-checked, completeness-reporting protocol. The scalar-binary Input-PTA
threshold slice now experimentally executes trained parent → invention → exact
extension → adaptation → conformance → promotion → publication → activation →
labeled drift → bit-exact restoration → fresh child activation across Python,
GNU Prolog, and `ptmrt`, with durable cross-generation evidence-use accounting.
That scalar-binary slice also supports complete bounded threshold candidate
sets across several numeric fields, independent per-candidate review,
adaptation-only deterministic comparison, and selection provenance that is
sealed before promotion holdout inspection. Intervals, categorical groups,
and the multi-PTA general control plane remain proposed.
