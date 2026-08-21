# Architecture record 0001: hybrid runtime boundaries

Status: accepted for the initial vertical slice.

## Decision

PTM is divided into six layers with versioned interfaces between them.

| Layer | Initial implementation | Responsibility |
| --- | --- | --- |
| Representation | Python | Raw/typed records, bounded transforms, typed syntax/relations, Boolean literals, and reversible provenance |
| Adaptive substrate | C++20 plus Python oracle | TA states, clauses, feedback, snapshots, and capacity allocation |
| Fredkin path | C++20 plus Python oracle | Lossless controlled swaps and conservative signal routing |
| Logic compiler | C++20 | Canonical operator DAG, simplification, TA/rule lowering, and backend planning |
| PA runtime | C++20 | Fixed-shape buffers, slot maps, compiled kernels, validation, and restoration handles |
| Symbolic control | GNU Prolog plus Python orchestration | Bounded search, rule checking, explanation, and artifact generation |

The Python scalar model is the executable semantic specification. Optimized
native implementations must pass the same golden-vector and snapshot tests.

## Data flow

```text
raw record
   |
   v
Class I representation -----> typed symbolic facts
   |
   +-------------------------> literal provenance ledger
   |
   v
packed literal-truth plane
   |
   v
TA/Fredkin adaptive substrate
   |                    ^
   | candidates         | repair/reactivation
   v                    |
Class II compiler/auditor <----> canonical logic IR
                                      |
                                      v
                             planned native kernels
   |
   | unresolved region
   v
Class III bounded search -----> feature | PA artifact | TA configuration
```

## PA execution model

The first native PA ABI supports 1024-bit (32x32) and 4096-bit (64x64)
payloads. Buffers are aligned to 64 bytes and expressed as arrays of 64-bit
words. A shared kernel consumes immutable instance descriptors and mutable
input buffers. We do not patch data into executable code.

The portable masked-threshold kernel is the first baseline:

```text
value = popcount(input AND selected_slots) >= minimum_true
```

It represents conjunctions, disjunctions, and cardinality thresholds while
returning diagnostic masks. SIMD specializations can replace it after the
baseline is profiled. Runtime dispatch must retain a portable implementation.
The versioned C ABI exposes both fixed shapes, and the Python binding checks its
ABI version before evaluation.

The structural Logic path adds a separate 32-instruction, 320-byte aligned
state block. Its shared native evaluator consumes a topological Boolean program
and five binding bits, returning the result and instruction-truth diagnostics.
This uses one immutable engine across many resident program states; it does not
modify executable code. See the
[Logic Class II consolidation record](logic-class-ii-consolidation.md).

Arbitrary rule structure lowers through a separate canonical Boolean DAG before
backend selection. Its first executors are a scalar semantic path and a
feature-major, 64-example packed CPU path. The graph preserves signed TM voting
with weighted thresholds. Optional sparse, warp-tile, and dense-bitset CUDA
clause backends attach at the packed TM boundary under device-specific
crossover measurements. Their vote axis independently selects a conventional
clause-output scan or fused atomic score aggregation; tensor execution is not
assumed to be appropriate for Boolean clause evaluation. See the
[logic compiler record](logic-compiler.md).

The adaptive substrate also has a direct prepared-TM path. Exact clause-major
TA state is bit-sliced across the automaton population, while feature-major
`uint64_t` inputs carry 64 examples per feature. The portable evaluator emits
both prediction and feedback clause words, clamped signed scores, and a
prediction mask without routing through the general graph. This immutable
snapshot image is replaced after learning changes its source states. See
[packed TM execution](packed-tm.md).

## Deployment boundary

The training system's deployable product is a deterministic, content-addressed
inference artifact rather than a mutable native object or backend-specific
memory image. The standardized exporter freezes packed-TM snapshots, fixed
Logic programs, and PA masked-threshold artifacts; lowers them to canonical
inference semantics; attaches their feature, binding, slot, and input/output
contracts; and reloads the emitted bytes for oracle comparison. All three use
the same container and task-neutral runtime API.

A separate `ptmrt` library consumes those artifacts through an independently
versioned, task-neutral C ABI. It can inspect a model and execute named tensor
ports from a file or memory buffer without Python, Prolog, training state, or a
mandatory accelerator. The scalar implementation is the portability baseline;
SIMD and CUDA remain optional exact dispatch targets. Declarative artifacts may
be embedded in applications or model containers, but they never contain
arbitrary executable code. See
[portable artifact and runtime reference](manual/reference/artifact-contract.md).

Packed-TM artifacts may also embed `ptm.preprocessing.v1`, an ordered and
bounded transform contract for numeric thresholds/ranges, typed categorical
equality/membership, Boolean values, and missingness. The Python and standalone
C++ paths enforce the same non-coercing value rules and stable literal order.
Connector parsing and richer transforms remain host responsibilities unless a
later portable contract explicitly versions them. See
[deterministic raw-record preprocessing](manual/reference/preprocessing.md).

## Prolog boundary

GNU Prolog is used as a compiler/search participant, not as the hot Boolean
evaluator. A Prolog result must lower to the versioned PA artifact schema before
it can enter inference. The accepted outputs are:

- a new Class I literal descriptor;
- a compiled Class II Boolean/threshold kernel artifact;
- a native TA clause configuration.

Unbounded recursion, arbitrary dynamic predicates, and unrestricted foreign
calls are excluded from generated inference artifacts. Search limits and the
allowed Horn-clause subset will be explicit and versioned.

The available templates perform exact monotone masked-threshold search, typed
feature-template selection, signed TA-clause search, and read-once Boolean
decision-tree search. Their finite candidate counts are checked in Python before
GNU Prolog is launched, and a subprocess deadline supplies a second bound.
Python re-evaluates every returned structure before it can instantiate a literal
catalog, emit a TA-clause configuration, lower to a fixed Logic program, or
create a content-addressed PA artifact. Counterexample repair leaves the parent
immutable and synthesizes a bounded XOR guard until full finite validation has
zero mismatches or an explicit iteration/search bound is exhausted.

## Consolidation lifecycle

```text
observe -> nominate -> validate -> compile -> shadow -> activate
                                            |          |
                                            v          v
                                         reject   audit continuously
                                                       |
                                  patch | branch | reopen | dissolve
```

Class II compilation absorbs behavior, not merely memory. Activation requires a
validation signature and a restorable source snapshot. A kernel remains under
shadow comparison after activation. Capacity is released only after the
configured acceptance window.

For the Logic corpus, specified operator behavior is compiled from the typed AST
into the fixed program state, then audited rather than learned from labels. This
artifact may serve a derived Boolean output immediately after exact shadow
acceptance, but TA capacity is not considered recycled until the contributing
clauses and restoration snapshot are explicitly bound through the registry.

Morphology never edits an active program. Specialization, generalization,
counterexample repair, conditional factoring, and equivalence merging create
immutable child programs with exhaustive behavior signatures. An audited child
replaces its parent through generation-tagged atomic rebinds; the child is
invisible until every source has moved. See the
[Logic morphology record](logic-morphology.md).

Candidate connectivity and active routing use different structures. A
rebuildable Union-Find groups nomination candidates. Once compiled, dense
source handles resolve through generation-tagged atomic words; artifact state
provides the publication barrier for group activation and reopening. See the
[Class II lifecycle](class-ii-lifecycle.md) for the concurrency contract.

The lifecycle control plane checkpoints immutable registry images containing
policies, artifact states, audit aggregates, mapping words and their exact
generations, and restoration references. Full post-transaction images form a
SHA-256-chained append log between atomic snapshots. Recovery constructs and
validates a new unpublished registry, rejects partial `activating` state, then
allows the orchestrator to publish it. See
[Class II persistence](class-ii-persistence.md).

## Reversibility boundary

Fredkin gates are bijective only when all outputs, including garbage lines, are
retained. This gives us reversible routing and makes data-path reconstruction
possible. Ordinary stochastic, saturating Tsetlin feedback is not itself
bijective. Exact learning rollback is therefore implemented through explicit
snapshots and event logs; it is not inferred from a projected gate output.
