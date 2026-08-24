# Trained-parent PTA model generations

> Status: current experimental architecture contract. This closes recurrent
> trained-parent threshold episodes for the scalar binary reference and
> portable packed runtime. It does not make the broader PTA control plane or
> other TM families implemented.

The trained-parent lifecycle separates representation from adaptation:

```text
frozen trained parent P
        ↓ GNU Prolog threshold invention and independent review
append-only extended parent P+
        ↓ exact structural and behavioral-equivalence proof
bounded adaptation
        ↓
behavior-changing child C
        ↓ reference/packed/ptmrt conformance and paired promotion audit
immutable generation publication and activation
        ↓ labeled drift
reopen and bit-exact restoration of P
        ↓ fresh, durably reserved evidence
genuinely new child C2 through the same bounded contract
```

## P to P+ is not learning

Literal identity alone does not preserve a trained model. `LiteralCatalog` and
`LiteralBatch.literal_ids` are insertion ordered, while every scalar clause
stores an interleaved positive/negative TA pair for each feature. A valid
extension therefore satisfies:

```text
P+.literal_ids[:P.feature_count] == P.literal_ids
P+.literal_ids[P.feature_count:] == approved_new_literal_ids
```

Every existing canonical descriptor and TA state remains byte-for-byte equal.
For each new feature, every clause appends exactly:

```text
(states_per_action, states_per_action)
```

`action_include()` is true only above `states_per_action`; both newborn TAs are
therefore excluded. The parent RNG state is copied without constructing or
advancing another random stream. The structural conditions prove that the new
literal has no inference or feedback effect for either truth value. An
independent raw-record oracle over the already-authorized parent-training
corpus and exhaustive small-feature tests provide defense in depth. Invention,
adaptation, promotion, and future live rows are not inspected by P+ creation.

Threshold approval still uses the existing boundary:

1. GNU Prolog returns the sole bounded threshold proposal considered by the
   first lifecycle.
2. `review_threshold_proposal()` validates it without catalog mutation.
3. A cloned parent catalog is reconstructed in its declared order.
4. `materialize_threshold_clause()` appends the canonical descriptor and
   crosses `lower_exact()` only as a derived `binary_clause`.
5. The frozen parent catalog and snapshot remain untouched.

The supported lifecycle performs GNU Prolog invention internally; it does not
accept a caller-supplied proposal that merely claims a Prolog origin. It
publishes content-addressed invention evidence binding the exact invention
corpus and reasoning session, collective query/protocol, GNU Prolog executable
digest and version, packaged module digests, and returned proposal identities.

The original `threshold` proposal remains `NotRepresentable`.

## P+ to C is bounded adaptation

Adaptation restores a new scalar machine from the immutable P+ snapshot and
trains only on the declared adaptation corpus for a bounded epoch count. It
never mutates P or P+. The resulting child carries a new content-addressed
adaptive snapshot and uses the P+ ordered literal manifest.

Adaptive behavior and deployment packaging have separate identities. An
`AdaptiveBehaviorIdentity` binds the child snapshot, ordered manifest,
preprocessing contract, and versioned scalar training semantics. P+ lineage,
adaptation evidence, proposal identity and provenance, promotion evidence, and
the inference artifact remain deployment/lineage claims. They cannot change the
consumed identity of an unchanged adaptive state. Repackaging that state against
a new holdout—or claiming a different adaptation corpus—can therefore produce a
different generation ID without producing a different behavior ID.

The lifecycle distinguishes these immutable corpora:

| Corpus | May be inspected by | Purpose |
| --- | --- | --- |
| parent training | original trainer | establish P |
| invention | GNU Prolog and threshold review | discover the new distinction |
| adaptation | child trainer | produce C from P+ |
| promotion holdout | audit only | compare P and C with labels |
| live/drift | post-activation audit | decide whether C became worse |

Every corpus has a canonical SHA-256 digest. Lifecycle example IDs are
disjoint, and identical labeled rows cannot cross invention, adaptation, and
promotion. Live data may deliberately revisit covariates after a concept
change, including with a changed label, but the same labeled observation is
not reusable merely by assigning it a new example ID.

`LifecycleCorpora` contains only the three pre-activation roles. Live/drift
evidence enters through `reopen_and_restore_for_drift()` after activation.
Episode-local separation is supplemented by a durable cross-generation
ledger. An immutable `EvidenceUsage` stores the exact labeled corpora for one
of three purposes: parent registration, candidate episode, or live drift. Its
content identity covers the dataset, purpose, subject generation, corpus
digests, observation identities, raw records, and labels.

Candidate and live evidence is committed through `evidence_reserved` before
GNU Prolog invention, adaptation, promotion inspection, or native drift
evaluation begins. A crash, rejection, failed native check, or other abandoned
attempt appends `evidence_abandoned`, which terminalizes that reservation
without making its observations fresh again. Controller recovery likewise
terminalizes a reservation left pending by a process interruption. An
abandoned reservation remains in the spent-evidence ledger but can never
authorize candidate creation or reopen. The strict v1 policy
rejects both a previously used observation identity and a previously used
`dataset + record + label` fingerprint across every role and episode. This is
intentionally conservative; any future relaxation requires a versioned policy
rather than an implicit exception.

## Conformance and promotion are different audits

Runtime conformance asks whether the child snapshot and portable execution
mean exactly the same thing. Scalar reference and packed Python predictions
must have zero mismatches, the artifact's embedded oracle must pass, and
`ptmrt verify` must return the same artifact identity. There is no tolerance.
Before a live drift decision, the service additionally runs every bounded raw
live record through `ptmrt run-record`. Native preprocessing outputs, score,
prediction, and artifact identity must exactly match the scalar child and
packed Python artifact on that live window. This execution is owned by
`ModelGenerationController.request_reopen()`; callers cannot authorize reopen
by supplying a preconstructed conformance report or verification Boolean. The
controller publishes a content-addressed `LiveRuntimeConformanceEvidence`
receipt containing the exact labeled live corpus, child state/artifact
identities, scalar feature/score/prediction vectors, packed predictions, native
feature/score/prediction vectors, and the `ptmrt` executable digest. That digest
is not treated as self-authenticating metadata: recovery and restoration hash
the controller's trusted `ptmrt` executable and rerun it over the receipt's
exact corpus before accepting the durable authorization.

Promotion asks whether C is preferable to P on the independent labeled
holdout. Each paired observation is classified as:

```text
both correct
both wrong
parent wrong / child correct  (improvement)
parent correct / child wrong  (regression)
```

The report also retains parent and child errors, disagreements, scores, and
class-stratified counts. The first-loop policy requires a minimum sample count,
strictly fewer child errors, at least one improvement, zero regressions, and
exact runtime conformance.

Parent/child disagreement without ground truth is not proof of drift. Reopen
uses an explicit `DriftAuditPolicy` with minimum observations, regressions,
regression rate, error increase, and per-class coverage, and still requires
more regressions than improvements. The exact policy is recorded with the
durable reopen event and revalidated during restoration. The strict fixture
policy rejects a single labeled regression; hysteresis and consecutive-window
policies remain future generalization work.

## Model-generation lineage sits above Class II

Class II `replace_active()` intentionally requires identical source IDs, source
kinds, input shape, and port semantics. An appended literal changes the source
vocabulary, so P+, C, and P are model generations above that same-source
registry. The Class II contract is not weakened or used to smuggle feature
expansion through morphology replacement. Each model generation may continue
to own its own Class II registry.

The UI-neutral `ModelGenerationController` derives its active generation from
an immutable SHA-256-chained event log. Candidate creation, promotion approval,
activation, reopen request, and parent restoration are separate events; the
content-addressed lineage node is never edited to carry lifecycle state.
Activation consumes the adaptive behavior identity. Candidate recording scans
the durable activation history and rejects any lineage whose behavior ID was
previously activated, even if the same snapshot and manifest are repackaged as
a new deployment generation with a new artifact or promotion holdout.
Restoration reopens the parent for genuinely newly adapted behaviors. The
experimental controller has no rehabilitation mechanism for a displaced
behavior.

Each lineage also records its expected activation sequence and the previously
activated lineage. C2 therefore names C1 as its episode predecessor even when
both children branch from the same bit-exact restored P0. These fields do not
authorize transitions: replay recomputes them from prior `activated` events,
and rejected candidates continue to occupy only the still-unfilled next
activation slot.

## Durability and restoration

Before any evidence may affect an episode, the store durably publishes its
exact `EvidenceUsage` and appends `evidence_reserved`. Before child activation,
the store then durably publishes:

1. the parent's adaptive restoration bundle;
2. P+ and C adaptive snapshots;
3. ordered literal and preprocessing manifests;
4. the child `.ptm` bytes;
5. the paired audit and immutable lineage node.

Every object is content addressed. Atomic publication synchronizes the
temporary file and, on POSIX, the containing directory after link or rename.
Typed loads recompute the object's identity and require it to equal the
requested address; a different valid object placed at that path fails closed.
The lifecycle event log is atomically replaced as a complete hash chain and a
separate durable head binds its terminal event ID, sequence, and complete-log
digest. Removing even a complete valid suffix therefore fails closed. The log
is published before its head; a process or power loss between them leaves a
detectable mismatch, while an ordinary publication exception attempts to
restore the prior complete checkpoint. The controller changes its in-memory
generation only after the activation or restoration event is durably appended.
Store instances in one process share a root-scoped event lock; cross-process
writers remain outside the experimental single-process contract.

Recovery replays the complete lifecycle state machine rather than selecting the
last routing-shaped event. Every frame must consume its required predecessor
with matching lineage, audit, behavior, bundle, policy, error counts, evidence
usage, activation sequence, and active route. Replay reconstructs the complete
cross-generation set of spent observation identities and labeled-row
fingerprints. A forged but hash-valid reservation that reuses either fails
closed. In particular, activation requires an immediately valid accepted exact
promotion, and parent restoration requires a valid reopen request whose labeled
drift satisfies its recorded policy. A hash-valid but impossible event makes
controller construction fail closed. Reopen replay and restoration also reload
the live-conformance receipt, reconstruct scalar features and scores and packed
predictions from the durable child and exact corpus, require the native vectors
to agree, verify the configured `ptmrt` bytes against the receipt digest, rerun
native inference, and reconstruct the complete paired drift audit. Controllers
recovering a history that contains a reopen therefore require the trusted
`ptmrt` path at construction. A self-asserted `ptmrt_verified` Boolean or a
structurally valid receipt fabricated from scalar outputs cannot authorize
restoration.

The same replay engine is authoritative while the process is live. It returns
the active route, registered dataset, pending reservation, candidate,
promotion, activation, reopen, activation ordinal/predecessor, retired
behaviors, and spent observation/fingerprint sets as one structured state.
Every mutating controller method replays and compares that durable route with
its cached route while holding the shared event lock, authorizes the proposed
transition from the replayed state, and appends before releasing the lock.
Native drift evaluation reserves and then releases the event lock while
`ptmrt` runs, but reacquires it and replays again immediately before committing
`reopen_requested`. Thus a hash-valid raw event tail cannot be accepted by a
running controller when the same tail would be rejected after restart.

Before any parent or child is registered, activated, restored, or recovered as
the active route, one deployability contract reloads and cross-validates its
snapshot, ordered manifest, preprocessing, and inference artifact. It checks
the artifact's feature order, exact TA-derived inclusion masks, threshold,
embedded conformance cases, identity signature, and role-specific evidence.

Before candidate recording, promotion, activation, and active-child recovery,
the controller additionally reloads and cross-validates the complete lineage
graph. This includes generation links and kinds, P/P+/C manifests and
snapshots, the appended invented literal, all corpus digests, proposal and
invention-evidence IDs, preprocessing order, audit/artifact identity, and
restoration parent.
The child `.ptm` is opened as part of that graph validation: its embedded
adaptive-snapshot, ordered-manifest, preprocessing, corpus/proposal, and
restoration signatures must all agree with the durable child generation. A
different valid artifact that merely matches the finite holdout cannot be
attached to the child's adaptive lineage.

An `AdaptiveRestorationBundle` binds the parent generation to:

- its full `TMSnapshot` TA states and RNG state;
- the ordered literal manifest and feature schema;
- the preprocessing contract;
- the deployed parent artifact;
- the parent training-corpus digest; and
- versioned training and RNG semantics.

Reopen verifies and restores all of those objects before resuming the parent.
Restoration is not a free-standing routing API: the active child must have a
durably stored labeled live audit, a matching lineage/restoration bundle, and
the immediately preceding `reopen_requested` transition. The parent generation,
artifact signature, and bundle must agree on the snapshot, manifest,
preprocessing, artifact, and parent-training digest.
The integration contract proves snapshot equality, literal-manifest equality,
prediction equality, and equality after the next training update. This is a
bit-exact Python-to-Python restoration guarantee for the recorded compatible
runtime. `python.random` state is explicitly versioned; long-term
cross-version or Python/C++ adaptive continuation is not claimed. A future
PTM-owned PRNG can replace that dependency under a new snapshot schema.

## Typed events

The service emits lifecycle events such as `parent_registered`,
`proposal_created`, `shadow_completed`, `artifact_published`, `activated`,
`reopen_requested`, and `artifact_reopened` through the existing telemetry
envelope. Existing workbench Events and related projections can consume them
without reconstructing private state or adding another lifecycle engine.
Telemetry is observational: a caller-supplied sink failure is retained on the
controller for inspection but cannot reverse or report failure for an already
durable lifecycle transition.

## Evidence

- `python/prolog_tsetlin/model_generation.py`
- `python/prolog_tsetlin/services/model_generation.py`
- `tests/python/test_model_generation_lifecycle.py`
- `tests/data/trained_parent_child_v1.hex`
- the `ptmrt_trained_parent_*` CTest cases
- the required `Trained-parent GNU Prolog / native lifecycle` CI job

The next PTA breadth milestone should prove a De-escalation PTA lifecycle under
this recurrent promotion/restoration substrate. Bounded multi-candidate Input
PTA selection is also still future work. CoTM/shared weights, regression,
graph, patch/CTM, and other families remain outside the implemented exact
target set.
