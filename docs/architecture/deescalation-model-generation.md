# De-escalation PTA model generations

> Status: current experimental architecture contract. This implements one
> scalar-binary literal-equivalence contraction slice. Clause removal,
> subsumption-driven contraction, automatic rehabilitation, and the general PTA
> control plane remain proposed.

The first De-escalation lifecycle exercises the trained-parent substrate in the
opposite direction from Input-PTA threshold invention:

```text
frozen trained parent P
        ↓ complete literal truth vectors on bounded proof evidence
GNU Prolog literal-equivalence derivation
        ↓ independently replayed and attested
deterministic one-literal contraction P-
        ↓ separate Python confirmation corpus
scalar / packed / ptmrt conformance
        ↓ independent paired promotion holdout
immutable contracted child C and lineage
        ↓ activation → labeled drift → reopen
bit-exact restoration of P
```

This is vocabulary contraction, not a claim that two literal descriptors are
universally equivalent. The evidence establishes equality on explicit bounded
corpora, while promotion and post-activation drift govern behavior elsewhere.

## Complete bounded proof

`invent_literal_contraction_for_corpus()` encodes every literal in the parent's
ordered manifest for every row of a `deescalation_proof` corpus. Literal IDs and
truth values enter a bounded `PTAReasoningSession`; no model state, artifact, or
catalog is mutated. The collective runs only De-escalation products:

```text
numeric_fields = ()
discover_thresholds = false
discover_intervals = false
derive_deescalation = true
derive_escalation = false
```

The result must contain at least one `literal_redundant` pair and must fit the
declared result budget without truncating literal-equivalence, literal-
subsumption, or clause-subsumption products. The durable
`PrologDeescalationEvidence` binds:

- the proof-corpus and complete reasoning-session digests;
- the parent snapshot and ordered-manifest identities;
- the exact query and candidate budget;
- every canonical equivalent pair and the deterministic selected pair; and
- the GNU Prolog version, executable digest, protocol, and packaged module
  digests measured before and after execution.

Python does not trust the returned pair. The collective service first validates
each redundancy against complete Python truth vectors. Candidate admission and
recovery then reconstruct the session from durable proof rows, rerun the
attested GNU Prolog query, require the complete pair set to match, and repeat
the deterministic selection. The earlier feature position survives; the later
equivalent position is removed.

`literal_redundant` does not become a new `lower_exact()` target. It authorizes
a model-generation transformation whose smaller snapshot and portable packed
artifact are independently reconstructed and checked. Unsupported PTA targets
continue to fail closed.

## Deterministic TA contraction

For a surviving feature position `s` and removed position `r`, every clause
consolidates like-polarity TA states before splicing the removed pair:

```text
positive[s] = max(positive[s], positive[r])
negative[s] = max(negative[s], negative[r])
remove positive[r], negative[r]
```

The maximum retains inclusion whenever either equivalent automaton was
included and preserves the stronger state when both have the same action. All
unrelated TA states, clause order, model configuration, and RNG state are
copied exactly. The literal manifest is likewise positional: it removes only
the selected later descriptor and preserves every other descriptor byte for
byte and in order.

Python requires the two literal columns to agree on both the Prolog proof
corpus and a separate `deescalation_confirmation` corpus. It then compares the
complete per-clause output vector, score, and prediction of P and P- on every
row of both corpora. Any difference rejects the episode. The componentwise
state merge deliberately creates a new adaptive behavior and does not promise
the parent's future training trajectory; rollback restores P when adaptive
continuation is required.

## Promotion and deployment

The first slice performs no child training after contraction. A
`contracted_parent` candidate and deployable `contracted_child` therefore name
the same snapshot, ordered manifest, and preprocessing contract; deployment
packaging and promotion evidence give the child its distinct generation ID.
Its `AdaptiveBehaviorIdentity` remains solely the snapshot, manifest,
preprocessing contract, and versioned training semantics.

Promotion uses a third immutable corpus. Runtime conformance remains exact:
the scalar snapshot, packed Python artifact, embedded conformance vectors, and
`ptmrt` must agree with zero mismatches. The paired promotion audit compares P
and C against labels. Because an exact contraction is a consolidation rather
than an improvement episode, the first policy requires:

```text
minimum observations reached
child errors <= parent errors
regressions == 0
runtime conformance exact
```

Strict-improvement promotion is intentionally rejected by this entry point.
The proof, confirmation, and promotion corpora must share one dataset while
using disjoint observation IDs and distinct labeled-row fingerprints. Their
complete rows are reserved before Prolog execution under the durable
`deescalation_episode` evidence purpose. Failed attempts are abandoned but
remain spent across future generations.

The content-addressed `LiteralContractionLineage` binds the parent, contracted
candidate, deployable child, adaptive behavior, restoration bundle, promotion
audit, De-escalation evidence, evidence usage, activation sequence, selected
literal IDs, and all three corpus digests. Controller replay reloads the child
artifact and complete object graph, reruns GNU Prolog, reconstructs the exact TA
contraction, and resolves the parent restoration bundle before accepting any
candidate, activation, or recovered route.

## Drift and restoration

The contracted model can diverge when future records distinguish descriptors
that were equal on the bounded proof and confirmation evidence. That is an
expected audited condition, not evidence that the Prolog derivation was false.
Live reopening uses fresh labels, fresh scalar/packed/native conformance, and
the same explicit `DriftAuditPolicy` as the Input-PTA lifecycle. A qualifying
regression window restores the original parent snapshot, ordered manifest,
preprocessing contract, artifact, and RNG state through its immutable
`AdaptiveRestorationBundle`.

The integration contract deliberately uses two numeric thresholds that agree
on proof, confirmation, and promotion rows but differ in a later threshold-gap
window. It proves contraction, native activation, labeled regression,
bit-exact restoration, and equality after the next parent training update.
The consumed contracted behavior ID cannot be repackaged and activated again
after restoration; genuinely new adaptive behavior remains eligible through
the recurrent generation controller.

## Current boundary

Implemented:

- one selected literal-equivalence pair per episode;
- complete bounded GNU Prolog derivation and Python replay;
- independent proof, confirmation, promotion, and live evidence;
- deterministic scalar snapshot contraction;
- portable packed artifact publication and `ptmrt` conformance;
- durable activation, drift reopen, and bit-exact parent restoration.

Not implemented by this slice:

- literal-subsumption or clause-subsumption transformations;
- removal of more than one literal in one episode;
- clause-bank contraction or sparse native lowering;
- a proof of universal descriptor equivalence;
- rehabilitation of a retired behavior; or
- CoTM/shared weights, TAUs, TSUs, graph, regression, patch/CTM, and the broad
  multi-PTA control plane.

## Evidence

- `python/prolog_tsetlin/model_generation.py`
- `python/prolog_tsetlin/services/model_generation.py`
- `prolog/pta_deescalation.pl`
- `tests/python/test_model_generation_lifecycle.py`
- `tests/python/test_reference_tm.py`
