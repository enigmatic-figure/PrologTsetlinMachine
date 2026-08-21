# Logic dataset representational-exhaustion baseline

This baseline measures the point at which a flat propositional representation
stops carrying useful structure for evaluated logic programs. It deliberately
uses a noise-free dataset so representation loss, learner limitations, and
ordinary optimization variance can be separated.

## Protocol

The paired local sources are:

- `logical_problems_dataset.csv`: natural-language expression, bindings, and
  Boolean result;
- `logical_problems_symbolic.csv`: the same rows with a compact symbolic
  expression and `0`/`1` bindings.

The connector verifies all 5,000 paired labels, tokenizes the symbolic grammar
without accepting unknown fragments, and retains both source forms. Their
combined content digest in this run is:

```text
sha256:6542d61f86ee6073ab46d37951cb4fe8b378a54f6adba0346afe99525f76dc7f
```

The fixed split is stratified, uses seed `20260806`, and contains 4,000 training
rows and 1,000 evaluation rows. The dataset is nearly balanced: 2,527 true and
2,473 false rows. Generated binary matrices and their literal-provenance schema
are written under `out/logic-dataset` and are content-digested in the manifest.

## Bounded representations

Four Class I encodings expose progressively more source structure:

| Encoding | Features | Unique vectors | Optimistic collision ceiling | Evaluation vectors seen in training | Train-signature lookup accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Token presence + five bindings | 18 | 2,899 | 83.66% | 59.40% | 51.70% |
| Cumulative token-count thresholds + bindings | 65 | 4,941 | 99.42% | 2.20% | 50.30% |
| Token-at-position one-hot + bindings | 330 | 5,000 | 100.00% | 0.00% | 50.50% |
| AST counts/depths/edges/two-hop paths + bindings | 197 | 4,999 | 100.00% | 0.00% | 50.50% |

The optimistic collision ceiling groups the complete dataset by its encoded bit
vector and chooses the majority label inside each group. It is an upper bound
for any deterministic classifier receiving only those bits, even with access
to all labels. The train-signature lookup instead uses only training labels and
falls back to the training majority for unseen evaluation vectors.

Token presence therefore destroys answer-relevant information: 684 vectors,
covering 2,169 rows, require both labels. Counts nearly eliminate collisions but
also eliminate signature reuse. Position one-hot is injective for this dataset,
so failure there cannot be blamed on input collisions.

The AST-relational encoding is also label-consistent while replacing absolute
token locations with reusable typed structure. Its one duplicate vector carries
the same label in both rows.

## Scalar TM baseline

The first locked model uses the deterministic scalar semantic implementation,
100 clauses, 100 states per action, specificity 3.9, threshold 15, five epochs,
and seed `20260806`.

| Encoding | Train accuracy | Evaluation accuracy | Mean included literals/clause | Dead clauses | Unique clause behaviors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Token presence | 57.90% | 58.40% | 4.89 | 19 | 43 |
| Count thresholds | 58.17% | 55.40% | 17.88 | 12 | 85 |
| Position one-hot | 59.07% | 55.70% | 55.39 | 7 | 92 |
| AST relational | 57.93% | 58.80% | 36.29 | 8 | 87 |

All native results are lowered to the canonical logic IR and checked against
both compiled scalar and packed-64 execution before being reported.

## Capacity sweep

Evaluation accuracy after five epochs does not rise monotonically with more
clauses:

| Encoding | 20 clauses | 50 clauses | 100 clauses | 200 clauses | 400 clauses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Token presence | 55.5% | 56.0% | 58.4% | **60.4%** | 56.3% |
| Count thresholds | **56.8%** | 55.8% | 55.4% | 54.7% | 55.1% |
| Position one-hot | 56.6% | **58.1%** | 55.7% | 54.1% | 54.4% |
| AST relational | **63.3%** | 61.2% | 58.8% | 56.9% | 55.6% |

At the best five-epoch clause count for each encoding, extending training also
fails to produce sustained improvement:

| Encoding/configuration | 1 epoch | 5 epochs | 20 epochs | 50 epochs |
| --- | ---: | ---: | ---: | ---: |
| Presence, 200 clauses | 58.3% | **60.4%** | 57.8% | 56.9% |
| Counts, 20 clauses | 56.3% | 56.8% | 56.6% | **57.7%** |
| Position, 50 clauses | 56.3% | **58.1%** | 56.4% | 56.0% |
| AST relational, 20 clauses | 62.0% | 63.3% | **63.8%** | 63.0% |

This is a baseline, not a claim that every TM formulation is capped at these
scores. Specificity, feedback variants, clause weighting, convolution, and
alternative training schedules have not yet been swept.

## What exhaustion looks like

The four encodings fail or plateau for different reasons:

1. Token presence is representationally exhausted. Opposite programs collapse
   to identical inputs, so no learner can recover their distinction.
2. Token counts retain multiplicity but discard order, scope, branch ownership,
   and variable-binding relationships. The near-perfect full-data collision
   ceiling hides the fact that almost every evaluation signature is new.
3. Position one-hot retains the complete token stream, but a flat clause must
   relearn equivalent subexpressions at different absolute positions and depths.
   It supplies the information without supplying the reusable relations needed
   to interpret it.
4. AST-relational features replace absolute positions with typed counts, depths,
   edges, and two-hop paths. They raise the observed frontier to 63.8% with only
   20 clauses, demonstrating useful structural reuse. Complete-tree vectors are
   still novel, and bounded local paths do not provide recursive composition.

Across the sweeps, more clauses and epochs tend to increase included-literal
counts and low-state saturation without creating a durable accuracy trend.
Clause behavior diversity eventually stalls or declines. That is the signature
we will use for escalation: information-preserving input, sustained feedback,
poor marginal accuracy, growing clause specificity, and weak semantic reuse.

The implemented Class I front end now exposes the typed expression tree,
parent/child and branch relations, operator identity, and bound variable values.
Class III can next search reusable evaluator rules, and Class II can compile
stable subexpression behavior back into fast native artifacts or new
propositional features.

That structural compilation path is now executable. The fixed Class II Logic
program reaches 100.0% on both training and held-out evaluation rows with zero
cross-evaluator mismatches. This does not revise the 63.8% flat-TM frontier;
it closes the gap through the higher-resolution compiled route described in the
[Logic Class II consolidation record](../architecture/logic-class-ii-consolidation.md).

## Reproduction

```powershell
.\scripts\benchmark-logic-dataset.ps1 -Epochs 5 -Clauses 100 -Repeats 1
```

The preparation step records the split digest, exact literal mapping, collision
statistics, truncation counts, and binary-file digests in
`out/logic-dataset/logic_baseline_manifest.json`.
