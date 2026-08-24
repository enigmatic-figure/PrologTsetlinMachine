# PTA threshold materialization

> Status: current experimental architecture contract. This is one completed
> vertical slice of the proposed PTA control plane, not the complete adaptive
> publication and reopen loop.

GNU Prolog Input PTA thresholds now have one explicit route into the portable
native substrate:

```text
observations + labels
        ↓
GNU Prolog invent_threshold/2
        ↓
PTAEscalationProposal(target = threshold)
        ↓  independent Python boundary review, no catalog mutation
ReviewedThresholdProposal
        ↓  explicit approval
canonical numeric_ge LiteralDescriptor
        ↓
derived binary_clause proposal
        ↓
lower_exact() + conjunction oracle
        ↓
ExecutableBinaryClause
        ↓
deterministic packed-TM compilation
        ↓
.ptm → Python runtime / ptmrt
```

The original `threshold` target remains `NotRepresentable`. Materialization is
not a new threshold lowerer: it changes the declared semantics by creating a
real catalog literal and a derived `binary_clause` proposal. Only that derived
proposal may produce `LoweredCandidate`.

## Review and approval boundary

`review_threshold_proposal()` accepts only the typed Input-PTA product shape:

- target `threshold`;
- structure `{field, operator: ge, threshold}`;
- one matching `numeric_ge` required-literal declaration;
- matching Input-PTA threshold insight;
- one-literal resource bound; and
- no weights or output assignments.

The review independently reconstructs the observed numeric states from a
`PTAReasoningSession`. Duplicate values are consolidated. A mixed-label value
is a barrier. The proposed boundary must lie strictly between two adjacent,
pure, differently labeled observed values. Review uses
`LiteralCatalog.preview_numeric_ge()`, so failure cannot partially mutate the
catalog.

`materialize_threshold_clause()` is the approval operation. It registers the
reviewed descriptor, derives a one-literal `binary_clause` proposal, and calls
the authoritative `lower_exact()` gate. The derived proposal preserves the
origin proposal's semantic and provenance IDs plus a digest of the labeled
observations used for review.

The observation digest is deliberately field-scoped: changing an unrelated
field does not stale a threshold whose raw transform and evidence are
unchanged. Numeric parameter types are also identity-bearing. Promotion never
coerces `75` to `75.0`; semantically equivalent typed thresholds may therefore
have distinct literal IDs and can later be considered by explicit
de-escalation reasoning.

## Native compilation semantics

`compile_threshold_artifact()` compiles the exact one-literal conjunction to a
minimal packed-TM image:

| Quantity | Value |
| --- | ---: |
| Boolean features | 1 |
| clauses | 1, positive polarity |
| included literals | the materialized positive literal |
| negative literals | none |
| vote threshold | 1 |

The embedded preprocessing contract evaluates the raw numeric field with
inclusive `numeric_ge` semantics. Conformance includes an observed value below
the boundary, the boundary itself, and an observed value above it. Export then
checks the scalar snapshot against the packed evaluator; the compiler checks
the packed artifact against `ExecutableBinaryClause`; the golden artifact is
also executed by the C++ runtime and `ptmrt verify`/`run-record` tests.

This artifact predicts **clause inactive/active**. It does not claim to be a
trained classifier for the source labels, because a discovered feature may be
used with either polarity or as part of a larger clause bank. That limitation
is recorded in the manifest.

## Provenance carried by the artifact

The validation signature records:

- origin threshold proposal semantic and provenance IDs;
- derived binary-clause proposal semantic and provenance IDs;
- dataset and field;
- lower/upper values and labels around the boundary;
- a content digest of all labeled observations for the field; and
- the exact semantic-oracle target and mismatch count.

Publication uses the shared atomic artifact publisher. The resulting bytes are
ordinary `packed_tm_binary_v1` and require neither Python nor GNU Prolog after
opening in `ptmrt`.

## Evidence

- `python/prolog_tsetlin/pta/threshold_artifact.py`
- `tests/python/test_pta_threshold_artifact.py`
- `tests/data/pta_threshold_clause_v1.hex`
- `tests/cpp/model_runtime_tests.cpp`
- the `ptmrt_pta_threshold_*` CTest cases

The next control-plane milestone remains broader: incorporate a PTA-derived
literal into a trained parent model, run a shadow audit, publish a lineage
child, and prove reopen/restore under drift.
