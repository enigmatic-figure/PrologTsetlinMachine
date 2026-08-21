# Bounded-search contracts

PTM validates every `ptm.search.request.v1` document before starting GNU
Prolog, including its candidate ceiling, example count, deadline, and
kind-specific structural bounds.

## Request contract

```json
{
  "schema": "ptm.search.request.v1",
  "kind": "decision-tree",
  "timeout_seconds": 30,
  "problem": {
    "slot_count": 2,
    "max_depth": 2,
    "examples": [[], [0], [1], [0, 1]],
    "labels": [0, 1, 1, 0]
  }
}
```

Unknown top-level fields are rejected. The deadline is between 0.1 and 300
seconds. JSON numbers and integers are type-checked rather than coerced from
strings or Booleans.

| Kind | Required problem fields | Bound |
| --- | --- | --- |
| `threshold` | `slot_count`, `max_selected`, `positive_examples`, `negative_examples` | At most 1,000,000 selected-mask/threshold candidates |
| `feature-template` | `candidates`, `labels`, `coverage` | 1–4096 registry-backed typed candidates |
| `ta-clause` | `feature_count`, `max_literals`, `examples`, `labels` | At most 1,000,000 signed-literal conjunctions |
| `decision-tree` | `slot_count`, `max_depth`, `examples`, `labels` | Read-once depth at most 8 and at most 1,000,000 trees |
| `repair` | decision-tree fields plus `parent`; optional top-level `max_iterations` | At most 256 counterexample iterations within the tree bound |

Examples contain zero-based indices of active Boolean inputs. TA literal
indices use `2 * feature` for a positive literal and `2 * feature + 1` for its
negation. Feature-template candidates contain `field_name`, `template_id`,
`data_type`, and `parameters`; IDs and types are checked against the Python
registry before Prolog runs.

A repair parent is either `{"leaf": false}` or a recursive branch:

```json
{
  "feature": 0,
  "false": {"leaf": false},
  "true": {"leaf": true}
}
```

## Result contract

Solved searches emit `ptm.search.result.v1` containing:

- the search kind and elapsed time;
- the declared candidate upper bound and dataset digest;
- zero mismatches after independent Python validation;
- a typed template, signed clause configuration, tree, or repair guard;
- repair counterexamples; and
- `exportable: true` when behavior fits the fixed five-binding Logic ABI.

Prolog output is never accepted directly as deployable behavior. Python checks
returned indices, literals, tree bounds, the read-once property, and every
labeled example before reporting success.

See [Run bounded symbolic search](../how-to/run-bounded-search.md) for the
operational workflow.
