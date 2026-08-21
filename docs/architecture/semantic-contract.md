# Semantic contract 0.1

This document defines meanings that optimized implementations may not change.

## Stable identities

- A `source_field_id` identifies a field definition, not a column position.
- A `literal_id` is derived from a canonical, versioned transform descriptor.
- IDs are unsigned 64-bit values. Hash collisions are detected when catalogs
  are assembled; they are never silently accepted.
- External JSON encodes unsigned 64-bit IDs as decimal strings so JavaScript
  parsers cannot silently lose precision. Native and Python APIs use integers.
- Slot indices are local to an artifact. They must always be interpreted through
  that artifact's mapping version.

## Representation pair

For each record the Class I layer can produce both:

- `x_ta`: a packed Boolean vector ordered by the catalog;
- `x_symbolic`: typed facts preserving original values.

Each Boolean literal has a descriptor containing its source field, transform,
parameters, null policy, and transform-catalog version. An optional evaluation
trace additionally records the record ID, raw value, and result. This connects
apparently independent thresholds back to their shared continuous variable.

The initial bounded transform catalog contains numeric thresholds/intervals,
categorical equality/membership, missingness, and token membership. Versioned
host adapters now provide regex, aggregates, relational identity, sequences,
and pairwise temporal windows, but these are not silently approximated in the
portable artifact's version 1 preprocessing contract.

## Bit-plane types

| Type | Meaning | Depends on example? | Sufficient for restore? |
| --- | --- | --- | --- |
| `LiteralTruth` | Result of a Class I literal | yes | no |
| `TAAction` | Exclude=0 or Include=1 | no, until learning | no |
| `LiteralCondition` | Included literal constrains clause; excluded literal yields neutral true | yes | no |
| `ClauseOutput` | Conjunction result | yes | no |
| `PAInputSlot` | Artifact-defined Boolean port | artifact-specific | no |

A `LogicProgram32` is not another bit plane. It is a versioned, fixed-shape
Class II state block containing typed instructions and backward operand masks.
Its five runtime binding bits retain literal-truth semantics; its instruction
truth mask is diagnostic output and cannot be used as a TA snapshot.

These types may share the same physical one-bit representation, but API names
and descriptors must preserve their distinct semantics.

## Scalar TA state

Each two-action TA has `2N` states. States `1..N` select Exclude and states
`N+1..2N` select Include. Increment and decrement saturate. The state transition
is therefore not reversible without recorded history.

Literal order for a record with `F` represented features is:

```text
x0, NOT x0, x1, NOT x1, ... x(F-1), NOT x(F-1)
```

An empty clause evaluates false during prediction. This prevents a population
of empty clauses from producing a positive prediction.

During feedback evaluation, an empty clause is an unconstrained conjunction
and evaluates true. Packed execution must retain separate prediction and
feedback clause words so this difference is not optimized away. Feature-major
batch word bit `i` always denotes example lane `i`; bits outside the explicit
valid-lane mask have no semantic value and must be suppressed from every
diagnostic and result.

A bit-sliced TA snapshot stores every state bit for every clause-major
automaton. Its derived Include mask is an execution index, not a sufficient
restoration representation.

## Fredkin literal condition

The lossless gate invocation is:

```text
fredkin(action, 1, literal_truth)
```

using the conventional positive-control definition: control `1` swaps the data
lines and control `0` preserves them. The complete result is:

```text
(action, literal_condition, garbage)
```

If action is Include, `literal_condition` equals `literal_truth`. If action is
Exclude, it equals the neutral conjunction value `1`. Discarding `garbage`
turns this into a useful projection but not a reversible operation.

## Snapshot completeness

An exact adaptive-substrate snapshot includes at least:

- all multi-state TA values;
- clause polarities and weights;
- thresholds, specificity, and feedback configuration;
- feature/literal catalog version and clause-to-literal mapping;
- random-generator state;
- implementation and snapshot schema versions.

Class II restoration handles must resolve to such a snapshot or an equivalent
deterministic event-log checkpoint.

Class II registry persistence is a separate but linked checkpoint. It must
retain artifact handles and states, mapping generations, audit aggregates,
policies, mapping versions, content-addressed artifact IDs, and adaptive-state
restoration handles. Recovery must validate the entire image before exposure;
an `activating` artifact or a partially mapped active artifact is never a valid
durable state. A torn final log frame may be discarded, but a complete corrupt
or ancestry-breaking frame must fail recovery.

## Morphology immutability

A deployed logic program is immutable. Morphology creates a content-addressed
child and retains all parent program IDs as restoration lineage. Exhaustive
32-bit behavior signatures are valid only for the current five-binding Logic
domain. A replacement child must pass shadow audit before source mappings move;
its publication may expose the old parent, the new child, or source fallback to
concurrent readers, but never an `activating` child.
