# Deterministic raw-record preprocessing

Status: `ptm.preprocessing.v1` is implemented for packed binary TM artifacts
in the Python exporter/loader and the standalone `ptmrt` ABI and CLI.

## Contract

The optional top-level `preprocessing` object converts one typed record into
the artifact's ordered Boolean feature vector. Its schema is deliberately
small enough to reproduce without Python:

```json
{
  "schema": "ptm.preprocessing.v1",
  "outputs": [
    {
      "field": "age",
      "field_id": "131...",
      "field_kind": "number",
      "literal_id": "710...",
      "null_policy": "error",
      "parameters": {"threshold": 18},
      "transform": "numeric_ge"
    }
  ]
}
```

Outputs are evaluated in array order. Their literal IDs must exactly equal
`features.literal_ids`, so preprocessing cannot silently reorder the model
inputs. Field and literal IDs use the existing content-derived 64-bit identity
rules and are serialized as canonical unsigned-decimal strings. A contract has
between one and 4096 unique outputs. Parsers also enforce a maximum nesting
depth of 8 and a 100,000-node decoded contract budget before constructing field
or transform objects. The output-count check happens before individual output
descriptors are traversed.

The v1 field kinds and transforms are:

| Field kind | Transform | Parameters | Result |
| --- | --- | --- | --- |
| `number` | `numeric_ge` | finite `threshold` | `value >= threshold` |
| `number` | `numeric_between` | finite `lower`, `upper`; Boolean inclusive flags | bounded interval membership |
| `category`, `boolean` | `category_eq` | typed `value` | exact typed equality |
| `category`, `boolean` | `category_in` | nonempty canonical `values` | exact typed membership |
| any supported kind | `is_missing` | none | field absent or explicitly null |

`false`, `true`, and `error` null policies apply to every transform except
`is_missing`, whose policy is always `false` because missingness is its result.
An absent field and an explicit null have the same meaning. Unknown record
fields are ignored; duplicate field names are rejected by the C ABI.

## Portable value rules

The exporter and native runtime enforce the same value domain:

- numeric values are finite binary64 numbers or integers in the exact
  binary64 range `[-2^53, 2^53]`; Booleans are not numbers;
- category values are UTF-8 strings without control characters, signed
  64-bit integers, or Booleans;
- category equality is type-strict, so `true`, `1`, and `"1"` are distinct;
- Boolean fields accept only Boolean values and Boolean category constants;
- a present, non-null value of the wrong type fails the record rather than
  coercing or approximating it.

Cyclic Python mappings, non-string object keys, non-JSON values, excessive
nesting, and oversized contracts fail with `ValueError`. The native JSON parser
uses the same depth, node, and output ceilings. These limits are part of v1 and
prevent a small record interface from becoming an unbounded allocation path.

These rules are part of the artifact semantics. Locale, host integer width,
CSV inference, and application-specific truthiness cannot affect a result.

## Export and execution

Build a contract from the existing literal catalog and validate with raw
records during export:

```python
schema = FeatureSchema.from_fields(age=FieldKind.NUMBER)
catalog = LiteralCatalog(schema)
catalog.numeric_ge("age", 18, null_policy=NullPolicy.ERROR)
preprocessing = PreprocessingContract.from_catalog(catalog)

artifact = export_packed_tm(
    snapshot,
    preprocessing=preprocessing,
    validation_records=({"age": 17}, {"age": 18}),
)
predictions = artifact.predict_records(({"age": 21},))
```

`iter_predict_records` accepts a one-pass iterable and packs bounded pages of
at most 64 records. It is the connector-neutral streaming boundary: a CSV,
Arrow, message-queue, or application adapter can yield typed mappings without
making its parsing and checkpoint rules part of the artifact.

The C ABI exposes `ptmrt_model_has_preprocessing` and
`ptmrt_model_preprocess_record`. The latter produces one `uint64_t` word with
value zero or one per feature, suitable as lane zero of the normal
feature-major input. Hosts can pack up to 64 materialized records before a
`ptmrt_model_run` call without changing model semantics.

The CLI provides an explicitly typed record syntax:

```powershell
ptmrt run-record model.ptm age:int=21 status:string=ready active:bool=true
ptmrt run-record model.ptm age:null
```

`int`, `float`, `string`, `bool`, and `null` are the only CLI value types. The
command prints both the materialized feature vector and the prediction.

## Deliberate v1 boundary

Token, image, regex, aggregate, relational, sequence, and temporal adapters are
available through the versioned Python host pipeline described in
[Streaming data connectors and record transforms](data-connectors.md). They
remain outside this portable v1 artifact contract because the standalone C
runtime does not implement them. Such deployments materialize derived scalar
fields upstream; export never substitutes a nearby transform. Multiclass heads,
regression, stateful stream windows, and connector checkpoint protocols remain
separate artifact and host-integration milestones.
