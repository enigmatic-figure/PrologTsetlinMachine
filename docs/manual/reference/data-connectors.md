# Streaming data connectors and record transforms

PTM's `ptm.records.v1` host layer turns external data into ordinary typed
Python mappings. `ptm.record_pipeline.v1` then derives deterministic scalar or
bounded-sequence fields before the existing `ptm.preprocessing.v1` contract
materializes Boolean model inputs.

Install the optional file/image adapters through the `data` profile in
[Install PTM](../how-to/install.md).

The base package still has no mandatory third-party dependency. Token adapters
and record transforms use the Python standard library; PyArrow and Pillow are
imported only when their connectors are called.

## Arrow and Parquet streams

`iter_arrow_records` accepts a `pyarrow.Table`, `RecordBatch`,
`RecordBatchReader`, or iterable of batches. It validates unique string column
names, supports projection, and splits incoming batches to the requested bound:

```python
records = iter_arrow_records(table, columns=("message", "event_time"), batch_size=256)
for record in records:
    consume(record)
```

`iter_parquet_records` uses an Arrow Dataset scanner for a file, a path list,
or a directory. It streams record batches instead of reading all row groups
into one table and accepts an optional Arrow filter expression for predicate
pushdown:

```python
records = iter_parquet_records(
    "events/",
    columns=("message", "event_time", "readings"),
    batch_size=1024,
    filter_expression=ds.field("year") == 2026,
)
predictions = artifact.iter_predict_records(pipeline.iter_transform(records))
```

Batch sizes are limited to 65,536 rows. Arrow scalar conversion follows
PyArrow's `RecordBatch.to_pylist()` types; later adapters reject types outside
their own declared domain rather than coercing them.

## Token adapter

`TokenAdapter` applies an explicit normalization form, optional case folding,
a bounded regular-expression tokenizer, and an overflow policy. Its serialized
descriptor records the runtime Unicode database version, so a tokenizer is not
quietly replayed under different Unicode tables.

```python
adapter = TokenAdapter("message", output_field="tokens", max_tokens=4096)
adapted = adapter.iter_adapt(records)
```

Each record receives `tokens` and `tokens__count`. The default is NFKC,
case-folded Unicode word tokens with internal apostrophes. Inputs, token length,
and token count have explicit ceilings; truncation occurs only when the
descriptor says `overflow="truncate"`.

## Image adapter

`ImageAdapter` accepts Pillow images or paths, converts to `L` or `RGB`, and
uses nearest-neighbor resize in v1. It emits stable scalar fields such as
`image_pixel_0000` or `image_pixel_0000_r`, which can feed ordinary numeric
literal catalogs.

```python
adapter = ImageAdapter(28, 28, mode="L", output_prefix="glyph")
record = adapter.adapt_image("digit.png", record={"label": 7})
```

The source image is bounded to 16,777,216 pixels and the target to 65,536
channel values. V1 does not perform color management, learned normalization,
augmentation, or nondeterministic resampling.

## Record transform pipeline

Pipelines are ordered, content-addressed, JSON-round-trippable descriptors.
Each output field is unique and cannot overwrite an input field. Later
transforms may consume earlier outputs.

```python
pipeline = RecordTransformPipeline((
    RegexTransform("message", "has_alert", r"\balert\b", ignore_case=True),
    AggregateTransform(("readings",), "mean", AggregateOperation.MEAN),
    RelationalTransform("lower", "upper", "ordered", RelationalOperation.LE),
    SequenceTransform("tokens", "has_ready", SequenceOperation.CONTAINS,
                      argument="ready"),
    TemporalTransform("event_time", "event_year", TemporalOperation.YEAR),
))
derived = pipeline.iter_transform(adapter.iter_adapt(records))
```

Supported operations are:

| Family | Operations and rules |
| --- | --- |
| Regex | search, full match, count, or extract; bounded pattern, input, and matches |
| Aggregate | count, sum, mean, min, max, any, or all over fields or one sequence |
| Relational | type-strict equality/inequality and same-type ordering |
| Sequence | length, typed unique count/contains, prefix, suffix, or indexed item |
| Temporal | UTC epoch/parts, seconds between fields, or a fixed-width pairwise window |

Temporal strings must be ISO-8601 with an explicit timezone; numeric values are
UTC epoch seconds. No transform reads the wall clock. Aggregates are record
local, and temporal windows compare two fields in the same record, so streaming
does not require hidden cross-record state.

Configurable regexes use a validated subset of Python's Unicode `re` syntax.
Literals, character classes, categories, anchors, groups, top-level alternation,
and a deliberately narrow quantified form are supported. A general alternation
branch may contain at most one unbounded quantifier, and it must be the final
term. Finite repeats are capped at 64 occurrences and share a 256-choice branch
budget. Repeated atoms must consume exactly one character; adjacent, nested,
and alternation-containing quantifiers are rejected. The built-in Unicode word
tokenizer is a separately audited delimiter-separated language with a fallback
branch that consumes each word run. Lookaround, backreferences, and
conditionals are also rejected. Thus patterns such as `(a+)+$`, `a*a*a*a*b`,
and `a+b` are never evaluated. Pattern, input, token, and match ceilings remain
defense in depth. Invalid descriptors raise `ValueError`, while failures
processing a record raise `ConnectorError` or `TransformError`.

Run the complete example with:

```bash
python examples/data_adapter_pipeline.py
python examples/data_adapter_pipeline.py --parquet records.parquet
```

The [host-pipeline explanation](../explanation/host-pipelines.md) defines why
these deterministic adapters remain outside the portable runtime boundary.
