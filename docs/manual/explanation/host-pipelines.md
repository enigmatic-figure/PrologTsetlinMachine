# Host pipelines and portable preprocessing

PTM has two deterministic preprocessing layers with different portability
promises.

`ptm.preprocessing.v1` is deliberately small enough for both Python and the
standalone C runtime to reproduce exactly. It turns typed numeric, category,
Boolean, and missing values into an artifact's ordered Boolean feature vector.

The `ptm.records.v1` and `ptm.record_pipeline.v1` host layers handle broader
integration concerns: Arrow/Parquet streaming, tokenization, images, regular
expressions, aggregates, relations, sequences, and temporal fields. They are
versioned and deterministic, but are not embedded in the v1 artifact runtime.

Keeping the layers separate prevents a native consumer from silently
approximating a transform it does not implement. A deployment without Python
materializes host-derived scalar fields upstream, then supplies portable raw
fields or already-computed Boolean features to `ptmrt`.

See the [preprocessing contract](../reference/preprocessing.md) and
[data-connector reference](../reference/data-connectors.md) for exact rules.
