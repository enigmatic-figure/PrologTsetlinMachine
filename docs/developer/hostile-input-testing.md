# Hostile-input testing

PTM treats `.ptm` files and raw records as untrusted input. Loading never
executes artifact bytes, imports Python objects, invokes GNU Prolog, or restores
training state. The Python loader and standalone `ptmrt` runtime validate the
container before model construction or conformance execution.

## Resource ceilings

The `ptm.model.v1` boundary enforces:

- 256 MiB maximum container size;
- 16 MiB maximum canonical JSON manifest;
- 16 levels and 1,000,000 decoded JSON nodes in the Python manifest loader;
- payload-specific dimensions and at most 16 conformance cases;
- SHA-256 integrity before JSON or payload interpretation;
- exact section sizes, reserved bytes, versions, kinds, and canonical JSON.

Embedded `ptm.preprocessing.v1` contracts additionally allow at most 4,096
outputs, 8 levels, and 100,000 decoded nodes. Both Python and C++ reject
non-finite numbers, invalid typed categories, duplicate identities, and
nonportable text.

## Regression strategy

The test extra installs Hypothesis and runs deterministic properties for:

- arbitrary byte strings passed to the generic artifact loader;
- single-byte mutations and sampled truncations of all golden model kinds;
- recursive JSON-like values passed to preprocessing construction;
- arbitrary typed and mistyped raw records;
- valid-digest manifests containing excessive nesting or non-finite constants.

The native suite independently checks truncated prefixes and suffixes, sampled
single-byte mutations, overflowing section lengths, and allocation-ceiling
sizes through `ptmrt_model_open_memory`. Every rejected open must leave the
model handle null.

Run the bounded corpus locally with:

```bash
python -m pytest tests/python/test_hostile_inputs.py
ctest --test-dir build --output-on-failure
```

These tests are bounded and deterministic so they run in normal Windows and
Linux CI. They complement, rather than claim to replace, coverage-guided native
fuzzing or process sandboxing planned for the universal runtime milestone.
