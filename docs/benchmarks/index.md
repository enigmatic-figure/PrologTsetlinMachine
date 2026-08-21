# Benchmarks and research records

Benchmarks require a versioned protocol, executable correctness gate, immutable
result bundle, and environment manifest. Papers and narrative reports interpret
named bundles; they do not serve as a live results database.

Current transitional records:

- [Logic dataset representational-exhaustion baseline](logic-dataset-baseline.md)
- [Research references](references.md)

The existing benchmark pages predate a shared result-manifest schema. Their
procedures and observations remain useful, but new publishable results should
record the source commit and dirty state, compiler and flags, hardware and
driver, backend selection, input shape and density, warm-up and sample method,
timing boundary, median and dispersion, sample count, correctness checksum,
exclusions, failed cases, and raw-data content hash.

```{toctree}
:hidden:
:maxdepth: 2

logic-dataset-baseline
protocol
references
```
