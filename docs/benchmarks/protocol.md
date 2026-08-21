# Benchmark protocol v1

Benchmarks require a versioned protocol, executable harness, and immutable result bundle.

## Required manifest fields (ptm.benchmark-manifest.v1)

- `commit` and `dirty` — source commit and working-tree cleanliness
- `environment` — OS, compiler, flags, hardware, driver
- `backend` — selected backend (scalar, avx2, cuda_sparse, etc.)
- `input_shape` — clause count, feature count, density, batch size
- `timing_boundary` — what is timed (kernel_only, resident, cold, etc.)
- `samples` — repetition count
- `median` and `mad` — median throughput and median absolute deviation
- `checksum` — correctness checksum/mat
- `exclusions` — skipped cases and reason
- `raw_data_hash` — SHA-256 of raw JSONL bundle

See [Benchmarks and research records](index.md) for the full charter. New results will be published as immutable bundles under `benchmarks/bundles/<hash>.jsonl` and linked from papers by bundle ID.
