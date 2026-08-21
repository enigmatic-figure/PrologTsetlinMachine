# CUDA packed TM execution

> Status: current architecture contract. Host measurements have been extracted
> to the [Packed TM CUDA benchmark](../benchmarks/packed-tm-cuda.md) record.

The optional CUDA path is a native benchmark and research boundary for exact
64-example packed Tsetlin inference. It does not alter `ptm_core`, make CUDA a
CPU-only dependency, or extend C ABI version 2.

## Build boundary

`PTM_ENABLE_CUDA` defaults to `OFF`. When enabled, CMake discovers the CUDA
compiler and builds a separate `PTM::cuda` static library from a C++20 host
wrapper and a CUDA C++17 device translation unit. CPU source files receive no
CUDA ISA flags, and the shared `ptm` C library continues to link only
`PTM::core`.

The Quadro RTX 5000 Max-Q development host uses Ubuntu 22.04 under WSL2, CUDA
11.8, GCC 11.4, and an SM 7.5 target. The RTX 4050 Laptop host uses CUDA 12.8,
GCC 11.4, and SM 8.9. A Colab Tesla T4 supplies an independent CUDA 12.8/SM 7.5
check. From a configured Linux or WSL shell, run the complete CUDA verification
boundary with:

```bash
bash scripts/verify-cuda-wsl.sh
```

`PTM_CUDA_ROOT`, `PTM_CUDA_ARCHITECTURES`, and `PTM_CUDA_BUILD_DIR` override
the script defaults. A system without a CUDA compiler remains covered by the
ordinary CPU verification path.

## Ownership and execution

`CudaPackedTMExecutor64` owns device buffers for compact clause offsets,
32-bit feature indices, literal-negation bytes, transposed positive/negative
Include bitsets, resident feature-major pages, valid masks, prediction and
feedback clause words, signed lane scores, and prediction masks. Immutable
model data is uploaded in the constructor.

Callers upload one or more pages explicitly, run a backend against the resident
pages any number of times, and download the final page-major outputs. Input,
kernel, and result-transfer durations are measured with CUDA events. The
benchmark additionally measures the two end-to-end scopes with a host
monotonic clock.

Clause evaluation and vote aggregation are independent backend axes. The
clause backends are:

- `cuda_sparse`: a conventional grid worker evaluates one `(page, clause)`
  pair by walking the prepared compact literal plan;
- `cuda_warp_tile`: one 32-thread warp owns a tile of 32 clauses, with one lane
  producing each clause word;
- `cuda_dense_bitset`: adjacent clause workers read coalesced packed
  positive/negative Include masks and scan features in a common order so
  feature-major input reads can broadcast across a warp;

The vote backends are:

- `two_stage`: a 64-thread vote kernel scans stored clause outputs, computes
  every signed lane score, clamps it, and assembles the prediction mask;
- `fused_atomic`: clause workers publish the same exact clause outputs while
  atomically accumulating their even/odd contributions into page-local lane
  scores. A reduced vote kernel only clamps those scores and assembles the
  prediction mask. Scores are cleared before every repeated evaluation.

None of the Boolean backends uses tensor cores. Fused benchmark names append
`_fused_vote` to the clause backend, for example
`cuda_sparse_fused_vote`.

## Exactness boundary

Every benchmark workload first evaluates every page through the forced scalar
CPU backend. Timing starts only after exact equality of:

- prediction and feedback clause words;
- all 64 signed, threshold-clamped scores per page;
- prediction masks;
- valid-example masks.

The dedicated CUDA suite covers empty and contradictory clauses, positive and
negative literals, odd clause counts, threshold clipping, arbitrary valid
masks, 1/17/63/64-lane partial batches, feature counts beyond packed-word
boundaries, high-density 129-feature tail words, and multi-page repeated
execution. The cross-product of all three clause backends and both vote
backends passes the same comparisons. Compute Sanitizer memcheck must report
zero errors before a performance result is accepted.

## Benchmark stream

The CUDA wrapper preserves `ptm.runtime-benchmark.v1` and adds optional fields.
The `capabilities` event contains the GPU name and ordinal, compute capability,
VRAM, driver/runtime versions, multiprocessor count, warp size, and supported
backend names. Measurement records add:

- `resident_pages`;
- `timing_scope` and `timing_source`;
- `input_upload_bytes` and `result_download_bytes`;
- `samples` and `mad_examples_per_second` alongside median throughput;
- `cuda_error_status`.

The three CUDA timing scopes are `kernel_only`,
`resident_device_end_to_end`, and `cold_host_end_to_end`. Model upload is
excluded because the immutable execution image is intended for reuse. Seeds,
workload seeds, and checksums remain decimal strings for JavaScript safety.

Build and emit a targeted dashboard stream from PowerShell with:

```powershell
.\scripts\benchmark-packed-tm-cuda-wsl.ps1 `
    -Clauses 32,256,1024 `
    -Features 64,1024 `
    -Densities 0.02,0.50 `
    -ResidentPages 1,16,256 `
    -Backend "scalar,avx2,cuda_sparse,cuda_sparse_fused_vote,cuda_warp_tile,cuda_warp_tile_fused_vote,cuda_dense_bitset,cuda_dense_bitset_fused_vote" `
    -Repeats 100 -Samples 5 -JsonLines
```

`-GpuSweep` selects the full handoff matrix: 32/64/256/1024 clauses,
64/256/1024/4096 features, 0.005/0.02/0.10/0.50 Include density, and
1/16/256/4096 resident pages. It is intentionally a large campaign.

## Initial host observation

Host measurement observations and routing evidence are recorded in the
[Packed TM CUDA benchmark](../benchmarks/packed-tm-cuda.md) benchmark record.
