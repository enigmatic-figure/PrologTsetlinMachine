# CUDA packed TM execution

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
check. Reproduce the complete local CUDA verification boundary from PowerShell
with:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash scripts/verify-cuda-wsl.sh
```

`PTM_CUDA_ROOT`, `PTM_CUDA_ARCHITECTURES`, and `PTM_CUDA_BUILD_DIR` override
the script defaults. A system without a CUDA compiler remains covered by the
ordinary Windows and WSL verification scripts.

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

The first correctness-gated runs on the Quadro host confirm the expected
latency boundary. CPU scalar remains decisively faster end-to-end for small
one-page plans because launch and transfer costs dominate. At 256 resident
pages, the crossover is no longer marginal:

| Clauses | Features | Density | Scalar host | Best CUDA resident | Best CUDA cold |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1024 | 0.02 | 0.66M/s | warp 13.44M/s | warp 10.34M/s |
| 256 | 1024 | 0.50 | 0.06M/s | warp 3.35M/s | warp 2.94M/s |
| 1024 | 1024 | 0.02 | 0.28M/s | sparse 7.15M/s | sparse 5.24M/s |

These are medians of three short samples, and every backend/scope checksum in
each workload group was identical. They establish a provisional routing rule:
keep one-page work on the CPU; route reused 256-page, 256/1024-clause plans to
CUDA on this host. They do not establish the exact lower page-count boundary.

The fan directly over that laptop's GPU was later confirmed to have been
inoperative for months. Its temperature data describe a hardware fault, not a
constraint created by this workflow. Those results remain useful correctness
evidence, while the healthy RTX 4050 and Tesla T4 supply the routing evidence.

## RTX 4050 observation

The RTX 4050 Laptop host completed an SM 8.9 CUDA 12.8 build, all eight CTest
suites, and Compute Sanitizer memcheck with zero errors. The inherited
1/16/256/4096-page workloads retained exact checksums across every requested
backend and timing scope. Sustained runs were warmed before comparison so
reported backend differences were taken at the same operating clock state.

The sparse/warp routing boundary on this host is shape-dependent. At 16 pages,
256-clause density-0.02 work remains CPU-favored, while density-0.50 and
1024-clause density-0.02 work already favor CUDA. At 256 pages both CUDA
geometries are decisively ahead for the inherited workloads.

The dense-bitset experiment establishes a separate high-density regime. These
are same-run median resident measurements; every comparison was
scalar-oracle-gated:

| Clauses | Features | Density | Pages | Prior best | Dense bitset | Dense/prior |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1024 | 0.50 | 256 | 4.454M/s | 10.645M/s | 2.39x |
| 1024 | 1024 | 0.50 | 256 | 1.555M/s | 4.049M/s | 2.60x |
| 256 | 1024 | 0.50 | 4096 | 6.587M/s | 13.192M/s | 2.00x |
| 1024 | 1024 | 0.50 | 4096 | 1.586M/s | 3.238M/s | 2.04x |

At 4096 pages, dense bitsets also improved cold throughput by 1.81x for 256
clauses and 2.35x for 1024 clauses. At density 0.10 the older backends usually
remain preferable, with one modest 1024-clause/256-page dense win; at density
0.02 they remain the default. These results define a provisional routing
regime for this RTX 4050, not a device-independent crossover.

## Fused vote observation

Fused atomics remove the full clause-output scan from vote aggregation, but
they do not remove clause-output publication: the exact prediction and
feedback words remain part of the result contract. The strongest kernel-only
measurements were in large, sparse clause sets:

| Device | Clauses | Density | Pages | Clause backend | Two-stage | Fused | Fused/two-stage |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| RTX 4050 SM 8.9 | 1024 | 0.005 | 256 | sparse | 238.661M/s | 355.370M/s | 1.49x |
| RTX 4050 SM 8.9 | 1024 | 0.02 | 256 | sparse | 78.163M/s | 89.869M/s | 1.15x |
| RTX 4050 SM 8.9 | 1024 | 0.005 | 4096 | sparse | 245.071M/s | 345.101M/s | 1.41x |
| RTX 4050 SM 8.9 | 1024 | 0.02 | 4096 | sparse | 81.660M/s | 92.595M/s | 1.13x |
| Tesla T4 SM 7.5 | 1024 | 0.005 | 256 | sparse | 104.892M/s | 139.730M/s | 1.33x |
| Tesla T4 SM 7.5 | 1024 | 0.02 | 256 | sparse | 30.129M/s | 32.706M/s | 1.09x |

The boundary is not simply "sparse means fused." On the RTX 4050 at 256
clauses, density 0.005, and 256 pages, sparse fusion fell from 708.896M/s to
457.531M/s because the original vote scan was already cheap relative to atomic
contention. At 4096 pages the same shape reversed and fusion improved
kernel-only throughput by 1.28x.

Downloading every intermediate clause word hides much of the kernel gain. On
the RTX 4050, the best 256-page resident full-result improvement was 3.1%, and
several 4096-page or dense cases were neutral or slower. The T4 retained an
8.5% best-route resident improvement at density 0.005 and 1.9% at density
0.02. Consequently, use fused voting as a measured route for large sparse
plans when kernel execution dominates; retain two-stage voting for small
clause counts and full-output or high-density work unless a device-specific
measurement favors fusion. A future prediction-only download contract could
expose more of the fused kernel gain without changing the current exact result
path.

The Colab SM 7.5 check is reproducible with
`scripts/verify-colab-cuda.py`. Upload the tracked working-tree archive as
`/content/ptm-source.tar.gz`, then execute the script with `colab exec -f`.
It configures explicitly for architecture 75, builds, runs the CUDA CTest,
runs Compute Sanitizer memcheck, and emits the representative routing matrix.
