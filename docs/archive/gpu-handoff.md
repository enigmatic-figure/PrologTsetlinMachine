# GPU development handoff

> Historical environment record. This page preserves benchmark provenance and
> host-specific recovery notes; it is not an installation guide. New users
> should start with [Installation](../manual/how-to/install.md) and
> [CUDA packed TM execution](../architecture/cuda-packed-tm.md).

> Status (2026-08-06): the optional build boundary, `cuda_sparse`,
> `cuda_warp_tile`, `cuda_dense_bitset`, and the independent
> `fused_atomic` vote strategy are implemented, correctness-gated, and
> sanitizer-clean on SM 7.5 and SM 8.9. See
> [CUDA packed TM execution](../architecture/cuda-packed-tm.md).

## RTX 4050 host checkpoint

The second host was reached on 2026-08-06. It has an NVIDIA GeForce RTX 4050
Laptop GPU with 6141 MiB VRAM, an Intel Core i5-13420H, and 7.71 GiB system
RAM. Ubuntu 22.04.5 under WSL2 exposes 3.7 GiB RAM plus 1 GiB swap.

Both CPU baselines are green: Windows and WSL passed all seven native suites,
all 41 Python tests, GNU Prolog compilation, and both examples. On Windows,
use the `py` launcher because the Microsoft Store `python.exe` alias precedes
the installed interpreter; `scripts/verify.ps1` now handles that selection.

The minimal WSL CUDA 12.8.2 toolchain (NVCC 12.8.93, CUDART development files,
and Compute Sanitizer) is installed from NVIDIA's `wsl-ubuntu` repository. An
SM 8.9 build passed all eight CTest suites, including the dedicated CUDA
intermediate-exactness suite, and Compute Sanitizer reports zero errors.
Building in WSL's ext4 filesystem is materially faster on this
memory-constrained host; use the durable native cache:

```powershell
wsl.exe -d Ubuntu-22.04 -- env `
    PTM_CUDA_ARCHITECTURES=89 `
    PTM_CUDA_BUILD_DIR=/home/finn/.cache/ptm-cuda-sm89 `
    bash scripts/verify-cuda-wsl.sh
```

The 1/16/256/4096-page routing campaign is complete for the three inherited
workloads with exact checksums. At 50% Include density,
`cuda_dense_bitset` improves same-run resident throughput by 2.00x to 2.60x
over the prior CUDA best at 256/4096 pages; sparse and warp-tile remain
preferred at low density. Fused voting improves the strongest 1024-clause
sparse kernel routes by 1.09x to 1.49x, but downloading the complete exact
clause-output matrix often hides that gain. See
[CUDA packed TM execution](../architecture/cuda-packed-tm.md) for the measured tables.

A Colab Tesla T4 independently compiled the same working tree for SM 7.5,
passed the expanded CUDA exactness suite, and completed Compute Sanitizer with
zero errors. `scripts/verify-colab-cuda.py` reproduces that build and a focused
routing sample after a tracked source archive is uploaded as
`/content/ptm-source.tar.gz`. The current next optimization boundary is a
prediction-only result path and automatic routing; C ABI version 2 remains
frozen.

## Second GPU host restart (completed)

Resume from commit `292d1e6` (`Add exact CUDA packed TM backends`). The Quadro
host completed all implementation, exactness, sanitizer, and provisional
256-page measurements, but the unusually thin fan directly over its GPU had
been inoperative for months. Its recorded temperatures reflect that hardware
fault and say nothing adverse about this workflow. Avoid sustained work on
that laptop until the fan is repaired.

The next host is expected to have an RTX 4050 Laptop GPU with 6 GB VRAM. Treat
that identification as provisional until `nvidia-smi` confirms it. Six GB is
ample for the current compact model, 4096 resident feature pages, clause
outputs, scores, and both layouts under test.

On arrival, record the new host before building:

```powershell
git log -2 --oneline
git status --short
nvidia-smi
wsl.exe --status
wsl.exe -d Ubuntu-22.04 -- nvidia-smi
```

Inside WSL, locate `nvcc` and record its version. If the GPU is an RTX 4050
(Ada), configure for SM 8.9. The reproducible verification commands are:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc `
    'PTM_CUDA_ARCHITECTURES=89 bash scripts/verify-cuda-wsl.sh'
```

```powershell
.\scripts\benchmark-packed-tm-cuda-wsl.ps1 `
    -CudaArchitectures 89 -Clauses 32 -Features 65 `
    -Densities 0.02 -ResidentPages 1,2 `
    -Backend scalar,cuda_sparse,cuda_warp_tile,cuda_dense_bitset `
    -Repeats 10 -Warmup 2 -Samples 3 -JsonLines
```

The small benchmark is a smoke test, not a performance claim. Confirm exact
checksums first, then measure longer runs at a stable operating clock. The
16/4096-page campaign, `cuda_dense_bitset`, and alternate fused vote
aggregation described here are now complete. ABI version 2 remained frozen
throughout.

This note is the restart point for the first CUDA implementation. It was
written on 2026-08-06 for a target laptop with an NVIDIA Quadro RTX 5000,
16 GB VRAM, an Intel Core i7, and 48 GB system RAM.

## Known-good source baseline

CPU runtime milestone commit: `8b6a57a` (`Add runtime-dispatched SIMD TM
kernels`). At that point the repository has:

- exact bit-sliced TA-state inference images;
- feature-major batches of 64 examples;
- prepared clause offsets plus compact feature/negation arrays;
- portable scalar, AVX2, and AVX-512 execution objects;
- CPUID/XCR0-safe, density-aware runtime selection;
- C ABI version 2 and dependency-free Python bindings;
- exact intermediate-result equivalence gates;
- the versioned `ptm.runtime-benchmark.v1` JSON Lines stream.

The Windows verification command passed all seven native suites, all 41 Python
tests, GNU Prolog compilation, and both examples. WSL SIMD and deliberately
SIMD-disabled builds each passed all seven native suites. The working tree was
clean after the milestone commit.

## First boot on the GPU host

Do not change code until the copied repository passes its CPU baseline. Record
the command output alongside the first benchmark results:

```powershell
git log -1 --oneline
git status --short
nvidia-smi
wsl.exe --status
wsl.exe --update
wsl.exe -d Ubuntu-22.04 -- nvidia-smi
```

Inside WSL, record:

```bash
uname -a
cat /etc/os-release
cmake --version
ninja --version
g++ --version
gprolog --version
nvcc --version
```

Then run the existing verification boundary:

```powershell
.\scripts\verify.ps1
```

```bash
bash scripts/verify-wsl.sh
```

If CUDA is used through WSL, install the NVIDIA display driver on Windows and
the CUDA toolkit in WSL. Do not install a Linux NVIDIA display driver inside
WSL. Confirm that `nvidia-smi` works from both sides before adding a CUDA
language target.

The laptop must be on AC power in its performance mode for measurements. Record
driver, toolkit, GPU name, VRAM, power state, and clocks rather than assuming
the model name fully describes its mobile configuration.

## Architectural boundary

CUDA must remain optional. A machine without a CUDA compiler or NVIDIA device
must configure, build, test, and run exactly as it does now.

Add a `PTM_ENABLE_CUDA` CMake option, defaulting to automatic discovery or OFF,
without attaching CUDA ISA flags to `ptm_core`. Place device code in separate
translation units and expose it through a baseline C++ dispatcher, following
the pattern already established for AVX2 and AVX-512.

Do not extend C ABI version 2 during the exploratory benchmark phase. First
keep experimental CUDA entry points within the native benchmark target. Once
the useful backend taxonomy and ownership/lifetime rules are measured, publish
them together under an intentional ABI version 3.

The CPU implementation remains the semantic oracle. CUDA is never permitted to
change:

- prediction versus feedback empty-clause behavior;
- valid-lane masking;
- positive/negative literal interpretation;
- even/odd clause polarity;
- signed vote totals and threshold clipping;
- every clause and feedback output word.

## Kernel sequence

Implement one variable at a time, in this order:

1. `cuda_sparse`: one logical worker per clause walking the prepared compact
   literal plan. This is the simplest exact device counterpart to the scalar
   CPU kernel.
2. `cuda_warp_tile`: one warp owns a group of clauses, with one lane producing
   one 64-example clause word. Test 32-clause groups first and use warp
   collectives only where they remove real synchronization or vote work.
3. `cuda_dense_bitset`: evaluate dense Include masks as coalesced packed words.
   This is a separate density regime, not a replacement for the sparse plan.
4. `cuda_vote`: split clause production from signed vote aggregation and test
   fused versus two-stage execution.
5. `cuda_int8_vote`: only after a dense clause-output matrix exists, compare
   conventional integer/tensor aggregation with the bitwise vote kernel.

Do not start with tensor cores. Clause truth is Boolean and the first question
is whether integer bit packing, warp geometry, and residency beat CPU SIMD.
Tensor cores are a later candidate for dense weighted multiclass voting.

Keep host and device state explicit. The first implementation should use owned
RAII device buffers for:

- clause literal offsets;
- compact feature indices;
- compact negation bytes or masks;
- resident feature-major `uint64_t` input pages;
- prediction and feedback clause words;
- signed lane scores and prediction masks.

Prepare and upload immutable model data once. Measure input upload, kernel, and
result download separately before considering streams, pinned buffers, or
overlap.

## Correctness gates

Every timed CUDA configuration must first run a forced scalar CPU evaluation
on identical generated state and input. Require exact equality of:

- all prediction clause words;
- all feedback clause words;
- all 64 signed scores;
- the prediction mask;
- the valid-example mask.

Include dedicated cases for empty clauses, contradictory clauses, negative
literals, partial batches of 1/17/63 lanes, feature counts crossing packed-word
boundaries, odd clause counts, and threshold clipping. Randomized equivalence
must cover every available CUDA backend, just as the CPU dispatch test covers
every executable SIMD backend.

Run Compute Sanitizer on small correctness workloads before performance work.
Never treat a matching final prediction as sufficient evidence when an
intermediate clause or score differs.

## Benchmark matrix

Extend, rather than replace, `ptm_packed_tm_benchmark`. Start with:

| Axis | Initial values |
| --- | --- |
| Clauses | 32, 64, 256, 1024 |
| Features | 64, 256, 1024, 4096 |
| Include density | 0.005, 0.02, 0.10, 0.50 |
| Resident 64-example pages | 1, 16, 256, 4096 |
| Backend | scalar, AVX2, CUDA sparse, CUDA warp, CUDA dense |
| CUDA vote | two-stage scan, fused atomic aggregation |

Report three timing scopes:

1. kernel-only time measured with CUDA events;
2. resident-device end-to-end evaluation;
3. cold host-to-device plus evaluation plus device-to-host time measured with
   a host monotonic clock.

Use warm-ups, synchronize at timing boundaries, retain checksums, and report
median plus dispersion across repeated samples. A kernel-launch win at one
resident page does not establish an end-to-end win. The principal result is a
routing table over density, shape, page count, reuse, and output requirements.

## Dashboard event contract

Preserve schema `ptm.runtime-benchmark.v1` while adding optional fields. Existing
consumers must continue to parse CPU-only records. Extend `capabilities` with a
`gpu` object containing at least:

- name and device ordinal;
- compute capability;
- total VRAM;
- driver and CUDA runtime versions;
- multiprocessor and warp sizes;
- supported backend names.

Use backend names such as `cuda_sparse`, `cuda_warp_tile`, and
`cuda_dense_bitset`; append `_fused_vote` for the alternate vote axis. Add
`timing_scope`, `resident_pages`, transfer byte counts, and CUDA error status
to measurement records. Keep seeds, checksums, and any other unsigned 64-bit
identifiers as decimal strings for JavaScript safety.

In JSONL mode, stdout remains exclusively machine-readable events. Build logs,
diagnostics, and profiler instructions belong on stderr.

## Stop conditions

Do not advance to the next kernel when:

- any intermediate differs from the scalar oracle;
- Compute Sanitizer reports an error;
- timing includes an implicit synchronization that is not declared;
- the benchmark omits transfer or residency scope;
- CUDA becomes a mandatory dependency for CPU-only users;
- a claimed crossover is based on a single short sample.

The first GPU milestone is complete: three exact CUDA Boolean clause backends
and two vote strategies are available, the benchmark produces measured
routing data, CPU-only builds remain green, and the result is bounded to the
tested RTX 4050 and Tesla T4 configurations.
