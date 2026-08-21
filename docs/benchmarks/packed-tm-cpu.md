# Packed TM CPU inference benchmark

This benchmark record captures CPU-only packed 64-example Tsetlin inference
performance measurements for the native adaptive substrate. It documents
throughput results for scalar, AVX2, and AVX-512 backends across varying
clause counts, feature counts, and Include densities.

## Protocol

Measurements follow the `ptm.runtime-benchmark.v1` schema with:

- **Timing scope**: kernel-only evaluation with resident feature-major input
  and caller-owned output buffers;
- **Correctness gate**: every workload compared against forced scalar result
  before timing;
- **Environment**: Intel Core i3-13100, Release build, CPUID/XCR0-checked
  dispatch;
- **Repetitions**: 5,000 for baseline, 2,000 for sweep studies.

Reproduce with:

```powershell
.\scripts\benchmark-packed-tm.ps1
```

Run the standard routing sweep as JSONL:

```powershell
.\scripts\benchmark-packed-tm.ps1 -Sweep -JsonLines
```

Or provide targeted axes:

```powershell
.\scripts\benchmark-packed-tm.ps1 `
    -Clauses 8,20,64 -Features 64,256 -Densities 0.005,0.02,0.10 `
    -ResidentPages 1,16 -Backend all -Repeats 2000 -JsonLines
```

## Baseline measurement (Release)

The Release benchmark uses 20 clauses, 256 represented features, approximately
2% included literals, one resident 64-example batch, and 5,000 repetitions.
One observed run before SIMD specialization produced:

| Path | Examples/second | Relative to scalar |
| --- | ---: | ---: |
| Scalar TM | 0.46M | 1.00x |
| Packed, including portable C++ row transpose | 1.27M | 2.78x |
| Packed, input already feature-major | 132.38M | 290.15x |

These are cache-hot synthetic inference measurements, not training or
end-to-end dataset throughput claims. The main architectural result is that
Class I should retain feature-major pages when a batch will be reused; repeated
row transposition consumes most of the otherwise available speedup.

## Dispatch sweep results

The dispatch harness sweeps clause count, represented feature count, Include
density, and every compiled backend. On the i3-13100, the standard Release
sweep showed:

- **AVX2**: about 1.5x faster than scalar for sparse 20-clause, 64-feature tiles;
- **Scalar**: faster for denser plans, which is why density participates in
  automatic selection;
- **AVX-512**: object compiled but unavailable on this CPU (correctly reported).

Automatic selection is conservative: vector kernels are chosen only for sparse
prepared plans averaging no more than one included literal per clause and for
16–128 clauses. Other shapes remain scalar.

## Capability notes

- Non-x86 builds contain only the portable kernel;
- x86 builds can be compiled with `-DPTM_ENABLE_X86_SIMD=OFF` for capability
  testing;
- Forced unavailable backends fail explicitly rather than silently falling back;
- Capability reports distinguish supported hardware from kernels present in
  the binary.

## Related documentation

- [Packed 64-example Tsetlin inference](../architecture/packed-tm.md) — execution contract and semantics;
- [CUDA packed TM execution](cuda-packed-tm-cuda.md) — GPU handoff measurements.
