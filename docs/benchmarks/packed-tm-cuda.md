# Packed TM CUDA benchmark record

This benchmark record captures CUDA packed 64-example Tsetlin inference
performance measurements and host routing evidence. It documents throughput
results for sparse, warp-tile, and dense-bitset clause backends combined with
two-stage and fused-atomic vote backends across multiple GPU devices.

## Protocol

Measurements follow the `ptm.runtime-benchmark.v1` schema with:

- **Timing scopes**: `kernel_only`, `resident_device_end_to_end`, and
  `cold_host_end_to_end`;
- **Correctness gate**: every workload compared against forced scalar CPU
  result before timing;
- **Environment metadata**: GPU name, ordinal, compute capability, VRAM,
  driver/runtime versions, multiprocessor count, warp size;
- **Validation**: Compute Sanitizer memcheck must report zero errors before
  performance results are accepted.

Reproduce the verification boundary from Linux/WSL:

```bash
bash scripts/verify-cuda-wsl.sh
```

Run a targeted dashboard stream:

```powershell
.\scripts\benchmark-packed-tm-cuda-wsl.ps1 `
    -Clauses 32,256,1024 `
    -Features 64,1024 `
    -Densities 0.02,0.50 `
    -ResidentPages 1,16,256 `
    -Backend "scalar,avx2,cuda_sparse,cuda_sparse_fused_vote,cuda_warp_tile,cuda_warp_tile_fused_vote,cuda_dense_bitset,cuda_dense_bitset_fused_vote" `
    -Repeats 100 -Samples 5 -JsonLines
```

Or run the full handoff matrix (intentionally large):

```powershell
.\scripts\benchmark-packed-tm-cuda-wsl.ps1 -GpuSweep -JsonLines
```

## Backend taxonomy

### Clause backends

| Backend | Description |
| --- | --- |
| `cuda_sparse` | Conventional grid worker evaluates one (page, clause) pair by walking prepared compact literal plan |
| `cuda_warp_tile` | One 32-thread warp owns a tile of 32 clauses, with one lane producing each clause word |
| `cuda_dense_bitset` | Adjacent clause workers read coalesced packed Include masks and scan features in common order |

### Vote backends

| Backend | Description |
| --- | --- |
| `two_stage` | 64-thread vote kernel scans stored clause outputs, computes signed lane scores, clamps, assembles prediction mask |
| `fused_atomic` | Clause workers atomically accumulate even/odd contributions into page-local lane scores; reduced vote kernel only clamps and assembles |

Fused benchmark names append `_fused_vote` to the clause backend.

## Exactness boundary

Every benchmark workload first evaluates every page through the forced scalar
CPU backend. Timing starts only after exact equality of:

- Prediction and feedback clause words;
- All 64 signed, threshold-clamped scores per page;
- Prediction masks;
- Valid-example masks.

The dedicated CUDA suite covers empty and contradictory clauses, positive and
negative literals, odd clause counts, threshold clipping, arbitrary valid
masks, 1/17/63/64-lane partial batches, feature counts beyond packed-word
boundaries, high-density 129-feature tail words, and multi-page repeated
execution.

## Routing evidence by device

### Quadro RTX 5000 Max-Q (SM 7.5, CUDA 11.8)

At 256 resident pages, the crossover is decisive:

| Clauses | Features | Density | Scalar host | Best CUDA resident | Best CUDA cold |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1024 | 0.02 | 0.66M/s | warp 13.44M/s | warp 10.34M/s |
| 256 | 1024 | 0.50 | 0.06M/s | warp 3.35M/s | warp 2.94M/s |
| 1024 | 1024 | 0.02 | 0.28M/s | sparse 7.15M/s | sparse 5.24M/s |

Provisional routing rule: keep one-page work on the CPU; route reused 256-page,
256/1024-clause plans to CUDA on this host.

### RTX 4050 Laptop (SM 8.9, CUDA 12.8)

The sparse/warp routing boundary is shape-dependent:

- At 16 pages, 256-clause density-0.02 work remains CPU-favored;
- At 16 pages, density-0.50 and 1024-clause density-0.02 already favor CUDA;
- At 256 pages both CUDA geometries are decisively ahead.

#### Dense bitset regime

| Clauses | Features | Density | Pages | Prior best | Dense bitset | Dense/prior |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1024 | 0.50 | 256 | 4.454M/s | 10.645M/s | 2.39x |
| 1024 | 1024 | 0.50 | 256 | 1.555M/s | 4.049M/s | 2.60x |
| 256 | 1024 | 0.50 | 4096 | 6.587M/s | 13.192M/s | 2.00x |
| 1024 | 1024 | 0.50 | 4096 | 1.586M/s | 3.238M/s | 2.04x |

At density 0.10 the older backends usually remain preferable, with one modest
1024-clause/256-page dense win; at density 0.02 they remain the default.

### Tesla T4 Colab (SM 7.5, CUDA 12.8)

Verified via `scripts/verify-colab-cuda.py`. Upload source archive as
`/content/ptm-source.tar.gz` and execute with `colab exec -f`.

## Fused vote observations

Strongest kernel-only improvements in large, sparse clause sets:

| Device | Clauses | Density | Pages | Clause backend | Two-stage | Fused | Fused/two-stage |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| RTX 4050 SM 8.9 | 1024 | 0.005 | 256 | sparse | 238.661M/s | 355.370M/s | 1.49x |
| RTX 4050 SM 8.9 | 1024 | 0.02 | 256 | sparse | 78.163M/s | 89.869M/s | 1.15x |
| RTX 4050 SM 8.9 | 1024 | 0.005 | 4096 | sparse | 245.071M/s | 345.101M/s | 1.41x |
| RTX 4050 SM 8.9 | 1024 | 0.02 | 4096 | sparse | 81.660M/s | 92.595M/s | 1.13x |
| Tesla T4 SM 7.5 | 1024 | 0.005 | 256 | sparse | 104.892M/s | 139.730M/s | 1.33x |
| Tesla T4 SM 7.5 | 1024 | 0.02 | 256 | sparse | 30.129M/s | 32.706M/s | 1.09x |

The boundary is not simply "sparse means fused." At 256 clauses, density 0.005,
and 256 pages on RTX 4050, sparse fusion fell from 708.896M/s to 457.531M/s
because the original vote scan was already cheap relative to atomic contention.

**Recommendation**: Use fused voting as a measured route for large sparse plans
when kernel execution dominates; retain two-stage voting for small clause counts
and full-output or high-density work unless device-specific measurement favors
fusion.

## Related documentation

- [CUDA packed TM execution](../architecture/cuda-packed-tm.md) — backend boundaries and exactness rules;
- [Packed TM CPU inference benchmark](packed-tm-cpu.md) — CPU-only measurements.
