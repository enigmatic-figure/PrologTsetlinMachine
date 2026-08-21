# Native API reference

> Transitional header index. Public headers under `include/ptm/` are the source of truth for C++ signatures, parameters, return values, exceptions, limits, and invariants. Doxygen/Breathe generation is planned but not yet wired into the build.

Authoritative sources: `include/ptm/c_api.h`, `include/ptm/*.hpp` (`#include "ptm/..."`).

See also: [C ABI](c-api.md) (versioned `extern "C"` boundary), [Packed TM execution](../../architecture/packed-tm.md).

## Generation

```text
# when doxygen is installed, regenerate XML and rebuild docs
doxygen Doxyfile            # produces xml/ from include/ptm/*.hpp
sphinx-build -b html docs out/html
sphinx-build -b man docs out/man   # emits ptm(1) / ptmrt(1) via conf.py man_pages
```

Expected `docs/conf.py` Breathe wiring when the toolchain is present:

```python
extensions += ["breathe"]
breathe_projects = {"ptm": "xml/"}
breathe_default_project = "ptm"
```

The table below is an authored header index; each header owns its documented surface.

## Header index

| Header | Responsibility | Key surface |
|---|---|---|
| `include/ptm/c_api.h` | Versioned `extern "C"` ABI, fixed-width payloads, status codes | `ptm_abi_version`, `ptm_threshold_*_eval`, `ptm_logic_program32_*`, `ptm_tm_model_*`, `ptm_cpu_capabilities_query` |
| `include/ptm/packed_tm.hpp` | 64-lane `PackedTMModel64`, TA state planes, Include masks, scalar/AVX2/AVX-512 kernels | `PackedTMModel64`, `PackedTMResult64`, `PackedTMBackend` |
| `include/ptm/packed_tm_cuda.hpp` | CUDA sparse / warp-tile kernels, resident pages, capability-gated dispatch | `CudaPackedTM`, `CudaCapabilities` |
| `include/ptm/logic_program.hpp` | `LogicProgram32` fixed 32-instruction program and evaluation | `LogicProgram32`, `FixedLogicInstruction`, `FixedLogicResult` |
| `include/ptm/logic_ir.hpp` | Canonical operator DAG, compiler IR | `LogicIR`, `LogicGraph` |
| `include/ptm/pa_kernel.hpp` | Class II masked-threshold kernel shared by PA and C ABI | `MaskedThresholdKernel`, `PAResult` |
| `include/ptm/pa_instance.hpp` | Cold/warm PA instance, fixed-shape buffers, slot maps | `PAInstance`, `PAConfig` |
| `include/ptm/class_ii_persistence.hpp` | Class II artifact persistence, content-hash, restoration handles | `ClassIIPersistence`, `ArtifactManifest` |
| `include/ptm/scalar_tm.hpp` | Scalar semantic oracle, golden-vector reference | `ScalarBinaryTsetlinMachine`, `TMSnapshot` |
| `include/ptm/bit_block.hpp` | Aligned bit-block payloads (`ptm_bitblock_1024/4096`) | `BitBlock1024`, `BitBlock4096` |
| `include/ptm/fredkin.hpp` | Lossless controlled swaps, conservative signal routing | `FredkinGate`, `FredkinResult` |
| `include/ptm/concurrent_mapping.hpp` | Concurrent slot maps, capacity allocation | `ConcurrentMapping` |
| `include/ptm/consolidation_registry.hpp` | Logic consolidation registry, canonicalization | `ConsolidationRegistry` |
| `include/ptm/runtime.h` | Runtime helpers, alignment, portable dispatch | `ptm::runtime::*` |

## C++ signature ownership

* Signatures, template parameters, `noexcept`/`[[nodiscard]]` annotations, and width/alignment requirements live in the header.
* Python re-exports (`prolog_tsetlin.native`) mirror availability: forcing an unavailable ISA (`PackedTMBackend`) returns `PTM_STATUS_BACKEND_UNAVAILABLE` rather than falling back.
* Threshold clipping, empty-clause semantics (prediction false vs feedback true), and per-clause valid-lane masking are exercised by `tests/python/test_packed_tm.py` golden vectors.

## Example: inspecting headers without Doxygen

```bash
grep -rn "PTM_API\|class PackedTM\|struct LogicProgram32" include/ptm/
```

When Doxygen/Breathe are wired, each header above will be rendered as:

```rst
.. doxygenfile:: include/ptm/packed_tm.hpp
   :project: ptm
```
