# Prolog Tsetlin Machine

Prolog Tsetlin Machine (PTM) is an experimental hybrid learning runtime. It
combines recyclable Tsetlin-automata populations, reversible Fredkin data
paths, provenance-preserving feature representation, and bounded Prolog
search/compilation.

This repository currently contains the first executable foundation, not a
production learner:

- a Class I representation layer with deterministic literal IDs, typed facts,
  packed Boolean rows, and per-literal provenance;
- a deterministic scalar binary Tsetlin Machine used as the semantic oracle;
- lossless Fredkin primitives that retain all three outputs;
- fixed 32x32 and 64x64 PA bit buffers and a portable masked-threshold kernel;
- a versioned C ABI, shared native library, and dependency-free Python binding;
- bounded GNU Prolog search that lowers exact rules to Class II PA artifacts;
- a canonical Boolean DAG, TM-to-IR compiler, preliminary cost planner, and
  scalar/64-example packed CPU execution paths;
- exact bit-sliced TA-state inference images with a direct 64-example clause,
  feedback, signed-vote, and prediction C ABI;
- capability-safe scalar/AVX2/AVX-512 runtime dispatch and a versioned JSONL
  crossover benchmark stream;
- optional CUDA sparse, 32-clause warp-tile, and dense-bitset execution with
  selectable two-stage or fused-atomic voting, resident input pages, exact
  intermediate-result gates, and Compute Sanitizer coverage;
- a fixed 32-instruction Class II Logic evaluator with content-addressed state,
  exact shadow validation, and a batched native C ABI;
- immutable Logic morphology with counterexample repair, branch factoring,
  equivalence merging, and generation-safe parent/child replacement;
- a transactional Class II registry with O(1) atomic source resolution and
  non-blocking shadow-audit recording;
- atomic Class II snapshots and SHA-256-chained event logs with deterministic
  replay, torn-tail recovery, and checkpoint compaction;
- snapshot/restore contracts for exact TA-state recovery;
- deterministic, content-addressed `.ptm` export for frozen packed TMs, fixed
  Logic programs, and PA masked thresholds, plus a standalone scalar `ptmrt`
  C ABI and inspect/verify/run CLI;
- versioned architecture, semantic, and PA-artifact contracts.

The intended production architecture is C++20 for the native execution core,
Python for orchestration and data integration, and GNU Prolog for bounded
symbolic search and offline artifact generation. Prolog is deliberately not in
the per-example inference loop.

## Quick start

On the provisioned Windows development machine, the complete build and
cross-layer verification is one command:

```powershell
.\scripts\verify.ps1
```

The Ubuntu 22.04/WSL path verifies the same C++, C ABI, Python/native, and GNU
Prolog boundaries with GCC and `libptm.so`:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash scripts/verify-wsl.sh
```

The Python reference path itself has no runtime dependencies:

```powershell
$env:PYTHONPATH = "$PWD\python"
python -m unittest discover -s tests/python -v
python examples/tabular_xor.py
python examples/prolog_threshold.py
```

Build and test the portable C++ core with any C++20 compiler:

```powershell
cmake -S . -B build -DPTM_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Export and run three portable inference artifacts through the same runtime:

```powershell
$env:PYTHONPATH = "$PWD\python"
py -3 examples/export_xor_artifact.py out/artifacts/xor-little-guy.ptm
py -3 examples/export_logic_artifact.py out/artifacts/conditional-little-guy.ptm
py -3 examples/export_pa_artifact.py out/artifacts/threshold-little-guy.ptm
.\build\ptmrt.exe inspect out/artifacts/xor-little-guy.ptm
.\build\ptmrt.exe verify out/artifacts/xor-little-guy.ptm
.\build\ptmrt.exe run out/artifacts/xor-little-guy.ptm 0,1
.\build\ptmrt.exe inspect out/artifacts/conditional-little-guy.ptm
.\build\ptmrt.exe verify out/artifacts/conditional-little-guy.ptm
.\build\ptmrt.exe run out/artifacts/conditional-little-guy.ptm 1,0,1,0,0
.\build\ptmrt.exe inspect out/artifacts/threshold-little-guy.ptm
.\build\ptmrt.exe verify out/artifacts/threshold-little-guy.ptm
.\build\ptmrt.exe run out/artifacts/threshold-little-guy.ptm 1,70
```

The `.ptm` file contains canonical inference data, human/research metadata,
stable feature identifiers, and bounded conformance vectors—not mutable
training state. See the portable-runtime design for the current packed-TM,
fixed-Logic, and PA scope and the planned ONNX, accelerator, and WebAssembly
extensions.

Build and run the NoisyXOR CPU compiler calibration when the reference data is
available locally:

```powershell
.\scripts\benchmark-logic.ps1
```

Prepare and run the paired, noise-free Logic dataset exhaustion baseline:

```powershell
.\scripts\benchmark-logic-dataset.ps1 -Repeats 1
```

Compile and shadow-audit the structural Class II evaluator:

```powershell
.\scripts\benchmark-logic-consolidation.ps1 -Repeats 100
```

Run the controlled-drift morphology and transactional replacement example:

```powershell
.\scripts\benchmark-logic-morphology.ps1
```

Run native Class II snapshot, replay, drift-reopen, and compaction:

```powershell
.\scripts\run-class-ii-persistence.ps1
```

Build and run the Class II mapping microbenchmark:

```powershell
.\scripts\benchmark-mapping.ps1
```

Run the scalar-backend-gated packed TM benchmark:

```powershell
.\scripts\benchmark-packed-tm.ps1
```

Emit the standard backend/density sweep as dashboard-ready JSON Lines:

```powershell
.\scripts\benchmark-packed-tm.ps1 -Sweep -JsonLines
```

On the provisioned CUDA/WSL host, verify all experimental GPU backends and run
Compute Sanitizer:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash scripts/verify-cuda-wsl.sh
```

Emit CPU/CUDA upload, kernel, resident, and cold timing records:

```powershell
.\scripts\benchmark-packed-tm-cuda-wsl.ps1 `
    -Clauses 32,256,1024 -Features 64,1024 `
    -Densities 0.02,0.50 -ResidentPages 1,16,256 -JsonLines
```

Set `PTM_GPROLOG` when GNU Prolog is not on `PATH`, and
`PTM_NATIVE_LIBRARY` when the shared library is outside the default `build`
directory. The verification script discovers the Visual Studio and
`C:\GNU-Prolog` installations currently provisioned on this system.

## Design boundaries

Four distinct bit planes are never conflated:

1. literal truth for one example;
2. TA Include/Exclude action;
3. the literal condition after action gating;
4. the resulting clause or PA output.

A TA action can be polled as one bit. It is not a complete TA snapshot. Exact
restoration also requires the multi-state automaton values, clause weights,
configuration, random-generator state, and mapping version.

Mutable instance data lives in aligned buffers. Machine code is shared and
immutable; PA behavior is selected by versioned descriptors, masks, slot maps,
and compiled artifacts.

See [Architecture](docs/architecture.md), [Semantic contract](docs/semantic-contract.md),
[Logic compiler](docs/logic-compiler.md), [Typed Logic AST](docs/logic-ast.md),
[Logic Class II consolidation](docs/logic-class-ii-consolidation.md),
[Logic morphology](docs/logic-morphology.md),
[Packed TM execution](docs/packed-tm.md),
[CUDA packed TM execution](docs/cuda-packed-tm.md),
[GPU development handoff](docs/gpu-handoff.md),
[Class II persistence](docs/class-ii-persistence.md),
[Portable model export and static runtime](docs/model-export-runtime.md),
[C ABI](docs/c-api.md),
[Logic dataset baseline](docs/logic-dataset-baseline.md),
[Class II lifecycle](docs/class-ii-lifecycle.md), and [Roadmap](docs/roadmap.md).
