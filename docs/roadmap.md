# Roadmap disposition

This page is retained as a disposition record while the former roadmap is
split between release history, capability evidence, RFCs, and tracked issues.
It is not a current release navigation page.

| Former item | Current home | Status |
| --- | --- | --- |
| Optional SIMD/CUDA runtime dispatch | [CUDA architecture](architecture/cuda-packed-tm.md) and benchmark work | partially implemented; correctness-gated follow-up remains |
| Thin language bindings | project issue tracker | deferred |
| Hostile-input fuzzing and WebAssembly build | [hostile-input testing](developer/hostile-input-testing.md) and project issue tracker | fuzzing ongoing; WebAssembly deferred |
| Exact ONNX lowering and PTM custom operator | RFC or project issue tracker | deferred pending exact operator contract |
| Artifact, Prolog bridge, and TUI module split | project issue tracker | planned refactoring |
| PTA reasoning/control plane | [PTA RFC](rfcs/pta-control-plane.md) | proposed |

Completed milestones are recorded in [release history](releases/changelog.md);
current capability claims belong in the [maturity baseline](manual/reference/maturity.md).

This compatibility page preserves the original public path. Completed milestones are now in the release history; future work is tracked in RFCs and issues.
