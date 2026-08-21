# Roadmap disposition

> Historical record: this disposition preserves the former `docs/roadmap.md`
> planning surface after the migration to capability evidence, release history,
> RFCs, and tracked issues. It is not a current roadmap; new planning lives in
> [release history](../releases/changelog.md), the
> [maturity baseline](../manual/reference/maturity.md), RFCs, and the project
> issue tracker.

This page is retained as a disposition record while the former roadmap is split
between release history, capability evidence, RFCs, and tracked issues. It is
not a current release navigation page.

| Former item | Current home | Status |
| --- | --- | --- |
| Optional SIMD/CUDA runtime dispatch | [CUDA architecture](../architecture/cuda-packed-tm.md) and benchmark work | partially implemented; correctness-gated follow-up remains |
| Thin language bindings | project issue tracker | deferred |
| Hostile-input fuzzing and WebAssembly build | [hostile-input testing](../developer/hostile-input-testing.md) and project issue tracker | fuzzing ongoing; WebAssembly deferred |
| Exact ONNX lowering and PTM custom operator | RFC or project issue tracker | deferred pending exact operator contract |
| Artifact, Prolog bridge, and TUI module split | project issue tracker | planned refactoring |
| PTA reasoning/control plane | [PTA RFC](../rfcs/pta-control-plane.md) | proposed |

Completed milestones are recorded in
[release history](../releases/changelog.md); current capability claims belong in
the [maturity baseline](../manual/reference/maturity.md).

This archived disposition preserves the former roadmap content at a stable
path.
