# Roadmap

## Milestone 0: executable contracts — complete

- [x] Class I bounded catalog and provenance ledger.
- [x] Python scalar TM and lossless Fredkin oracle.
- [x] Portable C++ bit blocks, Fredkin primitives, PA kernel, and scalar TM.
- [x] Cross-language kernel vectors and exact snapshot replay tests.

## Milestone 1: native vertical slice — in progress

- [x] Stable C ABI and dependency-free Python native binding.
- [x] Strongly typed 32x32 and 64x64 PA buffers.
- [x] Reproducible Python/native conformance tests.
- [x] Canonical Boolean IR and exact scalar-TM clause/vote lowering.
- [x] Portable scalar and 64-example bit-packed logic-program execution.
- [x] Preliminary density/batch/layout execution planner.
- [x] Fixed 32-instruction program ABI and prepared native batch execution.
- [x] NoisyXOR correctness-gated CPU calibration harness.
- [x] Batch-oriented feature-major representation input through the C ABI.
- [x] Exact bit-sliced TA-state images and portable 64-lane clause/vote evaluation.
- [x] Benchmark harness and scalar/AVX2/AVX-512 runtime dispatch.
- [x] Optional, capability-gated CUDA sparse and warp-tile Boolean backends.
- [x] CUDA dense-bitset backend with correctness-gated density routing.
- [x] Fused-atomic CUDA vote backend with SM 7.5/8.9 measured routing.

## Milestone 2: Class II lifecycle — complete

- [x] Maturity metrics and transactional consolidation registry.
- [x] Mapping/version, behavioral-signature, and restoration artifact contract.
- [x] O(1) dense atomic mapping with generation-tagged absorb/release writes.
- [x] Concurrent shadow/live audit window and drift policy.
- [x] Reopen and dissolve transitions with invisible partial publication.
- [x] Content-addressed typed-Logic evaluator and exact shadow activation audit.
- [x] Exception patching and specialized artifact branching.
- [x] Generation-safe transactional parent/child artifact replacement.
- [x] Exhaustive behavior signatures and equivalence merging for five inputs.
- [x] Atomic snapshots, hash-chained event-log persistence, replay, and compaction.

## Milestone 3: Class III bounded search — in progress

- [x] Discover and verify GNU Prolog 1.5 toolchain.
- [x] Implement a resource-bounded exact masked-threshold search template.
- [x] Lower Prolog solutions to content-addressed Class II artifacts.
- [x] Validate generated artifacts through the native runtime.
- [ ] Add typed feature-template and TA-clause-configuration outputs.
- [ ] Add bounded decision-tree and counterexample-guided repair templates.

## Milestone 4: data connectors and learned allocation

- [x] Paired natural/symbolic Logic CSV connector and deterministic split.
- [x] Provenanced presence, count-threshold, and position-aware encodings.
- [x] Collision ceilings and clause/state exhaustion calibration.
- [x] Safe typed Logic AST, fact interface, evaluator, and primitive-IR lowering.
- [x] Bounded AST count/depth/edge/two-hop relational encoding.
- [ ] Arrow/Parquet streaming and image/token adapters.
- [ ] Regex, aggregate, relational, sequence, and temporal transforms.
- [ ] Budgeted feature persistence and retirement.
- [ ] Multi-class, convolutional, regression, and graph adapters.

## Milestone 5: portable model artifacts and static inference — in progress

### Post-training export

- [x] Specify the deterministic, content-addressed `ptm.model.v1` container.
- [x] Export frozen packed-TM inference through the standardized
  freeze/lower/package routine.
- [x] Export fixed Logic-program inference through the same container and
  generic runtime contract.
- [x] Extend that routine to PA inference payloads while preserving source
  mappings, provenance, validation signatures, and restoration lineage.
- [x] Embed named input/output schemas, feature-catalog versions, stable
  provenance, research/licensing metadata, validation signatures, and bounded
  conformance vectors.
- [x] Keep canonical inference semantics independent from optional
  backend-specific prepared layouts and training-restoration attachments.
- [x] Reload every emitted packed-TM, Logic, or PA artifact and require exact
  agreement with its independent oracle before export succeeds.

### Universal inference runtime

- [x] Build the standalone, independently versioned `ptmrt` C ABI with generic
  open, describe, run, and close operations for files and memory buffers.
- [x] Implement an immutable, training-free scalar CPU executor.
- [ ] Add optional, correctness-gated SIMD/CUDA runtime dispatch.
- [x] Add `ptmrt inspect`, `verify`, and `run` CLI commands.
- [ ] Add thin language bindings over the same task-neutral tensor interface.
- [x] Verify deterministic concurrent inference and bounded, corrupted, and
  incompatible-artifact rejection on Windows and Linux.
- [ ] Add broader hostile-input fuzzing and a sandboxed/WebAssembly build.
- [ ] Add standard-operator ONNX lowering where exact and a compact PTM custom
  operator for artifacts that are better embedded through `ptmrt`.
