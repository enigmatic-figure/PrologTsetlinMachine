# Portable model export and static inference runtime

Status: packed-TM, fixed-Logic, and PA masked-threshold vertical slices
implemented; host integration and broader Milestone 5 work are in progress.

## Motivation

PTM can train, freeze, compile, validate, and content-address learned behavior.
The first three core slices now package that behavior as self-contained
objects that an unrelated application can retain and execute. Extending that
boundary to later learned artifact kinds matters because a
discovery that cannot leave the training workspace is difficult to reproduce,
combine, or distribute.

The deployment boundary has two inseparable parts:

1. a deterministic post-training export that captures portable inference
   semantics and their input/output contract;
2. a small, task-neutral runtime that loads that artifact and performs only
   immutable inference.

```text
mutable training state
        |
        | freeze and lower
        v
canonical inference artifact
        |
        | package, hash, and verify
        v
small immutable runtime object
        |
        +----> application
        +----> ONNX host or custom operator
        +----> another learned system
```

This is a distribution contract, not a second training representation.

## Training snapshots versus inference artifacts

A resumable training snapshot and a deployable inference artifact have
different completeness rules.

The training snapshot retains every multi-state automaton value, random stream,
feedback configuration, lifecycle checkpoint, and restoration dependency
needed to continue learning exactly. The inference artifact retains only the
canonical learned computation required to reproduce outputs: included
literals, clause polarity or weights, thresholds, compiled logic or PA state,
and the feature and output schemas that give those values meaning.

An export may reference or optionally attach a restoration snapshot for
research lineage, but the static runtime must neither require nor interpret
mutable training state. Backend-specific prepared layouts are optional caches;
they cannot be the only semantic representation in a portable artifact.
Training-only feedback and intermediate diagnostics become optional declared
output ports rather than mandatory deployment outputs.

## `ptm.model.v1` artifact contract

The first export format is a deterministic, content-addressed `ptm.model.v1`
bundle. The reference writer produces a single `.ptm` file with a fixed
little-endian header, a canonical UTF-8 JSON manifest, one typed binary
payload, and a SHA-256 trailer. The exact byte-level container is versioned
independently from the model schemas it carries.

The v1 container layout is:

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 8 | magic `PTMODEL\0` |
| 8 | 4 | container version |
| 12 | 4 | header size, currently 64 |
| 16 | 4 | numeric model kind |
| 20 | 4 | required flags, currently zero |
| 24 | 8 | manifest byte count |
| 32 | 8 | payload byte count |
| 40 | 24 | reserved, required to be zero |
| 64 | variable | canonical UTF-8 JSON manifest |
| next | variable | model-kind payload |
| final | 32 | SHA-256 of every preceding byte |

The raw trailer digest, rendered as `sha256:<hex>`, is the artifact identity.
The manifest declares the digest mechanism as `sha256-trailer-v1`; it cannot
contain its own resulting digest without introducing a circular encoding.

The complete artifact contract declares the following across its header,
manifest, and typed payload. Each implemented model kind carries the fields
relevant to its semantics; later operator/version fields arrive with the
payloads that require them.

- artifact, container, and payload schema versions;
- the content-digest mechanism covering the header, canonical manifest, and
  payload;
- producer version, model kind, and required runtime/operator versions;
- human-facing name, description, authorship, license, citations, intended use,
  and known limitations for research distribution;
- named input and output ports with data type, rank, shape bounds, layout, and
  semantic identifiers;
- the feature/literal catalog version and stable provenance identifiers;
- the canonical inference payload and any optional optimized payloads;
- task metadata such as classification labels or score interpretation, without
  making the execution API task-specific;
- validation signatures and small deterministic conformance vectors;
- source lineage, dataset digests, and optional restoration references;
- explicit dimensions and counts that the loader checks against its versioned
  resource ceilings before allocation.

The implemented canonical payload kinds are:

- `packed_tm_binary_v1` for frozen clause/vote inference;
- `logic_program32_v1` for fixed structural Logic programs;
- `masked_threshold_v1` for compiled PA thresholds.

Its payload begins with a fixed 32-byte header, followed by clause-major
positive and negative Include bitsets, then one to sixteen bounded conformance
cases. Each case carries a valid-lane mask, expected prediction mask,
feature-major packed words, and 64 expected signed scores. Empty clauses are
false, even clauses vote positive, odd clauses vote negative, and scores are
clamped symmetrically to the declared threshold.

The Logic payload also begins with a fixed 32-byte header. It carries one to 32
validated, eight-byte instructions with no forward references, followed by an
exhaustive 32-assignment conformance page for its five Boolean bindings. That
page records the result bit, true-instruction mask, and evaluated-instruction
mask for every assignment. The canonical opcodes are constant, input, NOT,
AND, OR, and XOR.

The PA payload begins with a fixed 32-byte header and a 1024- or 4096-slot
selection bitset. Its threshold is bounded by the selected-slot count. A
compact conformance page stores only the selected input, matched, and missing
slot words plus all 64 matched counts; runtime outputs remain dense and match
the native PA kernel contract. Export preserves the source PA artifact ID,
mapping version, slot bindings, provenance literal IDs, validation signature,
and restoration reference in the manifest.

Later schemas may add multiclass heads, convolutional or graph forms, and a
bounded composite DAG whose nodes can include other PTM artifacts. Unknown
required payloads or operators fail closed. The bundle contains declarative
data only: embedded native machine code, arbitrary Python, unrestricted Prolog,
and load-time network access are forbidden.

## Feature and input boundary

A model is not portable when its feature ordering exists only in the training
script. Every export therefore includes an introspectable input contract.

All artifacts must support already-materialized typed tensor or packed-feature
inputs. An artifact may additionally contain a bounded, versioned Class I
transform graph for raw records. In that case the same runtime may expose a
record-oriented convenience path. If a transform cannot be represented by the
portable catalog, export must require precomputed features and say so in the
manifest; it must not silently approximate training-time preprocessing.

This distinction allows a tiny learned Boolean feature to be embedded in a
larger tensor system without requiring that system to adopt PTM's data-loading
stack.

## Static runtime contract

`ptmrt` is a standalone inference library with an independently versioned C
ABI. "Static" means that loaded model behavior is immutable and training-free;
the reference library must also support static linking for small deployments.

The task-neutral lifecycle is:

```text
open file or memory -> verify -> describe ports -> run batch -> close
```

The initial ABI exposes:

- `ptmrt_abi_version`;
- `ptmrt_model_open_file` and `ptmrt_model_open_memory`;
- `ptmrt_model_describe`;
- `ptmrt_model_manifest_json` and `ptmrt_model_verify`;
- `ptmrt_model_run` over named, caller-owned tensor views;
- `ptmrt_model_close`;
- `ptmrt_status_message`.

The baseline runtime has no Python, Prolog, training, CUDA, or filesystem
dependency after an in-memory model is opened. A portable scalar CPU executor
is mandatory. SIMD, CUDA, and other accelerators are optional dispatch targets
that must pass artifact conformance vectors and exact semantic gates before
selection. Read-only handles are safe for concurrent inference with
caller-private input and output buffers.

A companion `ptmrt` CLI provides `inspect`, `verify`, and `run` commands so a
shared artifact can be evaluated without writing host code.

## Implemented slice

The current exporter, Python loader, static C ABI library, and CLI implement
packed-TM, fixed-Logic, and PA masked-threshold paths:

```powershell
$env:PYTHONPATH = "$PWD\python"
py -3 examples/export_xor_artifact.py out/artifacts/xor-little-guy.ptm
py -3 examples/export_logic_artifact.py out/artifacts/conditional-little-guy.ptm
py -3 examples/export_pa_artifact.py out/artifacts/threshold-little-guy.ptm

.\scripts\verify.ps1
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

The training-side `ptm export` entry point accepts a frozen scalar-TM snapshot
JSON file; `ptm export-logic` accepts the canonical `LogicProgram32` JSON form;
and `ptm export-pa` accepts a content-addressed Class II `PAArtifact` JSON file.
The Python APIs additionally accept validation signatures, stable literal IDs,
task labels, research metadata, and optional restoration references. TM export
derives inference-only Include masks and checks them against the original
multi-state snapshot oracle. Logic export exhaustively checks the compiled
payload against all 32 assignments of the source program. PA export checks up
to 64 deterministic assignments, exhaustively covering artifacts with six or
fewer selected slots, against the existing masked-threshold kernel. All three
emit canonical bytes, reload them, and verify embedded conformance data before
returning.

The standalone `ptmrt` library links only its scalar implementation and the C
standard/runtime libraries; it does not link `ptm_core`, Python, Prolog, CUDA,
or any training code. The packed-TM ports are:

| Direction | Name | Type and shape | Meaning |
| --- | --- | --- | --- |
| input | `features` | `uint64[F]` | one packed 64-lane word per feature |
| input | `valid_mask` | `uint64` scalar | lanes that contain examples |
| output | `predictions` | `uint64` scalar | one binary result bit per lane |
| output | `scores` | `int32[64]` | signed, threshold-clamped vote scores |

The fixed-Logic ports are:

| Direction | Name | Type and shape | Meaning |
| --- | --- | --- | --- |
| input | `bindings` | `uint64[5]` | one packed 64-lane word per binding |
| input | `valid_mask` | `uint64` scalar | lanes that contain assignments |
| output | `values` | `uint64` scalar | one Boolean result bit per lane |
| output | `true_instruction_masks` | `uint32[64]` | true instructions per lane |
| output | `evaluated_instruction_masks` | `uint32[64]` | evaluated instructions per lane |

The PA masked-threshold ports are:

| Direction | Name | Type and shape | Meaning |
| --- | --- | --- | --- |
| input | `slots` | `uint64[S]` | one packed 64-lane word per PA slot |
| input | `valid_mask` | `uint64` scalar | lanes that contain examples |
| output | `values` | `uint64` scalar | one threshold-result bit per lane |
| output | `matched_counts` | `uint32[64]` | selected slots matched per lane |
| output | `matched_slots` | `uint64[S]` | selected matching slots per lane |
| output | `missing_slots` | `uint64[S]` | selected missing slots per lane |

The present limitations are intentional and explicit: only binary packed TM,
five-binding `LogicProgram32`, and fixed 32x32/64x64 PA masked-threshold
artifacts with precomputed Boolean inputs are supported; each call uses the
fixed 64-lane packed interface; execution is scalar; and there is not yet a
raw-record transform graph, installed shared-library package, thin language
binding, ONNX bridge, accelerator dispatch, or WebAssembly build.

The native loader validates bounds, versions, reserved fields, declared model
dimensions and ports, TM bitset tails, Logic instruction graphs, conformance
structure, and the SHA-256 trailer. Full canonical-JSON and metadata-schema
validation currently belongs to the Python reference loader; the native
runtime treats non-execution metadata as opaque UTF-8 bytes after checking the
execution-critical fields.

## Export routine

The standardized exporter performs these stages explicitly:

1. freeze a selected training/lifecycle generation;
2. validate that its feature catalog, mappings, and output semantics are
   complete;
3. lower it to a canonical inference payload;
4. optionally prepare backend-specific acceleration sections;
5. attach provenance, validation signatures, and conformance vectors;
6. serialize deterministically and compute the artifact ID;
7. reload the emitted bytes and compare outputs with the training oracle
   before reporting success.

The current Python exporters perform this gate with the independent artifact
loader and the source-specific scalar oracle. Cross-language golden artifacts
for all three model kinds are loaded and executed by `ptmrt`. Invoking the
native runtime as an additional gate inside every export remains a packaging
step rather than a semantic dependency of the Python writer.

The same source state and export options must produce byte-identical output.
Partial, unaudited, activating, or hash-inconsistent lifecycle state is not
exportable.

## ONNX and composition

ONNX is an interoperability target, not the canonical storage format. Once the
runtime contract is stable, two bridges are useful:

- lower supported PTM computations to standard ONNX operators when that
  preserves exact semantics and reasonable size;
- provide a versioned PTM custom operator backed by `ptmrt` for compact models
  that do not lower well to standard operators.

The in-memory open API allows a `.ptm` payload to live inside another container
or model graph. This is the path for automata-discovered feature blocks,
attention components, or other micro-models to become reusable parts of larger
systems. Continuous tensors, trainable ONNX graphs, and automata-based
attention require the later multiclass/regression/sequence contracts; the v1
runtime should establish composition without pretending those model kinds
already exist.

## Acceptance boundary

Milestone 5 is complete when:

- equivalent exports are byte-for-byte deterministic and content-addressed;
- corrupted, oversized, incompatible, and executable-code-bearing artifacts
  are rejected before evaluation;
- packed TM, Logic program, and PA artifacts round-trip through one exporter
  and the same generic runtime API;
- exported conformance vectors match the independent Python oracle and native
  training runtime on Windows and Linux;
- a CPU-only consumer can inspect and execute an artifact without the training
  package or GNU Prolog;
- file and in-memory loading produce identical results and support concurrent
  read-only inference;
- at least one artifact is embedded through an ONNX bridge or equivalent host
  integration without changing its content identity.

The first release remains a research runtime. Its purpose is to make learned
discoveries durable, inspectable, and composable enough for other researchers
to reproduce and extend them.
