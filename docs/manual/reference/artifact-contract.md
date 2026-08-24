# Portable artifact and runtime reference

Status: `ptm.model.v1` supports packed-TM, fixed-Logic, and PA
masked-threshold payloads. Packed-TM artifacts may also carry
`ptm.preprocessing.v1` for raw-record inference.

## Container

A `.ptm` file contains a fixed little-endian header, canonical UTF-8 JSON
manifest, one typed binary payload, and a SHA-256 trailer.

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
The manifest names the mechanism `sha256-trailer-v1`; it cannot contain its
own resulting digest.

The portable manifest complexity contract is part of `ptm.model.v1`, not an
implementation-specific parser setting. A manifest is limited to 16 MiB of
canonical UTF-8 JSON, a maximum decoded value depth of 8 (the root is depth
zero), and at most 100,000 decoded JSON values. Object member names are not
values for the node count. Exporters reject manifests outside these bounds
before publication, and both the Python loader and `ptmrt` enforce the same
limits before schema-specific interpretation. The C header exposes the limits
as `PTMRT_MODEL_MANIFEST_MAX_BYTES`, `PTMRT_MODEL_MANIFEST_MAX_DEPTH`, and
`PTMRT_MODEL_MANIFEST_MAX_NODES`.

The manifest declares versions, producer, kind, runtime requirements,
research metadata, named ports, feature/literal identities, validation data,
source lineage, dimensions, and resource bounds. Unknown required payloads or
operators fail closed. The bundle is declarative: native machine code,
arbitrary Python, unrestricted Prolog, and load-time network access are
forbidden.

## Payload kinds

### Packed TM

`packed_tm_binary_v1` stores a 32-byte payload header, clause-major positive
and negative Include bitsets, and one to sixteen conformance cases. Each case
contains a valid-lane mask, expected prediction mask, feature-major packed
words, and 64 expected signed scores. Empty clauses are false; even clauses
vote positive, odd clauses vote negative, and scores are clamped symmetrically
to the declared threshold.

### Fixed Logic

`logic_program32_v1` stores a 32-byte payload header, one to 32 validated
eight-byte instructions with no forward references, and an exhaustive
32-assignment conformance page for five Boolean bindings. Opcodes are constant,
input, NOT, AND, OR, and XOR.

### PA masked threshold

`masked_threshold_v1` stores a 32-byte header and a 1024- or 4096-slot
selection bitset. The threshold is bounded by selected-slot count. Its
conformance page stores selected input, matched and missing slot words, and 64
matched counts. The manifest preserves the source PA artifact ID, mapping
version, slot bindings, literal provenance, validation signature, and
restoration reference.

## Input boundary

All artifacts accept already-materialized typed tensor or packed-feature
inputs. Packed TMs may additionally embed the bounded
[`ptm.preprocessing.v1`](preprocessing.md) contract. Unsupported host
transforms must be materialized upstream; export never approximates them.

## Runtime lifecycle

The immutable runtime lifecycle is:

```text
open file or memory -> verify -> describe ports -> run batch -> close
```

The `ptmrt` ABI provides file and memory opening, port description, manifest
access, verification, optional raw-record preprocessing, named tensor
execution, closing, and status messages. After an in-memory model is opened,
the baseline runtime has no Python, Prolog, training, CUDA, or filesystem
dependency. Handles are read-only and support concurrent inference with
caller-private buffers.

The native loader validates integrity, resource bounds, versions, reserved
fields, execution-critical manifest fields, model dimensions and ports,
payload structure, and embedded conformance cases. It checks that the manifest
is a UTF-8 JSON object but treats non-execution metadata as opaque. The Python
reference loader additionally enforces canonical JSON bytes and the complete
metadata schema. Native and Python verification therefore protect the same
execution boundary without claiming identical metadata validation.

### Packed-TM ports

| Direction | Name | Type and shape | Meaning |
| --- | --- | --- | --- |
| input | `features` | `uint64[F]` | one packed 64-lane word per feature |
| input | `valid_mask` | `uint64` scalar | lanes containing examples |
| output | `predictions` | `uint64` scalar | one binary result bit per lane |
| output | `scores` | `int32[64]` | signed threshold-clamped scores |

### Fixed-Logic ports

| Direction | Name | Type and shape | Meaning |
| --- | --- | --- | --- |
| input | `bindings` | `uint64[5]` | one packed word per binding |
| input | `valid_mask` | `uint64` scalar | lanes containing assignments |
| output | `values` | `uint64` scalar | one Boolean result bit per lane |
| output | `true_instruction_masks` | `uint32[64]` | true instructions per lane |
| output | `evaluated_instruction_masks` | `uint32[64]` | evaluated instructions per lane |

### PA ports

| Direction | Name | Type and shape | Meaning |
| --- | --- | --- | --- |
| input | `slots` | `uint64[S]` | one packed word per PA slot |
| input | `valid_mask` | `uint64` scalar | lanes containing examples |
| output | `values` | `uint64` scalar | one threshold-result bit per lane |
| output | `matched_counts` | `uint32[64]` | selected slots matched per lane |
| output | `matched_slots` | `uint64[S]` | selected matching slots per lane |
| output | `missing_slots` | `uint64[S]` | selected missing slots per lane |

## Current limits

Model heads are binary packed TM, five-binding `LogicProgram32`, and fixed
32x32/64x64 PA masked thresholds. Only packed TMs accept raw records. Each
inference call uses the fixed 64-lane packed interface and baseline execution
is scalar. Host token, image, regex, aggregate, relational, sequence, and
temporal transforms must be materialized before this runtime boundary.

See [Export, inspect, and verify artifacts](../how-to/export-artifacts.md) for
the workflow, [Embed the native runtime](../how-to/embed-ptmrt.md) for CMake
integration, and [Portable deployment](../explanation/portable-deployment.md)
for the design rationale.
