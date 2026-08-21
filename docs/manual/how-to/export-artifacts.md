# Export, inspect, and verify artifacts

Use this guide to freeze learned behavior into a portable `.ptm` file and
verify the result before inference.

## Choose the exporter

The installed `ptm` command accepts three source forms:

- a frozen scalar-TM snapshot for packed-TM export;
- canonical `LogicProgram32` JSON for fixed-Logic export;
- a validated Class II `PAArtifact` JSON file for masked-threshold export.

Run `ptm help artifacts` for current examples, and use each command's
`--help` output for parser-owned arguments and defaults.

## Apply the export gate

Every standardized exporter:

1. freezes a selected source generation;
2. validates feature catalogs, mappings, and output semantics;
3. lowers to the canonical inference payload;
4. attaches provenance, validation signatures, and conformance vectors;
5. serializes deterministically and computes the artifact identity;
6. reloads the emitted bytes; and
7. compares outputs with the independent source oracle before success.

The same source state and export options must produce byte-identical output.
Partial, unaudited, activating, or hash-inconsistent state is not exportable.

## Inspect before inference

Run `ptm help artifacts` for inspection and verification examples. For typed
raw-record inference, use `ptm help preprocessing`. The native `ptmrt` tool
provides equivalent inspect, verify, run, and run-record operations for an
installed standalone runtime.

Only packed-TM artifacts carrying `ptm.preprocessing.v1` accept raw records.
The result includes the deterministic Boolean feature vector alongside each
prediction. See the [preprocessing reference](../reference/preprocessing.md)
for the exact value rules.

The [artifact contract](../reference/artifact-contract.md) owns format,
payload, port, resource-bound, and runtime compatibility facts.
