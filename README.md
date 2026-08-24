# Prolog Tsetlin Machine

[![CI](https://github.com/enigmatic-figure/PrologTsetlinMachine/actions/workflows/ci.yml/badge.svg)](https://github.com/enigmatic-figure/PrologTsetlinMachine/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Prolog Tsetlin Machine (PTM) is an experimental toolkit for learning,
inspecting, searching, and exporting compact Boolean models. It combines a
Tsetlin Machine runtime, provenance-preserving feature transforms, bounded
symbolic search, and portable `.ptm` inference artifacts.

The easiest way to explore PTM is the keyboard-first terminal workbench. The
native runtime and GNU Prolog are optional; install them only when your use case
needs them.

## Try it

PTM requires Python 3.10 or newer. From a source checkout:

```bash
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows or
`source .venv/bin/activate` on Linux/macOS), then run:

```bash
python -m pip install --upgrade pip
python -m pip install ".[tui]"
ptm tui --demo xor
```

The Dashboard, persistent research telemetry, and task views guide the built-in
XOR journey through training, clause and TA inspection, temporal samples,
export, verification, and raw-record inference. Help is available throughout
the workbench; the generated
[help-topic and keyboard reference](docs/manual/reference/help-topics.md) is the
authoritative offline copy.

For the dependency-free Python core instead:

```bash
python -m pip install .
python examples/tabular_xor.py
```

For Arrow/Parquet streams and image files, install the data extra and run the
adapter pipeline example:

```bash
python -m pip install ".[data]"
python examples/data_adapter_pipeline.py
```

To exercise the complete portable-artifact path:

```bash
python examples/export_raw_xor_artifact.py out/raw-xor.ptm
ptm artifact inspect out/raw-xor.ptm --pretty
ptm artifact verify out/raw-xor.ptm
ptm artifact run-record out/raw-xor.ptm --pretty \
  --record '{"left": false, "right": true}'
```

On PowerShell, use the same command on one line or replace the trailing `\`
with a backtick. The result includes both the materialized Boolean features and
the predicted label.

With GNU Prolog installed, try bounded symbolic synthesis without preparing a
request file:

```bash
ptm search decision-tree --demo --pretty
ptm search repair --demo --output out/xor-repair.ptm --pretty
```

See [Installation](docs/manual/how-to/install.md) for native C/C++, GNU Prolog, CUDA, offline,
and troubleshooting instructions.

## What is included

- A deterministic scalar binary Tsetlin Machine and exact snapshot replay.
- Typed raw-record transforms with stable literal IDs and provenance.
- Bounded Arrow/Parquet streams, deterministic image/token adapters, and
  versioned regex, aggregate, relational, sequence, and temporal pipelines.
- Scalar, AVX2, AVX-512, and optional experimental CUDA packed-TM paths.
- Bounded GNU Prolog searches for thresholds, typed feature templates, signed
  TA clauses, read-once decision trees, and counterexample repair.
- Fixed Logic and masked-threshold Class II artifacts with lifecycle auditing.
- Deterministic `.ptm` export for binary packed TMs, Logic programs, and PA
  thresholds, including optional raw-record preprocessing.
- A standalone `ptmrt` C ABI and CLI for immutable inference without Python or
  Prolog.

PTM is currently a research release. Binary classification is the complete
packed-TM path; multiclass, regression, stateful stream processing, and richer
transforms embedded directly in the portable native runtime remain roadmap
work.

## Choose a path

| Goal | Start here |
| --- | --- |
| Explore interactively | [First workbench session](docs/manual/tutorials/first-tui-session.md) |
| Install PTM | [Installation guide](docs/manual/how-to/install.md) |
| Export a first Python model | [First Python model](docs/manual/tutorials/first-python-model.md) |
| Embed the native runtime | [First native consumer](docs/manual/tutorials/first-native-consumer.md) |
| Upgrade an existing checkout or integration | [Upgrade guide](docs/releases/upgrading.md) |
| Check supported and tested versions | [Compatibility matrix](docs/releases/compatibility.md) |
| Export or inspect `.ptm` artifacts | [Artifact how-to](docs/manual/how-to/export-artifacts.md) |
| Understand raw-record transforms | [Preprocessing reference](docs/manual/reference/preprocessing.md) |
| Stream Arrow/Parquet, images, or tokens | [Data connector reference](docs/manual/reference/data-connectors.md) |
| Run bounded symbolic search | [Bounded-search how-to](docs/manual/how-to/run-bounded-search.md) |
| Contribute or run the complete test matrix | [Contributor guide](docs/developer/contributing.md) |

## Project boundaries

Python owns orchestration, reference semantics, and data integration. C++20
owns portable and optimized execution. GNU Prolog is an offline, bounded search
participant and is never required in the per-record inference loop.

The portable artifact is the deployment boundary: training state is frozen and
lowered into an immutable, content-addressed `.ptm` file. Applications can use
the Python loader or the independently versioned `ptmrt` runtime.

For the user manual, developer guide, architecture contracts, RFCs, release
records, benchmarks, and archive, start at the
[documentation hub](docs/README.md). For background reading, use the linked
[research references](docs/benchmarks/references.md); large paper copies are intentionally
not stored in the repository.

## License

PTM is licensed under the [Apache License 2.0](LICENSE). See the
[changelog](docs/releases/changelog.md) for release history and pending changes.
