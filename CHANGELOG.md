# Changelog

All notable changes to Prolog Tsetlin Machine will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Keyboard-first Textual workbench with training telemetry, clause inspection,
  cancellation, environment preflight, and artifact export.
- Typed GNU Prolog searches for feature templates, signed TA clauses, bounded
  decision trees, and counterexample-guided repair.
- Deterministic `ptm.preprocessing.v1` raw-record transforms.
- Portable raw-record inference through Python, the `ptmrt` C ABI, and
  `ptmrt run-record`.
- Installable CMake package targets and clean-consumer verification.
- User-oriented installation, compatibility, upgrade, and consumer guides.
- Apache-2.0 project licensing, shared release versioning, and cross-platform
  continuous integration.
- `ptm artifact inspect`, `verify`, and `run-record` commands.
- Textual artifact loading, conformance verification, schema-driven typed
  record controls, and per-literal preprocessing traces.
- Versioned bounded-search JSON contracts, `ptm search` commands, cooperative
  GNU Prolog cancellation, and a Textual Search workspace with repair
  counterexamples and fixed-Logic export.
- Shared conceptual help topics, `ptm help TOPIC`, generated workbench control
  reference, and contextual TUI help sourced from the application bindings.

### Changed

- External research papers are linked from the documentation instead of being
  stored in the source tree.
- Windows GNU Prolog searches now run with a child-only `LINEDIT=gui=no`
  environment, detached input, and `CREATE_NO_WINDOW` so linedit does not open
  GUI consoles or close dialogs during CLI and Textual search jobs.
- Artifact manifests and preprocessing contracts now have explicit byte,
  nesting, and decoded-node ceilings, with deterministic Hypothesis properties
  and a native hostile mutation corpus covering their fail-closed behavior.
- Added bounded Arrow/Parquet record streams, deterministic Pillow and Unicode
  token adapters, and content-addressed regex, aggregate, relational, sequence,
  and timezone-aware temporal record pipelines behind the optional `data`
  installation profile.

### Removed

- Machine-specific research PDFs, an obsolete C# demonstration, and a legacy
  transcript from the supported source distribution.

## [0.1.0] - Unreleased

- Initial research release baseline.

[Unreleased]: https://github.com/enigmatic-figure/PrologTsetlinMachine/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/enigmatic-figure/PrologTsetlinMachine/releases/tag/v0.1.0
