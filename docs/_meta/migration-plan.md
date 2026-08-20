# Documentation migration plan

Status: checkpoints 1 and 2 are implemented; the remaining sequence is
proposed and each checkpoint requires its own reviewed change.

This plan applies the [documentation constitution](constitution.md) in bounded,
reviewable checkpoints. It is an implementation plan, not a source of current
product capability claims.

## Checkpoint 1: boundaries and compatibility scaffold

- Establish domain hubs, authority rules, states, and ownership.
- Inventory every current Markdown page and record its destination.
- Preserve current public paths.
- Build the existing Markdown corpus through Sphinx and MyST.
- Fail CI on broken local links, unclassified pages, Sphinx warnings, and stale
  generated navigation inputs.

Exit criterion: a contributor can determine where a fact belongs, every page
has an explicit migration action, and the published navigation separates
manual, developer, architecture, proposal, release, benchmark, and archive
material.

## Checkpoint 2: shared help sources

- Define the structured help-topic schema.
- Make the `argparse` tree authoritative for CLI syntax and defaults.
- Render topic content into `ptm help <topic>`, subcommand help, contextual TUI
  help, manual pages, and command-reference inputs.
- Add coverage tests for commands, topics, TUI bindings, examples, and links.

Exit criterion: no command or keyboard fact is independently maintained in
multiple user-facing surfaces.

## Checkpoint 3: content migration

- Split the current multi-journey consumer tutorial.
- Move reusable installation and embedding procedures into how-to guides.
- Separate conceptual explanation from task instructions and factual
  reference.
- Convert old paths to compatibility landings as content moves.

Exit criterion: each migrated page has one Diátaxis role and one factual owner.

## Checkpoint 4: generated reference and native formats

- Generate Python API reference from structured docstrings.
- Generate native API reference from public-header comments.
- Document Prolog predicates with modes, determinism, bounds, and effects.
- Generate CLI reference plus `ptm(1)` and `ptmrt(1)` from shared sources.
- Publish offline HTML and PDF outputs when the content build is stable.

## Checkpoint 5: capability and decision systems

- Introduce the machine-readable capability registry with evidence links.
- Generate public maturity tables and validate evidence in CI.
- Move open-ended planning to Discussions and scoped issues.
- Adopt numbered ADRs and graduate accepted RFCs into current contracts.
- Split the mixed roadmap into capability evidence, release history, and future
  work.

## Checkpoint 6: benchmark provenance

- Version benchmark protocols and result-manifest schemas.
- Record commit, dirty state, environment, backend, timing boundary, samples,
  dispersion, checksums, exclusions, and raw-data hashes.
- Publish current reports from immutable result bundles.
- Link papers to named bundles rather than editing papers as results change.
