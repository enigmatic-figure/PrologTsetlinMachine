# PTM documentation constitution

Status: accepted for the documentation system. Changes to this policy require
maintainer review and a recorded rationale.

## Purpose

PTM publishes documentation through several surfaces, but it keeps very few
authoritative sources. A command, contract, capability, or benchmark fact must
have one named owner. Other surfaces may adapt that fact to their audience;
they must not silently fork it.

## Audiences and domains

| Domain | Primary audience | Owns |
| --- | --- | --- |
| Manual | Users and artifact consumers | Tutorials, task-oriented how-to guides, user reference, conceptual explanations |
| Developer | Contributors and integrators | Contribution workflow, public APIs, cross-language internals, extension procedures, verification |
| Architecture | Implementers and reviewers | Current semantic contracts, accepted decisions, ABI/schema boundaries |
| RFCs | Participants evaluating a change | Proposed directions and open design questions |
| Releases | Upgraders and operators | Changelog, migration requirements, compatibility and release history |
| Benchmarks | Performance and research readers | Protocols, manifests, reproducibility records, named result bundles |
| Archive | Historical readers | Superseded interfaces, environment handoffs, and non-current plans |

Within the manual and developer domains, authors should keep tutorials,
how-to guides, reference, and explanation distinct. A page may link across
those roles, but it should have one primary job.

## Authority rules

1. The CLI parser owns command names, arguments, defaults, metavariables, and
   syntax. Generated command reference and man pages project those facts.
2. The structured help registry owns reusable topic IDs,
   examples, related commands, TUI controls, requirements, and manual links.
   CLI, TUI, manual, and man-page prose may vary in depth but share those facts.
3. Python docstrings and public native headers own individual public API
   signatures, parameters, return values, exceptions, limits, and invariants.
   Generated API reference projects them. Authored guides own cross-component
   workflows and design explanation.
4. Versioned specifications and accepted ADRs in this repository own current
   architecture and semantics. A contract-affecting code change updates its
   specification in the same change.
5. Discussions and RFCs own deliberation, not current truth. An accepted idea
   graduates into an ADR, implementation work, current documentation, and—when
   compatibility changes—a release or migration note.
6. A future capability registry will own maturity stages and evidence links.
   README, manual, TUI, and release reports will render that registry rather
   than maintain independent status claims.
7. Benchmark protocols and executable harnesses live with the code. Raw result
   bundles are immutable and content-addressed. Papers interpret named bundles;
   they do not replace current operational benchmark records.

## Document states

Every inventoried document has one state:

- `current`: describes implemented or presently supported behavior;
- `mixed`: combines current facts with material that must be split during
  migration;
- `proposed`: describes unaccepted or incomplete direction;
- `historical`: preserves provenance but is excluded from normal onboarding;
- `internal`: operational metadata that is not part of the published manual.

Proposed and historical pages must say so near the title. Current pages must
not present roadmap work as implemented behavior.

## Change lifecycle

```text
idea or research question
  -> discussion
  -> RFC when a concrete design needs review
  -> accepted ADR or explicit rejection
  -> implementation issue/change
  -> current contract/manual update
  -> release and migration note when users are affected
```

ADRs are immutable records after acceptance except for corrections and status
links. A later ADR supersedes an earlier decision. RFCs may evolve while under
review, but their status and decision link must remain explicit.

## Versioning and publication

- The default web manual documents the latest stable release once tagged
  releases exist. Development documentation is labeled with its source commit.
- Older manuals remain available from tags or a version selector. Current
  pages do not narrate obsolete behavior inline when an archive or upgrade note
  can carry that history.
- Persistent format, ABI, schema, and semantic versions are documented beside
  their implementations. Unsupported combinations fail closed.
- Existing public Markdown paths remain valid during migration. A page may
  become a compatibility landing page, but it is not removed until inbound
  links and release policy permit removal.

## Ownership and review

The code owner for a behavior owns its corresponding factual documentation.
The author of a change is responsible for updating affected manual, contract,
capability, and release sources. Reviewers should reject copied command facts,
unversioned contract changes, unsupported capability claims, and benchmark
tables without provenance.

The documentation inventory is enforced by `scripts/check_docs.py`. New
Markdown files must be classified when they are added. The Sphinx build runs
with warnings treated as errors, and repository-local links are checked
independently.

## Migration rule

The migration is compatibility-first. New hubs and metadata may point to a
transitional flat page. Content moves only when its new authoritative owner is
clear, and the old path then becomes a maintained compatibility entry or an
explicit redirect in the publishing layer.
