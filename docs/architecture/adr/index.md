# Architecture decision records

Accepted decisions are versioned beside the implementation. Use a four-digit
identifier and a short slug, for example `0002-artifact-versioning.md`.

Each ADR records:

- status: proposed, accepted, rejected, deprecated, or superseded;
- context and decision drivers;
- the decision and its exact scope;
- consequences and compatibility impact;
- implementation, contract, migration, and superseding links.

Accepted ADRs are not rewritten to reflect a later design. A later ADR marks
the earlier record superseded and updates the current architecture pages.

ADR 0001 was migrated from the original `docs/architecture.md` public path.
That old path remains as a compatibility landing; this directory is now the
canonical ADR location.

```{toctree}
:hidden:
:maxdepth: 1

0001-hybrid-runtime-boundaries
template
```
