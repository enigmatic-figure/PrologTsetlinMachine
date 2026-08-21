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

The existing [hybrid runtime boundary record](0001-hybrid-runtime-boundaries.md) predates
this directory and remains the accepted ADR 0001 until a compatibility-preserving
file migration is scheduled.
