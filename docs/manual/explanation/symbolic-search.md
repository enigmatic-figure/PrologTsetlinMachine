# Why symbolic search is bounded

PTM uses GNU Prolog as a finite, offline search participant rather than an
unrestricted runtime dependency. Each request declares a closed candidate
space, dataset, structural limits, and deadline before a subprocess starts.

This boundary provides three useful properties:

- the user can inspect the candidate ceiling before spending search time;
- `no_solution` is distinct from invalid input or runtime failure; and
- a proposed result is independently replayed against every labeled example
  in Python before it can become exportable behavior.

The Prolog process nominates a candidate; it does not publish one. Decision
trees and repairs must also pass depth, read-once, binding-count, and exact
lowering gates before PTM packages them as fixed-Logic artifacts.

See the [search contracts](../reference/search-contracts.md) for exact bounds
and [Run bounded symbolic search](../how-to/run-bounded-search.md) for usage.
