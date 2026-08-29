# How the workbench protects state

The terminal workbench is a keyboard-first view over existing PTM services. It
does not introduce a second training, artifact, preprocessing, or search
implementation.

Training and bounded search run outside Textual's event loop and report
structured progress. Cancellation targets the active job. Configuration
changes mark a completed result stale so it cannot be exported under settings
that did not produce it.

The canonical layout keeps system and dashboard state visible while task views
cover configuration, deterministic XOR training, sustained native MNIST
multiclass training, clause behavior, TA population diagnostics, literals,
predictions, sampled temporal inspection, portable artifact
export/loading/raw-record inference, and bounded
threshold/template/clause/tree/repair search. GNU Prolog remains optional until
a workflow needs it.

The workload boundary is explicit. XOR owns the current scalar snapshot,
clause/TA diagnostics, temporal samples, and packed artifact contract. MNIST
owns epoch validation and an exact final confusion matrix over ten native
one-vs-rest banks. Until PTM defines a portable multiclass adaptive snapshot
and artifact schema, the workbench reports those snapshot-derived surfaces as
unavailable and blocks export rather than projecting binary facts onto the
multiclass run.

The earlier five-view shell uses the same controllers and remains available as
`ptm tui --style classic`. It is a compatibility presentation rather than a
separate implementation of PTM semantics.

Artifact paths are verified before use, existing files are not silently
overwritten, and raw-record controls come from the loaded preprocessing schema.
The workbench therefore adapts authoritative service and contract state rather
than maintaining UI-only semantics.

See the [first workbench session](../tutorials/first-tui-session.md) for a
guided journey and [Use the terminal workbench](../how-to/tui.md) for setup and
troubleshooting.
