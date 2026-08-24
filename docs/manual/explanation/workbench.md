# How the workbench protects state

The terminal workbench is a keyboard-first view over existing PTM services. It
does not introduce a second training, artifact, preprocessing, or search
implementation.

Training and bounded search run outside Textual's event loop and report
structured progress. Cancellation targets the active job. Configuration
changes mark a completed result stale so it cannot be exported under settings
that did not produce it.

The canonical layout keeps system and dashboard state visible while task views
cover configuration, deterministic XOR training, clause behavior, TA
population diagnostics, literals, predictions, sampled temporal inspection,
portable artifact export/loading/raw-record inference, and bounded
threshold/template/clause/tree/repair search. Native runtime and GNU Prolog
capabilities remain optional until a workflow needs them.

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
