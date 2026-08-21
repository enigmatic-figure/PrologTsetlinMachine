# First terminal-workbench session

This tutorial explores the complete built-in XOR journey through the Textual
workbench. The native runtime and GNU Prolog are not required.

## Launch the workbench

Complete the [workbench installation](../how-to/tui.md), then launch its XOR
demonstration.

## Follow the displayed workflow

1. Open the Train view and start training with the displayed configuration.
2. Watch epoch progress and accuracy in the event dock.
3. Open Clauses and compare the predictions with learned clause state.
4. Open Artifacts and export the completed model as a `.ptm` file.
5. Load and verify the exported artifact in the same view.
6. Enter `true` or `false` for the generated `x0` and `x1` fields and run the
   record to inspect preprocessing and inference.

Configuration changes mark completed results stale, so the workbench will not
export an old result as though it represented new settings.

Open contextual help in any view for the current controls. The generated
[help-topic and keyboard reference](../reference/help-topics.md) is built from
the same registry as the application bindings.

Next, use [Run bounded symbolic search](../how-to/run-bounded-search.md) if you
want to add GNU Prolog to the workbench.
