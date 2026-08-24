# First terminal-workbench session

This tutorial explores the complete built-in XOR journey through the Textual
workbench. The native runtime and GNU Prolog are not required.

## Launch the workbench

Complete the [workbench installation](../how-to/tui.md), then launch its XOR
demonstration.

## Follow the displayed workflow

1. Open Config with `c`, inspect the displayed request, and start training with
   `t`.
2. Watch accuracy and sampled model movement in Dashboard, Graphs, Events, and
   Timeline.
3. Open Clauses with `3`, then inspect clause support, signed contribution,
   literal similarity, and a selected example's TA truth.
4. Open Artifacts with `7` and export the completed model as a `.ptm` file.
5. Load and verify the exported artifact in the same task view.
6. Enter `true` or `false` for the generated `x0` and `x1` fields and run the
   record to inspect preprocessing and inference.

Configuration changes mark completed results stale, so the workbench will not
export an old result as though it represented new settings.

Open help with `?` for the current controls. The generated
[help-topic and keyboard reference](../reference/help-topics.md) is built from
the same registry as the application bindings.

Next, use [Run bounded symbolic search](../how-to/run-bounded-search.md) if you
want to add GNU Prolog to the workbench.
