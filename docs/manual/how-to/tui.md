# Use the terminal workbench

Use this guide to install, launch, and troubleshoot PTM's Textual workbench.

## Install and launch

Install the optional TUI profile described in [Install PTM](install.md), then
launch the built-in XOR session:

```bash
ptm tui --demo xor
```

Use MNIST when you need a sustained, interactive multiclass workload:

```bash
ptm tui --demo mnist
```

The MNIST workload reads `data/mnist.pkl`, binarizes pixels with the recorded
`pixel > 0.3` rule, and trains ten native one-vs-rest clause banks. It reports
validation accuracy each epoch and a final per-class confusion view. The
current multiclass path does not expose a portable adaptive snapshot, so
clause/TA/literal inspection, temporal snapshot sampling, and `.ptm` export
remain unavailable and fail closed. Use XOR for those binary snapshot paths.

This launches PTM's canonical workbench. The former screen-per-step interface
remains available during the transition with `ptm tui --style classic`; the
`single_pane` style name remains accepted as a compatibility alias for the
canonical workbench.

Use a terminal at least 80 columns wide and 24 rows high. Windows Terminal and
current Linux and macOS terminal emulators provide the best color and keyboard
support.

The footer and help overlay show active controls. The generated
[help-topic and keyboard reference](../reference/help-topics.md) comes from the
same registry as Textual's bindings; this guide does not maintain a second key
map.

## Troubleshoot the workbench

- If the command says the TUI extra is missing, activate the intended virtual
  environment and install the `tui` profile.
- If the layout is cramped, enlarge the terminal to at least 80x24.
- If an IDE intercepts shortcuts, use a standalone terminal.
- Export failures appear in the event dock and do not overwrite an existing
  artifact.
- If MNIST reports that its native runner is unavailable, build the
  `ptm_mnist_ovr_benchmark` CMake target or set `PTM_MNIST_RUNNER` to a built
  executable.
- A blank optional record field means missing. Enter `null` for an explicit
  null and use the exact typed value rules in the
  [preprocessing reference](../reference/preprocessing.md).
- If Search reports that GNU Prolog is unavailable, follow
  [Run bounded symbolic search](run-bounded-search.md).
- Windows search jobs set `LINEDIT=gui=no` in the child and use
  `CREATE_NO_WINDOW`, so they do not open a GNU Prolog GUI console.

For a guided introduction, see the
[first terminal-workbench session](../tutorials/first-tui-session.md). The
[workbench explanation](../explanation/workbench.md) describes background
jobs, stale-result protection, and current boundaries.
