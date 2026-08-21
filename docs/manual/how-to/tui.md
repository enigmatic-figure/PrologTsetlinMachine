# Use the terminal workbench

Use this guide to install, launch, and troubleshoot PTM's Textual workbench.

## Install and launch

Install the optional TUI profile described in [Install PTM](install.md), then
launch the built-in XOR session:

```bash
ptm tui --demo xor
```

Use a terminal at least 80 columns wide and 24 rows high. Windows Terminal and
current Linux and macOS terminal emulators provide the best color and keyboard
support.

The footer and contextual help show active controls. The generated
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
