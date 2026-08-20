# Terminal workbench

The PTM terminal workbench is the recommended interactive introduction. It
runs in a regular terminal, starts with a built-in XOR session, and does not
require the native runtime or GNU Prolog.

## Install and launch

From a source checkout and activated virtual environment:

```bash
python -m pip install ".[tui]"
ptm tui --demo xor
```

Use a terminal at least 80 columns wide and 24 rows high. Windows Terminal,
modern Linux terminal emulators, and current macOS terminal alternatives work
best with Textual's color and keyboard support.

## First session

1. Press `t` to train with the displayed configuration.
2. Watch epoch progress and accuracy in the event dock.
3. Press `2` for predictions and `3` for clause state.
4. Press `e` to export the completed model as a `.ptm` artifact.
5. Press `4`, then `l`, to open and verify the exported artifact.
6. Enter `true` or `false` for the generated `x0` and `x1` fields, then press
   `r` to inspect preprocessing and run inference.

## Keyboard map

| Key | Action |
| --- | --- |
| `1`-`5` | Switch workbench views |
| `t` | Start training |
| `x` | Cancel the active job |
| `e` | Export the completed model |
| `l` | Load and verify the artifact path |
| `r` | Run the typed record in the loaded artifact |
| `F5` | Run the bounded Prolog request |
| `F6` | Cancel an active Prolog search |
| `o` | Open the Overview view |
| `c` | Open the Clauses view |
| `p` | Open the command palette |
| `Ctrl+L` | Collapse or expand the event dock |
| `?` | Open contextual help |
| `q` | Quit |

Configuration changes are validated before a job starts. A completed result is
marked stale when its source configuration changes, so it cannot be exported
accidentally as if it represented the new settings.

## What the workbench currently covers

- Environment and optional-capability preflight.
- Editable deterministic XOR training.
- Cancellable background execution and structured telemetry.
- Prediction and exact clause-state inspection.
- Portable packed-TM artifact export and safe path handling.
- Portable artifact loading, metadata inspection, and conformance verification.
- Schema-driven numeric, Boolean, and typed-category record controls.
- Raw-record inference with per-literal preprocessing traces.
- Editable, budgeted threshold/template/clause/tree/repair searches.
- Search cancellation, repair counterexamples, and fixed-Logic export.

Custom datasets, live native registries, raw-record inference for Logic/PA
artifacts, and an interactive view over the implemented streaming connectors
are later workbench milestones. The detailed [product and implementation
plan](textual-tui-plan.md) tracks those extensions.

## Troubleshooting

- If `ptm tui` says the extra is missing, activate the intended environment and
  run `python -m pip install ".[tui]"`.
- If the layout is cramped, enlarge the terminal to at least 80x24.
- If keys are intercepted by an IDE terminal, try a standalone terminal.
- Export failures appear in the event dock and do not overwrite an existing
  artifact.
- A blank optional record field is treated as missing. Enter `null` for an
  explicit null, `true`/`false` for Boolean values, a JSON number for numeric
  fields, and quotes around categories that must remain numeric-looking strings.
- If Search reports that GNU Prolog is unavailable, install it and place
  `gprolog` on `PATH`, or set `PTM_GPROLOG` before launching the workbench.
- Windows Search jobs automatically force `LINEDIT=gui=no` in each GNU Prolog
  child and use `CREATE_NO_WINDOW`; no GUI console or close dialog should open.
