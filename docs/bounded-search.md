# Bounded Prolog search

PTM treats GNU Prolog as a finite, offline search participant. Every supported
request is validated in Python before a subprocess starts, including its
candidate ceiling, example count, deadline, and kind-specific structural
bounds.

## Try the built-in searches

Install GNU Prolog, ensure `gprolog` is on `PATH` or set `PTM_GPROLOG`, then run:

```bash
ptm search threshold --demo --pretty
ptm search feature-template --demo --pretty
ptm search ta-clause --demo --pretty
ptm search decision-tree --demo --pretty
ptm search repair --demo --pretty
```

On Windows, PTM launches `gprolog.exe` headlessly with the child-only setting
`LINEDIT=gui=no`, redirected standard streams, and `CREATE_NO_WINDOW`. This
prevents GNU Prolog's linedit GUI console and close dialog from appearing; it
does not change the user's environment or affect interactive GNU Prolog
sessions. The `--gui-console` option belongs to GNU Prolog's compiler/linker and
is not a runtime disable switch. PTM's verification scripts use `gplc.exe` when
they only need to compile a Prolog source file.

Decision-tree and repair behavior can be lowered to the fixed five-binding
Logic runtime and exported without overwriting an existing file:

```bash
ptm search decision-tree --demo --output out/xor-tree.ptm --pretty
ptm artifact verify out/xor-tree.ptm
```

Successful commands exit with status 0. A valid search with no exact solution
emits a `no_solution` result and exits with status 3. Invalid requests or
runtime failures exit with status 2. Ctrl+C terminates and reaps the active GNU
Prolog process before the CLI exits.

## Request contract

Custom requests use `ptm.search.request.v1`:

```json
{
  "schema": "ptm.search.request.v1",
  "kind": "decision-tree",
  "timeout_seconds": 30,
  "problem": {
    "slot_count": 2,
    "max_depth": 2,
    "examples": [[], [0], [1], [0, 1]],
    "labels": [0, 1, 1, 0]
  }
}
```

Save the object and provide it to the matching command:

```bash
ptm search decision-tree xor-tree.json --pretty
```

The command kind and document kind must agree. Unknown top-level fields are
rejected. The deadline must be between 0.1 and 300 seconds and can be overridden
with `--timeout`. JSON numbers and integers are type-checked rather than coerced
from strings or Booleans.

### Problem fields

| Kind | Required problem fields | Bound |
| --- | --- | --- |
| `threshold` | `slot_count`, `max_selected`, `positive_examples`, `negative_examples` | At most 1,000,000 selected-mask/threshold candidates |
| `feature-template` | `candidates`, `labels`, `coverage` | 1–4096 registry-backed typed candidates |
| `ta-clause` | `feature_count`, `max_literals`, `examples`, `labels` | At most 1,000,000 signed-literal conjunctions |
| `decision-tree` | `slot_count`, `max_depth`, `examples`, `labels` | Read-once depth at most 8 and at most 1,000,000 trees |
| `repair` | decision-tree fields plus `parent`; optional top-level `max_iterations` | At most 256 counterexample iterations within the tree bound |

Examples contain the zero-based indices of active Boolean inputs. TA literal
indices use `2 * feature` for the positive literal and `2 * feature + 1` for
its negation. Feature-template candidates contain `field_name`, `template_id`,
`data_type`, and a `parameters` object; template IDs and types are checked
against the Python registry before Prolog runs.

A repair parent is recursively represented as either `{"leaf": false}` or:

```json
{
  "feature": 0,
  "false": {"leaf": false},
  "true": {"leaf": true}
}
```

## Result contract

Solved searches emit `ptm.search.result.v1` with:

- the search kind and elapsed time;
- the declared candidate upper bound and dataset digest;
- zero mismatches after independent Python validation;
- a typed template, signed clause configuration, tree, or repair guard;
- counterexamples for repair; and
- `exportable: true` when the behavior fits the fixed five-binding Logic ABI.

Prolog output is never accepted directly as a deployable result. Python checks
the returned index, literals, tree bounds, read-once property, and every labeled
example before the service reports success.

## Terminal workbench

Open Search in `ptm tui`, choose a search kind to load its bounded example, and
edit the JSON and deadline before running it. The status line shows the
candidate ceiling before launch. Repair results populate a counterexample table;
portable tree/repair results can be exported and then opened in the Artifact
workspace. The generated
[workbench-control tables](manual/reference/help-topics.md) list the current run
and cancel bindings.
