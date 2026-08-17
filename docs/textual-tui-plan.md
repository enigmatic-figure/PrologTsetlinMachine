# Textual TUI product and implementation plan

Status: implementation started. The initial vertical slice adds a shared XOR
training service and an optional Textual Overview/Train surface; the remaining
phases below continue to describe planned work. Textual remains outside runtime
contracts and inference hot paths.

## Executive direction

PTM should gain a keyboard-first **workbench**, not a terminal-shaped copy of
the Dear PyGUI dashboard. Its first-run experience should make the built-in XOR
example useful in under a minute, while its long-term shape should expose the
project's real strengths: provenance, exact oracle/native agreement, backend
selection, portable artifacts, bounded symbolic search, and the Class II audit
lifecycle.

The visual reference is Dolphie's dense but calm operational style: a compact
status header, selectable workspaces, live panels, a persistent event stream,
fast keyboard navigation, and recording/replay semantics. PTM should borrow
those interaction ideas rather than its database-specific information
architecture. The interface must remain usable over SSH, in a narrow terminal,
without a mouse, and without color.

The proposed command is:

```text
ptm tui [--workspace PATH] [--demo xor] [--replay SESSION.jsonl]
```

Textual should be an optional dependency so the dependency-free library and
existing `ptm` export commands remain dependency-free. A user without the extra
gets a short installation hint rather than an import traceback.

## What the repository can support

### Product surfaces already present

| Surface | Useful TUI capability | Current boundary |
| --- | --- | --- |
| Class I representation | inspect schemas, literals, encoded rows, typed facts, and provenance traces | Python API is ready; custom data ingestion is not a general product workflow yet |
| Scalar TM oracle | configure, train, predict, inspect TA state, snapshot | synchronous and intended as the semantic oracle, so UI work must leave the event loop |
| Native packed TM | show CPU capabilities, selected backend, clause outputs, feedback outputs, scores, and predictions | depends on a discoverable shared library; CUDA control is not exposed by the Python binding |
| Feature templates | browse template registry and analyze TA clause configurations | roadmap outputs that connect templates to bounded Prolog search are incomplete |
| Logic dataset/AST | load, split, encode, report collisions/signatures, inspect AST and primitive graph | valuable but more advanced than the first-run path |
| Prolog bridge | configure and run bounded monotone threshold search | GNU Prolog is optional and discovery can fail |
| Class II logic/morphology | inspect fixed programs, behavior signatures, repairs, factoring, and merges | Python models artifacts, but the live native registry and persistence control plane lack a Python facade |
| `.ptm` artifacts | export and load three artifact kinds; display manifests and conformance cases | the training-side CLI exports only; independent `ptmrt` owns inspect/verify/run |
| benchmark JSONL | stream environment, skipped cases, backend choice, timing scope, median rate, MAD, and checksum | schema is emitted by a separate executable/script and should be consumed, not reimplemented |

This distinction should be visible in the interface. Unsupported features are
labelled **planned** or **unavailable**, never represented by decorative panels
that imply live data exists.

### Lessons from the existing dashboard

Preserve the good entry-level loop: choose hyperparameters, train XOR, compare
predictions, inspect clause state, and export. Correct these structural issues
before building another frontend:

1. Move `DashboardState`'s domain operations into UI-neutral services. The TUI
   and GUI should call the same training, evaluation, and export code.
2. Replace module-global mutable state with a per-workspace session object.
3. Model long-running work as cancellable jobs with progress and structured
   events rather than synchronous callbacks and a text blob.
4. Never catch broad exceptions at the domain boundary. Convert known input,
   environment, native-runtime, and subprocess failures into typed user-facing
   failures; preserve unexpected tracebacks in a diagnostics view.
5. Derive tables from actual data. The current dashboard assumes exactly two
   XOR fields in its result formatting and only promises a future clause view.
6. Keep metadata and export paths in the session model, validate before writing,
   and show artifact verification as a separate result.

## Audience and journeys

### 1. Curious first-time user

Launch `ptm tui`, accept **Try XOR**, press `t` to train, watch epoch progress,
see four predictions and accuracy, open a clause, then press `e` to export a
verified artifact. Context help explains each parameter at the point of use.
No paper or manual is prerequisite.

### 2. Researcher iterating on a model

Open a workspace, load a data source, inspect literal provenance and collision
reports, compare runs with controlled seeds, drill into clauses and vote scores,
and save a reproducible session manifest. Parameter changes mark prior results
stale rather than silently mixing configurations.

### 3. Runtime/performance engineer

Attach to benchmark JSONL or open a saved stream, check detected CPU/GPU
capabilities, filter by backend/density/shape/timing scope, compare median
throughput and MAD, and verify checksums. Missing AVX/CUDA support is a normal
capability state, not an alarming error.

### 4. Class II / symbolic investigator

Inspect AST-to-primitive lowering, a fixed logic program, behavior signatures,
and morphology diffs; launch a bounded Prolog threshold search with candidate
count and deadline displayed up front. Later, when an orchestration API exists,
follow nomination, shadow audit, activation, drift, reopen, and replacement in
the same workspace.

### 5. Artifact consumer

Open a `.ptm`, inspect identity, kind, ports, feature/binding contract,
limitations, and conformance vectors; verify it; enter or load inputs; compare
results. This journey should delegate runtime truth to `ptmrt` (or future thin
bindings), never duplicate its decoder semantics in presentation code.

## Information architecture

### Persistent shell

```text
┌ PTM ─ xor-demo ─ TRAINING ─ scalar/oracle ─ seed 7 ─ 00:03 ───┐
│ Overview  Data  Train  Clauses  Logic  Artifacts  Benchmarks         │
├──────────────────────────────────────────────────────────────────────────┤
│ [active workspace content; panels reflow at terminal breakpoints]       │
│                                                                        │
├ Events ─ 14:02:11 INFO epoch 80/150 accuracy=75% ─────────────┤
│ t train  x cancel  / filter  p palette  ? help  q quit                │
└─────────────────────────────────────────────────────────────────────────┘
```

- **Header:** workspace, state, execution mode/backend, seed, and elapsed time.
- **Workspace bar:** stable top-level destinations with badges for activity,
  errors, and stale results.
- **Content:** a screen composed of focusable panels; selection drives a detail
  drawer rather than proliferating modals.
- **Event dock:** tail-like structured stream, collapsible with `ctrl+l`, with
  severity/source filters and pause/follow behavior.
- **Footer:** only currently valid bindings; `?` opens contextual help and `p`
  opens the command palette.

### Screens

**Welcome / setup** offers Try XOR, Open dataset, Open artifact, Watch benchmark,
Resume workspace, and Replay session. It runs an environment preflight for
terminal size, native library, CPU capabilities, GNU Prolog, `ptmrt`, CUDA
benchmark availability, and writable output directory.

**Overview** contains run state, latest accuracy and throughput, configuration
summary, environment/capabilities, recent artifacts, and a short next-action
card. It should be useful at 80x24, not merely a splash page.

**Data** contains source/schema, split, literal catalog, encoded rows, provenance
trace, and diagnostics (collision and evaluation-signature reports). Selecting
a cell traces raw field → transform → literal ID → packed position. Data
is paged or virtualized; it is never all rendered at once.

**Train** pairs an editable, validated configuration form with live progress,
accuracy, prediction table, and run comparison. Advanced settings are collapsed
by default. A changed configuration is visibly `DIRTY`; training captures an
immutable run configuration and seed.

**Clauses** provides a sortable clause table (polarity, output/feedback rates,
weight when applicable, included-literal count) and a detail view containing
TA state/action, named literals, provenance, clause output by row, and signed
vote contribution. The MVP may expose scalar snapshot facts; rate histories
require instrumentation and are explicitly deferred.

**Logic** has AST, primitive IR, fixed-program instruction, and behavior views.
A morphology comparison shows parent/child truth signatures and changed
assignments. A Search drawer displays the finite candidate bound, timeout,
allowed template, input examples, and resulting PA artifact. Registry lifecycle
telemetry remains a disabled subsection until a supported control-plane adapter
exists.

**Artifacts** is a library plus inspector. It displays artifact ID and kind,
metadata/limitations, named ports, catalog versions and stable IDs, conformance
cases, verification status, and inference input/output. Export is a reviewed
two-step action; overwrite requires confirmation. Copyable values should use
plain text without Rich markup.

**Benchmarks** tails versioned JSONL from stdin, a file, or a subprocess. It
shows environment, filters, backend availability, selected versus requested
backend, workload dimensions/density/pages, timing scope/source, median
examples/s, MAD, samples, checksum, and skips/errors. Small terminals use a
table plus numeric deltas; Unicode sparklines are an enhancement, never the only
representation.

### Responsive and accessible behavior

Define and test three layouts:

- **wide (≥120 columns):** two or three panels plus event dock;
- **standard (80–119):** master/detail panels with the event dock collapsed;
- **compact (<80 or <24 rows):** one panel at a time and a clear size warning,
  while help, cancel, logs, and quit remain reachable.

Color is redundant with text labels and symbols. Focus has a shape/border as
well as color. Respect `NO_COLOR`, offer high-contrast and ASCII-safe modes,
avoid rapid animation, and allow refresh rate reduction. Every action is
keyboard accessible with conventional Tab/Shift-Tab and arrow navigation.
Single-letter bindings do not fire while an input has focus.

## Interaction model

### Commands and safety

Global keys are `p` palette, `?` help, `/` filter/search, `ctrl+l` events,
`ctrl+s` save workspace, and `q` request quit. Screen-local keys include `t`
train, `x` cancel, `r` refresh/replay, Enter inspect, Space select, and `e`
export where applicable. The palette is the discoverable source of truth and
includes disabled-command reasons.

Training, search, benchmark launch, export, and verification have explicit
states: `idle → queued → running → cancelling → succeeded|failed|cancelled`.
Disable duplicate starts. Cancel subprocess groups and worker tasks cleanly.
Quit with active work offers wait, cancel and quit, or return. Destructive or
overwriting actions require a confirmation modal; ordinary navigation does not.

### Progressive disclosure and terminology

First-run copy says what a control changes before using specialized terms.
Short explanations link concepts without turning the footer into a manual:

- **clauses:** pattern voters;
- **states per action:** automaton memory depth;
- **specificity:** tendency toward more specific patterns;
- **threshold:** cap/scale for the signed vote;
- **Class I/II/III:** representation, consolidated behavior, bounded search.

An optional research mode reveals raw masks, instruction words, hashes,
generation tags, and diagnostic planes. It changes presentation only.

## Technical design

### Package shape

```text
python/prolog_tsetlin/
  services/                 # no Textual imports
    training.py             # TrainingRequest/Run and progress callback
    artifacts.py            # export/load/verify facade
    environment.py          # capability discovery
    benchmarks.py           # strict JSONL parser and stream controller
    logic.py                # AST/morphology/search presentation models
  tui/
    app.py                  # PTMApp, bindings, screen routing
    models.py               # SessionState, JobState, UI view models
    messages.py             # typed messages from services/workers
    screens/                # welcome + seven workspaces
    widgets/                # event log, status, tables, inspectors
    styles/                 # base, high-contrast, compact TCSS
```

Add a `tui` optional dependency group and a `ptm-tui` entry point; also route
`ptm tui` through a tiny lazy launcher. Keep all Textual imports under the TUI
package. Version-lock to a tested compatible Textual range rather than an
unbounded minimum, because widget and CSS behavior are part of this product.

### State and message flow

```text
Textual action
  → validate request on UI thread
  → service call in exclusive worker/thread or managed subprocess
  → typed Progress | Metric | Event | Result | Failure message
  → session reducer (single writer)
  → reactive view models
  → widgets update only changed rows/panels
```

The UI never reads a mutating machine while a training worker owns it. A worker
publishes immutable progress snapshots at a bounded frequency (target 4–10 Hz),
then transfers the completed model/run to the session reducer. Use monotonic
timestamps for rates/durations and wall-clock UTC only for display/persistence.
Bound all queues and coalesce progress events; never allow rendering to throttle
training or benchmark capture.

Subprocess adapters consume stdout and stderr separately, preserve raw lines,
parse recognized versioned JSONL into typed records, and turn unknown schema
versions/fields into visible compatibility events. They must not infer success
from pretty output: use exit status plus checksum/conformance status.

### Telemetry contract

Create one presentation-neutral event envelope before UI work expands:

```json
{
  "schema": "ptm.telemetry.v1",
  "sequence": 42,
  "timestamp_utc": "2026-08-17T14:02:11.123Z",
  "monotonic_ns": 123456789,
  "session_id": "...",
  "run_id": "...",
  "source": "training",
  "kind": "progress",
  "level": "info",
  "payload": {"epoch": 80, "epochs": 150, "accuracy": 0.75}
}
```

Payloads are typed by `kind`; required fields, units, nullable values, and
schema evolution rules belong in a separate contract when implemented. Do not
put secrets, full input rows, or arbitrary exception locals in events. The
recorder writes append-only JSONL with a session header, bounded flush policy,
and a terminal completion record. Replay uses the same reducer and widgets but
disables mutations and clearly labels simulated time.

Initial kinds: `session`, `capability`, `job_state`, `progress`, `metric`,
`prediction`, `artifact`, `benchmark`, `audit`, `log`, and `failure`. Include
stable dimensions (run, backend, artifact, clause where appropriate), explicit
units, and quality/status rather than encoding meaning in message strings.

### Integrations and ownership

- **Training:** extract the XOR workflow first, then accept a prepared literal
  batch. Epoch chunking is needed for meaningful progress and cancellation;
  verify that chunked fitting preserves deterministic semantics before use.
- **Native:** call existing capability and packed-evaluation APIs. Discovery
  failure falls back to the scalar oracle and is shown in preflight.
- **Artifacts:** use Python artifact APIs for training-side export/load. Invoke
  `ptmrt` for independent verification/inference until thin runtime bindings
  exist; capture its exact stdout/stderr and exit code.
- **Prolog:** retain candidate-count and subprocess deadline safeguards from the
  bridge. Present the command and bounds, not unrestricted query entry.
- **Class II registry:** add a read-only snapshot/telemetry facade before live
  panels. Do not bind the TUI directly to C++ persistence internals.
- **Benchmarks:** treat `ptm.benchmark.v1` JSONL as the source of truth and add
  parser fixtures for environment, result, skip, and future unknown records.

## Delivery roadmap

### Phase 0 — contracts and prototype (1–2 iterations)

- Interview/observe at least three target users completing XOR, artifact
  inspection, and benchmark diagnosis; turn failures into acceptance tests.
- Record golden terminal layouts at 80x24, 100x30, and 140x40.
- Define session/run/job view models, telemetry v1, error taxonomy, and JSONL
  benchmark parser.
- Extract UI-neutral training/export services without changing behavior.
- Build a throwaway Overview + event dock spike to validate Textual version,
  SSH terminals, Windows Terminal, WSL, cancellation, and update frequency.

**Exit:** architecture review accepts service boundaries; the spike stays
responsive under a synthetic high-rate stream; no new mandatory dependency.

### Phase 1 — impressive vertical slice (2–3 iterations)

- Ship Welcome, Overview, Train, Clauses (snapshot facts), Artifacts, contextual
  help, palette, event dock, preflight, and responsive themes.
- Make Try XOR → train → inspect → export → independently verify work.
- Save a minimal workspace manifest containing inputs by reference,
  configuration, seeds, run summaries, and artifact IDs.
- Add graceful no-native/no-`ptmrt`/no-Prolog states.

**Exit:** a clean environment reaches a verified XOR artifact in under two
minutes; all actions are keyboard-only; 80x24 is usable; cancel/quit never
leaves a child process; existing CLI and library tests pass.

### Phase 2 — data and diagnostics (2–3 iterations)

- Add Data schema/literal/row/provenance views and collision diagnostics.
- Generalize training beyond hard-coded XOR through a documented bounded input
  workflow; add immutable run comparison.
- Add paged/virtualized tables, filtering, exportable diagnostics, and replay of
  UI telemetry.
- Decide whether the Dear PyGUI entry point becomes a thin client of the shared
  services, is deprecated with a migration window, or remains an example.

**Exit:** no XOR-specific formatting exists in shared services or widgets;
large synthetic tables remain responsive and memory-bounded.

### Phase 3 — logic and benchmarking (2–4 iterations)

- Add Logic AST/IR/program/morphology inspectors and bounded search flow.
- Add benchmark live/file/stdin adapters, filtering, comparisons, and saved
  sessions with raw-record access.
- Add artifact inference and conformance-case exploration across all three
  artifact kinds.

**Exit:** unknown JSONL schema versions fail safely; benchmark checksums and
artifact conformance results are never obscured; bounded search shows limits
before launch and cancels reliably.

### Phase 4 — lifecycle operations (after control-plane APIs)

- Specify and implement a stable read-only Class II registry snapshot/event API.
- Visualize maturity, mapping generations, shadow/live audit windows, drift,
  replacement, reopen, and persistence/replay.
- Gate any mutation behind explicit orchestration commands, confirmation,
  authorization assumptions, and audit events.

**Exit:** snapshot plus event replay reconstructs the same displayed state;
generation changes and partial/unpublished transactions cannot be mistaken for
active routing.

## Verification strategy

### Automated layers

1. **Service unit tests:** deterministic XOR, request validation, typed errors,
   cancellation checkpoints, artifact paths, and environment discovery.
2. **Parser/contract tests:** golden telemetry and benchmark JSONL, malformed
   lines, oversized records, non-finite numbers, unknown kinds/versions,
   truncated streams, and stderr interleaving.
3. **Textual app tests:** use the framework's headless pilot to press every key,
   change focus, validate forms, start/cancel jobs, open/close modals, resize,
   and assert semantic widget state rather than fragile styling details.
4. **Snapshot tests:** normalized screen captures for the three breakpoints,
   ASCII/high-contrast/no-color modes, empty/loading/success/error/stale states.
5. **Integration tests:** mocked workers plus real scalar XOR; optional native,
   GNU Prolog, benchmark binary, and `ptmrt` tests behind capability markers.
6. **Soak/performance tests:** bounded memory during long streams, update burst
   coalescing, event-loop latency, repeated screen changes, and cancellation.

CI should run the core suite without Textual, then a TUI-extra job on supported
Python versions. Capture random seed, terminal dimensions, theme, locale, and
Textual version in failures. Do not require GPU or Prolog for the basic TUI job.

### Human acceptance checklist

- A newcomer can explain the prediction and find why a clause fired.
- A keyboard-only user can reach, operate, and leave every control.
- Screen-reader/plain-text logs contain the same state conveyed by color/graphs.
- Resize, disconnect-like cancellation, missing tools, corrupt artifacts, and
  malformed benchmark lines produce actionable recovery guidance.
- Researchers can copy exact IDs, values, masks, commands, and errors.
- Refresh never changes a trained result or hides that configuration is stale.

## Risks and decisions to settle early

| Risk/decision | Mitigation or decision gate |
| --- | --- |
| UI outruns real telemetry | inventory each displayed field and its authoritative producer; label deferred metrics |
| Event loop blocked by Python training | exclusive threaded worker, bounded progress, measured input latency; consider process isolation only if necessary |
| Chunked epochs change determinism | golden comparison against single `fit_literal_batch`; otherwise add a progress callback in the oracle |
| Textual becomes mandatory | optional extra, lazy import, core-only CI job |
| Duplicate `ptmrt` semantics | adapter delegates verify/run and parses a stable machine-readable mode |
| JSONL schema drift | versioned strict core plus preserved unknown fields/raw lines and compatibility errors |
| Huge rows/clauses/logs exhaust terminal | paging, bounded ring buffers, sampling summaries, explicit export for full data |
| Unicode/color portability | ASCII and no-color modes tested in CI; never encode status only visually |
| TUI becomes an unsafe control plane | read-only first; bounded operations; confirmations and audit events for later mutation |
| Two frontends diverge | shared services and view models; make an explicit Dear PyGUI support decision in Phase 2 |

## Success measures

Measure the product rather than panel count:

- median time and error count from launch to correct XOR training and verified
  export;
- percentage of first-run tasks completed without external documentation;
- cancellation latency and 95th-percentile UI input latency during work;
- peak memory under a one-hour benchmark/event stream;
- proportion of errors that identify a recovery action;
- accessibility completion rate at 80x24, no-color, and keyboard-only;
- automated coverage of every command state and every optional-tool absence.

The strongest first impression will come from truthfulness and fluency: PTM
starts quickly, teaches its concepts in context, keeps the log-centric feel of
terminal work, exposes exact diagnostic data on demand, and never pretends a
roadmap capability is already observable.
