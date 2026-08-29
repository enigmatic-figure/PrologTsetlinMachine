# Initial capacity campaign

This directory holds the small execution substrate for PTM's internal capacity
and cost measurements. It does not contain benchmark findings.

The first campaign has two tracks:

- `shared`: every implementation receives the same manifest-identified Boolean
  rows and labels;
- `native`: an implementation may use its own representation, with preparation
  cost reported separately.

Each attempted run is retained in JSONL, including failed and unsupported
attempts. The runner independently scores saved predictions. Timing keeps four
boundaries distinct: preprocessing/materialization, adaptive training, resident
inference, and a full PTA lifecycle episode. An unavailable boundary is `n/a`.

## Materials

The initial ceiling is XOR with controlled label noise, exhaustive parity,
PTM's four Logic encodings, WDBC, IMDb unigram presence, and MNIST 0-vs-8 bits.
Only the local synthetic and archived materials are wired today; the three
external datasets remain the next data-preparation slice.

Prepare the current local set from the repository root:

```text
python examples/logic_dataset_prepare.py --data-dir data/Logic --output-dir out/logic-dataset --encodings token_presence token_count_threshold position_one_hot ast_relational
python -m prolog_tsetlin.benchmark_campaign prepare-local-baselines . out/logic-dataset out/benchmark-campaign/materials
python -m prolog_tsetlin.benchmark_campaign prepare-synthetic out/benchmark-campaign/materials
```

Bulk material and raw results remain under ignored `out/` storage. Their
manifests carry the identities used by campaign records.

## Linux smoke

The incumbents are isolated at the commits in `incumbents.json`. Bootstrap them
with CPython 3.12; the script writes the resolved package freezes beside the
environments:

```text
bash scripts/bootstrap-benchmark-incumbents.sh out/benchmark-campaign/incumbents-linux
```

Build PTM's C++ scalar-training/packed-inference route:

```text
cmake -S . -B out/benchmark-campaign/ptm-native -G Ninja -DCMAKE_BUILD_TYPE=Release -DPTM_BUILD_TESTS=OFF -DPTM_BUILD_EXAMPLES=OFF -DPTM_BUILD_RUNTIME_CLI=OFF -DPTM_BUILD_BENCHMARKS=ON -DPTM_ENABLE_CUDA=OFF
cmake --build out/benchmark-campaign/ptm-native --target ptm_campaign_native_runner -j2
```

`run_smoke.py` sends the N=6 parity manifest through the Python semantic
reference, the PTM C++ route, pyTsetlinMachine, and TMU. This proves plumbing
only. It is not a performance or accuracy result.

`run_local_matrix.py` executes a predeclared, resumable local campaign. Its
default matrix covers generated XOR20 noise, the parity ladder, and all four
Logic encodings through PTM native, pyTsetlinMachine, and TMU. It retains the
exact plan and host/thread environment before the first run. Resume refuses a
changed plan, a changed host environment, duplicate run IDs, or records outside
the declared matrix:

```text
PYTHONPATH=python python3 benchmarks/initial_capacity/run_local_matrix.py --project-root . --material-root out/benchmark-campaign/materials --incumbent-root out/benchmark-campaign/incumbents-linux --ptm-native-executable out/benchmark-campaign/ptm-native/ptm_campaign_native_runner --ptm-commit <commit> --output out/benchmark-campaign/local-scout
```

Repeat `--variant` to restrict the material variants and `--total-clauses` to
declare a clause ladder. Repeat `--score-split` to retain several views of the
same fitted model; for example, `train` plus `validation` supports capacity and
generalization-gap analysis. By default the driver scores validation when
available. An explicit `evaluation` split is intended for a configuration
frozen after validation. PTM's native runner also retains independently checked
signed vote scores and separately timed clause-population diagnostics so that
measurement does not contaminate adaptive-training or resident-inference time.
For ladders whose smallest polarity has fewer clauses than the requested vote
threshold, `--threshold-policy clamp-to-polarity` records and applies the
deterministic rule `min(requested threshold, total clauses / 2)`.

After a completed native XOR ladder, `summarize_capacity_surface.py` verifies
the retained plan and vote-score digests, reconstructs clean XOR labels,
separates fit to flipped and unflipped training rows, and aggregates the clause
telemetry and per-state signed margins. It writes JSON and CSV beside the raw
campaign. Its one-standard-error clause selection is an exploratory,
deterministic reference rather than a deployed capacity governor:

```text
PYTHONPATH=python python benchmarks/initial_capacity/summarize_capacity_surface.py out/benchmark-campaign/xor-capacity-surface-v1 out/benchmark-campaign/materials
```

`package_colab.py` creates the deterministic allowlisted input archive consumed
by `colab_smoke.py`. On a CPU runtime the remote driver repeats the four-route
smoke. When both a GPU and `nvcc` are present, it additionally builds PTM with
CUDA and runs the existing packed-runtime correctness-gated GPU smoke. The
dataset routes remain labeled with their actual CPU backends. Invoke the T4
driver with `--require-gpu`; it then fails unless the requested CUDA backend
produces correctness-gated measurements.

Incumbent clause counts are per class. PTM's binary scalar clause pool instead
alternates positive and negative polarity. Campaign configurations must record
that distinction rather than comparing the raw clause-count fields as if they
were identical. The pinned pyTsetlinMachine code also has a fixed internal C
random stream but no public seed control; its records say so explicitly.

## MNIST JIT distillation scout

`run_mnist_jit_distillation.py` runs the bounded policy screen used to separate
feedback volume, feedback allocation, nonlinear priority weighting,
teacher-specific information, and soft-target timing. A common one-epoch
scalar PTM checkpoint is restored bit-exactly into fourteen fixed arms: the
four original references, uniform slowdown, three normalized disagreement
priorities, a protected baseline blend, student-only and shuffled-teacher
controls, delayed soft targets, and two additional agreement/teacher-advantage
diagnostics. Normalized arms calibrate once per bank from fixed pre-epoch
scores, solve the clipped probability budget exactly, then hold that scalar
while online states and probabilities evolve. Results retain both calibration
equality and the achieved online budget ratio.
The compact CPU ConvNet sees all 784 grayscale pixels; each exact PTM bank keeps
the existing 12-literal, thresholded representation. The mismatch is recorded
as part of the experiment rather than treated as a matched representation.
Multiclass selection and residual probabilities use unclipped signed clause
votes. The existing margin-clipped `score()` remains the binary inference and
training-gate contract, and the result records both forms for comparison. The
bounded bank schedule deliberately reuses a fixed 200-positive then
200-negative row order each epoch; results must be interpreted within that
training regime.

The policy screen uses `--skip-pta` so it measures controller and teacher
effects without launching 140 lifecycle cells. A later acceptance run may send
validation-selected finalists through executable Input/Escalation and
De-escalation PTA cells. Bulk checkpoints, logits, vectors, stores, and results
remain under ignored `out/` storage:

```text
python -m pip install -e ".[data,test]"
python -m pip install torch
PYTHONPATH=python python benchmarks/initial_capacity/run_mnist_jit_distillation.py --output out/benchmark-campaign/mnist-teacher-policy-screen --epochs 10 --teacher-epochs 3 --validation-rows 2000 --pta-audit-rows 2000 --skip-pta
```

`--skip-pta` is the isolation mode for controller and teacher experiments. It
leaves Input, Escalation, and De-escalation lifecycle execution uninvoked and
records `pta_skipped: true`; omit it only when PTA effects are themselves part
of the question.

The runner treats `mnist.pkl` and optional parent-checkpoint pickle files as
trusted local benchmark inputs; Python pickle is not an untrusted interchange
format. A new run publishes an immutable `plan.json` before training. Resume
requires the same configuration, source/runtime identities, and code digests,
then independently binds every reusable PTA cell to its parent snapshot's
content identity.

`analyze_mnist_checkpoint_scores.py` compares raw-vote and clipped-vote
multiclass decisions from an existing run without retraining. It reconstructs
the deterministic bank projections and validates them against the source
schema before publishing: legacy v1 runs must reproduce their reported clipped
accuracy, while v2 through v4 runs must reproduce both their primary raw-vote
accuracy and their separately stored clipped comparison.
