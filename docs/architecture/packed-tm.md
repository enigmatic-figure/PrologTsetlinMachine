# Packed 64-example Tsetlin inference

> Status: current architecture contract. Performance measurements have been
> extracted to the [Packed TM CPU inference benchmark](../benchmarks/packed-tm-cpu.md) record.

The native adaptive substrate now has a direct batch path that does not lower a
trained TM through the general Boolean graph. `PackedTMModel64` prepares one
immutable scalar-TM snapshot, retains its exact multi-state automata as bit
planes, derives packed Include masks, and evaluates up to 64 examples at once.

## Exact TA state planes

Automata retain clause-major literal order:

```text
automaton = clause * (2 * feature_count) + literal
literal   = x0, NOT x0, x1, NOT x1, ...
```

For a maximum state `2N`, the prepared model allocates
`bit_width(2N)` planes. Plane `b`, word `w`, bit `i` stores bit `b` of the
corresponding TA state. No Include/Exclude projection is used as a substitute
for state: every value in `1..2N` round-trips exactly. A separate immutable
Include mask is derived for clause execution.

This first prepared model is an inference image. When training changes its
source TA population, orchestration prepares a new image from the new snapshot.
Bit-parallel feedback mutation is a later execution specialization, not
silently approximated here.

## Example-major lanes

The execution input is feature-major:

```text
feature_words[f], bit lane -> feature f for example lane
```

One `uint64_t` therefore carries one feature for 64 examples. The explicit
valid-lane mask supports every partial batch from one through 64 and prevents
unused high bits from contributing to clauses or votes.

Class I `LiteralBatch.feature_major_words64()` transposes one page directly
from its packed literal rows. `NativePackedTsetlinMachine.evaluate_literal_batch`
passes those words across the C ABI without expanding them into 64 native
calls.

## Clause and vote semantics

For each included positive literal, the clause word is ANDed with its feature
word. An included negative literal uses the complemented feature word. Both
are bounded by the valid-lane mask.

The result retains two clause vectors:

- prediction outputs, where an empty clause is false;
- feedback outputs, where an unconstrained empty clause is true.

This distinction matches the scalar semantic oracle and is necessary for later
native feedback. Even-indexed clauses vote `+1`; odd-indexed clauses vote `-1`.
Every lane score is clamped to the configured threshold, and prediction is
`score > 0`.

Model preparation also flattens the Include masks into an immutable execution
plan: clause offsets index compact feature and negation arrays. This preserves
the exact state planes for restoration while removing packed-mask discovery
from the hot inference loop.

Three interchangeable kernels consume that plan:

- portable scalar evaluation walks only included literals and set output bits;
- AVX2 evaluates four clause words together and accumulates eight vote lanes at
  a time;
- AVX-512 evaluates eight clause words together and accumulates sixteen vote
  lanes at a time.

The vector kernels skip vote work for zero-output clauses. Every compiled SIMD
translation unit has its own ISA flag; the common library and dispatcher are
compiled for the baseline architecture. CPUID plus XCR0 checks therefore occur
before a vector function can be called, including verification that the OS
saves the required register state.

`automatic` dispatch is deliberately conservative. The current clause-parallel
vector kernels are selected only for sparse prepared plans averaging no more
than one included literal per clause and for 16--128 clauses. Other shapes
remain scalar. Callers can force any available backend
for calibration and equivalence testing; forcing an unavailable ISA fails
instead of silently falling back.

Non-x86 builds contain only the portable kernel. The same configuration can be
tested deliberately on x86 with `-DPTM_ENABLE_X86_SIMD=OFF`; capability reports
then distinguish supported hardware from kernels present in the binary.

## C ABI

`ptm_tm_model_create` copies clause-major `uint16_t` states into an immutable
opaque native model. The source array may be released after creation.
`ptm_tm_model_eval_packed64` consumes:

- one input word per feature;
- the valid-example mask;
- caller-owned prediction and feedback clause-word arrays;
- a fixed 320-byte result containing the valid mask, prediction mask, and 64
  signed scores.

Model evaluation is read-only and can be invoked concurrently when each caller
owns its input and output buffers. `ptm_tm_model_destroy` releases the prepared
state. All validation failures return status codes; no C++ exception crosses
the ABI.

`ptm_cpu_capabilities_query` reports the CPU brand, hardware/OS capability
flags, compiled kernels, and generally preferred ISA. Model-specific
`ptm_tm_model_selected_backend` reports the automatic choice after plan density
is known. `ptm_tm_model_eval_packed64_backend` accepts an explicit automatic,
scalar, AVX2, or AVX-512 request. Python exposes the same contract through
`native_cpu_capabilities()`, `PackedTMBackend`, `selected_backend`, and the
`backend=` evaluation argument.

## Correctness gates

Native and Python tests compare the packed path against independent scalar
evaluation for:

- every XOR clause, score, and prediction across a full 64-lane word;
- 1, 17, 37, 63, and 64 valid lanes;
- randomized TA states and inputs;
- feature counts spanning multiple 64-bit state words;
- empty and contradictory clauses;
- prediction and feedback clause semantics;
- state-plane round trips, threshold clipping, invalid dimensions, invalid
  states, and insufficient caller capacity;
- direct Class I literal-batch input;
- exact intermediate equivalence across every available scalar/AVX2/AVX-512
  backend on randomized plans;
- exact intermediate equivalence across optional CUDA sparse and warp-tile
  backends for multiple resident pages;
- CUDA edge cases covering 1/17/63/64 valid lanes, odd and 257-clause shapes,
  65/257-feature boundaries, empty and contradictory clauses, negative
  literals, threshold clipping, and repeated resident execution;
- zero Compute Sanitizer memcheck findings on the CUDA correctness suite;
- capability coherence, model-specific selection, and rejection of forced
  unavailable backends through C++, C, and Python.


Performance measurements and calibration data are recorded in the
[Packed TM CPU inference benchmark](../benchmarks/packed-tm-cpu.md) benchmark record.

The experimental WSL/CUDA route is documented separately in
[CUDA packed TM execution](cuda-packed-tm.md).
