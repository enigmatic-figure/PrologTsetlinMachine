# Logic evaluator Class II consolidation

The first structural Class II artifact closes the deterministic Logic benchmark
by compiling typed Boolean programs rather than enlarging the flat TA
population. It is a shared native execution kernel plus a separate fixed-shape
state partition:

```text
typed Logic AST
      |
      v
canonical primitive graph
      |
      v
32-instruction state block -----> shared native evaluator -----> Boolean result
      ^                                      |
      |                                      +---- diagnostic truth mask
five binding bits
```

No dataset label participates in parsing, lowering, or compilation. Labels are
used only by the validation and shadow-audit phases.

## Fixed program shape

Each aligned `ptm_logic_program32` occupies 320 bytes. It contains up to 32
topologically ordered instructions. An eight-byte instruction stores an opcode,
an argument, and a 32-bit mask naming earlier instruction outputs. Version 1
supports constants, the five inputs `A` through `E`, NOT, AND, OR, and XOR.

The kernel rejects unknown opcodes, bad input indices, malformed arity, forward
references, non-final roots, and out-of-range binding bits. Evaluation returns:

- the root truth value;
- a mask of true instruction outputs, usable as a compact proof diagnostic;
- a mask of evaluated instructions.

Machine code is shared and immutable. The aligned program blocks and five-bit
binding values are prepared once and remain resident, so repeated polling has
no Python-to-native gather phase. This is the shared-engine/state-hot-swap
design anticipated for fixed-shape PAs.

## Artifact and restoration boundary

`LogicEvaluatorArtifact` is content-addressed over the opcode contract, source
AST and primitive-graph schema versions, mapping version, validation signature,
and restoration handle. It declares a 32x32 literal-truth input plane whose
first five slots map `logic_binding:A` through `logic_binding:E`, making it
compatible with the existing consolidation registry after those external IDs
are interned into dense source handles. The corresponding state manifest interns
unique program blocks by content hash and maps each source row to a program ID
and binding byte.

The current experimental restoration handle names the deterministic flat-TM
retraining recipe at commit `74e455f`. It is sufficient to reproduce this
baseline, but it is not yet a persisted bit-exact TA snapshot. Production
absorption must replace it with the snapshot/event-log persistence still listed
on the roadmap.

## Shadow audit result

The fixed split remains seed `20260806`: 4,000 training rows and 1,000 held-out
shadow rows. The supplied corpus lowers as follows:

| Measurement | Result |
| --- | ---: |
| Program instruction capacity | 32 |
| Observed instruction range | 1–16 |
| Mean instructions | 9.3258 |
| Unique compiled programs | 4,717 |
| Training accuracy | 100.0% |
| Held-out shadow accuracy | 100.0% |
| Residual error after the 63.8% flat-TM frontier | 0.0% |

There are zero disagreements among the supplied labels, typed AST evaluator,
primitive graph evaluator, Python fixed-program oracle, and native kernel. The
native kernel also agrees with the fixed-program oracle on all 150,944 possible
five-bit bindings across all unique compiled programs. The resulting shadow
decision is `activate`.

On the provisioned Core i3-13100 Windows host, observed runs of 100 prepared
evaluations of the 5,000-program batch executed at 7.5–7.8 million
programs/second. This timing
includes C ABI dispatch, validation, evaluation, and result writes, but excludes
the roughly 0.24–0.27-second compilation and Python result-object
materialization.
It is a local baseline, not a general hardware claim.

Reproduce the build, audit, state manifest, artifact, and timing report with:

```powershell
.\scripts\benchmark-logic-consolidation.ps1 -Repeats 100
```

Outputs are written under `out/logic-consolidation`.

## What this proves—and what it does not

This result demonstrates the intended higher-resolution path:

```text
Class I typed structure -> compiled Class II artifact -> native Boolean output
```

It is not evidence that the 20-clause flat TM learned a perfect evaluator. The
artifact absorbs the stable, specified operator semantics exposed by Class I;
it does not memorize labels, but it also does not yet recycle a measured set of
contributing TAs. The next integration step is to route this artifact output as
a derived literal, release only the clauses it behaviorally supersedes, and use
the live auditor to reopen them if a future grammar or operator violates the
versioned contract.
