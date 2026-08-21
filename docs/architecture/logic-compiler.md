# Logic compiler and execution planner

The logic compiler is the common lowering boundary between TA clauses, Prolog
artifacts, and execution backends. It prevents a symbolic producer from being
tied to a particular CPU instruction sequence or future CUDA representation.

```text
TA state | Prolog rule | Class II artifact
                   |
                   v
          canonical Boolean DAG
                   |
                   v
        self-contained logic program
                   |
          +--------+---------+
          |                  |
     CPU scalar       CPU packed-64
                           (today)
          |
          +--- CUDA sparse / warp tile / INT8 vote (future)
```

## Typed intermediate representation

The initial IR supports constants, input ports, NOT, AND, OR, XOR, implication,
equivalence, cardinality thresholds, and signed weighted thresholds. Implication
and equivalence are front-end operations: they lower immediately to canonical
Boolean nodes. A signed threshold preserves the ordinary TM decision rule:

```text
sum(positive_clause_outputs) - sum(negative_clause_outputs) >= 1
```

Each graph uses structural interning, so identical subexpressions share one node.
Construction currently performs:

- constant folding and identity elimination;
- double-negation elimination;
- associative flattening and commutative ordering;
- AND/OR idempotence;
- contradiction and tautology detection;
- XOR duplicate/complement cancellation;
- implication elimination and contrapositive canonicalization;
- conjunction factoring and Boolean absorption;
- threshold constant removal, weight coalescing, and bound folding.

The compiled program contains only nodes reachable from its output and remaps
them into a compact topological instruction sequence. It does not retain a
pointer to the source graph.

## CPU execution layouts

The scalar evaluator consumes ordinary row-major byte features. The packed
evaluator transposes a batch into feature-major words:

```text
packed[feature][word] -> 64 examples of one Boolean feature
```

Every Boolean instruction then evaluates 64 examples with ordinary `uint64_t`
operations. Unit-weight thresholds use a bit-sliced `at_least[k]` recurrence.
Signed thresholds, including the final TM vote, currently use a portable lane
fallback after their clause inputs have been evaluated bitwise. This is a clear
optimization target for AVX2/AVX-512 and GPU backends.

This batch layout is distinct from a fixed 32x32 or 64x64 PA input matrix. The
former packs examples across machine-word lanes; the latter names stable logical
slots within one PA. A backend may tile both dimensions, but their semantics are
not interchangeable.

## Planner contract

The current planner chooses only executable backends: portable scalar CPU or
portable packed-64 CPU. It records graph size, depth, fan-in, referenced inputs,
and two density measures. The first conservative heuristic selects packed
execution when a full batch word is already available, repeated use amortizes
the transpose, or a sufficiently large dense row-major batch pays for packing.

CUDA is deliberately not advertised as available by the planner. Future device
backends will be capability-gated and selected from measured crossover tables
using at least:

- rule arity and logical density;
- batch size and input layout;
- graph depth and common-subexpression reuse;
- output type (Boolean, count, signed vote, or proof diagnostics);
- transfer cost and device residency;
- register pressure, occupancy, and achieved memory bandwidth.

The expected GPU split is bit-packed integer execution for clause logic, sparse
indexed execution for irregular low-density rules, and INT8/tensor acceleration
for sufficiently large dense weighted votes. Tensor cores are not treated as a
general Boolean execution foundation.

## NoisyXOR calibration

`scripts/benchmark-logic.ps1` trains the deterministic scalar semantic model on
a supplied binary dataset, lowers the learned TA configuration to the IR, and
requires exact agreement among source, compiled scalar, and compiled packed
predictions before timing anything.

On the development Intel Core i3-13100, the supplied 5,000-row NoisyXOR test
split produced this first Release-build calibration (5 training epochs, 25
timing repetitions):

| Path | Examples/second |
| --- | ---: |
| Source scalar TM traversal | 0.49 million |
| Compiled scalar IR | 4.14 million |
| Packed IR, transpose once | 17.76 million |
| Packed IR, including transpose each pass | 9.40 million |

These figures characterize one evolving implementation on one machine. They
are not CUDA comparisons or general performance claims. The benchmark prints a
checksum and aborts on any semantic mismatch.
