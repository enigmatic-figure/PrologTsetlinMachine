# Logic morphology and transactional replacement

Logic morphology changes the shape of a deployed rule without mutating an
active artifact in place. Every operation produces a new immutable
`LogicProgram32`, an exhaustive behavior signature, and a content-addressed
lineage artifact that retains its parent program IDs.

## Behavior signatures

The current Logic grammar has five Boolean bindings, so its complete behavior
fits in one 32-bit word. Assignment bit positions use `A` as bit 0 through `E`
as bit 4. Signature bit `i` is the program result for assignment `i`.

This makes several normally expensive questions exact and cheap:

- equivalence is a 32-bit equality test;
- behavioral distance is one XOR and popcount;
- specialization proves the child's true set is a subset of its parent;
- generalization proves the parent true set is a subset of its child;
- a point patch proves exactly one assignment changed.

Larger typed domains will require sampled signatures, symbolic proofs, or a
bounded decision procedure. They must not silently claim exhaustive equivalence.

## Version 1 operations

| Operation | Construction | Required invariant |
| --- | --- | --- |
| `specialize` | `parent AND guard` | introduces no new true assignments |
| `generalize` | `parent OR extension` | removes no true assignments |
| `patch_true` | `parent OR exact_cube` | changes one false assignment to true |
| `patch_false` | `parent AND NOT exact_cube` | changes one true assignment to false |
| `conditional_compose` | `if condition then branch_a else branch_b` | matches both routed branches on all 32 assignments |
| `equivalence_merge` | select the smallest equivalent program | all parents have identical signatures |

Construction uses structural interning, constant folding, associative
flattening, complement detection, absorption, XOR cancellation, and dead-node
removal. Conditional composition interns both branches into one builder, so
shared inputs and subexpressions are factored automatically. Results that do not
fit the 32-instruction shape raise `MorphologyCapacityError`; they are candidates
for a larger PA shape or Class III refactoring.

## Counterexample repair

An exact five-literal cube identifies one binding assignment. Applying an
exception therefore cannot disturb an already-correct neighboring assignment.
This is intentionally conservative: repeated point patches are not allowed to
grow without bound. Capacity pressure is the signal to factor related
counterexamples into a guard, split a specialized branch, or escalate to a
richer search.

The morphology artifact records:

- operation and mapping version;
- parent program IDs and behavior signatures;
- child program ID and behavior signature;
- changed assignments and child instruction count;
- counterexample evidence when applicable;
- all parent IDs as the restoration lineage;
- an exhaustive 32-assignment, zero-mismatch validation claim.

The program store persists both parent and child blocks, so dissolution can
resolve the recorded parent rather than attempting to reverse Boolean
simplification.

## Transactional registry handoff

An audited child commonly consumes the same sources as its active parent. A
normal activation would correctly reject those mappings as conflicts.
`replace_active(parent, child)` provides the explicit morphology handoff:

```text
parent ACTIVE + child SHADOWING/audit accepted
                    |
                    v
child ACTIVATING -- generation-tagged atomic rebind of every source
                    |
                    v
child ACTIVE ------ parent REOPENING
```

Each source word moves directly from the parent artifact/slot to the child
artifact/slot and advances its generation. While the child is `activating`, a
reader that observes a moved word falls back to the source substrate because
only `active` artifacts resolve. After every rebind succeeds, the child is
published active and the parent enters reopening. An in-flight reader may see
the old valid parent, the new valid child, or a safe fallback—never a partially
active child.

If a compare-and-swap fails, completed moves are rebound to the parent before
the child returns to shadowing. A native stress test performs 1,000 alternating
parent/child replacements under four concurrent readers and accepts only the
two valid artifact/slot pairs or fallback.

## Controlled drift result

The executable example compiles the first Logic dataset program, changes its
expected output at one assignment, and repairs that drift from the resulting
counterexample:

| Measurement | Result |
| --- | ---: |
| Parent instructions | 10 |
| Patched child instructions | 15 |
| Assignments changed | 1 |
| Parent drift mismatches | 1 |
| Child drift mismatches | 0 |
| Native exhaustive mismatches | 0 |
| Conditional factoring savings | 3 instructions |
| Equivalent-program merge savings | 13 instructions |

Reproduce the lineage artifact, parent/child program store, native validation,
and report with:

```powershell
.\scripts\benchmark-logic-morphology.ps1
```

Outputs are written under `out/logic-morphology`.
