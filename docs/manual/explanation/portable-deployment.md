# Portable deployment

A learned behavior is difficult to reproduce or compose when it cannot leave
the training workspace. PTM therefore separates mutable training state from a
canonical, immutable inference artifact.

```text
mutable training state
        |
        | freeze and lower
        v
canonical inference artifact
        |
        | package, hash, and verify
        v
small immutable runtime object
```

A resumable snapshot retains automaton values, random state, feedback
configuration, lifecycle checkpoints, and every dependency needed to continue
learning exactly. A deployable artifact retains only the canonical learned
computation, its feature/output schemas, provenance, and conformance evidence.
The runtime neither requires nor interprets mutable training state.

Backend-prepared layouts may be caches, but cannot be the only semantic form.
This keeps scalar execution mandatory and makes acceleration conditional on
passing the artifact's exact conformance gates.

ONNX is an interoperability target rather than the canonical format. A future
bridge may lower computations to standard operators when exact, or use a
versioned PTM custom operator. The in-memory runtime API already allows a `.ptm`
payload to live inside another container without changing its identity; no
ONNX bridge is claimed as implemented today.

See the [artifact reference](../reference/artifact-contract.md) for current
formats and ports.
