# Class II consolidation lifecycle

This subsystem separates candidate discovery, control-plane transitions, hot
source routing, and behavioral auditing. The separation is intentional: no one
data structure has suitable behavior for all four jobs.

## Candidate discovery

`DisjointSet` groups TAs or clauses that repeatedly participate in the same
candidate subformula. Union-by-size and path compression make rebuilding a
nomination epoch inexpensive.

Union-Find is not the active mapping. It handles union efficiently but cannot
split an absorbed set when an artifact reopens or dissolves. Candidate clusters
are therefore disposable and can be rebuilt from current evidence.

## Active mapping

`ConcurrentMappingTable` is a fixed-capacity dense array indexed by a 32-bit
runtime source handle. External TA, clause, and literal identifiers must first
be interned into that dense namespace.

Each source occupies one atomic 64-bit word:

| Bits | Field |
| ---: | --- |
| 0–11 | PA slot, supporting 0–4095 |
| 12–35 | Dense artifact handle |
| 36–62 | Mapping generation |
| 63 | Bound flag |

A raw lookup is one bounds check and one acquire-load. Resolving an active
consolidation adds one pointer load and one artifact-state acquire-load. There
is no hashing, allocation, reference count, or mutex on this path.

Artifact handles are dense and never reused during a registry lifetime. This
keeps stale mapping words from resolving to an unrelated replacement artifact.

Bind and release use compare-and-swap. Release increments the 27-bit generation
so a delayed writer cannot clear a newer mapping—the common ABA case. A process
that approaches 134,217,728 bind/release cycles on one source must rebuild the
mapping table before the generation wraps.

`is_lock_free()` reports whether the host implements the 64-bit atomic without
a library lock. It reports true on the current x64 MSVC environment.

## Publication protocol

Activation and reopening use artifact state as the publication barrier:

```text
shadowing
   |
   | audit accepted
   v
activating -- populate every source mapping -- publish ACTIVE
   |                                            |
   | conflict: CAS rollback                     v
   +---------------------------------------> hot readers resolve

ACTIVE -- publish REOPENING -- release mappings -- reopening
             |
             +-- hot readers immediately fall back to source TAs
```

Mappings written for an `activating` artifact are invisible because `resolve`
requires the artifact state to be `active`. Reopening publishes the non-active
state before clearing any mapping, so stale entries safely fall back rather
than route to a partially dismantled artifact. Partial activation is rolled
back with the exact generation-tagged words that were published.

An audited morphology child can replace an active parent with the same source
set. `try_rebind` changes each atomic word directly from the expected parent to
the child and increments its generation. The child remains invisible in
`activating` until all moves succeed; then it becomes active and the parent
enters reopening. Failed handoffs rebind completed entries to the parent, again
advancing generations so stale releases cannot affect either artifact.

Registry mutations use one control-plane mutex. This does not participate in
resolution or audit recording and makes multi-entry lifecycle transitions easy
to reason about. Runtime objects and audit windows remain at stable addresses
for the registry lifetime.

## Lifecycle

```text
nominated -> validated -> compiled -> shadowing -> activating -> active
                                            ^                       |
                                            |                       v
                                            +---- reopening <-------+

non-active states -> rejected
non-terminal states -> dissolved
```

Nomination checks configurable precision, support, recent movement, feedback,
reuse, and perturbation thresholds. A specification retains its mapping
version and restoration handle after rejection or dissolution so the
orchestrator can reconstruct the original TA substrate.

The fixed Logic evaluator exercises this same lifecycle with five
literal-truth sources mapped to slots 0–4 of a 32x32 plane. Its native
integration test performs exhaustive shadow observations before activation and
then verifies that every source resolves through the generation-tagged hot
mapping. The compiled instruction matrix remains a separate aligned state
partition owned by the artifact runtime.

## Shadow auditor

Each artifact owns a sequence-stamped atomic ring. Recording an expected/actual
pair reserves one sequence and updates one slot; it does not lock. A cold
snapshot scans the fixed window and ignores overwritten, future, or in-flight
entries. Concurrent uncertainty can temporarily reduce `observed`, which makes
decisions conservative rather than understating mismatches.

The policy treats shadow and live phases differently:

- shadowing requires a minimum sample count and a maximum mismatch rate before
  activation;
- live auditing requests reopening only when both mismatch count and rate cross
  their configured thresholds;
- a fresh audit window is published for every new shadow and live phase.

Each window carries its owning lifecycle phase. A recorder that races a phase
transition refuses to write if either the state or window phase has changed,
preventing late shadow samples from contaminating a new live audit.

Old windows are retained until registry destruction. This is a simple RCU-style
lifetime rule that keeps recorders free of reference counting. Reopen cycles are
expected to be rare; epoch reclamation can replace this policy if profiling
shows material retained memory.

## Durable recovery

`checkpoint()` captures an immutable registry image under the control-plane
mutex while using the auditor's existing conservative snapshot operation.
It retains exact raw mapping generations, artifact states, policies, audit
aggregates, mapping versions, and restoration handles. Restore validates the
whole image before publishing any runtime pointer or mapping to callers.

`ClassIIPersistence` writes those images through atomic snapshot replacement
and a SHA-256-chained append log. Event frames are complete post-transaction
images, making replay independent of transient `activating` steps. A torn final
frame is ignored; a complete corrupt or discontinuous frame is rejected. See
[Class II snapshot and event-log persistence](class-ii-persistence.md).

## Current benchmark

The Release microbenchmark uses 65,536 randomized dense sources and 20 million
iterations on the provisioned x64 machine. One observed run produced:

```text
mapping_lock_free=true  ns_per_lookup=1.7–2.0
active_resolve          ns_per_resolve=3.3–3.6
```

These are cache-resident microbenchmark results, not inference latency claims.
Run `scripts/benchmark-mapping.ps1` to measure the current machine and build.

## Present constraints

- Capacity is fixed when the registry is constructed; resizing requires a new
  registry and controlled republish.
- One dense source maps to at most one active artifact slot.
- Parent-to-child replacement requires the same source IDs, source kinds, input
  shape, and port semantic; slot positions and mapping versions may change.
- Metadata and lifecycle writes are serialized; only mapping reads and audit
  observations are designed as hot concurrent operations.
- The registry retains, but does not itself execute, the restoration snapshot.
- Artifact kernel payloads and restoration snapshots live in content-addressed
  stores; registry persistence retains and validates their stable references.
- Destruction requires all external reader and recorder threads to have stopped.
