# Class II snapshot and event-log persistence

Class II lifecycle state is durable without placing file I/O on inference or
audit-recording paths. The persistence service operates on immutable registry
images captured under the registry's control-plane mutex.

## Persisted state

Snapshot schema version 1 contains:

- source, artifact, and audit-window capacities;
- maturity and audit policies;
- every artifact specification and dense artifact handle;
- lifecycle state and current audit phase;
- audit sequence, observation count, and mismatch count;
- every nonzero raw mapping word, including unbound generation counters;
- mapping versions, content-addressed artifact IDs, and restoration handles.

The audit window is reconstructed with the same sequence end, observation
count, and mismatch count. Internal observation ordering is not preserved
because no current audit decision depends on it. Consequently, activation,
rejection, health, and reopen decisions are identical after restore.

Kernel payloads and adaptive-substrate snapshots remain in their respective
content-addressed stores. The registry snapshot persists their stable IDs and
restoration handles; it does not duplicate those potentially large objects.

## Snapshot envelope

`ClassIIPersistence::write_snapshot_atomic` serializes a canonical
little-endian binary payload into a versioned envelope. The complete envelope
is protected by SHA-256. It is written to a sibling temporary file, flushed to
stable storage with `_commit` on Windows or `fsync` on POSIX, and atomically
renamed over the prior checkpoint. POSIX also synchronizes the containing
directory.

SHA-256 detects accidental corruption and unmodified-digest tampering. It does
not authenticate a malicious writer that can replace both content and digest;
deployment requiring that threat model must sign or MAC the files.

## Append-only event log

Each event is a complete post-transaction registry image with:

- persistence schema version;
- monotonically increasing 64-bit sequence;
- previous event digest;
- bounded descriptive event name;
- payload length and immutable registry image;
- SHA-256 digest over the header, name, and payload.

The previous digest forms a hash chain. Full images are intentionally used for
the initial implementation: replay cost is bounded, every event can be
validated independently, and lifecycle operations do not need a second
partially overlapping state machine for delta application. If profiling shows
material write amplification, a later schema can add typed deltas while
retaining periodic full-image events.

`append_event` returns the only valid durable commit token for the new image.
The orchestrator must not report an in-memory lifecycle mutation committed
until that call succeeds. Persistence failure leaves the previous token as the
recovery boundary; the process should stop publishing further control-plane
mutations and recover from that token.

Hot `resolve()` calls and `record_observation()` calls perform no persistence
I/O. An orchestrator chooses audit checkpoint frequency and appends the
resulting control-plane image before making durability claims.

Snapshot and log files have one control-plane writer. Multiple processes must
coordinate above this API; concurrent writers to the same path are outside the
version 1 contract.

## Recovery

Recovery accepts either:

1. an atomic snapshot followed by zero or more events; or
2. a log whose first complete event is sequence 1 with a zero digest anchor.

It validates the snapshot envelope, every complete event digest, the chain and
sequence, and the semantic registry invariants. Replay then restores a new
registry before it is made visible to readers. In particular:

- `activating` is never accepted as a durable state;
- an active artifact must own all of its declared mappings;
- a bound word may point only to an active or reopening artifact;
- every word must match a declared source and slot;
- unbound words may retain only their generation;
- artifact and source handles remain dense, bounded, and unambiguous.

An incomplete final frame is classified as a torn tail and ignored. Complete
frames with invalid hashes or discontinuous ancestry are rejected rather than
silently skipped. The next append atomically removes a known torn tail before
writing its new frame.

## Compaction and crash points

Compaction first installs the latest image as the atomic snapshot and then
atomically replaces the event log with an empty file. This ordering makes every
crash point recoverable:

| Crash point | Recovery result |
| --- | --- |
| Before snapshot replacement | Old snapshot plus complete log |
| After snapshot replacement, before log truncation | New snapshot; old log prefix is recognized as already applied |
| After log truncation | New snapshot alone |
| During a later event append | New snapshot plus complete events; torn tail ignored |

Run the end-to-end example with:

```powershell
.\scripts\run-class-ii-persistence.ps1
```

The native persistence test additionally exercises log-only recovery,
corruption rejection, malformed-state rejection, exact mapping-generation and
audit recovery, torn-tail repair, and compaction.
