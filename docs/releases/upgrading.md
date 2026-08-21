# Upgrading PTM

PTM is currently pre-1.0. Artifact formats are versioned and fail closed, but
Python APIs, native ABIs, and source-level interfaces may still change between
minor releases. Upgrade deliberately rather than updating a live environment
in place.

## Before upgrading

1. Record the current PTM commit or release version.
2. Commit or copy local source changes; `git pull` must not be used as a backup.
3. Preserve `.ptm` artifacts, training snapshots, and Class II persistence
   files outside disposable build/virtual-environment directories.
4. Record `PTM_ABI_VERSION`, `PTMRT_ABI_VERSION`, and artifact schema versions
   used by any native host application.
5. Run the old version's tests or application smoke test to establish a known
   baseline.

## Recommended upgrade procedure

Use a new virtual environment and native install prefix so rollback remains a
directory switch rather than an emergency rebuild.

```bash
git fetch --tags origin
# Select a reviewed tag or commit here.
python -m venv .venv-next
```

Activate `.venv-next`, then install the desired profile:

```bash
python -m pip install --upgrade pip
python -m pip install ".[tui,data,test]"
python -m pytest tests/python
```

Build and install native components side by side:

```bash
cmake -S . -B out/build-next -DPTM_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build out/build-next --config Release
ctest --test-dir out/build-next -C Release --output-on-failure
cmake --install out/build-next --config Release --prefix out/install-next
```

Run the clean-consumer smoke project against `out/install-next`, then verify a
copy of every artifact/persistence type your application depends on. Switch the
application to the new environment/prefix only after those checks pass.

## Compatibility gates

| Boundary | Upgrade rule |
| --- | --- |
| Python package | Recreate the virtual environment when dependency bounds or Python minor versions change. |
| Native training C ABI | `PTM_ABI_VERSION` must exactly match the Python/native caller. Rebuild on mismatch. |
| Static inference C ABI | `PTMRT_ABI_VERSION` must exactly match compiled host headers. Recompile/relink on mismatch. |
| `.ptm` artifact | The loader checks container, model-kind payload, execution contract, and digest versions before inference. Keep the old runtime for unsupported artifacts. |
| Training snapshot | Schema must match exactly for restoration. Export a portable inference artifact before retiring an old trainer. |
| Class II persistence | Replay with the version that wrote the schema before attempting a migration. Never edit the binary/event log manually. |
| Preprocessing | Literal IDs and ordered `ptm.preprocessing.v1` outputs are part of model identity; do not reorder them during an upgrade. |
| Host record pipeline | Persist `ptm.record_pipeline.v1` and token/image adapter descriptors with application configuration; compare their content IDs before switching data paths. |

## Current 0.1 migration notes

- Both native ABI families are version 2. Host programs built against v1 must
  be recompiled against the current headers and libraries.
- `.ptm` container and packed-TM, fixed-Logic, and masked-threshold payloads
  remain version 1.
- Packed-TM artifacts may now embed `ptm.preprocessing.v1`; older precomputed
  artifacts remain valid.
- The TUI is tested with Textual 8.x and intentionally excludes Textual 9 until
  its compatibility is reviewed.
- Arrow/Parquet and image consumers must opt into the `data` extra. Existing
  dependency-free core installations do not acquire PyArrow or Pillow.
- Installed native consumers should use exported CMake targets rather than
  paths inside a build tree.

## Rollback

Keep the previous source revision, virtual environment, and native prefix until
the new release has processed representative inputs successfully. To roll back,
restore the old application configuration and reopen artifacts with the old
runtime. Do not convert or overwrite the only copy of a snapshot or persistence
log during an upgrade.

See the [compatibility matrix](compatibility.md) for tested toolchains and
the [portable runtime contract](../manual/reference/artifact-contract.md) for fail-closed
artifact behavior.
