# Contributing

## Development setup

Use an isolated Python environment and keep build outputs under `out/`:

```bash
python -m venv .venv
python -m pip install -e ".[tui,data,test]"
python -m pytest tests/python

cmake -S . -B out/build -DPTM_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build out/build --config Release
ctest --test-dir out/build -C Release --output-on-failure
```

GNU Prolog tests run when `gprolog` is on `PATH` or `PTM_GPROLOG` identifies the
executable. CUDA tests require an explicitly enabled CUDA build.

The PowerShell and WSL verification scripts reproduce the maintainers' full
cross-layer gates, but they are conveniences rather than installation APIs.
Contributions should not add assumptions about a specific username, drive,
WSL distribution name, compiler installation directory, or GNU Prolog path.

## Change expectations

- Preserve stable feature/literal identities and explicit bit-plane semantics.
- Keep Python as the reference oracle for optimized native paths.
- Bound symbolic search before launching GNU Prolog and revalidate results in
  Python before lowering them.
- Version persistent, ABI, artifact, JSONL, and preprocessing contract changes.
- Add or update clean-consumer coverage for packaging changes.
- Update `UPGRADING.md` whenever users must rebuild, migrate, or change config.
- Keep papers and large external datasets out of Git; add durable links to
  `docs/references.md` instead.

Run `git diff --check` before submitting a change. Do not commit `.venv`, build
trees, generated artifacts, local benchmark logs, or downloaded research PDFs.
