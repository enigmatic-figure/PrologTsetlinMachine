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

## Documentation changes

The [documentation constitution](docs/_meta/constitution.md) defines factual
ownership, document states, version policy, and the proposal-to-contract
lifecycle. Classify every new Markdown page in
[`docs/_meta/inventory.csv`](docs/_meta/inventory.csv), then run:

```bash
python scripts/check_docs.py
python scripts/check_markdown_links.py
python scripts/render_help_reference.py --check
python -m sphinx -W --keep-going -b html -c docs . out/docs/html
```

Install the documentation toolchain with `python -m pip install -e ".[docs]"`.
Do not copy CLI syntax, API signatures, capability status, or benchmark results
into a second authoritative source.
