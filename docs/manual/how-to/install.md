# Installing PTM

PTM can be installed in layers. Most people should start with the Python
workbench and add the native runtime or GNU Prolog only when needed.

## Supported installation profiles

| Profile | Provides | Required tools |
| --- | --- | --- |
| Python core | Reference TM, representation, artifact APIs, `ptm` CLI | Python 3.10+ |
| Terminal workbench | Interactive XOR workflow and telemetry | Python 3.10+, `tui` extra |
| Data adapters | Arrow/Parquet streams and image files | Python 3.10+, `data` extra |
| Native runtime | C/C++ libraries and `ptmrt` CLI | CMake 3.20+, C++20 compiler |
| Symbolic search | Bounded threshold and structure search | Python profile, GNU Prolog |
| CUDA research | Experimental GPU kernels and benchmarks | Native profile, compatible CUDA toolkit/GPU |

The exact tested versions and platform status are in the
[compatibility matrix](../../compatibility.md).

## Get the source

There is not yet a published PyPI package or binary installer. Install from a
Git checkout or a release source archive:

```bash
git clone https://github.com/enigmatic-figure/PrologTsetlinMachine.git
cd PrologTsetlinMachine
```

Until tagged releases begin, pin deployments to a commit hash rather than
tracking `main` implicitly.

## Python core or terminal workbench

Create an isolated environment. Python's `venv` environments are disposable;
do not copy one between machines.

```bash
python -m venv .venv
```

Activate it:

```text
Windows PowerShell:  .venv\Scripts\Activate.ps1
Windows cmd.exe:     .venv\Scripts\activate.bat
Linux/macOS:         source .venv/bin/activate
```

Install one profile:

```bash
# Dependency-free core
python -m pip install .

# Or the terminal workbench
python -m pip install ".[tui]"

# Arrow/Parquet and Pillow image adapters
python -m pip install ".[data]"

# Contributor/test environment
python -m pip install ".[tui,data,test]"
```

Confirm the command is available:

```bash
ptm --help
ptm tui --demo xor        # when the tui extra is installed
```

Editable installs (`python -m pip install -e ".[tui,data,test]"`) are intended for
contributors. Consumers should use the regular install shown above.

## Optional GNU Prolog

GNU Prolog is required only for bounded symbolic searches. Download a binary or
source distribution from the [GNU Prolog project](https://www.gprolog.org/),
then make `gprolog` available on `PATH`:

```bash
gprolog --consult-file /dev/null --query-goal halt
```

On platforms where that command is not on `PATH`, set `PTM_GPROLOG` to the
executable's absolute path. PTM does not assume a Windows installation folder
or a particular Linux distribution. The Python wheel includes PTM's two
versioned `.pl` search templates; GNU Prolog itself remains a system package.

From a source checkout, exercise the integration with:

```bash
python examples/prolog_threshold.py
python examples/prolog_structures.py
```

## Native runtime and C/C++ SDK

Configure and build with any supported C++20 toolchain:

```bash
cmake -S . -B out/build -DPTM_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build out/build --config Release
ctest --test-dir out/build -C Release --output-on-failure
```

Install into an isolated prefix that is easy to replace during upgrades:

```bash
cmake --install out/build --config Release --prefix out/install
```

The install tree contains:

- `bin/ptmrt` (or `ptmrt.exe`);
- the `ptm` shared C ABI library and `ptmrt` static runtime library;
- public headers under `include/ptm`;
- a relocatable `PTMConfig.cmake` exporting `PTM::core`, `PTM::c_api`, and
  `PTM::runtime`.

Continue with [Embed the native runtime](embed-ptmrt.md) to select an exported
CMake target and validate a clean consumer. The official CMake documentation
also explains the [`cmake --install` workflow](https://cmake.org/cmake/help/latest/guide/tutorial/Installation%20Commands%20and%20Concepts.html).

## Optional CUDA research profile

CUDA is not required for artifact export or scalar inference. Enable it only on
a machine with a supported NVIDIA driver, toolkit, and GPU:

```bash
cmake -S . -B out/cuda -DPTM_ENABLE_CUDA=ON -DPTM_BUILD_TESTS=ON
cmake --build out/cuda --config Release
ctest --test-dir out/cuda -C Release --output-on-failure
```

See [CUDA packed-TM execution](../../cuda-packed-tm.md) for the experimental
backend contract. WSL is one tested CUDA host, not an installation requirement.

## Offline installation

On a connected machine, build a wheel without downloading runtime dependencies:

```bash
python -m pip wheel . --no-deps --wheel-dir out/wheels
```

For the TUI, download its pinned dependency range at the same time:

```bash
python -m pip download ".[tui]" --dest out/wheels
```

Copy `out/wheels` to the offline machine and install from that directory:

```bash
python -m pip install --no-index --find-links out/wheels prolog-tsetlin-machine
```

Native consumers can copy the complete prefix produced by `cmake --install` to
an equivalent platform/architecture, or build from the source archive locally.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `ptm` is not found | Activate the virtual environment or run `.venv/bin/ptm` / `.venv\Scripts\ptm.exe`. |
| TUI extra is missing | Run `python -m pip install ".[tui]"` in the active environment. |
| GNU Prolog is not found | Put `gprolog` on `PATH` or set `PTM_GPROLOG`. |
| Native Python ABI mismatch | Rebuild the native library from the same checkout; core ABI is currently v2. |
| `ptmrt` rejects an artifact version | Use a runtime supporting the artifact's declared container/payload version; see [Upgrading](../../../UPGRADING.md). |
| CMake cannot find PTM | Set `CMAKE_PREFIX_PATH` to the install prefix, not the source or build directory. |
