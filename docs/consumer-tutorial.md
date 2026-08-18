# Clean consumer tutorial

This tutorial starts from a fresh checkout, trains and exports a raw-record XOR
artifact, installs the standalone runtime, and consumes its CMake package. It
does not rely on a provisioned workstation or WSL distribution name.

## 1. Create an isolated Python install

```bash
git clone https://github.com/enigmatic-figure/PrologTsetlinMachine.git
cd PrologTsetlinMachine
python -m venv .venv
```

Activate `.venv`, then install the regular (non-editable) package:

```bash
python -m pip install .
ptm --help
```

## 2. Export and consume a raw-record model in Python

```bash
python examples/export_raw_xor_artifact.py out/consumer/raw-xor.ptm
```

Create `out/consumer/predict.py`:

```python
from prolog_tsetlin import load_model_artifact

model = load_model_artifact("out/consumer/raw-xor.ptm")
records = (
    {"left": False, "right": False},
    {"left": False, "right": True},
    {"left": True, "right": False},
    {"left": True, "right": True},
)
print(model.predict_records(records))
```

Run it from the repository root:

```bash
python out/consumer/predict.py
```

Expected output:

```text
(0, 1, 1, 0)
```

## 3. Build and install the standalone runtime

```bash
cmake -S . -B out/consumer-build -DPTM_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build out/consumer-build --config Release
cmake --install out/consumer-build --config Release \
  --prefix out/consumer-install
```

Run the installed executable (append `.exe` on Windows):

```bash
out/consumer-install/bin/ptmrt verify out/consumer/raw-xor.ptm
out/consumer-install/bin/ptmrt run-record out/consumer/raw-xor.ptm \
  left:bool=false right:bool=true
```

The second command reports materialized features `[0,1]` and prediction `1`.
Python and the independent runtime have now agreed on preprocessing and model
execution.

## 4. Consume the installed CMake package

The repository includes a minimal out-of-tree consumer:

```bash
cmake -S tests/consumer -B out/consumer-sdk \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/PrologTsetlinMachine/out/consumer-install
cmake --build out/consumer-sdk --config Release
```

Run `out/consumer-sdk/ptm_consumer_smoke` on a single-config build or
`out\consumer-sdk\ptm_consumer_smoke.exe` on Windows. It includes
`ptm/runtime.h`, links `PTM::runtime`, and checks runtime ABI v2.

A real consumer CMake project needs only:

```cmake
find_package(PTM 0.1 CONFIG REQUIRED)
target_link_libraries(your_program PRIVATE PTM::runtime)
```

Use `PTM::c_api` for the adaptive/training C ABI or `PTM::core` for the native
C++ API.

## 5. Add optional capabilities

- Install `.[tui]` and run `ptm tui --demo xor` for the workbench.
- Install GNU Prolog and put `gprolog` on `PATH` for bounded symbolic search.
- Enable `PTM_ENABLE_CUDA=ON` only for the experimental CUDA profile.

None of those capabilities changes the content identity of the exported model.
