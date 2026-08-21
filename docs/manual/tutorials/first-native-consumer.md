# First native consumer

This tutorial takes the raw-record XOR artifact from the
[first Python model](first-python-model.md), runs it with the standalone
runtime, and links it into the repository's minimal out-of-tree CMake
consumer.

## Install the runtime

Follow [Embed the native runtime](../how-to/embed-ptmrt.md) through the
installation step, using `out/consumer-install` as the prefix. If necessary,
create the example artifact first:

```bash
python examples/export_raw_xor_artifact.py out/consumer/raw-xor.ptm
```

## Verify and run the artifact

On a single-config build, run:

```bash
out/consumer-install/bin/ptmrt verify out/consumer/raw-xor.ptm
out/consumer-install/bin/ptmrt run-record out/consumer/raw-xor.ptm \
  left:bool=false right:bool=true
```

Append `.exe` to `ptmrt` on Windows. The second command reports materialized
features `[0,1]` and prediction `1`. Python and the independent runtime have
now agreed on preprocessing and inference.

## Build the clean CMake consumer

Use the installed prefix rather than the source or build directory:

```bash
cmake -S tests/consumer -B out/consumer-sdk \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/PrologTsetlinMachine/out/consumer-install
cmake --build out/consumer-sdk --config Release
```

Run `out/consumer-sdk/ptm_consumer_smoke` on a single-config build or
`out\consumer-sdk\Release\ptm_consumer_smoke.exe` on a multi-config Windows
build. The program includes `ptm/runtime.h`, links `PTM::runtime`, checks the
runtime ABI, and evaluates a model.

The tutorial is complete when both the CLI and clean consumer accept the
installed runtime. For integration choices and reusable CMake snippets, see
[Embed the native runtime](../how-to/embed-ptmrt.md).
