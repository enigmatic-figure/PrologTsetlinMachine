# Embed the native runtime

Use this guide to build, install, and consume PTM's standalone inference
runtime from a C or CMake application.

## Build and install

Configure with a C++20 toolchain and install into an isolated prefix:

```bash
cmake -S . -B out/consumer-build -DPTM_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build out/consumer-build --config Release
cmake --install out/consumer-build --config Release \
  --prefix out/consumer-install
```

The prefix contains `ptmrt`, public headers under `include/ptm`, and a
relocatable `PTMConfig.cmake` package.

## Select a CMake target

```cmake
find_package(PTM 0.1 CONFIG REQUIRED)
target_link_libraries(your_program PRIVATE PTM::runtime)
```

Use `PTM::runtime` for immutable artifact inference, `PTM::c_api` for the
adaptive/training C ABI, or `PTM::core` for the native C++ API. Include public
headers as `#include "ptm/..."`.

Configure consumers with the install prefix:

```bash
cmake -S path/to/consumer -B out/consumer \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/out/consumer-install
cmake --build out/consumer --config Release
```

Do not point `CMAKE_PREFIX_PATH` at the source or build tree. The repository's
`tests/consumer` project is a minimal working integration.

## Validate compatibility

Check `ptmrt_abi_version()` against `PTMRT_ABI_VERSION` before using a loaded
model. For the adaptive ABI, check `ptm_abi_version()` against
`PTM_ABI_VERSION`. See the [runtime artifact reference](../reference/artifact-contract.md)
and [C ABI reference](../reference/c-api.md) for the two boundaries.

For a guided end-to-end exercise, continue with the
[first native consumer](../tutorials/first-native-consumer.md).
