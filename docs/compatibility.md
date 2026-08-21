# Compatibility matrix

This page distinguishes declared support from configurations exercised by the
repository's verification scripts. A tested row is evidence, not an exclusive
platform requirement.

## User-facing components

| Component | Supported boundary | Verified configurations |
| --- | --- | --- |
| Python core | CPython 3.10+ | CPython 3.10.12 on Linux; 3.14.6 on Windows |
| Terminal workbench | `textual>=8.2,<9` | Textual 8.x on Windows; headless test driver |
| C/C++ core | CMake 3.20+, C++20 compiler | GCC 11.4/CMake 3.22 on Ubuntu 22.04; MSVC 19.51/CMake 4.3 on Windows x64 |
| GNU Prolog bridge | GNU Prolog 1.4.5 or 1.5.x; optional | GNU Prolog 1.4.5 on Linux and 1.5-era Windows toolchain |
| Data adapters | PyArrow 18-25 and Pillow 12.x; optional `data` extra | PyArrow 25.0/Pillow 12.3 on Windows verification and Linux CI |
| Static runtime | Scalar CPU required; SIMD optional | x86-64 scalar, AVX2, and capability-gated AVX-512 |
| CUDA | Experimental, source build only | NVIDIA CUDA/WSL research configuration documented separately |

Windows and Linux are first-class tested hosts. WSL is useful for reproducing
one Linux/CUDA environment but is not required. macOS and non-x86 processors
may build through the portable scalar path, but are not currently release-gated
and should be treated as community-tested.

## Versioned data and ABI contracts

| Contract | Current version | Compatibility behavior |
| --- | ---: | --- |
| Native training C ABI (`ptm/c_api.h`) | 2 | Exact match required |
| Static runtime C ABI (`ptm/runtime.h`) | 2 | Exact match required; recompile host on change |
| `.ptm` container | 1 | Unknown container versions rejected |
| Packed binary TM payload | 1 | Unknown payload versions rejected |
| Fixed Logic payload | 1 | Unknown payload versions rejected |
| Masked-threshold payload | 1 | Unknown payload versions rejected |
| Raw preprocessing | `ptm.preprocessing.v1` | Unknown transforms/schema rejected; precomputed input remains supported |
| Scalar TM snapshot | 1 | Exact schema required for restoration |
| PA/Class II artifact | 1 | Exact schema and content digest required |
| Logic AST/program/morphology | 1 | Exact schema required |

Artifact compatibility is checked at load time. Metadata is not permission to
approximate an unsupported execution contract: loaders fail closed instead.

## Packaging support levels

- **Supported:** source checkout Python installs, locally built wheels, CMake
  build/install prefixes, and CMake `find_package(PTM)` consumers.
- **Not published yet:** PyPI distributions, OS packages, container images,
  signed binary installers, and stable shared-library packages.
- **Research-only:** CUDA dispatch, benchmark-specific scripts, and internal
  persistence experiments.

See [Installation](manual/how-to/install.md) for commands and [Upgrading](../UPGRADING.md)
for migration rules.
