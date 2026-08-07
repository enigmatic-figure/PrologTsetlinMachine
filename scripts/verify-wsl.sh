#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PTM_WSL_BUILD_DIR:-}" ]]; then
    build_dir="${PTM_WSL_BUILD_DIR}"
else
    build_dir="${project_root}/out/wsl-release"
    cache_file="${build_dir}/CMakeCache.txt"
    if [[ -f "${cache_file}" ]]; then
        cached_home="$(sed -n \
            's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${cache_file}" | head -n1)"
        if [[ -n "${cached_home}" && "${cached_home}" != "${project_root}" ]]; then
            path_tag="$(printf '%s' "${project_root}" | sha256sum | cut -c1-12)"
            build_dir="${project_root}/out/wsl-release-${path_tag}"
        fi
    fi
fi

for tool in cmake ninja g++ ctest gprolog gplc python3; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "Required WSL tool was not found: ${tool}" >&2
        exit 2
    fi
done

cmake -S "${project_root}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPTM_BUILD_TESTS=ON \
    -DPTM_BUILD_BENCHMARKS=OFF
cmake --build "${build_dir}" -j2
ctest --test-dir "${build_dir}" --output-on-failure

gplc -c --no-susp-warn \
    -o "${build_dir}/bounded_threshold_search.o" \
    "${project_root}/prolog/bounded_threshold_search.pl"

export PYTHONPATH="${project_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PTM_GPROLOG="$(command -v gprolog)"
export PTM_NATIVE_LIBRARY="${build_dir}/libptm.so"

cd "${project_root}"
python3 -m unittest discover -s tests/python -v
python3 examples/tabular_xor.py
python3 examples/prolog_threshold.py
