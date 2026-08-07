#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PTM_CUDA_BUILD_DIR:-}" ]]; then
    build_dir="${PTM_CUDA_BUILD_DIR}"
else
    path_tag="$(printf '%s' "${project_root}" | sha256sum | cut -c1-12)"
    build_dir="${project_root}/out/wsl-cuda-${path_tag}"
fi
if [[ -n "${PTM_CUDA_ROOT:-}" ]]; then
    cuda_root="${PTM_CUDA_ROOT}"
else
    cuda_root=""
    for candidate in /usr/local/cuda /usr/local/cuda-*; do
        if [[ -x "${candidate}/bin/nvcc" ]]; then
            cuda_root="${candidate}"
            break
        fi
    done
    cuda_root="${cuda_root:-/usr/local/cuda}"
fi
cuda_architectures="${PTM_CUDA_ARCHITECTURES:-75}"

for tool in cmake ninja g++ ctest; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "Required WSL tool was not found: ${tool}" >&2
        exit 2
    fi
done
if [[ ! -x "${cuda_root}/bin/nvcc" ]]; then
    echo "CUDA compiler was not found: ${cuda_root}/bin/nvcc" >&2
    exit 2
fi
if [[ ! -x "${cuda_root}/bin/compute-sanitizer" ]]; then
    echo "Compute Sanitizer was not found: ${cuda_root}/bin/compute-sanitizer" >&2
    exit 2
fi

cmake -S "${project_root}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPTM_BUILD_TESTS=ON \
    -DPTM_BUILD_BENCHMARKS=ON \
    -DPTM_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="${cuda_root}/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}"
cmake --build "${build_dir}" -j2
ctest --test-dir "${build_dir}" --output-on-failure
"${cuda_root}/bin/compute-sanitizer" \
    --tool memcheck --error-exitcode 99 \
    "${build_dir}/ptm_packed_tm_cuda_tests"
