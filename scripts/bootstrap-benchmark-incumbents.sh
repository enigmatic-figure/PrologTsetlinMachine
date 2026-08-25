#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "usage: $0 [OUTPUT_ROOT]" >&2
    exit 2
fi

output_root="${1:-out/benchmark-campaign/incumbents-linux}"
python_selector="${PTM_BENCHMARK_PYTHON:-3.12}"
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
constraint_path="${script_directory}/../benchmarks/initial_capacity/incumbent-constraints.txt"
pytm_commit="d6c1cf0e4aaa4a8ae2f2818ba27878fb89d31dc5"
tmu_commit="5605ff070a18549328028c907a9acf68e063346e"

mkdir -p "${output_root}/sources" "${output_root}/envs" "${output_root}/receipts"

prepare_source() {
    local name="$1"
    local repository="$2"
    local commit="$3"
    local source_path="${output_root}/sources/${name}"
    if [[ ! -d "${source_path}/.git" ]]; then
        git clone --filter=blob:none --no-checkout "${repository}" "${source_path}"
    fi
    git -C "${source_path}" fetch --depth 1 origin "${commit}"
    git -C "${source_path}" checkout --detach "${commit}"
    test "$(git -C "${source_path}" rev-parse HEAD)" = "${commit}"
}

prepare_environment() {
    local name="$1"
    local source_path="${output_root}/sources/${name}"
    local env_path="${output_root}/envs/${name}"
    if command -v uv >/dev/null 2>&1; then
        uv venv \
            --seed \
            --allow-existing \
            --link-mode copy \
            --python "${python_selector}" \
            "${env_path}"
    else
        python_bin="$(command -v "python${python_selector}" || true)"
        if [[ -z "${python_bin}" ]]; then
            echo "Python ${python_selector} is unavailable and uv is not installed" >&2
            exit 2
        fi
        "${python_bin}" -m venv "${env_path}"
    fi
    test "$("${env_path}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = "${python_selector}"
    "${env_path}/bin/python" -m pip install \
        --constraint "${constraint_path}" \
        --upgrade pip setuptools wheel
    "${env_path}/bin/python" -m pip install \
        --constraint "${constraint_path}" \
        "numpy==1.26.4"
    "${env_path}/bin/python" -m pip install \
        --constraint "${constraint_path}" \
        "${source_path}"
    "${env_path}/bin/python" -m pip freeze --all > "${output_root}/receipts/${name}-pip-freeze.txt"
    "${env_path}/bin/python" -VV > "${output_root}/receipts/${name}-python.txt"
}

prepare_source \
    "pytsetlinmachine" \
    "https://github.com/cair/pyTsetlinMachine.git" \
    "${pytm_commit}"
prepare_source \
    "tmu" \
    "https://github.com/cair/tmu.git" \
    "${tmu_commit}"
prepare_environment "pytsetlinmachine"
prepare_environment "tmu"
