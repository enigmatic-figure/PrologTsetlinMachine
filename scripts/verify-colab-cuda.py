#!/usr/bin/env python3
"""Build and verify an uploaded PTM source archive on a Colab CUDA runtime."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tarfile


CONTENT_ROOT = pathlib.Path("/content")
SOURCE_ARCHIVE = CONTENT_ROOT / "ptm-source.tar.gz"
SOURCE_ROOT = CONTENT_ROOT / "ptm-source"
BUILD_ROOT = CONTENT_ROOT / "ptm-build-sm75"


def run(*args: str, cwd: pathlib.Path | None = None) -> None:
    print(f"+ {' '.join(args)}", flush=True)
    result = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    result.check_returncode()


def reset_directory(path: pathlib.Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def main() -> None:
    if not SOURCE_ARCHIVE.is_file():
        raise FileNotFoundError(
            f"upload the working tree archive to {SOURCE_ARCHIVE} before running"
        )

    reset_directory(SOURCE_ROOT)
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)

    with tarfile.open(SOURCE_ARCHIVE, "r:gz") as archive:
        archive.extractall(SOURCE_ROOT, filter="data")

    run(
        "nvidia-smi",
        "--query-gpu=name,compute_cap,pstate,clocks.sm,memory.total",
        "--format=csv,noheader",
    )
    run("nvcc", "--version")

    configure = [
        "cmake",
        "-S",
        str(SOURCE_ROOT),
        "-B",
        str(BUILD_ROOT),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DPTM_ENABLE_CUDA=ON",
        "-DPTM_BUILD_BENCHMARKS=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=75",
    ]
    if shutil.which("ninja"):
        configure.extend(("-G", "Ninja"))
    run(*configure)
    run("cmake", "--build", str(BUILD_ROOT), "--parallel", "2")
    run(
        "ctest",
        "--test-dir",
        str(BUILD_ROOT),
        "-R",
        "packed_tm_cuda",
        "--output-on-failure",
    )

    sanitizer = shutil.which("compute-sanitizer")
    if sanitizer:
        run(
            sanitizer,
            "--tool",
            "memcheck",
            "--error-exitcode",
            "99",
            str(BUILD_ROOT / "ptm_packed_tm_cuda_tests"),
        )
    else:
        print("compute-sanitizer not present; exactness tests still passed", flush=True)

    run(
        str(BUILD_ROOT / "ptm_packed_tm_benchmark"),
        "--clauses",
        "1024",
        "--features",
        "1024",
        "--densities",
        "0.005,0.02,0.5",
        "--resident-pages",
        "256",
        "--backend",
        "cuda_sparse,cuda_sparse_fused_vote,"
        "cuda_warp_tile,cuda_warp_tile_fused_vote,"
        "cuda_dense_bitset,cuda_dense_bitset_fused_vote",
        "--repeats",
        "5",
        "--warmup",
        "3",
        "--samples",
        "3",
    )


if __name__ == "__main__":
    main()
