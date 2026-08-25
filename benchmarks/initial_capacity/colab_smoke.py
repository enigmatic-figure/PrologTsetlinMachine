#!/usr/bin/env python3
"""Remote driver for the allowlisted PTM campaign CPU/T4 plumbing smoke."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import traceback


CONTENT = Path("/content")
ARCHIVE = CONTENT / "ptm-campaign-input.tar.gz"
PROJECT = CONTENT / "ptm"
RESULTS = CONTENT / "ptm-campaign-results"
BUNDLE = CONTENT / "ptm-campaign-results.tar.gz"


def run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    stdout_copy: Path | None = None,
) -> None:
    rendered = " ".join(command)
    print(f"+ {rendered}", flush=True)
    completed = subprocess.run(
        command,
        check=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with (RESULTS / "driver.log").open("a", encoding="utf-8") as stream:
        stream.write(f"+ {rendered}\n")
        stream.write(completed.stdout)
        if completed.stdout and not completed.stdout.endswith("\n"):
            stream.write("\n")
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if stdout_copy is not None:
        stdout_copy.write_text(completed.stdout, encoding="utf-8")
    completed.check_returncode()


def first_line(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.splitlines()[0] if completed.stdout else "n/a"


def environment_receipt() -> dict[str, object]:
    gpu_name = "n/a"
    if shutil.which("nvidia-smi"):
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if names:
            gpu_name = names
    cpu_name = "n/a"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_name = line.split(":", 1)[1].strip()
                break
    archive = ARCHIVE.read_bytes() if ARCHIVE.is_file() else b""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu_name,
        "gpu": gpu_name,
        "cmake": first_line(["cmake", "--version"]),
        "compiler": first_line(["c++", "--version"]),
        "nvcc": first_line(["nvcc", "--version"])
        if shutil.which("nvcc")
        else "n/a",
        "input_archive_bytes": len(archive) if archive else "n/a",
        "input_archive_digest": (
            "sha256:" + hashlib.sha256(archive).hexdigest() if archive else "n/a"
        ),
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    status = "failed"
    failure = None
    try:
        if not ARCHIVE.is_file():
            raise FileNotFoundError(ARCHIVE)
        PROJECT.mkdir(parents=True, exist_ok=True)
        run(["tar", "-xzf", str(ARCHIVE), "-C", str(PROJECT)])
        run([sys.executable, "-m", "pip", "install", "uv"])
        run(
            [
                "bash",
                str(PROJECT / "scripts" / "bootstrap-benchmark-incumbents.sh"),
                str(CONTENT / "incumbents"),
            ]
        )
        generator = "Ninja" if shutil.which("ninja") else "Unix Makefiles"
        cuda_enabled = bool(shutil.which("nvidia-smi") and shutil.which("nvcc"))
        build = CONTENT / (
            f"ptm-native-{'cuda' if cuda_enabled else 'cpu'}-"
            + ("ninja" if generator == "Ninja" else "make")
        )
        run(
            [
                "cmake",
                "-S",
                str(PROJECT),
                "-B",
                str(build),
                "-G",
                generator,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DPTM_BUILD_TESTS=OFF",
                "-DPTM_BUILD_EXAMPLES=OFF",
                "-DPTM_BUILD_RUNTIME_CLI=OFF",
                "-DPTM_BUILD_BENCHMARKS=ON",
                f"-DPTM_ENABLE_CUDA={'ON' if cuda_enabled else 'OFF'}",
            ]
        )
        run(
            [
                "cmake",
                "--build",
                str(build),
                "--target",
                "ptm_campaign_native_runner",
                "-j2",
            ]
        )
        if cuda_enabled:
            run(
                [
                    "cmake",
                    "--build",
                    str(build),
                    "--target",
                    "ptm_packed_tm_benchmark",
                    "-j2",
                ]
            )
            run(
                [
                    str(build / "ptm_packed_tm_benchmark"),
                    "--clauses",
                    "64",
                    "--features",
                    "256",
                    "--densities",
                    "0.02",
                    "--resident-pages",
                    "16",
                    "--backend",
                    "cuda_warp_tile_fused_vote",
                    "--repeats",
                    "5",
                    "--warmup",
                    "2",
                    "--samples",
                    "2",
                    "--seed",
                    "20260825",
                    "--jsonl",
                ],
                stdout_copy=RESULTS / "gpu-packed-smoke.jsonl",
            )
        process_environment = dict(os.environ)
        process_environment["PYTHONPATH"] = str(PROJECT / "python")
        run(
            [
                sys.executable,
                str(PROJECT / "benchmarks" / "initial_capacity" / "run_smoke.py"),
                "--project-root",
                str(PROJECT),
                "--material-root",
                str(PROJECT / "out" / "benchmark-campaign" / "materials"),
                "--incumbent-root",
                str(CONTENT / "incumbents"),
                "--ptm-native-executable",
                str(build / "ptm_campaign_native_runner"),
                "--output",
                str(RESULTS / "smoke"),
            ],
            env=process_environment,
        )
        receipts = CONTENT / "incumbents" / "receipts"
        if receipts.is_dir():
            shutil.copytree(
                receipts,
                RESULTS / "incumbent-receipts",
                dirs_exist_ok=True,
            )
        status = "ok"
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        receipt = environment_receipt() | {
            "schema": "ptm.colab-plumbing-smoke.v1",
            "status": status,
            "failure": failure,
        }
        (RESULTS / "environment.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with tarfile.open(BUNDLE, "w:gz") as archive:
            archive.add(RESULTS, arcname=RESULTS.name)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
