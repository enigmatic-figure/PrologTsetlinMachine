#!/usr/bin/env python3
"""Create the deterministic, explicit Colab plumbing-smoke input archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile


ROOT_FILES = ("CMakeLists.txt", "LICENSE")
ROOT_DIRECTORIES = (
    "cmake",
    "include",
    "src",
    "python/prolog_tsetlin",
    "prolog",
    "benchmarks",
)
SINGLE_FILES = ("scripts/bootstrap-benchmark-incumbents.sh",)
MATERIAL_DIRECTORY = "out/benchmark-campaign/materials/parity-ladder/n-06"


def _files(project: Path) -> tuple[Path, ...]:
    selected = [project / name for name in (*ROOT_FILES, *SINGLE_FILES)]
    for name in (*ROOT_DIRECTORIES, MATERIAL_DIRECTORY):
        root = project / name
        selected.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    missing = [str(path) for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError("allowlisted input is absent: " + ", ".join(missing))
    return tuple(sorted(set(selected), key=lambda path: path.as_posix()))


def package(project_root: Path, output: Path) -> dict[str, object]:
    project = project_root.resolve()
    selected = _files(project)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in selected:
                    relative = path.relative_to(project).as_posix()
                    data = path.read_bytes()
                    information = tarfile.TarInfo(relative)
                    information.size = len(data)
                    information.mode = 0o755 if path.suffix in (".py", ".sh") else 0o644
                    information.mtime = 0
                    information.uid = 0
                    information.gid = 0
                    information.uname = ""
                    information.gname = ""
                    archive.addfile(information, BytesIO(data))
    payload = output.read_bytes()
    return {
        "schema": "ptm.colab-smoke-package.v1",
        "path": str(output),
        "files": len(selected),
        "bytes": len(payload),
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(package(arguments.project_root, arguments.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
