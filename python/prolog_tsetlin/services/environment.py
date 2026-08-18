"""Read-only environment preflight for interactive PTM frontends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import sys

from ..native import find_native_library


@dataclass(frozen=True, slots=True)
class Capability:
    component: str
    status: str
    detail: str


def _first_existing(paths: tuple[Path, ...]) -> Path | None:
    return next((path.resolve() for path in paths if path.is_file()), None)


def inspect_environment(workspace: Path) -> tuple[Capability, ...]:
    """Report optional tools without importing or starting native workloads."""

    root = workspace.resolve()
    runtime = shutil.which("ptmrt")
    if runtime is None:
        candidate = _first_existing(
            (
                root / "build" / "ptmrt.exe",
                root / "build" / "ptmrt",
                root / "build" / "Release" / "ptmrt.exe",
            )
        )
        runtime = str(candidate) if candidate is not None else None
    prolog = os.environ.get("PTM_GPROLOG") or shutil.which("gprolog")
    native = find_native_library()
    writable = os.access(root, os.W_OK)
    return (
        Capability("Python", "READY", sys.version.split()[0]),
        Capability("Scalar oracle", "READY", "deterministic reference backend"),
        Capability(
            "Native runtime",
            "READY" if native is not None else "OPTIONAL",
            str(native) if native is not None else "build libptm to enable native paths",
        ),
        Capability(
            "ptmrt",
            "READY" if runtime is not None else "OPTIONAL",
            runtime or "build the C++ runtime for independent verification",
        ),
        Capability(
            "GNU Prolog",
            "READY" if prolog else "OPTIONAL",
            prolog or "set PTM_GPROLOG to enable bounded symbolic search",
        ),
        Capability(
            "Workspace",
            "READY" if writable else "READ ONLY",
            str(root),
        ),
    )
