"""Resolve the Prolog resources shipped with PTM.

The source checkout is useful during development, while a wheel installs the
same files below ``sys.prefix/share/prolog-tsetlin-machine/prolog``. Keeping
that policy here prevents services from depending on repository layout.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from types import MappingProxyType


PROLOG_MODULES = frozenset(
    {
        "bounded_structure_search.pl",
        "bounded_threshold_search.pl",
        "pta_ontology.pl",
        "pta_input.pl",
        "pta_deescalation.pl",
        "pta_escalation.pl",
    }
)


class PrologResourceError(FileNotFoundError):
    """A required installed or checkout Prolog resource is unavailable."""


def prolog_module_candidates(
    filename: str,
    *,
    prefix: str | os.PathLike[str] | None = None,
) -> tuple[Path, ...]:
    """Return the portable lookup order for a known PTM Prolog module."""

    if type(filename) is not str:
        raise TypeError("filename must be a string")
    if filename not in PROLOG_MODULES:
        raise ValueError(f"unknown PTM Prolog module: {filename!r}")
    install_prefix = Path(sys.prefix if prefix is None else prefix)
    checkout = Path(__file__).resolve().parents[2] / "prolog" / filename
    installed = (
        install_prefix
        / "share"
        / "prolog-tsetlin-machine"
        / "prolog"
        / filename
    )
    return checkout, installed


def prolog_directory_candidates(
    *, prefix: str | os.PathLike[str] | None = None
) -> tuple[Path, ...]:
    """Return coherent checkout and installed Prolog resource roots."""

    install_prefix = Path(sys.prefix if prefix is None else prefix)
    return (
        Path(__file__).resolve().parents[2] / "prolog",
        install_prefix / "share" / "prolog-tsetlin-machine" / "prolog",
    )


def resolve_prolog_module(
    filename: str,
    *,
    prefix: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a packaged PTM Prolog module or fail explicitly."""

    candidates = prolog_module_candidates(filename, prefix=prefix)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise PrologResourceError(
        f"PTM Prolog module {filename!r} was not found; searched: {searched}"
    )


def resolve_prolog_module_set(
    filenames: tuple[str, ...],
    *,
    prefix: str | os.PathLike[str] | None = None,
) -> MappingProxyType[str, Path]:
    """Resolve a set of modules from one root, never a mixed installation."""

    if type(filenames) is not tuple or not filenames:
        raise TypeError("filenames must be a nonempty tuple")
    if len(set(filenames)) != len(filenames):
        raise ValueError("filenames must not contain duplicates")
    for filename in filenames:
        if type(filename) is not str:
            raise TypeError("filenames items must be strings")
        if filename not in PROLOG_MODULES:
            raise ValueError(f"unknown PTM Prolog module: {filename!r}")
    roots = prolog_directory_candidates(prefix=prefix)
    for root in roots:
        resolved = {filename: root / filename for filename in filenames}
        if all(path.is_file() for path in resolved.values()):
            return MappingProxyType(
                {filename: path.resolve() for filename, path in resolved.items()}
            )
    searched = ", ".join(str(root) for root in roots)
    raise PrologResourceError(
        "PTM Prolog modules were not found together in one resource root; "
        f"searched: {searched}"
    )


def resolve_gprolog(
    executable: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve GNU Prolog from an explicit path, ``PTM_GPROLOG``, or PATH."""

    if executable is not None:
        explicit_command = shutil.which(str(executable))
        candidates = (
            Path(executable),
            Path(explicit_command) if explicit_command else None,
        )
    else:
        configured = os.environ.get("PTM_GPROLOG")
        discovered = shutil.which("gprolog")
        candidates = (
            Path(configured) if configured else None,
            Path(discovered) if discovered else None,
        )
    for candidate in candidates:
        if (
            candidate is not None
            and candidate.is_file()
            and os.access(candidate, os.X_OK)
            and (os.name != "nt" or candidate.suffix.lower() in {".exe", ".com"})
        ):
            return candidate.resolve()
    if executable is not None:
        raise PrologResourceError(f"GNU Prolog executable was not found: {executable}")
    raise PrologResourceError("GNU Prolog was not found; set PTM_GPROLOG or update PATH")


def prolog_process_environment() -> dict[str, str]:
    """Return an isolated environment suitable for noninteractive GNU Prolog."""

    environment = dict(os.environ)
    if os.name == "nt":
        # GNU Prolog's Windows linedit build may otherwise create a GUI console.
        environment["LINEDIT"] = "gui=no"
    return environment


__all__ = [
    "PROLOG_MODULES",
    "PrologResourceError",
    "prolog_directory_candidates",
    "prolog_module_candidates",
    "prolog_process_environment",
    "resolve_gprolog",
    "resolve_prolog_module",
    "resolve_prolog_module_set",
]
