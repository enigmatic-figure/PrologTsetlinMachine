"""Validate that the declared Python public API has documented owners."""

from __future__ import annotations

import ast
import importlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "python"
API_PAGE = ROOT / "docs" / "manual" / "reference" / "python-api.md"
AUTOMODULE = re.compile(r"^\.\. automodule::\s+([\w.]+)$", re.MULTILINE)


def declared_public_names() -> list[str]:
    tree = ast.parse((PACKAGE_ROOT / "prolog_tsetlin" / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            return [element.value for element in node.value.elts if isinstance(element, ast.Constant)]
    raise SystemExit("python/prolog_tsetlin/__init__.py does not declare __all__")


def main() -> int:
    sys.path.insert(0, str(PACKAGE_ROOT))
    package = importlib.import_module("prolog_tsetlin")
    documented_modules = set(AUTOMODULE.findall(API_PAGE.read_text(encoding="utf-8")))
    failures: list[str] = []
    for name in declared_public_names():
        if not hasattr(package, name):
            failures.append(f"declared public symbol is not importable: {name}")
            continue
        value = getattr(package, name)
        owner = getattr(value, "__module__", "prolog_tsetlin")
        if owner not in documented_modules and owner not in {"prolog_tsetlin", "typing"} and not any(
            owner.startswith(f"{module}.") for module in documented_modules
        ):
            failures.append(f"public symbol owner is not documented: {name} ({owner})")
        if not isinstance(value, type) and not (getattr(value, "__doc__", "") or "").strip():
            failures.append(f"public symbol has no docstring: {name} ({owner})")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Python API: {len(declared_public_names())} public symbols have documented owners")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
