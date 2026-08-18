"""Fail when the release version or Apache licensing metadata drifts."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def release_version() -> str:
    source = (ROOT / "python/prolog_tsetlin/_version.py").read_text(
        encoding="utf-8"
    )
    match = re.fullmatch(
        r'"""[^\n]+"""\s+__version__ = "([0-9]+\.[0-9]+\.[0-9]+)"\s*',
        source,
    )
    if match is None:
        raise SystemExit("_version.py must contain one stable SemVer assignment")
    return match.group(1)


def main() -> int:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if 'license = "Apache-2.0"' not in pyproject:
        raise SystemExit("pyproject.toml must declare the Apache-2.0 SPDX license")
    if 'dynamic = ["version"]' not in pyproject:
        raise SystemExit("the package version must remain dynamically sourced")

    expected_attribute = "prolog_tsetlin._version.__version__"
    if f'version = {{attr = "{expected_attribute}"}}' not in pyproject:
        raise SystemExit(f"package version must use {expected_attribute}")

    version = release_version()
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    if "python/prolog_tsetlin/_version.py" not in cmake:
        raise SystemExit("CMake does not read the authoritative Python version")

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0, January 2004" not in license_text:
        raise SystemExit("LICENSE is not the Apache License 2.0 text")
    print(f"release metadata: {version}, Apache-2.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
