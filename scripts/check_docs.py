"""Validate PTM documentation classification and publication metadata."""

from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs" / "_meta" / "inventory.csv"
IGNORED_PARTS = {
    ".agents",
    ".codex",
    ".git",
    ".hypothesis",
    ".pytest_cache",
    ".venv",
    "build",
    "dist",
    "out",
}
FIELDS = (
    "path",
    "domain",
    "type",
    "state",
    "authority",
    "published",
    "destination",
    "action",
)
DOMAINS = {
    "architecture",
    "archive",
    "benchmarks",
    "developer",
    "manual",
    "meta",
    "operations",
    "releases",
    "rfcs",
}
TYPES = {
    "adr",
    "benchmark-record",
    "bibliography",
    "contract",
    "how-to",
    "inventory",
    "landing",
    "operations",
    "plan",
    "policy",
    "record",
    "reference",
    "release-notes",
    "rfc",
    "status",
    "tutorial",
}
STATES = {"current", "historical", "internal", "mixed", "proposed"}
AUTHORITIES = {"authoritative", "derived", "record", "transitional"}
ACTIONS = {"archive", "exclude", "move-compatible", "retain", "split"}


def markdown_files() -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.md")
        if not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
    }


def main() -> int:
    failures: list[str] = []
    with INVENTORY.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != FIELDS:
            failures.append(
                f"{INVENTORY.relative_to(ROOT)}: expected columns {', '.join(FIELDS)}"
            )
        rows = list(reader)

    classified: set[str] = set()
    previous_path = ""
    for line, row in enumerate(rows, start=2):
        prefix = f"{INVENTORY.relative_to(ROOT)}:{line}"
        path_text = row.get("path", "")
        normalized = PurePosixPath(path_text)
        if (
            not path_text
            or normalized.is_absolute()
            or ".." in normalized.parts
            or normalized.as_posix() != path_text
        ):
            failures.append(f"{prefix}: path must be normalized repository-relative POSIX")
        if path_text in classified:
            failures.append(f"{prefix}: duplicate path {path_text}")
        classified.add(path_text)
        if previous_path and path_text <= previous_path:
            failures.append(f"{prefix}: inventory paths must be strictly sorted")
        previous_path = path_text
        if row.get("domain") not in DOMAINS:
            failures.append(f"{prefix}: invalid domain {row.get('domain')!r}")
        if row.get("type") not in TYPES:
            failures.append(f"{prefix}: invalid type {row.get('type')!r}")
        if row.get("state") not in STATES:
            failures.append(f"{prefix}: invalid state {row.get('state')!r}")
        if row.get("authority") not in AUTHORITIES:
            failures.append(f"{prefix}: invalid authority {row.get('authority')!r}")
        if row.get("published") not in {"yes", "no"}:
            failures.append(f"{prefix}: published must be yes or no")
        if row.get("action") not in ACTIONS:
            failures.append(f"{prefix}: invalid action {row.get('action')!r}")
        if not row.get("destination"):
            failures.append(f"{prefix}: destination is required")
        if row.get("authority") == "transitional" and row.get("state") != "mixed":
            failures.append(f"{prefix}: transitional authority requires mixed state")
        if row.get("state") == "historical" and row.get("domain") != "archive":
            failures.append(f"{prefix}: historical pages belong in the archive domain")
        if row.get("published") == "no" and row.get("state") != "internal":
            failures.append(f"{prefix}: unpublished pages must be internal")

    discovered = markdown_files()
    for path_text in sorted(discovered - classified):
        failures.append(f"unclassified Markdown file: {path_text}")
    for path_text in sorted(classified - discovered):
        failures.append(f"inventory entry does not exist: {path_text}")

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"documentation inventory: {len(classified)} files classified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
