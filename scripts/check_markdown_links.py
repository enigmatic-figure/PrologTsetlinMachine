"""Validate repository-local links in Markdown files."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
IGNORED_PARTS = {".git", ".pytest_cache", ".venv", "build", "out"}


def markdown_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
    )


def local_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().strip("<>")
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None
    path_text = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not path_text:
        return None
    if path_text.startswith("/"):
        return ROOT / path_text.lstrip("/")
    return source.parent / path_text


def main() -> int:
    failures: list[str] = []
    files = markdown_files()
    for source in files:
        text = source.read_text(encoding="utf-8")
        for match in LINK.finditer(text):
            target = local_target(source, match.group(1))
            if target is not None and not target.exists():
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{source.relative_to(ROOT)}:{line}: missing {match.group(1)}"
                )
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"markdown links: {len(files)} files passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
