"""Run the complete repository documentation correctness boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main() -> int:
    run("scripts/render_help_reference.py", "--check")
    run("scripts/render_cli_reference.py", "--check")
    run("scripts/check_python_api.py")
    run("scripts/check_docs.py")
    run("scripts/check_markdown_links.py")
    run("-m", "sphinx", "-W", "--keep-going", "-b", "html", "-c", "docs", ".", "out/docs")
    run("-m", "sphinx", "-W", "--keep-going", "-b", "man", "-c", "docs", ".", "out/man")
    print("documentation checks: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
