"""Render or validate the generated shared-help manual page."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "python"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prolog_tsetlin.help_topics import render_manual_reference  # noqa: E402


OUTPUT = ROOT / "docs" / "manual" / "reference" / "help-topics.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of updating when the generated page is stale",
    )
    arguments = parser.parse_args()
    expected = render_manual_reference()
    current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else None
    if arguments.check:
        if current != expected:
            raise SystemExit(
                f"generated help reference is stale: run {Path(__file__).name}"
            )
        print(f"generated help reference: {OUTPUT.relative_to(ROOT)} is current")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(expected, encoding="utf-8", newline="\n")
    print(f"generated help reference: wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
