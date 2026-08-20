"""Sphinx configuration for the compatibility-first PTM documentation tree."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = runpy.run_path(str(ROOT / "python" / "prolog_tsetlin" / "_version.py"))

project = "Prolog Tsetlin Machine"
author = "PTM contributors"
copyright = "2026, PTM contributors"
version = str(VERSION["__version__"])
release = version

extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
root_doc = "docs/index"
exclude_patterns = [
    ".agents/**",
    ".codex/**",
    ".git/**",
    ".github/**",
    ".hypothesis/**",
    ".pytest_cache/**",
    ".venv/**",
    "build/**",
    "dist/**",
    "out/**",
    "AGENTS.md",
]

myst_enable_extensions = ["colon_fence", "deflist", "tasklist"]
myst_heading_anchors = 4

html_theme = "alabaster"
html_title = f"PTM {release} documentation"
html_show_sourcelink = True
