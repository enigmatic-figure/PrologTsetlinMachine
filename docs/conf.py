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

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = [
    ".agents/**",
    ".codex/**",
    ".gemini/**",
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

# Autodoc: Python package docstrings own the API reference (Checkpoint 4).
# Headers under include/ptm/*.hpp own native signatures; Doxygen+Breathe
# will be wired when doxygen is available in the build environment.
import sys

PACKAGE_ROOT = ROOT / "python"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

autosummary_generate = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
# Avoid duplicate index entries when re-exported symbols appear via __init__
autodoc_inherit_docstrings = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Sphinx generates many reference pages that are linked directly; the explicit
# toctree is for the primary navigation. Suppress noisy not_included warnings
# and duplicate-index warnings that autodoc can emit for re-exported enums.
suppress_warnings = [
    "toc.not_included",
    "autodoc",
    "ref.python.duplicate",
    "ref.duplicate",
]

myst_enable_extensions = ["colon_fence", "deflist", "tasklist"]
myst_heading_anchors = 4

html_theme = "alabaster"
html_title = f"PTM {release} documentation"
html_show_sourcelink = True

# Man pages: ptm(1) and ptmrt(1) share the argparse source with the CLI reference.
man_pages = [
    ("manual/reference/cli", "ptm", "PTM command line", [author], 1),
    ("manual/reference/c-api", "ptmrt", "PTM native runtime C ABI", [author], 1),
]
