import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Solana Agent"
copyright = "2025, Bevan Hunt"
author = "Bevan Hunt"
release = "18.0.2"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
