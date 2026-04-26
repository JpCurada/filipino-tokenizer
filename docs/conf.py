import os
import sys

# Make the package importable for autodoc
sys.path.insert(0, os.path.abspath(".."))

# -- Project info ------------------------------------------------------------
project = "Filipino Tokenizer"
author = "John Paul M. Curada"
copyright = "2026, John Paul M. Curada"
release = "0.1.0"

# -- Extensions --------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",      # Google / NumPy docstring styles
    "sphinx.ext.viewcode",      # [source] links on API pages
    "sphinx.ext.intersphinx",   # cross-links to Python docs
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = "Filipino Tokenizer"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#22c55e",
        "color-brand-content": "#16a34a",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4ade80",
        "color-brand-content": "#86efac",
    },
}

# -- General -----------------------------------------------------------------
exclude_patterns = ["_build"]
templates_path = ["_templates"]
