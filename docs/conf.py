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
    "source_repository": "https://github.com/JpCurada/filipino-tokenizer",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#22c55e",
        "color-brand-content": "#16a34a",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4ade80",
        "color-brand-content": "#86efac",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/JpCurada/filipino-tokenizer",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/filipino-tokenizer/",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 24 24">
                    <path d="M12 0L1.5 6v12L12 24l10.5-6V6L12 0zm0 2.18l8.5 4.9v9.84L12 21.82l-8.5-4.9V7.08L12 2.18z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# -- General -----------------------------------------------------------------
exclude_patterns = ["_build"]
templates_path = ["_templates"]
