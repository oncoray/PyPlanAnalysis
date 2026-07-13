# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make sure PyPlanAnalysis is importable even if the package hasn't been
# pip-installed in the environment building the docs (e.g. plain CI runner).
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------
project   = "PyPlanAnalysis"
author    = "Giovanni Parrella"
copyright = "2026, Giovanni Parrella"
release   = "1.0.0"

# -- General configuration -----------------------------------------------
extensions = [
    "sphinx.ext.autodoc",        # pull docstrings from the code
    "sphinx.ext.napoleon",       # understand NumPy-style docstrings
    "sphinx.ext.viewcode",       # add "[source]" links
    "sphinx.ext.autosummary",    # generate per-module summary tables
    #"sphinx.ext.intersphinx",    # link out to numpy/pandas/etc. docs
    "sphinx_autodoc_typehints",  # render type hints as part of the signature
    "myst_parser",               # allow README.md to be included as-is
]
autosummary_generate = True
add_module_names = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_mock_imports = ["SimpleITK"]  # avoid requiring a native lib just to build docs

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

#intersphinx_mapping = {
#    "python": ("https://docs.python.org/3", None),
#    "numpy": ("https://numpy.org/doc/stable/", None),
#    "pandas": ("https://pandas.pydata.org/docs/", None),
#}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output ----------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "PyPlanAnalysis documentation"
html_theme_options = {
    "sidebar_hide_name": False,
}
