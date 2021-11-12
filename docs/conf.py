# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath(os.pardir))


# -- Project information -----------------------------------------------------

project = "ABSOLV"
copyright = "2021, Simon Boothroyd"
author = "Simon Boothroyd"

# The short X.Y version
version = ""
# The full version, including alpha/beta/rc tags
release = ""


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "nbsphinx_link",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.autodoc_pydantic",
]

source_suffix = ".rst"

master_doc = "index"

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = 'default'

# Autodoc settings
autosummary_generate = True
autosummary_imported_members = False
autosummary_ignore___all__ = False
autosummary_context = {
    "exclude_modules": ["absolv.tests"],
}

autodoc_default_options = {
    "member-order": "bysource",
}
autodoc_mock_imports = ["openff", "mdtraj"]
autodoc_typehints = "description"

# Napoleon settings
napoleon_google_docstring = True
napoleon_use_rtype = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_preprocess_types = True

# autodoc_pydantic settings
autodoc_pydantic_show_config = False
autodoc_pydantic_model_show_config = False
autodoc_pydantic_show_validators = False
autodoc_pydantic_model_show_validators = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# nbsphinx settings
nbsphinx_execute = "never"

# sphinx bibtext settings
bibtex_bibfiles = [
    "user-guide/theory.bib",
    "user-guide/transformations.bib",
    "examples/equilibrium.bib",
    "examples/non-equilibrium.bib",
]

# Set up the intershinx mappings.
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "openff.toolkit": (
        "https://open-forcefield-toolkit.readthedocs.io/en/latest/",
        None,
    ),
    "openmm": ("http://docs.openmm.org/latest/api-python/", None),
}

# Set up mathjax.
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_immaterial"

html_theme_options = {
    # 'logo_svg':
    #     'logo.svg',
    "site_url": "https://github.com/SimonBoothroyd/absolv/",
    "repo_url": "https://github.com/SimonBoothroyd/absolv/",
    "repo_name": "SimonBoothroyd/absolv",
    "repo_type": "github",
    "globaltoc_depth": -1,
    "globaltoc_collapse": True,
    "globaltoc_includehidden": True,
    "features": [
        "navigation.expand",
        # 'navigation.tabs',
        # 'toc.integrate',
        "navigation.sections",
        # 'navigation.instant',
        # 'header.autohide',
        "navigation.top",
        # 'search.highlight',
        # 'search.share',
    ],
    "palette": [
        {
            "scheme": "default",
            "primary": "green",
            "accent": "light blue",
        },
    ],
}

html_show_sourcelink = False
html_copy_source = False

# Skip unnecessary footer text.
html_show_sphinx = False
html_show_copyright = False

html_static_path = ["_static"]

html_context = {
    "css_files": [
        "_static/basic.css",
        "_static/overrides.css",
        "_static/api.css",
    ]
}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "absolvdoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "absolv.tex", "ABSOLV Documentation", author, "manual"),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "absolv", "ABSOLV Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "absolv",
        "ABSOLV Documentation",
        author,
        "absolv",
        "Absolute solvation free energy calculations with OpenFF and OpenMM",
        "Miscellaneous",
    ),
]
