# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

from datetime import datetime
from importlib.metadata import version as get_version

# -- Project information -----------------------------------------------------
project = "SlickTune"
copyright = f"2026-{datetime.now().year}, SlickML"
author = "Amirhessam Tahmassebi"
version = get_version("slicktune")
release = version
language = "en"

# -- General configuration ---------------------------------------------------
# References:
# - https://www.sphinx-doc.org/en/master/usage/extensions/index.html
# - https://sphinx-design.readthedocs.io/en/furo-theme/
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "autoapi.extension",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx_design",
    "sphinx.ext.inheritance_diagram",
    "myst_parser",
    "sphinxcontrib.mermaid",
]
myst_enable_extensions = [
    "colon_fence",
]
# Auto link headings (needed for in-page TOC anchors)
myst_heading_anchors = 3
# Render ```mermaid fences via sphinxcontrib-mermaid
myst_fence_as_directive = [
    "mermaid",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

autoapi_dirs = [
    "../src/slicktune",
]

templates_path = [
    "_templates",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    ".ipynb_checkpoints",
]

pygments_style = "dracula"
pygments_dark_style = "dracula"

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = False

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#097EBE",
        "color-brand-content": "#097EBE",
    },
    "dark_css_variables": {
        "color-brand-primary": "#C302D5",
        "color-brand-content": "#C302D5",
    },
    "source_repository": "https://github.com/slickml/slick-tune",
    "source_branch": "master",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/slickml/slick-tune",
            "html": "🧩",
            "class": "",
        },
    ],
    "navigation_with_keys": True,
    "sidebar_hide_name": False,
}
html_title = "SlickTune"
html_logo = "_static/img/logo.png"
html_favicon = "_static/img/logo.png"
html_show_copyright = True
html_show_search_summary = True
html_show_sphinx = True
html_copy_source = False
html_output_encoding = "utf-8"
github_url = "https://github.com/slickml/slick-tune"

html_static_path = [
    "_static",
]
html_css_files = [
    "css/custom.css",
]
html_js_files = [
    "js/custom.js",
]
html_context = {
    "display_github": True,
    "github_user": "slickml",
    "github_repo": "slick-tune",
    "github_version": "master/docs/",
}

# -- Options for Auto-API-Docs -----------------------------------------------
autoapi_type = "python"
autoapi_template_dir = ""
autoapi_file_patterns = [
    "*.py",
    "*.pyi",
]
autoapi_generate_api_docs = True
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
    "inherited-members",
]
autoapi_ignore = [
    "*migrations*",
    "*TODO*",
]
autoapi_add_toctree_entry = False
autoapi_python_class_content = "class"
autoapi_member_order = "alphabetical"
autoapi_python_use_implicit_namespaces = False
autoapi_prepare_jinja_env = None
autoapi_keep_files = False
suppress_warnings: list[str] = []

# -- Options for View-Code ---------------------------------------------------
viewcode_follow_imported_members = True

# -- Options for Napoleon ----------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for TODOs -------------------------------------------------------
todo_include_todos = False
