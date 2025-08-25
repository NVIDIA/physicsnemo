# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os

from physicsnemo import __version__ as version

project = "NVIDIA PhysicsNeMo"
copyright = "2023, NVIDIA PhysicsNeMo Team"
author = "NVIDIA PhysicsNeMo Team"
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "recommonmark",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "nbsphinx",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

pdf_documents = [
    ("index", "rst2pdf", "Sample rst2pdf doc", "Your Name"),
]

napoleon_custom_sections = [("Variable Shape", "notes"), ("Forward", "params_style"), ("Outputs", "returns_style")]

# -- Options for HTML output -------------------------------------------------

# HTML theme options
html_theme = "nvidia_sphinx_theme"
html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#000000",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": False,
    # 'navigation_depth': 10,
    "sidebarwidth": 12,
    "includehidden": True,
    "titles_only": False,
}

# Additional html options
html_static_path = ["_static"]
html_css_files = [
    "css/nvidia_styles.css",
]
html_js_files = ["js/pk_scripts.js"]
# html_last_updated_fmt = ''

# Additional sphinx switches
math_number_all = False
todo_include_todos = True
numfig = True

_PREAMBLE = r"""
\usepackage{amsmath}
\usepackage{esint}
\usepackage{mathtools}
\usepackage{stmaryrd}
"""
latex_elements = {
    "preamble": _PREAMBLE,
    # other settings go here
}

latex_preamble = [
    (
        "\\usepackage{amssymb}",
        "\\usepackage{amsmath}",
        "\\usepackage{amsxtra}",
        "\\usepackage{bm}",
        "\\usepackage{esint}",
        "\\usepackage{mathtools}",
        "\\usepackage{stmaryrd}",
    ),
]

autosectionlabel_maxdepth = 1

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE.txt",
]

# Fake imports
autodoc_mock_imports = ["torch_scatter", "torch_cluster"]   # install of these packages takes very long

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
pdf_documents = [
    ("index", "rst2pdf", "Sample rst2pdf doc", "Your Name"),
]

# napoleon_custom_sections = ["Variable Shape"]
