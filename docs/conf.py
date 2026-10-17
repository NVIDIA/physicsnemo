# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

# Grids, cards, tabs, and dropdowns in docs/api/mesh come from sphinx-design. The
# published build (the physicsnemo-docs repository) always has it. Without it, a
# local build renders the same content as plain blocks instead of failing.
try:
    import sphinx_design  # noqa: F401

    extensions.append("sphinx_design")
    _HAVE_SPHINX_DESIGN = True
except ImportError:
    _HAVE_SPHINX_DESIGN = False

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
    "research.md",
]

# Fake imports
autodoc_mock_imports = ["torch_scatter", "torch_cluster"]   # install of these packages takes very long

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
pdf_documents = [
    ("index", "rst2pdf", "Sample rst2pdf doc", "Your Name"),
]

# napoleon_custom_sections = ["Variable Shape"]


# Mesh and DomainMesh present these TensorDict methods as part of their own API
# (they re-declare them for type checkers), so their pages keep them.
_TENSORDICT_MEMBERS_TO_KEEP = frozenset({"save", "load", "clone"})


def _skip_tensordict_members(app, what, name, obj, skip, options):
    """Leave the TensorDict API of tensorclasses to the TensorDict docs.

    ``@tensorclass`` copies every TensorDict method onto the class it decorates
    (``physicsnemo.mesh.Mesh`` gains about 360 of them), so documenting them
    would bury the class's own API. Dataclass fields stay: tensorclass turns them
    into properties whose getter tensordict 0.15 names ``_make_prop`` (a private
    name; recheck it when upgrading tensordict).
    """
    if what != "class" or name in _TENSORDICT_MEMBERS_TO_KEEP:
        return None
    target = getattr(obj, "fget", obj)
    target = getattr(target, "__func__", target)
    module = getattr(target, "__module__", None) or ""
    from_tensordict = module.startswith("tensordict")
    is_field = "_make_prop" in getattr(target, "__qualname__", "")
    return True if from_tensordict and not is_field else None


def _register_sphinx_design_fallbacks(app):
    """Render sphinx-design directives as plain blocks when the package is missing."""

    from docutils import nodes
    from docutils.parsers.rst import directives
    from sphinx.util import logging
    from sphinx.util.docutils import SphinxDirective

    class _AnyOption(dict):
        """Accept every option. docutils parses options only when the spec is truthy."""

        def __missing__(self, key):
            return directives.unchanged

        def __bool__(self):
            return True

    class _PlainBlock(SphinxDirective):
        """Show the title, if any, as a rubric and the body as plain content."""

        has_content = True
        optional_arguments = 1
        final_argument_whitespace = True
        option_spec = _AnyOption()

        def run(self):
            out = []
            if self.arguments:
                out.append(nodes.rubric(text=self.arguments[0]))
            box = nodes.container()
            self.state.nested_parse(self.content, self.content_offset, box)
            out.append(box)
            return out

    def _plain_role(name, rawtext, text, lineno, inliner, options=None, content=None):
        return [nodes.literal(rawtext, text)], []

    for name in (
        "grid",
        "grid-item",
        "grid-item-card",
        "card",
        "tab-set",
        "tab-item",
        "dropdown",
    ):
        app.add_directive(name, _PlainBlock)
    for name in (
        "bdg",
        "bdg-primary",
        "bdg-secondary",
        "bdg-success",
        "bdg-info",
        "bdg-warning",
        "bdg-light",
    ):
        app.add_role(name, _plain_role)
    logging.getLogger(__name__).warning(
        "sphinx-design is not installed, so cards and tabs render as plain blocks. "
        "Run `pip install sphinx-design` to see the published layout."
    )


def setup(app):
    app.connect("autodoc-skip-member", _skip_tensordict_members)
    if not _HAVE_SPHINX_DESIGN:
        _register_sphinx_design_fallbacks(app)
