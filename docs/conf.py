# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import pyvista
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "fedoo"
copyright = "2019, Etienne Prulière"
author = "Etienne Prulière"
master_doc = "index"

with open(
    os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"), "rb"
) as file:
    release = tomllib.load(file)["project"]["version"]

version = release
documentation_state = os.environ.get("FEDOO_DOC_STATE", "dev").lower()
if documentation_state not in {"beta", "stable", "dev"}:
    raise ValueError(
        "FEDOO_DOC_STATE must be one of 'beta', 'stable' or 'dev'; "
        f"got {documentation_state!r}."
    )

rst_epilog = f"""
.. |fedoo_version| replace:: {release}
.. |documentation_state| replace:: {documentation_state}
"""

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_gallery.gen_gallery",
    "pyvista.ext.plot_directive",
    "pyvista.ext.viewer_directive",
    "sphinx_design",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True
add_module_names = False
# Keep Python objects (classes, methods, functions) out of the page TOCs.
# They remain available through the autosummary tables in the page content.
toc_object_entries = False
# autodoc_mock_imports = ["simcoon"]
# autosummary_mock_imports = ["simcoon"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# html_theme = 'sphinxdoc'
html_logo = "_static/fedoo_logos_ss_fond.png"
html_title = f"Fedoo {release} ({documentation_state})"

html_theme_options = {
    # 'canonical_url': 'https://3mah.github.io/fedoo-docs/',
    "logo_only": True,
    # 'prev_next_buttons_location': 'bottom',
    "style_external_links": True,
    # 'vcs_pageview_mode': '',
    "style_nav_header_background": "#24445C",
    # Toc options
    "collapse_navigation": True,
    # 'sticky_navigation': True,
    "navigation_depth": 3,
    "includehidden": False,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Options for sphinx gallery for examples automatic generation ----------------------
pyvista.OFF_SCREEN = True
# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
# pyvista.set_jupyter_backend(None)

# -- Sphinx Gallery Options
# Examples listed in examples/precomputed_runs.toml are too heavy to be run at
# each build: they are rendered without being executed, and their page shows
# the precomputed animation and the run record (see tools/run_precomputed.py).
with open(
    os.path.join(os.path.dirname(__file__), "..", "examples", "precomputed_runs.toml"),
    "rb",
) as file:
    _precomputed = tomllib.load(file)
_precomputed_pages = {
    "examples/" + entry["path"][: -len(".py")]: entry
    for entry in _precomputed["example"]
}
_executed_pattern = (
    r"^(?!.*/(?:"
    + "|".join(re.escape(entry["path"]) for entry in _precomputed["example"])
    + r")$).*\.py$"
)

sphinx_gallery_conf = {
    # convert rst to md for ipynb
    "pypandoc": True,
    # path to your examples scripts
    "examples_dirs": ["../examples/"],
    # path where to save gallery generated examples
    "gallery_dirs": ["examples"],
    # Examples to execute: all except the precomputed ones (rendered only)
    "filename_pattern": _executed_pattern,
    # Basic examples first, then the categories in a fixed order
    "subsection_order": ExplicitOrder(
        [
            f"../examples/{category}"
            for category in (
                "basic",
                "boundary_conditions",
                "contact",
                "finite_strain",
                "plasticity",
                "shells",
                "dynamics",
                "thermal",
                "poromechanics",
                "custom_weak_forms",
                "homogenization",
                "mesh",
            )
        ]
        + ["*"]
    ),
    # Sort gallery example by file name instead of number of lines (default)
    "within_subsection_order": FileNameSortKey,
    "image_scrapers": (DynamicScraper(), "matplotlib"),
}


def _precomputed_banner(entry):
    """Return the rst inserted at the top of a precomputed example page."""
    run = entry.get("run", {})
    if not run:
        return (
            ".. note::\n\n"
            "   This example is too heavy to be run at each documentation\n"
            "   build. Its results have not been generated yet: run the\n"
            "   script to obtain them.\n\n"
        )
    versions = ", ".join(
        f"{name} {run[name]}"
        for name in ("fedoo", "simcoon", "ipctk", "numpy")
        if name in run
    )
    if "commit" in run:
        versions += f" (fedoo commit {run['commit']})"
    if "media" not in entry:
        media = None
    elif entry["media"].endswith(".gif"):
        url = f"{_precomputed['media_root']}/{entry['media']}"
        media = f'   <img src="{url}" style="width: 100%" alt="{entry["path"]}">\n'
    else:
        url = f"{_precomputed['media_root']}/{entry['media']}"
        media = (
            f'   <video controls loop muted playsinline style="width: 100%"'
            f' src="{url}"></video>\n'
        )
    what = "The animation below was" if media else "Its results were"
    note = (
        ".. note::\n\n"
        "   This example is too heavy to be run at each documentation build.\n"
        f"   {what} precomputed on "
        f"{run.get('date', '?')} with {versions}, in about "
        f"{run.get('wall_time_s', '?')} s ({run.get('machine', '?')}"
        + (f", BLAS {run['blas']}" if "blas" in run else "")
        + "). Run\n"
        "   the script to reproduce it with your fedoo version.\n\n"
    )
    if media is None:
        return note
    return note + ".. raw:: html\n\n" + media + "\n"


def _insert_precomputed_banner(app, docname, source):
    """Insert the banner after the title of a precomputed example page."""
    entry = _precomputed_pages.get(docname)
    if entry is None:
        return
    lines = source[0].split("\n")
    label = f".. _sphx_glr_{docname.replace('/', '_')}.py:"
    try:
        start = lines.index(label)
    except ValueError:
        return
    for i in range(start + 1, len(lines) - 1):
        underline = lines[i + 1]
        if lines[i].strip() and underline and set(underline) <= set("=-~^*"):
            lines.insert(i + 2, "\n" + _precomputed_banner(entry))
            source[0] = "\n".join(lines)
            return


def setup(app):
    app.connect("source-read", _insert_precomputed_banner)
