# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared colors, fonts, and rendering helpers for the PhysicsNeMo-Mesh figures.

Each figure script in this folder imports this module, so that every figure uses
one visual language:

- A neutral mesh with Okabe-Ito accents, which stay distinct for readers with
  color vision deficiency.
- Matplotlib text only, two lines per panel title: the call or object, then
  ``Mesh[m, s]`` with counts taken from the live object.
- PyVista renders as images without frames, text, or scalar bars.
- White, opaque PNGs of about 2100 px width, reduced to a 256-color palette.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

### Palette
FACE = "#eef2f5"  # neutral planar mesh fill
EDGE = "#9aa5ad"  # neutral planar mesh edges
SURFACE = "#9ecae1"  # shaded 3D surface
SURFACE_EDGE = "#304050"  # edges of a shaded 3D surface
FAINT = "#eef2f5"  # context geometry behind a highlight
FAINT_EDGE = "#c5ccd1"
RESULT = "#d55e00"  # vermillion: what a call returns, a query, a selection
SECOND = "#0072b2"  # blue: neighbors, the edge graph, a second category
SECOND_FILL = "#56b4e9"  # sky blue fill for the second category
FIXED = "#009e73"  # bluish green: fixed or anchored points
FOURTH = "#e69f00"  # orange: a fourth category only
TEXT = "#202830"

### Colormaps
SEQUENTIAL = "viridis"
DIVERGING = "RdBu_r"  # pair with matplotlib.colors.TwoSlopeNorm(vcenter=0.0)
ERROR = "cividis"

### Geometry rendering
VIEW_VECTOR = (1.0, -1.4, 0.9)
VIEW_UP = (0.0, 0.0, 1.0)
DPI = 160

# Flat lighting for small accent actors (handles, anchors, highlighted cells), so their
# rendered color stays close to the palette color.
ACCENT_LIGHTING = {"ambient": 0.5, "diffuse": 0.55, "specular": 0.0}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 14,
        "text.color": TEXT,
        "axes.edgecolor": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
    }
)


def mesh_label(mesh, noun: str = "cells") -> str:
    """Describe a mesh by its dimensions and its cell count, as in panel titles."""

    return f"Mesh[{mesh.n_manifold_dims}, {mesh.n_spatial_dims}], {mesh.n_cells} {noun}"


def clean_axis(axis, limits=None) -> None:
    """Remove the frame and ticks of a mesh panel, keeping equal aspect."""

    if limits is not None:
        (x0, x1), (y0, y1) = limits
        axis.set_xlim(x0, x1)
        axis.set_ylim(y0, y1)
    axis.set_aspect("equal")
    axis.set_axis_off()


def render(actors, window_size=(900, 900), zoom=1.4, view_vector=VIEW_VECTOR):
    """Render PyVista actors off-screen on white and return the image array.

    Parameters
    ----------
    actors : list of tuple
        ``(dataset, add_mesh_kwargs)`` pairs, drawn in order.
    window_size : tuple of int
        Image size in pixels.
    zoom : float
        Camera zoom after the camera is fitted to the actors.
    view_vector : tuple of float
        Direction from the focal point to the camera.
    """

    import pyvista as pv

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.enable_anti_aliasing("ssaa")
    plotter.set_background("white")
    for dataset, kwargs in actors:
        plotter.add_mesh(dataset, **{"ambient": 0.35, **kwargs})
    plotter.view_vector(view_vector, viewup=VIEW_UP)
    plotter.reset_camera()
    plotter.camera.zoom(zoom)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def tubes(pv_mesh, radius: float):
    """Draw the edges of a PyVista dataset as tubes, because off-screen line widths are ignored."""

    return pv_mesh.extract_all_edges().tube(radius=radius)


def spheres(points, radius: float):
    """Draw points as small spheres. ``points`` is an ``(n, 3)`` array."""

    import pyvista as pv

    return pv.PolyData(points).glyph(
        geom=pv.Sphere(radius=radius), scale=False, orient=False
    )


def save(figure, path: Path, tight: bool = False, full_color: bool = False) -> None:
    """Save a figure as an opaque white PNG, and report its size.

    The image is reduced to a 256-color palette, which keeps the file small. The
    octree method keeps small accent colors true, where median cut would merge them
    into the shading. Pass ``full_color=True`` for figures with smooth color
    gradients, where a palette shows bands.
    """

    path = Path(path)
    figure.savefig(
        path, dpi=DPI, facecolor="white", bbox_inches="tight" if tight else None
    )
    plt.close(figure)
    image = Image.open(path).convert("RGB")
    if not full_color:
        image = image.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
    image.save(path, optimize=True)
    print(f"Saved {path} ({path.stat().st_size / 1024:.0f} KB)")
