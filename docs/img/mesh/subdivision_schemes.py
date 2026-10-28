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

"""Render a coarse cube surface and its linear, Loop, and Butterfly subdivisions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pyvista as pv
from PIL import Image

from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives.surfaces import cube_surface

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "subdivision_schemes.png"

LEVELS = 3
MESH_COLOR = "#9ecae1"
EDGE_COLOR = "#304050"
CAGE_COLOR = "#d55e00"

plt.rcParams.update({"font.size": 13})


def _render(mesh, coarse):
    """Render a surface with the coarse cube's edges and vertices on top."""

    # Draw edges and vertices as geometry: off-screen OpenGL ignores line widths.
    cage = to_pyvista(coarse).extract_all_edges().tube(radius=0.009)
    corners = pv.PolyData(coarse.points.numpy()).glyph(
        geom=pv.Sphere(radius=0.035), scale=False, orient=False
    )
    plotter = pv.Plotter(window_size=(800, 800))
    plotter.enable_anti_aliasing("ssaa")
    plotter.add_mesh(
        to_pyvista(mesh),
        color=MESH_COLOR,
        show_edges=True,
        edge_color=EDGE_COLOR,
        line_width=0.6,
        ambient=0.35,
        diffuse=0.7,
    )
    plotter.add_mesh(cage, color=CAGE_COLOR)
    plotter.add_mesh(corners, color=CAGE_COLOR)
    plotter.set_background("white")
    plotter.camera_position = [(2.2, -2.9, 1.9), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    plotter.camera.zoom(1.05)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def main():
    """Generate the documentation image."""

    coarse = cube_surface.load()
    panels = [("cube_surface", f"{coarse.n_cells} triangles", coarse)]
    for scheme, kind in (
        ("linear", "interpolating, flat"),
        ("loop", "approximating, smooth"),
        ("butterfly", "interpolating, smooth"),
    ):
        mesh = coarse.subdivide(levels=LEVELS, filter=scheme)
        panels.append((f'filter="{scheme}"', kind, mesh))
    n_cells = panels[-1][2].n_cells

    figure, axes = plt.subplots(1, 4, figsize=(14.0, 4.4))
    for axis, (title, kind, mesh) in zip(axes, panels, strict=True):
        axis.imshow(_render(mesh, coarse))
        axis.set_title(f"{title}\n{kind}")
        axis.set_axis_off()
    figure.text(
        0.5,
        0.02,
        f"subdivide(levels={LEVELS}, filter=...) turns {coarse.n_cells} triangles "
        f"into {n_cells}. Orange: the edges and vertices of the coarse cube.",
        ha="center",
        fontsize=13,
    )
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=0.86, wspace=0.02)
    figure.savefig(OUTPUT, dpi=160, facecolor="white")
    # A 256-color palette keeps the file small and looks the same.
    Image.open(OUTPUT).convert("RGB").quantize(
        colors=256, method=Image.Quantize.FASTOCTREE
    ).save(OUTPUT, optimize=True)
    plt.close(figure)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
