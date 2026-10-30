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

"""Render boundary and facet meshes of a planar mesh and a tetrahedral mesh."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib.collections import LineCollection, PolyCollection
from PIL import Image

from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives.planar import annulus_2d
from physicsnemo.mesh.primitives.procedural import lumpy_ball

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "boundaries_facets.png"

MESH_COLOR = "#9ecae1"
EDGE_COLOR = "#304050"
FAINT_COLOR = "#eef2f5"
FAINT_EDGE = "#c5ccd1"
BOUNDARY_COLOR = "#d55e00"
FACET_COLOR = "#0072b2"
VIEW_VECTOR = (1.0, -1.4, 0.9)

plt.rcParams.update({"font.size": 13})


def _label(mesh, noun):
    """Describe a mesh by its dimensions and cell count."""

    return f"Mesh[{mesh.n_manifold_dims}, {mesh.n_spatial_dims}], {mesh.n_cells} {noun}"


def _draw_triangles(axis, mesh, facecolor, edgecolor):
    """Fill the triangles of a planar mesh."""

    points = mesh.points.numpy()
    axis.add_collection(
        PolyCollection(
            points[mesh.cells.numpy()],
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=0.8,
        )
    )


def _draw_edges(axis, mesh, color, linewidth):
    """Draw the cells of a planar edge mesh as line segments."""

    points = mesh.points.numpy()
    axis.add_collection(
        LineCollection(points[mesh.cells.numpy()], colors=color, linewidths=linewidth)
    )


def _render(dataset, color):
    """Render a shaded PyVista dataset off-screen and return the image."""

    plotter = pv.Plotter(window_size=(900, 900))
    plotter.enable_anti_aliasing("ssaa")
    plotter.add_mesh(
        dataset,
        color=color,
        show_edges=True,
        edge_color=EDGE_COLOR,
        line_width=0.6,
        # Lower diffuse and no specular keep a saturated color, such as the
        # vermillion boundary, on its hue instead of washing it out to orange.
        ambient=0.5,
        diffuse=0.55,
        specular=0.0,
    )
    plotter.set_background("white")
    plotter.view_vector(VIEW_VECTOR, viewup=(0.0, 0.0, 1.0))
    plotter.reset_camera()
    plotter.camera.zoom(1.5)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def main():
    """Generate the documentation image."""

    planar = annulus_2d.load(n_radial=3, n_angular=24)
    planar_boundary = planar.get_boundary_mesh()
    planar_facets = planar.get_facet_mesh()

    volume = lumpy_ball.load(noise_amplitude=0.2, seed=0)
    volume_boundary = volume.get_boundary_mesh()

    figure = plt.figure(figsize=(13.0, 10.4))
    grid = figure.add_gridspec(2, 6, height_ratios=(1.0, 1.25), hspace=0.18)
    planar_axes = [figure.add_subplot(grid[0, 2 * i : 2 * i + 2]) for i in range(3)]
    volume_axes = [figure.add_subplot(grid[1, 3 * i : 3 * i + 3]) for i in range(2)]

    _draw_triangles(planar_axes[0], planar, MESH_COLOR, EDGE_COLOR)
    planar_axes[0].set_title(f"annulus_2d\n{_label(planar, 'triangles')}")

    _draw_triangles(planar_axes[1], planar, FAINT_COLOR, FAINT_EDGE)
    _draw_edges(planar_axes[1], planar_boundary, BOUNDARY_COLOR, 3.0)
    planar_axes[1].set_title(f"get_boundary_mesh()\n{_label(planar_boundary, 'edges')}")

    _draw_edges(planar_axes[2], planar_facets, FACET_COLOR, 1.6)
    planar_axes[2].set_title(f"get_facet_mesh()\n{_label(planar_facets, 'edges')}")

    for axis in planar_axes:
        axis.set_xlim(-1.08, 1.08)
        axis.set_ylim(-1.08, 1.08)
        axis.set_aspect("equal")
        axis.set_axis_off()

    # Cut the ball open: drop the tetrahedra in the octant that faces the camera.
    facing = (volume.cell_centroids * volume.points.new_tensor(VIEW_VECTOR) > 0).all(1)
    cut = to_pyvista(volume).extract_cells((~facing).nonzero().squeeze(1).numpy())
    # Shrink each tetrahedron toward its centroid to show it.
    volume_axes[0].imshow(_render(cut.shrink(0.8), MESH_COLOR))
    volume_axes[0].set_title(
        f"lumpy_ball, cut open, tetrahedra shrunk\n{_label(volume, 'tetrahedra')}"
    )

    volume_axes[1].imshow(_render(to_pyvista(volume_boundary), BOUNDARY_COLOR))
    volume_axes[1].set_title(
        f"get_boundary_mesh()\n{_label(volume_boundary, 'triangles')}"
    )
    for axis in volume_axes:
        axis.set_axis_off()

    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.94)
    figure.savefig(OUTPUT, dpi=160, facecolor="white")
    # A 256-color palette keeps the file small and looks the same.
    Image.open(OUTPUT).convert("RGB").quantize(
        colors=256, method=Image.Quantize.FASTOCTREE
    ).save(OUTPUT, optimize=True)
    plt.close(figure)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
