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

"""Show the input and the output of the three meshers: fill_interior, mesh_implicit_domain, marching_cubes."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from _style import (
    DIVERGING,
    EDGE,
    FACE,
    RESULT,
    SEQUENTIAL,
    SURFACE,
    SURFACE_EDGE,
    TEXT,
    clean_axis,
    render,
    save,
)
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import TwoSlopeNorm

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.generate import (
    marching_cubes,
    mesh_implicit_domain,
    sdf_box,
    sdf_difference,
    sdf_sphere,
)
from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives.curves import circle_2d
from physicsnemo.mesh.tessellation import fill_interior

OUTPUT = Path(__file__).parent / "mesh_generators.png"

H = 0.15
THRESHOLD = 0.5
# An odd grid size puts the middle slice of the grid exactly at z = 0.
GRID = 49
BLOBS = ((-0.5, -0.1, 0.0), (0.5, 0.1, 0.1), (0.0, 0.55, -0.2))


def _triangles(axis, mesh):
    """Fill the triangles of a planar mesh."""

    points = mesh.points.numpy()
    axis.add_collection(
        PolyCollection(
            points[mesh.cells.numpy()], facecolors=FACE, edgecolors=EDGE, linewidths=0.8
        )
    )


def _edges(axis, mesh, color, linewidth):
    """Draw the cells of a planar edge mesh as line segments."""

    points = mesh.points.numpy()
    axis.add_collection(
        LineCollection(points[mesh.cells.numpy()], colors=color, linewidths=linewidth)
    )


def _dots(axis, points, color):
    """Draw points as white-rimmed dots."""

    axis.scatter(
        points[:, 0],
        points[:, 1],
        s=26,
        color=color,
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
        clip_on=False,
    )


def _arrow(axis):
    """Point from an input panel down to its output panel."""

    axis.annotate(
        "",
        xy=(0.5, -0.2),
        xytext=(0.5, -0.04),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>,head_width=0.4", "color": TEXT, "lw": 2.0},
        annotation_clip=False,
    )


def main():
    """Generate the documentation image."""

    ### fill_interior: two loops, an outer circle and a hole, as one boundary.
    outer = circle_2d.load(radius=1.0, n_points=40)
    hole = circle_2d.load(radius=0.38, n_points=20).translate(torch.tensor([0.3, 0.15]))
    boundary = Mesh.merge([outer, hole])
    filled = fill_interior(boundary, max_cell_size=0.02, smooth_iterations=3)

    ### mesh_implicit_domain: a box minus a disk, as a signed distance function.
    lo, hi = torch.tensor([-1.0, -0.8]), torch.tensor([1.0, 0.8])
    phi = sdf_difference(sdf_box(lo, hi), sdf_sphere(torch.tensor([0.2, 0.05]), 0.4))
    # The bounds box coincides with the outer box, so its faces stay exact.
    implicit = mesh_implicit_domain(phi, (lo.tolist(), hi.tolist()), H)
    gx, gy = torch.meshgrid(
        torch.linspace(-1.1, 1.1, 300), torch.linspace(-0.9, 0.9, 250), indexing="xy"
    )
    phi_grid = phi(torch.stack([gx, gy], dim=-1)).numpy()

    ### marching_cubes: an isosurface of three Gaussian blobs on a grid.
    axis_1d = torch.linspace(-1.5, 1.5, GRID)
    x, y, z = torch.meshgrid(axis_1d, axis_1d, axis_1d, indexing="ij")
    field = sum(
        torch.exp(-((x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2) / 0.25)
        for a, b, c in BLOBS
    )
    surface = marching_cubes(field, threshold=THRESHOLD, coords=(axis_1d,) * 3)
    surface_pv = to_pyvista(surface)
    cut = surface_pv.slice(normal="z", origin=(0.0, 0.0, 0.0)).tube(radius=0.018)
    surface_image = render(
        [
            (
                surface_pv,
                {
                    "color": SURFACE,
                    "show_edges": True,
                    "edge_color": SURFACE_EDGE,
                    "line_width": 0.4,
                    "smooth_shading": False,
                },
            ),
            (cut, {"color": RESULT}),
        ],
        zoom=1.3,
    )

    figure, axes = plt.subplots(
        2, 3, figsize=(14.0, 9.4), gridspec_kw={"hspace": 0.42, "wspace": 0.06}
    )
    disk_limits = ((-1.08, 1.08), (-1.08, 1.08))
    box_limits = ((-1.12, 1.12), (-0.92, 0.92))

    # Column 1: an exact boundary in, quality triangles that keep it out.
    axis = axes[0, 0]
    _edges(axis, boundary, TEXT, 1.6)
    _dots(axis, boundary.points.numpy(), RESULT)
    axis.set_title(
        f"input: two circle_2d loops, merged\n"
        f"Mesh[1, 2], {boundary.n_cells} edges, vertices in vermillion"
    )
    clean_axis(axis, disk_limits)
    _arrow(axis)
    axis = axes[1, 0]
    _triangles(axis, filled)
    _dots(axis, boundary.points.numpy(), RESULT)
    axis.set_title(
        f"fill_interior(boundary)\nMesh[2, 2], {filled.n_cells} triangles, exact boundary"
    )
    clean_axis(axis, disk_limits)

    # Column 2: an implicit function in, a mesh whose boundary follows its zero set.
    axis = axes[0, 1]
    extent = (-1.1, 1.1, -0.9, 0.9)
    axis.contourf(
        phi_grid,
        levels=16,
        extent=extent,
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vcenter=0.0),
    )
    axis.contour(phi_grid, levels=[0.0], colors=RESULT, linewidths=2.2, extent=extent)
    axis.set_title(
        "input: phi = sdf_difference(box, disk)\nnegative inside (blue), zero in vermillion"
    )
    clean_axis(axis, box_limits)
    _arrow(axis)
    axis = axes[1, 1]
    _triangles(axis, implicit)
    axis.contour(phi_grid, levels=[0.0], colors=RESULT, linewidths=1.4, extent=extent)
    axis.set_title(
        f"mesh_implicit_domain(phi, bounds, h={H})\n"
        f"Mesh[2, 2], {implicit.n_cells} triangles, boundary within $O(h^2)$"
    )
    clean_axis(axis, box_limits)

    # Column 3: a sampled scalar field in, its isosurface out.
    axis = axes[0, 2]
    middle = field[:, :, GRID // 2].numpy().T
    extent = (-1.5, 1.5, -1.5, 1.5)
    axis.imshow(middle, origin="lower", extent=extent, cmap=SEQUENTIAL)
    axis.contour(
        middle, levels=[THRESHOLD], colors=RESULT, linewidths=2.2, extent=extent
    )
    axis.set_title(
        f"input: field on a {GRID}\u00b3 grid, slice z = 0\nlevel {THRESHOLD} in vermillion"
    )
    clean_axis(axis)
    _arrow(axis)
    axis = axes[1, 2]
    axis.imshow(surface_image)
    axis.set_title(
        f"marching_cubes(field, threshold={THRESHOLD})\n"
        f"Mesh[2, 3], {surface.n_cells} triangles, surface only"
    )
    clean_axis(axis)

    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93)
    save(figure, OUTPUT)


if __name__ == "__main__":
    main()
