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

"""Convert a 2 x 2 x 2 voxel grid with from_pyvista in four ways."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from _style import (
    ACCENT_LIGHTING,
    FAINT_EDGE,
    RESULT,
    SECOND,
    SURFACE,
    SURFACE_EDGE,
    TEXT,
    render,
    save,
    spheres,
    tubes,
)

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import from_pyvista, to_pyvista

OUTPUT = Path(__file__).parent / "io_cells_to_simplices.png"

WINDOW = (900, 900)
ZOOM = 1.05
SOLID = {"show_edges": True, "edge_color": SURFACE_EDGE, "line_width": 1.0}


def _graph(mesh, context):
    """A graph as blue tubes and dark nodes, over the faint edges of the voxels."""

    return [
        (tubes(context, radius=0.008), {"color": FAINT_EDGE}),
        (to_pyvista(mesh).tube(radius=0.022), {"color": SECOND}),
        (spheres(mesh.points.numpy(), radius=0.06), {"color": TEXT}),
    ]


def main():
    """Generate the documentation image."""

    grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()

    tets = from_pyvista(grid)
    edges = from_pyvista(grid, manifold_dim=1)
    centroids = from_pyvista(grid, point_source="cell_centroids")
    dual = from_pyvista(grid, manifold_dim=1, point_source="cell_centroids")

    # Mark one voxel, the one nearest the camera, and the tetrahedra cut from it.
    voxel_centers = torch.from_numpy(np.asarray(grid.cell_centers().points)).float()
    marked = int(torch.argmax(voxel_centers @ torch.tensor([1.0, -1.4, 0.9])))
    lo = voxel_centers[marked] - 0.5
    inside = ((tets.cell_centroids > lo) & (tets.cell_centroids < lo + 1.0)).all(dim=1)
    voxel_mask = np.arange(grid.n_cells) == marked
    # Pull the tetrahedra of the marked voxel apart, away from its center.
    corners = tets.points[tets.cells[inside]]
    offsets = corners.mean(dim=1, keepdim=True) - voxel_centers[marked]
    exploded = Mesh(
        points=(corners + 0.9 * offsets).reshape(-1, 3),
        cells=torch.arange(corners.shape[0] * 4).reshape(-1, 4),
    )
    others = to_pyvista(tets.slice_cells(~inside)).shrink(0.92)
    plain_voxels = grid.extract_cells(np.flatnonzero(~voxel_mask)).shrink(0.86)
    marked_voxel = grid.extract_cells(np.flatnonzero(voxel_mask)).shrink(0.86)
    accent = {"color": RESULT, **ACCENT_LIGHTING}

    panels = [
        (
            "pv.ImageData(dimensions=(3, 3, 3))\n"
            f".cast_to_unstructured_grid()\n{grid.n_cells} voxels, {grid.n_points} points",
            [
                (plain_voxels, {"color": SURFACE, **SOLID}),
                (marked_voxel, {**accent, **SOLID}),
            ],
        ),
        (
            f"from_pyvista(grid)\nMesh[3, 3], {tets.n_cells} tetrahedra,\n"
            f"{int(inside.sum())} for each voxel",
            [
                (others, {"color": SURFACE, **SOLID}),
                (to_pyvista(exploded).shrink(0.9), {**accent, **SOLID}),
            ],
        ),
        (
            "from_pyvista(grid, manifold_dim=1)\n"
            f"Mesh[1, 3], {edges.n_points} points, {edges.n_cells} edges",
            _graph(edges, grid),
        ),
        (
            'from_pyvista(grid,\npoint_source="cell_centroids")\n'
            f"Mesh[0, 3], {centroids.n_points} points",
            [
                (tubes(grid, radius=0.008), {"color": FAINT_EDGE}),
                (spheres(centroids.points.numpy(), radius=0.075), accent),
            ],
        ),
        (
            'from_pyvista(grid, manifold_dim=1,\npoint_source="cell_centroids")\n'
            f"Mesh[1, 3], {dual.n_points} points, {dual.n_cells} edges",
            _graph(dual, grid),
        ),
    ]

    figure = plt.figure(figsize=(14.0, 7.6))
    layout = figure.add_gridspec(
        2, 3, left=0.005, right=0.995, top=0.9, bottom=0.01, hspace=0.32, wspace=0.04
    )
    slots = [layout[:, 0], layout[0, 1], layout[0, 2], layout[1, 1], layout[1, 2]]
    for slot, (title, actors) in zip(slots, panels):
        axis = figure.add_subplot(slot)
        axis.imshow(render(actors, window_size=WINDOW, zoom=ZOOM))
        axis.set_axis_off()
        axis.set_title(title, fontsize=12.5)
    figure.text(0.335, 0.47, "→", fontsize=40, ha="center", va="center", color=TEXT)
    save(figure, OUTPUT)


if __name__ == "__main__":
    main()
