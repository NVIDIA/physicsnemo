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

"""Render the four adjacency queries on one small planar triangle mesh."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

from physicsnemo.mesh.primitives.procedural import perturbed_grid

OUTPUT = Path(__file__).parent / "neighbors_adjacency.png"

EDGE_COLOR = "#9aa5ad"
FACE_COLOR = "#eef2f5"
QUERY_COLOR = "#d55e00"
NEIGHBOR_COLOR = "#0072b2"
NEIGHBOR_FILL = "#56b4e9"

plt.rcParams.update({"font.size": 13})


def _draw_cells(axis, points, cells, color, *, linewidth=0.8, zorder=1):
    """Fill triangles with one color and draw their edges."""

    axis.add_collection(
        PolyCollection(
            points[cells],
            facecolors=color,
            edgecolors=EDGE_COLOR if color == FACE_COLOR else "white",
            linewidths=linewidth,
            zorder=zorder,
        )
    )


def _draw_points(axis, points, color, size):
    """Draw vertices as dots with a white rim."""

    axis.scatter(
        points[:, 0],
        points[:, 1],
        s=size,
        color=color,
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
    )


def main():
    """Generate the documentation image."""

    mesh = perturbed_grid.load(n_x=6, n_y=6, perturbation_scale=0.04, seed=0)
    points = mesh.points.numpy()
    cells = mesh.cells.numpy()
    center = mesh.points.new_tensor([0.5, 0.5])
    query_point = int(torch.linalg.vector_norm(mesh.points - center, dim=1).argmin())

    point_neighbors = mesh.get_point_to_points_adjacency().to_list()[query_point]
    point_cells = mesh.get_point_to_cells_adjacency().to_list()[query_point]
    # Query the largest cell around the query point, so that it is easy to see.
    query_cell = point_cells[int(mesh.cell_areas[point_cells].argmax())]
    cell_neighbors = mesh.get_cell_to_cells_adjacency().to_list()[query_cell]
    cell_points = mesh.get_cell_to_points_adjacency().to_list()[query_cell]

    figure, axes = plt.subplots(2, 2, figsize=(10.4, 11.4))
    axes = axes.ravel()
    for axis in axes:
        _draw_cells(axis, points, cells, FACE_COLOR)

    axis = axes[0]
    axis.add_collection(
        LineCollection(
            [[points[query_point], points[j]] for j in point_neighbors],
            colors=NEIGHBOR_COLOR,
            linewidths=2.5,
            zorder=3,
        )
    )
    _draw_points(axis, points[point_neighbors], NEIGHBOR_COLOR, 110)
    _draw_points(axis, points[[query_point]], QUERY_COLOR, 220)
    axis.set_title(
        "point to points\n"
        f"get_point_to_points_adjacency()\n{len(point_neighbors)} neighbors"
    )

    axis = axes[1]
    _draw_cells(axis, points, cells[point_cells], NEIGHBOR_FILL, zorder=2)
    _draw_points(axis, points[[query_point]], QUERY_COLOR, 220)
    axis.set_title(
        f"point to cells\nget_point_to_cells_adjacency()\n{len(point_cells)} cells"
    )

    axis = axes[2]
    _draw_cells(axis, points, cells[cell_neighbors], NEIGHBOR_FILL, zorder=2)
    _draw_cells(axis, points, cells[[query_cell]], QUERY_COLOR, zorder=2)
    axis.set_title(
        "cell to cells (shared edge)\n"
        f"get_cell_to_cells_adjacency()\n{len(cell_neighbors)} cells"
    )

    axis = axes[3]
    _draw_cells(axis, points, cells[[query_cell]], QUERY_COLOR, zorder=2)
    _draw_points(axis, points[cell_points], NEIGHBOR_COLOR, 110)
    axis.set_title(
        f"cell to points\nget_cell_to_points_adjacency()\n{len(cell_points)} points"
    )

    for axis in axes:
        axis.set_xlim(-0.04, 1.04)
        axis.set_ylim(-0.04, 1.04)
        axis.set_aspect("equal")
        axis.set_axis_off()

    figure.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", markersize=12, color=QUERY_COLOR),
            Patch(color=QUERY_COLOR),
            Line2D(
                [], [], marker="o", linestyle="", markersize=9, color=NEIGHBOR_COLOR
            ),
            Patch(color=NEIGHBOR_FILL),
        ],
        labels=["query point", "query cell", "neighbor points", "neighbor cells"],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=14,
    )
    figure.subplots_adjust(
        left=0.02, right=0.98, bottom=0.07, top=0.92, wspace=0.08, hspace=0.3
    )
    figure.savefig(OUTPUT, dpi=160, facecolor="white")
    # A 256-color palette keeps the file small and looks the same.
    Image.open(OUTPUT).convert("RGB").quantize(
        colors=256, method=Image.Quantize.FASTOCTREE
    ).save(OUTPUT, optimize=True)
    plt.close(figure)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
