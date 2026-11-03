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

"""Draw a five-point mesh beside the tensors that store it: geometry and data."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from _style import (
    EDGE,
    FIXED,
    FOURTH,
    RESULT,
    SECOND,
    SECOND_FILL,
    TEXT,
    clean_axis,
    save,
)
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import ConnectionPatch, Rectangle

from physicsnemo.mesh import Mesh

OUTPUT = Path(__file__).parent / "mesh_data_model.png"

MONO = {"family": "DejaVu Sans Mono"}
CELL_COLORS = [SECOND_FILL, FOURTH, FIXED, "#cc79a7"]  # Okabe-Ito, one per triangle
ROW = 0.5  # table row height, in table-axis units
COL = 0.95  # table column width


def _build():
    """A unit square split into four triangles around its center, with data."""

    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    cells = torch.tensor([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    mesh = Mesh(points=points, cells=cells)
    x, y = mesh.points.unbind(-1)
    mesh.point_data["temperature"] = 300.0 + 20.0 * x + 10.0 * y
    mesh.point_data["velocity"] = torch.stack([1.0 - 0.6 * y, 0.2 + 0.4 * x], dim=-1)
    mesh.cell_data["pressure"] = 1.0 - mesh.cell_centroids[:, 1]
    mesh.global_data["Re"] = torch.tensor(1000.0)
    return mesh


def _table(axis, x, y, values, fmt, header, color=TEXT, row_colors=None, mark=None):
    """Draw a tensor as a grid of boxes. ``(x, y)`` is the top-left corner."""

    values = values.reshape(values.shape[0], -1)
    n_rows, n_cols = values.shape
    axis.text(
        x + n_cols * COL / 2,
        y + 0.3,
        header,
        ha="center",
        va="bottom",
        color=color,
        **MONO,
    )
    for i in range(n_rows):
        top = y - i * ROW
        if row_colors is not None:
            axis.add_patch(
                Rectangle((x - 0.14, top - ROW), 0.12, ROW, color=row_colors[i], lw=0)
            )
        for j in range(n_cols):
            axis.add_patch(
                Rectangle(
                    (x + j * COL, top - ROW),
                    COL,
                    ROW,
                    facecolor=to_rgba(RESULT, 0.16) if i == mark else "white",
                    edgecolor=EDGE,
                    linewidth=1.0,
                )
            )
            axis.text(
                x + (j + 0.5) * COL,
                top - ROW / 2,
                fmt.format(float(values[i, j])),
                ha="center",
                va="center",
                fontsize=12,
                **MONO,
            )
    return x + n_cols * COL


def _bracket(axis, x0, x1, y, text, color):
    """A horizontal bracket above a group of tables, with a label."""

    axis.plot([x0, x0, x1, x1], [y - 0.12, y, y, y - 0.12], color=color, lw=1.6)
    axis.text((x0 + x1) / 2, y + 0.1, text, ha="center", va="bottom", color=color)


def main():
    """Generate the documentation image."""

    mesh = _build()
    points = mesh.points.numpy()

    figure = plt.figure(figsize=(14.0, 6.0))
    mesh_axis = figure.add_axes((0.01, 0.08, 0.33, 0.74))
    table_axis = figure.add_axes((0.36, 0.02, 0.63, 0.96))

    ### The mesh: one color per triangle, point and cell labels, velocity arrows
    mesh_axis.add_collection(
        PolyCollection(
            points[mesh.cells.numpy()],
            facecolors=[to_rgba(c, 0.35) for c in CELL_COLORS],
            edgecolors=TEXT,
            linewidths=1.4,
        )
    )
    for c, centroid in enumerate(mesh.cell_centroids.numpy()):
        offset = 0.25 * (centroid - points[4])
        mesh_axis.text(
            *(centroid + offset), f"c{c}", ha="center", va="center", fontsize=14, **MONO
        )
    velocity = mesh.point_data["velocity"].numpy()
    mesh_axis.quiver(
        points[:, 0],
        points[:, 1],
        velocity[:, 0],
        velocity[:, 1],
        color=SECOND,
        scale=6.5,
        width=0.007,
        zorder=3,
    )
    colors = [TEXT] * 4 + [RESULT]
    mesh_axis.scatter(
        points[:, 0], points[:, 1], s=110, c=colors, edgecolor="white", lw=1.2, zorder=4
    )
    for i, (px, py) in enumerate(points):
        dx = -0.05 if px < 0.75 or i == 2 else 0.05
        dy = -0.05 if py < 0.75 else 0.05
        if i == 4:
            dx, dy = -0.07, 0.0
        mesh_axis.text(
            px + dx,
            py + dy,
            str(i),
            ha="right" if dx < 0 else "left",
            va="center" if i == 4 else ("top" if dy < 0 else "bottom"),
            fontsize=14,
            color=colors[i],
            **MONO,
        )
    clean_axis(mesh_axis, ((-0.2, 1.42), (-0.2, 1.3)))
    mesh_axis.set_title(
        "Mesh(points, cells)\n"
        f"Mesh[{mesh.n_manifold_dims}, {mesh.n_spatial_dims}], "
        f"{mesh.n_points} points, {mesh.n_cells} triangles"
    )

    ### The tensors: geometry on the left, data on the right
    axis = table_axis
    axis.set_xlim(0.0, 12.0)
    axis.set_ylim(0.2, 9.8)
    axis.set_axis_off()

    top_points, top_cells = 7.9, 3.8
    x_geometry, x_data = 0.9, 5.9
    small = {"va": "top", "fontsize": 11, **MONO}
    for i in range(mesh.n_points):
        y = top_points - (i + 0.5) * ROW
        axis.text(0.75, y, str(i), ha="right", va="center", color=colors[i], **MONO)
    _table(axis, x_geometry, top_points, mesh.points, "{:.1f}", "points", mark=4)
    shape = f"{tuple(mesh.points.shape)}\n(n_points, n_spatial_dims)"
    axis.text(x_geometry, top_points - 5 * ROW - 0.15, shape, **small)
    for c in range(mesh.n_cells):
        y = top_cells - (c + 0.5) * ROW
        axis.text(0.75, y, f"c{c}", ha="right", va="center", **MONO)
    end = _table(
        axis,
        x_geometry,
        top_cells,
        mesh.cells,
        "{:.0f}",
        "cells",
        row_colors=CELL_COLORS,
    )
    shape = f"{tuple(mesh.cells.shape)}\n(n_cells, n_manifold_dims + 1)"
    axis.text(x_geometry, top_cells - 4 * ROW - 0.15, shape, **small)

    def group(x0, x1, y, name):
        """The name of a data dictionary over its tables."""

        axis.text((x0 + x1) / 2, y + 0.82, name, ha="center", va="bottom", **MONO)
        axis.plot([x0, x1], [y + 0.78, y + 0.78], color=EDGE, lw=1.0)

    x_velocity = x_data + 2.7
    _table(
        axis,
        x_data,
        top_points,
        mesh.point_data["temperature"],
        "{:.0f}",
        '["temperature"]',
        mark=4,
    )
    end_data = _table(
        axis,
        x_velocity,
        top_points,
        mesh.point_data["velocity"],
        "{:.1f}",
        '["velocity"]',
        color=SECOND,
        mark=4,
    )
    group(x_data - 0.9, end_data + 0.1, top_points, "point_data")
    _table(
        axis, x_data, top_cells, mesh.cell_data["pressure"], "{:.2f}", '["pressure"]'
    )
    group(x_data - 0.9, x_data + COL + 0.9, top_cells, "cell_data")
    x_global = x_velocity + COL / 2
    _table(
        axis, x_global, top_cells, mesh.global_data["Re"].reshape(1), "{:.0f}", '["Re"]'
    )
    group(x_global - 0.9, x_global + COL + 0.9, top_cells, "global_data")
    axis.text(
        x_global + COL / 2,
        top_cells - ROW - 0.15,
        "one value for\nthe whole mesh",
        ha="center",
        va="top",
        fontsize=12,
    )

    _bracket(axis, 0.3, end + 0.1, 9.3, "geometry", TEXT)
    _bracket(axis, x_data - 1.0, x_global + COL + 0.9, 9.3, "data", TEXT)

    ### A link from the first row of cells to its triangle
    figure.add_artist(
        ConnectionPatch(
            xyA=(0.3, top_cells - 0.5 * ROW),
            coordsA=axis.transData,
            xyB=tuple(mesh.cell_centroids[0].numpy() + [0.12, -0.06]),
            coordsB=mesh_axis.transData,
            arrowstyle="-|>",
            color=TEXT,
            lw=1.4,
            mutation_scale=16,
            connectionstyle="arc3,rad=-0.25",
        )
    )
    save(figure, OUTPUT)


if __name__ == "__main__":
    main()
