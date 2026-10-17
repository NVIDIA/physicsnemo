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


"""Render batching by Mesh.merge and fixed or bucketed padding for torch.compile."""

from pathlib import Path

import _style as style
import matplotlib.pyplot as plt
import torch
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.planar import annulus_2d, l_shape
from physicsnemo.mesh.primitives.procedural import lumpy_sphere
from physicsnemo.mesh.primitives.surfaces import cylinder, sphere_icosahedral

OUTPUT = Path(__file__).parent / "batching_merge_pad.png"

POWER = 1.5
FIXED_N_POINTS = 700
COLORS = (style.SECOND, style.RESULT)
FILLS = ("#cfe6f5", "#f6d2bd")  # light fills of the two sample colors


def _merge_panel(axis, strip):
    """Merge two planar meshes and draw the result colored by sample."""

    a = annulus_2d.load(n_radial=3, n_angular=16)
    b = l_shape.load(size=1.6, subdivisions=3).translate(torch.tensor([1.35, -0.8]))
    batch = Mesh.merge([a, b])
    # The cells of b now index the points of b, which follow the points of a.
    if not torch.equal(batch.cells[a.n_cells :], b.cells + a.n_points):
        raise RuntimeError("Mesh.merge did not offset the cells of b")
    sizes = torch.tensor([a.n_points, b.n_points])
    node_to_graph = torch.repeat_interleave(torch.arange(2), sizes)

    points = batch.points.numpy()
    cell_to_graph = node_to_graph[batch.cells[:, 0]].numpy()
    axis.add_collection(
        PolyCollection(
            points[batch.cells.numpy()],
            facecolors=[FILLS[g] for g in cell_to_graph],
            edgecolors="white",
            linewidths=1.0,
        )
    )
    axis.scatter(
        points[:, 0],
        points[:, 1],
        s=45,
        c=[COLORS[g] for g in node_to_graph.numpy()],
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
    )
    axis.text(0.0, -1.22, f"a: {a.n_points} points", ha="center", color=COLORS[0])
    axis.text(
        2.15,
        -1.22,
        f"b: {b.n_points} points, cell indices + {a.n_points}",
        ha="center",
        color=COLORS[1],
    )
    axis.set_title(
        "Mesh.merge([a, b])\n"
        f"Mesh[2, 2], {batch.n_points} points, {batch.n_cells} cells"
    )
    style.clean_axis(axis, ((-1.1, 3.05), (-1.4, 1.1)))

    strip.imshow(
        node_to_graph[None, :].numpy(),
        cmap=ListedColormap(COLORS),
        aspect="auto",
        extent=(0, batch.n_points, 0, 1),
    )
    strip.set_xticks([0, a.n_points, batch.n_points])
    strip.set_yticks([])
    for side in ("top", "right", "left"):
        strip.spines[side].set_visible(False)
    strip.set_title("node_to_graph", fontsize=13)
    strip.set_xlabel("torch.repeat_interleave(torch.arange(2), sizes)")


def _padding_panel(axis):
    """Compare bucketed padding with one fixed size, for meshes of four sizes."""

    meshes = [
        ("sphere_icosahedral\nsubdivisions=1", sphere_icosahedral.load(subdivisions=1)),
        ("lumpy_sphere\nsubdivisions=2", lumpy_sphere.load(subdivisions=2)),
        ("cylinder", cylinder.load()),
        ("sphere_icosahedral\nsubdivisions=3", sphere_icosahedral.load(subdivisions=3)),
    ]
    if FIXED_N_POINTS < max(mesh.n_points for _, mesh in meshes):
        raise RuntimeError("FIXED_N_POINTS must fit the largest mesh")
    for row, (_, mesh) in enumerate(meshes):
        padded = mesh.pad_to_next_power(power=POWER)
        axis.barh(row, mesh.n_points, color=style.SECOND_FILL, height=0.6)
        axis.barh(
            row,
            padded.n_points - mesh.n_points,
            left=mesh.n_points,
            color="white",
            edgecolor=style.SECOND,
            hatch="///",
            linewidth=0.8,
            height=0.6,
        )
        # One column of labels to the right of the fixed size.
        axis.text(
            FIXED_N_POINTS * 1.12,
            row,
            f"{mesh.n_points} → {padded.n_points}",
            va="center",
        )
    axis.axvline(FIXED_N_POINTS, color=style.RESULT, linestyle="--", linewidth=2.0)

    buckets = [int(POWER**n) for n in range(9, 17)]
    axis.set_xscale("log")
    axis.set_xticks(buckets, [str(b) for b in buckets], fontsize=11)
    axis.minorticks_off()
    axis.grid(axis="x", color=style.FAINT_EDGE, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_xlim(30, 1900)
    axis.set_xlabel(f"points, with bucket edges floor({POWER}^n)")
    axis.set_yticks(range(len(meshes)), [name for name, _ in meshes], fontsize=11)
    axis.invert_yaxis()
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    # End the axis line past the fixed size, short of the label column.
    axis.spines["bottom"].set_bounds(30, 800)
    axis.set_title(f"mesh.pad_to_next_power(power={POWER})\npoints per mesh")
    return [
        Patch(color=style.SECOND_FILL),
        Patch(facecolor="white", edgecolor=style.SECOND, hatch="///"),
        Line2D([], [], color=style.RESULT, linestyle="--", linewidth=2.0),
    ], [
        "points",
        "padding to the next bucket",
        f"pad(target_n_points={FIXED_N_POINTS})",
    ]


def main():
    """Generate the documentation image."""

    figure = plt.figure(figsize=(13.5, 5.8))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 1.05),
        height_ratios=(1.0, 0.05),
        wspace=0.42,
        hspace=0.5,
    )
    _merge_panel(figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[1, 0]))
    handles, labels = _padding_panel(figure.add_subplot(grid[:, 1]))
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.subplots_adjust(left=0.03, right=0.98, bottom=0.2, top=0.88)
    style.save(figure, OUTPUT)


if __name__ == "__main__":
    main()
