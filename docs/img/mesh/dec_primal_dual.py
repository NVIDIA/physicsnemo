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


"""Show the stencils behind LSQ and DEC derivatives: a vertex's neighbors, and its circumcentric dual cell."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from _style import (
    EDGE,
    FACE,
    FIXED,
    RESULT,
    SECOND,
    TEXT,
    clean_axis,
    save,
)
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.geometry import (
    compute_circumcenters,
    compute_cotan_weights_fem,
    compute_dual_volumes_0,
)
from physicsnemo.mesh.primitives.planar import equilateral_triangle

OUTPUT = Path(__file__).parent / "dec_primal_dual.png"

DOT = dict(markeredgecolor="white", markeredgewidth=1.2, zorder=5)


def _patch():
    """An equilateral triangle patch with its interior vertices jittered."""

    base = equilateral_triangle.load(subdivisions=3)
    on_boundary = torch.zeros(base.n_points, dtype=torch.bool)
    on_boundary[base.get_boundary_mesh().cells.unique()] = True
    generator = torch.Generator().manual_seed(0)
    spacing = float(base.cell_areas.mean().mul(4 / math.sqrt(3)).sqrt())
    noise = (torch.rand(base.n_points, 2, generator=generator) - 0.5) * 0.25 * spacing
    points = base.points + noise * (~on_boundary).unsqueeze(-1)
    return Mesh(points=points, cells=base.cells), on_boundary


def _draw_mesh(axis, mesh):
    """Draw the triangles as a neutral background."""

    axis.add_collection(
        PolyCollection(
            mesh.points[mesh.cells].numpy(),
            facecolors=FACE,
            edgecolors=EDGE,
            linewidths=1.0,
        )
    )


def main():
    """Generate the documentation image."""

    mesh, on_boundary = _patch()
    # The interior vertex nearest the centroid of the patch.
    distance = (mesh.points - mesh.points.mean(dim=0)).norm(dim=-1)
    distance[on_boundary] = math.inf
    vertex = int(distance.argmin())
    center = mesh.points[vertex]

    ### LSQ stencil: the vertex's neighbors.
    neighbors = torch.tensor(mesh.get_point_to_points_adjacency().to_list()[vertex])

    ### DEC stencil: the circumcenters of the incident triangles, in angular order.
    circumcenters = compute_circumcenters(mesh.points[mesh.cells])
    incident = torch.tensor(mesh.get_point_to_cells_adjacency().to_list()[vertex])
    offsets = circumcenters[incident] - center
    order = torch.atan2(offsets[:, 1], offsets[:, 0]).argsort()
    incident = incident[order]
    dual_cell = circumcenters[incident]
    dual_area = float(compute_dual_volumes_0(mesh)[vertex])
    x, y = dual_cell.unbind(dim=-1)
    shoelace = 0.5 * float((x * y.roll(-1) - x.roll(-1) * y).sum())
    # For acute triangles, the dual cell is the polygon of the circumcenters.
    torch.testing.assert_close(shoelace, dual_area, rtol=1e-4, atol=0.0)

    # One primal edge (vertex, neighbor) and its dual edge, which joins the
    # circumcenters of the two triangles that share it.
    neighbor = int(neighbors[(mesh.points[neighbors] - center)[:, 0].argmax()])
    sharing = incident[(mesh.cells[incident] == neighbor).any(dim=-1)]
    dual_edge = circumcenters[sharing]
    weights, edges = compute_cotan_weights_fem(mesh)
    match = ((edges == vertex) | (edges == neighbor)).all(dim=-1)
    weight = float(weights[match][0])
    ratio = float(
        (dual_edge[0] - dual_edge[1]).norm() / (mesh.points[neighbor] - center).norm()
    )
    # The cotangent weight is the ratio of the dual and primal edge lengths.
    torch.testing.assert_close(ratio, weight, rtol=1e-4, atol=0.0)

    points = mesh.points.numpy()
    # Zoom in on the stencil, two rings of cells around the vertex.
    reach = 2.3 * float((mesh.points[neighbors] - center).norm(dim=-1).mean())
    cx, cy = center.tolist()
    limits = ((cx - reach, cx + reach), (cy - 0.8 * reach, cy + 0.8 * reach))

    figure, (lsq_axis, dec_axis) = plt.subplots(1, 2, figsize=(13.0, 5.9))

    _draw_mesh(lsq_axis, mesh)
    for target in neighbors.tolist():
        lsq_axis.annotate(
            "",
            xy=points[target],
            xytext=points[vertex],
            arrowprops=dict(
                arrowstyle="-|>",
                color=SECOND,
                linewidth=2.2,
                mutation_scale=18,
                shrinkA=0,
                shrinkB=7,
            ),
        )
    lsq_axis.plot(
        *points[neighbors.numpy()].T,
        "o",
        color=SECOND,
        markersize=13,
        linestyle="none",
        **DOT,
    )
    lsq_axis.plot(*points[vertex], "o", color=RESULT, markersize=16, **DOT)
    lsq_axis.set_title(
        f'method="lsq": get_point_to_points_adjacency()\n'
        f"fit a linear function to {len(neighbors)} neighbors"
    )

    _draw_mesh(dec_axis, mesh)
    dec_axis.add_patch(
        Polygon(
            dual_cell.numpy(),
            closed=True,
            facecolor=RESULT,
            alpha=0.22,
            edgecolor="none",
        )
    )
    dec_axis.add_patch(
        Polygon(
            dual_cell.numpy(), closed=True, fill=False, edgecolor=RESULT, linewidth=2.0
        )
    )
    dec_axis.plot(
        *points[[vertex, neighbor]].T,
        color=SECOND,
        linewidth=4.0,
        solid_capstyle="round",
        zorder=3,
    )
    dec_axis.plot(
        *dual_edge.numpy().T,
        color=FIXED,
        linewidth=4.0,
        solid_capstyle="round",
        zorder=4,
    )
    dec_axis.plot(
        *circumcenters.numpy().T,
        "o",
        color=TEXT,
        markersize=6,
        linestyle="none",
        zorder=4,
    )
    dec_axis.plot(*points[vertex], "o", color=RESULT, markersize=16, **DOT)
    dec_axis.plot(*points[neighbor], "o", color=SECOND, markersize=13, **DOT)
    midpoint = dual_edge.mean(dim=0).numpy()
    dec_axis.annotate(
        rf"$w_{{ij}} = {weight:.2f}$",
        xy=midpoint,
        xytext=(midpoint[0] + 0.035, midpoint[1] + 0.105),
        color=FIXED,
        fontsize=15,
        arrowprops=dict(arrowstyle="-", color=FIXED, linewidth=1.2, shrinkB=2),
    )
    dec_axis.set_title(
        'method="dec": circumcentric dual mesh\n'
        rf"dual cell area $|{{\star}}v_i| = {dual_area:.4f}$, "
        r"$w_{ij} = |{\star}e_{ij}|\,/\,|e_{ij}|$"
    )

    for axis in (lsq_axis, dec_axis):
        clean_axis(axis, limits)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=RESULT,
            linestyle="none",
            markersize=12,
            label="vertex i",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color=SECOND,
            linestyle="none",
            markersize=10,
            label="neighbors j",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color=TEXT,
            linestyle="none",
            markersize=5,
            label="circumcenters",
        ),
        Patch(facecolor=RESULT, alpha=0.3, edgecolor=RESULT, label="dual cell of i"),
        Line2D([], [], color=SECOND, linewidth=4, label="primal edge"),
        Line2D([], [], color=FIXED, linewidth=4, label="dual edge"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.4,
    )
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.1, top=0.88, wspace=0.05)
    save(figure, OUTPUT)


if __name__ == "__main__":
    main()
