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


"""Pair five mesh defects with the call that detects each one and the call that repairs it."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from _style import (
    EDGE,
    FACE,
    RESULT,
    SECOND,
    SECOND_FILL,
    TEXT,
    clean_axis,
    save,
)
from matplotlib.collections import PolyCollection

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.planar import unit_square
from physicsnemo.mesh.repair import (
    clean_mesh,
    fill_holes,
    fix_orientation,
    remove_degenerate_cells,
    remove_isolated_points,
)
from physicsnemo.mesh.validation import compute_mesh_statistics, validate

OUTPUT = Path(__file__).parent / "mesh_defects.png"

LIMITS = ((-0.12, 1.12), (-0.12, 1.12))
REVERSED_FILL = "#f6c7a6"  # a light vermillion, so that the winding arrow stays visible
DOT = dict(markeredgecolor="white", markeredgewidth=1.2, linestyle="none", zorder=5)


def _grid():
    """A 5 x 5 vertex grid on the unit square, 32 triangles."""

    return unit_square.load(subdivisions=2)


def _draw(axis, mesh, facecolors=FACE):
    """Draw the cells of a mesh, seen from above."""

    axis.add_collection(
        PolyCollection(
            mesh.points[mesh.cells][..., :2].numpy(),
            facecolors=facecolors,
            edgecolors=EDGE,
            linewidths=1.0,
        )
    )


def _winding(axis, mesh, highlight=None):
    """Draw a circular arrow inside each cell, in the order of its vertices."""

    corners = mesh.points[mesh.cells][..., :2]
    centers = corners.mean(dim=1)
    edge_a = corners[:, 1] - corners[:, 0]
    edge_b = corners[:, 2] - corners[:, 0]
    signs = torch.sign(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
    for cell, (center, sign) in enumerate(zip(centers.tolist(), signs.tolist())):
        flagged = cell == highlight
        color = RESULT if flagged else "#7d8a94"
        radius = 0.057
        angles = [math.pi / 2 + sign * 1.6 * math.pi * k / 24 for k in range(25)]
        xs = [center[0] + radius * math.cos(a) for a in angles]
        ys = [center[1] + radius * math.sin(a) for a in angles]
        axis.plot(xs[:-1], ys[:-1], color=color, linewidth=2.4 if flagged else 1.5)
        axis.annotate(
            "",
            xy=(xs[-1], ys[-1]),
            xytext=(xs[-3], ys[-3]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                linewidth=0,
                mutation_scale=15 if flagged else 11,
            ),
        )


def _duplicates():
    """Split the grid down x = 0.5 so that the right half has its own copy of the seam points."""

    mesh = _grid()
    points, cells = mesh.points, mesh.cells.clone()
    seam = torch.nonzero(torch.isclose(points[:, 0], torch.tensor(0.5))).flatten()
    copies = torch.arange(len(seam)) + mesh.n_points
    right = mesh.cell_centroids[:, 0] > 0.5
    remap = torch.arange(mesh.n_points)
    remap[seam] = copies
    cells[right] = remap[cells[right]]
    broken = Mesh(points=torch.cat([points, points[seam]]), cells=cells)

    report = validate(broken)
    pairs = report["duplicate_vertex_pairs"]
    fixed, stats = clean_mesh(broken)
    if fixed.n_points != mesh.n_points:
        raise RuntimeError(f"clean_mesh left {fixed.n_points} points")
    right_faces = [SECOND_FILL if r else FACE for r in right.tolist()]

    def before(axis):
        _draw(axis, broken, facecolors=right_faces)
        ring = broken.points[pairs[:, 0]].numpy()
        axis.plot(
            *ring.T,
            "o",
            markersize=16,
            markerfacecolor="none",
            markeredgecolor=SECOND,
            markeredgewidth=2.4,
            linestyle="none",
            zorder=5,
        )

    def after(axis):
        _draw(axis, fixed)
        axis.plot(*fixed.points[seam].numpy().T, "o", color=SECOND, markersize=9, **DOT)

    return (
        (
            "duplicate points",
            "validate()",
            f"n_duplicate_vertices: {report['n_duplicate_vertices']}",
            before,
        ),
        (
            "clean_mesh()",
            f"n_points {stats['n_points_before_merge']} → {fixed.n_points}",
            after,
        ),
    )


def _degenerate():
    """Add a zero-area cell whose three corners lie on one grid line."""

    mesh = _grid()
    line = torch.nonzero(torch.isclose(mesh.points[:, 1], torch.tensor(0.5))).flatten()
    line = line[mesh.points[line, 0].argsort()][1:4]
    broken = Mesh(points=mesh.points, cells=torch.cat([mesh.cells, line.unsqueeze(0)]))

    report = validate(broken)
    bad = report["degenerate_cell_indices"]
    fixed, stats = remove_degenerate_cells(broken)

    def before(axis):
        _draw(axis, broken)
        for cell in bad.tolist():
            corners = broken.points[broken.cells[cell]].numpy()
            axis.plot(
                *corners.T, color=RESULT, linewidth=6, solid_capstyle="round", zorder=4
            )
            axis.plot(*corners.T, "o", color=RESULT, markersize=9, **DOT)

    return (
        (
            "zero-area cell",
            "validate()",
            f"n_degenerate_cells: {report['n_degenerate_cells']}",
            before,
        ),
        (
            "remove_degenerate_cells()",
            f"n_cells {stats['n_cells_original']} → {stats['n_cells_final']}",
            lambda axis: _draw(axis, fixed),
        ),
    )


def _isolated():
    """Add three points that no cell uses."""

    mesh = _grid()
    extra = torch.tensor([[0.13, 1.06], [1.06, 0.62], [0.55, -0.07]])
    broken = Mesh(points=torch.cat([mesh.points, extra]), cells=mesh.cells)

    report = compute_mesh_statistics(broken)
    fixed, stats = remove_isolated_points(broken)

    def before(axis):
        _draw(axis, broken)
        axis.plot(*extra.numpy().T, "o", color=RESULT, markersize=11, **DOT)

    return (
        (
            "unused points",
            "compute_mesh_statistics()",
            f"n_isolated_vertices: {report['n_isolated_vertices']}",
            before,
        ),
        (
            "remove_isolated_points()",
            f"n_points {stats['n_points_original']} → {stats['n_points_final']}",
            lambda axis: _draw(axis, fixed),
        ),
    )


def _reversed():
    """Lift the grid into a curved Mesh[2, 3] and reverse the vertex order of one cell."""

    mesh = _grid()
    x, y = mesh.points.unbind(dim=-1)
    z = 0.4 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)
    flipped = 13
    # unit_square winds its cells clockwise. Reverse all of them, so that the
    # normals point up, then reverse one cell back.
    cells = mesh.cells[:, [0, 2, 1]]
    cells[flipped] = mesh.cells[flipped]
    broken = Mesh(points=torch.stack([x, y, z], dim=-1), cells=cells)

    # The normal of a reversed cell points down, against its neighbors.
    downward = torch.nonzero(broken.cell_normals[:, 2] < 0).flatten()
    if downward.tolist() != [flipped]:
        raise RuntimeError(f"downward normals at cells {downward.tolist()}")
    fixed, stats = fix_orientation(broken)
    if not bool((fixed.cell_normals[:, 2] > 0).all()):
        raise RuntimeError("fix_orientation left a downward normal")
    faces = [
        REVERSED_FILL if cell == flipped else FACE for cell in range(broken.n_cells)
    ]

    def before(axis):
        _draw(axis, broken, facecolors=faces)
        _winding(axis, broken, highlight=flipped)

    def after(axis):
        _draw(axis, fixed)
        _winding(axis, fixed)

    return (
        (
            "reversed cell, Mesh[2, 3]",
            "cell_normals",
            f"{len(downward)} normal points down",
            before,
        ),
        ("fix_orientation()", f"n_cells_flipped: {stats['n_cells_flipped']}", after),
    )


def _hole():
    """Remove the two cells of one grid square to open a hole."""

    mesh = _grid()
    centroids = mesh.cell_centroids
    inside = ((centroids - torch.tensor([0.375, 0.625])).abs() < 0.125).all(dim=-1)
    broken = mesh.slice_cells(~inside)

    boundary = broken.get_boundary_mesh()
    fixed, stats = fill_holes(broken, max_hole_edges=10)
    # The outer loop has 16 edges, more than max_hole_edges, so it is the one skipped.
    if (stats["n_holes_filled"], stats["n_holes_skipped"]) != (1, 1):
        raise RuntimeError(f"fill_holes filled or skipped other loops: {stats}")
    added = [SECOND_FILL if i >= broken.n_cells else FACE for i in range(fixed.n_cells)]

    def before(axis):
        _draw(axis, broken)
        for edge in boundary.points[boundary.cells].numpy():
            axis.plot(
                *edge.T, color=RESULT, linewidth=3.2, solid_capstyle="round", zorder=4
            )

    def after(axis):
        _draw(axis, fixed, facecolors=added)

    return (
        ("hole", "get_boundary_mesh()", f"{boundary.n_cells} boundary edges", before),
        (
            "fill_holes(max_hole_edges=10)",
            f"{stats['n_holes_filled']} filled, {stats['n_holes_skipped']} skipped (outer loop)",
            after,
        ),
    )


def main():
    """Generate the documentation image."""

    columns = [_duplicates(), _degenerate(), _isolated(), _reversed(), _hole()]
    figure, axes = plt.subplots(2, 5, figsize=(14.0, 6.9))
    for column, (defect, repair) in enumerate(columns):
        name, call, finding, draw_before = defect
        repair_call, repair_result, draw_after = repair
        draw_before(axes[0, column])
        draw_after(axes[1, column])
        axes[0, column].set_title(f"{name}\n{call}\n{finding}", fontsize=12.5)
        axes[1, column].set_title(f"{repair_call}\n{repair_result}", fontsize=12.5)
        for axis in axes[:, column]:
            clean_axis(axis, LIMITS)
    for row, label in enumerate(("detect", "repair")):
        axes[row, 0].text(
            -0.08,
            0.5,
            label,
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=16,
            color=TEXT,
        )
    figure.subplots_adjust(
        left=0.03, right=0.995, bottom=0.01, top=0.86, wspace=0.08, hspace=0.42
    )
    save(figure, OUTPUT, tight=True)


if __name__ == "__main__":
    main()
