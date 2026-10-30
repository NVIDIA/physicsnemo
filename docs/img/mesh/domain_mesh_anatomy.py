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

"""Draw the potential-flow cylinder DomainMesh of the User Guide and its three fields."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from _style import (
    DIVERGING,
    EDGE,
    RESULT,
    SECOND,
    TEXT,
    clean_axis,
    mesh_label,
    save,
)
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
from matplotlib.tri import Triangulation

from physicsnemo.mesh import DomainMesh
from physicsnemo.mesh.primitives.planar import annulus_2d

OUTPUT = Path(__file__).parent / "domain_mesh_anatomy.png"

GLOBAL = TEXT  # the freestream arrow and the global_data box
MONO = {"family": "DejaVu Sans Mono"}


def _build():
    """The code of the "Building a Domain" section of the User Guide, verbatim."""

    interior = annulus_2d.load(
        inner_radius=0.5, outer_radius=2.0, n_radial=8, n_angular=64
    )

    ### Potential flow around a cylinder of radius a = 0.5, freestream U_inf = (1, 0)
    x, y = interior.points.unbind(-1)
    r2 = x**2 + y**2
    U = torch.stack([1 - 0.25 * (x**2 - y**2) / r2**2, -0.5 * x * y / r2**2], dim=-1)
    interior.point_data["U"] = U
    interior.point_data["C_p"] = 1 - (U**2).sum(dim=-1)

    ### Split the boundary of the annulus by the radius of each edge
    boundary = interior.get_boundary_mesh()
    radius = boundary.cell_centroids.norm(dim=-1)
    wall = boundary.slice_cells(radius < 1.0).clean()
    farfield = boundary.slice_cells(radius > 1.0).clean()

    return DomainMesh(
        interior=interior,
        boundaries={"no_slip": wall, "freestream": farfield},
        global_data={"U_inf": torch.tensor([1.0, 0.0])},
    )


def _segments(mesh):
    """The edges of a Mesh[1, 2] as an array of segments."""

    return mesh.points.numpy()[mesh.cells.numpy()]


def _fields(data):
    """List the fields of a TensorDict with their per-row shapes."""

    return ", ".join(
        f"{key} {tuple(data[key].shape[1:])}" for key in sorted(data.keys())
    )


def _box(axis, x, y, width, height, color, lines, fill="white"):
    """Draw a rounded box with left-aligned lines of text."""

    axis.add_patch(
        FancyBboxPatch(
            (x, y - height),
            width,
            height,
            boxstyle="round,pad=0.0,rounding_size=0.12",
            facecolor=fill,
            edgecolor=color,
            linewidth=2.2,
        )
    )
    for i, (text, style) in enumerate(lines):
        axis.text(x + 0.22, y - 0.38 - 0.42 * i, text, va="center", **style)


def _draw_domain(axis, dm):
    """The interior colored by C_p, the boundaries, and the freestream arrow."""

    interior = dm.interior
    points = interior.points.numpy()
    triangulation = Triangulation(points[:, 0], points[:, 1], interior.cells.numpy())
    c_p = interior.point_data["C_p"].numpy()
    shading = axis.tripcolor(
        triangulation,
        c_p,
        shading="gouraud",
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vmin=float(c_p.min()), vcenter=0.0, vmax=float(c_p.max())),
    )
    axis.triplot(triangulation, color="white", linewidth=0.35, alpha=0.6)

    # A sparse quiver of U: every sixth angle on three of the inner rings.
    radius = interior.points.norm(dim=-1).round(decimals=4)
    ring = torch.bucketize(radius, torch.unique(radius))
    angle = torch.atan2(interior.points[:, 1], interior.points[:, 0])
    k = (angle / (2 * torch.pi / 64)).round().long()
    sparse = torch.isin(ring, torch.tensor([1, 3, 5])) & ((k + ring) % 6 == 0)
    U = interior.point_data["U"][sparse].numpy()
    axis.quiver(
        points[sparse.numpy(), 0],
        points[sparse.numpy(), 1],
        U[:, 0],
        U[:, 1],
        color=TEXT,
        scale=14.0,
        width=0.005,
        zorder=3,
    )

    colors = {"no_slip": RESULT, "freestream": SECOND}
    for key in dm.boundaries.keys():
        axis.add_collection(
            LineCollection(
                _segments(dm.boundaries[key]),
                colors=colors[key],
                linewidths=4.0,
                capstyle="round",
                zorder=4,
            )
        )

    # The freestream velocity, a global_data vector, enters from the left.
    u_inf = dm.global_data["U_inf"].numpy()
    start = (-2.95, 0.0)
    axis.annotate(
        "",
        xy=(start[0] + 0.75 * u_inf[0], start[1] + 0.75 * u_inf[1]),
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": GLOBAL,
            "lw": 2.6,
            "mutation_scale": 22,
        },
    )
    axis.text(-2.58, 0.18, "U_inf", ha="center", va="bottom", color=GLOBAL, **MONO)
    axis.text(0.0, 0.0, "no_slip", ha="center", va="center", color=RESULT, **MONO)
    axis.text(1.45, 1.62, "freestream", ha="left", va="bottom", color=SECOND, **MONO)
    clean_axis(axis, ((-3.05, 2.1), (-2.1, 2.1)))
    return shading


def _draw_tree(axis, dm):
    """The three fields of the DomainMesh as a tree of boxes."""

    interior = dm.interior
    axis.set_xlim(0.0, 9.6)
    axis.set_ylim(0.95, 8.0)
    axis.set_axis_off()

    axis.text(0.1, 7.7, "dm = DomainMesh(...)", va="center", fontsize=15, **MONO)
    trunk, left, width = 0.45, 3.45, 6.05

    def branch(y, name, color, targets):
        """A branch from the trunk to a field name, and on to its boxes."""

        axis.plot([trunk, trunk + 0.4], [y, y], color=EDGE, linewidth=1.6)
        axis.text(trunk + 0.55, y, name, va="center", color=color, **MONO)
        for target in targets:
            axis.plot(
                [3.0, 3.2, 3.2, left],
                [y, y, target, target],
                color=EDGE,
                linewidth=1.6,
            )

    # Each box: (top, height, border color, lines of text).
    branch(6.7, "interior", TEXT, [6.7])
    _box(
        axis,
        left,
        7.32,
        width,
        1.25,
        EDGE,
        [
            (mesh_label(interior, "triangles") + f", {interior.n_points} points", {}),
            (f"point_data: {_fields(interior.point_data)}", MONO),
        ],
    )

    keys = list(dm.boundaries.keys())
    tops = [5.55 - 1.7 * i for i in range(len(keys))]
    branch(3.9, "boundaries", TEXT, [top - 0.8 for top in tops])
    for key, top in zip(keys, tops):
        color = {"no_slip": RESULT, "freestream": SECOND}[key]
        patch = dm.boundaries[key]
        _box(
            axis,
            left,
            top,
            width,
            1.6,
            color,
            [
                (f'"{key}"', {"color": color, **MONO}),
                (mesh_label(patch, "edges") + f", {patch.n_points} points", {}),
                (f"point_data: {_fields(patch.point_data)}", MONO),
            ],
        )

    branch(1.53, "global_data", GLOBAL, [1.53])
    _box(
        axis, left, 1.95, width, 0.85, GLOBAL, [(_fields_global(dm.global_data), MONO)]
    )
    axis.plot([trunk, trunk], [7.45, 1.53], color=EDGE, linewidth=1.6)


def _fields_global(data):
    """List global_data entries with their full shapes."""

    return ", ".join(f"{key}: {tuple(data[key].shape)}" for key in sorted(data.keys()))


def main():
    """Generate the documentation image."""

    dm = _build()

    figure = plt.figure(figsize=(14.0, 7.1))
    domain_axis = figure.add_axes((0.01, 0.13, 0.5, 0.76))
    tree_axis = figure.add_axes((0.53, 0.06, 0.46, 0.83))

    shading = _draw_domain(domain_axis, dm)
    domain_axis.set_title(
        "potential flow around a cylinder\n"
        f"interior {mesh_label(dm.interior, 'triangles')}"
    )
    colorbar_axis = figure.add_axes((0.14, 0.08, 0.24, 0.025))
    colorbar = figure.colorbar(shading, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label('interior.point_data["C_p"]', **MONO)

    _draw_tree(tree_axis, dm)
    save(figure, OUTPUT)


if __name__ == "__main__":
    main()
