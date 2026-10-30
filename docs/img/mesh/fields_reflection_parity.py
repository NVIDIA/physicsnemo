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


"""Render a scalar, a vector, and a pseudoscalar field before and after the mirror y -> -y."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from _style import (
    DIVERGING,
    EDGE,
    FACE,
    SECOND,
    SEQUENTIAL,
    TEXT,
    clean_axis,
    save,
)
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.tri import Triangulation

from physicsnemo.mesh.fields import FieldSchema
from physicsnemo.mesh.primitives.planar import unit_square

OUTPUT = Path(__file__).parent / "fields_reflection_parity.png"

GAP = 0.12  # distance from the mirror line to the mesh
LIMITS = ((-0.62, 0.68), (-1.0 - GAP - 0.12, 1.0 + GAP + 0.14))


def _draw(axis, mesh, field=None, cmap=None, norm=None):
    """Draw the triangles, filled with a scalar field or a flat color."""

    points = mesh.points.numpy()
    cells = mesh.cells.numpy()
    if field is None:
        axis.add_collection(
            PolyCollection(
                points[cells], facecolors=FACE, edgecolors=EDGE, linewidths=0.8
            )
        )
        return None
    triangulation = Triangulation(points[:, 0], points[:, 1], cells)
    image = axis.tripcolor(
        triangulation, field.numpy(), cmap=cmap, norm=norm, shading="gouraud"
    )
    axis.triplot(triangulation, color=EDGE, linewidth=0.8)
    return image


def main():
    """Generate the documentation image."""

    # A unit square just above the mirror line y = 0.
    mesh = unit_square.load(subdivisions=2).translate([-0.5, GAP])
    x, y = mesh.points.unbind(dim=-1)
    mesh.point_data["pressure"] = x + 0.8 * y
    # The velocity (0.55, 0.25 + 0.5 x^2) has the vorticity w = dv_y/dx - dv_x/dy = x.
    mesh.point_data["velocity"] = torch.stack(
        [torch.full_like(x, 0.55), 0.25 + 0.5 * x**2], dim=-1
    )
    mesh.point_data["vorticity"] = x.clone()

    schema = FieldSchema.parse(
        {
            "pressure": {"rank": 0},
            "velocity": {"rank": 1},
            "vorticity": {"rank": 0, "parity": "odd"},
        }
    )
    schema.check(mesh.point_data, label="point data")

    M = torch.diag(torch.tensor([1.0, -1.0]))  # y -> -y
    mirrored = mesh.transform(M, transform_point_data={"velocity": True})
    # transform() treats every field as a true tensor, so apply det(M) by hand.
    mirrored.point_data["vorticity"] = (
        torch.linalg.det(M) * mesh.point_data["vorticity"]
    )

    torch.testing.assert_close(mirrored.points, mesh.points @ M.T)
    torch.testing.assert_close(
        mirrored.point_data["pressure"], mesh.point_data["pressure"]
    )
    torch.testing.assert_close(
        mirrored.point_data["velocity"], mesh.point_data["velocity"] @ M.T
    )

    pressure_norm = plt.Normalize(
        float(mesh.point_data["pressure"].min()),
        float(mesh.point_data["pressure"].max()),
    )
    w_max = float(mesh.point_data["vorticity"].abs().max())
    vorticity_norm = TwoSlopeNorm(vcenter=0.0, vmin=-w_max, vmax=w_max)
    titles = (
        f"pressure\n{schema['pressure']}\n" r"unchanged: $p' = p$",
        f"velocity\n{schema['velocity']}\n" r"transform(): $v' = M\,v$",
        f"vorticity\n{schema['vorticity']}\n"
        r"by hand: $\omega' = \det(M)\,\omega$",
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 8.6))
    images = {}
    for panel_mesh in (mesh, mirrored):
        data = panel_mesh.point_data
        images["pressure"] = _draw(
            axes[0], panel_mesh, data["pressure"], SEQUENTIAL, pressure_norm
        )
        _draw(axes[1], panel_mesh)
        points = panel_mesh.points
        axes[1].quiver(
            points[:, 0].numpy(),
            points[:, 1].numpy(),
            data["velocity"][:, 0].numpy(),
            data["velocity"][:, 1].numpy(),
            color=SECOND,
            angles="xy",
            scale_units="xy",
            scale=4.0,
            width=0.012,
            zorder=3,
        )
        images["vorticity"] = _draw(
            axes[2], panel_mesh, data["vorticity"], DIVERGING, vorticity_norm
        )

    for axis, title in zip(axes, titles, strict=True):
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.2)
        axis.set_title(title, fontsize=14)
        clean_axis(axis, LIMITS)
    axes[1].text(
        0.06,
        0.0,
        "mirror y = 0",
        ha="center",
        va="center",
        color=TEXT,
        backgroundcolor="white",
    )
    for label, y_center in (("original", 0.5 + GAP), ("mirrored", -0.5 - GAP)):
        axes[0].text(
            -0.6,
            y_center,
            label,
            rotation=90,
            ha="right",
            va="center",
            fontsize=16,
        )

    for axis, key in ((axes[0], "pressure"), (axes[2], "vorticity")):
        colorbar = figure.colorbar(
            images[key],
            cax=axis.inset_axes([0.15, -0.05, 0.7, 0.025]),
            orientation="horizontal",
        )
        colorbar.set_label(key)
    figure.subplots_adjust(left=0.04, right=0.99, bottom=0.08, top=0.86, wspace=0.06)
    save(figure, OUTPUT, tight=True, full_color=True)


if __name__ == "__main__":
    main()
