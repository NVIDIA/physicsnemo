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

"""Render scalar, vector, and rank-2 tensor fields before and after a rotation."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.collections import PolyCollection
from matplotlib.patches import Ellipse, Polygon
from matplotlib.tri import Triangulation
from PIL import Image

from physicsnemo.mesh.fields import FieldSchema
from physicsnemo.mesh.primitives.planar import unit_square

OUTPUT = Path(__file__).parent / "fields_rank_transforms.png"

ANGLE = math.radians(30.0)
EDGE_COLOR = "#9aa5ad"
FACE_COLOR = "#eef2f5"
VECTOR_COLOR = "#0072b2"
TENSOR_COLOR = "#d55e00"
LIMIT = 0.92
SQUARE = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]

plt.rcParams.update({"font.size": 13})


def _draw_mesh(axis, mesh, *, scalar=None, norm=None):
    """Draw the triangles, filled with a scalar field or a flat color."""

    points = mesh.points.numpy()
    cells = mesh.cells.numpy()
    if scalar is None:
        axis.add_collection(
            PolyCollection(
                points[cells],
                facecolors=FACE_COLOR,
                edgecolors=EDGE_COLOR,
                linewidths=0.8,
            )
        )
        return None
    triangulation = Triangulation(points[:, 0], points[:, 1], cells)
    image = axis.tripcolor(
        triangulation, scalar.numpy(), cmap="viridis", norm=norm, shading="gouraud"
    )
    axis.triplot(triangulation, color=EDGE_COLOR, linewidth=0.8)
    return image


def _draw_ellipses(axis, points, tensors, scale):
    """Draw each symmetric positive-definite tensor as an ellipse."""

    eigenvalues, eigenvectors = torch.linalg.eigh(tensors)
    for center, values, vectors in zip(points, eigenvalues, eigenvectors):
        major = vectors[:, 1]
        axis.add_patch(
            Ellipse(
                center.tolist(),
                width=2 * scale * float(values[1]),
                height=2 * scale * float(values[0]),
                angle=math.degrees(math.atan2(major[1], major[0])),
                facecolor=TENSOR_COLOR,
                edgecolor=TENSOR_COLOR,
                alpha=0.75,
                zorder=3,
            )
        )


def main():
    """Generate the documentation image."""

    mesh = unit_square.load(subdivisions=2).translate([-0.5, -0.5])
    x, y = mesh.points.unbind(dim=-1)
    mesh.point_data["pressure"] = x + y**2
    mesh.point_data["velocity"] = torch.stack([0.6 + 0.2 * y, 0.3 * x], dim=-1)
    stress = torch.zeros(mesh.n_points, 2, 2)
    stress[:, 0, 0] = 1.0 + 0.6 * x
    stress[:, 0, 1] = stress[:, 1, 0] = 0.25 * y
    stress[:, 1, 1] = 0.35
    mesh.point_data["stress"] = stress

    schema = FieldSchema.parse(
        {
            "pressure": {"rank": 0},
            "velocity": {"rank": 1},
            "stress": {"rank": 2, "symmetric": True},
        }
    )
    schema.check(mesh.point_data, label="point data")

    rotated = mesh.rotate(ANGLE, transform_point_data=True)
    schema.check(rotated.point_data, label="rotated point data")

    R = torch.tensor(
        [[math.cos(ANGLE), -math.sin(ANGLE)], [math.sin(ANGLE), math.cos(ANGLE)]]
    )
    torch.testing.assert_close(rotated.points, mesh.points @ R.T)
    torch.testing.assert_close(
        rotated.point_data["pressure"], mesh.point_data["pressure"]
    )
    torch.testing.assert_close(
        rotated.point_data["velocity"], mesh.point_data["velocity"] @ R.T
    )
    torch.testing.assert_close(rotated.point_data["stress"], R @ stress @ R.T)

    norm = plt.Normalize(
        float(mesh.point_data["pressure"].min()),
        float(mesh.point_data["pressure"].max()),
    )
    titles = (
        f"pressure\n{schema['pressure']}\n" r"unchanged: $p' = p$",
        f"velocity\n{schema['velocity']}\n" r"rotated: $v' = R\,v$",
        f"stress\n{schema['stress']}\n" r"rotated: $T' = R\,T\,R^{\mathsf{T}}$",
    )
    rows = (("Original", mesh), ("Rotated by 30°", rotated))
    interior = (mesh.points.abs() < 0.5 - 1e-6).all(dim=1)

    figure, axes = plt.subplots(2, 3, figsize=(12.0, 9.4))
    for row, (row_label, panel_mesh) in enumerate(rows):
        points = panel_mesh.points
        data = panel_mesh.point_data
        image = _draw_mesh(axes[row, 0], panel_mesh, scalar=data["pressure"], norm=norm)
        _draw_mesh(axes[row, 1], panel_mesh)
        axes[row, 1].quiver(
            points[:, 0].numpy(),
            points[:, 1].numpy(),
            data["velocity"][:, 0].numpy(),
            data["velocity"][:, 1].numpy(),
            color=VECTOR_COLOR,
            angles="xy",
            scale_units="xy",
            scale=3.5,
            # A thin shaft with a short head, so each arrow reads as an arrow.
            width=0.008,
            headwidth=3.0,
            headlength=3.5,
            headaxislength=3.0,
            # Center each arrow on its point, so the arrows of the right column
            # stay within the panel.
            pivot="middle",
            zorder=3,
        )
        _draw_mesh(axes[row, 2], panel_mesh)
        # Only interior vertices get an ellipse, so that every ellipse stays
        # inside the mesh.
        _draw_ellipses(
            axes[row, 2], points[interior], data["stress"][interior], scale=0.1
        )
        axes[row, 0].text(
            -0.06,
            0.5,
            row_label,
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=16,
        )
        for axis in axes[row]:
            if row == 1:
                axis.add_patch(
                    Polygon(
                        SQUARE,
                        fill=False,
                        edgecolor="#555555",
                        linestyle="--",
                        linewidth=1.0,
                    )
                )
            axis.set_xlim(-LIMIT, LIMIT)
            axis.set_ylim(-LIMIT, LIMIT)
            axis.set_aspect("equal")
            axis.set_axis_off()
    for axis, title in zip(axes[0], titles, strict=True):
        axis.set_title(title, fontsize=14)

    colorbar = figure.colorbar(
        image,
        cax=axes[1, 0].inset_axes([0.15, -0.06, 0.7, 0.04]),
        orientation="horizontal",
    )
    colorbar.set_label("pressure")
    figure.subplots_adjust(
        left=0.05, right=0.99, bottom=0.08, top=0.88, wspace=0.04, hspace=0.06
    )
    figure.savefig(OUTPUT, dpi=160, facecolor="white", bbox_inches="tight")
    # Drop the unused alpha channel to keep the file small.
    Image.open(OUTPUT).convert("RGB").save(OUTPUT, optimize=True)
    plt.close(figure)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
