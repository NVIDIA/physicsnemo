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

"""Show how embed, extrude, and project change the dimensions of a mesh."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from _style import (
    EDGE,
    FACE,
    FAINT_EDGE,
    SEQUENTIAL,
    SURFACE,
    SURFACE_EDGE,
    TEXT,
    clean_axis,
    mesh_label,
    render,
    save,
    spheres,
    tubes,
)
from matplotlib.collections import LineCollection, PolyCollection

from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives.curves import circle_2d
from physicsnemo.mesh.primitives.planar import unit_square
from physicsnemo.mesh.primitives.surfaces import plane
from physicsnemo.mesh.projections import embed, extrude, project

OUTPUT = Path(__file__).parent / "projections_embed_extrude.png"

SHADED = {"color": SURFACE, "show_edges": True, "edge_color": SURFACE_EDGE}


def _draw_planar(axis, mesh):
    """Draw a planar triangle mesh or a planar curve in matplotlib."""

    points = mesh.points.numpy()
    segments = points[mesh.cells.numpy()]
    if mesh.n_manifold_dims == 2:
        axis.add_collection(
            PolyCollection(segments, facecolors=FACE, edgecolors=EDGE, linewidths=1.0)
        )
    else:
        axis.add_collection(
            LineCollection(segments, colors=SURFACE_EDGE, linewidths=2.2)
        )
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=22,
            color=SURFACE_EDGE,
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )
    low, high = points.min(axis=0), points.max(axis=0)
    pad = 0.12 * float((high - low).max())
    clean_axis(axis, ((low[0] - pad, high[0] + pad), (low[1] - pad, high[1] + pad)))


def _arrow(axis, start, end):
    """Draw an arrow in the axes-fraction coordinates of ``axis``."""

    axis.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>,head_width=0.4", "color": TEXT, "lw": 2.0},
        annotation_clip=False,
    )


def main():
    """Generate the documentation image."""

    # Row 1: a planar surface gains a spatial axis, then a manifold dimension.
    square = unit_square.load(subdivisions=1)
    square_3d = embed(square, 3)
    prism = extrude(square_3d, vector=[0.0, 0.0, 0.5])

    # Row 2: a planar curve. extrude needs a 3D vector here, so embed comes first.
    curve = circle_2d.load(n_points=24)
    curve_3d = embed(curve, 3)
    tube = extrude(curve_3d, vector=[0.0, 0.0, 1.0])

    # Column 4: a curved sheet loses its height under project.
    sheet = plane.load(size=2.0, subdivisions=16)
    xy = sheet.points[:, :2]
    height = 0.55 * torch.exp(-(xy - torch.tensor([0.2, -0.1])).square().sum(-1) / 0.3)
    sheet = sheet.with_points(
        sheet.points + height[:, None] * torch.tensor([0.0, 0.0, 1.0])
    )
    shadow = project(sheet, keep_dims=[0, 1])

    sheet_pv = to_pyvista(sheet)
    sheet_pv.point_data["z"] = sheet.points[:, 2].numpy()
    images = {
        "square_3d": render([(to_pyvista(square_3d), SHADED)], zoom=1.15),
        "prism": render([(to_pyvista(prism).shrink(0.8), SHADED)], zoom=0.95),
        "curve_3d": render(
            [
                (tubes(to_pyvista(curve_3d), 0.022), {"color": SURFACE_EDGE}),
                (spheres(curve_3d.points.numpy(), 0.045), {"color": SURFACE_EDGE}),
            ],
            zoom=1.15,
        ),
        "tube": render(
            [(to_pyvista(tube).compute_normals(auto_orient_normals=True), SHADED)],
            zoom=1.15,
        ),
        "sheet": render(
            [
                (
                    sheet_pv,
                    {
                        "scalars": "z",
                        "cmap": SEQUENTIAL,
                        "show_edges": True,
                        "edge_color": FAINT_EDGE,
                        "show_scalar_bar": False,
                    },
                )
            ],
            zoom=1.2,
        ),
    }

    figure, axes = plt.subplots(
        2, 4, figsize=(14.0, 8.0), gridspec_kw={"wspace": 0.16, "hspace": 0.3}
    )
    panels = [
        (axes[0, 0], square, "unit_square.load()", "triangles"),
        (axes[0, 1], images["square_3d"], "embed(square, 3)", "triangles"),
        (axes[0, 2], images["prism"], "extrude(square_3d, [0, 0, 0.5])", "tetrahedra"),
        (axes[1, 0], curve, "circle_2d.load()", "edges"),
        (axes[1, 1], images["curve_3d"], "embed(curve, 3)", "edges"),
        (axes[1, 2], images["tube"], "extrude(curve_3d, [0, 0, 1])", "triangles"),
        (axes[0, 3], images["sheet"], "curved sheet, color = z", "triangles"),
        (axes[1, 3], shadow, "project(sheet, keep_dims=[0, 1])", "triangles"),
    ]
    meshes = {
        id(axes[0, 1]): square_3d,
        id(axes[0, 2]): prism,
        id(axes[1, 1]): curve_3d,
        id(axes[1, 2]): tube,
        id(axes[0, 3]): sheet,
    }
    for axis, content, call, noun in panels:
        if hasattr(content, "points"):
            _draw_planar(axis, content)
            mesh = content
        else:
            axis.imshow(content[50:-50, 50:-50])
            axis.set_axis_off()
            mesh = meshes[id(axis)]
        axis.set_title(f"{call}\n{mesh_label(mesh, noun)}")

    for row in range(2):
        for column in range(2):
            _arrow(axes[row, column], (1.0, 0.5), (1.17, 0.5))
    _arrow(axes[0, 3], (0.5, -0.02), (0.5, -0.17))
    # A rule sets the lossy column apart from the lossless chains.
    figure.add_artist(
        plt.Line2D([0.745, 0.745], [0.06, 0.94], color=FAINT_EDGE, linewidth=1.5)
    )

    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.92)
    save(figure, OUTPUT, full_color=True)


if __name__ == "__main__":
    main()
