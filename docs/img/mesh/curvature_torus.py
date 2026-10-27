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


"""Render Gaussian and mean curvature on a torus, and the angle defect at an icosahedron vertex."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _style import (
    DIVERGING,
    EDGE,
    FACE,
    RESULT,
    SECOND,
    SECOND_FILL,
    SURFACE_EDGE,
    TEXT,
    clean_axis,
    render,
    save,
)
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon, Wedge
from matplotlib.ticker import MaxNLocator, MultipleLocator

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.geometry import compute_vertex_angle_sums
from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral, torus

OUTPUT = Path(__file__).parent / "curvature_torus.png"

TORUS_VIEW = (0.0, -1.0, 1.15)


def _norm(values):
    """A colormap norm centered at zero.

    A field that changes sign stretches each side to its own extreme. A field of
    one sign gets a symmetric range, so that its colorbar shows it has one sign.
    """

    low, high = float(values.min()), float(values.max())
    if low < 0.0 < high:
        return TwoSlopeNorm(vcenter=0.0, vmin=low, vmax=high)
    bound = max(abs(low), abs(high))
    return TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)


def _render_field(mesh, values, norm):
    """Render the torus colored by a vertex field through a centered norm."""

    surface = to_pyvista(mesh)
    rgb = colormaps[DIVERGING](norm(values.numpy()))[:, :3]
    surface.point_data["rgb"] = (255 * rgb).astype(np.uint8)
    actors = [
        (
            surface,
            dict(
                scalars="rgb",
                rgb=True,
                smooth_shading=True,
                show_edges=True,
                edge_color=SURFACE_EDGE,
                line_width=0.3,
                ambient=0.45,
                diffuse=0.6,
                specular=0.1,
            ),
        )
    ]
    return render(actors, window_size=(1200, 820), zoom=1.65, view_vector=TORUS_VIEW)


def _draw_angle_defect(axis, n_triangles, defect):
    """Lay the triangles around one vertex flat and mark the gap they leave."""

    theta = (2 * math.pi - defect) / n_triangles
    start = math.pi / 2 + defect / 2
    for k in range(n_triangles):
        a0, a1 = start + k * theta, start + (k + 1) * theta
        corners = [
            (0.0, 0.0),
            (math.cos(a0), math.sin(a0)),
            (math.cos(a1), math.sin(a1)),
        ]
        axis.add_patch(
            Polygon(corners, closed=True, facecolor=FACE, edgecolor=EDGE, linewidth=1.5)
        )
        if k == n_triangles - 1:
            axis.add_patch(
                Wedge(
                    (0.0, 0.0),
                    0.3,
                    math.degrees(a0),
                    math.degrees(a1),
                    facecolor=SECOND_FILL,
                    edgecolor=SECOND,
                    linewidth=1.2,
                )
            )
            middle = 0.5 * (a0 + a1)
            axis.text(
                0.45 * math.cos(middle),
                0.45 * math.sin(middle),
                r"$\theta_j$",
                ha="center",
                va="center",
                color=TEXT,
                fontsize=15,
            )
    axis.add_patch(
        Wedge(
            (0.0, 0.0),
            1.0,
            math.degrees(math.pi / 2 - defect / 2),
            math.degrees(math.pi / 2 + defect / 2),
            facecolor=RESULT,
            alpha=0.18,
            edgecolor=RESULT,
            linestyle="--",
            linewidth=1.5,
        )
    )
    axis.plot(
        0.0,
        0.0,
        "o",
        color=RESULT,
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=4,
    )
    axis.text(
        0.0,
        1.08,
        rf"gap $2\pi - \Sigma\,\theta_j$ = {math.degrees(defect):.0f}°",
        ha="center",
        va="bottom",
        color=RESULT,
        fontsize=15,
    )


def main():
    """Generate the documentation image."""

    mesh = torus.load(major_radius=1.0, minor_radius=0.35, n_major=96, n_minor=48)
    # torus.load() winds its cells with inward normals. Reverse them, so that
    # the normals point outward and the mean curvature comes out positive.
    mesh = Mesh(points=mesh.points, cells=mesh.cells[:, [0, 2, 1]])
    gaussian = mesh.gaussian_curvature_vertices
    mean = mesh.mean_curvature_vertices
    print(f"K in [{float(gaussian.min()):.3f}, {float(gaussian.max()):.3f}]")
    print(f"H in [{float(mean.min()):.3f}, {float(mean.max()):.3f}]")

    icosahedron = sphere_icosahedral.load(subdivisions=0)
    angle_sum = float(compute_vertex_angle_sums(icosahedron)[0])
    defect = 2 * math.pi - angle_sum
    n_triangles = int((icosahedron.cells == 0).any(dim=-1).sum())
    print(f"icosahedron vertex 0: angle sum {math.degrees(angle_sum):.1f} deg")

    figure, axes = plt.subplots(
        1, 3, figsize=(14.0, 5.2), gridspec_kw={"width_ratios": (1.35, 1.35, 1.0)}
    )
    titles = []
    # K changes sign, so its two halves of the colorbar have different scales.
    # Ticks at even steps keep the labels apart.
    for axis, values, name, symbol, locator in (
        (axes[0], gaussian, "gaussian_curvature_vertices", "K", MultipleLocator(2.0)),
        (axes[1], mean, "mean_curvature_vertices", "H", MaxNLocator(nbins=5)),
    ):
        norm = _norm(values)
        axis.imshow(_render_field(mesh, values, norm))
        titles.append(
            f"{name}\n{mesh.n_points} points, {symbol} from"
            f" {float(values.min()):.2f} to {float(values.max()):.2f}"
        )
        axis.set_axis_off()
        colorbar = figure.colorbar(
            ScalarMappable(norm=norm, cmap=DIVERGING),
            cax=axis.inset_axes([0.2, -0.07, 0.6, 0.04]),
            orientation="horizontal",
        )
        colorbar.locator = locator
        colorbar.update_ticks()
        colorbar.set_label(symbol)

    _draw_angle_defect(axes[2], n_triangles, defect)
    titles.append(
        f"compute_vertex_angle_sums(mesh)\nicosahedron vertex, {n_triangles} angles,"
        f" sum {math.degrees(angle_sum):.0f}°"
    )
    clean_axis(axes[2], ((-1.25, 1.25), (-1.0, 1.35)))
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.12, top=0.84, wspace=0.04)
    # Put the three titles on one line, although the panels differ in height.
    for axis, title in zip(axes, titles, strict=True):
        box = axis.get_position()
        figure.text(0.5 * (box.x0 + box.x1), 0.95, title, ha="center", va="top")
    save(figure, OUTPUT)


if __name__ == "__main__":
    main()
