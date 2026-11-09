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

"""Draw one primitive for each Mesh[m, s] with m <= s <= 3, shaded by codimension."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from _style import (
    ACCENT_LIGHTING,
    EDGE,
    RESULT,
    SECOND,
    SURFACE,
    SURFACE_EDGE,
    TEXT,
    VIEW_VECTOR,
    clean_axis,
    render,
    save,
    spheres,
)
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Patch

from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives.curves import circle_2d, helix_3d, line_segments_1d
from physicsnemo.mesh.primitives.planar import annulus_2d
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.mesh.primitives.volumes import cube_volume

OUTPUT = Path(__file__).parent / "mesh_dimensions.png"

# Background tint of a panel, by codimension (2 stands for 2 or more).
TINTS = {0: "#e3eef7", 1: "#fbeadb", 2: "#eeeff1"}
TINT_LABELS = {
    0: "codimension 0\nthe cells fill the space",
    1: "codimension 1\none normal (arrows)",
    2: "codimension 2 or more\nmany normals (disk)",
}
NOUNS = {0: "points", 1: "edges", 2: "triangles", 3: "tetrahedra"}


def _primitives():
    """The mesh of each filled panel, keyed by its Mesh[m, s]."""

    segments = line_segments_1d.load(n_segments=4, gap=0.25)
    circle = circle_2d.load(n_points=20)
    sphere = sphere_icosahedral.load(subdivisions=1)
    return {
        (0, 1): ("line_segments_1d.to_point_cloud()", segments.to_point_cloud()),
        (1, 1): ("line_segments_1d", segments),
        (0, 2): ("circle_2d.to_point_cloud()", circle.to_point_cloud()),
        (1, 2): ("circle_2d", circle),
        (2, 2): ("annulus_2d", annulus_2d.load(n_radial=3, n_angular=20)),
        (0, 3): (
            "sphere_icosahedral.to_point_cloud()",
            sphere_icosahedral.load(subdivisions=2).to_point_cloud(),
        ),
        (1, 3): ("helix_3d", helix_3d.load(height=3.0, n_turns=2.0, n_points=80)),
        (2, 3): ("sphere_icosahedral", sphere),
        (3, 3): ("cube_volume, cut open", cube_volume.load(subdivisions=3)),
    }


def _tinted(image, tint):
    """Replace the white background of a rendered image by a tint."""

    image = image.astype(float) / 255.0
    whiteness = image.min(axis=-1, keepdims=True)
    # Blend by how white each pixel is, so antialiased edges keep their shape.
    weight = np.clip((whiteness - 0.92) / 0.08, 0.0, 1.0)
    return image * (1 - weight) + np.array(to_rgb(tint)) * weight


def _draw_1d(axis, mesh):
    """A mesh in 1D space, drawn on a number line."""

    x = mesh.points[:, 0].numpy()
    axis.plot([x.min() - 0.15, x.max() + 0.15], [0, 0], color=EDGE, lw=1.0, zorder=1)
    if mesh.n_manifold_dims == 1:
        segments = [[(x[a], 0.0), (x[b], 0.0)] for a, b in mesh.cells.numpy()]
        axis.add_collection(LineCollection(segments, colors=SECOND, linewidths=4.0))
    axis.scatter(
        x, np.zeros_like(x), s=70, color=TEXT, edgecolor="white", lw=1.2, zorder=3
    )
    clean_axis(axis, ((x.min() - 0.2, x.max() + 0.2), (-0.12, 0.12)))


def _draw_2d(axis, mesh):
    """A mesh in 2D space, with cell normals on a curve."""

    points = mesh.points.numpy()
    if mesh.n_manifold_dims == 2:
        axis.add_collection(
            PolyCollection(
                points[mesh.cells.numpy()],
                facecolors=SURFACE,
                edgecolors=SURFACE_EDGE,
                linewidths=0.8,
            )
        )
    elif mesh.n_manifold_dims == 1:
        axis.add_collection(
            LineCollection(points[mesh.cells.numpy()], colors=SECOND, linewidths=3.0)
        )
        centers = mesh.cell_centroids.numpy()
        normals = mesh.cell_normals.numpy()
        axis.quiver(
            centers[:, 0],
            centers[:, 1],
            normals[:, 0],
            normals[:, 1],
            color=RESULT,
            scale=11.0,
            width=0.011,
            zorder=4,
        )
    size = 26 if mesh.n_manifold_dims == 2 else 60
    axis.scatter(
        points[:, 0],
        points[:, 1],
        s=size,
        color=TEXT,
        edgecolor="white",
        lw=1.0,
        zorder=3,
    )
    clean_axis(axis, ((-1.45, 1.45), (-1.45, 1.45)))


def _facing_octant(mesh):
    """Mask of the cells in the octant that faces the camera."""

    offsets = mesh.cell_centroids - mesh.points.mean(dim=0)
    return (offsets * offsets.new_tensor(VIEW_VECTOR) > 0).all(dim=1)


def _render_3d(mesh):
    """A mesh in 3D space, with point normals on a surface and a normal disk on a curve."""

    radius = 0.045
    actors = []
    if mesh.n_manifold_dims == 0:
        actors.append((spheres(mesh.points.numpy(), radius=radius), {"color": TEXT}))
    elif mesh.n_manifold_dims == 1:
        dataset = to_pyvista(mesh)
        actors.append((dataset.tube(radius=0.035), {"color": SECOND}))
        # Every vector in the disk is normal to the curve at its center.
        i = mesh.n_points // 2 + 10
        tangent = (mesh.points[i + 1] - mesh.points[i - 1]).numpy()
        disk = pv.Disc(
            center=mesh.points[i].numpy(),
            inner=0.0,
            outer=0.75,
            normal=tangent,
            c_res=60,
        )
        actors.append((disk, {"color": "#8a949c", "opacity": 0.55}))
        actors.append(
            (
                spheres(mesh.points[i : i + 1].numpy(), radius=0.07),
                {"color": RESULT, **ACCENT_LIGHTING},
            )
        )
    elif mesh.n_manifold_dims == 2:
        actors.append(
            (
                to_pyvista(mesh),
                {
                    "color": SURFACE,
                    "show_edges": True,
                    "edge_color": SURFACE_EDGE,
                    "line_width": 0.8,
                },
            )
        )
        arrows = pv.PolyData(mesh.points.numpy())
        arrows["normals"] = mesh.point_normals.numpy()
        glyphs = arrows.glyph(
            orient="normals",
            scale=False,
            factor=0.42,
            geom=pv.Arrow(tip_length=0.3, tip_radius=0.12, shaft_radius=0.045),
        )
        actors.append((glyphs, {"color": RESULT, **ACCENT_LIGHTING}))
    else:
        # Drop the cells in the octant that faces the camera, and shrink the rest.
        kept = (~_facing_octant(mesh)).nonzero().squeeze(1).numpy()
        cut = to_pyvista(mesh).extract_cells(kept)
        actors.append(
            (
                cut.shrink(0.8),
                {
                    "color": SURFACE,
                    "show_edges": True,
                    "edge_color": SURFACE_EDGE,
                    "line_width": 0.6,
                },
            )
        )
    zoom = 0.95 if mesh.n_manifold_dims == 3 else 1.25
    return render(actors, window_size=(800, 800), zoom=zoom)


def main():
    """Generate the documentation image."""

    figure = plt.figure(figsize=(14.0, 10.6))
    grid = figure.add_gridspec(
        3,
        4,
        height_ratios=(0.5, 1.0, 1.0),
        left=0.05,
        right=0.995,
        top=0.915,
        bottom=0.01,
        hspace=0.2,
        wspace=0.08,
    )
    for name, mesh in _primitives().values():
        m, s = mesh.n_manifold_dims, mesh.n_spatial_dims
        axis = figure.add_subplot(grid[s - 1, m])
        tint = TINTS[min(mesh.codimension, 2)]
        box = axis.get_position()
        figure.add_artist(
            FancyBboxPatch(
                (box.x0, box.y0),
                box.width,
                box.height,
                boxstyle="round,pad=0.004,rounding_size=0.01",
                transform=figure.transFigure,
                facecolor=tint,
                edgecolor="none",
                zorder=0,
            )
        )
        if s == 1:
            _draw_1d(axis, mesh)
        elif s == 2:
            _draw_2d(axis, mesh)
        else:
            axis.imshow(_tinted(_render_3d(mesh), tint))
            axis.set_axis_off()
        axis.set_zorder(1)
        count = mesh.n_points if m == 0 else mesh.n_cells
        if m == 3:
            # The panel draws the cube with one octant removed.
            count = f"{int((~_facing_octant(mesh)).sum())} of {count}"
        figure.text(
            (box.x0 + box.x1) / 2,
            box.y1 + 0.006,
            f"{name}\nMesh[{m}, {s}], {count} {NOUNS[m]}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Column and row headers.
    for m in range(4):
        box = figure.add_subplot(grid[2, m]).get_position()
        figure.axes[-1].remove()
        figure.text(
            (box.x0 + box.x1) / 2,
            0.975,
            f"n_manifold_dims = {m}",
            ha="center",
            va="center",
            fontsize=13,
            family="DejaVu Sans Mono",
        )
    for s in (1, 2, 3):
        probe = figure.add_subplot(grid[s - 1, 0])
        box = probe.get_position()
        probe.remove()
        figure.text(
            0.022,
            (box.y0 + box.y1) / 2,
            f"n_spatial_dims = {s}",
            rotation=90,
            ha="center",
            va="center",
            fontsize=13,
            family="DejaVu Sans Mono",
        )

    # The empty upper triangle: a manifold never has more dimensions than its space.
    probe = figure.add_subplot(grid[0, 2:])
    box = probe.get_position()
    probe.remove()
    figure.text(
        (box.x0 + box.x1) / 2,
        (box.y0 + box.y1) / 2,
        "empty: always n_manifold_dims ≤ n_spatial_dims",
        ha="center",
        va="center",
        fontsize=13,
        color="#8a949c",
    )

    # The legend fills the empty panel of Mesh[3, 2].
    probe = figure.add_subplot(grid[1, 3])
    box = probe.get_position()
    probe.remove()
    figure.legend(
        handles=[
            Patch(facecolor=TINTS[c], edgecolor="none", label=TINT_LABELS[c])
            for c in TINTS
        ],
        loc="center left",
        ncol=1,
        frameon=False,
        fontsize=12.5,
        labelspacing=1.4,
        handlelength=1.8,
        borderaxespad=0.0,
        handleheight=1.8,
        bbox_to_anchor=(box.x0 + 0.004, (box.y0 + box.y1) / 2),
    )
    save(figure, OUTPUT)


if __name__ == "__main__":
    main()
