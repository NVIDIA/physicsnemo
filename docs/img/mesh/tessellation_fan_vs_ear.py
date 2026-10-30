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

"""Draw a polygon soup, its triangulation, and a vertex-0 fan against ear clipping."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from _style import (
    EDGE,
    FACE,
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
from matplotlib.patches import Polygon

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.neighbors import Adjacency
from physicsnemo.mesh.tessellation import triangulate

OUTPUT = Path(__file__).parent / "tessellation_fan_vs_ear.png"

# One fill color per source polygon (parent_index 0, 1, 2).
PARENT_COLORS = (SECOND_FILL, FOURTH, FIXED)

QUAD = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
HEXAGON = [
    (1.95 + 0.55 * np.cos(a), 0.5 + 0.55 * np.sin(a))
    for a in np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
]
# A non-convex arrow. Vertex 0 sits on the shaft, so its fan leaves the polygon.
ARROW = [
    (2.8, 0.4),
    (3.5, 0.4),
    (3.5, 0.75),
    (4.3, 0.25),
    (3.5, -0.25),
    (3.5, 0.1),
    (2.8, 0.1),
]


def _shoelace(xy):
    """Signed area of a polygon ring."""

    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _triangle_areas(xy, triangles):
    """Unsigned areas of triangles given by vertex index triples."""

    a, b, c = (xy[triangles[:, k]] for k in range(3))
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    return 0.5 * np.abs(cross)


def _outline(axis, xy, color=TEXT, linewidth=2.0, fill="none"):
    """Draw one closed polygon outline."""

    axis.add_patch(
        Polygon(xy, closed=True, facecolor=fill, edgecolor=color, linewidth=linewidth)
    )


def _vertices(axis, xy):
    """Draw polygon vertices as white-rimmed dots."""

    axis.scatter(
        xy[:, 0],
        xy[:, 1],
        s=46,
        color=TEXT,
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
        clip_on=False,
    )


def main():
    """Generate the documentation image."""

    rings = [np.array(r, dtype=np.float64) for r in (QUAD, HEXAGON, ARROW)]
    sizes = [len(r) for r in rings]
    points = torch.tensor(np.concatenate(rings), dtype=torch.float32)
    offsets = torch.tensor(np.concatenate([[0], np.cumsum(sizes)]))
    polygons = Adjacency(offsets=offsets, indices=torch.arange(int(offsets[-1])))
    cells, parent_index = triangulate(points, polygons)
    mesh = Mesh.from_polygons(points, polygons)
    xy = points.numpy().astype(np.float64)
    cells_np, parents_np = cells.numpy(), parent_index.numpy()

    # The arrow alone: a hand-built vertex-0 fan against what triangulate returns.
    arrow_xy = rings[2]
    k = len(arrow_xy)
    fan = np.array([[0, i, i + 1] for i in range(1, k - 1)])
    arrow_cells = cells_np[parents_np == 2] - int(offsets[2])
    polygon_area = abs(_shoelace(arrow_xy))
    fan_area = _triangle_areas(arrow_xy, fan).sum()
    ear_area = _triangle_areas(arrow_xy, arrow_cells).sum()
    # A fan triangle leaves the polygon when its orientation opposes the ring's.
    ring_sign = np.sign(_shoelace(arrow_xy))
    a, b, c = (arrow_xy[fan[:, j]] for j in range(3))
    fan_sign = np.sign(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    outside = fan_sign != ring_sign

    # The arrow panels zoom in 1.5 times. Their limits keep every panel equally tall.
    soup_limits = ((-0.12, 4.42), (-0.66, 1.12))
    zoom = 1.5
    arrow_limits = ((2.68, 4.42), (0.25 - 0.89 / zoom, 0.25 + 0.89 / zoom))
    soup_width = soup_limits[0][1] - soup_limits[0][0]
    arrow_width = zoom * (arrow_limits[0][1] - arrow_limits[0][0])
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(14.0, 3.0),
        gridspec_kw={
            "width_ratios": [soup_width, soup_width, arrow_width, arrow_width]
        },
    )

    # (a) The polygon soup with its CSR offsets.
    axis = axes[0]
    for ring in rings:
        _outline(axis, ring, fill=FACE)
        _vertices(axis, ring)
    for ring, size in zip(rings, sizes, strict=True):
        cx = 0.5 * (ring[:, 0].min() + ring[:, 0].max())
        axis.text(cx, -0.42, f"{size} vertices", ha="center", va="top", color=TEXT)
    axis.set_title(
        f"(a) polygon soup, Adjacency\noffsets = {offsets.tolist()}", loc="left"
    )
    clean_axis(axis, soup_limits)

    # (b) triangulate: each triangle colored by its parent polygon.
    axis = axes[1]
    axis.add_collection(
        PolyCollection(
            xy[cells_np],
            facecolors=[PARENT_COLORS[p] for p in parents_np],
            edgecolors="white",
            linewidths=1.6,
        )
    )
    for ring in rings:
        _outline(axis, ring)
    for parent, ring in enumerate(rings):
        cx = 0.5 * (ring[:, 0].min() + ring[:, 0].max())
        count = int((parents_np == parent).sum())
        axis.text(cx, -0.42, f"{count} triangles", ha="center", va="top", color=TEXT)
    axis.set_title(
        f"(b) triangulate(points, polygons)\n"
        f"Mesh[{mesh.n_manifold_dims}, {mesh.n_spatial_dims}], {mesh.n_cells} triangles, "
        "fill = parent_index",
        loc="left",
    )
    clean_axis(axis, soup_limits)

    # (c) Vertex-0 fan, drawn by hand: the triangles outside the polygon are hatched.
    axis = axes[2]
    _outline(axis, arrow_xy, fill=FACE)
    for triangle, out in zip(fan, outside, strict=True):
        axis.add_patch(
            Polygon(
                arrow_xy[triangle],
                closed=True,
                facecolor="none",
                edgecolor=RESULT if out else EDGE,
                hatch="///" if out else None,
                linewidth=1.4,
            )
        )
    _outline(axis, arrow_xy)
    _vertices(axis, arrow_xy)
    axis.annotate(
        "0", arrow_xy[0], xytext=(-12, 6), textcoords="offset points", color=TEXT
    )
    axis.set_title(
        f"(c) vertex-0 fan, not used\narea {fan_area:.2f} > {polygon_area:.2f}",
        loc="left",
        color=RESULT,
    )
    clean_axis(axis, arrow_limits)

    # (d) Ear clipping, which triangulate returns for this polygon.
    axis = axes[3]
    axis.add_collection(
        PolyCollection(
            arrow_xy[arrow_cells],
            facecolors=SECOND_FILL,
            edgecolors="white",
            linewidths=1.6,
        )
    )
    _outline(axis, arrow_xy, color=SECOND)
    _vertices(axis, arrow_xy)
    axis.set_title(
        f"(d) ear clipping, returned\narea {ear_area:.2f} = {polygon_area:.2f}",
        loc="left",
    )
    clean_axis(axis, arrow_limits)

    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.80, wspace=0.06)
    save(figure, OUTPUT, tight=True)


if __name__ == "__main__":
    main()
