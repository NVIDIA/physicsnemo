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


"""Show the boxes of a BVH with a point query, and a slice through a signed distance field."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from _style import (
    DIVERGING,
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from physicsnemo.mesh.primitives.planar import annulus_2d
from physicsnemo.mesh.primitives.procedural import lumpy_sphere
from physicsnemo.mesh.spatial import BVH, signed_distance_field

OUTPUT = Path(__file__).parent / "spatial_bvh_sdf.png"

DEPTHS = {
    1: ("#4a555e", 2.6),
    3: ("#8f9aa3", 1.6),
    5: ("#c3cbd1", 1.0),
}  # depth: (gray, line width)
QUERY = torch.tensor([[0.43, 0.6]])
SLICE_RESOLUTION = 240


def _node_depths(bvh):
    """Depth of each BVH node, found by walking down from the root (node 0)."""

    depth = torch.full((bvh.n_nodes,), -1, dtype=torch.long)
    depth[0] = 0
    level = torch.tensor([0])
    while len(level):
        children = torch.cat([bvh.node_left_child[level], bvh.node_right_child[level]])
        children = children[children >= 0]
        depth[children] = depth[level].max() + 1
        level = children
    return depth


def main():
    """Generate the documentation image."""

    ### (a) BVH over an annulus.
    annulus = annulus_2d.load(n_radial=3, n_angular=24)
    bvh = BVH.from_mesh(annulus)
    depth = _node_depths(bvh)
    candidates = bvh.find_candidate_cells(QUERY).to_list()[0]
    lo, hi = bvh.node_aabb_min, bvh.node_aabb_max
    contains = ((lo <= QUERY) & (QUERY <= hi)).all(dim=-1)
    print(f"BVH: {bvh.n_nodes} nodes, candidates {candidates}")

    ### (b) Signed distance on the slice z = 0 through a closed surface.
    surface = lumpy_sphere.load(subdivisions=4, noise_amplitude=0.4, seed=1)
    extent = 1.7
    axis_values = torch.linspace(-extent, extent, SLICE_RESOLUTION)
    gx, gy = torch.meshgrid(axis_values, axis_values, indexing="xy")
    grid = torch.stack([gx, gy, torch.zeros_like(gx)], dim=-1)
    sdf, _, _ = signed_distance_field(surface, grid)
    probes = torch.tensor(
        [
            [1.45, 0.9, 0.0],
            [-1.4, 1.1, 0.0],
            [-1.3, -1.25, 0.0],
            [1.2, -1.35, 0.0],
            # Inside. Its closest point lies near the slice, so its arrow meets the curve.
            [0.175, 0.475, 0.0],
        ]
    )
    probe_sdf, hit_points, _ = signed_distance_field(surface, probes)

    figure, (bvh_axis, sdf_axis) = plt.subplots(
        1, 2, figsize=(13.0, 6.9), gridspec_kw={"width_ratios": (1.0, 1.12)}
    )

    cells = annulus.points[annulus.cells].numpy()
    facecolors = [
        SECOND_FILL if i in candidates else FACE for i in range(annulus.n_cells)
    ]
    bvh_axis.add_collection(
        PolyCollection(cells, facecolors=facecolors, edgecolors=EDGE, linewidths=0.8)
    )
    for level, (gray, width) in sorted(DEPTHS.items(), reverse=True):
        for node in torch.nonzero(depth == level).flatten().tolist():
            on_path = bool(contains[node])
            (x0, y0), (x1, y1) = lo[node].tolist(), hi[node].tolist()
            bvh_axis.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=SECOND if on_path else gray,
                    linewidth=width + (0.6 if on_path else 0.0),
                    zorder=3 if on_path else 2,
                )
            )
    bvh_axis.plot(
        *QUERY[0].tolist(),
        "*",
        color=RESULT,
        markersize=22,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=5,
    )
    bvh_axis.set_title(
        f"BVH.from_mesh(), boxes at depths 1, 3, 5\n{bvh.n_nodes} nodes over {annulus.n_cells} cells,"
        f" {len(candidates)} candidates"
    )
    clean_axis(bvh_axis, ((-1.08, 1.08), (-1.08, 1.08)))

    values = sdf.numpy()
    bound = float(abs(values).max())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)
    filled = sdf_axis.contourf(
        gx.numpy(), gy.numpy(), values, levels=24, cmap=DIVERGING, norm=norm
    )
    sdf_axis.contour(
        gx.numpy(), gy.numpy(), values, levels=[0.0], colors=RESULT, linewidths=2.6
    )
    for probe, hit in zip(
        probes[:, :2].tolist(), hit_points[:, :2].tolist(), strict=True
    ):
        sdf_axis.annotate(
            "",
            xy=hit,
            xytext=probe,
            arrowprops=dict(
                arrowstyle="-|>",
                color=TEXT,
                linewidth=1.8,
                mutation_scale=16,
                shrinkA=4,
                shrinkB=0,
            ),
        )
        sdf_axis.plot(
            *probe,
            "o",
            color=TEXT,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=5,
        )
    sdf_axis.set_title(
        f"signed_distance_field(), slice z = 0\nlumpy_sphere, {surface.n_cells} triangles"
    )
    clean_axis(sdf_axis, ((-extent, extent), (-extent, extent)))
    colorbar = figure.colorbar(filled, ax=sdf_axis, fraction=0.045, pad=0.02)
    colorbar.set_label("signed distance (negative inside)")

    handles = [
        Line2D(
            [],
            [],
            marker="*",
            color=RESULT,
            linestyle="none",
            markersize=16,
            label="query point",
        ),
        Line2D([], [], color=SECOND, linewidth=2.5, label="boxes that hold it"),
        Patch(facecolor=SECOND_FILL, edgecolor=EDGE, label="candidate cells"),
        Line2D([], [], color=RESULT, linewidth=2.6, label="surface (distance 0)"),
        Line2D(
            [],
            [],
            marker="o",
            color=TEXT,
            linewidth=1.8,
            markersize=7,
            label="hit_points",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.4,
    )
    figure.subplots_adjust(left=0.01, right=0.97, bottom=0.1, top=0.88, wspace=0.08)
    save(figure, OUTPUT, tight=True)
    print(f"probe distances {[round(v, 3) for v in probe_sdf.tolist()]}")


if __name__ == "__main__":
    main()
