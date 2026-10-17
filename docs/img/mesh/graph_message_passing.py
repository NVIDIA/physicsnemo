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


"""Render message passing on the edge graph of a mesh with plain tensor indexing."""

from pathlib import Path

import _style as style
import matplotlib.pyplot as plt
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch

from physicsnemo.mesh.primitives.procedural import perturbed_grid

OUTPUT = Path(__file__).parent / "graph_message_passing.png"

STEPS = 3
LEVELS = 12  # discrete color steps, which also keep the PNG palette small
LIMITS = ((-0.05, 1.05), (-0.05, 1.05))


def _edges(axis, graph, colors, linewidth, zorder=1):
    """Draw the cells of a Mesh[1, 2] as line segments."""

    points = graph.points.numpy()
    axis.add_collection(
        LineCollection(
            points[graph.cells.numpy()],
            colors=colors,
            linewidths=linewidth,
            zorder=zorder,
        )
    )


def _nodes(axis, points, size, zorder=5, **kwargs):
    """Draw nodes as dots with a white rim."""

    axis.scatter(
        points[:, 0],
        points[:, 1],
        s=size,
        edgecolor="white",
        linewidth=1.2,
        zorder=zorder,
        **kwargs,
    )


def _colorbar(figure, cax, norm, cmap, label):
    """A horizontal colorbar in its own axes."""

    figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
        label=label,
    )


def main():
    """Generate the documentation image."""

    mesh = perturbed_grid.load(n_x=6, n_y=6, perturbation_scale=0.04, seed=0)
    graph = mesh.to_edge_graph()
    points = graph.points.numpy()

    ### Graph construction: each undirected edge becomes two directed pairs.
    adjacency = graph.get_point_to_points_adjacency()
    i, j = adjacency.expand_to_pairs()  # node i receives a message from node j
    center = graph.points.new_tensor([0.5, 0.5])
    query = int(torch.linalg.vector_norm(graph.points - center, dim=1).argmin())
    senders = j[i == query].numpy()

    ### Edge features from geometry: on a Mesh[1, n], the cell measure is the edge length.
    length = graph.cell_areas

    ### Node field: a smooth ramp plus seeded noise, then mean aggregation over neighbors.
    generator = torch.Generator().manual_seed(0)
    ramp = torch.sin(torch.pi * (graph.points[:, 0] - 0.5))
    x = ramp + 0.7 * torch.randn(graph.n_points, generator=generator)
    smoothed = x.clone()
    for _ in range(STEPS):
        smoothed = torch.zeros_like(smoothed).index_add_(0, i, smoothed[j])
        smoothed = smoothed / adjacency.counts

    sequential = plt.get_cmap(style.SEQUENTIAL, LEVELS)
    diverging = plt.get_cmap(style.DIVERGING, LEVELS)
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 11.4))
    axes = axes.ravel()
    figure.subplots_adjust(
        left=0.02, right=0.98, bottom=0.1, top=0.92, wspace=0.08, hspace=0.4
    )

    axis = axes[0]
    _edges(axis, graph, style.FAINT_EDGE, 1.4)
    for sender in senders:
        axis.add_patch(
            FancyArrowPatch(
                points[sender],
                points[query],
                arrowstyle="-|>",
                # Small heads and margins, so that the shortest arrow keeps a shaft.
                mutation_scale=15,
                shrinkA=6,
                shrinkB=9,
                color=style.SECOND,
                linewidth=2.2,
                zorder=3,
            )
        )
    _nodes(axis, points, 40, color=style.EDGE)
    _nodes(axis, points[senders], 110, color=style.SECOND)
    _nodes(axis, points[[query]], 220, color=style.RESULT)
    axis.set_title(
        "adjacency = graph.get_point_to_points_adjacency()\n"
        "i, j = adjacency.expand_to_pairs()\n"
        f"{len(i)} directed pairs = 2 x {graph.n_cells} edges"
    )

    axis = axes[1]
    length_norm = Normalize(float(length.min()), float(length.max()))
    _edges(axis, graph, sequential(length_norm(length.numpy())), 4.0)
    _nodes(axis, points, 40, color=style.EDGE)
    axis.set_title(f"graph.cell_areas\nedge lengths, {graph.n_cells} edges")
    _colorbar(
        figure,
        axis.inset_axes((0.2, -0.08, 0.6, 0.035)),
        length_norm,
        sequential,
        None,  # the panel title names the quantity
    )

    bound = float(x.abs().max())
    field_norm = TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)
    for axis, field, title in (
        (axes[2], x, f"node field x\n{graph.n_points} nodes, a ramp plus noise"),
        (
            axes[3],
            smoothed,
            "torch.zeros_like(x).index_add_(0, i, x[j])\n"
            f"/ adjacency.counts, repeated {STEPS} times\n"
            "mean of the neighbors, plain tensors",
        ),
    ):
        _edges(axis, graph, style.FAINT_EDGE, 1.4)
        _nodes(axis, points, 200, c=field.numpy(), cmap=diverging, norm=field_norm)
        axis.set_title(title)
    _colorbar(
        figure, figure.add_axes((0.35, 0.05, 0.3, 0.016)), field_norm, diverging, "x"
    )

    for axis in axes:
        style.clean_axis(axis, LIMITS)
    style.save(figure, OUTPUT)


if __name__ == "__main__":
    main()
