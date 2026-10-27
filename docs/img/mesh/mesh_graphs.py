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


"""Render a curved triangle mesh, its edge graph, and its dual graph, each a Mesh[1, 3]."""

from pathlib import Path

import _style as style
import matplotlib.pyplot as plt
import numpy as np
import torch

from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives.procedural import perturbed_grid
from physicsnemo.mesh.projections import embed

OUTPUT = Path(__file__).parent / "mesh_graphs.png"

EDGE_RADIUS = 0.009
NODE_RADIUS = 0.024
VIEW_VECTOR = (0.8, -1.2, 1.5)


def _curved_sheet():
    """A perturbed 6 x 6 grid lifted into 3D and bent into a gentle dome."""

    mesh = embed(
        perturbed_grid.load(n_x=6, n_y=6, perturbation_scale=0.04, seed=0),
        target_n_spatial_dims=3,
    )
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    bump = 0.28 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    return mesh.with_points(
        mesh.points + bump[:, None] * mesh.points.new_tensor([0, 0, 1])
    )


def _faint(mesh):
    """The surface as faint context, moved down a little so that graph edges stay on top."""

    below = mesh.with_points(mesh.points - mesh.points.new_tensor([0, 0, 0.03]))
    return (
        to_pyvista(below),
        dict(color=style.FAINT, show_edges=False, ambient=0.6, diffuse=0.4),
    )


def _trim(image, margin=12):
    """Crop the white border of a rendered image."""

    ink = (image[..., :3] < 250).any(axis=2)
    rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    r0, r1 = max(rows[0] - margin, 0), rows[-1] + margin
    c0, c1 = max(cols[0] - margin, 0), cols[-1] + margin
    return image[r0:r1, c0:c1]


def _graph(graph, color):
    """A Mesh[1, 3] drawn as tubes for its edges and spheres for its nodes."""

    # Soft lighting keeps the rendered colors close to the palette.
    light = dict(color=color, ambient=0.55, diffuse=0.5, specular=0.0)
    return [
        (style.tubes(to_pyvista(graph), EDGE_RADIUS), light),
        (style.spheres(graph.points.numpy(), NODE_RADIUS), light),
    ]


def main():
    """Generate the documentation image."""

    mesh = _curved_sheet()
    edge_graph = mesh.to_edge_graph()
    dual_graph = mesh.to_dual_graph()
    # On a triangle mesh, the edge graph is the facet mesh.
    if not torch.equal(edge_graph.cells, mesh.get_facet_mesh().cells):
        raise RuntimeError("to_edge_graph() and get_facet_mesh() disagree")

    surface = [
        (to_pyvista(mesh), dict(color=style.SURFACE)),
        (style.tubes(to_pyvista(mesh), 0.0025), dict(color=style.SURFACE_EDGE)),
    ]
    panels = [
        (
            surface,
            f"triangle mesh\nMesh[2, 3], {mesh.n_points} points, {mesh.n_cells} triangles",
        ),
        (
            [_faint(mesh), *_graph(edge_graph, style.SECOND)],
            "mesh.to_edge_graph()\n"
            f"Mesh[1, 3], {edge_graph.n_points} nodes, {edge_graph.n_cells} edges",
        ),
        (
            [
                _faint(mesh),
                (
                    style.tubes(to_pyvista(edge_graph), 0.004),
                    dict(color=style.FAINT_EDGE),
                ),
                *_graph(dual_graph, style.RESULT),
            ],
            "mesh.to_dual_graph()\n"
            f"Mesh[1, 3], {dual_graph.n_points} nodes, {dual_graph.n_cells} edges",
        ),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.9))
    for axis, (actors, title) in zip(axes, panels, strict=True):
        image = style.render(
            actors, window_size=(1000, 900), zoom=1.0, view_vector=VIEW_VECTOR
        )
        axis.imshow(_trim(image))
        axis.set_title(title)
        axis.set_axis_off()
    figure.text(
        0.5,
        0.03,
        "In a Mesh[1, n], points are the nodes, cells are the edges, "
        "and point_data and cell_data hold node and edge features.",
        ha="center",
    )
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=0.9, wspace=0.02)
    style.save(figure, OUTPUT)


if __name__ == "__main__":
    main()
