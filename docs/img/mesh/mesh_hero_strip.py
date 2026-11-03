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

"""Render one blob as a point cloud, an edge graph, a surface, and a cut-open volume."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from _style import (
    SECOND,
    SEQUENTIAL,
    SURFACE,
    SURFACE_EDGE,
    render,
    save,
    spheres,
)

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.generate import mesh_implicit_domain
from physicsnemo.mesh.io import to_pyvista

OUTPUT = Path(__file__).parent / "mesh_hero_strip.png"

VIEW = (1.0, -1.5, 0.9)
WINDOW = (900, 900)
BOUNDS = ([-1.5, -1.4, -1.3], [1.5, 1.4, 1.3])


def _blob(x):
    """A smooth, slightly lumpy blob: negative inside, positive outside."""

    radius = (x / x.new_tensor([1.1, 1.0, 1.0])).norm(dim=-1)
    lumps = 0.09 * torch.sin(3.0 * x[..., 0]) * torch.cos(2.5 * x[..., 1])
    return radius - 1.0 - lumps - 0.07 * torch.sin(3.5 * x[..., 2] + 1.0)


def _field(points):
    """A smooth scalar field that colors every panel the same way."""

    x, y, z = torch.as_tensor(points, dtype=torch.float64).unbind(-1)
    return (torch.sin(2.2 * x + 0.6) + torch.cos(2.0 * y) + 1.2 * z).numpy()


def _keys(triangles):
    """One integer per triangle, independent of the vertex order."""

    a, b, c = triangles.sort(dim=1).values.long().unbind(1)
    return (a * 2**20 + b) * 2**20 + c


def _trim(image, pad=8):
    """Crop the white margin around a rendered image."""

    mask = (image < 250).any(axis=-1)
    rows, cols = np.nonzero(mask)
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad, image.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad, image.shape[1])
    return image[r0:r1, c0:c1]


def main():
    """Generate the documentation image."""

    torch.manual_seed(0)
    volume = mesh_implicit_domain(_blob, BOUNDS, h=0.17).to(torch.float32)
    surface = volume.get_boundary_mesh()
    coarse = mesh_implicit_domain(_blob, BOUNDS, h=0.38).get_boundary_mesh()

    ### Mesh[0, 3]: points sampled uniformly on the surface
    cell_indices = torch.multinomial(surface.cell_areas, 1400, replacement=True)
    cloud = Mesh(points=surface.sample_random_points_on_cells(cell_indices))

    ### Mesh[1, 3]: the edge graph of a coarse surface
    graph = coarse.to(torch.float32).to_edge_graph()

    limits = (float(_field(surface.points).min()), float(_field(surface.points).max()))
    common = {"cmap": SEQUENTIAL, "clim": limits, "show_scalar_bar": False}

    cloud_pv = spheres(cloud.points.numpy(), radius=0.03)
    cloud_pv["f"] = _field(cloud_pv.points)
    graph_tubes = to_pyvista(graph).tube(radius=0.014)
    nodes = spheres(graph.points.numpy(), radius=0.035)
    surface_pv = to_pyvista(surface)
    surface_pv["f"] = _field(surface_pv.points)

    # Cut the volume open: drop the tetrahedra in the quarter that faces the
    # camera, then split the exposed triangles into the outer skin and the cut.
    centroids = volume.cell_centroids
    kept = volume.slice_cells(~((centroids[:, 0] > 0.0) & (centroids[:, 1] < 0.0)))
    exposed = kept.get_boundary_mesh()
    on_skin = torch.isin(_keys(exposed.cells), _keys(surface.cells))
    skin_pv = to_pyvista(exposed.slice_cells(on_skin))
    skin_pv["f"] = _field(skin_pv.points)
    walls_pv = to_pyvista(exposed.slice_cells(~on_skin))

    edges = {"show_edges": True, "edge_color": SURFACE_EDGE, "line_width": 0.4}
    # A white surface just inside the cloud and the graph hides their far sides,
    # so that each panel shows only what faces the camera.
    hidden = {"color": "white", "ambient": 1.0, "diffuse": 0.0, "specular": 0.0}
    cloud_occluder = to_pyvista(surface.with_points(0.96 * surface.points))
    graph_occluder = to_pyvista(coarse.with_points(0.96 * coarse.points))
    panels = [
        (
            cloud,
            [
                (cloud_occluder, hidden),
                (cloud_pv, {"scalars": "f", **common, "smooth_shading": True}),
            ],
        ),
        (
            graph,
            [
                (graph_occluder, hidden),
                (graph_tubes, {"color": SECOND}),
                (nodes, {"color": SURFACE_EDGE, "smooth_shading": True}),
            ],
        ),
        (surface, [(surface_pv, {"scalars": "f", **common, **edges})]),
        (
            volume,
            [
                (skin_pv, {"scalars": "f", **common, **edges}),
                (walls_pv, {"color": SURFACE, **edges}),
            ],
        ),
    ]

    figure, axes = plt.subplots(1, 4, figsize=(14.0, 3.4))
    for axis, (mesh, actors) in zip(axes, panels):
        image = _trim(render(actors, window_size=WINDOW, zoom=1.0, view_vector=VIEW))
        axis.imshow(image)
        axis.set_axis_off()
    figure.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.1, wspace=0.06)
    for axis, (mesh, _) in zip(axes, panels):
        box = axis.get_position()
        figure.text(
            (box.x0 + box.x1) / 2,
            0.01,
            f"Mesh[{mesh.n_manifold_dims}, {mesh.n_spatial_dims}]",
            ha="center",
            va="bottom",
            fontsize=15,
        )
    save(figure, OUTPUT)


if __name__ == "__main__":
    main()
