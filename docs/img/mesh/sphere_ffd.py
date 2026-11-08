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

"""Render lattice free-form deformation of a sphere with its control lattice."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from _style import RESULT, SURFACE, SURFACE_EDGE, mesh_label, save, spheres

from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "sphere_ffd.png"

CAMERA = [(7.2, -7.2, 5.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
LATTICE = "#9aa5ad"

ORIGIN = (-1.15, -1.15, -1.15)
EXTENT = (2.3, 2.3, 2.3)


def lattice_nodes(resolution, basis):
    """Basis-associated control positions, shape (*resolution, 3)."""
    if basis == "bernstein":
        coordinates = [torch.linspace(0.0, 1.0, n) for n in resolution]
    else:
        coordinates = [
            (torch.arange(n, dtype=torch.float32) - 1) / (n - 3) for n in resolution
        ]
    axes = [
        origin + extent * coordinate
        for coordinate, origin, extent in zip(coordinates, ORIGIN, EXTENT)
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def lattice_wireframe(nodes):
    """Build a PyVista wireframe of the lattice edges along every axis."""
    n1, n2, n3, _ = nodes.shape
    index = np.arange(n1 * n2 * n3).reshape(n1, n2, n3)
    segments = []
    for a, b in (
        (index[:-1, :, :], index[1:, :, :]),
        (index[:, :-1, :], index[:, 1:, :]),
        (index[:, :, :-1], index[:, :, 1:]),
    ):
        segments.append(np.stack([a.ravel(), b.ravel()], axis=1))
    segments = np.concatenate(segments, axis=0)
    lines = np.column_stack([np.full(len(segments), 2), segments]).ravel()
    return pv.PolyData(nodes.reshape(-1, 3).numpy(), lines=lines)


def taper_displacements(nodes, strength=0.55):
    """Bernstein example: taper the lattice toward the top, growing with z."""
    z = (nodes[..., 2] - ORIGIN[2]) / EXTENT[2]
    shrink = -strength * z.square()
    return torch.stack(
        [shrink * nodes[..., 0], shrink * nodes[..., 1], torch.zeros_like(z)], dim=-1
    )


def sculpt_displacements(nodes):
    """B-spline example with three independently localized shape features."""
    displacements = torch.zeros_like(nodes)
    features = (
        # Center, displacement direction, and compact support radius.
        ((0.75, 0.0, 0.25), (0.9, 0.0, 0.2), 0.95),
        ((0.0, -0.75, -0.2), (0.0, -0.75, -0.2), 0.9),
        ((-0.2, 0.1, 0.75), (0.1, -0.1, -0.7), 0.85),
    )
    for center, direction, radius in features:
        center_t = nodes.new_tensor(center)
        direction_t = nodes.new_tensor(direction)
        distance = torch.linalg.vector_norm(nodes - center_t, dim=-1)
        normalized_distance = (distance / radius).clamp_max(1)
        weight = (1 - normalized_distance.square()).square()
        displacements += weight.unsqueeze(-1) * direction_t
    return displacements


def _render(actors, camera):
    """Render PyVista actors from a fixed camera, so that panels share one scale."""

    plotter = pv.Plotter(off_screen=True, window_size=(900, 900))
    plotter.enable_anti_aliasing("ssaa")
    plotter.set_background("white")
    for dataset, kwargs in actors:
        plotter.add_mesh(dataset, **{"ambient": 0.35, **kwargs})
    plotter.camera_position = camera
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def _surface(mesh):
    """Return the shaded surface actor of a mesh."""

    return (
        to_pyvista(mesh),
        {
            "color": SURFACE,
            "show_edges": True,
            "edge_color": SURFACE_EDGE,
            "line_width": 0.6,
            "diffuse": 0.7,
        },
    )


def _arrows(starts, vectors, color):
    """Draw displacement arrows as solid geometry."""

    arrows = [
        pv.Arrow(
            start=start,
            direction=vector,
            scale=float(np.linalg.norm(vector)),
            tip_radius=0.18,
            shaft_radius=0.06,
            tip_length=0.35,
        )
        for start, vector in zip(starts, vectors, strict=True)
    ]
    return (pv.merge(arrows), {"color": color})


def _panel(mesh, nodes, color, radius):
    """Render a surface inside its control lattice, drawn as tubes and spheres."""

    wireframe = lattice_wireframe(nodes)
    actors = [
        _surface(mesh),
        (wireframe.tube(radius=radius), {"color": color}),
        (spheres(wireframe.points, 2.4 * radius), {"color": color}),
    ]
    return _render(actors, CAMERA)


def main():
    """Generate the documentation image."""

    sphere = sphere_icosahedral.load(radius=1.0, subdivisions=3)

    bernstein_resolution = (4, 4, 4)
    bernstein_nodes = lattice_nodes(bernstein_resolution, "bernstein")
    bernstein_displacements = taper_displacements(bernstein_nodes)
    tapered = sphere.free_form_deform(
        bernstein_displacements,
        origin=list(ORIGIN),
        extent=list(EXTENT),
        basis="bernstein",
    )

    bspline_resolution = (8, 8, 8)
    bspline_nodes = lattice_nodes(bspline_resolution, "bspline")
    bspline_displacements = sculpt_displacements(bspline_nodes)
    sculpted = sphere.free_form_deform(
        bspline_displacements,
        origin=list(ORIGIN),
        extent=list(EXTENT),
        basis="bspline",
    )

    panels = (
        ("sphere, 4x4x4 lattice", sphere, bernstein_nodes, LATTICE, 0.012),
        (
            'basis="bernstein", 4x4x4 lattice',
            tapered,
            bernstein_nodes + bernstein_displacements,
            RESULT,
            0.012,
        ),
        (
            'basis="bspline", 8x8x8 lattice',
            sculpted,
            bspline_nodes + bspline_displacements,
            RESULT,
            0.006,
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 5.3))
    for axis, (title, mesh, nodes, color, radius) in zip(axes, panels, strict=True):
        axis.imshow(_panel(mesh, nodes, color, radius))
        axis.set_title(f"{title}\n{mesh_label(mesh)}")
        axis.set_axis_off()
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.88, wspace=0.02)
    save(figure, OUTPUT)


if __name__ == "__main__":
    main()
