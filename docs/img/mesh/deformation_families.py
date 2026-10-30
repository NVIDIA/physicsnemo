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

"""Apply the six deformation families to one flat sheet, seen from one camera."""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from _style import (
    ACCENT_LIGHTING,
    EDGE,
    FIXED,
    RESULT,
    SURFACE,
    SURFACE_EDGE,
    VIEW_UP,
    VIEW_VECTOR,
    save,
    spheres,
)
from matplotlib.lines import Line2D

from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives.surfaces import plane, sphere_icosahedral

OUTPUT = Path(__file__).parent / "deformation_families.png"

SEED = 0
HANDLE = (0.3, 0.2)
LIFT = 0.55
RADIUS = 0.6
# One camera for every panel, so that the shapes compare directly.
FOCUS = (0.0, 0.0, -0.12)
CAMERA_DISTANCE = 5.1
SPHERE_RADIUS = 0.04
TARGET = "#c5ccd1"  # the shrinkwrap target, behind the result


def _render(actors):
    """Render PyVista actors off-screen from the shared fixed camera."""

    import pyvista as pv

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(900, 720))
    plotter.enable_anti_aliasing("ssaa")
    plotter.set_background("white")
    for dataset, kwargs in actors:
        plotter.add_mesh(dataset, **{"ambient": 0.35, **kwargs})
    direction = np.asarray(VIEW_VECTOR) / np.linalg.norm(VIEW_VECTOR)
    eye = tuple(np.asarray(FOCUS) + CAMERA_DISTANCE * direction)
    plotter.camera_position = [eye, FOCUS, VIEW_UP]
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def _segments(starts, ends, radius):
    """Draw straight segments between paired points as tubes."""

    import pyvista as pv

    starts, ends = np.asarray(starts), np.asarray(ends)
    points = np.concatenate([starts, ends])
    n = len(starts)
    lines = np.column_stack([np.full(n, 2), np.arange(n), np.arange(n) + n]).ravel()
    return pv.PolyData(points, lines=lines).tube(radius=radius)


def main():
    """Generate the documentation image."""

    torch.manual_seed(SEED)
    sheet = plane.load(size=2.0, subdivisions=16)
    points = sheet.points
    x, y = points[:, 0], points[:, 1]
    edge = (x.abs() > 0.999) | (y.abs() > 0.999)
    up = torch.tensor([0.0, 0.0, 1.0])

    # A dense field: a smooth bump plus seeded noise, one vector per point.
    bump = 0.4 * torch.exp(-((x - HANDLE[0]) ** 2 + (y - HANDLE[1]) ** 2) / 0.25)
    noise = 0.035 * torch.randn(sheet.n_points)
    d = (bump + noise)[:, None] * up

    handle = torch.tensor([[HANDLE[0], HANDLE[1], 0.0]])
    lift = torch.tensor([[0.0, 0.0, LIFT]])
    corners = torch.tensor([[sx, sy, 0.0] for sx, sy in product((-1.0, 1.0), repeat=2)])
    # The corners and the handle are coplanar, which makes the affine tail singular.
    # A sixth fixed control below the sheet keeps the fit well posed.
    anchors = torch.cat([corners, torch.tensor([[0.0, -1.2, -0.3]])])
    controls = torch.cat([anchors, handle])
    control_moves = torch.cat([torch.zeros(len(anchors), 3), lift])

    # A 4 x 4 x 2 Bernstein lattice. The sheet is flat, so the box needs an explicit
    # positive z extent.
    origin = torch.tensor([-1.0, -1.0, -0.25])
    extent = torch.tensor([2.0, 2.0, 0.5])
    resolution = (4, 4, 2)
    lattice = torch.zeros(*resolution, 3)
    lattice[1:3, 1:3, :, 2] = 0.5

    target = sphere_icosahedral.load(radius=1.6, subdivisions=4).translate(
        torch.tensor([0.0, 0.0, -1.3])
    )
    # Drawn slightly inside the target, so that the wrapped sheet stays visible on it.
    backdrop = sphere_icosahedral.load(radius=1.585, subdivisions=4).translate(
        torch.tensor([0.0, 0.0, -1.3])
    )

    results = [
        sheet.displace(d),
        sheet.sobolev_deform(d, length_scale=0.3, fixed_points=edge),
        sheet.morph(handle, lift, radius=RADIUS),
        sheet.radial_basis_function_deform(controls, control_moves),
        sheet.free_form_deform(lattice, origin=origin, extent=extent),
        sheet.shrinkwrap(target),
    ]

    import pyvista as pv

    surface = {"color": SURFACE, "show_edges": True, "edge_color": SURFACE_EDGE}
    lifted = (handle + lift).numpy()
    accent = {"color": RESULT, **ACCENT_LIGHTING}
    fixed = {"color": FIXED, **ACCENT_LIGHTING}
    handle_actor = (spheres(lifted, 2.2 * SPHERE_RADIUS), accent)
    lift_actor = (_segments(handle.numpy(), lifted, 0.012), accent)

    edge_points = results[1].points[edge].numpy()
    # The compact support of the handle, drawn as a ring on the flat part of the sheet.
    angles = np.linspace(0.0, 2.0 * np.pi, 97)
    ring = np.column_stack(
        [HANDLE[0] + RADIUS * np.cos(angles), HANDLE[1] + RADIUS * np.sin(angles)]
    )
    ring = np.column_stack([ring, np.full(len(angles), 0.004)])
    support = _segments(ring[:-1], ring[1:], 0.011)

    nodes = np.stack(
        np.meshgrid(*[np.linspace(0.0, 1.0, n) for n in resolution], indexing="ij"), -1
    )
    nodes = origin.numpy() + extent.numpy() * nodes + lattice.numpy()
    starts, ends = [], []
    for axis in range(3):
        head = [slice(None)] * 3
        tail = [slice(None)] * 3
        head[axis], tail[axis] = slice(None, -1), slice(1, None)
        starts.append(nodes[tuple(head)].reshape(-1, 3))
        ends.append(nodes[tuple(tail)].reshape(-1, 3))
    cage = _segments(np.concatenate(starts), np.concatenate(ends), 0.008)

    target_pv = to_pyvista(backdrop)
    cap = target_pv.clip(normal="z", origin=(0.0, 0.0, -0.4), invert=False)

    extras = [
        [],
        [(spheres(edge_points, 0.7 * SPHERE_RADIUS), fixed)],
        [
            (support, accent),
            handle_actor,
            lift_actor,
        ],
        [
            (spheres(anchors.numpy(), 2.2 * SPHERE_RADIUS), fixed),
            handle_actor,
            lift_actor,
        ],
        [
            (cage, {"color": EDGE}),
            (spheres(nodes.reshape(-1, 3), 1.1 * SPHERE_RADIUS), accent),
        ],
        [(cap, {"color": TARGET, "show_edges": False})],
    ]
    titles = [
        ("displace(d)", "dense, one vector per point"),
        ("sobolev_deform(d, length_scale=0.3)", "dense, smoothed, boundary fixed"),
        (f"morph(c, dc, radius={RADIUS})", "compact, sparse handles"),
        ("radial_basis_function_deform(c, dc)", "global, exact at controls"),
        ("free_form_deform(lattice)", "lattice of controls"),
        ("shrinkwrap(target)", "conform to a surface"),
    ]

    figure, axes = plt.subplots(2, 3, figsize=(14.0, 8.6))
    for axis, mesh, extra, (call, family) in zip(
        axes.flat, results, extras, titles, strict=True
    ):
        image = _render([(to_pyvista(mesh), surface), *extra])
        axis.imshow(image[20:-20, 20:-20])
        axis.set_title(f"{call}\n{family}")
        axis.set_axis_off()

    legend = [
        Line2D([], [], marker="o", linestyle="", color=RESULT, markersize=10),
        Line2D([], [], marker="o", linestyle="", color=FIXED, markersize=10),
        Line2D([], [], color=EDGE, linewidth=3),
        Line2D([], [], marker="s", linestyle="", color=TARGET, markersize=12),
    ]
    figure.legend(
        legend,
        ["control or handle", "fixed", "FFD lattice", "shrinkwrap target"],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.text(
        0.5,
        0.055,
        f"Every panel deforms the same plane.load(size=2.0, subdivisions=16), "
        f"Mesh[{sheet.n_manifold_dims}, {sheet.n_spatial_dims}], {sheet.n_cells} triangles.",
        ha="center",
    )
    figure.subplots_adjust(
        left=0.0, right=1.0, bottom=0.09, top=0.93, wspace=0.02, hspace=0.18
    )
    save(figure, OUTPUT)


if __name__ == "__main__":
    main()
