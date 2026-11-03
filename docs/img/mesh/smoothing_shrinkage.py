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

"""Show how Laplacian smoothing shrinks a closed surface, and faster on coarse meshes."""

from pathlib import Path

import matplotlib.pyplot as plt
from _style import (
    FOURTH,
    RESULT,
    SECOND,
    SURFACE,
    SURFACE_EDGE,
    TEXT,
    VIEW_UP,
    VIEW_VECTOR,
    save,
)

from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives.procedural import lumpy_sphere
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.mesh.smoothing import smooth_laplacian

OUTPUT = Path(__file__).parent / "smoothing_shrinkage.png"

RELAXATION = 0.1
SNAPSHOTS = (0, 50, 500)
CURVE_ITERS = (0, 10, 25, 50, 100, 150, 200, 300, 400, 500)
LEVELS = {1: FOURTH, 2: SECOND, 3: RESULT}

# One fixed camera for every snapshot, so that the shrinkage shows.
CAMERA_DISTANCE = 3.6


def _render(mesh):
    """Render a surface off-screen from the shared fixed camera."""

    import pyvista as pv

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
    plotter.enable_anti_aliasing("ssaa")
    plotter.set_background("white")
    # Outward normals light the outside of the closed surface.
    plotter.add_mesh(
        to_pyvista(mesh).compute_normals(auto_orient_normals=True),
        color=SURFACE,
        show_edges=True,
        edge_color=SURFACE_EDGE,
        line_width=0.6,
        ambient=0.5,
        diffuse=0.7,
    )
    norm = sum(v * v for v in VIEW_VECTOR) ** 0.5
    eye = tuple(CAMERA_DISTANCE * v / norm for v in VIEW_VECTOR)
    plotter.camera_position = [eye, (0.0, 0.0, 0.0), VIEW_UP]
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def _mean_radius(mesh):
    """Mean distance of the points from the origin."""

    return float(mesh.points.norm(dim=-1).mean())


def main():
    """Generate the documentation image."""

    lumpy = lumpy_sphere.load(subdivisions=3, seed=0)
    snapshots = [
        lumpy
        if n_iter == 0
        else smooth_laplacian(lumpy, n_iter=n_iter, relaxation_factor=RELAXATION)
        for n_iter in SNAPSHOTS
    ]

    curves = {}
    for level in LEVELS:
        sphere = sphere_icosahedral.load(subdivisions=level)
        curves[level] = (
            sphere,
            [
                _mean_radius(
                    sphere
                    if n_iter == 0
                    else smooth_laplacian(
                        sphere, n_iter=n_iter, relaxation_factor=RELAXATION
                    )
                )
                for n_iter in CURVE_ITERS
            ],
        )

    figure = plt.figure(figsize=(14.0, 4.3))
    grid = figure.add_gridspec(
        1, 4, width_ratios=[1.0, 1.0, 1.0, 1.25], wspace=0.04, left=0.0, right=0.985
    )
    for column, (n_iter, mesh) in enumerate(zip(SNAPSHOTS, snapshots, strict=True)):
        axis = figure.add_subplot(grid[0, column])
        axis.imshow(_render(mesh)[60:-60, 60:-60])
        first = "lumpy_sphere, " if n_iter == 0 else ""
        axis.set_title(f"{first}n_iter={n_iter}\nmean radius {_mean_radius(mesh):.2f}")
        axis.set_axis_off()

    axis = figure.add_subplot(grid[0, 3])
    for level, color in LEVELS.items():
        sphere, radii = curves[level]
        axis.plot(
            CURVE_ITERS,
            radii,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=4,
            label=f"{sphere.n_cells} triangles",
        )
    final = curves[3][1][-1]
    axis.annotate(
        f"{final:.2f}",
        (CURVE_ITERS[-1], final),
        xytext=(-8, -16),
        textcoords="offset points",
        ha="right",
        color=RESULT,
    )
    axis.set_xlabel("n_iter")
    axis.set_ylabel("mean radius")
    axis.set_ylim(0.0, 1.05)
    axis.set_title(
        f"sphere_icosahedral, subdivisions 1, 2, 3\nrelaxation_factor={RELAXATION}"
    )
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        handles[::-1], labels[::-1], loc="upper right", frameon=False, fontsize=12
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=TEXT)

    figure.subplots_adjust(top=0.84, bottom=0.14)
    save(figure, OUTPUT)


if __name__ == "__main__":
    main()
