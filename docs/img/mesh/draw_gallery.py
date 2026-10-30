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

"""Render four Mesh.draw calls side by side, each drawn by Mesh.draw itself."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pyvista as pv
import torch

sys.path.insert(0, str(Path(__file__).parent))
import _style as S  # noqa: E402

from physicsnemo.mesh.primitives.planar import annulus_2d  # noqa: E402
from physicsnemo.mesh.primitives.procedural import lumpy_sphere  # noqa: E402

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "draw_gallery.png"


def main():
    """Generate the documentation image."""

    planar = annulus_2d.load(n_radial=4, n_angular=32)
    x, y = planar.points.unbind(dim=-1)
    planar.point_data["temperature"] = x**2 - 0.5 * y
    angle = torch.atan2(planar.cell_centroids[:, 1], planar.cell_centroids[:, 0])
    planar.cell_data["pressure"] = torch.cos(2 * angle)
    planar.point_data["velocity"] = torch.stack([-y, x], dim=-1) * (1.0 + x)[:, None]

    surface = lumpy_sphere.load(subdivisions=3, noise_amplitude=0.4, seed=1)
    surface.point_data["height"] = surface.points[:, 2]

    figure, grid = plt.subplots(2, 2, figsize=(12.0, 9.6))
    axes = grid.ravel()

    planar.draw(show=False, ax=axes[0], cell_scalars="pressure")
    axes[0].set_title(
        f'draw(cell_scalars="pressure")\n{S.mesh_label(planar, "triangles")}'
    )

    planar.draw(show=False, ax=axes[1], point_scalars="temperature")
    axes[1].set_title(
        f'draw(point_scalars="temperature")\n'
        f"Mesh[{planar.n_manifold_dims}, {planar.n_spatial_dims}], {planar.n_points} points"
    )

    # backend="pyvista" draws on a Plotter. Its image goes into the matplotlib panel.
    plotter = pv.Plotter(off_screen=True, window_size=(900, 900))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    surface.draw(
        backend="pyvista",
        show=False,
        ax=plotter,
        point_scalars="height",
        alpha_points=0.0,
        backend_options={"show_scalar_bar": False},
    )
    plotter.view_vector(S.VIEW_VECTOR, viewup=S.VIEW_UP)
    plotter.reset_camera()
    plotter.camera.zoom(1.35)
    axes[2].imshow(plotter.screenshot(return_img=True))
    plotter.close()
    height = surface.point_data["height"]
    figure.colorbar(
        plt.cm.ScalarMappable(
            plt.Normalize(float(height.min()), float(height.max())), S.SEQUENTIAL
        ),
        ax=axes[2],
        label="height",
    )
    axes[2].set_title(
        f'draw(point_scalars="height", backend="pyvista")\n'
        f"{S.mesh_label(surface, 'triangles')}"
    )

    planar.draw(show=False, ax=axes[3], point_scalars="velocity")
    axes[3].set_title(
        f'draw(point_scalars="velocity")\n'
        f"|velocity| of a {tuple(planar.point_data['velocity'].shape)} field"
    )

    for axis in axes:
        S.clean_axis(axis)
    figure.subplots_adjust(
        left=0.01, right=0.95, bottom=0.02, top=0.93, wspace=0.12, hspace=0.22
    )
    S.save(figure, OUTPUT, tight=True)


if __name__ == "__main__":
    main()
