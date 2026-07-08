# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate deterministic figures for field smoothing and guarded updates.

Run from the repository root with PhysicsNeMo mesh extras installed::

    python docs/img/mesh/generate_guarded_normal_update.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.smoothing import smooth_point_field

OUTPUT_DIR = Path(__file__).parent


def _grid_mesh(n: int, *, surface: bool) -> Mesh:
    """Create a consistently oriented triangular grid."""
    axis = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    if surface:
        zz = 0.12 * torch.sin(math.pi * xx) * torch.sin(math.pi * yy)
        points = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=-1)
    else:
        points = torch.stack((xx.flatten(), yy.flatten()), dim=-1)

    cells: list[tuple[int, int, int]] = []
    for row in range(n - 1):
        for column in range(n - 1):
            lower_left = row * n + column
            lower_right = lower_left + 1
            upper_left = lower_left + n
            upper_right = upper_left + 1
            if (row + column) % 2 == 0:
                cells.extend(
                    (
                        (lower_left, lower_right, upper_right),
                        (lower_left, upper_right, upper_left),
                    )
                )
            else:
                cells.extend(
                    (
                        (lower_left, lower_right, upper_left),
                        (lower_right, upper_right, upper_left),
                    )
                )
    return Mesh(points=points, cells=torch.tensor(cells, dtype=torch.int64))


def _save_field_smoothing() -> None:
    """Evaluate and render ``smooth_point_field``."""
    mesh = _grid_mesh(27, surface=False)
    x, y = mesh.points.unbind(dim=-1)
    checker = torch.sin(10.0 * x + 1.2) * torch.sin(9.0 * y - 0.4)
    broad_signal = 1.4 * torch.exp(-3.5 * ((x + 0.25) ** 2 + (y - 0.1) ** 2))
    field = broad_signal + 0.65 * checker
    smoothed = smooth_point_field(mesh, field, n_iter=12, relaxation_factor=0.22)
    limit = float(torch.stack((field.abs().amax(), smoothed.abs().amax())).amax())

    triangulation = mtri.Triangulation(
        x.numpy(), y.numpy(), triangles=mesh.cells.numpy()
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, values, title in zip(
        axes,
        (field, smoothed),
        ("Input Point Field", "Smoothed Point Field"),
        strict=True,
    ):
        image = axis.tripcolor(
            triangulation,
            values.numpy(),
            shading="gouraud",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.triplot(triangulation, color="white", linewidth=0.2, alpha=0.75)
        axis.set_aspect("equal")
        axis.set_title(title, fontsize=17, weight="bold")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    figure.colorbar(image, ax=axes, shrink=0.86, label="point-field value")
    figure.savefig(OUTPUT_DIR / "smooth_point_field.png", dpi=150)
    plt.close(figure)


def _save_guarded_update() -> None:
    """Evaluate and render ``Mesh.guarded_normal_update``."""
    mesh = _grid_mesh(23, surface=True)
    x, y, _ = mesh.points.unbind(dim=-1)
    direction = torch.zeros_like(mesh.points)
    direction[:, 2] = 0.32 * torch.exp(-2.8 * ((x - 0.25) ** 2 + y**2))
    weights = ((y + 0.9) / 0.35).clamp(0.0, 1.0)
    updated, diagnostics = mesh.guarded_normal_update(
        direction,
        point_weights=weights,
        smoothing_iterations=5,
        smoothing_relaxation=0.2,
        max_step=0.18,
    )
    if not diagnostics.accepted:
        raise RuntimeError(
            f"Documentation update was rejected: {diagnostics.validation}"
        )

    displacement = (updated.points - mesh.points).norm(dim=-1)
    value_max = max(float(displacement.amax()), 1.0e-12)
    figure = plt.figure(figsize=(12, 5), constrained_layout=True)
    axes = [figure.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
    for axis, current, values, title in zip(
        axes,
        (mesh, updated),
        (torch.zeros_like(displacement), displacement),
        ("Original Surface", "Guarded Normal Update"),
        strict=True,
    ):
        points = current.points.numpy()
        surface = axis.plot_trisurf(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            triangles=current.cells.numpy(),
            cmap="viridis",
            linewidth=0.2,
            edgecolor="white",
            antialiased=True,
            shade=False,
        )
        surface.set_array(values[current.cells].mean(dim=-1).numpy())
        surface.set_clim(0.0, value_max)
        axis.view_init(elev=27, azim=-55)
        axis.set_box_aspect((2.0, 2.0, 0.8))
        axis.set_title(title, fontsize=17, weight="bold")
        axis.set_axis_off()
    colorbar = figure.colorbar(
        ScalarMappable(norm=Normalize(0.0, value_max), cmap="viridis"),
        ax=axes,
        shrink=0.78,
    )
    colorbar.set_label("accepted displacement magnitude")
    figure.savefig(OUTPUT_DIR / "guarded_normal_update.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """Generate both public-operation figures."""
    with torch.no_grad():
        _save_field_smoothing()
        _save_guarded_update()
    print(f"Saved mesh API figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
