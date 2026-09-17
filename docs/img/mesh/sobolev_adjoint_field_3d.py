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

"""Render raw and Sobolev-filtered upward updates on a sheet."""

from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import to_pyvista

OUTPUT = Path(__file__).parent / "sobolev_adjoint_field_3d.png"

RESOLUTION = 13
LENGTH_SCALE = 0.18
DISPLAY_STEP = 0.52
SCALAR_NAME = "Normalized upward update"
MESH_EDGE = "#3e4a4f"
ARROW_COLOR = "#18272d"
FIXED_COLOR = "#c36a2d"


def _make_sheet() -> tuple[Mesh, torch.Tensor]:
    """Create a triangulated square sheet embedded in 3D."""

    axis = torch.linspace(-1.0, 1.0, RESOLUTION)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack(
        (x.reshape(-1), y.reshape(-1), torch.zeros(RESOLUTION**2)),
        dim=-1,
    )

    cells = []
    for row in range(RESOLUTION - 1):
        for column in range(RESOLUTION - 1):
            lower_left = row * RESOLUTION + column
            lower_right = lower_left + 1
            upper_left = lower_left + RESOLUTION
            upper_right = upper_left + 1
            cells.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )

    connectivity = torch.tensor(cells, dtype=torch.long)
    fixed_points = (
        (points[:, 0] == -1)
        | (points[:, 0] == 1)
        | (points[:, 1] == -1)
        | (points[:, 1] == 1)
    )
    return Mesh(points=points, cells=connectivity), fixed_points


def _target_points(
    mesh: Mesh,
    fixed_points: torch.Tensor,
) -> torch.Tensor:
    """Pull the center upward with deterministic vertex-scale variation."""

    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    boundary_window = torch.cos(0.5 * torch.pi * x)
    boundary_window = boundary_window * torch.cos(0.5 * torch.pi * y)
    bump = torch.exp(-(x.square() + y.square()) / 0.45) * boundary_window

    indices = torch.arange(mesh.n_points)
    checkerboard = (indices // RESOLUTION + indices % RESOLUTION) % 2
    checkerboard = 1.0 - 2.0 * checkerboard.to(mesh.points.dtype)
    upward_pull = bump * (0.30 + 0.08 * checkerboard)
    upward_pull[fixed_points] = 0

    target_points = mesh.points.clone()
    target_points[:, 2] = target_points[:, 2] + upward_pull
    return target_points


def _initial_adjoint(
    mesh: Mesh,
    target_points: torch.Tensor,
    fixed_points: torch.Tensor,
    *,
    use_sobolev: bool,
) -> torch.Tensor:
    """Differentiate the initial objective with respect to candidate vertices."""

    candidate_vertices = mesh.points.detach().clone().requires_grad_()
    raw_displacement = candidate_vertices - mesh.points.detach()
    if use_sobolev:
        deformed = mesh.sobolev_deform(
            raw_displacement,
            length_scale=LENGTH_SCALE,
            fixed_points=fixed_points,
            max_iterations=128,
            implementation="torch",
        )
    else:
        free_displacement = torch.where(
            fixed_points[:, None],
            torch.zeros_like(raw_displacement),
            raw_displacement,
        )
        deformed = mesh.displace(free_displacement)

    objective = (deformed.points - target_points).square().mean()
    return torch.autograd.grad(objective, candidate_vertices)[0]


def _unique_edges(cells: torch.Tensor) -> torch.Tensor:
    """Return undirected triangle edges without duplicates."""

    edges = torch.cat(
        (cells[:, (0, 1)], cells[:, (1, 2)], cells[:, (2, 0)]),
        dim=0,
    )
    return torch.unique(torch.sort(edges, dim=1).values, dim=0)


def _roughness(field: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Measure neighbor variation after removing the field magnitude."""

    root_mean_square = field.square().mean().sqrt()
    root_mean_square = root_mean_square.clamp_min(torch.finfo(field.dtype).eps)
    normalized = field / root_mean_square
    differences = normalized[edges[:, 0]] - normalized[edges[:, 1]]
    return differences.square().sum(dim=-1).mean()


def main() -> None:
    """Generate the documentation image."""

    mesh, fixed_points = _make_sheet()
    target_points = _target_points(mesh, fixed_points)
    raw_adjoint = _initial_adjoint(
        mesh,
        target_points,
        fixed_points,
        use_sobolev=False,
    )
    sobolev_adjoint = _initial_adjoint(
        mesh,
        target_points,
        fixed_points,
        use_sobolev=True,
    )

    edges = _unique_edges(mesh.cells)
    raw_roughness = _roughness(raw_adjoint, edges)
    sobolev_roughness = _roughness(sobolev_adjoint, edges)
    if sobolev_roughness >= raw_roughness:
        raise RuntimeError("Sobolev filtering did not reduce adjoint roughness")

    raw_update = -raw_adjoint
    sobolev_update = -sobolev_adjoint
    shared_scale = torch.stack(
        (
            torch.linalg.vector_norm(raw_update, dim=-1).max(),
            torch.linalg.vector_norm(sobolev_update, dim=-1).max(),
        )
    ).max()
    shared_scale = shared_scale.clamp_min(torch.finfo(mesh.points.dtype).eps)
    raw_update = raw_update / shared_scale
    sobolev_update = sobolev_update / shared_scale

    points = mesh.points.numpy()
    fixed = fixed_points.numpy()
    grid = np.arange(mesh.n_points).reshape(RESOLUTION, RESOLUTION)
    selected = grid[1:-1:2, 1:-1:2].reshape(-1)
    panels = (
        (
            "Before: raw upward update",
            raw_update.numpy(),
            raw_roughness.item(),
        ),
        (
            "After: Sobolev-filtered update",
            sobolev_update.numpy(),
            sobolev_roughness.item(),
        ),
    )

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 500))
    plotter.enable_anti_aliasing("ssaa")
    for column, (title, update, roughness) in enumerate(panels):
        deformed_points = points + DISPLAY_STEP * update
        panel_mesh = to_pyvista(mesh)
        panel_mesh.points = deformed_points
        panel_mesh.point_data[SCALAR_NAME] = update[:, 2]

        plotter.subplot(0, column)
        plotter.add_mesh(
            panel_mesh,
            scalars=SCALAR_NAME,
            cmap="OrRd",
            clim=(0.0, 1.0),
            show_edges=True,
            edge_color=MESH_EDGE,
            line_width=0.55,
            smooth_shading=True,
            ambient=0.3,
            diffuse=0.7,
            show_scalar_bar=column == 1,
            scalar_bar_args={
                "title": "",
                "color": "black",
                "label_font_size": 10,
                "vertical": False,
                "position_x": 0.31,
                "position_y": 0.02,
                "width": 0.62,
                "height": 0.07,
                "n_labels": 3,
            },
        )
        plotter.add_arrows(
            deformed_points[selected] + np.array((0.0, 0.0, 0.012)),
            update[selected],
            mag=0.13,
            color=ARROW_COLOR,
        )
        plotter.add_points(
            points[fixed] + np.array((0.0, 0.0, 0.012)),
            color=FIXED_COLOR,
            point_size=8.0,
            render_points_as_spheres=True,
        )
        plotter.add_text(
            f"{title}\nnormalized edge roughness  {roughness:.2f}",
            position="upper_left",
            font_size=12,
            color="black",
        )
        if column == 0:
            plotter.add_text(
                "● fixed boundary",
                position="lower_left",
                font_size=9,
                color=FIXED_COLOR,
            )
        else:
            plotter.add_text(
                "normalized upward update",
                position="lower_left",
                font_size=9,
                color="#35444a",
            )
        plotter.camera_position = [
            (2.8, -4.0, 2.4),
            (0.0, 0.0, 0.12),
            (0.0, 0.0, 1.0),
        ]
        plotter.camera.parallel_projection = True
        plotter.camera.zoom(1.02)

    plotter.link_views()
    plotter.set_background("white")
    plotter.render()
    plotter.screenshot(OUTPUT, transparent_background=False)
    plotter.close()

    print(f"Raw adjoint roughness: {raw_roughness.item():.4f}")
    print(f"Sobolev adjoint roughness: {sobolev_roughness.item():.4f}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
