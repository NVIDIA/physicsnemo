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

"""Render raw and Sobolev-filtered vertex adjoint fields."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import torch
from matplotlib.colors import TwoSlopeNorm

from physicsnemo.mesh import Mesh

OUTPUT = Path(__file__).parent / "sobolev_adjoint_field.png"

NVIDIA_GREEN = "#76B900"
MESH_EDGE = "#263238"
ARROW_COLOR = "#17242a"
FIXED_EDGE = "#49575d"


def _make_square(resolution: int = 9) -> tuple[Mesh, torch.Tensor]:
    """Create a triangulated unit square and its boundary mask."""

    axis = torch.linspace(0.0, 1.0, resolution)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)

    cells = []
    for row in range(resolution - 1):
        for column in range(resolution - 1):
            lower_left = row * resolution + column
            lower_right = lower_left + 1
            upper_left = lower_left + resolution
            upper_right = upper_left + 1
            cells.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )

    connectivity = torch.tensor(cells, dtype=torch.long)
    fixed_points = (
        (points[:, 0] == 0)
        | (points[:, 0] == 1)
        | (points[:, 1] == 0)
        | (points[:, 1] == 1)
    )
    return Mesh(points=points, cells=connectivity), fixed_points


def _target_points(
    mesh: Mesh,
    resolution: int,
    fixed_points: torch.Tensor,
) -> torch.Tensor:
    """Create a smooth target with deterministic vertex-scale variation."""

    centered = mesh.points - 0.5
    radius_squared = centered.square().sum(dim=-1)
    boundary_window = torch.sin(torch.pi * mesh.points[:, 0])
    boundary_window = boundary_window * torch.sin(torch.pi * mesh.points[:, 1])

    indices = torch.arange(mesh.n_points)
    checkerboard = (indices // resolution + indices % resolution) % 2
    checkerboard = 1.0 - 2.0 * checkerboard.to(mesh.points.dtype)

    vertical_target = 0.15 * torch.exp(-radius_squared / 0.06)
    vertical_target = vertical_target + 0.04 * checkerboard * boundary_window
    target_displacement = torch.zeros_like(mesh.points)
    target_displacement[:, 1] = vertical_target
    target_displacement[fixed_points] = 0
    return mesh.points + target_displacement


def _initial_vertex_adjoint(
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
            length_scale=0.15,
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


def _normalized_edge_roughness(
    field: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Measure neighbor variation after removing the field magnitude."""

    root_mean_square = field.square().mean().sqrt()
    root_mean_square = root_mean_square.clamp_min(torch.finfo(field.dtype).eps)
    normalized = field / root_mean_square
    differences = normalized[edges[:, 0]] - normalized[edges[:, 1]]
    return differences.square().sum(dim=-1).mean()


def _draw_field(
    axis,
    triangulation: mtri.Triangulation,
    points,
    normalized_field,
    fixed_points,
    *,
    title: str,
    roughness: float,
    norm: TwoSlopeNorm,
):
    """Draw one vertex field with shared color and arrow scales."""

    surface = axis.tripcolor(
        triangulation,
        normalized_field[:, 1],
        shading="gouraud",
        cmap="coolwarm",
        norm=norm,
    )
    axis.triplot(
        triangulation,
        color=MESH_EDGE,
        linewidth=0.45,
        alpha=0.28,
    )

    free_points = ~fixed_points
    arrow_field = 0.075 * normalized_field[free_points]
    axis.quiver(
        points[free_points, 0],
        points[free_points, 1],
        arrow_field[:, 0],
        arrow_field[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0048,
        headwidth=3.7,
        headlength=4.8,
        color=ARROW_COLOR,
        alpha=0.9,
        zorder=4,
    )
    axis.scatter(
        points[fixed_points, 0],
        points[fixed_points, 1],
        s=20,
        facecolor="white",
        edgecolor=FIXED_EDGE,
        linewidth=0.8,
        zorder=5,
    )

    axis.set_title(title, fontsize=15, fontweight="bold", pad=18)
    axis.text(
        0.5,
        1.015,
        f"normalized edge roughness  {roughness:.2f}",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#4c5b61",
    )
    axis.set_xlim(-0.055, 1.055)
    axis.set_ylim(-0.055, 1.055)
    axis.set_aspect("equal")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_xticks((0.0, 0.5, 1.0))
    axis.set_yticks((0.0, 0.5, 1.0))
    axis.tick_params(colors="#4c5b61")
    axis.spines[["top", "right"]].set_visible(False)
    return surface


def main() -> None:
    """Generate the documentation image."""

    resolution = 9
    mesh, fixed_points = _make_square(resolution)
    target_points = _target_points(mesh, resolution, fixed_points)
    raw_adjoint = _initial_vertex_adjoint(
        mesh,
        target_points,
        fixed_points,
        use_sobolev=False,
    )
    sobolev_adjoint = _initial_vertex_adjoint(
        mesh,
        target_points,
        fixed_points,
        use_sobolev=True,
    )

    edges = _unique_edges(mesh.cells)
    raw_roughness = _normalized_edge_roughness(raw_adjoint, edges)
    sobolev_roughness = _normalized_edge_roughness(sobolev_adjoint, edges)
    if sobolev_roughness >= raw_roughness:
        raise RuntimeError("Sobolev filtering did not reduce adjoint roughness")

    shared_scale = torch.stack((raw_adjoint.abs().max(), sobolev_adjoint.abs().max()))
    shared_scale = shared_scale.max().clamp_min(torch.finfo(mesh.points.dtype).eps)
    raw_normalized = raw_adjoint / shared_scale
    sobolev_normalized = sobolev_adjoint / shared_scale

    points = mesh.points.detach().cpu().numpy()
    cells = mesh.cells.detach().cpu().numpy()
    fixed = fixed_points.detach().cpu().numpy()
    raw_field = raw_normalized.detach().cpu().numpy()
    sobolev_field = sobolev_normalized.detach().cpu().numpy()
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.7), dpi=160)
    panels = (
        (
            axes[0],
            raw_field,
            "Before: raw vertex adjoint",
            raw_roughness.item(),
        ),
        (
            axes[1],
            sobolev_field,
            "After: Sobolev-filtered adjoint",
            sobolev_roughness.item(),
        ),
    )
    surface = None
    for axis, field, title, roughness in panels:
        surface = _draw_field(
            axis,
            triangulation,
            points,
            field,
            fixed,
            title=title,
            roughness=roughness,
            norm=norm,
        )

    figure.suptitle(
        "Sobolev filtering removes vertex-scale adjoint oscillation",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.49,
        0.53,
        "→",
        ha="center",
        va="center",
        fontsize=28,
        color=NVIDIA_GREEN,
        fontweight="bold",
    )
    figure.text(
        0.46,
        0.025,
        "Colors and arrows share one scale. Open circles mark fixed vertices.",
        ha="center",
        fontsize=10.5,
        color="#4c5b61",
    )
    if surface is None:
        raise RuntimeError("No field panels were rendered")
    colorbar = figure.colorbar(
        surface,
        ax=axes,
        fraction=0.036,
        pad=0.045,
        shrink=0.82,
    )
    colorbar.set_label("Normalized vertical adjoint", fontsize=11)
    colorbar.ax.tick_params(labelsize=9.5)

    figure.subplots_adjust(
        left=0.065,
        right=0.9,
        bottom=0.12,
        top=0.84,
        wspace=0.25,
    )
    figure.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Raw adjoint roughness: {raw_roughness.item():.4f}")
    print(f"Sobolev adjoint roughness: {sobolev_roughness.item():.4f}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
