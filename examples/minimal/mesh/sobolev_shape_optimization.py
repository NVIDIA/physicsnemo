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

"""Compare dense and Sobolev per-vertex shape optimization.

The target combines a smooth bump with an alternating vertex-scale
perturbation. Direct displacement passes this high-frequency residual into its
adjoint. Sobolev deformation solves a uniform-mass P1 Helmholtz problem in
the forward pass, so autograd applies the corresponding smooth adjoint.
"""

from __future__ import annotations

import torch

from physicsnemo.mesh import Mesh


def make_square(
    resolution: int,
    device: torch.device,
) -> tuple[Mesh, torch.Tensor]:
    """Create a triangulated unit square and its boundary mask."""

    axis = torch.linspace(0.0, 1.0, resolution, device=device)
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
    connectivity = torch.tensor(cells, dtype=torch.long, device=device)

    fixed_points = (
        (points[:, 0] == 0)
        | (points[:, 0] == 1)
        | (points[:, 1] == 0)
        | (points[:, 1] == 1)
    )
    return Mesh(points=points, cells=connectivity), fixed_points


def make_target(
    mesh: Mesh, resolution: int, fixed_points: torch.Tensor
) -> torch.Tensor:
    """Create a smooth vertical bump with deterministic vertex-scale noise."""

    centered = mesh.points - 0.5
    radius_squared = centered.square().sum(dim=-1)
    boundary_window = torch.sin(torch.pi * mesh.points[:, 0])
    boundary_window = boundary_window * torch.sin(torch.pi * mesh.points[:, 1])

    indices = torch.arange(mesh.n_points, device=mesh.points.device)
    checkerboard = (indices // resolution + indices % resolution) % 2
    checkerboard = 1.0 - 2.0 * checkerboard.to(mesh.points.dtype)

    vertical_target = 0.15 * torch.exp(-radius_squared / 0.06)
    vertical_target = vertical_target + 0.04 * checkerboard * boundary_window
    target_displacement = torch.zeros_like(mesh.points)
    target_displacement[:, 1] = vertical_target
    target_displacement[fixed_points] = 0
    return mesh.points + target_displacement


def unique_edges(cells: torch.Tensor) -> torch.Tensor:
    """Return undirected triangle edges without duplicates."""

    edges = torch.cat(
        (cells[:, (0, 1)], cells[:, (1, 2)], cells[:, (2, 0)]),
        dim=0,
    )
    return torch.unique(torch.sort(edges, dim=1).values, dim=0)


def normalized_edge_roughness(field: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Measure neighbor variation after removing the field magnitude."""

    root_mean_square = field.square().mean().sqrt()
    root_mean_square = root_mean_square.clamp_min(torch.finfo(field.dtype).eps)
    normalized = field / root_mean_square
    edge_difference = normalized[edges[:, 0]] - normalized[edges[:, 1]]
    return edge_difference.square().sum(dim=-1).mean()


def objective(
    mesh: Mesh,
    design_vertices: torch.Tensor,
    target_points: torch.Tensor,
    fixed_points: torch.Tensor,
    use_sobolev: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate either direct or Sobolev deformation."""

    raw_displacement = design_vertices - mesh.points.detach()
    if use_sobolev:
        deformed = mesh.sobolev_deform(
            raw_displacement,
            length_scale=0.18,
            fixed_points=fixed_points,
            max_iterations=64,
        )
    else:
        free_displacement = torch.where(
            fixed_points[:, None],
            torch.zeros_like(raw_displacement),
            raw_displacement,
        )
        deformed = mesh.displace(free_displacement)

    loss = (deformed.points - target_points).square().mean()
    return loss, deformed.points


def optimize(
    mesh: Mesh,
    target_points: torch.Tensor,
    fixed_points: torch.Tensor,
    use_sobolev: bool,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """Optimize candidate vertex coordinates and retain their initial adjoint."""

    design_vertices = torch.nn.Parameter(mesh.points.detach().clone())
    initial_loss, _ = objective(
        mesh,
        design_vertices,
        target_points,
        fixed_points,
        use_sobolev,
    )
    initial_loss.backward()
    initial_adjoint = design_vertices.grad.detach().clone()

    optimizer = torch.optim.Adam((design_vertices,), lr=0.08)
    for _ in range(50):
        optimizer.zero_grad()
        loss, _ = objective(
            mesh,
            design_vertices,
            target_points,
            fixed_points,
            use_sobolev,
        )
        loss.backward()
        optimizer.step()

    final_loss, final_points = objective(
        mesh,
        design_vertices,
        target_points,
        fixed_points,
        use_sobolev,
    )
    return (
        initial_loss.detach().item(),
        final_loss.detach().item(),
        initial_adjoint,
        final_points.detach(),
    )


def main() -> None:
    """Run both optimizations and verify their observable behavior."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolution = 7
    mesh, fixed_points = make_square(resolution, device)
    target_points = make_target(mesh, resolution, fixed_points)
    edges = unique_edges(mesh.cells)

    dense = optimize(mesh, target_points, fixed_points, use_sobolev=False)
    sobolev = optimize(mesh, target_points, fixed_points, use_sobolev=True)

    dense_roughness = normalized_edge_roughness(dense[2], edges)
    sobolev_roughness = normalized_edge_roughness(sobolev[2], edges)

    if dense[1] >= 0.1 * dense[0]:
        raise RuntimeError("dense displacement optimization did not reduce its loss")
    if sobolev[1] >= 0.1 * sobolev[0]:
        raise RuntimeError("Sobolev deformation optimization did not reduce its loss")
    if sobolev_roughness >= 0.6 * dense_roughness:
        raise RuntimeError(
            "the Sobolev adjoint was not smoother than the dense adjoint"
        )

    torch.testing.assert_close(sobolev[3][fixed_points], mesh.points[fixed_points])

    print(f"device: {device}")
    print(f"dense loss:    {dense[0]:.6e} -> {dense[1]:.6e}")
    print(f"Sobolev loss: {sobolev[0]:.6e} -> {sobolev[1]:.6e}")
    print(f"dense adjoint roughness:    {dense_roughness.item():.4f}")
    print(f"Sobolev adjoint roughness: {sobolev_roughness.item():.4f}")


if __name__ == "__main__":
    main()
