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

from __future__ import annotations

from collections.abc import Sequence

import torch

from physicsnemo.core.function_spec import FunctionSpec

from ._warp_impl import mesh_to_voxel_fraction_warp


class MeshToVoxelFraction(FunctionSpec):
    r"""Compute mesh-voxel volume fractions on a regular 3D grid.

    This functional estimates the fraction of each voxel that lies inside a
    triangle mesh using Warp kernels and Monte Carlo sampling.

    For each voxel, it first performs an AABB-overlap query with mesh triangles.
    If no triangles overlap the voxel, it classifies only the voxel center as
    inside or outside. If triangles overlap, it uniformly samples points inside
    the voxel and estimates the occupancy fraction:

    .. math::

       f_{ijk} \approx \frac{1}{N_s}\sum_{s=1}^{N_s}\mathbb{1}\left(x_s \in \Omega\right),

    where :math:`N_s` is ``n_samples`` and :math:`\Omega` is the mesh interior.

    Parameters
    ----------
    mesh_vertices : torch.Tensor
        Vertex positions with shape ``(n_vertices, 3)``.
    mesh_indices : torch.Tensor
        Triangle connectivity as shape ``(n_faces, 3)`` or flattened shape
        ``(3 * n_faces,)``.
    origin : torch.Tensor | Sequence[float]
        Lower corner of the voxel grid as a length-3 vector.
    voxel_size : float
        Edge length of each cubic voxel.
    grid_dims : Sequence[int]
        Grid resolution ``(nx, ny, nz)``.
    n_samples : int, optional
        Number of Monte Carlo samples per overlapping voxel. Default is ``64``.
    seed : int, optional
        Random seed offset used per voxel. Default is ``42``.
    open_mesh : bool, optional
        If ``True``, uses winding-number sign queries for open meshes.
        Default is ``False``.
    winding_number_threshold : float, optional
        Winding-number threshold used when ``open_mesh=True``.
    winding_number_accuracy : float, optional
        Winding-number query accuracy used when ``open_mesh=True``.
    implementation : str | None, optional
        Explicit backend selection. Defaults to dispatch behavior.

    Returns
    -------
    torch.Tensor
        Volume fractions in ``[0, 1]`` with shape ``(nz, ny, nx)`` and dtype
        ``torch.float32``.

    Notes
    -----
    - This functional provides a Warp implementation.
    - The operation is stochastic over overlapping voxels; use ``seed`` for
      reproducible runs.
    """

    _BENCHMARK_CASES = (
        ("small-subdiv2-64^3-s16", 2, 64, 16, False),
        ("medium-subdiv3-96^3-s32", 3, 96, 32, False),
        ("large-subdiv3-128^3-s64-open", 3, 128, 64, True),
    )

    @FunctionSpec.register(
        name="warp",
        required_imports=("warp>=0.6.0",),
        rank=0,
        baseline=True,
    )
    def warp_forward(
        mesh_vertices: torch.Tensor,
        mesh_indices: torch.Tensor,
        origin: torch.Tensor | Sequence[float],
        voxel_size: float,
        grid_dims: Sequence[int] | torch.Tensor,
        n_samples: int = 64,
        seed: int = 42,
        open_mesh: bool = False,
        winding_number_threshold: float = 0.5,
        winding_number_accuracy: float = 2.0,
    ) -> torch.Tensor:
        """Run the Warp backend voxel-fraction estimator on a triangle mesh."""
        return mesh_to_voxel_fraction_warp(
            mesh_vertices=mesh_vertices,
            mesh_indices=mesh_indices,
            origin=origin,
            voxel_size=voxel_size,
            grid_dims=grid_dims,
            n_samples=n_samples,
            seed=seed,
            open_mesh=open_mesh,
            winding_number_threshold=winding_number_threshold,
            winding_number_accuracy=winding_number_accuracy,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Build benchmark inputs across mesh complexity and grid workloads."""
        device = torch.device(device)

        # Build benchmark cases with increasing grid resolution/workload.
        for seed, (label, subdivisions, grid_n, n_samples, open_mesh) in enumerate(
            cls._BENCHMARK_CASES
        ):
            n_rings = 4 * (2**subdivisions)
            n_segments = 8 * (2**subdivisions)

            phi = torch.linspace(0.0, torch.pi, n_rings + 2, device=device)[1:-1]
            theta = torch.linspace(0.0, 2.0 * torch.pi, n_segments + 1, device=device)[
                :-1
            ]
            phi_g, theta_g = torch.meshgrid(phi, theta, indexing="ij")

            sin_phi = phi_g.sin()
            ring_points = torch.stack(
                [sin_phi * theta_g.cos(), sin_phi * theta_g.sin(), phi_g.cos()],
                dim=-1,
            ).reshape(-1, 3)

            mesh_vertices = torch.cat(
                [
                    torch.tensor([[0.0, 0.0, 1.0]], device=device),
                    ring_points,
                    torch.tensor([[0.0, 0.0, -1.0]], device=device),
                ]
            ).to(torch.float32)

            south_idx = n_rings * n_segments + 1
            j = torch.arange(n_segments, device=device)
            j_next = (j + 1) % n_segments

            north_fan = torch.stack([torch.zeros_like(j), 1 + j, 1 + j_next], dim=1)

            r = torch.arange(n_rings - 1, device=device).unsqueeze(1)
            base = 1 + r * n_segments
            p00 = base + j
            p01 = base + j_next
            p10 = base + n_segments + j
            p11 = base + n_segments + j_next
            body_tris = torch.stack(
                [
                    torch.stack([p00, p10, p11], dim=-1),
                    torch.stack([p00, p11, p01], dim=-1),
                ],
                dim=2,
            ).reshape(-1, 3)

            last = south_idx - n_segments
            south_fan = torch.stack(
                [last + j, torch.full_like(j, south_idx), last + j_next], dim=1
            )

            mesh_indices = (
                torch.cat([north_fan, body_tris, south_fan])
                .to(torch.int32)
                .contiguous()
            )
            mesh_vertices = mesh_vertices.contiguous()

            # Define a padded cubic domain around the mesh bounds.
            bbox_min = mesh_vertices.min(dim=0).values
            bbox_max = mesh_vertices.max(dim=0).values
            extent_value = float((bbox_max - bbox_min).amax().detach().cpu().item())
            extent_value = extent_value if extent_value > 0.0 else 1.0
            padding = 0.1 * extent_value
            origin = (bbox_min - padding).to(torch.float32).contiguous()
            voxel_size = (extent_value + 2.0 * padding) / float(grid_n)

            yield (
                label,
                (
                    mesh_vertices,
                    mesh_indices,
                    origin,
                    voxel_size,
                    (grid_n, grid_n, grid_n),
                ),
                {
                    "n_samples": n_samples,
                    "seed": 2026 + seed,
                    "open_mesh": open_mesh,
                    "winding_number_threshold": 0.5,
                    "winding_number_accuracy": 2.0,
                },
            )


mesh_to_voxel_fraction = MeshToVoxelFraction.make_function("mesh_to_voxel_fraction")


__all__ = ["MeshToVoxelFraction", "mesh_to_voxel_fraction"]
