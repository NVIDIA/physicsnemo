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

import math

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from ._warp_impl import ray_mesh_intersect_warp

_RAY_MESH_INTERSECT_BENCHMARK_CASES = (
    ("small", 8, 20, 4096),
    ("medium", 16, 40, 16384),
    ("large", 32, 80, 65536),
)


def _make_uv_sphere_mesh(
    *,
    n_rings: int,
    n_segments: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Build UV-sphere vertex positions.
    phi = torch.linspace(0.0, torch.pi, n_rings + 2, device=device)[1:-1]
    theta = torch.linspace(0.0, 2.0 * torch.pi, n_segments + 1, device=device)[:-1]
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

    # Build UV-sphere triangle connectivity.
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

    mesh_indices = torch.cat([north_fan, body_tris, south_fan]).to(torch.int32)
    return mesh_vertices.contiguous(), mesh_indices.contiguous()


def _make_ray_grid(
    *,
    num_rays: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid_side = int(math.ceil(math.sqrt(float(num_rays))))
    grid_coords = torch.linspace(-1.5, 1.5, grid_side, device=device)
    x_coords, y_coords = torch.meshgrid(grid_coords, grid_coords, indexing="ij")
    xy_coords = torch.stack([x_coords.reshape(-1), y_coords.reshape(-1)], dim=-1)[
        :num_rays
    ]

    z_origins = torch.full((num_rays, 1), -2.0, device=device)
    ray_origins = torch.cat([xy_coords, z_origins], dim=-1).to(torch.float32)
    ray_directions = torch.zeros((num_rays, 3), device=device, dtype=torch.float32)
    ray_directions[:, 2] = 1.0
    return ray_origins.contiguous(), ray_directions.contiguous()


class RayMeshIntersect(FunctionSpec):
    """Intersect rays with a triangle mesh using Warp.

    ``ray_mesh_intersect`` builds a Warp ``Mesh`` acceleration structure from
    triangle vertices and indices, casts each input ray against the mesh, and
    returns the closest hit within ``max_distance``. Ray directions do not need
    to be normalized; the Warp implementation normalizes them before querying so
    returned hit distances are expressed in mesh-space length units.

    Parameters
    ----------
    mesh_vertices : torch.Tensor
        Mesh vertex positions with shape ``(num_vertices, 3)``.
    mesh_indices : torch.Tensor
        Triangle connectivity with shape ``(num_faces, 3)`` or a flattened
        equivalent.
    ray_origins : torch.Tensor
        Ray origins with shape ``(..., 3)``.
    ray_directions : torch.Tensor
        Ray directions with the same shape as ``ray_origins``.
    max_distance : float, optional
        Maximum ray distance. Default is ``1e8``.
    warp_mesh : wp.Mesh | None, optional
        Prepared Warp mesh returned by an earlier ``ray_mesh_intersect`` call
        with ``return_warp_mesh=True``. If provided, the mesh tensors are not
        used to rebuild a Warp ``Mesh``.
    return_warp_mesh : bool, optional
        If ``True``, append the Warp ``Mesh`` used for the query to the output
        tuple so it can be passed back through ``warp_mesh`` on later calls.
    implementation : str, optional
        Explicit implementation name. Currently only ``"warp"`` is registered.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        By default, a tuple ``(hit_mask, hit_distance, hit_points, face_ids,
        hit_normals)``. If ``return_warp_mesh=True``, the returned tuple is
        ``(hit_mask, hit_distance, hit_points, face_ids, hit_normals,
        warp_mesh)``. Missed rays have ``False`` in ``hit_mask``, infinite
        ``hit_distance``, zero ``hit_points`` and ``hit_normals``, and ``-1``
        ``face_ids``.

    Notes
    -----
    ``hit_normals`` are the mesh query normals returned by Warp. They preserve
    the mesh winding orientation and are not flipped to face the incoming ray.
    For repeated queries against a static mesh, call with
    ``return_warp_mesh=True`` once and pass the returned Warp mesh back through
    ``warp_mesh`` on later calls.
    """

    _BENCHMARK_CASES = _RAY_MESH_INTERSECT_BENCHMARK_CASES

    @FunctionSpec.register(
        name="warp",
        required_imports=("warp>=1.0.0",),
        rank=0,
        baseline=True,
    )
    def warp_forward(
        mesh_vertices: torch.Tensor,
        mesh_indices: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        max_distance: float = 1.0e8,
        warp_mesh: wp.Mesh | None = None,
        return_warp_mesh: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            wp.Mesh,
        ]
    ):
        """Run the Warp backend ray/mesh intersection query."""
        return ray_mesh_intersect_warp(
            mesh_vertices=mesh_vertices,
            mesh_indices=mesh_indices,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            max_distance=max_distance,
            warp_mesh=warp_mesh,
            return_warp_mesh=return_warp_mesh,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield benchmark inputs for batched ray/mesh intersection."""
        device = torch.device(device)
        for label, n_rings, n_segments, num_rays in cls._BENCHMARK_CASES:
            mesh_vertices, mesh_indices = _make_uv_sphere_mesh(
                n_rings=n_rings,
                n_segments=n_segments,
                device=device,
            )
            ray_origins, ray_directions = _make_ray_grid(
                num_rays=num_rays,
                device=device,
            )
            yield (
                f"{label}-uv-sphere-tris{2 * n_rings * n_segments}-rays{num_rays}",
                (mesh_vertices, mesh_indices, ray_origins, ray_directions),
                {"max_distance": 10.0},
            )


ray_mesh_intersect = RayMeshIntersect.make_function("ray_mesh_intersect")

__all__ = ["RayMeshIntersect", "ray_mesh_intersect"]
