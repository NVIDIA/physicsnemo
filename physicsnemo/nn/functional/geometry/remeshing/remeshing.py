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

"""Tensor functional for Warp-accelerated surface remeshing."""

from __future__ import annotations

from typing import Literal

import torch

from physicsnemo.core.function_spec import FunctionSpec

from ._config import WarpRemeshOptions

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _validate_inputs(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    n_clusters: int,
    max_iterations: int | None,
    warp_options: WarpRemeshOptions | None,
) -> None:
    """Validate the tensor-level remeshing contract."""
    if not isinstance(mesh_vertices, torch.Tensor):
        raise TypeError("mesh_vertices must be a torch.Tensor")
    if not isinstance(mesh_indices, torch.Tensor):
        raise TypeError("mesh_indices must be a torch.Tensor")
    if mesh_vertices.ndim != 2 or mesh_vertices.shape[1] != 3:
        raise ValueError("mesh_vertices must have shape (n_vertices, 3)")
    if mesh_indices.ndim != 2 or mesh_indices.shape[1] != 3:
        raise ValueError("mesh_indices must have shape (n_faces, 3)")
    if mesh_vertices.shape[0] < 3:
        raise ValueError("mesh_vertices must contain at least three vertices")
    if mesh_indices.shape[0] < 1:
        raise ValueError("mesh_indices must contain at least one triangle")
    if not torch.is_floating_point(mesh_vertices):
        raise TypeError(
            f"mesh_vertices must use a floating-point dtype, got {mesh_vertices.dtype}"
        )
    if mesh_indices.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            f"mesh_indices must use an integer dtype, got {mesh_indices.dtype}"
        )
    if mesh_vertices.device != mesh_indices.device:
        raise ValueError("mesh_vertices and mesh_indices must be on the same device")

    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
        raise TypeError(
            f"n_clusters must be an integer, got {type(n_clusters).__name__}"
        )
    if n_clusters < 3:
        raise ValueError(f"n_clusters must be at least 3, got {n_clusters}")
    if n_clusters > mesh_vertices.shape[0]:
        raise ValueError(
            "n_clusters cannot exceed the input vertex count; got "
            f"n_clusters={n_clusters} and n_vertices={mesh_vertices.shape[0]}"
        )

    if max_iterations is not None:
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise TypeError(
                "max_iterations must be an integer or None, got "
                f"{type(max_iterations).__name__}"
            )
        if max_iterations < 0:
            raise ValueError(
                f"max_iterations must be non-negative, got {max_iterations}"
            )

    if warp_options is not None and not isinstance(warp_options, WarpRemeshOptions):
        raise TypeError(
            "warp_options must be a WarpRemeshOptions instance or None, got "
            f"{type(warp_options).__name__}"
        )


def _make_uv_sphere(
    n_rings: int,
    n_segments: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct tensor-only benchmark geometry without importing mesh APIs."""
    phi = torch.linspace(0.0, torch.pi, n_rings + 2, device=device)[1:-1]
    theta = torch.linspace(0.0, 2.0 * torch.pi, n_segments + 1, device=device)[:-1]
    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing="ij")
    sin_phi = phi_grid.sin()
    ring_points = torch.stack(
        [
            sin_phi * theta_grid.cos(),
            sin_phi * theta_grid.sin(),
            phi_grid.cos(),
        ],
        dim=-1,
    ).reshape(-1, 3)
    mesh_vertices = torch.cat(
        [
            torch.tensor([[0.0, 0.0, 1.0]], device=device),
            ring_points,
            torch.tensor([[0.0, 0.0, -1.0]], device=device),
        ]
    ).to(torch.float32)

    south_index = n_rings * n_segments + 1
    segment = torch.arange(n_segments, device=device)
    next_segment = (segment + 1) % n_segments
    north_fan = torch.stack(
        [torch.zeros_like(segment), 1 + segment, 1 + next_segment], dim=1
    )

    ring = torch.arange(n_rings - 1, device=device).unsqueeze(1)
    base = 1 + ring * n_segments
    p00 = base + segment
    p01 = base + next_segment
    p10 = base + n_segments + segment
    p11 = base + n_segments + next_segment
    body = torch.stack(
        [
            torch.stack([p00, p10, p11], dim=-1),
            torch.stack([p00, p11, p01], dim=-1),
        ],
        dim=2,
    ).reshape(-1, 3)

    last_ring = south_index - n_segments
    south_fan = torch.stack(
        [
            last_ring + segment,
            torch.full_like(segment, south_index),
            last_ring + next_segment,
        ],
        dim=1,
    )
    mesh_indices = torch.cat([north_fan, body, south_fan]).to(torch.int64)
    return mesh_vertices.contiguous(), mesh_indices.contiguous()


class Remeshing(FunctionSpec):
    """Remesh a CUDA triangle surface represented by tensors.

    This low-level functional performs area-weighted centroidal clustering,
    projects cluster centers onto the source surface, and reconstructs compact
    triangle connectivity. The operation is intentionally non-differentiable.
    Most users should call :func:`physicsnemo.mesh.remeshing.remesh`, which
    accepts and returns :class:`physicsnemo.mesh.Mesh` objects.

    Parameters
    ----------
    mesh_vertices : torch.Tensor
        Floating-point vertex coordinates with shape ``(n_vertices, 3)`` on
        CUDA.
    mesh_indices : torch.Tensor
        Integer triangle connectivity with shape ``(n_faces, 3)`` on the same
        CUDA device.
    n_clusters : int
        Target output vertex count between 3 and ``n_vertices``, inclusive.
    max_iterations : int | None, optional
        Maximum centroid-relaxation iterations. ``None`` uses four iterations.
    warp_options : WarpRemeshOptions | None, optional
        Performance and initialization controls for the Warp backend.
    implementation : {"warp"} | None, optional
        Explicit backend selection. Only ``"warp"`` is currently available.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Remeshed vertices and triangle indices. Vertex dtype and device match
        ``mesh_vertices``; indices use ``torch.int64`` on the same device.
    """

    _BENCHMARK_CASES = (
        ("small-v482-k64", 15, 32, 64),
        ("medium-v1986-k256", 31, 64, 256),
        ("large-v8066-k1024", 63, 128, 1_024),
    )

    @FunctionSpec.register(
        name="warp",
        required_imports=("warp>=1.14.0",),
        rank=0,
        baseline=True,
    )
    def warp_forward(
        mesh_vertices: torch.Tensor,
        mesh_indices: torch.Tensor,
        n_clusters: int,
        *,
        max_iterations: int | None = None,
        warp_options: WarpRemeshOptions | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Warp-accelerated CUDA remeshing."""
        from ._warp_impl import remeshing_warp

        options = warp_options or WarpRemeshOptions()
        return remeshing_warp(
            mesh_vertices.detach(),
            mesh_indices.detach(),
            n_clusters,
            -1 if max_iterations is None else max_iterations,
            float(options.search_radius_scale),
            float(options.voxel_width_scale),
            options.hash_grid_resolution,
            options.farthest_point_threshold,
            options.farthest_point_oversampling,
        )

    @classmethod
    def dispatch(
        cls,
        mesh_vertices: torch.Tensor,
        mesh_indices: torch.Tensor,
        n_clusters: int,
        *,
        max_iterations: int | None = None,
        warp_options: WarpRemeshOptions | None = None,
        implementation: Literal["warp"] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate inputs and dispatch the CUDA implementation."""
        implementations = cls._get_impls()
        cls._check_impl(implementation, implementations)
        _validate_inputs(
            mesh_vertices,
            mesh_indices,
            n_clusters,
            max_iterations,
            warp_options,
        )
        if mesh_vertices.device.type != "cuda":
            raise ValueError("The Warp remeshing functional requires CUDA tensors.")

        selected = implementations["warp" if implementation is None else implementation]
        if not selected.available:
            raise ImportError("The Warp remeshing backend requires warp>=1.14.0.")
        return selected.func(
            mesh_vertices,
            mesh_indices,
            n_clusters,
            max_iterations=max_iterations,
            warp_options=warp_options,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative tensor-only remeshing workloads."""
        device = torch.device(device)
        for label, n_rings, n_segments, n_clusters in cls._BENCHMARK_CASES:
            vertices, indices = _make_uv_sphere(n_rings, n_segments, device)
            yield (label, (vertices, indices, n_clusters), {})


remeshing = Remeshing.make_function("remeshing")

__all__ = ["Remeshing", "WarpRemeshOptions", "remeshing"]
