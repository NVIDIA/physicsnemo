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

import sys

import torch
from jaxtyping import Float, Int, Integer

from physicsnemo.core.function_spec import FunctionSpec

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}
# Warp allocates two int32 arrays with ``resolution**3`` entries. A resolution
# of 256 bounds their combined storage to 128 MiB before point storage.
_MAX_HASH_GRID_RESOLUTION = 256
_MAX_FINITE_FLOAT = sys.float_info.max


def _validate_inputs(
    mesh_vertices: Float[torch.Tensor, "n_vertices 3"],
    mesh_indices: Integer[torch.Tensor, "n_faces 3"],
    n_clusters: int,
    max_iterations: int,
    search_radius_scale: float,
    voxel_width_scale: float,
    hash_grid_resolution: int,
    farthest_point_threshold: int,
    farthest_point_oversampling: int,
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
            "n_clusters cannot exceed the input vertex count. Got "
            f"n_clusters={n_clusters} and n_vertices={mesh_vertices.shape[0]}"
        )

    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError(
            f"max_iterations must be an integer, got {type(max_iterations).__name__}"
        )
    if max_iterations < 0:
        raise ValueError(f"max_iterations must be non-negative, got {max_iterations}")

    for name, value in (
        ("search_radius_scale", search_radius_scale),
        ("voxel_width_scale", voxel_width_scale),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
        if not (value > 0.0 and value <= _MAX_FINITE_FLOAT):
            raise ValueError(f"{name} must be finite and positive")

    for name, value, minimum in (
        ("hash_grid_resolution", hash_grid_resolution, 1),
        ("farthest_point_threshold", farthest_point_threshold, 0),
        ("farthest_point_oversampling", farthest_point_oversampling, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}, got {value}")

    if hash_grid_resolution > _MAX_HASH_GRID_RESOLUTION:
        raise ValueError(
            "hash_grid_resolution must be at most "
            f"{_MAX_HASH_GRID_RESOLUTION}, got {hash_grid_resolution}"
        )


def _make_uv_sphere(
    n_rings: int,
    n_segments: int,
    device: torch.device,
) -> tuple[
    Float[torch.Tensor, "n_vertices 3"],
    Int[torch.Tensor, "n_faces 3"],
]:
    """Construct tensor-only benchmark geometry without mesh-layer imports.

    The import architecture forbids ``physicsnemo.nn`` from depending on
    ``physicsnemo.mesh``, so this generator cannot use the UV-sphere primitive.
    """
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
    """Remesh a triangle surface represented by tensors.

    This low-level functional performs area-weighted centroidal clustering,
    projects cluster centers onto the source surface, and reconstructs compact
    triangle connectivity. The operation is intentionally non-differentiable.
    Most users should call :func:`physicsnemo.mesh.remeshing.remesh`, which
    accepts and returns :class:`physicsnemo.mesh.Mesh` objects.

    Parameters
    ----------
    mesh_vertices : torch.Tensor
        Floating-point vertex coordinates with shape ``(n_vertices, 3)``.
    mesh_indices : torch.Tensor
        Integer triangle connectivity with shape ``(n_faces, 3)`` on the same
        device.
    n_clusters : int
        Target output vertex count between 3 and ``n_vertices``, inclusive.
    max_iterations : int, optional
        Maximum centroid-relaxation iterations. Default is ``4``.
    search_radius_scale : float, optional
        Hash-grid query radius relative to
        ``sqrt(surface_area / n_clusters)``. Default is ``1.6``.
    voxel_width_scale : float, optional
        Spatial-stratification voxel width relative to
        ``sqrt(surface_area / n_clusters)``. Default is ``1.15``.
    hash_grid_resolution : int, optional
        Resolution of each axis of the centroid hash grid. Must be at most
        ``256``, which bounds its two dense cell-offset arrays to 128 MiB.
        Default is ``128``.
    farthest_point_threshold : int, optional
        Use farthest-point initialization when ``n_clusters`` is at most this
        value. Set to ``0`` to always use voxel initialization. Default is
        ``256``.
    farthest_point_oversampling : int, optional
        Area-weighted farthest-point candidate-pool size as a multiple of
        ``n_clusters``. Default is ``4``.
    implementation : {"warp"} | None, optional
        Explicit backend selection. Only ``"warp"`` is currently available.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Remeshed vertices and triangle indices. Vertex dtype and device match
        ``mesh_vertices``. Indices use ``torch.int64`` on the same device.

    Raises
    ------
    TypeError
        If tensor or scalar inputs have invalid types.
    ValueError
        If tensor shapes, devices, counts, geometry, or tuning values are
        invalid.
    KeyError
        If ``implementation`` does not name a registered backend.
    ImportError
        If Warp is unavailable.
    RuntimeError
        If topology reconstruction cannot produce a nonempty manifold surface.

    Notes
    -----
    Remeshing is intentionally non-differentiable. Warp computes in centered
    and scaled coordinates in float32, then restores the input vertex dtype and
    coordinate frame. Centroid sampling uses a fixed random seed, although
    floating-point atomics can still introduce small run-to-run differences.
    Spatial clustering can weld sheets or thin features separated by less than
    the mean cluster spacing. Projection can also map distinct centroids to the
    same surface position. Output vertices are not welded by position.
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
        mesh_vertices: Float[torch.Tensor, "n_vertices 3"],
        mesh_indices: Integer[torch.Tensor, "n_faces 3"],
        n_clusters: int,
        *,
        max_iterations: int = 4,
        search_radius_scale: float = 1.6,
        voxel_width_scale: float = 1.15,
        hash_grid_resolution: int = 128,
        farthest_point_threshold: int = 256,
        farthest_point_oversampling: int = 4,
    ) -> tuple[
        Float[torch.Tensor, "n_output_vertices 3"],
        Int[torch.Tensor, "n_output_faces 3"],
    ]:
        """Run the Warp remeshing backend on CPU or CUDA.

        Parameters
        ----------
        mesh_vertices : torch.Tensor
            Floating-point vertex coordinates with shape ``(n_vertices, 3)``.
        mesh_indices : torch.Tensor
            Integer triangle connectivity with shape ``(n_faces, 3)``.
        n_clusters : int
            Target output vertex count.
        max_iterations : int, optional
            Maximum centroid-relaxation iterations.
        search_radius_scale : float, optional
            Scale factor for the centroid hash-grid query radius.
        voxel_width_scale : float, optional
            Scale factor for spatial-stratification voxel width.
        hash_grid_resolution : int, optional
            Resolution of each centroid hash-grid axis.
        farthest_point_threshold : int, optional
            Largest target that uses farthest-point initialization.
        farthest_point_oversampling : int, optional
            Candidate-pool multiplier for farthest-point initialization.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Remeshed vertices and triangle connectivity.

        Raises
        ------
        TypeError
            If tensor or scalar inputs have invalid types.
        ValueError
            If tensor shapes, devices, counts, geometry, or tuning values are
            invalid.
        ImportError
            If Warp is unavailable.
        RuntimeError
            If topology reconstruction cannot produce a nonempty manifold
            surface.
        """
        _validate_inputs(
            mesh_vertices,
            mesh_indices,
            n_clusters,
            max_iterations,
            search_radius_scale,
            voxel_width_scale,
            hash_grid_resolution,
            farthest_point_threshold,
            farthest_point_oversampling,
        )
        if mesh_vertices.device.type not in ("cpu", "cuda"):
            raise ValueError(
                "The Warp remeshing functional supports CPU and CUDA tensors. "
                f"Got device {mesh_vertices.device}."
            )

        from ._warp_impl import remeshing_warp

        return remeshing_warp(
            mesh_vertices.detach(),
            mesh_indices.detach(),
            n_clusters,
            max_iterations,
            float(search_radius_scale),
            float(voxel_width_scale),
            hash_grid_resolution,
            farthest_point_threshold,
            farthest_point_oversampling,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative tensor-only remeshing workloads.

        Parameters
        ----------
        device : torch.device or str, optional
            Device on which to construct the benchmark tensors.

        Yields
        ------
        tuple
            Benchmark label, positional arguments, and keyword arguments.
        """
        device = torch.device(device)
        for label, n_rings, n_segments, n_clusters in cls._BENCHMARK_CASES:
            vertices, indices = _make_uv_sphere(n_rings, n_segments, device)
            yield (label, (vertices, indices, n_clusters), {})


remeshing = Remeshing.make_function("remeshing")

__all__ = ["Remeshing", "remeshing"]
