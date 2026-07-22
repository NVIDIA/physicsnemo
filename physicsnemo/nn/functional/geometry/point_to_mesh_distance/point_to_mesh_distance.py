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

"""Backend-dispatched point-to-triangle-mesh distance."""

from __future__ import annotations

from typing import Literal

import torch
from jaxtyping import Float, Int

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import point_to_mesh_distance_torch
from ._warp_impl import point_to_mesh_distance_warp


def _benchmark_inputs(
    num_faces: int,
    num_queries: int,
    device: torch.device,
    *,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct deterministic, nondegenerate triangle-soup benchmark inputs."""

    generator = torch.Generator(device=device).manual_seed(4701)
    centers = torch.rand(num_faces, 3, generator=generator, device=device)
    offsets = torch.tensor(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
        device=device,
    )
    vertices = (centers.unsqueeze(1) + offsets).reshape(-1, 3)
    faces = torch.arange(3 * num_faces, dtype=torch.long, device=device).reshape(
        num_faces, 3
    )
    queries = torch.rand(num_queries, 3, generator=generator, device=device)
    return (
        vertices.requires_grad_(requires_grad),
        faces,
        queries.requires_grad_(requires_grad),
    )


class PointToMeshDistance(FunctionSpec):
    r"""Find unsigned distances and closest points on a triangle mesh.

    For every query :math:`q_i`, this functional finds its closest point
    :math:`p_i` on the closed triangles of an unbatched target mesh and returns
    either :math:`\lVert q_i-p_i\rVert_2` or its square. Query points may have
    arbitrary leading dimensions, all of which share one target mesh.

    Hard nearest-face selection is discrete and treated as non-differentiable.
    Once a face is selected, its closest-point projection is recomputed from
    the original tensors with native Torch operations. Consequently, distances
    and closest points propagate gradients through query coordinates and target
    vertex coordinates almost everywhere. Gradients are not uniquely defined
    where nearest faces tie, at triangle-region boundaries, or on degenerate
    triangles.

    The Torch backend supports float32 and float64 on CPU and CUDA. The Warp
    backend supports float32 on CPU and CUDA and never downcasts unsupported
    inputs. Automatic dispatch uses Torch on CPU and Warp only for CUDA
    float32; either backend may be requested explicitly where supported.

    Parameters
    ----------
    mesh_vertices : torch.Tensor
        Target vertex coordinates with shape ``(num_vertices, 3)`` and dtype
        float32 or float64. Coordinates must be finite.
    mesh_indices : torch.Tensor
        Triangle connectivity with shape ``(num_faces, 3)`` and dtype int32 or
        int64. The mesh must contain at least one triangle, every triangle must
        be nondegenerate, and every index must address ``mesh_vertices``.
    input_points : torch.Tensor
        Query coordinates with shape ``(..., 3)``. Dtype and device must match
        ``mesh_vertices``.
    squared : bool, optional
        Return squared Euclidean distances when ``True``. Default is ``False``.
    implementation : {"warp", "torch"} or None, optional
        Explicit backend. ``None`` selects Warp for CUDA float32 coordinates
        when available and Torch otherwise. Explicit Warp also supports CPU
        float32 execution, primarily for parity and benchmarking.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(distance, closest_points)``. ``distance`` has shape
        ``input_points.shape[:-1]`` and ``closest_points`` has the same shape as
        ``input_points``.

    Notes
    -----
    The Torch baseline performs a memory-bounded exhaustive face search, then
    recomputes only the winning projections with autograd enabled. Its search
    cost is ``O(num_queries * num_faces)``. The Warp backend performs only the
    discrete nearest-face search with an accelerated BVH; the winning
    projection remains in native Torch and therefore has the same gradient
    contract as the baseline. Because this functional is stateless, Warp
    rebuilds the target BVH on every call. CUDA Graph capture is not part of
    the current contract.
    """

    _FORWARD_BENCHMARK_CASES = (
        ("small-f128-q1024", 128, 1024),
        ("medium-f512-q4096", 512, 4096),
        ("large-f2048-q16384", 2048, 16384),
    )
    _BACKWARD_BENCHMARK_CASES = (
        ("small-f64-q512-all-gradients", 64, 512),
        ("medium-f256-q2048-all-gradients", 256, 2048),
    )

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        mesh_vertices: Float[torch.Tensor, "num_vertices 3"],
        mesh_indices: Int[torch.Tensor, "num_faces 3"],
        input_points: Float[torch.Tensor, "... 3"],
        *,
        squared: bool = False,
    ) -> tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "... 3"]]:
        """Find correspondences with Warp and project with native Torch."""

        return point_to_mesh_distance_warp(
            mesh_vertices,
            mesh_indices,
            input_points,
            squared=squared,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        mesh_vertices: Float[torch.Tensor, "num_vertices 3"],
        mesh_indices: Int[torch.Tensor, "num_faces 3"],
        input_points: Float[torch.Tensor, "... 3"],
        *,
        squared: bool = False,
    ) -> tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "... 3"]]:
        """Compute closest-surface distances with the Torch backend."""

        return point_to_mesh_distance_torch(
            mesh_vertices,
            mesh_indices,
            input_points,
            squared=squared,
        )

    @classmethod
    def dispatch(
        cls,
        mesh_vertices: torch.Tensor,
        mesh_indices: torch.Tensor,
        input_points: torch.Tensor,
        *,
        squared: bool = False,
        implementation: Literal["torch", "warp"] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select Warp only for CUDA float32 coordinates by default."""

        if implementation is None:
            implementations = cls._get_impls()
            warp_implementation = implementations.get("warp")
            use_warp = (
                isinstance(mesh_vertices, torch.Tensor)
                and isinstance(input_points, torch.Tensor)
                and mesh_vertices.is_cuda
                and input_points.is_cuda
                and mesh_vertices.dtype == torch.float32
                and input_points.dtype == torch.float32
            )
            if use_warp and warp_implementation is not None:
                if warp_implementation.available:
                    implementation = "warp"
                else:
                    cls._warn_fallback(warp_implementation, implementations["torch"])
                    implementation = "torch"
            else:
                implementation = "torch"

        return super().dispatch(
            mesh_vertices,
            mesh_indices,
            input_points,
            squared=squared,
            implementation=implementation,
        )

    @classmethod
    def compare_forward(
        cls,
        output: tuple[torch.Tensor, torch.Tensor],
        reference: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Compare distances and closest points across search backends."""

        for actual, expected in zip(output, reference, strict=True):
            torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)

    @classmethod
    def compare_backward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare query or target gradients across search backends."""

        torch.testing.assert_close(output, reference, atol=2.0e-5, rtol=2.0e-5)

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark cases."""

        device = torch.device(device)
        for label, num_faces, num_queries in cls._FORWARD_BENCHMARK_CASES:
            vertices, faces, queries = _benchmark_inputs(
                num_faces, num_queries, device, requires_grad=False
            )
            yield label, (vertices, faces, queries), {"squared": True}

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield cases with gradients through queries and target vertices."""

        device = torch.device(device)
        for label, num_faces, num_queries in cls._BACKWARD_BENCHMARK_CASES:
            vertices, faces, queries = _benchmark_inputs(
                num_faces, num_queries, device, requires_grad=True
            )
            yield label, (vertices, faces, queries), {"squared": True}


point_to_mesh_distance = PointToMeshDistance.make_function("point_to_mesh_distance")

__all__ = ["PointToMeshDistance", "point_to_mesh_distance"]
