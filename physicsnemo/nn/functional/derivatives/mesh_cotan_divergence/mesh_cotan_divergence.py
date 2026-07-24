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

import torch

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import mesh_cotan_divergence_torch
from ._warp_impl import mesh_cotan_divergence_warp


def _make_benchmark_case(
    *, device: torch.device | str, n_points: int, n_dims: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic edge inputs owned by this functional's spec."""
    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    points = torch.rand((n_points, n_dims), generator=generator, device=device)

    edge_start = torch.arange(n_points - 1, device=device, dtype=torch.int64)
    chain_edges = torch.stack((edge_start, edge_start + 1), dim=-1)
    skip_edges = torch.stack((edge_start[:-1], edge_start[:-1] + 2), dim=-1)
    edges = torch.cat((chain_edges, skip_edges), dim=0)

    edge_vectors = points[edges[:, 1]] - points[edges[:, 0]]
    info = torch.finfo(points.dtype)
    eps = min(info.tiny**0.25, info.eps)
    cotan_weights = edge_vectors.norm(dim=-1).clamp(min=eps).reciprocal()
    dual_volumes = torch.ones((n_points,), dtype=torch.float32, device=device)
    return (
        points.to(torch.float32),
        edges,
        cotan_weights.to(torch.float32),
        dual_volumes,
    )


class MeshCotanDivergence(FunctionSpec):
    r"""Compute cotangent/DEC divergence of a vertex vector field.

    For each canonical undirected edge ``(i, j)``, this functional uses the
    midpoint vector-field flat

    .. math::

       \frac{X_i + X_j}{2} \cdot (x_j - x_i)

    and accumulates the weighted oriented flux at the endpoints before
    normalizing by dual vertex volumes.

    Parameters
    ----------
    points : torch.Tensor
        Vertex coordinates with shape ``(n_points, n_dims)``.
    edges : torch.Tensor
        Canonical undirected vertex pairs with shape ``(n_edges, 2)``.
    cotan_weights : torch.Tensor
        One cotangent weight per edge with shape ``(n_edges,)``.
    dual_volumes : torch.Tensor
        One positive dual volume per vertex with shape ``(n_points,)``.
    vector_field : torch.Tensor
        Vertex vector field with the same shape as ``points``.

    Returns
    -------
    torch.Tensor
        Scalar divergence at each vertex with shape ``(n_points,)`` and the
        same dtype as ``vector_field``.
    """

    _COMPARE_ATOL = 5e-6
    _COMPARE_RTOL = 5e-6

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: torch.Tensor,
        edges: torch.Tensor,
        cotan_weights: torch.Tensor,
        dual_volumes: torch.Tensor,
        vector_field: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch cotangent divergence to the Warp backend."""
        return mesh_cotan_divergence_warp(
            points=points,
            edges=edges,
            cotan_weights=cotan_weights,
            dual_volumes=dual_volumes,
            vector_field=vector_field,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: torch.Tensor,
        edges: torch.Tensor,
        cotan_weights: torch.Tensor,
        dual_volumes: torch.Tensor,
        vector_field: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch cotangent divergence to eager PyTorch."""
        return mesh_cotan_divergence_torch(
            points=points,
            edges=edges,
            cotan_weights=cotan_weights,
            dual_volumes=dual_volumes,
            vector_field=vector_field,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark and parity input cases."""
        for label, n_points, n_dims in (
            ("small-2d-n512", 512, 2),
            ("medium-3d-n1024", 1024, 3),
        ):
            points, edges, weights, volumes = _make_benchmark_case(
                device=device,
                n_points=n_points,
                n_dims=n_dims,
                seed=10100 + n_points + n_dims,
            )
            vector_field = points.square()
            yield label, (points, edges, weights, volumes, vector_field), {}

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative backward benchmark and parity input cases."""
        for label, args, kwargs in cls.make_inputs_forward(device=device):
            points, edges, weights, volumes, vector_field = args
            yield (
                f"{label}-backward",
                (
                    points.detach().clone().requires_grad_(True),
                    edges,
                    weights.detach().clone().requires_grad_(True),
                    volumes.detach().clone().requires_grad_(True),
                    vector_field.detach().clone().requires_grad_(True),
                ),
                kwargs,
            )

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare forward outputs across implementations."""
        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_ATOL,
            rtol=cls._COMPARE_RTOL,
        )

    @classmethod
    def compare_backward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare backward gradients across implementations."""
        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_ATOL,
            rtol=cls._COMPARE_RTOL,
        )


mesh_cotan_divergence = MeshCotanDivergence.make_function("mesh_cotan_divergence")


__all__ = ["MeshCotanDivergence", "mesh_cotan_divergence"]
