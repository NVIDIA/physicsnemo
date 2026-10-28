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

from ._torch_impl import mesh_lsq_gradient_torch
from ._warp_impl import mesh_lsq_gradient_warp


class MeshLSQGradient(FunctionSpec):
    r"""Weighted least-squares gradient reconstruction on unstructured entities.

    This functional computes gradients from unstructured neighborhoods provided
    as CSR adjacency (`neighbor_offsets`, `neighbor_indices`).

    For each entity :math:`i`, it solves the weighted least-squares problem:

    .. math::

       \nabla \phi_i = \arg\min_g
       \sum_{j \in \mathcal{N}(i)} w_{ij} \left(g^T(x_j - x_i) - (\phi_j - \phi_i)\right)^2

    with inverse-distance weighting:

    .. math::

       w_{ij} = ||x_j - x_i||^{-\alpha}

    where :math:`\alpha` is ``weight_power``.

    Parameters
    ----------
    points : torch.Tensor
        Entity coordinates with shape ``(n_entities, dims)``.
    values : torch.Tensor
        Scalar or tensor values with shape ``(n_entities,)`` or
        ``(n_entities, ...)``.
    neighbor_offsets : torch.Tensor
        CSR offsets with shape ``(n_entities + 1,)``.
    neighbor_indices : torch.Tensor
        CSR flattened neighbor indices with shape ``(nnz,)``.
    weight_power : float, optional
        Inverse-distance exponent used for weighting.
    min_neighbors : int, optional
        Entities with fewer than this count get zero gradients.
    safe_epsilon : float | None, optional
        Positive floor applied to squared neighbor distances before
        inverse-distance weighting. When ``None``, a dtype-derived default
        is used by each backend.
    implementation : {"warp", "torch"} or None
        Explicit backend selection. When ``None``, dispatch selects by rank.

    Returns
    -------
    torch.Tensor
        Gradients with shape ``(n_entities, dims)`` for scalar values or
        ``(n_entities, dims, ...)`` for tensor values.
    """

    ### Benchmark input presets (small -> large workload).
    _BENCHMARK_CASES = (
        ("small-1d-scalar-n2048-k16", 2048, 1, 16, False),
        ("small-2d-scalar-n1024-k16", 1024, 2, 16, False),
        ("medium-3d-scalar-n2048-k16", 2048, 3, 16, False),
        ("medium-3d-vector-n2048-k16", 2048, 3, 16, True),
    )

    _COMPARE_ATOL = 5e-3
    _COMPARE_RTOL = 5e-3
    _COMPARE_BACKWARD_ATOL = 8e-3
    _COMPARE_BACKWARD_RTOL = 8e-3

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: torch.Tensor,
        values: torch.Tensor,
        neighbor_offsets: torch.Tensor,
        neighbor_indices: torch.Tensor,
        weight_power: float = 2.0,
        min_neighbors: int = 0,
        safe_epsilon: float | None = None,
    ) -> torch.Tensor:
        """Dispatch mesh LSQ gradients to the Warp backend."""
        ### Warp backend implementation.
        return mesh_lsq_gradient_warp(
            points=points,
            values=values,
            neighbor_offsets=neighbor_offsets,
            neighbor_indices=neighbor_indices,
            weight_power=weight_power,
            min_neighbors=min_neighbors,
            safe_epsilon=safe_epsilon,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: torch.Tensor,
        values: torch.Tensor,
        neighbor_offsets: torch.Tensor,
        neighbor_indices: torch.Tensor,
        weight_power: float = 2.0,
        min_neighbors: int = 0,
        safe_epsilon: float | None = None,
    ) -> torch.Tensor:
        """Dispatch mesh LSQ gradients to eager PyTorch."""
        ### PyTorch backend implementation.
        return mesh_lsq_gradient_torch(
            points=points,
            values=values,
            neighbor_offsets=neighbor_offsets,
            neighbor_indices=neighbor_indices,
            weight_power=weight_power,
            min_neighbors=min_neighbors,
            safe_epsilon=safe_epsilon,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark and parity input cases."""
        device = torch.device(device)

        ### Build deterministic synthetic CSR neighborhoods and signal fields.
        for (
            label,
            n_entities,
            n_dims,
            k_neighbors,
            vector_values,
        ) in cls._BENCHMARK_CASES:
            generator = torch.Generator(device=device)
            generator.manual_seed(1234 + n_entities + n_dims)

            ### Generate point cloud and fixed-k CSR adjacency.
            points = torch.rand(
                (n_entities, n_dims), generator=generator, device=device
            )
            dists = torch.cdist(points, points)
            knn = torch.topk(dists, k=k_neighbors + 1, largest=False, dim=1).indices[
                :, 1:
            ]
            offsets = torch.arange(
                0,
                n_entities * k_neighbors + 1,
                k_neighbors,
                device=device,
                dtype=torch.int64,
            )
            indices = knn.reshape(-1).to(torch.int64)

            ### Build scalar/vector fields from analytic trigonometric signals.
            if vector_values:
                values = torch.stack(
                    [
                        torch.sin(2.0 * torch.pi * points[:, 0]),
                        torch.cos(2.0 * torch.pi * points[:, 1]),
                        torch.sin(2.0 * torch.pi * points[:, -1]),
                    ],
                    dim=-1,
                ).to(torch.float32)
            else:
                values = (
                    torch.sin(2.0 * torch.pi * points[:, 0])
                    + 0.5 * torch.cos(2.0 * torch.pi * points[:, -1])
                ).to(torch.float32)

            ### Yield the labeled functional input case.
            yield (
                label,
                (
                    points.to(torch.float32),
                    values,
                    offsets,
                    indices,
                ),
                {
                    "weight_power": 2.0,
                    "min_neighbors": 0,
                },
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative backward benchmark and parity input cases."""
        device = torch.device(device)

        ### Build representative scalar/vector LSQ inputs for backward parity.
        backward_cases = (
            ("backward-2d-scalar-n512-k12", 512, 2, 12, False),
            ("backward-3d-scalar-n768-k12", 768, 3, 12, False),
            ("backward-3d-vector-n768-k12", 768, 3, 12, True),
        )

        for label, n_entities, n_dims, k_neighbors, vector_values in backward_cases:
            generator = torch.Generator(device=device)
            generator.manual_seed(8411 + n_entities + n_dims)

            ### Build deterministic KNN-CSR adjacency.
            points = torch.rand(
                (n_entities, n_dims), generator=generator, device=device
            )
            dists = torch.cdist(points, points)
            knn = torch.topk(dists, k=k_neighbors + 1, largest=False, dim=1).indices[
                :, 1:
            ]
            offsets = torch.arange(
                0,
                n_entities * k_neighbors + 1,
                k_neighbors,
                device=device,
                dtype=torch.int64,
            )
            indices = knn.reshape(-1).to(torch.int64)

            ### Build differentiable scalar/vector field values.
            if vector_values:
                values = torch.stack(
                    [
                        torch.sin(2.0 * torch.pi * points[:, 0]),
                        torch.cos(2.0 * torch.pi * points[:, 1]),
                        torch.sin(2.0 * torch.pi * points[:, -1]),
                    ],
                    dim=-1,
                ).to(torch.float32)
            else:
                values = (
                    torch.sin(2.0 * torch.pi * points[:, 0])
                    + 0.5 * torch.cos(2.0 * torch.pi * points[:, -1])
                ).to(torch.float32)

            values = values.detach().clone().requires_grad_(True)

            ### Keep point coordinates fixed for backward parity on field gradients.
            yield (
                label,
                (
                    points.to(torch.float32),
                    values,
                    offsets,
                    indices,
                ),
                {
                    "weight_power": 2.0,
                    "min_neighbors": 0,
                },
            )

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare forward outputs across implementations."""
        ### Validate forward parity across backends.
        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_ATOL,
            rtol=cls._COMPARE_RTOL,
        )

    @classmethod
    def compare_backward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare backward gradients across implementations."""
        ### Validate backward parity across backends.
        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_BACKWARD_ATOL,
            rtol=cls._COMPARE_BACKWARD_RTOL,
        )


mesh_lsq_gradient = MeshLSQGradient.make_function("mesh_lsq_gradient")


__all__ = ["MeshLSQGradient", "mesh_lsq_gradient"]
