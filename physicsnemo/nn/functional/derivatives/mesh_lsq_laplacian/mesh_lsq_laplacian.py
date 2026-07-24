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

from ._torch_impl import mesh_lsq_laplacian_torch
from ._warp_impl import mesh_lsq_laplacian_warp


def _make_knn_csr_case(
    *,
    device: torch.device | str,
    n_entities: int,
    n_dims: int,
    k_neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic point-cloud KNN CSR benchmark inputs."""
    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    points = torch.rand((n_entities, n_dims), generator=generator, device=device)
    dists = torch.cdist(points, points)
    knn = torch.topk(dists, k=k_neighbors + 1, largest=False, dim=1).indices[:, 1:]
    offsets = torch.arange(
        0,
        n_entities * k_neighbors + 1,
        k_neighbors,
        device=device,
        dtype=torch.int64,
    )
    indices = knn.reshape(-1).to(torch.int64)
    return points.to(torch.float32), offsets, indices


class MeshLSQLaplacian(FunctionSpec):
    r"""Compute a double-LSQ Laplacian of an unstructured field.

    This functional first reconstructs the LSQ gradient and then traces its
    second LSQ derivative over the same CSR neighborhoods. Trailing value
    dimensions are treated as independent components. This is the
    unstructured analogue of ``div(grad(values))`` and is distinct from the
    intrinsic cotangent Laplace-Beltrami operator.

    Parameters
    ----------
    points : torch.Tensor
        Entity coordinates with shape ``(n_entities, dims)``.
    values : torch.Tensor
        Values with shape ``(n_entities, ...)``.
    neighbor_offsets : torch.Tensor
        CSR offsets with shape ``(n_entities + 1,)``.
    neighbor_indices : torch.Tensor
        CSR flattened neighbor indices with shape ``(nnz,)``.
    weight_power : float, optional
        Inverse-distance exponent used for weighting.
    min_neighbors : int, optional
        Entities with fewer than this count get zero gradients.
    safe_epsilon : float | None, optional
        Positive floor applied to squared neighbor distances.
    implementation : {"warp", "torch"} or None
        Explicit backend selection. When ``None``, dispatch selects by rank.

    Returns
    -------
    torch.Tensor
        Laplacian with the same shape as ``values``.
    """

    # Composing two first-order reconstructions amplifies the fp32 difference
    # between Warp QR and torch.linalg.lstsq on ill-conditioned neighborhoods.
    _COMPARE_ATOL = 3.0e-2
    _COMPARE_RTOL = 3.0e-2

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
        """Dispatch LSQ mesh Laplacian to the Warp backend."""
        return mesh_lsq_laplacian_warp(
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
        """Dispatch LSQ mesh Laplacian to eager PyTorch."""
        return mesh_lsq_laplacian_torch(
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
        for label, n_entities, n_dims, k_neighbors in (
            ("small-2d-n512-k12", 512, 2, 12),
            ("medium-3d-n1024-k16", 1024, 3, 16),
        ):
            points, offsets, indices = _make_knn_csr_case(
                device=device,
                n_entities=n_entities,
                n_dims=n_dims,
                k_neighbors=k_neighbors,
                seed=8100 + n_entities + n_dims,
            )
            values = (
                points.square().sum(dim=-1)
                if n_dims == 2
                else torch.stack(
                    (points.square().sum(dim=-1), points.prod(dim=-1)), dim=-1
                )
            )
            yield label, (points, values, offsets, indices), {}

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative backward benchmark and parity input cases."""
        for label, args, kwargs in cls.make_inputs_forward(device=device):
            points, values, offsets, indices = args
            yield (
                f"{label}-backward",
                (
                    points.detach().clone().requires_grad_(True),
                    values.detach().clone().requires_grad_(True),
                    offsets,
                    indices,
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


mesh_lsq_laplacian = MeshLSQLaplacian.make_function("mesh_lsq_laplacian")


__all__ = ["MeshLSQLaplacian", "mesh_lsq_laplacian"]
