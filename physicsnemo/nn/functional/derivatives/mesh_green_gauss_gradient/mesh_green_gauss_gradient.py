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

from ._torch_impl import mesh_green_gauss_gradient_torch
from ._warp_impl import mesh_green_gauss_gradient_warp
from .utils import build_neighbors


class MeshGreenGaussGradient(FunctionSpec):
    r"""Compute cell-centered gradients using Green-Gauss face flux balances.

    This functional reconstructs gradients from cell-centered values on
    simplicial meshes (2D triangles or 3D tetrahedra) using:

    .. math::

       \nabla \phi_i \approx \frac{1}{V_i}
       \sum_{f \in \partial i} \phi_f \, \mathbf{A}_{i,f}

    where :math:`V_i` is cell volume/area, :math:`\mathbf{A}_{i,f}` is outward
    face-area vector, and face value :math:`\phi_f` uses centered interpolation
    on interior faces:

    .. math::

       \phi_f = \tfrac{1}{2}(\phi_i + \phi_j)

    while boundary faces use :math:`\phi_f=\phi_i`.

    Parameters
    ----------
    points : torch.Tensor
        Mesh point coordinates with shape ``(n_points, dims)`` for ``dims`` in
        ``{2, 3}``.
    cells : torch.Tensor
        Simplicial connectivity with shape ``(n_cells, dims+1)``.
    neighbors : torch.Tensor
        Precomputed cell-neighbor indices with shape ``(n_cells, n_faces)``,
        where boundary faces are marked with ``-1``.
    values : torch.Tensor
        Cell-centered values with shape ``(n_cells,)`` or ``(n_cells, ...)``.
    implementation : {"warp", "torch"} or None
        Explicit backend selection. When ``None``, dispatch selects by rank.

    Returns
    -------
    torch.Tensor
        Reconstructed gradients with shape ``(n_cells, dims)`` for scalar
        values or ``(n_cells, dims, ...)`` for tensor values.
    """

    ### Benchmark input presets (small -> large workload).
    _BENCHMARK_CASES = (
        ("2d-tri-24x24-scalar", 24, 24, False),
        ("2d-tri-36x36-scalar", 36, 36, False),
        ("2d-tri-36x36-vector", 36, 36, True),
    )

    _COMPARE_ATOL = 3e-4
    _COMPARE_RTOL = 3e-4
    _COMPARE_BACKWARD_ATOL = 8e-3
    _COMPARE_BACKWARD_RTOL = 8e-3

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: torch.Tensor,
        cells: torch.Tensor,
        neighbors: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch Green-Gauss gradients to the Warp backend."""
        ### Warp backend implementation.
        return mesh_green_gauss_gradient_warp(
            points=points,
            cells=cells,
            neighbors=neighbors,
            values=values,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: torch.Tensor,
        cells: torch.Tensor,
        neighbors: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch Green-Gauss gradients to eager PyTorch."""
        ### PyTorch backend implementation.
        return mesh_green_gauss_gradient_torch(
            points=points,
            cells=cells,
            neighbors=neighbors,
            values=values,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark and parity input cases."""
        device = torch.device(device)

        ### Build deterministic triangulated meshes and scalar/vector cell values.
        for label, nx, ny, vector_values in cls._BENCHMARK_CASES:
            ### Construct a structured point set and split each quad into two triangles.
            x = torch.linspace(0.0, 1.0, nx, device=device, dtype=torch.float32)
            y = torch.linspace(0.0, 1.0, ny, device=device, dtype=torch.float32)
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)

            ### Add small deterministic interior jitter for non-degenerate geometry.
            rng = torch.Generator(device=device)
            rng.manual_seed(2027 + nx + ny)
            perturb = 0.01 * (
                torch.rand(
                    points.shape, generator=rng, device=device, dtype=points.dtype
                )
                - 0.5
            )
            border = (
                (points[:, 0] <= 0.0)
                | (points[:, 0] >= 1.0)
                | (points[:, 1] <= 0.0)
                | (points[:, 1] >= 1.0)
            )
            perturb[border] = 0.0
            points = torch.clamp(points + perturb, 0.0, 1.0).contiguous()

            cells = []
            for i in range(nx - 1):
                for j in range(ny - 1):
                    p00 = i * ny + j
                    p10 = (i + 1) * ny + j
                    p01 = i * ny + (j + 1)
                    p11 = (i + 1) * ny + (j + 1)
                    cells.append((p00, p10, p11))
                    cells.append((p00, p11, p01))
            cells = torch.tensor(cells, device=device, dtype=torch.int64).contiguous()
            neighbors = build_neighbors(cells).to(dtype=torch.int64).contiguous()
            centroids = points[cells].mean(dim=1)
            base = (
                torch.sin(2.0 * torch.pi * centroids[:, 0])
                + 0.4 * torch.cos(2.0 * torch.pi * centroids[:, 1])
            ).to(torch.float32)

            if vector_values:
                values = torch.stack(
                    (
                        base,
                        torch.cos(2.0 * torch.pi * centroids[:, 0] - 0.2),
                        torch.sin(2.0 * torch.pi * centroids[:, 1] + 0.3),
                    ),
                    dim=-1,
                ).to(torch.float32)
            else:
                values = base

            yield (
                label,
                (
                    points,
                    cells,
                    neighbors,
                    values,
                ),
                {},
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative backward benchmark and parity input cases."""
        device = torch.device(device)

        ### Build differentiable cell-value inputs for backward parity checks.
        backward_cases = (
            ("backward-2d-tri-24x24-scalar", 24, 24, False),
            ("backward-2d-tri-32x32-vector", 32, 32, True),
        )

        for label, nx, ny, vector_values in backward_cases:
            ### Construct a structured point set and split each quad into two triangles.
            x = torch.linspace(0.0, 1.0, nx, device=device, dtype=torch.float32)
            y = torch.linspace(0.0, 1.0, ny, device=device, dtype=torch.float32)
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)

            ### Add small deterministic interior jitter for non-degenerate geometry.
            rng = torch.Generator(device=device)
            rng.manual_seed(2027 + nx + ny)
            perturb = 0.01 * (
                torch.rand(
                    points.shape, generator=rng, device=device, dtype=points.dtype
                )
                - 0.5
            )
            border = (
                (points[:, 0] <= 0.0)
                | (points[:, 0] >= 1.0)
                | (points[:, 1] <= 0.0)
                | (points[:, 1] >= 1.0)
            )
            perturb[border] = 0.0
            points = torch.clamp(points + perturb, 0.0, 1.0).contiguous()

            cells = []
            for i in range(nx - 1):
                for j in range(ny - 1):
                    p00 = i * ny + j
                    p10 = (i + 1) * ny + j
                    p01 = i * ny + (j + 1)
                    p11 = (i + 1) * ny + (j + 1)
                    cells.append((p00, p10, p11))
                    cells.append((p00, p11, p01))
            cells = torch.tensor(cells, device=device, dtype=torch.int64).contiguous()
            neighbors = build_neighbors(cells).to(dtype=torch.int64).contiguous()
            centroids = points[cells].mean(dim=1)
            base = (
                torch.sin(2.0 * torch.pi * centroids[:, 0])
                + 0.4 * torch.cos(2.0 * torch.pi * centroids[:, 1])
            ).to(torch.float32)

            if vector_values:
                values = torch.stack(
                    (
                        base,
                        torch.cos(2.0 * torch.pi * centroids[:, 0] - 0.2),
                        torch.sin(2.0 * torch.pi * centroids[:, 1] + 0.3),
                    ),
                    dim=-1,
                ).to(torch.float32)
            else:
                values = base

            values = values.detach().clone().requires_grad_(True)
            yield (
                label,
                (
                    points,
                    cells,
                    neighbors,
                    values,
                ),
                {},
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


mesh_green_gauss_gradient = MeshGreenGaussGradient.make_function(
    "mesh_green_gauss_gradient"
)


__all__ = ["MeshGreenGaussGradient", "mesh_green_gauss_gradient"]
