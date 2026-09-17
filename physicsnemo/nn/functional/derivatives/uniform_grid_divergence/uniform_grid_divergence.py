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

from ._torch_impl import uniform_grid_divergence_torch
from ._warp_impl import uniform_grid_divergence_warp


class UniformGridDivergence(FunctionSpec):
    r"""Compute periodic divergence on a uniform grid.

    This functional accepts channel-first vector fields with shape
    ``(dim, *grid_shape)`` where ``dim`` matches the 1D/2D/3D grid
    dimensionality. Divergence is computed as the trace of the Jacobian,

    .. math::

       \nabla \cdot u = \sum_i \partial_i u_i.

    Parameters
    ----------
    vector_field : torch.Tensor
        Channel-first vector field with shape ``(dim, *grid_shape)``.
    spacing : float | Sequence[float], optional
        Uniform spacing per grid axis. A scalar applies the same spacing to
        every axis.
    order : int, optional
        Central-difference accuracy order. Supported values match
        :func:`physicsnemo.nn.functional.uniform_grid_gradient`.
    implementation : {"warp", "torch"} or None
        Explicit backend selection. When ``None``, rank-based backend dispatch
        is used.

    Returns
    -------
    torch.Tensor
        Scalar divergence field with shape ``grid_shape``.
    """

    _BENCHMARK_CASES = (
        ("1d-n8192-o2", (8192,), 0.01, 2),
        ("1d-n8192-o4", (8192,), 0.01, 4),
        ("2d-512x512-o2", (512, 512), (0.01, 0.02), 2),
        ("2d-384x384-o4", (384, 384), (0.01, 0.02), 4),
        ("3d-96x96x96-o2", (96, 96, 96), 0.02, 2),
    )

    _BACKWARD_CASES = (
        ("1d-grad-n4096-o2", (4096,), 0.01, 2),
        ("2d-grad-256x256-o2", (256, 256), (0.01, 0.02), 2),
        ("2d-grad-192x192-o4", (192, 192), (0.01, 0.02), 4),
        ("3d-grad-64x64x64-o2", (64, 64, 64), 0.02, 2),
    )

    _COMPARE_ATOL = 1e-5
    _COMPARE_RTOL = 1e-5

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        vector_field: torch.Tensor,
        spacing: float | Sequence[float] = 1.0,
        order: int = 2,
    ) -> torch.Tensor:
        """Dispatch uniform-grid divergence to the Warp backend."""
        return uniform_grid_divergence_warp(
            vector_field=vector_field,
            spacing=spacing,
            order=order,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        vector_field: torch.Tensor,
        spacing: float | Sequence[float] = 1.0,
        order: int = 2,
    ) -> torch.Tensor:
        """Dispatch uniform-grid divergence to eager PyTorch."""
        return uniform_grid_divergence_torch(
            vector_field=vector_field,
            spacing=spacing,
            order=order,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark and parity input cases."""
        device = torch.device(device)
        for label, shape, spacing, order in cls._BENCHMARK_CASES:
            vector_field = _make_periodic_vector_field(shape, device=device)
            yield label, (vector_field,), {"spacing": spacing, "order": order}

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative backward benchmark and parity input cases."""
        device = torch.device(device)
        for label, shape, spacing, order in cls._BACKWARD_CASES:
            vector_field = (
                _make_periodic_vector_field(shape, device=device)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            yield label, (vector_field,), {"spacing": spacing, "order": order}

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


def _make_periodic_vector_field(
    shape: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Construct smooth periodic vector fields for benchmark cases."""
    axes = tuple(
        torch.arange(n, device=device, dtype=torch.float32) / float(n) for n in shape
    )
    if len(shape) == 1:
        (x0,) = axes
        return torch.sin(2.0 * torch.pi * x0).unsqueeze(0)

    if len(shape) == 2:
        x0, x1 = axes
        xx, yy = torch.meshgrid(x0, x1, indexing="ij")
        return torch.stack(
            (
                torch.sin(2.0 * torch.pi * xx),
                torch.cos(2.0 * torch.pi * yy),
            ),
            dim=0,
        )

    x0, x1, x2 = axes
    xx, yy, zz = torch.meshgrid(x0, x1, x2, indexing="ij")
    return torch.stack(
        (
            torch.sin(2.0 * torch.pi * xx),
            torch.cos(2.0 * torch.pi * yy),
            0.5 * torch.sin(2.0 * torch.pi * zz),
        ),
        dim=0,
    )


uniform_grid_divergence = UniformGridDivergence.make_function("uniform_grid_divergence")

__all__ = ["UniformGridDivergence", "uniform_grid_divergence"]
