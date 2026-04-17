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

from ._torch_impl import uniform_grid_gradient_torch_multi
from ._warp_impl import uniform_grid_gradient_warp_multi


class UniformGridGradient(FunctionSpec):
    r"""Compute periodic central-difference gradients on a uniform grid.

    This functional computes first-order and/or second-order
    derivatives of a scalar field defined on a 1D/2D/3D uniform Cartesian
    grid with periodic indexing.

    For each axis :math:`k`, the first derivative is:

    .. math::

       \partial_k f(\mathbf{i}) \approx
       \frac{f(\mathbf{i}+\hat{e}_k) - f(\mathbf{i}-\hat{e}_k)}{2\,\Delta x_k}

    and the pure second derivative is:

    .. math::

       \partial_{kk} f(\mathbf{i}) \approx
       \frac{f(\mathbf{i}+\hat{e}_k)-2f(\mathbf{i})+f(\mathbf{i}-\hat{e}_k)}
       {\Delta x_k^2}

    with periodic wrap-around at boundaries.

    Parameters
    ----------
    field : torch.Tensor
        Scalar grid field with shape ``(n0,)``, ``(n0,n1)``, or ``(n0,n1,n2)``.
    spacing : float | Sequence[float], optional
        Uniform spacing per axis. Use a scalar for isotropic spacing or a
        sequence matching field dimensionality.
    order : int, optional
        Central-difference accuracy order. Supported values are ``2`` and ``4``.
    derivative_orders : int | Sequence[int], optional
        Derivative orders to compute. Supported values are ``1``, ``2``, or
        ``(1, 2)``.
    include_mixed : bool, optional
        Include mixed second derivatives when requesting second derivatives.
        Mixed terms are appended in axis-pair order ``(x,y)``, ``(x,z)``,
        ``(y,z)``.
    implementation : {"warp", "torch"} or None
        Explicit backend selection. When ``None``, rank-based backend dispatch
        is used.

    Returns
    -------
    torch.Tensor
        Gradient tensor of shape ``(num_derivatives, *field.shape)``.
    """

    ### Benchmark input presets (small -> large workload).
    _BENCHMARK_CASES = (
        ("1d-n8192-o2-d1", (8192,), 0.01, 2, 1),
        ("1d-n8192-o4-d1", (8192,), 0.01, 4, 1),
        ("2d-512x512-o2-d1", (512, 512), (0.01, 0.02), 2, 1),
        ("2d-512x512-o2-d2", (512, 512), (0.01, 0.02), 2, 2),
        ("3d-128x128x128-o2-d1", (128, 128, 128), 0.02, 2, 1),
        ("3d-96x96x96-o2-d2", (96, 96, 96), 0.02, 2, 2),
    )

    _COMPARE_ATOL = 1e-5
    _COMPARE_RTOL = 1e-5
    _COMPARE_BACKWARD_ATOL = 1e-5
    _COMPARE_BACKWARD_RTOL = 1e-5

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        field: torch.Tensor,
        spacing: float | Sequence[float] = 1.0,
        order: int = 2,
        derivative_orders: int | Sequence[int] = 1,
        include_mixed: bool = False,
    ) -> torch.Tensor:
        """Dispatch uniform-grid gradients to the Warp backend."""
        return uniform_grid_gradient_warp_multi(
            field=field,
            spacing=spacing,
            order=order,
            derivative_orders=derivative_orders,
            include_mixed=include_mixed,
        )

    @FunctionSpec.register(name="torch", rank=2, baseline=True)
    def torch_forward(
        field: torch.Tensor,
        spacing: float | Sequence[float] = 1.0,
        order: int = 2,
        derivative_orders: int | Sequence[int] = 1,
        include_mixed: bool = False,
    ) -> torch.Tensor:
        """Dispatch uniform-grid gradients to eager PyTorch."""
        return uniform_grid_gradient_torch_multi(
            field=field,
            spacing=spacing,
            order=order,
            derivative_orders=derivative_orders,
            include_mixed=include_mixed,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark and parity input cases."""
        device = torch.device(device)

        ### Build periodic analytic fields for benchmark and parity coverage.
        for label, shape, spacing, order, derivative_order in cls._BENCHMARK_CASES:
            if len(shape) == 1:
                x = torch.linspace(0.0, 1.0, shape[0], device=device)
                field = torch.sin(2.0 * torch.pi * x)
            elif len(shape) == 2:
                x0 = torch.linspace(0.0, 1.0, shape[0], device=device)
                x1 = torch.linspace(0.0, 1.0, shape[1], device=device)
                xx, yy = torch.meshgrid(x0, x1, indexing="ij")
                field = torch.sin(2.0 * torch.pi * xx) + 0.5 * torch.cos(
                    2.0 * torch.pi * yy
                )
            else:
                x0 = torch.linspace(0.0, 1.0, shape[0], device=device)
                x1 = torch.linspace(0.0, 1.0, shape[1], device=device)
                x2 = torch.linspace(0.0, 1.0, shape[2], device=device)
                xx, yy, zz = torch.meshgrid(x0, x1, x2, indexing="ij")
                field = (
                    torch.sin(2.0 * torch.pi * xx)
                    + 0.5 * torch.cos(2.0 * torch.pi * yy)
                    + 0.25 * torch.sin(2.0 * torch.pi * zz)
                )

            ### Yield the labeled functional input case.
            yield (
                label,
                (field.to(torch.float32),),
                {
                    "spacing": spacing,
                    "order": order,
                    "derivative_orders": derivative_order,
                    "include_mixed": False,
                },
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative backward benchmark and parity input cases."""
        device = torch.device(device)

        ### Build representative differentiable fields for backward parity.
        backward_cases = (
            ("1d-grad-n4096-o2-d1", (4096,), 0.01, 2, 1),
            ("2d-grad-256x256-o2-d1", (256, 256), (0.01, 0.02), 2, 1),
            ("2d-grad-256x256-o2-d2", (256, 256), (0.01, 0.02), 2, 2),
            ("2d-grad-256x256-o4-d1", (256, 256), (0.01, 0.02), 4, 1),
            ("3d-grad-96x96x96-o2-d1", (96, 96, 96), 0.02, 2, 1),
            ("3d-grad-64x64x64-o2-d2", (64, 64, 64), 0.02, 2, 2),
        )

        for label, shape, spacing, order, derivative_order in backward_cases:
            if len(shape) == 1:
                x = torch.linspace(0.0, 1.0, shape[0], device=device)
                field = torch.sin(2.0 * torch.pi * x)
            elif len(shape) == 2:
                x0 = torch.linspace(0.0, 1.0, shape[0], device=device)
                x1 = torch.linspace(0.0, 1.0, shape[1], device=device)
                xx, yy = torch.meshgrid(x0, x1, indexing="ij")
                field = torch.sin(2.0 * torch.pi * xx) + 0.5 * torch.cos(
                    2.0 * torch.pi * yy
                )
            else:
                x0 = torch.linspace(0.0, 1.0, shape[0], device=device)
                x1 = torch.linspace(0.0, 1.0, shape[1], device=device)
                x2 = torch.linspace(0.0, 1.0, shape[2], device=device)
                xx, yy, zz = torch.meshgrid(x0, x1, x2, indexing="ij")
                field = (
                    torch.sin(2.0 * torch.pi * xx)
                    + 0.5 * torch.cos(2.0 * torch.pi * yy)
                    + 0.25 * torch.sin(2.0 * torch.pi * zz)
                )

            ### Yield differentiable field inputs for backward dispatch.
            yield (
                label,
                (field.to(torch.float32).detach().clone().requires_grad_(True),),
                {
                    "spacing": spacing,
                    "order": order,
                    "derivative_orders": derivative_order,
                    "include_mixed": False,
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


uniform_grid_gradient = UniformGridGradient.make_function("uniform_grid_gradient")


__all__ = ["UniformGridGradient", "uniform_grid_gradient"]
