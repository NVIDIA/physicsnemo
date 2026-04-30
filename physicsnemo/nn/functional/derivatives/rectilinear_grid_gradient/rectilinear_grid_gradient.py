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

from ._torch_impl import rectilinear_grid_gradient_torch_multi
from ._warp_impl import (
    rectilinear_grid_gradient_warp_multi,
)


class RectilinearGridGradient(FunctionSpec):
    r"""Compute periodic gradients on rectilinear grids with nonuniform spacing.

    This functional computes first-order and/or second-order
    derivatives of a scalar field on a 1D/2D/3D rectilinear grid where each
    axis has independent, potentially nonuniform coordinate spacing.

    For each axis :math:`k`, first-order nonuniform central differencing is:

    .. math::

       \partial_k f_i \approx
       a_i\,f_{i-1} + b_i\,f_i + c_i\,f_{i+1}

    with

    .. math::

       a_i = -\frac{h_i^+}{h_i^-(h_i^-+h_i^+)}, \quad
       b_i = \frac{h_i^+ - h_i^-}{h_i^- h_i^+}, \quad
       c_i = \frac{h_i^-}{h_i^+(h_i^-+h_i^+)}

    and pure second derivatives are:

    .. math::

       \partial_{kk} f_i \approx
       \tilde{a}_i\,f_{i-1} + \tilde{b}_i\,f_i + \tilde{c}_i\,f_{i+1}

    with

    .. math::

       \tilde{a}_i = \frac{2}{h_i^-(h_i^-+h_i^+)}, \quad
       \tilde{b}_i = -\frac{2}{h_i^- h_i^+}, \quad
       \tilde{c}_i = \frac{2}{h_i^+(h_i^-+h_i^+)}

    where :math:`h_i^-` and :math:`h_i^+` are left/right periodic distances
    along that axis.

    Parameters
    ----------
    field : torch.Tensor
        Scalar grid field with shape ``(n0,)``, ``(n0,n1)``, or ``(n0,n1,n2)``.
    coordinates : Sequence[torch.Tensor]
        Per-axis coordinate tensors ``(x0, x1, x2)`` matching field dimensions.
        Each axis tensor must be rank-1, strictly increasing, and length
        compatible with ``field.shape[axis]``.
    periods : float | Sequence[float] | None, optional
        Period length per axis. If ``None``, each axis is inferred as
        ``coords[-1] - coords[0] + (coords[1] - coords[0])``.
    derivative_orders : int | Sequence[int], optional
        Derivative orders to compute. Supported values are ``1``, ``2``, or
        ``(1, 2)``.
    include_mixed : bool, optional
        Include mixed second derivatives when requesting second derivatives.
        Mixed terms are appended in axis-pair order ``(x,y)``, ``(x,z)``,
        ``(y,z)``.
    implementation : {"warp", "torch"} or None
        Explicit backend selection. When ``None``, dispatch selects by rank.

    Returns
    -------
    torch.Tensor
        Gradient tensor of shape ``(num_derivatives, *field.shape)``.
    """

    ### Benchmark input presets (small -> large workload).
    _BENCHMARK_CASES = (
        ("1d-n8192-d1", (8192,), 1),
        ("1d-n512-d2", (512,), 2),
        ("2d-384x384-d1", (384, 384), 1),
        ("2d-256x256-d2", (256, 256), 2),
        ("3d-96x96x96-d1", (96, 96, 96), 1),
        ("3d-64x64x64-d2", (64, 64, 64), 2),
    )

    _COMPARE_ATOL = 5e-2
    _COMPARE_RTOL = 5e-2
    _COMPARE_BACKWARD_ATOL = 5e-2
    _COMPARE_BACKWARD_RTOL = 5e-2

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        field: torch.Tensor,
        coordinates: Sequence[torch.Tensor],
        periods: float | Sequence[float] | None = None,
        derivative_orders: int | Sequence[int] = 1,
        include_mixed: bool = False,
    ) -> torch.Tensor:
        """Dispatch rectilinear gradients to the Warp backend."""
        return rectilinear_grid_gradient_warp_multi(
            field=field,
            coordinates=coordinates,
            periods=periods,
            derivative_orders=derivative_orders,
            include_mixed=include_mixed,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        field: torch.Tensor,
        coordinates: Sequence[torch.Tensor],
        periods: float | Sequence[float] | None = None,
        derivative_orders: int | Sequence[int] = 1,
        include_mixed: bool = False,
    ) -> torch.Tensor:
        """Dispatch rectilinear gradients to eager PyTorch."""
        return rectilinear_grid_gradient_torch_multi(
            field=field,
            coordinates=coordinates,
            periods=periods,
            derivative_orders=derivative_orders,
            include_mixed=include_mixed,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark and parity input cases."""
        device = torch.device(device)

        ### Build periodic nonuniform rectilinear coordinates and analytic fields.
        for label, shape, derivative_order in cls._BENCHMARK_CASES:
            if len(shape) == 1:
                n0 = shape[0]
                s0 = torch.linspace(0.0, 1.0, n0 + 1, device=device)[:-1]
                x0 = s0 + 0.04 * torch.sin(2.0 * torch.pi * s0)
                field = torch.sin(2.0 * torch.pi * x0)
                coordinates = (x0.to(torch.float32),)
                periods = 1.0
            elif len(shape) == 2:
                n0, n1 = shape
                s0 = torch.linspace(0.0, 1.0, n0 + 1, device=device)[:-1]
                s1 = torch.linspace(0.0, 1.0, n1 + 1, device=device)[:-1]
                x0 = s0 + 0.04 * torch.sin(2.0 * torch.pi * s0)
                x1 = s1 + 0.03 * torch.sin(2.0 * torch.pi * s1)
                xx, yy = torch.meshgrid(x0, x1, indexing="ij")
                field = torch.sin(2.0 * torch.pi * xx) + 0.5 * torch.cos(
                    2.0 * torch.pi * yy
                )
                coordinates = (x0.to(torch.float32), x1.to(torch.float32))
                periods = (1.0, 1.0)
            else:
                n0, n1, n2 = shape
                s0 = torch.linspace(0.0, 1.0, n0 + 1, device=device)[:-1]
                s1 = torch.linspace(0.0, 1.0, n1 + 1, device=device)[:-1]
                s2 = torch.linspace(0.0, 1.0, n2 + 1, device=device)[:-1]
                x0 = s0 + 0.04 * torch.sin(2.0 * torch.pi * s0)
                x1 = s1 + 0.03 * torch.sin(2.0 * torch.pi * s1)
                x2 = s2 + 0.02 * torch.sin(2.0 * torch.pi * s2)
                xx, yy, zz = torch.meshgrid(x0, x1, x2, indexing="ij")
                field = (
                    torch.sin(2.0 * torch.pi * xx)
                    + 0.5 * torch.cos(2.0 * torch.pi * yy)
                    + 0.25 * torch.sin(2.0 * torch.pi * zz)
                )
                coordinates = (
                    x0.to(torch.float32),
                    x1.to(torch.float32),
                    x2.to(torch.float32),
                )
                periods = (1.0, 1.0, 1.0)

            ### Yield each labeled benchmark/parity input.
            yield (
                label,
                (field.to(torch.float32), coordinates),
                {
                    "periods": periods,
                    "derivative_orders": derivative_order,
                    "include_mixed": False,
                },
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative backward benchmark and parity input cases."""
        device = torch.device(device)

        ### Build differentiable field inputs for backward parity checks.
        backward_cases = (
            ("1d-grad-n4096-d1", (4096,), 1),
            ("1d-grad-n512-d2", (512,), 2),
            ("2d-grad-256x256-d1", (256, 256), 1),
            ("2d-grad-192x192-d2", (192, 192), 2),
            ("3d-grad-80x80x80-d1", (80, 80, 80), 1),
            ("3d-grad-56x56x56-d2", (56, 56, 56), 2),
        )

        for label, shape, derivative_order in backward_cases:
            if len(shape) == 1:
                n0 = shape[0]
                s0 = torch.linspace(0.0, 1.0, n0 + 1, device=device)[:-1]
                x0 = s0 + 0.04 * torch.sin(2.0 * torch.pi * s0)
                field = torch.sin(2.0 * torch.pi * x0)
                coordinates = (x0.to(torch.float32),)
                periods = 1.0
            elif len(shape) == 2:
                n0, n1 = shape
                s0 = torch.linspace(0.0, 1.0, n0 + 1, device=device)[:-1]
                s1 = torch.linspace(0.0, 1.0, n1 + 1, device=device)[:-1]
                x0 = s0 + 0.04 * torch.sin(2.0 * torch.pi * s0)
                x1 = s1 + 0.03 * torch.sin(2.0 * torch.pi * s1)
                xx, yy = torch.meshgrid(x0, x1, indexing="ij")
                field = torch.sin(2.0 * torch.pi * xx) + 0.5 * torch.cos(
                    2.0 * torch.pi * yy
                )
                coordinates = (x0.to(torch.float32), x1.to(torch.float32))
                periods = (1.0, 1.0)
            else:
                n0, n1, n2 = shape
                s0 = torch.linspace(0.0, 1.0, n0 + 1, device=device)[:-1]
                s1 = torch.linspace(0.0, 1.0, n1 + 1, device=device)[:-1]
                s2 = torch.linspace(0.0, 1.0, n2 + 1, device=device)[:-1]
                x0 = s0 + 0.04 * torch.sin(2.0 * torch.pi * s0)
                x1 = s1 + 0.03 * torch.sin(2.0 * torch.pi * s1)
                x2 = s2 + 0.02 * torch.sin(2.0 * torch.pi * s2)
                xx, yy, zz = torch.meshgrid(x0, x1, x2, indexing="ij")
                field = (
                    torch.sin(2.0 * torch.pi * xx)
                    + 0.5 * torch.cos(2.0 * torch.pi * yy)
                    + 0.25 * torch.sin(2.0 * torch.pi * zz)
                )
                coordinates = (
                    x0.to(torch.float32),
                    x1.to(torch.float32),
                    x2.to(torch.float32),
                )
                periods = (1.0, 1.0, 1.0)

            yield (
                label,
                (
                    field.to(torch.float32).detach().clone().requires_grad_(True),
                    coordinates,
                ),
                {
                    "periods": periods,
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


rectilinear_grid_gradient = RectilinearGridGradient.make_function(
    "rectilinear_grid_gradient"
)


__all__ = ["RectilinearGridGradient", "rectilinear_grid_gradient"]
