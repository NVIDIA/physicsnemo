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

from .._request_utils import (
    normalize_derivative_orders,
    normalize_include_mixed,
    validate_mixed_request,
)
from ._torch_impl import (
    meshless_fd_derivatives_torch,
    meshless_fd_stencil_points_torch,
)


class MeshlessFDDerivatives(FunctionSpec):
    """Compute meshless finite-difference derivatives from local stencil values.

    This functional expects values already sampled on a canonical
    Cartesian ``{-1, 0, 1}`` stencil around each query point.
    It does not build stencil coordinates internally; it only maps stencil
    values to derivative estimates using central finite-difference formulas.

    Parameters
    ----------
    stencil_values : torch.Tensor
        Values sampled on a canonical ``{-1,0,1}`` stencil with shape
        ``(num_points, stencil_size)`` or ``(num_points, stencil_size, channels)``.
        Stencil sizes must be ``3``, ``9``, or ``27``.
    spacing : float | Sequence[float], optional
        Stencil spacing per axis.
    derivative_orders : int | Sequence[int], optional
        Derivative orders to compute. Supported values are ``1``, ``2``, or
        ``(1, 2)``.
    include_mixed : bool, optional
        Include mixed second derivatives when requesting second derivatives.
    implementation : {"torch"} or None
        Implementation to use. When ``None``, dispatch selects the available
        implementation.

    Returns
    -------
    torch.Tensor
        Stacked derivatives with shape ``(num_derivatives, num_points)`` for scalar
        input or ``(num_derivatives, num_points, channels)`` for vector input.

    Notes
    -----
    Derivative stack ordering is deterministic: first derivatives, then pure
    second derivatives, then mixed second derivatives in axis-combination
    order.

    The stencil size infers dimensionality:
    ``3 -> 1D``, ``9 -> 2D``, ``27 -> 3D``.
    """

    _BENCHMARK_CASES = (
        ("1d-scalar-n4096", 4096, 1, 0.01, 1, False, 1),
        ("2d-scalar-n4096-o1", 4096, 2, (0.01, 0.02), 1, False, 1),
        ("2d-vector-n4096-o2", 4096, 2, (0.01, 0.02), 2, True, 2),
        ("3d-scalar-n2048-o2", 2048, 3, (0.01, 0.015, 0.02), 2, True, 1),
    )

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(
        stencil_values: torch.Tensor,
        spacing: float | Sequence[float] = 1.0,
        derivative_orders: int | Sequence[int] = 1,
        include_mixed: bool = False,
    ) -> torch.Tensor:
        """Dispatch meshless finite-difference derivatives to the torch backend."""
        requested_orders = normalize_derivative_orders(
            derivative_orders=derivative_orders,
            function_name="meshless_fd_derivatives",
        )
        mixed_terms = normalize_include_mixed(
            include_mixed=include_mixed,
            function_name="meshless_fd_derivatives",
        )

        ndim = _infer_dim_from_stencil_size(stencil_values)
        if ndim is not None:
            validate_mixed_request(
                derivative_orders=requested_orders,
                include_mixed=mixed_terms,
                ndim=ndim,
                function_name="meshless_fd_derivatives",
            )

        outputs: list[torch.Tensor] = []
        if 1 in requested_orders:
            outputs.append(
                meshless_fd_derivatives_torch(
                    stencil_values=stencil_values,
                    spacing=spacing,
                    order=1,
                    return_mixed_derivs=False,
                )
            )
        if 2 in requested_orders:
            outputs.append(
                meshless_fd_derivatives_torch(
                    stencil_values=stencil_values,
                    spacing=spacing,
                    order=2,
                    return_mixed_derivs=mixed_terms,
                )
            )
        return torch.cat(outputs, dim=0)

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark and parity input cases."""
        device = torch.device(device)
        for (
            label,
            num_points,
            dim,
            spacing,
            order,
            return_mixed_derivs,
            channels,
        ) in cls._BENCHMARK_CASES:
            points = torch.rand(num_points, dim, device=device, dtype=torch.float32)
            stencil_points = meshless_fd_stencil_points_torch(points, spacing=spacing)
            stencil_values = cls._evaluate_stencil(stencil_points, channels=channels)
            yield (
                label,
                (stencil_values,),
                {
                    "spacing": spacing,
                    "derivative_orders": order,
                    "include_mixed": return_mixed_derivs,
                },
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative backward benchmark and parity input cases."""
        device = torch.device(device)
        backward_cases = (
            ("1d-grad-n2048", 2048, 1, 0.01, 1, False, 2),
            ("2d-grad-n2048-o2", 2048, 2, (0.01, 0.02), 2, True, 2),
            ("3d-grad-n1024-o2", 1024, 3, (0.01, 0.015, 0.02), 2, True, 1),
        )
        for (
            label,
            num_points,
            dim,
            spacing,
            order,
            return_mixed_derivs,
            channels,
        ) in backward_cases:
            points = torch.rand(num_points, dim, device=device, dtype=torch.float32)
            stencil_points = meshless_fd_stencil_points_torch(points, spacing=spacing)
            stencil_values = (
                cls._evaluate_stencil(stencil_points, channels=channels)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            yield (
                label,
                (stencil_values,),
                {
                    "spacing": spacing,
                    "derivative_orders": order,
                    "include_mixed": return_mixed_derivs,
                },
            )

    @staticmethod
    def _evaluate_stencil(
        stencil_points: torch.Tensor,
        channels: int,
    ) -> torch.Tensor:
        """Generate smooth multi-channel stencil values for benchmark inputs."""
        x = stencil_points[..., 0]
        if stencil_points.shape[-1] == 1:
            values = [torch.sin(2.0 * x) + 0.3 * x.square()]
        elif stencil_points.shape[-1] == 2:
            y = stencil_points[..., 1]
            values = [
                torch.sin(1.4 * x) * torch.cos(0.7 * y) + 0.2 * x * y,
                x.square() + y.pow(3),
            ]
        else:
            y = stencil_points[..., 1]
            z = stencil_points[..., 2]
            values = [
                torch.sin(1.2 * x) * torch.cos(0.8 * y) * torch.sin(0.6 * z)
                + 0.1 * x * y * z,
                x.square() + 0.5 * y.square() - z,
            ]

        stacked = torch.stack(values[:channels], dim=-1)
        if channels == 1:
            return stacked[..., 0]
        return stacked


meshless_fd_derivatives = MeshlessFDDerivatives.make_function("meshless_fd_derivatives")


__all__ = [
    "MeshlessFDDerivatives",
    "meshless_fd_derivatives",
]


def _infer_dim_from_stencil_size(stencil_values: torch.Tensor) -> int | None:
    """Infer dimensionality from stencil shape when it is structurally valid."""
    if stencil_values.ndim not in (2, 3):
        return None
    return {3: 1, 9: 2, 27: 3}.get(stencil_values.shape[1])
