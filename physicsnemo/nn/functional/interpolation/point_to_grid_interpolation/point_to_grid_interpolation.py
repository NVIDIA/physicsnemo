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

from typing import List, Tuple

import torch
from torch import Tensor

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import point_to_grid_interpolation_torch
from ._warp_impl import point_to_grid_interpolation_warp


class PointToGridInterpolation(FunctionSpec):
    r"""Scatter point values back onto a structured grid using interpolation weights.

    This functional maps values defined at query points onto a regular Cartesian
    grid in 1D/2D/3D. It is the adjoint-style counterpart to
    ``grid_to_point_interpolation``.

    For point values :math:`v_n` at points :math:`\mathbf{x}_n`, the gridded field
    :math:`g_i` is accumulated as:

    .. math::

       g_i = \sum_{n} w_i(\mathbf{x}_n)\, v_n

    where :math:`w_i(\mathbf{x}_n)` are interpolation weights for grid node
    :math:`i` at query point :math:`\mathbf{x}_n`.

    Parameters
    ----------
    query_points : torch.Tensor
        Query points with shape ``(num_points, dims)``.
    point_values : torch.Tensor
        Values at query points with shape ``(num_points, channels)``.
    grid : list[tuple[float, float, int]]
        Grid extent and resolution metadata.
    interpolation_type : str, optional
        Interpolation method name.
    mem_speed_trade : bool, optional
        Forwarded to the underlying grid-to-point implementation.
    implementation : {"warp", "torch"} or None
        Implementation to use. When ``None``, dispatch selects by rank.

    Notes
    -----
    - ``query_points`` and ``point_values`` currently support ``torch.float32``.
    - The ``warp`` and ``torch`` backends are intended to be numerically aligned.
    - ``warp`` is the default dispatch path for ``point_to_grid_interpolation``.
    """

    _BENCHMARK_CASES = (
        ("1d-nearest-g2048-n8192", 1, 2048, 8192, "nearest_neighbor"),
        ("1d-linear-g2048-n8192", 1, 2048, 8192, "linear"),
        ("2d-smooth1-g128-n1024", 2, 128, 1024, "smooth_step_1"),
        ("2d-smooth2-g128-n1024", 2, 128, 1024, "smooth_step_2"),
        ("3d-linear-g32-n512", 3, 32, 512, "linear"),
        ("3d-smooth2-g32-n512", 3, 32, 512, "smooth_step_2"),
        ("3d-gaussian-g32-n512", 3, 32, 512, "gaussian"),
    )
    _COMPARE_ATOL = 5e-5
    _COMPARE_RTOL = 1e-4
    _COMPARE_BACKWARD_ATOL = 2e-2
    _COMPARE_BACKWARD_RTOL = 5e-2

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        query_points: Tensor,
        point_values: Tensor,
        grid: List[Tuple[float, float, int]],
        interpolation_type: str = "smooth_step_2",
        mem_speed_trade: bool = True,
    ) -> Tensor:
        return point_to_grid_interpolation_warp(
            query_points,
            point_values,
            grid,
            interpolation_type=interpolation_type,
            mem_speed_trade=mem_speed_trade,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        query_points: Tensor,
        point_values: Tensor,
        grid: List[Tuple[float, float, int]],
        interpolation_type: str = "smooth_step_2",
        mem_speed_trade: bool = True,
    ) -> Tensor:
        return point_to_grid_interpolation_torch(
            query_points,
            point_values,
            grid,
            interpolation_type=interpolation_type,
            mem_speed_trade=mem_speed_trade,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        for label, dims, grid_size, num_points, interp_name in cls._BENCHMARK_CASES:
            grid = [(-1.0, 2.0, grid_size)] * dims
            query_points = torch.stack(
                [
                    torch.linspace(0.0, 1.0, num_points, device=device)
                    for _ in range(dims)
                ],
                axis=-1,
            )
            point_values = torch.stack(
                (
                    torch.sin(query_points.sum(dim=-1)),
                    torch.cos(query_points.prod(dim=-1)),
                ),
                dim=-1,
            )
            yield (
                label,
                (query_points, point_values, grid),
                {"interpolation_type": interp_name, "mem_speed_trade": True},
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        for label, dims, grid_size, num_points, interp_name in cls._BENCHMARK_CASES:
            grid = [(-1.0, 2.0, grid_size)] * dims
            query_points = torch.stack(
                [
                    torch.linspace(0.0, 1.0, num_points, device=device)
                    for _ in range(dims)
                ],
                axis=-1,
            ).requires_grad_(True)
            qp_detached = query_points.detach()
            point_values = torch.stack(
                (
                    torch.sin(qp_detached.sum(dim=-1)),
                    torch.cos(qp_detached.prod(dim=-1)),
                ),
                dim=-1,
            ).requires_grad_(True)
            yield (
                label,
                (query_points, point_values, grid),
                {"interpolation_type": interp_name, "mem_speed_trade": True},
            )

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_ATOL,
            rtol=cls._COMPARE_RTOL,
        )

    @classmethod
    def compare_backward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_BACKWARD_ATOL,
            rtol=cls._COMPARE_BACKWARD_RTOL,
        )


point_to_grid_interpolation = PointToGridInterpolation.make_function(
    "point_to_grid_interpolation"
)


__all__ = [
    "PointToGridInterpolation",
    "point_to_grid_interpolation",
]
