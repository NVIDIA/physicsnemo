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

import warnings
from typing import List, Tuple

import torch
from torch import Tensor

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import interpolation_torch
from ._warp_impl import interpolation_warp


class GridToPointInterpolation(FunctionSpec):
    r"""Interpolate values from a structured grid at query point locations.

    This functional evaluates a scalar or multi-channel field defined on a regular
    Cartesian grid at arbitrary query points in 1D, 2D, or 3D.

    For a query point :math:`\mathbf{x}` and a grid field :math:`f`, interpolation
    is computed as a weighted sum over local stencil points:

    .. math::

       \hat{f}(\mathbf{x}) = \sum_{i \in \mathcal{N}(\mathbf{x})}
       w_i(\mathbf{x})\, f_i

    where :math:`\mathcal{N}(\mathbf{x})` is the interpolation neighborhood and
    :math:`w_i(\mathbf{x})` are interpolation weights.

    The interpolation mode controls the stencil and weights:

    - ``nearest_neighbor``: nearest grid point (piecewise constant, 1-point stencil)
    - ``linear``: multilinear interpolation (2^d stencil in d dimensions)
    - ``smooth_step_1``: multilinear-style interpolation with smooth-step weights
      :math:`3t^2 - 2t^3`
    - ``smooth_step_2``: multilinear-style interpolation with quintic smooth-step
      weights :math:`t^3(6t^2 - 15t + 10)`
    - ``gaussian``: local Gaussian weighting over a larger fixed stencil

    Notes
    -----
    - Grid spacing and extents are provided by ``grid``.
    - The ``warp`` and ``torch`` backends are intended to be numerically aligned.
    - ``warp`` is the default dispatch path for ``grid_to_point_interpolation``.
    - The deprecated ``interpolation`` alias defaults to ``torch`` unless an
      explicit ``implementation`` is provided.

    Parameters
    ----------
    query_points: torch.Tensor
        Points at which interpolation is to be performed.
    context_grid: torch.Tensor
        Source grid from which values are interpolated.
    grid: list[tuple[float, float, int]]
        Describes the grid's range and resolution.
    interpolation_type: str, optional
        Interpolation method name, by default ``"smooth_step_2"``.
    mem_speed_trade: bool, optional
        Trade-off between memory usage and speed.
    implementation : {"warp", "torch"} or None
        Implementation to use. When ``None``, dispatch selects the available
        implementation.

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
        context_grid: Tensor,
        grid: List[Tuple[float, float, int]],
        interpolation_type: str = "smooth_step_2",
        mem_speed_trade: bool = True,
    ) -> Tensor:
        return interpolation_warp(
            query_points,
            context_grid,
            grid,
            interpolation_type=interpolation_type,
            mem_speed_trade=mem_speed_trade,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        query_points: Tensor,
        context_grid: Tensor,
        grid: List[Tuple[float, float, int]],
        interpolation_type: str = "smooth_step_2",
        mem_speed_trade: bool = True,
    ) -> Tensor:
        return interpolation_torch(
            query_points,
            context_grid,
            grid,
            interpolation_type=interpolation_type,
            mem_speed_trade=mem_speed_trade,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        for label, dims, grid_size, num_points, interp_name in cls._BENCHMARK_CASES:
            grid = [(-1.0, 2.0, grid_size)] * dims
            linspace = [torch.linspace(x[0], x[1], x[2], device=device) for x in grid]
            mesh_grid = torch.meshgrid(linspace, indexing="ij")
            mesh_grid = torch.stack(mesh_grid, dim=0)
            context_grid = torch.zeros_like(mesh_grid[0:1])
            for power, coord in enumerate(mesh_grid, start=1):
                context_grid = context_grid + coord.unsqueeze(0) ** power
            context_grid = torch.sin(context_grid)
            query_points = torch.stack(
                [
                    torch.linspace(0.0, 1.0, num_points, device=device)
                    for _ in range(dims)
                ],
                axis=-1,
            )
            yield (
                label,
                (query_points, context_grid, grid),
                {"interpolation_type": interp_name, "mem_speed_trade": True},
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        for label, dims, grid_size, num_points, interp_name in cls._BENCHMARK_CASES:
            grid = [(-1.0, 2.0, grid_size)] * dims
            linspace = [torch.linspace(x[0], x[1], x[2], device=device) for x in grid]
            mesh_grid = torch.meshgrid(linspace, indexing="ij")
            mesh_grid = torch.stack(mesh_grid, dim=0)
            context_grid = torch.zeros_like(mesh_grid[0:1])
            for power, coord in enumerate(mesh_grid, start=1):
                context_grid = context_grid + coord.unsqueeze(0) ** power
            context_grid = torch.sin(context_grid).requires_grad_(True)
            query_points = torch.stack(
                [
                    torch.linspace(0.0, 1.0, num_points, device=device)
                    for _ in range(dims)
                ],
                axis=-1,
            ).requires_grad_(True)
            yield (
                label,
                (query_points, context_grid, grid),
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


grid_to_point_interpolation = GridToPointInterpolation.make_function(
    "grid_to_point_interpolation"
)


def interpolation(*args, **kwargs):
    """Deprecated alias for ``grid_to_point_interpolation``."""
    warnings.warn(
        "`interpolation` is deprecated and will be removed in a future release. "
        "Use `grid_to_point_interpolation` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Preserve historical default behavior for the deprecated alias while still
    # allowing explicit backend selection overrides.
    kwargs.setdefault("implementation", "torch")
    return grid_to_point_interpolation(*args, **kwargs)


__all__ = [
    "GridToPointInterpolation",
    "grid_to_point_interpolation",
    "interpolation",
]
