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

from typing import List, Tuple

import torch
from torch import Tensor

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import interpolation_torch
from ._warp_impl import interpolation_warp


class Interpolation(FunctionSpec):
    """Interpolate values from a grid at query point locations.

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

    Notes
    -----
    TODO: ``torch`` is the default dispatch implementation for now. The
    Warp implementation will be promoted to the default after additional
    validation and testing.
    """

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=1)
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

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
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
    def make_inputs(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        cases = [
            ("1d-nearest", 1, 2048, 8192, "nearest_neighbor"),
            ("1d-linear", 1, 2048, 8192, "linear"),
            ("2d-smooth1", 2, 128, 1024, "smooth_step_1"),
            ("2d-smooth2", 2, 128, 1024, "smooth_step_2"),
            ("3d-linear", 3, 32, 512, "linear"),
            ("3d-smooth2", 3, 32, 512, "smooth_step_2"),
            ("3d-gaussian", 3, 32, 512, "gaussian"),
        ]
        for label, dims, grid_size, num_points, interp_name in cases:
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
                f"{label}-g{grid_size}-n{num_points}",
                (query_points, context_grid, grid),
                {"interpolation_type": interp_name, "mem_speed_trade": True},
            )


interpolation = Interpolation.make_function("interpolation")


__all__ = [
    "Interpolation",
    "interpolation",
]
