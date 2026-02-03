# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""
This layer is a compilable, ball-query operation.

By default, it will project a grid of points to a 1D set of points.

It does not support batch size > 1.
"""

import torch
import torch.nn as nn
from einops import rearrange

from physicsnemo.nn.functional import radius_search


class BQWarp(nn.Module):
    """
    Warp-based ball-query layer for finding neighboring points within a specified radius.

    This layer uses an accelerated ball query implementation to efficiently find points
    within a specified radius of query points.

    Only supports batch size 1.
    """

    def __init__(
        self,
        radius: float = 0.25,
        neighbors_in_radius: int | None = 10,
    ):
        """
        Initialize the BQWarp layer.

        Args:
            radius: Radius for ball query operation
            neighbors_in_radius: Maximum number of neighbors to return within radius. If None, all neighbors will be returned.
        """
        super().__init__()

        self.radius = radius
        self.neighbors_in_radius = neighbors_in_radius

    def forward(
        self, x: torch.Tensor, p_grid: torch.Tensor, reverse_mapping: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs ball query operation to find neighboring points and their features.

        This method uses the Warp-accelerated ball query implementation to find points
        within a specified radius. It can operate in two modes:
        - Forward mapping: Find points from x that are near p_grid points (reverse_mapping=False)
        - Reverse mapping: Find points from p_grid that are near x points (reverse_mapping=True)

        Args:
            x: Tensor of shape (batch_size, num_points, 3+features) containing point coordinates
               and their features
            p_grid: Tensor of shape (batch_size, grid_x, grid_y, grid_z, 3) containing grid point
                   coordinates
            reverse_mapping: Boolean flag to control the direction of the mapping:
                            - True: Find p_grid points near x points
                            - False: Find x points near p_grid points

        Returns:
            tuple containing:
                - mapping: Tensor containing indices of neighboring points
                - outputs: Tensor containing coordinates of the neighboring points
        """

        if x.shape[0] != 1 or p_grid.shape[0] != 1:
            raise ValueError("BQWarp only supports batch size 1")

        if p_grid.shape[-1] != x.shape[-1] or x.shape[-1] != 3:
            raise ValueError("The last dimension of p_grid and x must be 3")

        if p_grid.ndim != 3:
            if p_grid.ndim == 4:
                p_grid = rearrange(p_grid, "b nx ny c -> b (nx ny) c")
            elif p_grid.ndim == 5:
                p_grid = rearrange(p_grid, "b nx ny nz c -> b (nx ny nz) c")
            else:
                raise ValueError("p_grid must be 3D, 4D, 5D only")

        if reverse_mapping:
            mapping, outputs = radius_search(
                x[0],
                p_grid[0],
                self.radius,
                self.neighbors_in_radius,
                return_points=True,
            )
            mapping = mapping.unsqueeze(0)
            outputs = outputs.unsqueeze(0)
        else:
            mapping, outputs = radius_search(
                p_grid[0],
                x[0],
                self.radius,
                self.neighbors_in_radius,
                return_points=True,
            )
            mapping = mapping.unsqueeze(0)
            outputs = outputs.unsqueeze(0)

        return mapping, outputs
