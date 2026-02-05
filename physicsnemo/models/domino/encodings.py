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

"""
This code contains the DoMINO model architecture.
The DoMINO class contains an architecture to model both surface and
volume quantities together as well as separately (controlled using
the config.yaml file)
"""

import torch
import torch.nn as nn
from einops import rearrange

from physicsnemo.nn import BQWarp

from .mlps import LocalPointConv


class LocalGeometryEncoding(nn.Module):
    """
    A local geometry encoding module.

    This will apply a ball query to the input features, mapping the point cloud
    to the volume mesh, and then apply a local point convolution to the output.

    Args:
        radius: The radius of the ball query.
        neighbors_in_radius: The number of neighbors in the radius of the ball query.
        total_neighbors_in_radius: The total number of neighbors in the radius of the ball query.
        base_layer: The number of neurons in the hidden layer of the MLP.
        activation: The activation function to use in the MLP.
        grid_resolution: The resolution of the grid.
    """

    def __init__(
        self,
        radius: float,
        neighbors_in_radius: int,
        total_neighbors_in_radius: int,
        base_layer: int,
        activation: nn.Module,
        grid_resolution: tuple[int, int, int],
    ):
        super().__init__()
        self.bq_warp = BQWarp(
            radius=radius,
            neighbors_in_radius=neighbors_in_radius,
        )
        self.local_point_conv = LocalPointConv(
            input_features=total_neighbors_in_radius,
            base_layer=base_layer,
            output_features=neighbors_in_radius,
            activation=activation,
        )
        self.grid_resolution = grid_resolution

    def forward(
        self,
        encoding_g: torch.Tensor,
        volume_mesh_centers: torch.Tensor,
        p_grid: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = volume_mesh_centers.shape[0]
        nx, ny, nz = self.grid_resolution

        p_grid = torch.reshape(p_grid, (batch_size, nx * ny * nz, 3))
        mapping, outputs = self.bq_warp(
            volume_mesh_centers, p_grid, reverse_mapping=False
        )

        mapping = mapping.type(torch.int64)
        mask = mapping != 0

        encoding_g_inner = []
        for j in range(encoding_g.shape[1]):
            geo_encoding = rearrange(encoding_g[:, j], "b nx ny nz -> b 1 (nx ny nz)")

            geo_encoding_sampled = torch.index_select(
                geo_encoding, 2, mapping.flatten()
            )
            geo_encoding_sampled = torch.reshape(geo_encoding_sampled, mask.shape)
            geo_encoding_sampled = geo_encoding_sampled * mask

            encoding_g_inner.append(geo_encoding_sampled)
        encoding_g_inner = torch.cat(encoding_g_inner, dim=2)
        encoding_g_inner = self.local_point_conv(encoding_g_inner)

        return encoding_g_inner


class MultiGeometryEncoding(nn.Module):
    """
    Module to apply multiple local geometry encodings

    This will stack several local geometry encodings together, and concatenate the results.

    Args:
        radii: The list of radii of the local geometry encodings.
        neighbors_in_radius: The list of number of neighbors in the radius of the local geometry encodings.
        geo_encoding_type: The type of geometry encoding to use. Can be "both", "stl", or "sdf".
        base_layer: The number of neurons in the hidden layer of the MLP.
        activation: The activation function to use in the MLP.
        grid_resolution: The resolution of the grid.
    """

    def __init__(
        self,
        radii: list[float],
        neighbors_in_radius: list[int],
        geo_encoding_type: str,
        n_upstream_radii: int,
        base_layer: int,
        activation: nn.Module,
        grid_resolution: tuple[int, int, int],
    ):
        super().__init__()

        self.local_geo_encodings = nn.ModuleList(
            [
                LocalGeometryEncoding(
                    radius=r,
                    neighbors_in_radius=n,
                    total_neighbors_in_radius=self.calculate_total_neighbors_in_radius(
                        geo_encoding_type, n, n_upstream_radii
                    ),
                    base_layer=base_layer,
                    activation=activation,
                    grid_resolution=grid_resolution,
                )
                for r, n in zip(radii, neighbors_in_radius)
            ]
        )

    def calculate_total_neighbors_in_radius(
        self, geo_encoding_type: str, neighbors_in_radius: int, n_upstream_radii: int
    ) -> int:
        if geo_encoding_type == "both":
            total_neighbors_in_radius = neighbors_in_radius * (n_upstream_radii + 1)
        elif geo_encoding_type == "stl":
            total_neighbors_in_radius = neighbors_in_radius * (n_upstream_radii)
        elif geo_encoding_type == "sdf":
            total_neighbors_in_radius = neighbors_in_radius

        return total_neighbors_in_radius

    def forward(
        self,
        encoding_g: torch.Tensor,
        volume_mesh_centers: torch.Tensor,
        p_grid: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                local_geo_encoding(encoding_g, volume_mesh_centers, p_grid)
                for local_geo_encoding in self.local_geo_encodings
            ],
            dim=-1,
        )
