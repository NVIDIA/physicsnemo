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

r"""
DoMINO Encoding Modules.

This module contains encoding layers for the DoMINO model architecture,
including local and multi-scale geometry encodings.
"""

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.nn import BQWarp

from .mlps import LocalPointConv


class LocalGeometryEncoding(Module):
    r"""
    A local geometry encoding module.

    This module applies a ball query to map point cloud features onto a volume mesh,
    then applies a local point convolution to process the spatial relationships.
    It is used to capture local geometric context around each point.

    Parameters
    ----------
    radius : float
        The radius of the ball query for neighbor searching.
    neighbors_in_radius : int
        The number of neighbors to consider within the ball query radius.
    total_neighbors_in_radius : int
        The total number of input features per neighbor (accounts for encoding type).
    base_layer : int
        The number of neurons in the hidden layer of the local point convolution MLP.
    activation : nn.Module
        The activation function to use in the MLP.
    grid_resolution : tuple[int, int, int]
        The resolution of the 3D grid as :math:`(N_x, N_y, N_z)`.

    Forward
    -------
    encoding_g : torch.Tensor
        Geometry encoding tensor of shape :math:`(B, C, N_x, N_y, N_z)` where
        :math:`C` is the number of encoding channels.
    volume_mesh_centers : torch.Tensor
        Volume mesh center coordinates of shape :math:`(B, N_{vol}, 3)`.
    p_grid : torch.Tensor
        Grid points tensor of shape :math:`(B, N_x, N_y, N_z, 3)`.

    Outputs
    -------
    torch.Tensor
        Local geometry encoding of shape :math:`(B, N_{vol}, D_{out})` where
        :math:`D_{out}` equals ``neighbors_in_radius``.

    See Also
    --------
    :class:`~physicsnemo.nn.BQWarp` : Ball query warp module used for neighbor searching.
    :class:`~physicsnemo.models.domino.mlps.LocalPointConv` : Local point convolution layer.
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
        super().__init__(meta=None)
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
        encoding_g: Float[torch.Tensor, " batch channels nx ny nz"],
        volume_mesh_centers: Float[torch.Tensor, " batch num_points 3"],
        p_grid: Float[torch.Tensor, " batch nx ny nz 3"],
    ) -> Float[torch.Tensor, " batch num_points out_features"]:
        r"""
        Compute local geometry encoding.

        Parameters
        ----------
        encoding_g : torch.Tensor
            Geometry encoding tensor of shape :math:`(B, C, N_x, N_y, N_z)`.
        volume_mesh_centers : torch.Tensor
            Volume mesh center coordinates of shape :math:`(B, N_{vol}, 3)`.
        p_grid : torch.Tensor
            Grid points tensor of shape :math:`(B, N_x, N_y, N_z, 3)`.

        Returns
        -------
        torch.Tensor
            Local geometry encoding of shape :math:`(B, N_{vol}, D_{out})`.
        """
        # Input validation
        if not torch.compiler.is_compiling():
            if encoding_g.ndim != 5:
                raise ValueError(
                    f"Expected encoding_g to be 5D (B, C, Nx, Ny, Nz), "
                    f"got {encoding_g.ndim}D with shape {tuple(encoding_g.shape)}"
                )
            if volume_mesh_centers.ndim != 3 or volume_mesh_centers.shape[-1] != 3:
                raise ValueError(
                    f"Expected volume_mesh_centers of shape (B, N, 3), "
                    f"got shape {tuple(volume_mesh_centers.shape)}"
                )
            if p_grid.ndim != 5 or p_grid.shape[-1] != 3:
                raise ValueError(
                    f"Expected p_grid of shape (B, Nx, Ny, Nz, 3), "
                    f"got shape {tuple(p_grid.shape)}"
                )

        batch_size = volume_mesh_centers.shape[0]
        nx, ny, nz = self.grid_resolution

        # Reshape grid for ball query
        p_grid = torch.reshape(p_grid, (batch_size, nx * ny * nz, 3))

        # Perform ball query to find neighbors
        mapping, outputs = self.bq_warp(
            volume_mesh_centers, p_grid, reverse_mapping=False
        )

        mapping = mapping.type(torch.int64)
        mask = mapping != 0

        # Sample geometry encoding at neighbor locations
        encoding_g_inner = []
        for j in range(encoding_g.shape[1]):
            geo_encoding = rearrange(encoding_g[:, j], "b nx ny nz -> b 1 (nx ny nz)")

            geo_encoding_sampled = torch.index_select(
                geo_encoding, 2, mapping.flatten()
            )
            geo_encoding_sampled = torch.reshape(geo_encoding_sampled, mask.shape)
            geo_encoding_sampled = geo_encoding_sampled * mask

            encoding_g_inner.append(geo_encoding_sampled)

        # Concatenate and apply local point convolution
        encoding_g_inner = torch.cat(encoding_g_inner, dim=2)
        encoding_g_inner = self.local_point_conv(encoding_g_inner)

        return encoding_g_inner


class MultiGeometryEncoding(Module):
    r"""
    Module to apply multiple local geometry encodings at different scales.

    This module stacks several :class:`LocalGeometryEncoding` modules with different
    radii and neighbor counts, then concatenates their outputs to create a
    multi-scale geometry representation.

    Parameters
    ----------
    radii : list[float]
        List of radii for each local geometry encoding scale.
    neighbors_in_radius : list[int]
        List of neighbor counts for each scale, corresponding to ``radii``.
    geo_encoding_type : str
        The type of geometry encoding. Can be ``"both"``, ``"stl"``, or ``"sdf"``.
    n_upstream_radii : int
        Number of upstream radii used for computing total neighbors.
    base_layer : int
        The number of neurons in the hidden layer of each local point convolution MLP.
    activation : nn.Module
        The activation function to use in the MLPs.
    grid_resolution : tuple[int, int, int]
        The resolution of the 3D grid as :math:`(N_x, N_y, N_z)`.

    Forward
    -------
    encoding_g : torch.Tensor
        Geometry encoding tensor of shape :math:`(B, C, N_x, N_y, N_z)`.
    volume_mesh_centers : torch.Tensor
        Volume mesh center coordinates of shape :math:`(B, N_{vol}, 3)`.
    p_grid : torch.Tensor
        Grid points tensor of shape :math:`(B, N_x, N_y, N_z, 3)`.

    Outputs
    -------
    torch.Tensor
        Concatenated multi-scale geometry encoding of shape :math:`(B, N_{vol}, D_{total})`
        where :math:`D_{total}` is the sum of all ``neighbors_in_radius`` values.

    See Also
    --------
    :class:`LocalGeometryEncoding` : Single-scale local geometry encoding module.
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
        super().__init__(meta=None)

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
        r"""
        Calculate total neighbors based on encoding type.

        Parameters
        ----------
        geo_encoding_type : str
            The type of geometry encoding (``"both"``, ``"stl"``, or ``"sdf"``).
        neighbors_in_radius : int
            Base number of neighbors in radius.
        n_upstream_radii : int
            Number of upstream radii.

        Returns
        -------
        int
            Total number of neighbors in radius for the encoding.
        """
        if geo_encoding_type == "both":
            total_neighbors_in_radius = neighbors_in_radius * (n_upstream_radii + 1)
        elif geo_encoding_type == "stl":
            total_neighbors_in_radius = neighbors_in_radius * (n_upstream_radii)
        elif geo_encoding_type == "sdf":
            total_neighbors_in_radius = neighbors_in_radius
        else:
            raise ValueError(
                f"Invalid geo_encoding_type: {geo_encoding_type}. "
                f"Must be 'both', 'stl', or 'sdf'."
            )

        return total_neighbors_in_radius

    def forward(
        self,
        encoding_g: Float[torch.Tensor, "batch channels nx ny nz"],
        volume_mesh_centers: Float[torch.Tensor, "batch num_points 3"],
        p_grid: Float[torch.Tensor, "batch nx ny nz 3"],
    ) -> Float[torch.Tensor, "batch num_points total_features"]:
        r"""
        Compute multi-scale geometry encoding.

        Parameters
        ----------
        encoding_g : torch.Tensor
            Geometry encoding tensor of shape :math:`(B, C, N_x, N_y, N_z)`.
        volume_mesh_centers : torch.Tensor
            Volume mesh center coordinates of shape :math:`(B, N_{vol}, 3)`.
        p_grid : torch.Tensor
            Grid points tensor of shape :math:`(B, N_x, N_y, N_z, 3)`.

        Returns
        -------
        torch.Tensor
            Concatenated multi-scale geometry encoding of shape :math:`(B, N_{vol}, D_{total})`.
        """
        # Input validation
        if not torch.compiler.is_compiling():
            if encoding_g.ndim != 5:
                raise ValueError(
                    f"Expected encoding_g to be 5D (B, C, Nx, Ny, Nz), "
                    f"got {encoding_g.ndim}D with shape {tuple(encoding_g.shape)}"
                )
            if volume_mesh_centers.ndim != 3 or volume_mesh_centers.shape[-1] != 3:
                raise ValueError(
                    f"Expected volume_mesh_centers of shape (B, N, 3), "
                    f"got shape {tuple(volume_mesh_centers.shape)}"
                )
            if p_grid.ndim != 5 or p_grid.shape[-1] != 3:
                raise ValueError(
                    f"Expected p_grid of shape (B, Nx, Ny, Nz, 3), "
                    f"got shape {tuple(p_grid.shape)}"
                )

        # Apply each local geometry encoding and concatenate results
        return torch.cat(
            [
                local_geo_encoding(encoding_g, volume_mesh_centers, p_grid)
                for local_geo_encoding in self.local_geo_encodings
            ],
            dim=-1,
        )
