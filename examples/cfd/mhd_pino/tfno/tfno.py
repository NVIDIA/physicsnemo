# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import physicsnemo  # noqa: F401 for docs
import physicsnemo.models.layers as layers
from .spectral_layers import (
    FactorizedSpectralConv1d,
    FactorizedSpectralConv2d,
    FactorizedSpectralConv3d,
    FactorizedSpectralConv4d,
)

from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.models.module import Module

# ===================================================================
# ===================================================================
# 1D TFNO
# ===================================================================
# ===================================================================


class TFNO1DEncoder(nn.Module):
    """1D Spectral encoder for TFNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    rank : float, optional
        Rank of the decomposition, by default 1.0
    factorization : {'CP', 'TT', 'Tucker'}, optional
        Tensor factorization to use to decompose the tensor, by default 'CP'
    fixed_rank_modes : List[int], optional
        A list of modes for which the initial value is not modified, by default None
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict, optional
        Additional arguments to initialization of factorized tensors, by default dict()
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_fno_layers: int = 4,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
        rank: float = 1.0,
        factorization: str = "cp",
        fixed_rank_modes: List[int] = None,
        decomposition_kwargs: dict = dict(),
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.activation_fn = activation_fn

        # TensorLy arguments
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs

        # Add relative coordinate feature
        self.coord_features = coord_features
        if self.coord_features:
            self.in_channels = self.in_channels + 1

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding]
        self.pad = padding[:1]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes]

        # build lift
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) -> None:
        """construct network for lifting variables to latent space."""
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            layers.Conv1dFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            layers.Conv1dFCLayer(int(self.fno_width / 2), self.fno_width)
        )

    def build_fno(self, num_fno_modes: List[int]) -> None:
        """construct FNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        # Build Neural Fourier Operators
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                FactorizedSpectralConv1d(
                    self.fno_width,
                    self.fno_width,
                    num_fno_modes[0],
                    self.rank,
                    self.factorization,
                    self.fixed_rank_modes,
                    self.decomposition_kwargs,
                )
            )
            self.conv_layers.append(nn.Conv1d(self.fno_width, self.fno_width, 1))

    def forward(self, x: Tensor) -> Tensor:
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        # (left, right)
        x = F.pad(x, (0, self.pad[0]), mode=self.padding_type)
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        x = x[..., : self.ipad[0]]
        return x

    def meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        """Creates 1D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x = shape[0], shape[2]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1)
        return grid_x

    def grid_to_points(self, value: Tensor) -> Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: Tensor, shape: List[int]) -> Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], value.size(-1))
        return torch.permute(output, (0, 2, 1))


# ===================================================================
# ===================================================================
# 2D TFNO
# ===================================================================
# ===================================================================


class TFNO2DEncoder(nn.Module):
    """2D Spectral encoder for TFNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    rank : float, optional
        Rank of the decomposition, by default 1.0
    factorization : {'CP', 'TT', 'Tucker'}, optional
        Tensor factorization to use to decompose the tensor, by default 'CP'
    fixed_rank_modes : List[int], optional
        A list of modes for which the initial value is not modified, by default None
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict, optional
        Additional arguments to initialization of factorized tensors, by default dict()
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_fno_layers: int = 4,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
        rank: float = 1.0,
        factorization: str = "cp",
        fixed_rank_modes: List[int] = None,
        decomposition_kwargs: dict = dict(),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.activation_fn = activation_fn

        # TensorLy arguments
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs

        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 2

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding]
        padding = padding + [0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:2]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes]

        # build lift
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) -> None:
        """construct network for lifting variables to latent space."""
        # Initial lift network
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            layers.Conv2dFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            layers.Conv2dFCLayer(int(self.fno_width / 2), self.fno_width)
        )

    def build_fno(self, num_fno_modes: List[int]) -> None:
        """construct TFNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        # Build Neural Fourier Operators
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                FactorizedSpectralConv2d(
                    self.fno_width,
                    self.fno_width,
                    num_fno_modes[0],
                    num_fno_modes[1],
                    self.rank,
                    self.factorization,
                    self.fixed_rank_modes,
                    self.decomposition_kwargs,
                )
            )
            self.conv_layers.append(nn.Conv2d(self.fno_width, self.fno_width, 1))

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D FNO"
            )

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        # (left, right, top, bottom)
        x = F.pad(x, (0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        # remove padding
        x = x[..., : self.ipad[0], : self.ipad[1]]

        return x

    def meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        """Creates 2D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y = shape[0], shape[2], shape[3]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        return torch.cat((grid_x, grid_y), dim=1)

    def grid_to_points(self, value: Tensor) -> Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: Tensor, shape: List[int]) -> Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], shape[3], value.size(-1))
        return torch.permute(output, (0, 3, 1, 2))


# ===================================================================
# ===================================================================
# 3D TFNO
# ===================================================================
# ===================================================================


class TFNO3DEncoder(nn.Module):
    """3D Spectral encoder for TFNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    rank : float, optional
        Rank of the decomposition, by default 1.0
    factorization : {'CP', 'TT', 'Tucker'}, optional
        Tensor factorization to use to decompose the tensor, by default 'CP'
    fixed_rank_modes : List[int], optional
        A list of modes for which the initial value is not modified, by default None
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict, optional
        Additional arguments to initialization of factorized tensors, by default dict()
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_fno_layers: int = 4,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
        rank: float = 1.0,
        factorization: str = "cp",
        fixed_rank_modes: List[int] = None,
        decomposition_kwargs: dict = dict(),
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.activation_fn = activation_fn

        # TensorLy arguments
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs

        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 3

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding, padding]
        padding = padding + [0, 0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:3]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes, num_fno_modes]

        # build lift
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) -> None:
        """construct network for lifting variables to latent space."""
        # Initial lift network
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            layers.Conv3dFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            layers.Conv3dFCLayer(int(self.fno_width / 2), self.fno_width)
        )

    def build_fno(self, num_fno_modes: List[int]) -> None:
        """construct FNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        # Build Neural Fourier Operators
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                FactorizedSpectralConv3d(
                    self.fno_width,
                    self.fno_width,
                    num_fno_modes[0],
                    num_fno_modes[1],
                    num_fno_modes[2],
                    self.rank,
                    self.factorization,
                    self.fixed_rank_modes,
                    self.decomposition_kwargs,
                )
            )
            self.conv_layers.append(nn.Conv3d(self.fno_width, self.fno_width, 1))

    def forward(self, x: Tensor) -> Tensor:
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        # (left, right, top, bottom, front, back)
        x = F.pad(
            x,
            (0, self.pad[2], 0, self.pad[1], 0, self.pad[0]),
            mode=self.padding_type,
        )
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        x = x[..., : self.ipad[0], : self.ipad[1], : self.ipad[2]]
        return x

    def meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        """Creates 3D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y, size_z = shape[0], shape[2], shape[3], shape[4]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        return torch.cat((grid_x, grid_y, grid_z), dim=1)

    def grid_to_points(self, value: Tensor) -> Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 4, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: Tensor, shape: List[int]) -> Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], shape[3], shape[4], value.size(-1))
        return torch.permute(output, (0, 4, 1, 2, 3))


# ===================================================================
# ===================================================================
# 4D TFNO
# ===================================================================
# ===================================================================


class TFNO4DEncoder(nn.Module):
    """4D Spectral encoder for TFNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    rank : float, optional
        Rank of the decomposition, by default 1.0
    factorization : {'CP', 'TT', 'Tucker'}, optional
        Tensor factorization to use to decompose the tensor, by default 'CP'
    fixed_rank_modes : List[int], optional
        A list of modes for which the initial value is not modified, by default None
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict, optional
        Additional arguments to initialization of factorized tensors, by default dict()
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_fno_layers: int = 4,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
        rank: float = 1.0,
        factorization: str = "cp",
        fixed_rank_modes: List[int] = None,
        decomposition_kwargs: dict = dict(),
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.activation_fn = activation_fn

        # TensorLy arguments
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs

        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 4

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding, padding, padding]
        padding = padding + [0, 0, 0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:4]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes, num_fno_modes, num_fno_modes]

        # build lift
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) -> None:
        """construct network for lifting variables to latent space."""
        # Initial lift network
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            layers.ConvNdFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            layers.ConvNdFCLayer(int(self.fno_width / 2), self.fno_width)
        )

    def build_fno(self, num_fno_modes: List[int]) -> None:
        """construct TFNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        # Build Neural Fourier Operators
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                FactorizedSpectralConv4d(
                    self.fno_width,
                    self.fno_width,
                    num_fno_modes[0],
                    num_fno_modes[1],
                    num_fno_modes[2],
                    num_fno_modes[3],
                    self.rank,
                    self.factorization,
                    self.fixed_rank_modes,
                    self.decomposition_kwargs,
                )
            )
            self.conv_layers.append(
                layers.ConvNdKernel1Layer(self.fno_width, self.fno_width)
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        # (left, right, top, bottom, front, back, past, future)
        x = F.pad(
            x,
            (0, self.pad[3], 0, self.pad[2], 0, self.pad[1], 0, self.pad[0]),
            mode=self.padding_type,
        )
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        x = x[..., : self.ipad[0], : self.ipad[1], : self.ipad[2], : self.ipad[3]]
        return x

    def meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        """Creates 4D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y, size_z, size_t = (
            shape[0],
            shape[2],
            shape[3],
            shape[4],
            shape[5],
        )
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
        grid_t = torch.linspace(0, 1, size_t, dtype=torch.float32, device=device)
        grid_x, grid_y, grid_z, grid_t = torch.meshgrid(
            grid_x, grid_y, grid_z, grid_t, indexing="ij"
        )
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        grid_t = grid_t.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        return torch.cat((grid_x, grid_y, grid_z, grid_t), dim=1)

    def grid_to_points(self, value: Tensor) -> Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 4, 5, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: Tensor, shape: List[int]) -> Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(
            shape[0], shape[2], shape[3], shape[4], shape[5], value.size(-1)
        )
        return torch.permute(output, (0, 5, 1, 2, 3, 4))


# ===================================================================
# ===================================================================
# General TFNO Model
# ===================================================================
# ===================================================================


class TFNO(nn.Module):
    """Tensor Factorized Fourier neural operator (FNO) model.

    Note
    ----
    The TFNO architecture supports options for 1D, 2D, 3D and 4D fields which can
    be controlled using the `dimension` parameter.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    decoder_layers : int, optional
        Number of decoder layers, by default 1
    decoder_layer_size : int, optional
        Number of neurons in decoder layers, by default 32
    decoder_activation_fn : str, optional
        Activation function for decoder, by default "silu"
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    latent_channels : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding : int, optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : str, optional
        Activation function, by default "gelu"
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    rank : float, optional
        Rank of the decomposition, by default 1.0
    factorization : {'CP', 'TT', 'Tucker'}, optional
        Tensor factorization to use to decompose the tensor, by default 'CP'
    fixed_rank_modes : List[int], optional
        A list of modes for which the initial value is not modified, by default None
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict, optional
        Additional arguments to initialization of factorized tensors, by default dict()

    Example
    -------
    >>> # define the 2d TFNO model
    >>> model = physicsnemo.models.fno.TFNO(
    ...     in_channels=4,
    ...     out_channels=3,
    ...     decoder_layers=2,
    ...     decoder_layer_size=32,
    ...     dimension=2,
    ...     latent_channels=32,
    ...     num_fno_layers=2,
    ...     padding=0,
    ... )
    >>> input = torch.randn(32, 4, 32, 32) #(N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([32, 3, 32, 32])

    Note
    ----
    Reference: Rosofsky, Shawn G. and Huerta, E. A. "Magnetohydrodynamics with
    Physics Informed Neural Operators." arXiv preprint arXiv:2302.08332 (2023).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        decoder_layers: int = 1,
        decoder_layer_size: int = 32,
        decoder_activation_fn: str = "silu",
        dimension: int = 2,
        latent_channels: int = 32,
        num_fno_layers: int = 4,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: int = 8,
        padding_type: str = "constant",
        activation_fn: str = "gelu",
        coord_features: bool = True,
        rank: float = 1.0,
        factorization: str = "cp",
        fixed_rank_modes: List[int] = None,
        decomposition_kwargs: dict = dict(),
    ) -> None:
        super().__init__()

        self.num_fno_layers = num_fno_layers
        self.num_fno_modes = num_fno_modes
        self.padding = padding
        self.padding_type = padding_type
        self.activation_fn = layers.get_activation(activation_fn)
        self.coord_features = coord_features
        self.dimension = dimension

        # TensorLy arguments
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs

        # decoder net
        self.decoder_net = FullyConnected(
            in_features=latent_channels,
            layer_size=decoder_layer_size,
            out_features=out_channels,
            num_layers=decoder_layers,
            activation_fn=decoder_activation_fn,
        )

        TFNOModel = self.getTFNOEncoder()

        self.spec_encoder = TFNOModel(
            in_channels,
            num_fno_layers=self.num_fno_layers,
            fno_layer_size=latent_channels,
            num_fno_modes=self.num_fno_modes,
            padding=self.padding,
            padding_type=self.padding_type,
            activation_fn=self.activation_fn,
            coord_features=self.coord_features,
            rank=self.rank,
            factorization=self.factorization,
            fixed_rank_modes=self.fixed_rank_modes,
            decomposition_kwargs=self.decomposition_kwargs,
        )

    def getTFNOEncoder(self):
        "Return correct TFNO ND Encoder"
        if self.dimension == 1:
            return TFNO1DEncoder
        elif self.dimension == 2:
            return TFNO2DEncoder
        elif self.dimension == 3:
            return TFNO3DEncoder
        elif self.dimension == 4:
            return TFNO4DEncoder
        else:
            raise NotImplementedError(
                "Invalid dimensionality. Only 1D, 2D, 3D and 4D FNO implemented"
            )

    def forward(self, x: Tensor) -> Tensor:
        # Fourier encoder
        y_latent = self.spec_encoder(x)

        # Reshape to pointwise inputs if not a conv FC model
        y_shape = y_latent.shape
        y_latent, y_shape = self.spec_encoder.grid_to_points(y_latent)

        # Decoder
        y = self.decoder_net(y_latent)

        # Convert back into grid
        y = self.spec_encoder.points_to_grid(y, y_shape)

        return y
