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

r"""Fourier Neural Operator (FNO) encoder layers.

This module contains reusable FNO encoder building blocks that can be used
in various FNO-based architectures.
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

import physicsnemo.nn as layers
from physicsnemo.core.module import Module


class FNO1DEncoder(Module):
    r"""1D Spectral encoder for FNO.

    This encoder applies a lifting network followed by spectral convolution layers
    in the Fourier domain for 1D input data.

    Parameters
    ----------
    in_channels : int, optional, default=1
        Number of input channels.
    num_fno_layers : int, optional, default=4
        Number of spectral convolutional layers.
    fno_layer_size : int, optional, default=32
        Latent features size in spectral convolutions.
    num_fno_modes : Union[int, List[int]], optional, default=16
        Number of Fourier modes kept in spectral convolutions.
    padding : Union[int, List[int]], optional, default=8
        Domain padding for spectral convolutions.
    padding_type : str, optional, default="constant"
        Type of padding for spectral convolutions.
    activation_fn : nn.Module, optional, default=nn.GELU()
        Activation function.
    coord_features : bool, optional, default=True
        Use coordinate grid as additional feature map.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, L)` where :math:`B` is batch size,
        :math:`C_{in}` is the number of input channels, and :math:`L` is the
        sequence length (spatial dimension).

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{latent}, L)` where :math:`C_{latent}`
        is ``fno_layer_size``.

    Examples
    --------
    >>> import torch
    >>> encoder = FNO1DEncoder(in_channels=3, fno_layer_size=32, num_fno_modes=8)
    >>> x = torch.randn(4, 3, 64)
    >>> output = encoder(x)
    >>> output.shape
    torch.Size([4, 32, 64])
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
    ) -> None:
        super().__init__()

        self._input_channels = in_channels
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.activation_fn = activation_fn

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
        self._build_lift_network()
        self._build_fno(num_fno_modes)

    def _build_lift_network(self) -> None:
        r"""Construct network for lifting variables to latent space."""
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            layers.Conv1dFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            layers.Conv1dFCLayer(int(self.fno_width / 2), self.fno_width)
        )

    def _build_fno(self, num_fno_modes: List[int]) -> None:
        r"""Construct FNO spectral convolution layers.

        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions.
        """
        # Build Neural Fourier Operators
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                layers.SpectralConv1d(self.fno_width, self.fno_width, num_fno_modes[0])
            )
            self.conv_layers.append(nn.Conv1d(self.fno_width, self.fno_width, 1))

    def forward(self, x: Float[Tensor, "B C_in L"]) -> Float[Tensor, "B C_latent L"]:
        r"""Forward pass of the 1D FNO encoder."""
        # Input validation: single check for ndim and channels
        if not torch.compiler.is_compiling():
            if x.ndim != 3 or x.shape[1] != self._input_channels:
                raise ValueError(
                    f"Expected 3D input (B, {self._input_channels}, L), "
                    f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
                )

        # Add coordinate features if enabled
        if self.coord_features:
            coord_feat = self._meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        # Lift input to latent space
        x = self.lift_network(x)

        # Apply padding for spectral convolution
        x = F.pad(x, (0, self.pad[0]), mode=self.padding_type)

        # Apply spectral convolution layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        # Remove padding
        x = x[..., : self.ipad[0]]
        return x

    def _meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        r"""Create 1D meshgrid feature.

        Parameters
        ----------
        shape : List[int]
            Tensor shape as ``[batch, channels, L]``.
        device : torch.device
            Device model is on.

        Returns
        -------
        Tensor
            Meshgrid tensor of shape :math:`(B, 1, L)`.
        """
        bsize, size_x = shape[0], shape[2]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1)
        return grid_x

    def grid_to_points(self, value: Tensor) -> Tuple[Tensor, List[int]]:
        r"""Convert from grid-based (image) to point-based representation.

        Parameters
        ----------
        value : Tensor
            Grid tensor of shape :math:`(B, C, L)`.

        Returns
        -------
        Tuple[Tensor, List[int]]
            Tuple of (flattened tensor, original shape).
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: Tensor, shape: List[int]) -> Tensor:
        r"""Convert from point-based to grid-based (image) representation.

        Parameters
        ----------
        value : Tensor
            Point tensor of shape :math:`(B \times X, C)`.
        shape : List[int]
            Original grid shape as ``[B, C, L]``.

        Returns
        -------
        Tensor
            Grid tensor of shape :math:`(B, C, L)`.
        """
        output = value.reshape(shape[0], shape[2], value.size(-1))
        return torch.permute(output, (0, 2, 1))


class FNO2DEncoder(Module):
    r"""2D Spectral encoder for FNO.

    This encoder applies a lifting network followed by spectral convolution layers
    in the Fourier domain for 2D input data.

    Parameters
    ----------
    in_channels : int, optional, default=1
        Number of input channels.
    num_fno_layers : int, optional, default=4
        Number of spectral convolutional layers.
    fno_layer_size : int, optional, default=32
        Latent features size in spectral convolutions.
    num_fno_modes : Union[int, List[int]], optional, default=16
        Number of Fourier modes kept in spectral convolutions.
    padding : Union[int, List[int]], optional, default=8
        Domain padding for spectral convolutions.
    padding_type : str, optional, default="constant"
        Type of padding for spectral convolutions.
    activation_fn : nn.Module, optional, default=nn.GELU()
        Activation function.
    coord_features : bool, optional, default=True
        Use coordinate grid as additional feature map.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, H, W)` where :math:`B` is batch size,
        :math:`C_{in}` is the number of input channels, and :math:`H, W` are spatial
        dimensions.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{latent}, H, W)` where :math:`C_{latent}`
        is ``fno_layer_size``.

    Examples
    --------
    >>> import torch
    >>> encoder = FNO2DEncoder(in_channels=3, fno_layer_size=32, num_fno_modes=8)
    >>> x = torch.randn(4, 3, 32, 32)
    >>> output = encoder(x)
    >>> output.shape
    torch.Size([4, 32, 32, 32])
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
    ) -> None:
        super().__init__()
        self._input_channels = in_channels
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.activation_fn = activation_fn

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
        self._build_lift_network()
        self._build_fno(num_fno_modes)

    def _build_lift_network(self) -> None:
        r"""Construct network for lifting variables to latent space."""
        # Initial lift network
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            layers.Conv2dFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            layers.Conv2dFCLayer(int(self.fno_width / 2), self.fno_width)
        )

    def _build_fno(self, num_fno_modes: List[int]) -> None:
        r"""Construct FNO spectral convolution layers.

        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions.
        """
        # Build Neural Fourier Operators
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                layers.SpectralConv2d(
                    self.fno_width, self.fno_width, num_fno_modes[0], num_fno_modes[1]
                )
            )
            self.conv_layers.append(nn.Conv2d(self.fno_width, self.fno_width, 1))

    def forward(
        self, x: Float[Tensor, "B C_in H W"]
    ) -> Float[Tensor, "B C_latent H W"]:
        r"""Forward pass of the 2D FNO encoder."""
        # Input validation: single check for ndim and channels
        if not torch.compiler.is_compiling():
            if x.ndim != 4 or x.shape[1] != self._input_channels:
                raise ValueError(
                    f"Expected 4D input (B, {self._input_channels}, H, W), "
                    f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
                )

        # Add coordinate features if enabled
        if self.coord_features:
            coord_feat = self._meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        # Lift input to latent space
        x = self.lift_network(x)

        # Apply padding for spectral convolution
        x = F.pad(x, (0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)

        # Apply spectral convolution layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        # Remove padding
        x = x[..., : self.ipad[0], : self.ipad[1]]

        return x

    def _meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        r"""Create 2D meshgrid feature.

        Parameters
        ----------
        shape : List[int]
            Tensor shape as ``[batch, channels, height, width]``.
        device : torch.device
            Device model is on.

        Returns
        -------
        Tensor
            Meshgrid tensor of shape :math:`(B, 2, H, W)`.
        """
        bsize, size_x, size_y = shape[0], shape[2], shape[3]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        return torch.cat((grid_x, grid_y), dim=1)

    def grid_to_points(self, value: Tensor) -> Tuple[Tensor, List[int]]:
        r"""Convert from grid-based (image) to point-based representation.

        Parameters
        ----------
        value : Tensor
            Grid tensor of shape :math:`(B, C, H, W)`.

        Returns
        -------
        Tuple[Tensor, List[int]]
            Tuple of (flattened tensor, original shape).
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: Tensor, shape: List[int]) -> Tensor:
        r"""Convert from point-based to grid-based (image) representation.

        Parameters
        ----------
        value : Tensor
            Point tensor of shape :math:`(B \times H \times W, C)`.
        shape : List[int]
            Original grid shape as ``[B, C, H, W]``.

        Returns
        -------
        Tensor
            Grid tensor of shape :math:`(B, C, H, W)`.
        """
        output = value.reshape(shape[0], shape[2], shape[3], value.size(-1))
        return torch.permute(output, (0, 3, 1, 2))


class FNO3DEncoder(Module):
    r"""3D Spectral encoder for FNO.

    This encoder applies a lifting network followed by spectral convolution layers
    in the Fourier domain for 3D input data.

    Parameters
    ----------
    in_channels : int, optional, default=1
        Number of input channels.
    num_fno_layers : int, optional, default=4
        Number of spectral convolutional layers.
    fno_layer_size : int, optional, default=32
        Latent features size in spectral convolutions.
    num_fno_modes : Union[int, List[int]], optional, default=16
        Number of Fourier modes kept in spectral convolutions.
    padding : Union[int, List[int]], optional, default=8
        Domain padding for spectral convolutions.
    padding_type : str, optional, default="constant"
        Type of padding for spectral convolutions.
    activation_fn : nn.Module, optional, default=nn.GELU()
        Activation function.
    coord_features : bool, optional, default=True
        Use coordinate grid as additional feature map.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, D, H, W)` where :math:`B` is batch
        size, :math:`C_{in}` is the number of input channels, and :math:`D, H, W` are
        spatial dimensions.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{latent}, D, H, W)` where :math:`C_{latent}`
        is ``fno_layer_size``.

    Examples
    --------
    >>> import torch
    >>> encoder = FNO3DEncoder(in_channels=3, fno_layer_size=32, num_fno_modes=8)
    >>> x = torch.randn(4, 3, 16, 16, 16)
    >>> output = encoder(x)
    >>> output.shape
    torch.Size([4, 32, 16, 16, 16])
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
    ) -> None:
        super().__init__()

        self._input_channels = in_channels
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.activation_fn = activation_fn

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
        self._build_lift_network()
        self._build_fno(num_fno_modes)

    def _build_lift_network(self) -> None:
        r"""Construct network for lifting variables to latent space."""
        # Initial lift network
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            layers.Conv3dFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            layers.Conv3dFCLayer(int(self.fno_width / 2), self.fno_width)
        )

    def _build_fno(self, num_fno_modes: List[int]) -> None:
        r"""Construct FNO spectral convolution layers.

        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions.
        """
        # Build Neural Fourier Operators
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                layers.SpectralConv3d(
                    self.fno_width,
                    self.fno_width,
                    num_fno_modes[0],
                    num_fno_modes[1],
                    num_fno_modes[2],
                )
            )
            self.conv_layers.append(nn.Conv3d(self.fno_width, self.fno_width, 1))

    def forward(
        self, x: Float[Tensor, "B C_in D H W"]
    ) -> Float[Tensor, "B C_latent D H W"]:
        r"""Forward pass of the 3D FNO encoder."""
        # Input validation: single check for ndim and channels
        if not torch.compiler.is_compiling():
            if x.ndim != 5 or x.shape[1] != self._input_channels:
                raise ValueError(
                    f"Expected 5D input (B, {self._input_channels}, D, H, W), "
                    f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
                )

        # Add coordinate features if enabled
        if self.coord_features:
            coord_feat = self._meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        # Lift input to latent space
        x = self.lift_network(x)

        # Apply padding for spectral convolution
        x = F.pad(
            x,
            (0, self.pad[2], 0, self.pad[1], 0, self.pad[0]),
            mode=self.padding_type,
        )

        # Apply spectral convolution layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        # Remove padding
        x = x[..., : self.ipad[0], : self.ipad[1], : self.ipad[2]]
        return x

    def _meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        r"""Create 3D meshgrid feature.

        Parameters
        ----------
        shape : List[int]
            Tensor shape as ``[batch, channels, depth, height, width]``.
        device : torch.device
            Device model is on.

        Returns
        -------
        Tensor
            Meshgrid tensor of shape :math:`(B, 3, D, H, W)`.
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
        r"""Convert from grid-based (image) to point-based representation.

        Parameters
        ----------
        value : Tensor
            Grid tensor of shape :math:`(B, C, D, H, W)`.

        Returns
        -------
        Tuple[Tensor, List[int]]
            Tuple of (flattened tensor, original shape).
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 4, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: Tensor, shape: List[int]) -> Tensor:
        r"""Convert from point-based to grid-based (image) representation.

        Parameters
        ----------
        value : Tensor
            Point tensor of shape :math:`(B \times D \times H \times W, C)`.
        shape : List[int]
            Original grid shape as ``[B, C, D, H, W]``.

        Returns
        -------
        Tensor
            Grid tensor of shape :math:`(B, C, D, H, W)`.
        """
        output = value.reshape(shape[0], shape[2], shape[3], shape[4], value.size(-1))
        return torch.permute(output, (0, 4, 1, 2, 3))


class FNO4DEncoder(Module):
    r"""4D Spectral encoder for FNO.

    This encoder applies a lifting network followed by spectral convolution layers
    in the Fourier domain for 4D input data (3D spatial + time).

    Parameters
    ----------
    in_channels : int, optional, default=1
        Number of input channels.
    num_fno_layers : int, optional, default=4
        Number of spectral convolutional layers.
    fno_layer_size : int, optional, default=32
        Latent features size in spectral convolutions.
    num_fno_modes : Union[int, List[int]], optional, default=16
        Number of Fourier modes kept in spectral convolutions.
    padding : Union[int, List[int]], optional, default=8
        Domain padding for spectral convolutions.
    padding_type : str, optional, default="constant"
        Type of padding for spectral convolutions.
    activation_fn : nn.Module, optional, default=nn.GELU()
        Activation function.
    coord_features : bool, optional, default=True
        Use coordinate grid as additional feature map.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, X, Y, Z, T)` where :math:`B` is batch
        size, :math:`C_{in}` is the number of input channels, and :math:`X, Y, Z, T`
        are spatial and temporal dimensions.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{latent}, X, Y, Z, T)` where
        :math:`C_{latent}` is ``fno_layer_size``.

    Examples
    --------
    >>> import torch
    >>> encoder = FNO4DEncoder(in_channels=3, fno_layer_size=32, num_fno_modes=4)
    >>> x = torch.randn(2, 3, 8, 8, 8, 8)
    >>> output = encoder(x)
    >>> output.shape
    torch.Size([2, 32, 8, 8, 8, 8])
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
    ) -> None:
        super().__init__()

        self._input_channels = in_channels
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.activation_fn = activation_fn

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
        self._build_lift_network()
        self._build_fno(num_fno_modes)

    def _build_lift_network(self) -> None:
        r"""Construct network for lifting variables to latent space."""
        # Initial lift network
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            layers.ConvNdFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            layers.ConvNdFCLayer(int(self.fno_width / 2), self.fno_width)
        )

    def _build_fno(self, num_fno_modes: List[int]) -> None:
        r"""Construct FNO spectral convolution layers.

        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions.
        """
        # Build Neural Fourier Operators
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                layers.SpectralConv4d(
                    self.fno_width,
                    self.fno_width,
                    num_fno_modes[0],
                    num_fno_modes[1],
                    num_fno_modes[2],
                    num_fno_modes[3],
                )
            )
            self.conv_layers.append(
                layers.ConvNdKernel1Layer(self.fno_width, self.fno_width)
            )

    def forward(
        self, x: Float[Tensor, "B C_in X Y Z T"]
    ) -> Float[Tensor, "B C_latent X Y Z T"]:
        r"""Forward pass of the 4D FNO encoder."""
        # Input validation: single check for ndim and channels
        if not torch.compiler.is_compiling():
            if x.ndim != 6 or x.shape[1] != self._input_channels:
                raise ValueError(
                    f"Expected 6D input (B, {self._input_channels}, X, Y, Z, T), "
                    f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
                )

        # Add coordinate features if enabled
        if self.coord_features:
            coord_feat = self._meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        # Lift input to latent space
        x = self.lift_network(x)

        # Apply padding for spectral convolution
        x = F.pad(
            x,
            (0, self.pad[3], 0, self.pad[2], 0, self.pad[1], 0, self.pad[0]),
            mode=self.padding_type,
        )

        # Apply spectral convolution layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        # Remove padding
        x = x[..., : self.ipad[0], : self.ipad[1], : self.ipad[2], : self.ipad[3]]
        return x

    def _meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        r"""Create 4D meshgrid feature.

        Parameters
        ----------
        shape : List[int]
            Tensor shape as ``[batch, channels, x, y, z, t]``.
        device : torch.device
            Device model is on.

        Returns
        -------
        Tensor
            Meshgrid tensor of shape :math:`(B, 4, X, Y, Z, T)`.
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
        r"""Convert from grid-based (image) to point-based representation.

        Parameters
        ----------
        value : Tensor
            Grid tensor of shape :math:`(B, C, X, Y, Z, T)`.

        Returns
        -------
        Tuple[Tensor, List[int]]
            Tuple of (flattened tensor, original shape).
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 4, 5, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: Tensor, shape: List[int]) -> Tensor:
        r"""Convert from point-based to grid-based (image) representation.

        Parameters
        ----------
        value : Tensor
            Point tensor of shape :math:`(B \times X \times Y \times Z \times T, C)`.
        shape : List[int]
            Original grid shape as ``[B, C, X, Y, Z, T]``.

        Returns
        -------
        Tensor
            Grid tensor of shape :math:`(B, C, X, Y, Z, T)`.
        """
        output = value.reshape(
            shape[0], shape[2], shape[3], shape[4], shape[5], value.size(-1)
        )
        return torch.permute(output, (0, 5, 1, 2, 3, 4))
