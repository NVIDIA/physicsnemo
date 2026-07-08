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

from typing import Callable, Sequence, Tuple, Union

import torch
import torch as th

from physicsnemo.nn.module.hpx import HEALPixLayer

from .normalization import ConditionalLayerNorm


#
# Helper: standard LayerNorm over channel dimension for (B, C, H, W)
#
class _LayerNormOverChannels(th.nn.Module):
    """Applies nn.LayerNorm over the channel dimension for (B, C, H, W) tensors."""

    def __init__(self, channel_depth: int, eps: float = 1e-5):
        super().__init__()
        self.norm = th.nn.LayerNorm(channel_depth, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


#
# RECURRENT BLOCKS
#


class ConvGRUBlock(th.nn.Module):
    """Class that implements a Convolutional GRU
    Code modified from
    https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        kernel_size: int = 1,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        kernel_size: int, optional
            Size of the convolutioonal kernel
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()

        self.channels = in_channels
        self.conv_gates = geometry_layer(
            layer=torch.nn.Conv2d,
            in_channels=in_channels + self.channels,
            out_channels=2 * self.channels,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
        )
        self.conv_can = geometry_layer(
            layer=torch.nn.Conv2d,
            in_channels=in_channels + self.channels,
            out_channels=self.channels,  # for candidate neural memory
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
        )
        self.h = th.zeros(1, 1, 1, 1)

    def forward(self, inputs: Sequence) -> Sequence:
        """Forward pass of the ConvGRUBlock

        Parameters
        ----------
        inputs: Sequence
            Input to the forward pass

        Returns
        -------
        Sequence
            Result of the forward pass
        """
        if inputs.shape != self.h.shape:
            self.h = th.zeros_like(inputs)
        combined = th.cat([inputs, self.h], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = th.split(combined_conv, self.channels, dim=1)
        reset_gate = th.sigmoid(gamma)
        update_gate = th.sigmoid(beta)

        combined = th.cat([inputs, reset_gate * self.h], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = th.tanh(cc_cnm)

        h_next = (1 - update_gate) * self.h + update_gate * cnm
        self.h = h_next

        return inputs + h_next

    def reset(self):
        """Reset the update gates"""
        self.h = th.zeros_like(self.h)


#
# CONV BLOCKS
#


class BasicConvBlock(th.nn.Module):
    """Convolution block consisting of n subsequent convolutions and activations"""

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        latent_channels: int = None,
        activation: th.nn.Module = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutioonal kernel
        dilation: int, optional
            Spacing between kernel points, passed to torch.nn.Conv2d
        n_layers:
            Number of convolutional layers
        latent_channels:
            Number of latent channels
        activation: torch.nn.Module, optional
            Activation function to use
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        if latent_channels is None:
            latent_channels = max(in_channels, out_channels)
        convblock = []
        for n in range(n_layers):
            convblock.append(
                geometry_layer(
                    layer=torch.nn.Conv2d,
                    in_channels=in_channels if n == 0 else latent_channels,
                    out_channels=out_channels if n == n_layers - 1 else latent_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                )
            )
            if activation is not None:
                convblock.append(activation)
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x):
        """Forward pass of the BasicConvBlock

        Parameters
        ----------
        x: torch.Tensor
            inputs to the forward pass

        Returns
        -------
        torch.Tensor
            result of the forward pass
        """
        return self.convblock(x)


class ConvNeXtBlock(th.nn.Module):
    """Class implementing a modified ConvNeXt network as described in https://arxiv.org/pdf/2201.03545.pdf
    and shown in figure 4
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,  # not used, but required for hydra instantiation
        upscale_factor: int = 4,
        activation: th.nn.Module = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutioonal kernels
        dilation: int, optional
            Spacing between kernel points, passed to torch.nn.Conv2d
        upscale_factor: int, optional
            Upscale factor to apply on the number of latent channels
        latent_channels: int, optional
            Number of latent channels
        activation: torch.nn.Module, optional
            Activation function to use between layers
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()

        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        # Convolution block
        convblock = []
        # 3x3 convolution increasing channels
        convblock.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=in_channels,
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock.append(activation)
        # 3x3 convolution maintaining increased channels
        convblock.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock.append(activation)
        # Linear postprocessing
        convblock.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x):
        """Forward pass of the ConvNextBlock

        Parameters
        ----------
        x: torch.Tensor
            inputs to the forward pass

        Returns
        -------
        torch.Tensor
            result of the forward pass
        """
        return self.skip_module(x) + self.convblock(x)


class DoubleConvNeXtBlock(th.nn.Module):
    """Modification of ConvNeXtBlock block this time putting two sequentially
    in a single block with the number of channels in the middle being the
    number of latent channels
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,  # not used, but required for hydra instantiation
        upscale_factor: int = 4,
        latent_channels: int = 1,
        activation: th.nn.Module = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        conditional_layer_norm: Callable = None,
        conditional_layer_norm_once: bool = False,
        dropout: float = 0.0,
    ):
        """
        Parameters:
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        latent_channels: int, optional
            Number of latent channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutioonal kernels
        dilation: int, optional
            Spacing between kernel points, passed to torch.nn.Conv2d
        upscale_factor: int, optional
            Upscale factor to apply on the number of latent channels
        activation: torch.nn.Module, optional
            Activation function to use between layers
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()

        self.cln_enabled = conditional_layer_norm is not None

        if in_channels == int(latent_channels):
            self.skip_module1 = lambda x: (
                x
            )  # Identity-function required in forward pass
        else:
            self.skip_module1 = geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=in_channels,
                out_channels=int(latent_channels),
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        if out_channels == int(latent_channels):
            self.skip_module2 = lambda x: (
                x
            )  # Identity-function required in forward pass
        else:
            self.skip_module2 = geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=int(latent_channels),
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )

        # check if we're applying a layer norm at the beginning
        # we've got two ConvNeXt equivalent blocks in the layer, so have two entry points
        # TODO: conditional layer norm once is doing two things, it's applying a norm on block entry
        # and switch from conditional to non-conditional layer norm. This is not ideal and should be fixed once we determine
        # what works best.
        if conditional_layer_norm_once:
            if conditional_layer_norm is not None:
                # Conditional norm at the beginning of the block
                self.entry_norm1 = conditional_layer_norm(channel_depth=in_channels)
                self.entry_norm2 = conditional_layer_norm(channel_depth=latent_channels)
            else:
                # Regular layer normalization at the beginning of the block
                self.entry_norm1 = _LayerNormOverChannels(channel_depth=in_channels)
                self.entry_norm2 = _LayerNormOverChannels(channel_depth=latent_channels)
        else:
            # No normalization at the beginning
            self.entry_norm1 = None
            self.entry_norm2 = None

        # 1st ConvNeXt block, the output of this one remains internal
        convblock1 = []
        # 3x3 convolution establishing latent channels channels
        convblock1.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=in_channels,
                out_channels=int(latent_channels),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )

        # Apply batch norm and conditional layer norm if needed
        if conditional_layer_norm is not None and not conditional_layer_norm_once:
            cln = conditional_layer_norm(channel_depth=int(latent_channels))
            convblock1.append(cln)

        if activation is not None:
            convblock1.append(activation)
        if dropout > 0.0:
            convblock1.append(th.nn.Dropout2d(p=dropout))

        # 1x1 convolution establishing increased channels
        convblock1.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=int(latent_channels),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=1,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )

        # Apply layer norm if needed
        if conditional_layer_norm_once:
            convblock1.append(
                _LayerNormOverChannels(
                    channel_depth=int(latent_channels * upscale_factor)
                )
            )
        elif conditional_layer_norm is not None:
            cln = conditional_layer_norm(
                channel_depth=int(latent_channels * upscale_factor)
            )
            convblock1.append(cln)

        if activation is not None:
            convblock1.append(activation)
        if dropout > 0.0:
            convblock1.append(th.nn.Dropout2d(p=dropout))

        # 1x1 convolution returning to latent channels
        convblock1.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=int(latent_channels),
                kernel_size=1,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock1.append(activation)
        if dropout > 0.0:
            convblock1.append(th.nn.Dropout2d(p=dropout))
        self.convblock1 = th.nn.ModuleList(convblock1)

        # 2nd ConNeXt block, takes the output of the first convnext block
        convblock2 = []
        # 3x3 convolution establishing latent channels channels
        convblock2.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=int(latent_channels),
                out_channels=int(latent_channels),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        # Apply batch norm and conditional layer norm if needed
        if conditional_layer_norm is not None and not conditional_layer_norm_once:
            cln = conditional_layer_norm(channel_depth=int(latent_channels))
            convblock2.append(cln)

        if activation is not None:
            convblock2.append(activation)
        if dropout > 0.0:
            convblock2.append(th.nn.Dropout2d(p=dropout))

        # 1x1 convolution establishing increased channels
        convblock2.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=int(latent_channels),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=1,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        # Apply layer norm if needed
        if conditional_layer_norm_once:
            convblock2.append(
                _LayerNormOverChannels(
                    channel_depth=int(latent_channels * upscale_factor)
                )
            )
        elif conditional_layer_norm is not None:
            cln = conditional_layer_norm(
                channel_depth=int(latent_channels * upscale_factor)
            )
            convblock2.append(cln)

        if activation is not None:
            convblock2.append(activation)
        if dropout > 0.0:
            convblock2.append(th.nn.Dropout2d(p=dropout))

        # 1x1 convolution reducing to output channels
        convblock2.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock2.append(activation)
        if dropout > 0.0:
            convblock2.append(th.nn.Dropout2d(p=dropout))
        self.convblock2 = th.nn.ModuleList(convblock2)

    def forward(self, x, conditions_cln=None):
        """Forward pass of the DoubleConvNextBlock

        Parameters
        ----------
        x: torch.Tensor
            inputs to the forward pass

        Returns
        -------
        torch.Tensor
            result of the forward pass
        conditions_cln: torch.Tensor, optional
            conditions for the conditional layer normalization
        """

        # TODO: performance of skip connectioni hasn't been compared
        # check  cln(x) vs. cln(x1_residual + x) in the future
        # save residual for the first block
        x1_residual = self.skip_module1(x)

        # entry norm for the first block
        if self.entry_norm1 is not None:
            if conditions_cln is not None:
                x = self.entry_norm1(x, conditions=conditions_cln)
            else:
                x = self.entry_norm1(x)

        # internal convnext result
        for layer in self.convblock1:
            if isinstance(layer, ConditionalLayerNorm):
                x = layer(x, conditions=conditions_cln)
            else:
                x = layer(x)
        x1 = x1_residual + x

        # save residual for the second block
        x2_residual = self.skip_module2(x1)

        # entry norm for the second block
        if self.entry_norm2 is not None:
            if conditions_cln is not None:
                x1 = self.entry_norm2(x1, conditions=conditions_cln)
            else:
                x1 = self.entry_norm2(x1)

        # return second convnext result
        for layer in self.convblock2:
            if isinstance(layer, ConditionalLayerNorm):
                x1 = layer(x1, conditions=conditions_cln)
            else:
                x1 = layer(x1)
        return x2_residual + x1


class Multi_SymmetricConvNeXtBlock(th.nn.Module):
    """
    Wrapper for SymmetricConvNeXtBlock that allows serial linking of blocks.
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        upscale_factor: int = 4,
        n_layers: int = 1,
        activation: th.nn.Module = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        dropout: float = 0.0,
        conditional_layer_norm: Callable = None,
        conditional_layer_norm_once: bool = False,
    ):
        """
        Parameters
        ----------
        n_layers: int, optional
            The number of SymmetricConvNeXt Blocks
        conditional_layer_norm: Callable, optional
            Callable for physicsnemo.models.dlwp_healpix_layers.normalization.ConditionalLayerNorm.
            Callable can be passed in by setting _partial_ to True in hydra config. If None,
            conditional layer normalization is not applied.
        conditional_layer_norm_once: bool, optional
            Whether or not to apply conditional layer normalization only once. If True,
            the conditional layer normalization is applied only once, otherwise it is applied
            for each block.
        """
        super().__init__()

        # Create a ModuleList to store complete blocks
        self.blocks = th.nn.ModuleList()
        # flag for conditional layer normalization
        self.cln_enabled = conditional_layer_norm is not None

        for i in range(n_layers):
            curr_in = in_channels if i == 0 else out_channels
            self.blocks.append(
                SymmetricConvNeXtBlock(
                    geometry_layer=geometry_layer,
                    in_channels=curr_in,
                    latent_channels=latent_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    upscale_factor=upscale_factor,
                    activation=activation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    dropout=dropout,
                    conditional_layer_norm=conditional_layer_norm
                    if conditional_layer_norm is not None
                    else None,
                    conditional_layer_norm_once=conditional_layer_norm_once,
                ),
            )

    def forward(self, x, conditions_cln=None):
        out = x
        for block in self.blocks:
            out = block(out, conditions_cln=conditions_cln)
        return out


class SymmetricConvNeXtBlock(th.nn.Module):
    """Another modification of ConvNeXtBlock block this time using 4 layers and adding
    a layer that instead of going from in_channels to latent*upscale channesl goes to
    latent channels first
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,  # not used, but required for hydra instantiation
        upscale_factor: int = 4,
        activation: th.nn.Module = None,
        enable_nhwc: bool = False,
        use_block_skip_connection: bool = True,
        enable_healpixpad: bool = False,
        dropout: float = 0.0,
        conditional_layer_norm: th.nn.Module = None,
        conditional_layer_norm_once: bool = False,
    ):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        latent_channels: int, optional
            Number of latent channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutioonal kernels
        dilation: int, optional
            Spacing between kernel points, passed to torch.nn.Conv2d
        upscale_factor: int, optional
            Upscale factor to apply on the number of latent channels
        activation: torch.nn.Module, optional
            Activation function to use between layers
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        use_block_skip_connection: bool, optional
            Whether or not to use block-level skip connection
        dropout: float, optional
            Dropout probability to apply after the first convolution
        conditional_layer_norm: th.nn.Module, optional
            conditional layer normalization. If None,
            no conditional layer normalization is applied.
        conditional_layer_norm_once: bool, optional
            Whether or not to apply conditional layer normalization only once. If True,
            the conditional layer normalization is applied only once, otherwise it is applied
            for each block.
        """

        super().__init__()

        self.use_block_skip_connection = use_block_skip_connection
        self.activation = activation
        self.dropout = dropout > 0.0
        self.cln_enabled = conditional_layer_norm is not None

        if use_block_skip_connection:
            if in_channels == int(out_channels):
                self.skip_module = lambda x: x
            else:
                self.skip_module = geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                )

        # check if we're applying a layer norm at the beginning
        # TODO: conditional layer norm once is doing two things, it's applying a norm on block entry
        # and switch from conditional to non-conditional layer norm. This is not ideal and should be fixed once we determine
        # what works best.
        if conditional_layer_norm_once:
            if conditional_layer_norm is not None:
                # Conditional norm at the beginning of the block
                self.entry_norm = conditional_layer_norm(channel_depth=in_channels)
            else:
                # Regular layer normalization at the beginning of the block
                self.entry_norm = _LayerNormOverChannels(channel_depth=in_channels)
        else:
            # No normalization at the beginning
            self.entry_norm = None

        # Collect conv->norm->activation->dropout operations in list for sequential execution
        convblock = []

        # 3x3: in → latent
        convblock.append(
            geometry_layer(
                layer=th.nn.Conv2d,
                in_channels=in_channels,
                out_channels=int(latent_channels),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        # Apply layer norm if needed
        if conditional_layer_norm is not None and not conditional_layer_norm_once:
            cln = conditional_layer_norm(channel_depth=int(latent_channels))
            convblock.append(cln)

        if activation is not None:
            convblock.append(activation)
        if dropout > 0.0:
            convblock.append(th.nn.Dropout2d(p=dropout))

        # 1x1: latent → latent * upscale
        convblock.append(
            geometry_layer(
                layer=th.nn.Conv2d,
                in_channels=int(latent_channels),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=1,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )

        # Apply layer norm if needed
        if conditional_layer_norm_once:
            convblock.append(
                _LayerNormOverChannels(
                    channel_depth=int(latent_channels * upscale_factor)
                )
            )
        elif conditional_layer_norm is not None:
            cln = conditional_layer_norm(
                channel_depth=int(latent_channels * upscale_factor)
            )
            convblock.append(cln)

        if activation is not None:
            convblock.append(activation)
        if dropout > 0.0:
            convblock.append(th.nn.Dropout2d(p=dropout))

        # 1x1: upscale → latent
        convblock.append(
            geometry_layer(
                layer=th.nn.Conv2d,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=int(latent_channels),
                kernel_size=1,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )

        # Apply layer norm if needed
        if conditional_layer_norm_once:
            convblock.append(_LayerNormOverChannels(channel_depth=int(latent_channels)))
        elif conditional_layer_norm is not None:
            cln = conditional_layer_norm(channel_depth=int(latent_channels))
            convblock.append(cln)

        if activation is not None:
            convblock.append(activation)
        if dropout > 0.0:
            convblock.append(th.nn.Dropout2d(p=dropout))

        # 3x3: latent → out (no norm on this one, following convnext)
        convblock.append(
            geometry_layer(
                layer=th.nn.Conv2d,
                in_channels=int(latent_channels),
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock.append(activation)
        if dropout > 0.0:
            convblock.append(th.nn.Dropout2d(p=dropout))

        self.convblock = th.nn.ModuleList(convblock)

    def forward(self, x, conditions_cln=None):
        """
        Forward pass of the SymmetricConvNextBlock, broken into steps for support of conditional layer normalization
        Parameters
        ----------
        x: torch.Tensor
            inputs to the forward pass
        conditions_cln: torch.Tensor, optional
            Condition for the conditional layer normalization, if applicable
        Returns
        -------
        torch.Tensor
            result of the forward pass
        """

        # TODO: performance of skip connectioni hasn't been compared
        # check  cln(x) vs. cln(x1_residual + x) in the future
        # Save residual
        residual = self.skip_module(x) if self.use_block_skip_connection else 0

        if self.entry_norm is not None:
            if conditions_cln is not None:
                x = self.entry_norm(x, conditions=conditions_cln)
            else:
                x = self.entry_norm(x)

        for layer in self.convblock:
            if isinstance(layer, ConditionalLayerNorm):
                x = layer(x, conditions=conditions_cln)
            else:
                x = layer(x)

        return x + residual


#
# UPSAMPLING BLOCKS
#


class TransposedConvUpsample(th.nn.Module):
    """This class provides a wrapper for a HEALPix (or other) tensor data
    around the torch.nn.ConvTranspose2d class.
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        out_channels: int = 1,
        upsampling: int = 2,
        activation: th.nn.Module = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry of the tensor being bassed to MaxPool2d
        in_channels: int, optional
            The number of input channels
        out_channels: int, optional
            The number of output channels
        upsampling: int, optional
            Size used for upsampling
        activation: torch.nn.Module, optional
            Activation function used in upsampling
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        upsampler = []
        # Upsample transpose conv
        upsampler.append(
            geometry_layer(
                layer=torch.nn.ConvTranspose2d,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=upsampling,
                stride=upsampling,
                padding=0,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            upsampler.append(activation)
        self.upsampler = th.nn.Sequential(*upsampler)

    def forward(self, x):
        """Forward pass of the TransposedConvUpsample layer

        Parameters
        ----------
        x: torch.Tensor
            The values to upsample

        Returns
        -------
        torch.Tensor
            The upsampled values
        """
        return self.upsampler(x)


#
# Helper classes
#


class Interpolate(th.nn.Module):
    """Helper class that handles interpolation
    This is done as a class so that scale and mode can be stored
    """

    def __init__(self, scale_factor: Union[int, Tuple], mode: str = "nearest"):
        """
        Parameters:
        ----------
        scale_factor: Union[int , Tuple]
            Multiplier for spatial size, passed to torch.nn.functional.interpolate
        mode: str, optional
            Interpolation mode used for upsampling, passed to torch.nn.functional.interpolate
        """
        super().__init__()
        self.interp = th.nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, inputs):
        """Forward pass of the Interpolate layer

        Parameters
        ----------
        x: torch.Tensor
            inputs to interpolate

        Returns
        -------
        torch.Tensor
            the interpolated values
        """
        return self.interp(inputs, scale_factor=self.scale_factor, mode=self.mode)
