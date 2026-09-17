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
Implementation of the Deep Learning Weather Prediction (DLWP) decoder on the HEALPix mesh.

This class contains the implementation of the Deep Learning Weather Prediction (DLWP) decoder on the HEALPix mesh.
"""

from typing import Sequence

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint


class UNetDecoder(torch.nn.Module):
    """Generic UNetDecoder that can be applied to arbitrary meshes."""

    def __init__(
        self,
        conv_block: DictConfig,
        up_sampling_block: DictConfig,
        output_layer: DictConfig,
        recurrent_block: DictConfig = None,
        n_channels: Sequence = (64, 32, 16),
        n_layers: Sequence = (1, 2, 2),
        output_channels: int = 1,
        dilations: list = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        per_level_cln: list[bool] = None,
        per_level_checkpointing: list[bool] = None,
    ):
        """
        Parameters
        ----------
        conv_block: DictConfig
            dictionary of instantiable parameters for the convolutional block
        up_sampling_blockoder: DictConfig
            dictionary of instantiable parameters for the upsampling block
        output_layer: DictConfig
            dictionary of instantiable parameters for the output layer
        recurrent_block: DictConfig, optional
            dictionary of instantiable parameters for the recurrent block
            recurrent blocks are not used if this is None
        n_channels: Sequence, optional
            The number of channels in each decoder layer
        n_layers:, Sequence, optional
            Number of layers to use for the convolutional blocks
        output_channels: int, optional
            Number of output channels
        dilations: list, optional
            List of dialtions to use for the the convolutional blocks
        enable_nhwc: bool, optional
            If channel last format should be used
        enable_healpixpad, bool, optional
            If the healpixpad library should be used if installed
        per_level_cln: list[bool], optional
            If the CLN should be applied to each level of the decoder
            If None, the CLN will based on the conv_block.conditional_layer_norm attribute
        per_level_checkpointing: list[bool], optional
            If the checkpointing should be applied to each level of the decoder
            If None, the checkpointing will not be applied

        per level options are lists of booleans of the same length as n_channels,
        if only one value is provided, it will be applied to all levels. The level layout is a
        mirror of the encoder. The first value in the list will be applied to the lowest level
        in the decoder and the last value in the list will be applied to the highest level in the decoder.
        Example:
        n_channels = [16, 32, 64]
        per_level_cln = [True, False, False] will apply CLN to the lowest level (16 channels) and not the second and third levels.
        Not this is the opposite of the encoder.
        """
        super().__init__()
        self.channel_dim = 1

        if per_level_cln is not None and len(per_level_cln) != len(n_channels):
            raise ValueError(
                "per_level_cln must be a list of booleans of the same length as n_channels"
                f"Got {len(per_level_cln)} for per_level_cln and {len(n_channels)} for n_channels"
            )
        per_level_cln = (
            per_level_cln if per_level_cln is not None else [True] * len(n_channels)
        )

        if per_level_checkpointing is not None and len(per_level_checkpointing) != len(
            n_channels
        ):
            raise ValueError(
                "per_level_checkpointing must be a list of booleans of the same length as n_channels"
                f"Got {len(per_level_checkpointing)} for per_level_checkpointing and {len(n_channels)} for n_channels"
            )
        self.per_level_checkpointing = (
            per_level_checkpointing
            if per_level_checkpointing is not None
            else [False] * len(n_channels)
        )

        if dilations is None:
            dilations = [1 for _ in range(len(n_channels))]

        self.decoder = []
        for n, curr_channel in enumerate(n_channels):
            if n == 0:
                up_sample_module = None
            else:
                up_sample_module = instantiate(
                    config=up_sampling_block,
                    in_channels=curr_channel,
                    out_channels=curr_channel,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                )

            next_channel = (
                n_channels[n + 1] if n < len(n_channels) - 1 else n_channels[-1]
            )

            block_config = conv_block.copy()
            if (
                "conditional_layer_norm" in block_config
                and block_config.conditional_layer_norm is not None
            ):
                if not per_level_cln[n]:
                    block_config.conditional_layer_norm = None

            conv_module = instantiate(
                config=block_config,
                in_channels=curr_channel * 2 if n > 0 else curr_channel,
                latent_channels=curr_channel,
                out_channels=next_channel,
                dilation=dilations[n],
                n_layers=n_layers[n],
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )

            if recurrent_block is not None:
                rec_module = instantiate(
                    config=recurrent_block,
                    in_channels=next_channel,
                    enable_healpixpad=enable_healpixpad,
                )
            else:
                rec_module = None

            self.decoder.append(
                torch.nn.ModuleDict(
                    {
                        "upsamp": up_sample_module,
                        "conv": conv_module,
                        "recurrent": rec_module,
                    }
                )
            )

        self.decoder = torch.nn.ModuleList(self.decoder)
        self.output_layer = instantiate(
            config=output_layer,
            in_channels=curr_channel,
            out_channels=output_channels,
            dilation=dilations[-1],
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
        )

    def _forward_layer_pass(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        skip_connection: torch.Tensor = None,
        conditions_cln: torch.Tensor = None,
    ) -> torch.Tensor:
        """Helper function that performs the forward pass of a single layer of the decoder.

        Parameters
        ----------
        layer: torch.nn.Module
            The layer to forward pass
        x: torch.Tensor
            The input tensor
        skip_connection: torch.Tensor, optional
            The skip connection tensor
        conditions_cln: torch.Tensor, optional
            The conditional inputs for the normalization layers.

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        if layer["upsamp"] is not None:
            up = layer["upsamp"](x)
            x = torch.cat([up, skip_connection], dim=self.channel_dim)
        if hasattr(layer["conv"], "cln_enabled") and layer["conv"].cln_enabled:
            if conditions_cln is not None:
                x = layer["conv"](x, conditions_cln=conditions_cln)
            else:
                raise ValueError(
                    "Conditional inputs are required for layers with cln_enabled=True"
                )
        else:
            x = layer["conv"](x)

        return x

    def forward(
        self, inputs: Sequence, conditions_cln: Sequence = None
    ) -> torch.Tensor:
        """
        Forward pass of the HEALPix Unet decoder

        Parameters
        ----------
        inputs: Sequence
            The inputs to decode
        conditions_cln: Sequence, optional
            The conditional inputs for the normalization layers.

        Returns
        -------
        torch.Tensor: The decoded values
        """
        x = inputs[-1]
        for n, layer in enumerate(self.decoder):
            skip_connection = inputs[-1 - n] if layer["upsamp"] is not None else None
            if self.per_level_checkpointing[n]:
                x = checkpoint(
                    self._forward_layer_pass,
                    layer,
                    x,
                    skip_connection,
                    conditions_cln,
                    use_reentrant=False,
                )
            else:
                x = self._forward_layer_pass(layer, x, skip_connection, conditions_cln)

            if layer["recurrent"] is not None:
                x = layer["recurrent"](x)

        return self.output_layer(x)

    def reset(self):
        """Resets the state of the decoder layers"""
        for layer in self.decoder:
            if layer["recurrent"] is not None:
                layer["recurrent"].reset()
