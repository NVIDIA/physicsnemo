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
Implementation of the Deep Learning Weather Prediction (DLWP) encoder on the HEALPix mesh.

This class contains the implementation of the Deep Learning Weather Prediction (DLWP) encoder on the HEALPix mesh.
"""

from typing import Sequence

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint


class UNetEncoder(torch.nn.Module):
    """Generic UNetEncoder that can be applied to arbitrary meshes."""

    def __init__(
        self,
        conv_block: DictConfig,
        down_sampling_block: DictConfig,
        recurrent_block: DictConfig = None,
        input_channels: int = 3,
        n_channels: Sequence = (16, 32, 64),
        n_layers: Sequence = (2, 2, 1),
        dilations: list = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        per_level_cln: Sequence[bool] = None,
        per_level_checkpointing: Sequence[bool] = None,
    ):
        """
        Parameters
        ----------
        conv_block: DictConfig
            dictionary of instantiable parameters for the convolutional block
        down_sampling_block: DictConfig
            dictionary of instantiable parameters for the downsample block
        recurrent_block: DictConfig, optional
            dictionary of instantiable parameters for the recurrent block
            recurrent blocks are not used if this is None
        input_channels: int, optional
            Number of input channels
        n_channels: Sequence, optional
            The number of channels in each encoder layer
        n_layers:, Sequence, optional
            Number of layers to use for the convolutional blocks
        dilations: list, optional
            List of dialtions to use for the the convolutional blocks
        enable_nhwc: bool, optional
            If channel last format should be used
        enable_healpixpad, bool, optional
            If the healpixpad library should be used (if installed)
        per_level_cln: list[bool] | None, optional
            If the CLN should be applied to each level of the encoder
            If None, the CLN will based on the conv_block.conditional_layer_norm attribute
        per_level_checkpointing: list[bool] | None, optional
            If the checkpointing should be applied to each level of the encoder
            If None, the checkpointing will not be applied

        per level options are lists of booleans of the same length as n_channels,
        if only one value is provided, it will be applied to all levels. The first
        level is the highest (top) level in the unet encoder and the last level is the lowest (bottom) level.
        The value in the list will be applied to the top level, the second value to the next level down, etc.
        Example:
        n_channels = [16, 32, 64]
        per_level_cln = [True, False, False] will apply CLN to the highest level (16 channels) and not the second and third levels.

        """
        super().__init__()
        self.n_channels = n_channels

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

        old_channels = input_channels
        self.encoder = []
        for n, curr_channel in enumerate(n_channels):
            modules = list()
            if n > 0:
                modules.append(
                    instantiate(
                        config=down_sampling_block,
                        enable_nhwc=enable_nhwc,
                        enable_healpixpad=enable_healpixpad,
                    )
                )

            block_config = conv_block.copy()
            if (
                "conditional_layer_norm" in block_config
                and block_config.conditional_layer_norm is not None
            ):
                if not per_level_cln[n]:
                    block_config.conditional_layer_norm = None

            modules.append(
                instantiate(
                    config=block_config,
                    in_channels=old_channels,
                    latent_channels=curr_channel,
                    out_channels=curr_channel,
                    dilation=dilations[n],
                    n_layers=n_layers[n],
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                )
            )
            old_channels = curr_channel

            self.encoder.append(torch.nn.Sequential(*modules))

        self.encoder = torch.nn.ModuleList(self.encoder)

    def _forward_layer_pass(
        self,
        layer_group: torch.nn.Module,
        inp: torch.Tensor,
        conditions_cln: torch.Tensor = None,
    ) -> torch.Tensor:
        """Helper function that performs the forward pass of a single layer of the encoder.

        Parameters
        ----------
        layer_group: torch.nn.Module
            The layer group to forward pass
        inp: torch.Tensor
            The input tensor
        conditions_cln: torch.Tensor, optional
            The conditional inputs for the normalization layers.

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        interim_output = inp
        for layer in layer_group:
            if getattr(layer, "cln_enabled", False):
                if conditions_cln is None:
                    raise ValueError(
                        "Conditional inputs are required for layers with cln_enabled=True"
                    )
                interim_output = layer(interim_output, conditions_cln=conditions_cln)
            else:
                interim_output = layer(interim_output)

        return interim_output

    def forward(
        self, inputs: Sequence, conditions_cln: torch.Tensor = None
    ) -> Sequence:
        """
        Forward pass of the HEALPix Unet encoder

        Parameters
        ----------
        inputs: Sequence
            The inputs to encode
        conditions_cln: torch.Tensor, optional
            The conditional inputs for the normalization layers.

        Returns
        -------
        Sequence: The encoded values
        """
        outputs = []
        for n, layer_group in enumerate(self.encoder):
            interim_output = inputs
            if self.per_level_checkpointing[n]:
                interim_output = checkpoint(
                    self._forward_layer_pass,
                    layer_group,
                    interim_output,
                    conditions_cln,
                    use_reentrant=False,
                )
            else:
                interim_output = self._forward_layer_pass(
                    layer_group, interim_output, conditions_cln
                )
            outputs.append(interim_output)
            inputs = outputs[-1]
        return outputs

    def reset(self):
        """Resets the state of the decoder layers"""
        pass
