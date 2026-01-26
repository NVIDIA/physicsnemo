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

import logging
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn import HEALPixFoldFaces, HEALPixUnfoldFaces

from .layers import _legacy_hydra_targets_warning, _remap_obj

logger = logging.getLogger(__name__)


@dataclass
class MetaData(ModelMetaData):
    r"""Metadata for the DLWP HEALPix UNet model."""

    # Optimization
    jit: bool = False
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class HEALPixUNet(Module):
    r"""Deep Learning Weather Prediction (DLWP) UNet on the HEALPix mesh.

    Parameters
    ----------
    encoder : DictConfig
        Instantiable configuration for the U-Net encoder block.
    decoder : DictConfig
        Instantiable configuration for the U-Net decoder block.
    input_channels : int
        Number of prognostic input channels per time step.
    output_channels : int
        Number of prognostic output channels per time step.
    n_constants : int
        Number of constant input channels provided for all faces.
    decoder_input_channels : int
        Number of prescribed decoder inputs per time step.
    input_time_dim : int
        Number of input time steps :math:`T_{in}`.
    output_time_dim : int
        Number of output time steps :math:`T_{out}`.
    presteps : int, optional
        Number of warm-up steps used to initialize recurrent states, by default ``0``.
    enable_nhwc : bool, optional
        If ``True``, use channels-last tensors.
    enable_healpixpad : bool, optional
        Enable CUDA HEALPix padding when the optional dependency is installed.
    couplings : list, optional
        Optional coupling specifications appended to the input feature channels.

    Forward
    -------
    inputs : Sequence[torch.Tensor]
        Input tensors shaped :math:`(B, F, T_{in}, C_{in}, H, W)` plus optional
        decoder inputs and constants. When ``couplings`` is provided an additional
        coupling tensor shaped :math:`(T_{in}, B, C_c, H, W)` is expected.
    output_only_last : bool, optional
        If ``True``, return only the last forecast step; otherwise return the full
        integration horizon.

    Outputs
    -------
    torch.Tensor
        Predictions shaped :math:`(B, F, T_{out}, C_{out}, H, W)`.

    """

    __model_checkpoint_version__ = "0.2.0"
    __supported_model_checkpoint_version__ = {
        "0.1.0": _legacy_hydra_targets_warning,
    }

    @classmethod
    def _backward_compat_arg_mapper(
        cls, version: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        r"""
        Map arguments from older checkpoints to the current format.

        Parameters
        ----------
        version : str
            Version of the checkpoint being loaded.
        args : Dict[str, Any]
            Arguments dictionary from the checkpoint.

        Returns
        -------
        Dict[str, Any]
            Updated arguments dictionary compatible with the current version.
        """
        args = super()._backward_compat_arg_mapper(version, args)
        if version != "0.1.0":
            return args

        return _remap_obj(args)

    def __init__(
        self,
        encoder: DictConfig,
        decoder: DictConfig,
        input_channels: int,
        output_channels: int,
        n_constants: int,
        decoder_input_channels: int,
        input_time_dim: int,
        output_time_dim: int,
        presteps: int = 0,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        couplings: list = [],
    ):
        r"""Initialize the DLWP HEALPix UNet."""
        super().__init__(meta=MetaData())

        if len(couplings) > 0:
            if n_constants == 0:
                raise NotImplementedError(
                    "support for coupled models with no constant fields is not available at this time."
                )
            if decoder_input_channels == 0:
                raise NotImplementedError(
                    "support for coupled models with no decoder inputs (TOA insolation) is not available at this time."
                )

        # add coupled fields to input channels for model initialization
        self.coupled_channels = self._compute_coupled_channels(couplings)
        self.couplings = couplings
        self.train_couplers = None
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.channel_dim = 2  # Now 2 with [B, F, C*T, H, W]. Was 1 in old data format with [B, T*C, F, H, W]
        self.enable_nhwc = enable_nhwc
        self.enable_healpixpad = enable_healpixpad

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(
                f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                f"{self.output_time_dim} and {self.input_time_dim})"
            )

        # Build the model layers
        self.fold = HEALPixFoldFaces()
        self.unfold = HEALPixUnfoldFaces(num_faces=12)
        self.encoder = instantiate(
            config=encoder,
            input_channels=self._compute_input_channels(),
            enable_nhwc=self.enable_nhwc,
            enable_healpixpad=self.enable_healpixpad,
        )
        self.encoder_depth = len(self.encoder.n_channels)
        self.decoder = instantiate(
            config=decoder,
            output_channels=self._compute_output_channels(),
            enable_nhwc=self.enable_nhwc,
            enable_healpixpad=self.enable_healpixpad,
        )

    @property
    def integration_steps(self):
        r"""
        Number of implicit forward integration steps.

        Returns
        -------
        int
            Integration horizon :math:`T_{out} / T_{in}` (minimum 1).
        """
        return max(self.output_time_dim // self.input_time_dim, 1)

    def _compute_input_channels(self) -> int:
        r"""
        Calculate total number of input channels consumed by the encoder.

        Returns
        -------
        int
            Total channel count including couplings and constants.
        """
        return (
            self.input_time_dim * (self.input_channels + self.decoder_input_channels)
            + self.n_constants
            + self.coupled_channels
        )

    def _compute_coupled_channels(self, couplings):
        r"""
        Compute the number of additional coupling channels.

        Parameters
        ----------
        couplings : list
            Coupling configuration dictionaries.

        Returns
        -------
        int
            Total coupling channels contributed to the encoder inputs.
        """
        return sum(
            len(c["params"]["variables"]) * len(c["params"]["input_times"])
            for c in couplings
        )

    def _compute_output_channels(self) -> int:
        r"""
        Compute the total number of output channels in the model.

        Returns
        -------
        int
            Output channel count for each integration step.
        """
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels

    def _reshape_inputs(self, inputs: Sequence, step: int = 0) -> torch.Tensor:
        r"""
        Concatenate prognostic, decoder, constant, and coupling inputs for the encoder.

        Parameters
        ----------
        inputs : Sequence
            Tensors arranged as ``[prognostics, decoder_inputs, constants]`` with
            optional couplings.
        step : int, optional
            Integration step index.

        Returns
        -------
        torch.Tensor
            Folded encoder input shaped :math:`(B \cdot F, C, H, W)`.
        """

        if len(self.couplings) > 0:
            result = [
                inputs[0].flatten(
                    start_dim=self.channel_dim, end_dim=self.channel_dim + 1
                ),
                inputs[1][
                    :,
                    :,
                    slice(step * self.input_time_dim, (step + 1) * self.input_time_dim),
                    ...,
                ].flatten(
                    start_dim=self.channel_dim, end_dim=self.channel_dim + 1
                ),  # DI
                inputs[2].expand(
                    *tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1])
                ),  # constants
                inputs[3].permute(0, 2, 1, 3, 4),  # coupled inputs
            ]
            res = torch.cat(result, dim=self.channel_dim)

        else:
            if not (self.n_constants > 0 or self.decoder_input_channels > 0):
                res = inputs[0].flatten(
                    start_dim=self.channel_dim, end_dim=self.channel_dim + 1
                )
                return self.fold(res)
            if self.n_constants == 0:
                result = [
                    inputs[0].flatten(
                        start_dim=self.channel_dim, end_dim=self.channel_dim + 1
                    ),  # inputs
                    inputs[1][
                        :,
                        :,
                        slice(
                            step * self.input_time_dim, (step + 1) * self.input_time_dim
                        ),
                        ...,
                    ].flatten(self.channel_dim, self.channel_dim + 1),  # DI
                ]
                res = torch.cat(result, dim=self.channel_dim)

                # fold faces into batch dim
                res = self.fold(res)

                return res
            if self.decoder_input_channels == 0:
                result = [
                    inputs[0].flatten(
                        start_dim=self.channel_dim, end_dim=self.channel_dim + 1
                    ),  # inputs
                    inputs[1].expand(
                        *tuple([inputs[0].shape[0]] + len(inputs[1].shape) * [-1])
                    ),  # constants
                ]
                res = torch.cat(result, dim=self.channel_dim)

                # fold faces into batch dim
                res = self.fold(res)

                return res

            result = [
                inputs[0].flatten(
                    start_dim=self.channel_dim, end_dim=self.channel_dim + 1
                ),  # inputs
                inputs[1][
                    :,
                    :,
                    slice(step * self.input_time_dim, (step + 1) * self.input_time_dim),
                    ...,
                ].flatten(self.channel_dim, self.channel_dim + 1),  # DI
                inputs[2].expand(
                    *tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1])
                ),  # constants
            ]
            res = torch.cat(result, dim=self.channel_dim)

        # fold faces into batch dim
        res = self.fold(res)

        return res

    def _reshape_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        r"""
        Reshape decoder output back to explicit time and channel dimensions.

        Parameters
        ----------
        outputs : torch.Tensor
            Decoder output shaped :math:`(B \cdot F, C, H, W)`.

        Returns
        -------
        torch.Tensor
            Unfolded tensor shaped :math:`(B, F, T_{out}, C_{out}, H, W)`.
        """

        # unfold:
        outputs = self.unfold(outputs)

        # extract shape and reshape
        shape = tuple(outputs.shape)
        res = torch.reshape(
            outputs,
            shape=(
                shape[0],
                shape[1],
                1 if self.is_diagnostic else self.input_time_dim,
                -1,
                *shape[3:],
            ),
        )

        return res

    def forward(self, inputs: Sequence, output_only_last: bool = False) -> torch.Tensor:
        r"""
        Forward pass of the HEALPix UNet.

        Parameters
        ----------
        inputs : Sequence
            List ``[prognostics, decoder_inputs, constants]`` or with couplings
            ``[prognostics, decoder_inputs, constants, couplings]`` where
            prognostics and decoder inputs have shape :math:`(B, F, T, C, H, W)` and
            constants have shape :math:`(F, C, H, W)`.
        output_only_last : bool, optional
            If ``True``, return only the final forecast step. Defaults to ``False``.

        Returns
        -------
        torch.Tensor
            Model outputs shaped :math:`(B, F, T_{out}, C_{out}, H, W)`.
        """
        if not torch.compiler.is_compiling():
            if inputs[0].ndim != 6:
                raise ValueError(
                    "HEALPixUNet.forward expects prognostics shaped (B, F, T, C, H, W)"
                )

        outputs = []
        for step in range(self.integration_steps):
            if step == 0:
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        list(inputs[0:3]) + [inputs[3][step]], step
                    )
                else:
                    input_tensor = self._reshape_inputs(inputs, step)
            else:
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        [outputs[-1]] + list(inputs[1:3]) + [inputs[3][step]], step
                    )
                else:
                    input_tensor = self._reshape_inputs(
                        [outputs[-1]] + list(inputs[1:]), step
                    )
            encodings = self.encoder(input_tensor)
            decodings = self.decoder(encodings)

            reshaped = self._reshape_outputs(decodings)  # Absolute prediction
            outputs.append(reshaped)

        if output_only_last:
            res = outputs[-1]
        else:
            res = torch.cat(outputs, dim=self.channel_dim)

        return res
