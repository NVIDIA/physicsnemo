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

import logging
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import torch as th
from hydra.utils import instantiate
from omegaconf import DictConfig

from physicsnemo.models.dlwp_healpix_layers import HEALPixFoldFaces, HEALPixUnfoldFaces
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

logger = logging.getLogger(__name__)


@dataclass
class MetaData(ModelMetaData):
    """Metadata for the DLWP HEALPix Model"""

    name: str = "DLWP_HEALPixRec"
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


class HEALPixRecUNet(Module):
    """Deep Learning Weather Prediction (DLWP) recurrent UNet model on the HEALPix mesh."""

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
        delta_time: str = "6h",
        reset_cycle: str = "24h",
        presteps: int = 1,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        couplings: list = [],
    ):
        """
        Parameters
        ----------
        encoder: DictConfig
            dictionary of instantiable parameters for the U-net encoder
        decoder: DictConfig
            dictionary of instantiable parameters for the U-net decoder
        input_channels: int
            number of input channels expected in the input array schema. Note this should be the
            number of input variables in the data, NOT including data reshaping for the encoder part.
        output_channels: int
            number of output channels expected in the output array schema, or output variables
        n_constants: int
            number of optional constants expected in the input arrays. If this is zero, no constants
            should be provided as inputs to `forward`.
        decoder_input_channels: int
            number of optional prescribed variables expected in the decoder input array
            for both inputs and outputs. If this is zero, no decoder inputs should be provided as inputs to `forward`.
        input_time_dim: int
            number of time steps in the input array
        output_time_dim: int
            number of time steps in the output array
        delta_time: str, optional
            hours between two consecutive data points
        reset_cycle: str, optional
            hours after which the recurrent states are reset to zero and re-initialized. Set np.infty
            to never reset the hidden states.
        presteps: int, optional
            number of model steps to initialize recurrent states.
        enable_nhwc: bool, optional
            Model with [N, H, W, C] instead of [N, C, H, W]
        enable_healpixpad: bool, optional
            Enable CUDA HEALPixPadding if installed
        couplings: list, optional
            sequence of dictionaries that describe coupling mechanisms
        """
        super().__init__()
        self.channel_dim = 2  # Now 2 with [B, F, T*C, H, W]. Was 1 in old data format with [B, T*C, F, H, W]

        self.input_channels = input_channels

        if n_constants == 0 and decoder_input_channels == 0:
            raise NotImplementedError(
                "support for models with no constant fields and no decoder inputs (TOA insolation) is not available at this time."
            )
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
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.delta_t = int(pd.Timedelta(delta_time).total_seconds() // 3600)
        self.reset_cycle = int(pd.Timedelta(reset_cycle).total_seconds() // 3600)
        self.presteps = presteps
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
        """Number of integration steps"""
        return max(self.output_time_dim // self.input_time_dim, 1)

    def _compute_input_channels(self) -> int:
        """Calculate total number of input channels in the model"""
        return (
            self.input_time_dim * (self.input_channels + self.decoder_input_channels)
            + self.n_constants
            + self.coupled_channels
        )

    def _compute_coupled_channels(self, couplings):
        """Get number of coupled channels

        Returns
        -------
        int
            The number of coupled channels
        """
        c_channels = 0
        for c in couplings:
            c_channels += len(c["params"]["variables"]) * len(
                c["params"]["input_times"]
            )
        return c_channels

    def _compute_output_channels(self) -> int:
        """Compute the total number of output channels in the model"""
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels

    def _reshape_inputs(self, inputs: Sequence, step: int = 0) -> th.Tensor:
        """
        Returns a single tensor to pass into the model encoder/decoder. Squashes the time/channel dimension and
        concatenates in constants and decoder inputs.

        Parameters
        ----------
        inputs: Sequence
            list of expected input tensors (inputs, decoder_inputs, constants)
        step: int, optional
            step number in the sequence of integration_steps

        Returns
        -------
        torch.Tensor: reshaped Tensor in expected shape for model encoder
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
            res = th.cat(result, dim=self.channel_dim)

        else:
            if self.n_constants == 0:
                result = [
                    inputs[0].flatten(
                        start_dim=self.channel_dim, end_dim=self.channel_dim + 1
                    ),
                    inputs[1][
                        :,
                        :,
                        slice(
                            step * self.input_time_dim, (step + 1) * self.input_time_dim
                        ),
                        ...,
                    ].flatten(
                        start_dim=self.channel_dim, end_dim=self.channel_dim + 1
                    ),  # DI
                ]
                res = th.cat(result, dim=self.channel_dim)

                # fold faces into batch dim
                res = self.fold(res)

                return res

            if self.decoder_input_channels == 0:
                result = [
                    inputs[0].flatten(
                        start_dim=self.channel_dim, end_dim=self.channel_dim + 1
                    ),
                    inputs[1].expand(
                        *tuple([inputs[0].shape[0]] + len(inputs[1].shape) * [-1])
                    ),  # constants
                ]
                res = th.cat(result, dim=self.channel_dim)

                # fold faces into batch dim
                res = self.fold(res)

                return res

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
            ]
            res = th.cat(result, dim=self.channel_dim)

        # fold faces into batch dim
        res = self.fold(res)
        return res

    def _reshape_outputs(self, outputs: th.Tensor) -> th.Tensor:
        """Returns a maultiple tensors to from the model decoder.
        Splits the time/channel dimensions.

        Parameters
        ----------
        inputs: Sequence
            list of expected input tensors (inputs, decoder_inputs, constants)
        step: int, optional
            step number in the sequence of integration_steps

        Returns
        -------
        torch.Tensor: reshaped Tensor in expected shape for model outputs
        """
        # unfold:
        outputs = self.unfold(outputs)

        # extract shape and reshape
        shape = tuple(outputs.shape)
        res = th.reshape(
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

    def _initialize_hidden(
        self, inputs: Sequence, outputs: Sequence, step: int
    ) -> None:
        """Initialize the hidden layers

        Parameters
        ----------
        inputs: Sequence
            Inputs to use to initialize the hideen layers
        outputs: Sequence
            Outputs to use to initialize the hideen layers
        step: int
            Current step number of the initialization
        """
        self.reset()
        for prestep in range(self.presteps):
            if step < self.presteps:
                s = step + prestep
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        inputs=[
                            inputs[0][
                                :,
                                :,
                                s * self.input_time_dim : (s + 1) * self.input_time_dim,
                            ]
                        ]
                        + list(inputs[1:3])
                        + [inputs[3][prestep]],
                        step=step + prestep,
                    )
                else:
                    input_tensor = self._reshape_inputs(
                        inputs=[
                            inputs[0][
                                :,
                                :,
                                s * self.input_time_dim : (s + 1) * self.input_time_dim,
                            ]
                        ]
                        + list(inputs[1:]),
                        step=step + prestep,
                    )
            else:
                s = step - self.presteps + prestep
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        inputs=[outputs[s - 1]]
                        + list(inputs[1:3])
                        + [inputs[3][step - (prestep - self.presteps)]],
                        step=s + 1,
                    )
                else:
                    input_tensor = self._reshape_inputs(
                        inputs=[outputs[s - 1]] + list(inputs[1:]), step=s + 1
                    )
            # Forward the data through the model to initialize hidden states
            self.decoder(self.encoder(input_tensor))

    def forward(self, inputs: Sequence, output_only_last=False) -> th.Tensor:
        """
        Forward pass of the HEALPixUnet

        Parameters
        ----------
        inputs: Sequence
            Inputs to the model, of the form [prognostics|TISR|constants]
            [B, F, T, C, H, W] is the format for prognostics and TISR
            [F, C, H, W] is the format for constants
        output_only_last: bool, optional
            If only the last dimension of the outputs should be returned

        Returns
        -------
        th.Tensor: Predicted outputs
        """
        self.reset()
        outputs = []
        for step in range(self.integration_steps):
            # (Re-)initialize recurrent hidden states
            if (step * (self.delta_t * self.input_time_dim)) % self.reset_cycle == 0:
                self._initialize_hidden(inputs=inputs, outputs=outputs, step=step)

            # Construct concatenated input: [prognostics|TISR|constants]
            if step == 0:
                s = self.presteps
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        inputs=[
                            inputs[0][
                                :,
                                :,
                                s * self.input_time_dim : (s + 1) * self.input_time_dim,
                            ]
                        ]
                        + list(inputs[1:3])
                        + [inputs[3][s]],
                        step=s,
                    )
                else:
                    input_tensor = self._reshape_inputs(
                        inputs=[
                            inputs[0][
                                :,
                                :,
                                s * self.input_time_dim : (s + 1) * self.input_time_dim,
                            ]
                        ]
                        + list(inputs[1:]),
                        step=s,
                    )
            else:
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        inputs=[outputs[-1]]
                        + list(inputs[1:3])
                        + [inputs[3][self.presteps + step]],
                        step=step + self.presteps,
                    )
                else:
                    input_tensor = self._reshape_inputs(
                        inputs=[outputs[-1]] + list(inputs[1:]),
                        step=step + self.presteps,
                    )

            # Forward through model
            encodings = self.encoder(input_tensor)
            decodings = self.decoder(encodings)

            # Residual prediction
            reshaped = self._reshape_outputs(
                input_tensor[:, : self.input_channels * self.input_time_dim] + decodings
            )
            outputs.append(reshaped)

        if output_only_last:
            return outputs[-1]

        return th.cat(outputs, dim=self.channel_dim)

    def reset(self):
        """Resets the state of the network"""
        self.encoder.reset()
        self.decoder.reset()
