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
Implementation of the Deep Learning Weather Prediction (DLWP) recurrent UNet on the HEALPix mesh.

This class provides the core functionality for the DLWP recurrent UNet on the HEALPix mesh.
It handles the forward pass of the model, the backward pass, and the initialization of the hidden states.
It also supports coupling the model with external inputs from various earth system components.

"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn.module.hpx import HEALPixFoldFaces, HEALPixUnfoldFaces

from .layers import (
    _backward_compat_dlesym_v1_args,
    _dlesym_v02_version_mismatch_warning,
    _legacy_hydra_targets_warning,
    _remap_obj,
)

logger = logging.getLogger(__name__)


@dataclass
class MetaData(ModelMetaData):
    r"""Metadata for the DLWP HEALPix recurrent model."""

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
    r"""
    Deep Learning Weather Prediction (DLWP) recurrent UNet on the HEALPix mesh.

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
        Number of constant channels provided for all faces.
    decoder_input_channels : int
        Number of prescribed decoder input channels per time step.
    input_time_dim : int
        Number of input time steps :math:`T_{in}`.
    output_time_dim : int
        Number of output time steps :math:`T_{out}`.
    delta_time : str, optional
        Time difference between samples, e.g., ``\"6h\"``. Defaults to ``\"6h\"``.
    reset_cycle : str, optional
        Period for recurrent state reset, e.g., ``\"24h\"``. Defaults to ``\"24h\"``.
    presteps : int, optional
        Number of warm-up steps used to initialize recurrent states.
    enable_nhwc : bool, optional
        If ``True``, use channels-last tensors.
    enable_healpixpad : bool, optional
        Enable CUDA HEALPix padding when available.
    couplings : list, optional
        Optional coupling specifications appended to the input feature channels.
    residual_prediction : bool, optional
        If ``True``, the model will predict the residual of the input and the output
        and add it to the output. If ``False``, the model will predict the output directly.
    couplings_time_first : bool, optional
        If ``True``, the couplings will be passed to the model in the time dimension first.
        If ``False``, the couplings will be passed to the model in the channel dimension first.
    constraints : list[DictConfig], optional
        Optional constraints to be applied to the model outputs.
    is_diagnostic : bool, optional
        If ``True``, the model runs in diagnostic mode: it performs a single
        forward step and produces exactly one output time
        (``output_time_dim`` must be ``1``). Defaults to ``False``.

    Forward
    -------
    inputs : Sequence[torch.Tensor]
        Inputs shaped :math:`(B, F, T_{in}, C_{in}, H, W)` plus decoder inputs,
        constants, and optional coupling tensors.
    output_only_last : bool, optional
        If ``True``, return only the final forecast step.

    Outputs
    -------
    torch.Tensor
        Predictions shaped :math:`(B, F, T_{out}, C_{out}, H, W)`.

    """

    __model_checkpoint_version__ = "0.3.0"
    __supported_model_checkpoint_version__ = {
        "0.1.0": _legacy_hydra_targets_warning,
        "0.2.0": _dlesym_v02_version_mismatch_warning,
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
        if version == "0.1.0":
            args = _remap_obj(args)
        if version in ("0.1.0", "0.2.0"):
            args = _backward_compat_dlesym_v1_args(args)
        return args

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
        residual_prediction: bool = True,
        couplings_time_first: bool = True,
        constraints: list[DictConfig] = None,
        is_diagnostic: bool = False,
    ):
        r"""Initialize the recurrent DLWP HEALPix UNet."""
        super().__init__(meta=MetaData())
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
        self.residual_prediction = residual_prediction
        self.couplings_time_first = couplings_time_first

        # A diagnostic model performs a single forward step and produces
        # exactly one output time.
        self.is_diagnostic = is_diagnostic
        if self.is_diagnostic and self.output_time_dim != 1:
            raise ValueError(
                "A diagnostic model (is_diagnostic=True) must have "
                f"output_time_dim == 1 (got {self.output_time_dim})."
            )

        # We can't have a diagnostic model that tries to predict a residual
        if self.residual_prediction and self.is_diagnostic:
            raise ValueError(
                "A diagnostic model cannot predict a residual. Please set "
                "residual_prediction to False when is_diagnostic is True."
            )

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

        self.constraints = None
        self.set_constraints(constraints)

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
        Calculate total number of input channels.

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
        Get the number of coupled channels.

        Parameters
        ----------
        couplings : list
            Coupling configuration dictionaries.

        Returns
        -------
        int
            The number of coupled channels.
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
                inputs[3].permute(0, 2, 1, 3, 4)
                if self.couplings_time_first
                else inputs[3],  # coupled inputs
            ]
            res = torch.cat(result, dim=self.channel_dim)

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
                res = torch.cat(result, dim=self.channel_dim)

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
                res = torch.cat(result, dim=self.channel_dim)

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
            res = torch.cat(result, dim=self.channel_dim)

        # fold faces into batch dim
        res = self.fold(res)
        if self.enable_nhwc:
            res = res.to(memory_format=torch.channels_last)
        return res

    def set_constraints(self, constraints: list[DictConfig] = None):
        r"""
        Set constraints (e.g., non-negative) to be applied to model outputs.

        Parameters
        ----------
        constraints : list[DictConfig], optional
            Hydra instantiable constraint configurations.
        """
        if constraints is not None:
            # Use a ModuleList (rather than a plain list) so the constraint
            # modules are part of the module tree: their buffers (e.g. the
            # non-persistent ``thresholds`` and ``var_indices`` in
            # ``NonnegativeConstraint``) are then correctly moved by
            # ``.to(device)`` along with the rest of the model.
            self.constraints = torch.nn.ModuleList(
                [instantiate(constraints[constraint]) for constraint in constraints]
            )

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

    def _initialize_hidden(
        self,
        inputs: Sequence,
        outputs: Sequence,
        step: int,
        conditions_cln: Sequence = None,
    ) -> None:
        r"""
        Initialize the recurrent hidden states.

        Parameters
        ----------
        inputs : Sequence
            Input tensors used for warm-up.
        outputs : Sequence
            Outputs accumulated so far.
        step : int
            Current integration step index.

        Returns
        -------
        None
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
                # Only the prognostic channels of the previous output are fed
                # back in; any extra (diagnostic) output channels are dropped.
                prognostics = outputs[s - 1][:, :, :, : self.input_channels]
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        inputs=[prognostics]
                        + list(inputs[1:3])
                        + [inputs[3][step - (prestep - self.presteps)]],
                        step=s + 1,
                    )
                else:
                    input_tensor = self._reshape_inputs(
                        inputs=[prognostics] + list(inputs[1:]),
                        step=s + 1,
                    )
            if conditions_cln is not None:
                self.decoder(
                    self.encoder(input_tensor, conditions_cln=conditions_cln),
                    conditions_cln=conditions_cln,
                )
            else:
                self.decoder(self.encoder(input_tensor))

    def forward(
        self,
        inputs: Sequence,
        output_only_last: bool = False,
        conditions_cln=None,
    ) -> torch.Tensor:
        r"""
        Forward pass of the recurrent HEALPix UNet.

        Parameters
        ----------
        inputs : Sequence
            List ``[prognostics, decoder_inputs, constants]`` or
            ``[prognostics, decoder_inputs, constants, couplings]`` with shapes
            consistent with :math:`(B, F, T, C, H, W)`.
        output_only_last : bool, optional
            If ``True``, return only the final forecast step.

        Returns
        -------
        torch.Tensor
            Model outputs shaped :math:`(B, F, T_{out}, C_{out}, H, W)`.
        """
        if not torch.compiler.is_compiling():
            if inputs[0].ndim != 6:
                raise ValueError(
                    "HEALPixRecUNet.forward expects prognostics shaped "
                    "(B, F, T, C, H, W)"
                )

        self.reset()
        outputs = []
        for step in range(self.integration_steps):
            # (Re-)initialize recurrent hidden states
            if (step * (self.delta_t * self.input_time_dim)) % self.reset_cycle == 0:
                if conditions_cln is not None:
                    self._initialize_hidden(
                        inputs=inputs,
                        outputs=outputs,
                        step=step,
                        conditions_cln=conditions_cln[step],
                    )
                else:
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
                # Only the prognostic channels of the previous output are fed
                # back in; any extra (diagnostic) output channels are dropped.
                prognostics = outputs[-1][:, :, :, : self.input_channels]
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        inputs=[prognostics]
                        + list(inputs[1:3])
                        + [inputs[3][self.presteps + step]],
                        step=step + self.presteps,
                    )
                else:
                    input_tensor = self._reshape_inputs(
                        inputs=[prognostics] + list(inputs[1:]),
                        step=step + self.presteps,
                    )

            if conditions_cln is not None:
                kwargs = {"conditions_cln": conditions_cln[step]}
            else:
                kwargs = {}

            encodings = self.encoder(input_tensor, **kwargs)
            decodings = self.decoder(encodings, **kwargs)

            combined = self._reshape_outputs(decodings)
            prognostics = combined[:, :, :, : self.input_channels]
            if self.residual_prediction:
                prognostics = prognostics + self._reshape_outputs(
                    input_tensor[:, : self.input_channels * self.input_time_dim]
                )
            diagnostics = combined[:, :, :, self.input_channels :]
            out = torch.cat([prognostics, diagnostics], dim=3)

            if self.constraints is not None:
                for constraint in self.constraints:
                    out = constraint(out)

            outputs.append(out)

        if output_only_last:
            return outputs[-1]

        return torch.cat(outputs, dim=self.channel_dim)

    def reset(self):
        r"""Reset the state of the encoder and decoder recurrent blocks."""
        self.encoder.reset()
        self.decoder.reset()
