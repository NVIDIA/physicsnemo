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

"""PhysicsNeMo FNO and thermal physics losses for FP-DDM."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from physicsnemo import ModelMetaData, Module
from physicsnemo.models.fno import FNO


def _boundary_mask(values: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[..., 0, :] = True
    mask[..., -1, :] = True
    mask[..., :, 0] = True
    mask[..., :, -1] = True
    return mask


def thermal_residual(
    temperature: torch.Tensor,
    inputs: torch.Tensor,
    source_scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Return the interior residual of ``div(k grad(T)) + q = 0``.

    Parameters are expected in nondimensional form. ``source_scale`` converts
    the normalized heat-source channel into the conductivity-temperature
    scaling used by the divergence term.
    """

    conductivity = inputs[:, 2:3]
    heat_source = inputs[:, 4:5]
    height, width = temperature.shape[-2:]
    dy = 1.0 / max(height - 1, 1)
    dx = 1.0 / max(width - 1, 1)
    grad_x = torch.gradient(temperature, spacing=(dx,), dim=(3,))[0]
    grad_y = torch.gradient(temperature, spacing=(dy,), dim=(2,))[0]
    flux_x = conductivity * grad_x
    flux_y = conductivity * grad_y
    div_x = torch.gradient(flux_x, spacing=(dx,), dim=(3,))[0]
    div_y = torch.gradient(flux_y, spacing=(dy,), dim=(2,))[0]
    return div_x + div_y + source_scale * heat_source


def thermal_losses(
    temperature: torch.Tensor,
    inputs: torch.Tensor,
    source_scale: torch.Tensor | float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean-squared PDE and Dirichlet boundary residuals."""

    residual = thermal_residual(temperature, inputs, source_scale)
    pde_loss = residual[..., 1:-1, 1:-1].square().mean()
    temperature_bc = inputs[:, 3:4]
    boundary_loss = (
        (temperature - temperature_bc)[_boundary_mask(temperature)].square().mean()
    )
    return pde_loss, boundary_loss


class ThermalPINO(Module):
    """Physics-informed FNO local solver with thermal nondimensionalization."""

    def __init__(self, config: Mapping[str, object]):
        """Build the PhysicsNeMo FNO and register normalization constants."""

        super().__init__(meta=ModelMetaData())
        normalization = config["normalization"]
        if not isinstance(normalization, Mapping):
            raise TypeError("model.normalization must be a mapping")

        width = int(config.get("width", 32))
        self.fno = FNO(
            in_channels=int(config.get("in_channels", 5)),
            out_channels=int(config.get("out_channels", 1)),
            decoder_layers=int(config.get("decoder_layers", 1)),
            decoder_layer_size=int(config.get("decoder_width", width)),
            dimension=2,
            latent_channels=width,
            num_fno_layers=int(config.get("layers", 4)),
            num_fno_modes=int(config.get("modes", 16)),
            padding=int(config.get("padding", 0)),
            coord_features=False,
        )
        self.register_buffer(
            "length_scale",
            torch.tensor(float(normalization["length"]), dtype=torch.float32),
        )
        self.register_buffer(
            "conductivity_scale",
            torch.tensor(float(normalization["conductivity"]), dtype=torch.float32),
        )
        self.register_buffer(
            "temperature_reference",
            torch.tensor(
                float(normalization["temperature_reference"]), dtype=torch.float32
            ),
        )
        self.register_buffer(
            "temperature_scale",
            torch.tensor(
                float(normalization["temperature_scale"]), dtype=torch.float32
            ),
        )
        self.register_buffer(
            "heat_source_scale",
            torch.tensor(float(normalization["heat_source"]), dtype=torch.float32),
        )
        self.use_dimensional_input = False
        self.use_dimensional_output = False
        self.enforce_boundary = False

    @property
    def nondimensional_source_scale(self) -> torch.Tensor:
        """Return the source coefficient in the nondimensional PDE."""

        return (
            self.heat_source_scale
            * self.length_scale.square()
            / (self.conductivity_scale * self.temperature_scale)
        )

    def set_dimensional_mode(self, input_enabled: bool, output_enabled: bool) -> None:
        """Control dimensional conversion at the model input and output."""

        self.use_dimensional_input = input_enabled
        self.use_dimensional_output = output_enabled

    def set_boundary_enforcement(self, enabled: bool) -> None:
        """Enable or disable exact Dirichlet values on the model output."""

        self.enforce_boundary = enabled

    def normalize_input(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convert dimensional model inputs into nondimensional channels."""

        return torch.cat(
            [
                inputs[:, 0:1] / self.length_scale,
                inputs[:, 1:2] / self.length_scale,
                inputs[:, 2:3] / self.conductivity_scale,
                (inputs[:, 3:4] - self.temperature_reference) / self.temperature_scale,
                inputs[:, 4:5] / self.heat_source_scale,
            ],
            dim=1,
        )

    def denormalize_output(self, output: torch.Tensor) -> torch.Tensor:
        """Convert normalized temperature predictions into dimensional values."""

        return output * self.temperature_scale + self.temperature_reference

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict a local temperature field from five thermal input channels."""

        boundary_source = inputs
        if self.use_dimensional_input:
            inputs = self.normalize_input(inputs)
        output = self.fno(inputs)
        if self.use_dimensional_output:
            output = self.denormalize_output(output)
        if self.enforce_boundary:
            temperature_bc = boundary_source[:, 3:4]
            if self.use_dimensional_input and not self.use_dimensional_output:
                temperature_bc = (
                    temperature_bc - self.temperature_reference
                ) / self.temperature_scale
            elif not self.use_dimensional_input and self.use_dimensional_output:
                temperature_bc = self.denormalize_output(temperature_bc)
            output = torch.where(_boundary_mask(temperature_bc), temperature_bc, output)
        return output

    def loss_terms(
        self, inputs: torch.Tensor, *, normalize_input: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate thermal PDE and boundary losses for a training batch."""

        if normalize_input:
            inputs = self.normalize_input(inputs)
        prediction = self.fno(inputs)
        return thermal_losses(
            prediction, inputs, source_scale=self.nondimensional_source_scale
        )
