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

"""Fixed field decoding and the Rotor37 training objective."""

import torch
from torch import nn


class FieldDecoder(nn.Module):
    """Reconstruct positive surface fields from basis coefficients.

    The first ``pressure_rank`` coefficients multiply the pressure modes and
    the remaining coefficients multiply the temperature modes. Adding the
    training mean log fields and exponentiating gives positive pressure and
    temperature, and density follows from the ideal-gas relation. The decoder
    has no learned parameters.
    """

    def __init__(self, basis, stats):
        super().__init__()
        for name in ("pressure", "temperature"):
            self.register_buffer(f"{name}_modes", self._tensor(basis[f"{name}_modes"]))
            self.register_buffer(
                f"{name}_log_mean", self._tensor(basis[f"{name}_log_mean"])
            )
            eigenvalues = self._tensor(basis[f"{name}_eigenvalues"])
            # Pressure coefficients are weighted by their eigenvalues and
            # temperature coefficients by the square roots, each with mean one.
            weights = eigenvalues if name == "pressure" else eigenvalues.sqrt()
            self.register_buffer(f"{name}_weights", weights / weights.mean())
        self.pressure_rank = len(self.pressure_modes)
        self.coefficient_count = self.pressure_rank + len(self.temperature_modes)
        self.register_buffer("gas_constant", self._tensor(stats["gas_constant"]))
        self.register_buffer("field_mean", self._tensor(stats["fields"]["mean"]))
        self.register_buffer("field_std", self._tensor(stats["fields"]["std"]))
        self.register_buffer("edges", torch.as_tensor(basis["edges"], dtype=torch.long))
        self.register_buffer("jump_scale", self._tensor(basis["pressure_jump_scale"]))

    @staticmethod
    def _tensor(values):
        return torch.as_tensor(values, dtype=torch.float32)

    def physical(self, coefficients):
        """Return density, pressure and temperature with shape ``[B, V, 3]``."""
        coefficients = coefficients.float()
        pressure = (
            coefficients[:, : self.pressure_rank] @ self.pressure_modes
            + self.pressure_log_mean
        ).exp()
        temperature = (
            coefficients[:, self.pressure_rank :] @ self.temperature_modes
            + self.temperature_log_mean
        ).exp()
        density = pressure / (self.gas_constant * temperature)
        return torch.stack((density, pressure, temperature), dim=-1)

    def forward(self, coefficients):
        """Return fields normalized with the training statistics."""
        return (self.physical(coefficients) - self.field_mean) / self.field_std


def pressure_jump_loss(predicted, target, edges, scale):
    """Mean squared error of signed pressure differences across mesh edges.

    ``predicted`` and ``target`` hold normalized pressure with shape
    ``[B, V]``. The result has one value per case divided by ``scale``.
    """
    error = predicted - target
    residual = error[:, edges[:, 0]] - error[:, edges[:, 1]]
    return residual.square().mean(dim=1) / scale


def loss_components(prediction, batch, decoder):
    """Return the per-case terms of the training objective."""
    coefficient_error = (prediction.coefficients - batch["coefficients"]).square()
    pressure_rank = decoder.pressure_rank
    return {
        "field": (prediction.fields - batch["fields"]).square().mean(dim=(1, 2)),
        "global": (prediction.globals - batch["globals"]).square().mean(dim=1),
        "pressure_coefficient": (
            coefficient_error[:, :pressure_rank] * decoder.pressure_weights
        ).mean(dim=1),
        "temperature_coefficient": (
            coefficient_error[:, pressure_rank:] * decoder.temperature_weights
        ).mean(dim=1),
        "jump": pressure_jump_loss(
            prediction.fields[..., 1],
            batch["fields"][..., 1],
            decoder.edges,
            decoder.jump_scale,
        ),
    }
