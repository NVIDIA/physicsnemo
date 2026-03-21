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

"""Flexible loss calculator for configurable target fields."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from utils import FieldSpec, parse_target_config

# Default delta for Huber loss
DEFAULT_HUBER_DELTA = 1.0


# ---------------------------------------------------------------------------
# Core loss functions operating on tensors
# ---------------------------------------------------------------------------


def compute_huber(
    pred: torch.Tensor, target: torch.Tensor, delta: float = DEFAULT_HUBER_DELTA
) -> torch.Tensor:
    """Huber loss (smooth L1) for scalar fields.

    Huber loss is quadratic for small errors and linear for large errors,
    making it more robust to outliers than MSE.

    Args:
        pred: Predictions tensor
        target: Targets tensor
        delta: Threshold at which to switch from quadratic to linear.

    Returns:
        Mean Huber loss as a scalar tensor.
    """
    return F.huber_loss(pred, target, reduction="mean", delta=delta)


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error loss."""
    return torch.mean((pred - target) ** 2.0)


def compute_rmse(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Relative Mean Squared Error (normalized by target magnitude)."""
    num = torch.mean((pred - target) ** 2.0)
    denom = torch.mean(target**2.0)
    return num / (denom + eps)


def compute_huber_vector(
    pred: torch.Tensor, target: torch.Tensor, delta: float = DEFAULT_HUBER_DELTA
) -> torch.Tensor:
    """Huber loss for vector fields, summed across components.

    Args:
        pred: Predictions of shape [batch, points, dim]
        target: Targets of shape [batch, points, dim]
        delta: Threshold at which to switch from quadratic to linear.

    Returns:
        Sum of per-component Huber losses.
    """
    # Compute Huber loss per component
    total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    for i in range(pred.shape[-1]):
        total_loss = total_loss + F.huber_loss(
            pred[:, :, i], target[:, :, i], reduction="mean", delta=delta
        )
    return total_loss


def compute_mse_vector(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE for vector fields, summed across components.

    Args:
        pred: Predictions of shape [batch, points, dim]
        target: Targets of shape [batch, points, dim]

    Returns:
        Sum of per-component MSE losses.
    """
    # Compute mean squared diff per component, keeping last dim
    diff_sq = torch.mean((pred - target) ** 2.0, dim=(0, 1))
    return torch.sum(diff_sq)


def compute_rmse_vector(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Relative MSE for vector fields, normalized per component then summed.

    Args:
        pred: Predictions of shape [batch, points, dim]
        target: Targets of shape [batch, points, dim]
        eps: Small value to avoid division by zero.

    Returns:
        Sum of per-component relative MSE losses.
    """
    # Compute mean squared diff per component
    diff_sq = torch.mean((pred - target) ** 2.0, dim=(0, 1))
    # Compute mean squared target per component
    target_sq = torch.mean(target**2.0, dim=(0, 1))
    return torch.sum(diff_sq / (target_sq + eps))


LOSS_FUNCTIONS_SCALAR = {
    "huber": compute_huber,
    "mse": compute_mse,
    "rmse": compute_rmse,
}

LOSS_FUNCTIONS_VECTOR = {
    "huber": compute_huber_vector,
    "mse": compute_mse_vector,
    "rmse": compute_rmse_vector,
}


# ---------------------------------------------------------------------------
# LossCalculator class
# ---------------------------------------------------------------------------


class LossCalculator:
    """Configurable loss calculator for scalar and vector target fields.

    Computes loss for each configured target field separately, then combines them.
    Supports Huber, MSE, and RMSE (relative MSE) loss types.

    For vector fields, computes per-component losses and sums them.
    The final loss is normalized by the total number of channels.

    Parameters
    ----------
    target_config : dict[str, str]
        Mapping of field names to types. Order determines channel indices.
        Example: {"pressure": "scalar", "velocity": "vector", "turbulence": "scalar"}
    loss_type : Literal["huber", "mse", "rmse"], optional
        Type of loss to compute. Default is "huber".
        - "huber": Huber loss (smooth L1), robust to outliers
        - "mse": Mean Squared Error
        - "rmse": Relative MSE (normalized by target magnitude)
    vector_dim : int, optional
        Dimensionality of vector fields. Default is 3.
    prefix : str, optional
        Prefix for all loss names (e.g., "surface" -> "loss/surface/pressure").
        Default is empty string.
    normalize_by_channels : bool, optional
        Whether to normalize the total loss by the number of channels.
        Default is True.

    Examples
    --------
    >>> calc = LossCalculator(
    ...     target_config={"pressure": "scalar", "wall_shear": "vector"},
    ...     loss_type="huber",
    ...     prefix="surface",
    ... )
    >>> pred = torch.randn(2, 100, 4)  # [batch, points, channels]
    >>> target = torch.randn(2, 100, 4)
    >>> total_loss, loss_dict = calc(pred, target)
    """

    def __init__(
        self,
        target_config: dict[str, str],
        loss_type: Literal["huber", "mse", "rmse"] = "mse",
        vector_dim: int = 3,
        prefix: str = "",
        normalize_by_channels: bool = True,
    ):
        self.loss_type = loss_type
        self.vector_dim = vector_dim
        self.prefix = prefix
        self.normalize_by_channels = normalize_by_channels

        # Validate loss type
        if loss_type not in LOSS_FUNCTIONS_SCALAR:
            raise ValueError(
                f"Unknown loss type '{loss_type}'. "
                f"Available: {list(LOSS_FUNCTIONS_SCALAR.keys())}"
            )

        # Parse target config to build field specifications using shared utility
        self.field_specs = parse_target_config(target_config, vector_dim)
        self.total_channels = sum(spec.dim for spec in self.field_specs)

    def _make_key(self, *parts: str) -> str:
        """Construct a loss key with optional prefix."""
        key = "/".join(parts)
        return f"loss/{self.prefix}/{key}" if self.prefix else f"loss/{key}"

    def _compute_scalar_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        name: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute loss for a scalar field [batch, points].

        Returns:
            Tuple of (loss_value, {loss_key: loss_value})
        """
        loss_fn = LOSS_FUNCTIONS_SCALAR[self.loss_type]
        loss = loss_fn(pred, target)
        return loss, {self._make_key(name): loss}

    def _compute_vector_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        name: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute loss for a vector field [batch, points, dim].

        Returns:
            Tuple of (loss_value, {loss_key: loss_value})
        """
        loss_fn = LOSS_FUNCTIONS_VECTOR[self.loss_type]
        loss = loss_fn(pred, target)
        return loss, {self._make_key(name): loss}

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute losses for all configured fields.

        Args:
            pred: Predicted values, shape [batch, points, channels].
            target: Target values, shape [batch, points, channels].

        Returns:
            Tuple of:
                - total_loss: Combined loss as a scalar tensor
                - loss_dict: Dictionary of loss name -> scalar tensor value
        """
        if pred.shape[-1] != self.total_channels:
            raise ValueError(
                f"Expected {self.total_channels} channels based on target config, "
                f"but got {pred.shape[-1]}."
            )

        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        loss_dict = {}

        for spec in self.field_specs:
            pred_field = pred[:, :, spec.start_index : spec.end_index]
            target_field = target[:, :, spec.start_index : spec.end_index]

            if spec.field_type == "scalar":
                field_loss, field_dict = self._compute_scalar_loss(
                    pred_field.squeeze(-1), target_field.squeeze(-1), spec.name
                )
            else:
                field_loss, field_dict = self._compute_vector_loss(
                    pred_field, target_field, spec.name
                )

            total_loss = total_loss + field_loss
            loss_dict.update(field_dict)

        # Normalize by total channels if requested
        if self.normalize_by_channels:
            total_loss = total_loss / self.total_channels

        # Add total loss to dict
        total_key = f"loss/{self.prefix}" if self.prefix else "loss/total"
        loss_dict[total_key] = total_loss

        return total_loss, loss_dict

    def __repr__(self) -> str:
        fields_str = ", ".join(
            f"{s.name}:{s.field_type}[{s.start_index}:{s.end_index}]"
            for s in self.field_specs
        )
        return (
            f"LossCalculator(fields=[{fields_str}], "
            f"loss_type='{self.loss_type}', prefix='{self.prefix}')"
        )


# ---------------------------------------------------------------------------
# Convenience functions for direct use
# ---------------------------------------------------------------------------


def loss_fn_surface(
    output: torch.Tensor, target: torch.Tensor, loss_type: Literal["mse", "rmse"]
) -> torch.Tensor:
    """Calculate loss for surface data by handling scalar and vector components separately.

    This is a convenience function that matches the original implementation.
    For new code, prefer using LossCalculator for configurability.

    Assumes surface data format: [temp (1), pressure (1), wall_shear (3)]

    Args:
        output: Predicted surface values from the model [batch, points, 5]
        target: Ground truth surface values [batch, points, 5]
        loss_type: Type of loss to calculate ("mse" or "rmse")

    Returns:
        Combined scalar and vector loss as a scalar tensor
    """
    # Separate the scalar and vector components:
    output_temp, output_pres, output_wss = torch.split(output, [1, 1, 3], dim=2)
    target_temp, target_pres, target_wss = torch.split(target, [1, 1, 3], dim=2)

    num_temp = torch.mean((output_temp - target_temp) ** 2.0)
    num_pres = torch.mean((output_pres - target_pres) ** 2.0)
    wss_diff_sq = torch.mean((target_wss - output_wss) ** 2.0, (0, 1))

    if loss_type == "mse":
        masked_loss_pres = num_pres
        masked_loss_temp = num_temp
        masked_loss_ws = torch.sum(wss_diff_sq)
    else:
        denom_pres = torch.mean(target_pres**2.0)
        masked_loss_pres = num_pres / denom_pres

        denom_temp = torch.mean(target_temp**2.0)
        masked_loss_temp = num_temp / denom_temp

        # Compute the mean diff**2 of the WSS, leave the last dimension:
        masked_loss_ws_num = wss_diff_sq
        masked_loss_ws_denom = torch.mean(target_wss**2.0, (0, 1))
        masked_loss_ws = torch.sum(masked_loss_ws_num / masked_loss_ws_denom)

    loss = masked_loss_pres + masked_loss_temp + masked_loss_ws

    return loss / 5.0


def loss_fn_volume(
    output: torch.Tensor, target: torch.Tensor, loss_type: Literal["mse", "rmse"]
) -> torch.Tensor:
    """Calculate loss for volume data by handling scalar and vector components separately.

    Assumes volume data format: [velocity (3), pressure (1), turbulence (1)]

    Args:
        output: Predicted volume values from the model [batch, points, 5]
        target: Ground truth volume values [batch, points, 5]
        loss_type: Type of loss to calculate ("mse" or "rmse")

    Returns:
        Combined scalar and vector loss as a scalar tensor
    """
    # Separate the scalar and vector components:
    output_vel, output_pres, output_turb = torch.split(output, [3, 1, 1], dim=2)
    target_vel, target_pres, target_turb = torch.split(target, [3, 1, 1], dim=2)

    num_pres = torch.mean((output_pres - target_pres) ** 2.0)
    num_turb = torch.mean((output_turb - target_turb) ** 2.0)
    vel_diff_sq = torch.mean((target_vel - output_vel) ** 2.0, (0, 1))

    if loss_type == "mse":
        masked_loss_pres = num_pres
        masked_loss_turb = num_turb
        masked_loss_vel = torch.sum(vel_diff_sq)
    else:
        denom_pres = torch.mean(target_pres**2.0)
        masked_loss_pres = num_pres / denom_pres

        denom_turb = torch.mean(target_turb**2.0)
        masked_loss_turb = num_turb / denom_turb

        masked_loss_vel_num = vel_diff_sq
        masked_loss_vel_denom = torch.mean(target_vel**2.0, (0, 1))
        masked_loss_vel = torch.sum(masked_loss_vel_num / masked_loss_vel_denom)

    loss = masked_loss_pres + masked_loss_turb + masked_loss_vel

    return loss / 5.0
