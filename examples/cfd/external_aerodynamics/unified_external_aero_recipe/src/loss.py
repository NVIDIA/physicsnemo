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


def compute_relative_mse(
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
    "rmse": compute_relative_mse,
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
    n_spatial_dims : int, optional
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
        loss_type: Literal["huber", "mse", "rmse"] = "huber",
        n_spatial_dims: int = 3,
        prefix: str = "",
        normalize_by_channels: bool = True,
    ):
        self.loss_type = loss_type
        self.n_spatial_dims = n_spatial_dims
        self.prefix = prefix
        self.normalize_by_channels = normalize_by_channels

        # Validate loss type
        if loss_type not in LOSS_FUNCTIONS_SCALAR:
            raise ValueError(
                f"Unknown loss type '{loss_type}'. "
                f"Available: {list(LOSS_FUNCTIONS_SCALAR.keys())}"
            )

        # Parse target config to build field specifications using shared utility
        self.field_specs = parse_target_config(target_config, n_spatial_dims)
        self.total_channels = sum(spec.dim for spec in self.field_specs)

    def _make_key(self, *parts: str) -> str:
        """Construct a loss key with optional prefix."""
        segments = ["loss"]
        if self.prefix:
            segments.append(self.prefix)
        segments.extend(parts)
        return "/".join(segments)

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
