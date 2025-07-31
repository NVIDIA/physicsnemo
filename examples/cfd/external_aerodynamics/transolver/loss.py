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

import torch
from typing import Literal


def loss_fn(
    pred: torch.Tensor,
    target: torch.Tensor,
    others: dict[str, torch.Tensor],
    mode: Literal["surface", "volume"],
) -> torch.Tensor:
    """
    Compute the main loss function for the model.

    Args:
        pred: Predicted tensor from the model.
        target: Ground truth tensor.
        others: Dictionary of additional tensors (e.g., surface_areas, surface_normals, stream_velocity).

    Returns:
        Loss value as a scalar tensor.
    """
    if mode == "surface":
        loss = loss_fn_surface(pred, target, "mse")
    elif mode == "volume":
        loss = loss_fn_volume(pred, target, "mse")
    # 100 * integral_loss_fn(pred, target, others["surface_areas"], others["surface_normals"], others["stream_velocity"])
    return loss


def loss_fn_volume(
    pred: torch.Tensor, target: torch.Tensor, mode: Literal["mse", "rmse"]
) -> torch.Tensor:
    """
    Compute the main loss function for the model.
    """

    if mode == "rmse":
        dims = (0, 1)
    else:
        dims = None

    num = torch.sum((pred - target) ** 2.0, dims)
    if mode == "rmse":
        denom = torch.sum(target**2.0, dims)
    else:
        denom = pred.shape[1]

    return torch.mean(num / denom)


def loss_fn_surface(
    output: torch.Tensor, target: torch.Tensor, loss_type: Literal["mse", "rmse"]
) -> torch.Tensor:
    """Calculate loss for surface data by handling scalar and vector components separately.

    Args:
        output: Predicted surface values from the model.
        target: Ground truth surface values.
        loss_type: Type of loss to calculate ("mse" or "rmse").

    Returns:
        Combined scalar and vector loss as a scalar tensor.
    """
    # Separate the scalar and vector components:
    output_pressure, output_sheer = torch.split(output, [1, 3], dim=2)
    target_pressure, target_sheer = torch.split(target, [1, 3], dim=2)

    numerator_pressure = torch.mean((output_pressure - target_pressure) ** 2.0)
    numerator_sheer = torch.mean((target_sheer - output_sheer) ** 2.0, (0, 1))

    if loss_type == "mse":
        loss_pressure = numerator_pressure
        loss_wall_sheer = torch.sum(numerator_sheer)
    else:
        denom = torch.mean((target_pressure) ** 2.0)
        loss_pressure = numerator_pressure / denom

        # Compute the mean diff**2 of the vector component, leave the last dimension:
        denom_sheer = torch.mean((target_sheer) ** 2.0, (0, 1))
        loss_wall_sheer = torch.sum(numerator_sheer / denom_sheer)

    loss = loss_pressure + loss_wall_sheer

    return loss / 4.0


def integral_loss_fn(
    output: torch.Tensor,
    target: torch.Tensor,
    area: torch.Tensor,
    normals: torch.Tensor,
    stream_velocity: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute the integral loss (sum of lift and drag losses).

    Args:
        output: Predicted tensor.
        target: Ground truth tensor.
        area: Surface area tensor.
        normals: Surface normals tensor.
        stream_velocity: Stream velocity tensor (optional).

    Returns:
        Scalar tensor representing the sum of lift and drag losses.
    """
    drag_loss = drag_loss_fn(output, target, area, normals, stream_velocity)
    lift_loss = lift_loss_fn(output, target, area, normals, stream_velocity)
    return lift_loss + drag_loss


def lift_loss_fn(
    output: torch.Tensor,
    target: torch.Tensor,
    area: torch.Tensor,
    normals: torch.Tensor,
    stream_velocity: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute the lift loss.

    Args:
        output: Predicted tensor.
        target: Ground truth tensor.
        area: Surface area tensor.
        normals: Surface normals tensor.
        stream_velocity: Stream velocity tensor (optional).

    Returns:
        Scalar tensor representing the lift loss.
    """
    vel_inlet = stream_velocity  # Get this from the dataset
    # mask = abs(target - padded_value) > 1e-3

    output_true = target * area * (vel_inlet) ** 2.0
    output_pred = output * area * (vel_inlet) ** 2.0

    normals = torch.select(normals, 2, 2)
    output_true_0 = output_true.select(2, 0)
    output_pred_0 = output_pred.select(2, 0)

    pres_true = output_true_0 * normals
    pres_pred = output_pred_0 * normals

    wz_true = output_true[:, :, -1]
    wz_pred = output_pred[:, :, -1]

    masked_pred = torch.mean(pres_pred + wz_pred, (1))
    masked_truth = torch.mean(pres_true + wz_true, (1))

    loss = (masked_pred - masked_truth) ** 2.0
    loss = torch.mean(loss)
    return loss


def drag_loss_fn(
    output: torch.Tensor,
    target: torch.Tensor,
    area: torch.Tensor,
    normals: torch.Tensor,
    stream_velocity: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute the drag loss.

    Args:
        output: Predicted tensor.
        target: Ground truth tensor.
        area: Surface area tensor.
        normals: Surface normals tensor.
        stream_velocity: Stream velocity tensor (optional).

    Returns:
        Scalar tensor representing the drag loss.
    """
    vel_inlet = stream_velocity  # Get this from the dataset
    # mask = abs(target - padded_value) > 1e-3
    output_true = target * area * (vel_inlet) ** 2.0
    output_pred = output * area * (vel_inlet) ** 2.0

    pres_true = output_true[:, :, 0] * normals[:, :, 0]
    pres_pred = output_pred[:, :, 0] * normals[:, :, 0]

    wx_true = output_true[:, :, 1]
    wx_pred = output_pred[:, :, 1]

    masked_pred = torch.mean(pres_pred + wx_pred, (1))
    masked_truth = torch.mean(pres_true + wx_true, (1))

    loss = (masked_pred - masked_truth) ** 2.0
    loss = torch.mean(loss)
    return loss
