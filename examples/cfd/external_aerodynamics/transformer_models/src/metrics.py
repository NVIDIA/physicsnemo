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

import torch
import torch.distributed as dist
from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.distributed import DistributedManager

from utils import tensorwise


def all_reduce_dict(
    metrics: dict[str, torch.Tensor], dm: DistributedManager
) -> dict[str, torch.Tensor]:
    """
    Reduces a dictionary of metrics across all distributed processes.

    Args:
        metrics: Dictionary of metric names to torch.Tensor values.
        dm: DistributedManager instance for distributed context.

    Returns:
        Dictionary of reduced metrics.
    """
    # TODO - update this to use domains and not the full world

    if dm.world_size == 1:
        return metrics

    # Pack the metrics together:
    merged_metrics = torch.stack(list(metrics.values()), dim=-1)

    dist.all_reduce(merged_metrics)
    merged_metrics = merged_metrics / dm.world_size

    # Unstack metrics:
    metrics = {key: merged_metrics[i] for i, key in enumerate(metrics.keys())}
    return metrics


@tensorwise
def metrics_fn(
    pred: torch.Tensor,
    target: torch.Tensor,
    dm: DistributedManager,
    mode: str,
) -> dict[str, torch.Tensor]:
    """
    Computes metrics for either surface or volume data.

    Args:
        pred: Predicted values (unnormalized).
        target: Target values (unnormalized).
        others: Dictionary containing normalization statistics.
        dm: DistributedManager instance for distributed context.
        mode: Either "surface" or "volume".

    Returns:
        Dictionary of computed metrics.
    """
    with torch.no_grad():
        if mode == "surface":
            metrics = metrics_fn_surface(pred, target, dm)
        elif mode == "volume":
            metrics = metrics_fn_volume(pred, target, dm)
        else:
            raise ValueError(f"Unknown data mode: {mode}")

        metrics = all_reduce_dict(metrics, dm)
        return metrics


def metrics_fn_volume(
    pred: torch.Tensor,
    target: torch.Tensor,
    dm: DistributedManager,
) -> dict[str, torch.Tensor]:
    """
    Placeholder for volume metrics computation.

    Args:
        pred: Predicted values.
        target: Target values.
        others: Dictionary containing additional statistics.
        dm: DistributedManager instance for distributed context.
        norm_factors: Dictionary of normalization factors.

    Raises:
        NotImplementedError: Always, as this function is not yet implemented.
    """

    #
    pressure_pred = pred[:, :, 3]
    pressure_target = target[:, :, 3]

    velocity_pred = torch.sqrt(torch.sum(pred[:, :, 0:3] ** 2.0, dim=2))
    velocity_target = torch.sqrt(torch.sum(target[:, :, 0:3] ** 2.0, dim=2))

    # L1 errors
    l1_num = torch.abs(pred - target)
    l1_num = torch.sum(l1_num, dim=1)

    l1_denom = torch.abs(target)
    l1_denom = torch.sum(l1_denom, dim=1)

    l1 = l1_num / l1_denom

    # L1 errors velocity
    l1_num_vel = torch.abs(velocity_pred - velocity_target)
    l1_num_vel = torch.sum(l1_num_vel)

    l1_denom_vel = torch.abs(velocity_target)
    l1_denom_vel = torch.sum(l1_denom_vel)

    l1_vel = l1_num_vel / l1_denom_vel

    # MAE
    mae_num = torch.abs(pred - target)
    mae_num_vel = torch.abs(velocity_pred - velocity_target)
    mae_pressure = torch.abs(pressure_pred - pressure_target)

    # L2 errors
    l2_num = (pred - target) ** 2
    l2_num = torch.sum(l2_num, dim=1)
    l2_num = torch.sqrt(l2_num)

    l2_denom = target**2
    l2_denom = torch.sum(l2_denom, dim=1)
    l2_denom = torch.sqrt(l2_denom)

    l2 = l2_num / l2_denom

    # L2 errors velocity
    l2_num_vel = (velocity_pred - velocity_target) ** 2
    l2_num_vel = torch.sum(l2_num_vel)
    l2_num_vel = torch.sqrt(l2_num_vel)

    l2_denom_vel = velocity_target**2
    l2_denom_vel = torch.sum(l2_denom_vel)
    l2_denom_vel = torch.sqrt(l2_denom_vel)

    l2_vel = l2_num_vel / l2_denom_vel

    metrics = {
        "l2_pressure_vol": torch.mean(l2[:, 3]),
        "l2_velocity_x": torch.mean(l2[:, 0]),
        "l2_velocity_y": torch.mean(l2[:, 1]),
        "l2_velocity_z": torch.mean(l2[:, 2]),
        "l2_nut": torch.mean(l2[:, 4]),
        "l1_pressure_vol": torch.mean(l1[:, 3]),
        "l1_velocity_x": torch.mean(l1[:, 0]),
        "l1_velocity_y": torch.mean(l1[:, 1]),
        "l1_velocity_z": torch.mean(l1[:, 2]),
        "l1_nut": torch.mean(l1[:, 4]),
        "mae_pressure_vol": torch.mean(mae_pressure),
        "mae_velocity_x": torch.mean(mae_num[:, :, 0]),
        "mae_velocity_y": torch.mean(mae_num[:, :, 1]),
        "mae_velocity_z": torch.mean(mae_num[:, :, 2]),
        "mae_nut": torch.mean(mae_num[:, 4]),
        "l2_velocity": torch.mean(l2_vel),
        "l1_velocity": torch.mean(l1_vel),
        "mae_velocity": torch.mean(mae_num_vel),
    }

    return metrics


def metrics_fn_surface(
    pred: torch.Tensor,
    target: torch.Tensor,
    dm: DistributedManager,
) -> dict[str, torch.Tensor]:
    """
    Computes L2 surface metrics between prediction and target.

    Args:
        pred: Predicted values (normalized).
        target: Target values (normalized).
        others: Dictionary containing normalization statistics.
        dm: DistributedManager instance for distributed context.
        norm_factors: Dictionary with 'mean' and 'std' for unnormalization.

    Returns:
        Dictionary of L2 surface metrics for pressure and shear components.
    """
    # Unnormalize the surface values for L2:
    # target = target * norm_factors["std"] + norm_factors["mean"]
    # pred = pred * norm_factors["std"] + norm_factors["mean"]

    pressure_pred = pred[:, :, 0]
    pressure_target = target[:, :, 0]

    wall_shear_pred = torch.sqrt(torch.sum(pred[:, :, 1:4] ** 2.0, dim=2))
    wall_shear_target = torch.sqrt(torch.sum(target[:, :, 1:4] ** 2.0, dim=2))

    # MAE
    mae_num = torch.abs(pred - target)
    mae_wall_shear = torch.abs(wall_shear_pred - wall_shear_target)
    mae_pressure = torch.abs(pressure_pred - pressure_target)

    # L1 errors
    l1_num = torch.abs(pred - target)
    l1_num = torch.sum(l1_num, dim=1)

    l1_denom = torch.abs(target)
    l1_denom = torch.sum(l1_denom, dim=1)

    l1 = l1_num / l1_denom

    # L1 errors for wall shear stress
    l1_num_ws = torch.abs(wall_shear_pred - wall_shear_target)
    l1_num_ws = torch.sum(l1_num_ws)

    l1_denom_ws = torch.abs(wall_shear_target)
    l1_denom_ws = torch.sum(l1_denom_ws)

    l1_ws = l1_num_ws / l1_denom_ws

    # L2 errors
    l2_num = (pred - target) ** 2
    l2_num = torch.sum(l2_num, dim=1)
    l2_num = torch.sqrt(l2_num)

    l2_denom = target**2
    l2_denom = torch.sum(l2_denom, dim=1)
    l2_denom = torch.sqrt(l2_denom)

    l2 = l2_num / l2_denom

    # L2 errors for wall shear stress
    l2_num_ws = (wall_shear_pred - wall_shear_target) ** 2
    l2_num_ws = torch.sum(l2_num_ws)
    l2_num_ws = torch.sqrt(l2_num_ws)

    l2_denom_ws = wall_shear_target**2
    l2_denom_ws = torch.sum(l2_denom_ws)
    l2_denom_ws = torch.sqrt(l2_denom_ws)

    l2_ws = l2_num_ws / l2_denom_ws

    metrics = {
        "l2_pressure_surf": torch.mean(l2[:, 0]),
        "l2_shear_x": torch.mean(l2[:, 1]),
        "l2_shear_y": torch.mean(l2[:, 2]),
        "l2_shear_z": torch.mean(l2[:, 3]),
        "l1_pressure_surf": torch.mean(l1[:, 0]),
        "l1_shear_x": torch.mean(l1[:, 1]),
        "l1_shear_y": torch.mean(l1[:, 2]),
        "l1_shear_z": torch.mean(l1[:, 3]),
        "mae_pressure_surf": torch.mean(mae_pressure),
        "mae_shear_x": torch.mean(mae_num[:, :, 1]),
        "mae_shear_y": torch.mean(mae_num[:, :, 2]),
        "mae_shear_z": torch.mean(mae_num[:, :, 3]),
        "l2_wall_shear_stress": torch.mean(l2_ws),
        "l1_wall_shear_stress": torch.mean(l1_ws),
        "mae_wall_shear_stress": torch.mean(mae_wall_shear),
    }

    return metrics
