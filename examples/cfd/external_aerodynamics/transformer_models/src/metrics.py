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
from physicsnemo.distributed import DistributedManager, fused_all_reduce

from utils import tensorwise


@tensorwise
def metrics_fn(
    pred: torch.Tensor,
    target: torch.Tensor,
    dm: DistributedManager,
    mode: str,
) -> dict[str, torch.Tensor]:
    """
    Computes metrics for either surface or volume data.

    Each rank collapses its fields to scalar means, then one fused ``AVG``
    collective averages those across ranks (a mean-of-means). This weights
    every rank equally, which equals the true mean because every rank holds
    the same number of samples (the val sampler pads to even shards).

    Args:
        pred: Predicted values (unnormalized).
        target: Target values (unnormalized).
        dm: DistributedManager instance for distributed context.
        mode: Either "surface" or "volume".

    Returns:
        Flat dictionary of metric names to scalar tensors, globally averaged
        across ranks.
    """
    with torch.no_grad():
        if mode == "surface":
            fields = metrics_fn_surface(pred, target, dm)
        elif mode == "volume":
            fields = metrics_fn_volume(pred, target, dm)
        else:
            raise ValueError(f"Unknown data mode: {mode!r}")

        # Rank-local mean per field, then one fused AVG across ranks.
        rank_means = {key: value.mean() for key, value in fields.items()}
        return fused_all_reduce(rank_means, op=dist.ReduceOp.AVG)


def metrics_fn_volume(
    pred: torch.Tensor,
    target: torch.Tensor,
    dm: DistributedManager,
) -> dict[str, torch.Tensor]:
    """Compute volume-field error metrics: relative L2, relative L1, and MAE.

    Covers pressure, the three velocity components, velocity magnitude, and nut.

    Args:
        pred: Predicted volume fields, shape (B, N, C).
        target: Target volume fields, shape (B, N, C).
        dm: DistributedManager instance for distributed context.

    Returns:
        Each metric name mapped to its per-element error tensor, which
        :func:`metrics_fn` collapses to a scalar mean and averages across ranks.
    """
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
        "l2_pressure_vol": l2[:, 3],
        "l2_velocity_x": l2[:, 0],
        "l2_velocity_y": l2[:, 1],
        "l2_velocity_z": l2[:, 2],
        "l2_nut": l2[:, 4],
        "l1_pressure_vol": l1[:, 3],
        "l1_velocity_x": l1[:, 0],
        "l1_velocity_y": l1[:, 1],
        "l1_velocity_z": l1[:, 2],
        "l1_nut": l1[:, 4],
        "mae_pressure_vol": mae_pressure,
        "mae_velocity_x": mae_num[:, :, 0],
        "mae_velocity_y": mae_num[:, :, 1],
        "mae_velocity_z": mae_num[:, :, 2],
        "mae_nut": mae_num[:, 4],
        "l2_velocity": l2_vel,
        "l1_velocity": l1_vel,
        "mae_velocity": mae_num_vel,
    }

    return metrics


def metrics_fn_surface(
    pred: torch.Tensor,
    target: torch.Tensor,
    dm: DistributedManager,
) -> dict[str, torch.Tensor]:
    """Compute surface-field error metrics: relative L2, relative L1, and MAE.

    Covers pressure, the three wall-shear components, and wall-shear-stress
    magnitude.

    Args:
        pred: Predicted surface fields, shape (B, N, C).
        target: Target surface fields, shape (B, N, C).
        dm: DistributedManager instance for distributed context.

    Returns:
        Each metric name mapped to its per-element error tensor, which
        :func:`metrics_fn` collapses to a scalar mean and averages across ranks.
    """
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
        "l2_pressure_surf": l2[:, 0],
        "l2_shear_x": l2[:, 1],
        "l2_shear_y": l2[:, 2],
        "l2_shear_z": l2[:, 3],
        "l1_pressure_surf": l1[:, 0],
        "l1_shear_x": l1[:, 1],
        "l1_shear_y": l1[:, 2],
        "l1_shear_z": l1[:, 3],
        "mae_pressure_surf": mae_pressure,
        "mae_shear_x": mae_num[:, :, 1],
        "mae_shear_y": mae_num[:, :, 2],
        "mae_shear_z": mae_num[:, :, 3],
        "l2_wall_shear_stress": l2_ws,
        "l1_wall_shear_stress": l1_ws,
        "mae_wall_shear_stress": mae_wall_shear,
    }

    return metrics
