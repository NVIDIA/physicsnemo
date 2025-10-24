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
import torch.distributed as dist
from physicsnemo.distributed import ShardTensor
from physicsnemo.distributed import DistributedManager


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

    for key, value in metrics.items():
        dist.all_reduce(value)
        value = value / dm.world_size
        metrics[key] = value

    return metrics


def metrics_fn(
    pred: torch.Tensor,
    target: torch.Tensor,
    others: dict[str, torch.Tensor],
    dm: DistributedManager,
    mode: str,
    norm_factors: dict[str, torch.Tensor],
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
            metrics = metrics_fn_surface(pred, target, others, dm, norm_factors)
        elif mode == "volume":
            metrics = metrics_fn_volume(pred, target, others, dm, norm_factors)
        else:
            raise ValueError(f"Unknown data mode: {mode}")

        metrics = all_reduce_dict(metrics, dm)
        return metrics


def metrics_fn_volume(
    pred: torch.Tensor,
    target: torch.Tensor,
    others: dict[str, torch.Tensor],
    dm: DistributedManager,
    norm_factors: dict[str, torch.Tensor],
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
    raise NotImplementedError("Volume metrics not yet implemented.")


def metrics_fn_surface(
    pred: torch.Tensor,
    target: torch.Tensor,
    others: dict[str, torch.Tensor],
    dm: DistributedManager,
    norm_factors: dict[str, torch.Tensor],
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
    target = target * norm_factors["std"] + norm_factors["mean"]
    pred = pred * norm_factors["std"] + norm_factors["mean"]

    l2_num = (pred - target) ** 2
    l2_num = torch.sum(l2_num, dim=1)
    l2_num = torch.sqrt(l2_num)

    l2_denom = target**2
    l2_denom = torch.sum(l2_denom, dim=1)
    l2_denom = torch.sqrt(l2_denom)

    l2 = l2_num / l2_denom

    metrics = {
        "l2_pressure": torch.mean(l2[:, 0]),
        "l2_shear_x": torch.mean(l2[:, 1]),
        "l2_shear_y": torch.mean(l2[:, 2]),
        "l2_shear_z": torch.mean(l2[:, 3]),
    }

    return metrics


def metrics_fn_surface_pressure(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Computes mean squared error between predicted and target surface pressure.

    Args:
        pred: Predicted surface pressure.
        target: Target surface pressure.

    Returns:
        Mean squared error as a torch.Tensor.
    """
    return torch.mean((pred - target) ** 2.0)
