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

from physicsnemo.distributed import DistributedManager


def all_reduce_dict(metrics, dm):
    # TODO - update this to use domains and not the full world

    if dm.world_size == 1:
        return metrics

    for key, value in metrics.items():
        dist.all_reduce(value)
        value = value / dm.world_size
        metrics[key] = value

    return metrics


def metrics_fn(pred, target, dm):
    return metrics_fn_surface(pred, target, dm)


def metrics_fn_surface(pred, target, dm):

    l2_num = (pred - target) ** 2
    l2_num = torch.sum(l2_num, dim=1)
    l2_num = torch.sqrt(l2_num)

    l2_denom = target**2
    l2_denom = torch.sum(l2_denom, dim=1)
    l2_denom = torch.sqrt(l2_denom)

    l2 = l2_num / l2_denom

    metrics = {
        "l2_pressure": torch.mean(l2[:, 0]),
        "l2_sheer_x": torch.mean(l2[:, 1]),
        "l2_sheer_y": torch.mean(l2[:, 2]),
        "l2_sheer_z": torch.mean(l2[:, 3]),
    }

    return metrics


def metrics_fn_surface_pressure(pred, target):
    return torch.mean((pred - target) ** 2.0)
