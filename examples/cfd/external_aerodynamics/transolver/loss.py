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


def loss_fn(pred, target):
    return loss_fn_surface(pred, target, "mse")


def loss_fn_surface(
    output: torch.Tensor, target: torch.Tensor, loss_type: Literal["mse", "rmse"]
) -> torch.Tensor:
    """Calculate loss for surface data by handling scalar and vector components separately.

    Args:
        output: Predicted surface values from the model
        target: Ground truth surface values
        loss_type: Type of loss to calculate ("mse" or "rmse")

    Returns:
        Combined scalar and vector loss as a scalar tensor
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
