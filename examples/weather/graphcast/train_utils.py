# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
import numpy as np
from physicsnemo.models.graphcast.utils.graph_utils import deg2rad


def normalized_grid_cell_area(lat: Tensor, unit="deg") -> Tensor:
    """Normalized area of the latitude-longitude grid cell"""
    if unit == "deg":
        lat = deg2rad(lat)
    area = torch.abs(torch.cos(lat))
    return area / torch.mean(area)


def prepare_input(
    invar,
    cos_zenith=None,
    num_history=0,
    static_data=None,
    step=None,
    time_idx=None,
    stride=1,
    dt=6.0,
    num_samples_per_year=1459,
    device="cuda",
):
    """Prepare input by adding history, cos zenith angle, and static data, if applicable"""

    # Add history
    if num_history > 0:
        # flatten the history dimension
        invar = invar.view(invar.size(0), -1, *(invar.size()[3:]))

    # Add cos zenith
    if cos_zenith is not None:
        cos_zenith = torch.squeeze(cos_zenith, dim=2)
        cos_zenith = torch.clamp(cos_zenith, min=0.0) - 1.0 / np.pi
        invar = torch.concat(
            (invar, cos_zenith[:, step - 1 : num_history + step, ...]), dim=1
        )

    # Add static data
    if static_data is not None:
        invar = torch.concat((invar, static_data), dim=1)

    # Add clock variables
    if time_idx is not None:
        # Precompute the tensors to concatenate
        sin_day_of_year = torch.zeros(1, num_history + 1, 721, 1440, device=device)
        cos_day_of_year = torch.zeros(1, num_history + 1, 721, 1440, device=device)
        sin_time_of_day = torch.zeros(1, num_history + 1, 721, 1440, device=device)
        cos_time_of_day = torch.zeros(1, num_history + 1, 721, 1440, device=device)

        for i in range(num_history + 1):
            # Calculate the adjusted time index
            adjusted_time_idx = (time_idx - i) % num_samples_per_year

            # Compute hour of the year and its decomposition into day of year and time of day
            hour_of_year = adjusted_time_idx * stride * dt
            day_of_year = hour_of_year // 24
            time_of_day = hour_of_year % 24

            # Normalize to the range [0, pi/2]
            normalized_day_of_year = torch.tensor(
                (day_of_year / 365) * (np.pi / 2), dtype=torch.float32, device=device
            )
            normalized_time_of_day = torch.tensor(
                (time_of_day / (24 - dt)) * (np.pi / 2),
                dtype=torch.float32,
                device=device,
            )

            # Fill the tensors for the current step
            sin_day_of_year[0, i] = torch.sin(normalized_day_of_year)
            cos_day_of_year[0, i] = torch.cos(normalized_day_of_year)
            sin_time_of_day[0, i] = torch.sin(normalized_time_of_day)
            cos_time_of_day[0, i] = torch.cos(normalized_time_of_day)

        # Concatenate the new channels to invar
        invar = torch.cat(
            (invar, sin_day_of_year, cos_day_of_year, sin_time_of_day, cos_time_of_day),
            dim=1,
        )

    return invar


def count_trainable_params(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): Model to count parameters of.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
