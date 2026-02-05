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

from typing import Tuple

import torch
from torch import nn, Tensor


class GeometricL2Loss(nn.Module):
    """L2 loss on a lat-lon grid where the loss is computed over the sphere
    i.e. the errors are weighted by cos(lat).
    """

    def __init__(
        self,
        lat_range: Tuple[int, int] = (-90, 90),
        num_lats: int = 721,
        lat_indices_used: Tuple[int, int] = (0, 720),
        input_dims: int = 4,
    ):
        super().__init__()

        lats = torch.linspace(lat_range[0], lat_range[1], num_lats)
        dlat = lats[1] - lats[0]
        lats[0] = _correct_lat_at_pole(lats[0], dlat)
        lats[-1] = _correct_lat_at_pole(lats[-1], dlat)
        lats = torch.deg2rad(lats[lat_indices_used[0] : lat_indices_used[1]])
        weights = torch.cos(lats)
        weights = weights / torch.sum(weights)
        weights = torch.reshape(
            weights,
            (1,) * (input_dims - 2) + (lat_indices_used[1] - lat_indices_used[0], 1),
        )
        self.register_buffer("weights", weights)

    def forward(self, pred: Tensor, true: Tensor) -> Tensor:
        err = torch.square(pred - true)
        err = torch.sum(err * self.weights, dim=-2)
        return torch.mean(err)


def _correct_lat_at_pole(lat, dlat):
    """Adjust latitude at the poles to avoid zero weight at pole."""
    correction = dlat / 4
    if lat == 90:
        lat -= correction
    elif lat == -90:
        lat += correction
    return lat
