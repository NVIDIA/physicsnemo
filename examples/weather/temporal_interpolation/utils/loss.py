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
from torch import nn


class GeometricL2Loss(nn.Module):
    """Latitude weighted L2 (MSE) loss.

    Parameters
    ----------
    lat_range : tuple[float, float], optional
        Range of latitudes to cover, by default (-90.0, 90.0)
    num_lats : int, optional
        Number of latitudes in lat_range, by default 721
    num_lats_cropped : int, optional
        Use the first num_lats_cropped latitudes, by default 720
    input_dims : int, optional
        Number of dimensions in the input tensors passed to `forward`, by
        default 4.

    Forward
    -------
    pred : torch.Tensor
        Predicted values, shape (..., num_lats_cropped, num_lons),
        number of dimensions must equal ``input_dims``
    true : torch.Tensor
        True values, shape equal to pred

    Outputs
    -------
    torch.Tensor
        The computed loss
    """

    def __init__(
        self,
        lat_range: tuple[float, float] = (-90.0, 90.0),
        num_lats: int = 721,
        num_lats_cropped: int = 720,
        input_dims: int = 4,
    ):
        super().__init__()

        lats = torch.linspace(lat_range[0], lat_range[1], num_lats)
        if lat_range[0] == -90:  # special handling for poles
            lats[0] = 0.5 * (lats[0] + lats[1])
        if lat_range[1] == 90:
            lats[-1] = 0.5 * (lats[-2] + lats[-1])
        lats = torch.deg2rad(lats[:num_lats_cropped])
        weights = torch.cos(lats)
        weights = weights / torch.sum(weights)
        weights = torch.reshape(
            weights, (1,) * (input_dims - 2) + (num_lats_cropped, 1)
        )
        self.register_buffer("weights", weights)

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        if not (pred.ndim == true.ndim == self.weights.ndim):
            raise ValueError(
                "Shape mismatch: pred, true and weights must have the same number of dimensions."
            )
        if pred.shape != true.shape:
            raise ValueError("Shape mismatch: pred and true must have the same shape")
        err = torch.square(pred - true)
        err = torch.sum(err * self.weights, dim=-2)
        return torch.mean(err)
