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

import enum
import math

import torch
import torch.nn as nn
from torch import Tensor


class SirenLayerType(enum.Enum):
    """
    SiReN layer types.
    """

    FIRST = enum.auto()
    HIDDEN = enum.auto()
    LAST = enum.auto()


class SirenLayer(nn.Module):
    """
    SiReN layer.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    layer_type : SirenLayerType
        Layer type.
    omega_0 : float
        Omega_0 parameter in SiReN.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_type: SirenLayerType = SirenLayerType.HIDDEN,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.layer_type = layer_type
        self.omega_0 = omega_0

        self.linear = nn.Linear(in_features, out_features, bias=True)

        self.apply_activation = layer_type in {
            SirenLayerType.FIRST,
            SirenLayerType.HIDDEN,
        }

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset layer parameters."""
        weight_ranges = {
            SirenLayerType.FIRST: 1.0 / self.in_features,
            SirenLayerType.HIDDEN: math.sqrt(6.0 / self.in_features) / self.omega_0,
            SirenLayerType.LAST: math.sqrt(6.0 / self.in_features),
        }
        weight_range = weight_ranges[self.layer_type]
        nn.init.uniform_(self.linear.weight, -weight_range, weight_range)

        k_sqrt = math.sqrt(1.0 / self.in_features)
        nn.init.uniform_(self.linear.bias, -k_sqrt, k_sqrt)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.apply_activation:
            x = torch.sin(self.omega_0 * x)
        return x
