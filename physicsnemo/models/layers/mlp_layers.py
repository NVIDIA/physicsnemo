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
from torch import nn

from .activations import get_activation


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | list[int] | None = None,
        out_features: int | None = None,
        act_layer: nn.Module | str = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        if isinstance(hidden_features, int):
            hidden_features = [
                hidden_features,
            ]
        elif hidden_features is None:
            hidden_features = [
                in_features,
            ]

        # If the activation is a string, get it.
        # If it's a type, instantiate it.
        # If it's a module, leave it be.
        if isinstance(act_layer, str):
            act_layer = get_activation(act_layer)
        elif isinstance(act_layer, nn.Module):
            pass
        else:
            act_layer = act_layer()
            if not isinstance(act_layer, nn.Module):
                raise ValueError(
                    f"Activation layer must be a string or a module, got {type(act_layer)}"
                )

        layers = []
        input_dim = in_features
        for hidden_dim in hidden_features:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act_layer)
            if drop != 0:
                layers.append(nn.Dropout(drop))
            input_dim = hidden_dim

        # Add the last layers:
        layers.append(nn.Linear(input_dim, out_features))
        if drop != 0:
            layers.append(nn.Dropout(drop))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)
