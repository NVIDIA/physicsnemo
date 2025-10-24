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
import torch.nn as nn

from physicsnemo.models.mlp import FullyConnected


class MoEGatingNet(nn.Module):
    """
    MoE Gating Network
    """

    def __init__(
        self,
        hidden_dim=128,
        num_layers=3,
        num_experts=3,
        num_feature_per_expert=3,
        activation_fn="relu",
        use_moe_bias=True,
        include_normals=True,
    ):
        super().__init__()
        self.use_moe_bias = use_moe_bias
        out_features = num_experts + 1 if use_moe_bias else num_experts
        self.mlp = FullyConnected(
            in_features=num_feature_per_expert * num_experts + 3
            if include_normals
            else num_feature_per_expert * num_experts,
            layer_size=hidden_dim,
            out_features=out_features,
            num_layers=num_layers,
            activation_fn=activation_fn,
        )

    def forward(self, x):
        x = self.mlp(x)
        if self.use_moe_bias:
            # Split into expert weights and bias weight
            expert_weights = torch.softmax(x[..., :-1], dim=-1)
            bias = x[..., -1:]
            return torch.cat([expert_weights, bias], dim=-1)
        else:
            return torch.softmax(x, dim=-1)
