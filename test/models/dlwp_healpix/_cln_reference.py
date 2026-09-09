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

# Reference (old) implementation of ConditionalLayerNorm for testing.
# This is a copy of the original code before optimization.

import copy
from typing import List

import torch

try:
    from apex.normalization import FusedLayerNorm

    _APEX_AVAILABLE = True
except ImportError:
    _APEX_AVAILABLE = False


class ConditionalLayerNormReference(torch.nn.Module):
    def __init__(
        self,
        condition_shape: int,
        channel_depth: int,
        mlp_hidden_dims: List[int] = [128, 128],
        activation: torch.nn.Module = None,
        eps: float = 1e-5,
        n_faces: int = 12,
        norm_op: str = "torch",
        init_cln_to_zero: bool = False,
        scale_center: float = 0.0,
    ):
        super().__init__()
        self.eps = eps
        self.condition_shape = condition_shape
        self.channel_depth = channel_depth
        self.hidden_dims = mlp_hidden_dims
        self.activation = activation if activation is not None else torch.nn.Identity()
        self.gamma_mlp = self._make_mlp(
            self.condition_shape, self.hidden_dims, self.channel_depth, self.activation
        )
        self.beta_mlp = self._make_mlp(
            self.condition_shape, self.hidden_dims, self.channel_depth, self.activation
        )
        self.n_faces = n_faces
        self.scale_center = scale_center

        if init_cln_to_zero:
            self.gamma_mlp[-1].weight.data.zero_()
            self.beta_mlp[-1].weight.data.zero_()
            self.gamma_mlp[-1].bias.data.zero_()
            self.beta_mlp[-1].bias.data.zero_()

        if norm_op == "torch":
            self.norm = torch.nn.LayerNorm(channel_depth, elementwise_affine=False)
        elif norm_op == "apex":
            if not _APEX_AVAILABLE:
                raise ImportError(
                    "Apex FusedLayerNorm requested but apex is not available"
                )
            self.norm = FusedLayerNorm(channel_depth, elementwise_affine=False)

    def _make_mlp(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        activation: torch.nn.Module,
    ) -> torch.nn.Sequential:
        layers = []
        for hdim in hidden_dims:
            layers.append(torch.nn.Linear(in_dim, hdim))
            if activation:
                # some variations of _make_mlp may have duplicate activation submodule buffers (e.g. CappedGELU ``cap``)
                # so we need to deepcopy the activation submodule buffers
                layers.append(copy.deepcopy(activation))
            in_dim = hdim
        layers.append(torch.nn.Linear(in_dim, out_dim))
        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x)

        gamma = self.scale_center + self.gamma_mlp(conditions)[:, None, None, :]
        beta = self.beta_mlp(conditions)[:, None, None, :]

        gamma = gamma.repeat_interleave(self.n_faces, dim=0)
        beta = beta.repeat_interleave(self.n_faces, dim=0)

        x = gamma * x_norm + beta
        return x.permute(0, 3, 1, 2)
