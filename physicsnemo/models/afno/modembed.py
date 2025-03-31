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

from typing import Type

import torch
from torch import Tensor, nn


class PositionalEmbedding(nn.Module):
    """
    A module for generating positional embeddings based on timesteps.

    Parameters:
    -----------
    num_channels : int
        Number of channels for the embedding.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

        freqs = torch.pi * torch.arange(
            start=1, end=self.num_channels // 2 + 1, dtype=torch.float32
        )
        self.register_buffer("freqs", freqs)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1).outer(self.freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class OneHotEmbedding(nn.Module):
    """
    A module for generating one-hot embeddings based on timesteps.

    Parameters:
    -----------
    num_channels : int
        Number of channels for the embedding.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        ind = torch.arange(num_channels)
        ind = ind.view(1, len(ind))
        self.register_buffer("indices", ind)

    def forward(self, t: Tensor) -> Tensor:
        ind = t * (self.num_channels - 1)
        return torch.clamp(1 - torch.abs(ind - self.indices), min=0)


class ModEmbedNet(nn.Module):
    """
    A network that generates a timestep embedding and processes it with an MLP.

    Parameters:
    -----------
    max_time : float, optional
        Maximum input time. The inputs to `forward` is should be in the range [0, max_time].
    dim : int, optional
        The dimensionality of the time embedding.
    depth : int, optional
        The number of layers in the MLP.
    activation_fn:
        The activation function, default GELU.
    method : str, optional
        The embedding method. Either "sinusoidal" (default) or "onehot".
    """

    def __init__(
        self,
        max_time: float = 1.0,
        dim: int = 64,
        depth: int = 1,
        activation_fn: Type[nn.Module] = nn.GELU,
        method: str = "sinusoidal",
    ):
        super().__init__()
        self.max_time = max_time
        self.method = method
        if method == "onehot":
            self.onehot_embed = OneHotEmbedding(dim)
        elif method == "sinusoidal":
            self.sinusoid_embed = PositionalEmbedding(dim)
        else:
            raise ValueError(f"Embedding '{method}' not supported")

        self.dim = dim

        blocks = []
        for _ in range(depth):
            blocks.extend([nn.Linear(dim, dim), activation_fn()])
        self.mlp = nn.Sequential(*blocks)

    def forward(self, t: Tensor) -> Tensor:
        t = t / self.max_time
        if self.method == "onehot":
            emb = self.onehot_embed(t)
        elif self.method == "sinusoidal":
            emb = self.sinusoid_embed(t)

        return self.mlp(emb)
