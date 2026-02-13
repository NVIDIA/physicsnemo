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

from typing import Type

from jaxtyping import Float
from torch import Tensor, nn

from physicsnemo.core.module import Module
from physicsnemo.nn.module.embedding_layers import (
    OneHotEmbedding,
    SinusoidalTimestepEmbedding,
)

# Backward compatibility: modembed used "PositionalEmbedding" for the sinusoidal
# timestep embedding (distinct from nn.embedding_layers.PositionalEmbedding).
PositionalEmbedding = SinusoidalTimestepEmbedding


class ModEmbedNet(Module):
    r"""Network that generates a timestep embedding and processes it with an MLP.

    Parameters
    ----------
    max_time : float, optional, default=1.0
        Maximum input time. The inputs to ``forward`` should be in the range
        ``[0, max_time]``.
    dim : int, optional, default=64
        The dimensionality of the time embedding.
    depth : int, optional, default=1
        The number of layers in the MLP.
    activation_fn : Type[nn.Module], optional, default=nn.GELU
        The activation function class.
    method : str, optional, default="sinusoidal"
        The embedding method. Either ``"sinusoidal"`` or ``"onehot"``.

    Forward
    -------
    t : torch.Tensor
        Input tensor, shape ``B ...`` (e.g. :math:`(B,)` or :math:`(B, 1)`),
        containing timesteps in range ``[0, max_time]``.

    Outputs
    -------
    torch.Tensor
        Output tensor, shape ``B D``, where :math:`D` is ``dim``.

    Examples
    --------
    >>> import torch
    >>> embed_net = ModEmbedNet(max_time=1.0, dim=64, depth=2)
    >>> t = torch.tensor([0.0, 0.5, 1.0])
    >>> embedding = embed_net(t)
    >>> embedding.shape
    torch.Size([3, 64])

    See Also
    --------
    :mod:`~physicsnemo.nn.module.embedding_layers` :
        Embedding layers; this module uses
        :class:`~physicsnemo.nn.module.embedding_layers.SinusoidalTimestepEmbedding`
        and :class:`~physicsnemo.nn.module.embedding_layers.OneHotEmbedding`.
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
            self.sinusoid_embed = SinusoidalTimestepEmbedding(dim)
        else:
            raise ValueError(f"Embedding '{method}' not supported")

        self.dim = dim

        blocks = []
        for _ in range(depth):
            blocks.extend([nn.Linear(dim, dim), activation_fn()])
        self.mlp = nn.Sequential(*blocks)

    def forward(self, t: Float[Tensor, "B ..."]) -> Float[Tensor, "B D"]:
        r"""Forward pass computing the modulation embedding."""
        # Normalize time to [0, 1]
        t = t / self.max_time

        # Compute base embedding
        if self.method == "onehot":
            emb = self.onehot_embed(t)
        elif self.method == "sinusoidal":
            emb = self.sinusoid_embed(t)

        # Process through MLP
        return self.mlp(emb)
