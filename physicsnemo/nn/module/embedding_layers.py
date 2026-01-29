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

from typing import Literal

import numpy as np
import torch

from physicsnemo.nn.module.utils.utils import _validate_amp


class FourierEmbedding(torch.nn.Module):
    """
    Generates Fourier embeddings for timesteps, primarily used in the NCSN++
    architecture.

    This class generates embeddings by first multiplying input tensor `x` and
    internally stored random frequencies, and then concatenating the cosine and sine of
    the resultant.

    Parameters:
    -----------
    num_channels : int
        The number of channels in the embedding. The final embedding size will be
        2 * num_channels because of concatenation of cosine and sine results.
    scale : int, optional
        A scale factor applied to the random frequencies, controlling their range
        and thereby the frequency of oscillations in the embedding space. By default 16.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled. Defaults to False.
    """

    def __init__(self, num_channels: int, scale: int = 16, amp_mode: bool = False):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)
        self.amp_mode = amp_mode

    def forward(self, x):
        freqs = self.freqs
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if x.dtype != self.freqs.dtype:
                freqs = self.freqs.to(x.dtype)

        x = x.ger((2 * np.pi * freqs))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class PositionalEmbedding(torch.nn.Module):
    """
    A module for generating positional embeddings based on timesteps.
    This embedding technique is employed in the DDPM++ and ADM architectures.

    Parameters:
    -----------
    num_channels : int
        Number of channels for the embedding.
    max_positions : int, optional
        Maximum number of positions for the embeddings, by default 10000.
    endpoint : bool, optional
        If True, the embedding considers the endpoint. By default False.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled. Defaults to False.
    learnable : bool, optional
        A boolean flag indicating whether learnable positional embedding is enabled. Defaults to False.
    freq_embed_dim: int, optional
        The dimension of the frequency embedding. Defaults to None, in which case it will be set to num_channels.
    mlp_hidden_dim: int, optional
        The dimension of the hidden layer in the MLP. Defaults to None, in which case it will be set to 2 * num_channels.
        Only applicable if learnable is True; if learnable is False, this parameter is ignored.
    embed_fn: Literal["cos_sin", "np_sin_cos"], optional
        The function to use for embedding into sin/cos features (allows for swapping the order of sin/cos). Defaults to 'cos_sin'.
        Options:
            - 'cos_sin': Uses torch to compute frequency embeddings and returns in order (cos, sin)
            - 'np_sin_cos': Uses numpy to compute frequency embeddings and returns in order (sin, cos)
    """

    def __init__(
        self,
        num_channels: int,
        max_positions: int = 10000,
        endpoint: bool = False,
        amp_mode: bool = False,
        learnable: bool = False,
        freq_embed_dim: int | None = None,
        mlp_hidden_dim: int | None = None,
        embed_fn: Literal["cos_sin", "np_sin_cos"] = "cos_sin",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.amp_mode = amp_mode
        self.learnable = learnable
        self.embed_fn = embed_fn

        if freq_embed_dim is None:
            freq_embed_dim = num_channels
        self.freq_embed_dim = freq_embed_dim

        if learnable:
            if mlp_hidden_dim is None:
                mlp_hidden_dim = 2 * num_channels
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(freq_embed_dim, mlp_hidden_dim, bias=True),
                torch.nn.SiLU(),
                torch.nn.Linear(mlp_hidden_dim, num_channels, bias=True),
            )

        freqs = torch.arange(start=0, end=self.freq_embed_dim // 2, dtype=torch.float32)
        freqs = freqs / (self.freq_embed_dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x):
        x = torch.outer(x, self.freqs)

        if self.embed_fn == "cos_sin":
            x = torch.cat([x.cos(), x.sin()], dim=1)
        elif self.embed_fn == "np_sin_cos":
            x = torch.cat([x.sin(), x.cos()], dim=1)

        if self.learnable:
            x = self.mlp(x)
        return x
