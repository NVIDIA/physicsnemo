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

from typing import Literal

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.core.module import Module
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


class SinusoidalTimestepEmbedding(Module):
    r"""Sinusoidal embedding for timesteps (e.g. for modulation / diffusion).

    For input :math:`x` (timestep) and :math:`D =` ``num_channels`` (even), the
    frequencies are :math:`\omega_k = k\pi` for :math:`k = 1, \ldots, D/2`, and
    the output is the concatenation of cosines and sines:

    .. math::

        \mathrm{embed}(x) = \big[ \cos(x \omega_1), \ldots, \cos(x \omega_{D/2}),
        \sin(x \omega_1), \ldots, \sin(x \omega_{D/2}) \big] \in \mathbb{R}^D.

    This is a simpler scheme than :class:`PositionalEmbedding` (which uses
    geometric frequencies and optional learnable MLP) which works well for timestep
    conditioning in ModAFNO model.

    Parameters
    ----------
    num_channels : int
        Number of output channels. Must be even.

    Forward
    -------
    x : torch.Tensor
        Input tensor, shape ``B ...`` (e.g. :math:`(B,)` or :math:`(B, 1)`),
        containing timesteps.

    Outputs
    -------
    torch.Tensor
        Output tensor, shape ``B D`` with :math:`D =` ``num_channels``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.nn.module.embedding_layers import SinusoidalTimestepEmbedding
    >>> emb = SinusoidalTimestepEmbedding(num_channels=64)
    >>> t = torch.tensor([0.0, 0.5, 1.0])
    >>> out = emb(t)
    >>> out.shape
    torch.Size([3, 64])

    See Also
    --------
    :class:`PositionalEmbedding` :
        DDPM/ADM-style positional embedding (geometric frequencies, optional MLP).
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        freqs = torch.pi * torch.arange(
            start=1, end=self.num_channels // 2 + 1, dtype=torch.float32
        )
        self.register_buffer("freqs", freqs)

    def forward(self, x: Float[Tensor, "B ..."]) -> Float[Tensor, "B D"]:
        r"""Forward pass computing sinusoidal embeddings."""
        x = x.view(-1).outer(self.freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class OneHotEmbedding(Module):
    r"""Soft one-hot embedding for normalized timesteps in :math:`[0, 1]`.

    The :math:`i`-th channel is :math:`\max(0, 1 - |t(D-1) - i|)`, giving
    a soft one-hot vector of dimension :math:`D`. Used for timestep conditioning
    in ModAFNO when ``method="onehot"``.

    Parameters
    ----------
    num_channels : int
        Number of channels (embedding dimension).

    Forward
    -------
    t : torch.Tensor
        Input tensor, shape ``B ...``, with normalized timesteps in ``[0, 1]``.

    Outputs
    -------
    torch.Tensor
        Output tensor, shape ``B D``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.nn.module.embedding_layers import OneHotEmbedding
    >>> emb = OneHotEmbedding(num_channels=64)
    >>> t = torch.tensor([[0.0], [0.5], [1.0]])
    >>> out = emb(t)
    >>> out.shape
    torch.Size([3, 64])
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        ind = torch.arange(num_channels, dtype=torch.float32)
        self.register_buffer("indices", ind.view(1, -1))

    def forward(self, t: Float[Tensor, "B ..."]) -> Float[Tensor, "B D"]:
        r"""Forward pass computing soft one-hot embeddings."""
        ind = (t.view(-1, 1) * (self.num_channels - 1)).to(self.indices.dtype)
        return torch.clamp(1 - torch.abs(ind - self.indices), min=0)
