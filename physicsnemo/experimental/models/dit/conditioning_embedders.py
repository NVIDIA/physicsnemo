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

"""Conditioning embedders for DiT models."""

import math
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

from physicsnemo.core import Module

from .layers import Linear, PositionalEmbedding


@runtime_checkable
class ConditioningEmbedder(Protocol):
    r"""Protocol for conditioning embedders used in DiT.

    Computes conditioning embedding from timestep and optional additional inputs.
    Implementations must define a forward method and specify their output dimension.

    Forward
    -------
    t : torch.Tensor
        Timestep tensor of shape :math:`(B,)`.
    **kwargs
        Additional conditioning inputs (e.g., condition, class_labels).

    Outputs
    -------
    torch.Tensor
        Conditioning embedding of shape :math:`(B, D)` where :math:`D` is ``output_dim``.
    """

    @property
    def output_dim(self) -> int:
        r"""Output dimension of conditioning embedding."""
        ...

    def forward(self, t: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Compute conditioning embedding from timestep and optional inputs."""
        ...


class ZeroConditioningEmbedder(Module):
    r"""Zero conditioning embedder for unconditional/deterministic models (``condition_dim=0``).

    Returns empty tensors of shape :math:`(B, 0)` for conditioning, allowing
    AdaLN blocks to operate in bias-only mode (0 × D weight + D bias).
    This is useful when a deterministic model which uses constant timestep/condition values
    is trained using the DiT-style adaptive layer norm mechanism. In this case, the MLP weight matrix
    can be folded into a fixed bias parameter to reduce parameters at inference.

    Parameters
    ----------
    None
        No constructor parameters.

    Forward
    -------
    t : torch.Tensor
        Timestep tensor of shape :math:`(B,)`. Unused; provided for interface compatibility.
    **kwargs
        Additional conditioning inputs. Ignored.

    Outputs
    -------
    torch.Tensor
        Empty conditioning tensor of shape :math:`(B, 0)`.
    """

    def __init__(self):
        super().__init__()
        self._output_dim = 0

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.empty(t.shape[0], 0, device=t.device, dtype=t.dtype)


class DiTConditionEmbedder(Module):
    r"""DiT-style conditioning embedder.

    Processes timestep and condition independently, then adds them together at the end.

    Parameters
    ----------
    hidden_size : int
        Output embedding dimension, matching DiT hidden_size.
    condition_dim : int, optional, default=0
        Input condition dimension. If 0, no condition embedding is used.
    amp_mode : bool, optional, default=False
        Whether mixed-precision (AMP) training is enabled.
    **timestep_embed_kwargs
        Keyword arguments passed to :class:`physicsnemo.nn.PositionalEmbedding`
        for the timestep embedding.

    Forward
    -------
    t : torch.Tensor
        Timestep tensor of shape :math:`(B,)`.
    condition : torch.Tensor, optional
        Condition tensor of shape :math:`(B, \text{condition\_dim})`.

    Outputs
    -------
    torch.Tensor
        Conditioning embedding of shape :math:`(B, \text{hidden\_size})`.
    """

    def __init__(
        self,
        hidden_size: int,
        condition_dim: int = 0,
        amp_mode: bool = False,
        **timestep_embed_kwargs: Any,
    ):
        super().__init__()
        self._output_dim = hidden_size

        self.t_embedder = PositionalEmbedding(
            num_channels=hidden_size,
            amp_mode=amp_mode,
            learnable=True,
            **timestep_embed_kwargs,
        )

        self.cond_embedder = (
            Linear(
                in_features=condition_dim,
                out_features=hidden_size,
                bias=False,
                amp_mode=amp_mode,
                init_mode="kaiming_uniform",
                init_weight=0,
                init_bias=0,
            )
            if condition_dim
            else None
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self, t: torch.Tensor, condition: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        c = self.t_embedder(t)

        if self.cond_embedder is not None and condition is not None:
            c = c + self.cond_embedder(condition)

        return c


class EDMConditionEmbedder(Module):
    r"""EDM/SongUNet-style conditioning embedder.

    Combines timestep and condition before the MLP.

    Parameters
    ----------
    emb_channels : int
        Output embedding dimension (typically 4 * hidden_size).
    noise_channels : int
        Dimension of positional embedding for the noise/timestep label.
    condition_dim : int, optional, default=0
        Input condition dimension. If 0, no condition embedding.
    condition_dropout : float, optional, default=0.0
        Dropout probability for conditions during training.
    legacy_condition_bias : bool, optional, default=False
        If ``True``, includes a bias term even when ``condition_dim`` is 0.
    max_positions : int, optional, default=10000
        Maximum positions for positional embedding.

    Forward
    -------
    t : torch.Tensor
        Timestep/noise_labels tensor of shape :math:`(B,)`.
    condition : torch.Tensor, optional
        Condition tensor of shape :math:`(B, \text{condition\_dim})`.

    Outputs
    -------
    torch.Tensor
        Conditioning embedding of shape :math:`(B, \text{emb\_channels})`.
    """

    def __init__(
        self,
        emb_channels: int,
        noise_channels: int,
        condition_dim: int = 0,
        condition_dropout: float = 0.0,
        legacy_condition_bias: bool = False,
        max_positions: int = 10000,
        **kwargs, # ignore extra kwargs
    ):
        super().__init__()
        self._output_dim = emb_channels
        self.condition_dropout = condition_dropout
        self.legacy_condition_bias = legacy_condition_bias

        self.map_noise = PositionalEmbedding(
            num_channels=noise_channels,
            max_positions=max_positions,
            endpoint=True,
            learnable=False,  # No MLP here - added below
            embed_fn="np_sin_cos",
        )

        # Condition embedding (added before MLP)
        self.map_condition = None
        if condition_dim > 0 or legacy_condition_bias:
            self.map_condition = nn.Linear(condition_dim, noise_channels)

        # MLP: Linear → SiLU → Linear (no final SiLU - moved to AdaLN)
        self.map_layer0 = nn.Linear(noise_channels, emb_channels)
        self.map_layer1 = nn.Linear(emb_channels, emb_channels)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self, t: torch.Tensor, condition: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        emb = self.map_noise(t)

        # Add condition embedding before final MLP
        if self.map_condition is not None and condition is not None:
            tmp = condition
            if self.training and self.condition_dropout:
                tmp = tmp * (
                    torch.rand([t.shape[0], 1], device=tmp.device)
                    >= self.condition_dropout
                ).to(tmp.dtype)
            emb = emb + self.map_condition(tmp * math.sqrt(self.map_condition.in_features))

        # MLP
        emb = torch.nn.functional.silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)

        return emb


class ConditioningEmbedderType(Enum):
    r"""Conditioning embedder types for DiT models.

    Used by :func:`~physicsnemo.experimental.models.dit.conditioning_embedders.get_conditioning_embedder`
    to select the conditioning embedder implementation.
    """

    DIT = DiTConditionEmbedder
    EDM = EDMConditionEmbedder
    ZERO = ZeroConditioningEmbedder


def get_conditioning_embedder(
    conditioning_embedder: ConditioningEmbedderType = ConditioningEmbedderType.DIT,
    **kwargs: Any,
) -> ConditioningEmbedder:
    r"""Factory function to create conditioning embedders.

    Parameters
    ----------
    conditioning_embedder : ConditioningEmbedderType
        The type of conditioning embedder to use.
        Options:
            - DIT: DiT-style, maps timestep and condition independently (late fusion).
            - EDM: EDM/SongUNet-style, combines timestep and condition before MLP (early fusion).
            - ZERO: Returns empty (B, 0) tensors for bias-only AdaLN (unconditional/ViT-style inference).
    **kwargs
        Keyword arguments passed to the embedder constructor.
        See :class:`~physicsnemo.experimental.models.dit.conditioning_embedders.DiTConditionEmbedder` or
        :class:`~physicsnemo.experimental.models.dit.conditioning_embedders.EDMConditionEmbedder` for available options.

    Returns
    -------
    ConditioningEmbedder
        An instance implementing the :class:`~physicsnemo.experimental.models.dit.conditioning_embedders.ConditioningEmbedder` protocol.
    """
    if conditioning_embedder == ConditioningEmbedderType.ZERO:
        return conditioning_embedder.value()
    else:
        return conditioning_embedder.value(**kwargs)
