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

"""Pooling layers for aggregating variable-length point/token sequences into
fixed-size embeddings.

These modules reduce a sequence of per-point features ``(B, N, C)`` to a
single embedding ``(B, D)`` per batch element.  Two strategies are provided:

* :class:`AttentionPooling` — learns per-point importance weights before
  aggregating, followed by a small projector MLP.
* :class:`MeanPooling` — simple mean over the sequence dimension, followed by
  a single linear projection.

Both support optional L2 normalization of the output embedding (useful for
downstream models like Gaussian processes that are sensitive to embedding
magnitude) and optional spectral normalization of their linear layers (useful
for distance-preserving / SNGP-style pipelines).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.nn.module.activations import get_activation


def _maybe_spectral_norm(layer: nn.Module, enable: bool) -> nn.Module:
    """Optionally wrap *layer* with spectral normalization."""
    if enable:
        return torch.nn.utils.parametrizations.spectral_norm(layer)
    return layer


class AttentionPooling(nn.Module):
    """Learnable attention-weighted pooling over a variable-length sequence.

    Computes per-element importance scores via a small attention network,
    aggregates the weighted sum, and projects it to a fixed-size embedding.

    Parameters
    ----------
    feat_dim : int, optional
        Dimension of each input token/point feature. Default is 256.
    embed_dim : int, optional
        Output embedding dimension. Default is 32.
    hidden : int, optional
        Hidden dimension of the attention scoring network. Default is 128.
    projector_hidden : list[int] | None, optional
        Hidden layer sizes for the projector MLP.  Each hidden layer is
        followed by *activation* and ``LayerNorm``.  Default is
        ``[256, 128]``.
    activation : str, optional
        Activation function name passed to
        :func:`~physicsnemo.nn.module.activations.get_activation`.
        Default is ``"relu"``.
    spectral_norm : bool, optional
        If ``True``, apply spectral normalization to all linear layers.
        Default is ``False``.
    normalize : bool, optional
        If ``True``, L2-normalize the output embedding. Default is ``False``.
    target_scale : float, optional
        Radius of the embedding sphere when ``normalize=True``. Ignored
        when ``normalize=False``. Default is 1.0.

    Examples
    --------
    >>> pool = AttentionPooling(feat_dim=64, embed_dim=16)
    >>> x = torch.randn(2, 1000, 64)
    >>> pool(x).shape
    torch.Size([2, 16])
    """

    def __init__(
        self,
        feat_dim: int = 256,
        embed_dim: int = 32,
        hidden: int = 128,
        projector_hidden: list[int] | None = None,
        activation: str = "relu",
        spectral_norm: bool = False,
        normalize: bool = False,
        target_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if projector_hidden is None:
            projector_hidden = [256, 128]

        sn = spectral_norm
        self.attention = nn.Sequential(
            _maybe_spectral_norm(nn.Linear(feat_dim, hidden), sn),
            nn.Tanh(),
            _maybe_spectral_norm(nn.Linear(hidden, 1), sn),
        )

        proj_layers: list[nn.Module] = []
        in_dim = feat_dim
        for h in projector_hidden:
            proj_layers.append(_maybe_spectral_norm(nn.Linear(in_dim, h), sn))
            proj_layers.append(get_activation(activation))
            proj_layers.append(nn.LayerNorm(h))
            in_dim = h
        proj_layers.append(_maybe_spectral_norm(nn.Linear(in_dim, embed_dim), sn))
        self.projector = nn.Sequential(*proj_layers)

        self.normalize = normalize
        self.target_scale = target_scale

    def forward(
        self,
        point_feats: Float[torch.Tensor, "batch points feat_dim"],
    ) -> Float[torch.Tensor, "batch embed_dim"]:
        """Aggregate point features into a single embedding per batch element.

        Parameters
        ----------
        point_feats : torch.Tensor
            Input features of shape ``(B, N, feat_dim)``.

        Returns
        -------
        torch.Tensor
            Pooled embedding of shape ``(B, embed_dim)``.
        """
        attn_scores = self.attention(point_feats)
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_sum = (attn_weights * point_feats).sum(dim=1)
        out = self.projector(weighted_sum)
        if self.normalize:
            out = nn.functional.normalize(out, dim=-1) * self.target_scale
        return out


class MeanPooling(nn.Module):
    """Mean pooling over the sequence dimension followed by a linear projection.

    Parameters
    ----------
    feat_dim : int, optional
        Dimension of each input token/point feature. Default is 256.
    embed_dim : int, optional
        Output embedding dimension. Default is 32.
    spectral_norm : bool, optional
        If ``True``, apply spectral normalization to the projection layer.
        Default is ``False``.
    normalize : bool, optional
        If ``True``, L2-normalize the output embedding. Default is ``False``.
    target_scale : float, optional
        Radius of the embedding sphere when ``normalize=True``. Ignored
        when ``normalize=False``. Default is 1.0.

    Examples
    --------
    >>> pool = MeanPooling(feat_dim=64, embed_dim=16)
    >>> x = torch.randn(2, 1000, 64)
    >>> pool(x).shape
    torch.Size([2, 16])
    """

    def __init__(
        self,
        feat_dim: int = 256,
        embed_dim: int = 32,
        spectral_norm: bool = False,
        normalize: bool = False,
        target_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.projector = _maybe_spectral_norm(
            nn.Linear(feat_dim, embed_dim), spectral_norm
        )
        self.normalize = normalize
        self.target_scale = target_scale

    def forward(
        self,
        point_feats: Float[torch.Tensor, "batch points feat_dim"],
    ) -> Float[torch.Tensor, "batch embed_dim"]:
        """Aggregate point features via mean pooling and project.

        Parameters
        ----------
        point_feats : torch.Tensor
            Input features of shape ``(B, N, feat_dim)``.

        Returns
        -------
        torch.Tensor
            Pooled embedding of shape ``(B, embed_dim)``.
        """
        pooled = point_feats.mean(dim=1)
        out = self.projector(pooled)
        if self.normalize:
            out = nn.functional.normalize(out, dim=-1) * self.target_scale
        return out
