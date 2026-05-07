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

r"""Patch and positional embedding modules for the Cross_Unet PV-power model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    r"""Sinusoidal positional embedding with a fixed maximum length.

    Computes the standard transformer sin/cos positional encoding once at
    construction and exposes the prefix matching the input length on each
    forward call.

    Parameters
    ----------
    d_model : int
        Embedding dimension :math:`D`.
    max_len : int, optional, default=5000
        Maximum sequence length supported by the cached table.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    r"""Per-channel patch embedding for time-series.

    Splits the input along the time axis into overlapping patches with the
    given ``patch_len`` and ``stride`` (after replication padding), then
    projects each patch to ``d_model`` and adds a sinusoidal positional
    embedding.

    Parameters
    ----------
    d_model : int
        Output embedding dimension :math:`D`.
    patch_len : int
        Patch length along the time axis.
    stride : int
        Stride between consecutive patches.
    padding : int
        Trailing replication padding applied before patching.
    dropout : float
        Dropout probability applied to the embedded patches.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, L)` with channel dimension first.

    Outputs
    -------
    torch.Tensor
        Patch embeddings of shape :math:`(B \cdot C, P, D)` where
        :math:`P` is the number of patches.
    int
        The original number of channels :math:`C`, returned so that
        downstream code can rearrange the merged batch-channel dimension.
    """

    def __init__(
        self,
        d_model: int,
        patch_len: int,
        stride: int,
        padding: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
