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
r"""Reusable adaLN-Zero pieces: the :class:`AdaLNModulation` projection and the
:func:`modulate` / :func:`gated_residual` helpers that apply it. Rank-agnostic,
so they serve 3D :math:`(B, L, C)` and 4D :math:`(B, T, X, C)` states alike."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float


def _broadcast(param: Float[torch.Tensor, "batch channels"], ndim: int) -> torch.Tensor:
    # (B, C) -> (B, 1, ..., 1, C) so a per-sample modulation broadcasts over a
    # hidden-state tensor of arbitrary rank (3D (B, L, C), 4D (B, T, X, C), ...).
    shape = (param.shape[0],) + (1,) * (ndim - 2) + (param.shape[1],)
    return param.view(shape)


def modulate(
    x_normed: Float[torch.Tensor, "batch ... hidden_size"],
    shift: Float[torch.Tensor, "batch hidden_size"],
    scale: Float[torch.Tensor, "batch hidden_size"],
) -> torch.Tensor:
    r"""Apply an adaLN ``(shift, scale)``: :math:`x (1 + \text{scale}) + \text{shift}`.

    Parameters
    ----------
    x_normed : torch.Tensor
        Layer-normed hidden states :math:`(B, \dots, C)`.
    shift, scale : torch.Tensor
        Per-sample modulation :math:`(B, C)`, broadcast to ``x_normed.ndim``.

    Returns
    -------
    torch.Tensor
        Modulated tensor, same shape as ``x_normed``.
    """
    return torch.addcmul(
        _broadcast(shift, x_normed.ndim),
        x_normed,
        1 + _broadcast(scale, x_normed.ndim),
    )


def gated_residual(
    residual: Float[torch.Tensor, "batch ... hidden_size"],
    branch_out: Float[torch.Tensor, "batch ... hidden_size"],
    gate: Float[torch.Tensor, "batch hidden_size"],
    drop_path: Optional[nn.Module] = None,
) -> torch.Tensor:
    r"""Gated residual add: :math:`\text{residual} + \text{gate} \cdot \text{branch\_out}`.

    Parameters
    ----------
    residual : torch.Tensor
        Residual stream :math:`(B, \dots, C)`.
    branch_out : torch.Tensor
        Sub-layer output to gate, same shape as ``residual``.
    gate : torch.Tensor
        Per-sample gate :math:`(B, C)`, broadcast to ``residual.ndim``.
    drop_path : nn.Module, optional, default=None
        Stochastic-depth module applied to the gate (drops the whole branch).

    Returns
    -------
    torch.Tensor
        Updated residual, same shape as ``residual``.
    """
    gate = _broadcast(gate, residual.ndim)
    if drop_path is not None:
        gate = drop_path(gate)
    return torch.addcmul(residual, gate, branch_out)


class AdaLNModulation(nn.Module):
    r"""adaLN-Zero conditioning projection: ``c -> 3 * n_blocks`` modulation tensors.

    Emits ``n_blocks`` ``(shift, scale, gate)`` triples via ``SiLU + Linear``, in
    flat ``[shift, scale, gate] * n_blocks`` order, so one projection can drive
    several sub-layers (e.g. attention + MLP with ``n_blocks=2``). Norms and the
    modulation are applied at the call site (:func:`modulate`,
    :func:`gated_residual`); conditioning must be pre-activation.

    Parameters
    ----------
    embedding_dim : int
        Channel dimension :math:`C` of the hidden states.
    condition_embed_dim : int
        Channel dimension of the conditioning embedding.
    n_blocks : int, optional, default=1
        Number of ``(shift, scale, gate)`` triples to emit.
    zero_init : bool, optional, default=True
        Zero-init the projection (adaLN-Zero) so each residual branch starts as
        identity.

    Forward
    -------
    c : torch.Tensor
        Conditioning embedding :math:`(B, D_c)`.

    Outputs
    -------
    Tuple[torch.Tensor, ...]
        ``3 * n_blocks`` tensors :math:`(B, C)`, in ``[shift, scale, gate] * n_blocks`` order.
    """

    def __init__(
        self,
        embedding_dim: int,
        condition_embed_dim: int,
        n_blocks: int = 1,
        zero_init: bool = True,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.zero_init = zero_init
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_embed_dim, 3 * n_blocks * embedding_dim, bias=True),
        )
        if zero_init:
            self.initialize_weights()

    def initialize_weights(self) -> None:
        r"""Zero the projection linear when ``zero_init`` is set (adaLN-Zero)."""
        if self.zero_init:
            nn.init.zeros_(self.modulation[-1].weight)
            nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        c: Float[torch.Tensor, "batch condition_embed_dim"],
    ) -> Tuple[torch.Tensor, ...]:
        return self.modulation(c).chunk(3 * self.n_blocks, dim=-1)
