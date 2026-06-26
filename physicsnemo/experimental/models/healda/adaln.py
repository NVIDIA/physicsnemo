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
"""ndim-agnostic adaptive layer norm zero (adaLN-Zero) modulation."""

from typing import Literal, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.nn.module.dit_layers import get_layer_norm


def _broadcast(param: Float[torch.Tensor, "batch channels"], ndim: int) -> torch.Tensor:
    # (B, C) -> (B, 1, ..., 1, C) so a per-sample modulation broadcasts over a
    # hidden-state tensor of arbitrary rank (3D (B, L, C), 4D (B, T, X, C), ...).
    shape = (param.shape[0],) + (1,) * (ndim - 2) + (param.shape[1],)
    return param.view(shape)


class AdaLayerNormZero(nn.Module):
    r"""Adaptive layer norm zero (adaLN-Zero) modulation, agnostic to input rank.

    Emits ``n_blocks`` ``(shift, scale, gate)`` triples from the conditioning
    embedding via ``SiLU + Linear``. The first block's ``shift``/``scale`` are
    applied to the affine-free layer-normed ``x`` here; its ``gate`` is returned
    for the caller's gated residual. For ``n_blocks > 1`` the remaining blocks'
    ``(shift, scale, gate)`` are returned unapplied, so one projection can drive
    several sub-layers (e.g. grouped attention + feed-forward, as in the standard
    DiT block). Modulation vectors are broadcast to match ``x.ndim``, so the same
    module serves 3D :math:`(B, L, C)` and 4D :math:`(B, T, X, C)` states.

    The ``SiLU`` is applied inside this module, so the conditioning embedder must
    emit a pre-activation embedding.

    Parameters
    ----------
    embedding_dim : int
        Channel dimension :math:`C` of the hidden states.
    condition_embed_dim : int
        Channel dimension of the conditioning embedding.
    n_blocks : int, optional, default=1
        Number of ``(shift, scale, gate)`` triples to emit.
    zero_init : bool, optional, default=True
        If ``True``, zero the modulation ``Linear`` (adaLN-Zero) at construction
        and in :meth:`initialize_weights`, so each residual branch starts as
        identity. If ``False``, the modulation keeps its default initialization.
    layernorm_backend : Literal["apex", "torch"], optional, default="torch"
        Backend for the affine-free :func:`~physicsnemo.nn.module.dit_layers.get_layer_norm`.
    norm_eps : float, optional, default=1e-6
        Epsilon for the layer norm.

    Forward
    -------
    x : torch.Tensor
        Hidden states of shape :math:`(B, \dots, C)` (any rank :math:`\geq 2`).
    c : torch.Tensor
        Conditioning embedding of shape :math:`(B, D_c)`.

    Outputs
    -------
    Tuple[torch.Tensor, ...]
        ``(normed, gate)`` for ``n_blocks == 1``, where ``normed`` is the
        modulated layer-normed ``x`` and ``gate`` is broadcast to ``x.ndim``. For
        ``n_blocks > 1``, the remaining ``(shift, scale, gate)`` of each later
        block follow, each broadcast to ``x.ndim``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.healda.adaln import AdaLayerNormZero
    >>> norm = AdaLayerNormZero(embedding_dim=64, condition_embed_dim=32)
    >>> x = torch.randn(2, 5, 64)
    >>> c = torch.randn(2, 32)
    >>> normed, gate = norm(x, c)
    >>> normed.shape, gate.shape
    (torch.Size([2, 5, 64]), torch.Size([2, 1, 64]))
    """

    def __init__(
        self,
        embedding_dim: int,
        condition_embed_dim: int,
        n_blocks: int = 1,
        zero_init: bool = True,
        layernorm_backend: Literal["apex", "torch"] = "torch",
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.zero_init = zero_init
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_embed_dim, 3 * n_blocks * embedding_dim, bias=True),
        )
        self.norm = get_layer_norm(
            embedding_dim, layernorm_backend, elementwise_affine=False, eps=norm_eps
        )
        if zero_init:
            self.initialize_weights()

    def initialize_weights(self) -> None:
        r"""Zero the modulation linear when ``zero_init`` is set (adaLN-Zero).

        Returns
        -------
        None
            Modifies parameters in-place; a no-op when ``zero_init`` is ``False``.
        """
        if self.zero_init:
            nn.init.zeros_(self.modulation[-1].weight)
            nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        x: Float[torch.Tensor, "batch ... hidden_size"],
        c: Float[torch.Tensor, "batch condition_embed_dim"],
    ) -> Tuple[torch.Tensor, ...]:
        chunks = self.modulation(c).chunk(3 * self.n_blocks, dim=-1)
        shift, scale, gate = chunks[0], chunks[1], chunks[2]
        normed = self.norm(x) * (1 + _broadcast(scale, x.ndim)) + _broadcast(
            shift, x.ndim
        )
        outputs = [normed, _broadcast(gate, x.ndim)]
        outputs.extend(_broadcast(extra, x.ndim) for extra in chunks[3:])
        return tuple(outputs)
