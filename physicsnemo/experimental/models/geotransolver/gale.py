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

from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.version_check import check_version_spec
from physicsnemo.models.transolver.Physics_Attention import (
    PhysicsAttentionIrregularMesh,
    gumbel_softmax,
)
from physicsnemo.models.transolver.transolver import MLP

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module

# Check optional dependency availability
TE_AVAILABLE = check_version_spec("transformer_engine", "0.1.0", hard_fail=False)
if TE_AVAILABLE:
    import transformer_engine.pytorch as te

class GALE(PhysicsAttentionIrregularMesh):
    r"""Geometry-Aware Latent Embeddings (GALE) attention layer.

    This is an extension of the Transolver PhysicsAttention mechanism to support
    cross-attention with a context vector, built from geometry and global embeddings.
    GALE combines self-attention on learned physical state slices with cross-attention
    to geometry-aware context, using a learnable mixing weight to blend the two.

    Parameters
    ----------
    dim : int
        Input dimension of the features.
    heads : int, optional
        Number of attention heads. Default is 8.
    dim_head : int, optional
        Dimension of each attention head. Default is 64.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    slice_num : int, optional
        Number of learned physical state slices. Default is 64.
    use_te : bool, optional
        Whether to use Transformer Engine backend when available. Default is True.
    plus : bool, optional
        Whether to use Transolver++ features. Default is False.
    context_dim : int, optional
        Dimension of the context vector for cross-attention. Default is 0.

    Notes
    -----
    The mixing between self-attention and cross-attention is controlled by a learnable
    parameter ``state_mixing`` which is passed through a sigmoid function to ensure
    the mixing weight stays in \([0, 1]\).

    See Also
    --------
    :class:`physicsnemo.models.transolver.Physics_Attention.PhysicsAttentionIrregularMesh` : Base physics attention class.
    :class:`GALE_block` : Transformer block using GALE attention.
    """

    def __init__(
        self,
        dim,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_te: bool = True,
        plus: bool = False,
        context_dim: int = 0,
    ):
        super().__init__(dim, heads, dim_head, dropout, slice_num, use_te, plus)

        linear_layer = te.Linear if self.use_te else nn.Linear

        # We have additional parameters, here:
        self.cross_q = linear_layer(dim_head, dim_head)
        self.cross_k = linear_layer(context_dim, dim_head)
        self.cross_v = linear_layer(context_dim, dim_head)

        # This is the learnable mixing weight between self and cross attention.
        # We start near 0.0 since it is passed through a sigmoid to keep the
        # mixing weight between 0 and 1.
        self.state_mixing = nn.Parameter(torch.tensor(0.0))

    def compute_slice_attention_cross(
        self, slice_tokens: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute cross-attention between slice tokens and context.

        Parameters
        ----------
        slice_tokens : torch.Tensor
            Slice tokens of shape \((B, H, N, D)\) where \(B\) is batch size, \(H\) is number of heads, \(N\) is number of slices, and \(D\) is head dimension.
        context : torch.Tensor
            Context tensor of shape \((B, H, N_c, D_c)\) where \(N_c\) is number of context slices and \(D_c\) is context dimension.

        Returns
        -------
        torch.Tensor
            Cross-attention output of shape \((B, H, N, D)\).
        """

        # Project the slice and context tokens:

        q_input = torch.cat(slice_tokens, dim=-2)
        q = self.cross_q(q_input)
        
        k = self.cross_k(context)
        v = self.cross_v(context)

        # Compute the attention:
        if self.use_te:
            q = rearrange(q, "b h s d -> b s h d")
            k = rearrange(k, "b h s d -> b s h d")
            v = rearrange(v, "b h s d -> b s h d")
            cross_attention = self.attn_fn(q, k, v)
            cross_attention = rearrange(
                cross_attention, "b s (h d) -> b h s d", h=self.heads, d=self.dim_head
            )
        else:
            cross_attention = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False
            )
        cross_attention = torch.split(cross_attention, slice_tokens[0].shape[-2], dim=-2)


        return cross_attention

    def forward(
        self, x: tuple[torch.Tensor, ...], context: tuple[torch.Tensor, ...] | None = None
    ) -> torch.Tensor:
        r"""Forward pass of the GALE module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape \((B, N, C)\) where \(B\) is batch size, \(N\) is number of tokens, and \(C\) is number of channels.
        context : torch.Tensor, optional
            Context tensor for cross-attention of shape \((B, H, S_c, D_c)\) where \(H\) is number of heads, \(S_c\) is number of context slices, and \(D_c\) is context dimension. If None, only self-attention is applied. Default is None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape \((B, N, C)\), same shape as input.
        """
        # Project the inputs onto learned spaces:
        if self.plus:
            x_mid = [ self.project_input_onto_slices(_x) for _x in x ]
            # In transolver ++, fx_mid is gone.
            # x_mid is used to compute the projections instead:
            fx_mid = [ _x_mid for _x_mid in x_mid ]
        else:
            x_mid, fx_mid = zip(*[ self.project_input_onto_slices(_x) for _x in x ])

        # Perform the linear projection of learned latent space onto slices:
        slice_projections = [ self.in_project_slice(_x_mid) for _x_mid in x_mid ]

        # Slice projections has shape [B, N_head, N_tokens, Head_dim], but head_dim may have changed!
        # Use the slice projections and learned spaces to compute the slices, and their weights:
        slice_weights, slice_tokens = zip(*[self.compute_slices_from_projections(proj, _fx_mid) for proj, _fx_mid in zip(slice_projections, fx_mid)])
        # slice_weights has shape [Batch, N_heads, N_tokens, Slice_num]
        # slice_tokens has shape  [Batch, N_heads, N_tokens, head_dim]
        # Apply attention to the slice tokens
        if self.use_te:
            self_slice_token = [ self.compute_slice_attention_te(_slice_token) for _slice_token in slice_tokens ]
        else:
            self_slice_token = [ self.compute_slice_attention_sdpa(_slice_token) for _slice_token in slice_tokens ]
        
        # HERE, we are differing: apply cross-attention with physical states:
        if context is not None:
            # cross_slice_token = self.compute_slice_attention_cross(
            #     slice_tokens, context
            # )
            cross_slice_token = [ self.compute_slice_attention_cross([_slice_token], context)[0] 
                for _slice_token in slice_tokens 
            ]
            
            # Apply learnable mixing:
            mixing_weight = torch.sigmoid(self.state_mixing)
            out_slice_token = [ mixing_weight * sst + (1 - mixing_weight) * cst
                for sst, cst in zip(self_slice_token, cross_slice_token)
            ]

        else:
            # Just keep self attention:
            out_slice_token = self_slice_token

        # Shape unchanged

        # Deslice:
        outputs = [
            self.project_attention_outputs(ost, sw) for ost, sw in zip(out_slice_token, slice_weights)
        ]

        # Outputs now has the same shape as the original input x

        return outputs


class GALE_block(nn.Module):
    r"""Transformer encoder block using GALE attention.

    This block replaces standard self-attention with the GALE (Geometry-Aware Latent
    Embeddings) attention mechanism, which combines physics-aware self-attention with
    cross-attention to geometry and global context.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    hidden_dim : int
        Hidden dimension of the transformer.
    dropout : float
        Dropout rate.
    act : str, optional
        Activation function name. Default is "gelu".
    mlp_ratio : int, optional
        Ratio of MLP hidden dimension to ``hidden_dim``. Default is 4.
    last_layer : bool, optional
        Whether this is the last layer in the model. Default is False.
    out_dim : int, optional
        Output dimension (only used if ``last_layer=True``). Default is 1.
    slice_num : int, optional
        Number of learned physical state slices. Default is 32.
    use_te : bool, optional
        Whether to use Transformer Engine backend. Default is True.
    plus : bool, optional
        Whether to use Transolver++ features. Default is False.
    context_dim : int, optional
        Dimension of the context vector for cross-attention. Default is 0.

    Notes
    -----
    The block applies layer normalization before the attention operation and uses
    residual connections after both the attention and MLP layers.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        use_te=True,
        plus: bool = False,
        context_dim: int = 0,
    ):
        super().__init__()

        if use_te and not TE_AVAILABLE:
            raise ImportError(
                "Transformer Engine is not installed. Please install it with: pip install transformer-engine>=0.1.0"
            )

        self.last_layer = last_layer
        if use_te:
            self.ln_1 = te.LayerNorm(hidden_dim)
        else:
            self.ln_1 = nn.LayerNorm(hidden_dim)

        self.Attn = GALE(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            use_te=use_te,
            plus=plus,
            context_dim=context_dim,
        )

        if use_te:
            self.ln_mlp1 = te.LayerNormMLP(
                hidden_size=hidden_dim,
                ffn_hidden_size=hidden_dim * mlp_ratio,
            )
        else:
            self.ln_mlp1 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                MLP(
                    hidden_dim,
                    hidden_dim * mlp_ratio,
                    hidden_dim,
                    n_layers=0,
                    res=False,
                    act=act,
                    use_te=False,
                ),
            )

    def forward(self, fx: tuple[torch.Tensor, ...], global_context: tuple[torch.Tensor, ...]) -> torch.Tensor:
        r"""Forward pass of the GALE block.

        Parameters
        ----------
        fx : torch.Tensor
            Input tensor of shape \((B, N, C)\) where \(B\) is batch size, \(N\) is number of tokens, and \(C\) is hidden dimension.
        global_context : torch.Tensor
            Global context tensor for cross-attention of shape \((B, H, S_c, D_c)\) where \(H\) is number of heads, \(S_c\) is number of context slices, and \(D_c\) is context dimension.

        Returns
        -------
        torch.Tensor
            Output tensor of shape \((B, N, C)\), same shape as input.
        """
        
        normed_inputs = [ self.ln_1(_fx) for _fx in fx ]
        attn = self.Attn(normed_inputs, global_context)
        
        fx = [ attn[i] + normed_inputs[i] for i in range(len(normed_inputs)) ]
        
        fx = [ self.ln_mlp1(_fx) + _fx for _fx in fx ]

        return fx

