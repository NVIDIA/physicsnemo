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

"""FLARE (Fast Low-rank Attention Routing Engine) and FLARE++ attention layers.

:class:`FLARE` and :class:`FLAREPlusPlus` are token mixers with cost linear in
the number of tokens. The ``_flare_self_attention*`` kernels below are shared
with :class:`~physicsnemo.nn.module.gale.GALE`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float


def _flare_self_attention(
    x_mid: Float[torch.Tensor, "B H N D"],
    q_global: nn.Parameter,
    self_k: nn.Module,
    self_v: nn.Module,
    scale: float,
) -> Float[torch.Tensor, "B H N D"]:
    r"""FLARE two-pass self-attention kernel.

    Computes low-rank attention via learned global queries: first aggregate
    token values into global slots, then distribute back to tokens.

    Parameters
    ----------
    x_mid : torch.Tensor
        Projected input of shape :math:`(B, H, N, D)`.
    q_global : nn.Parameter
        Learned global queries of shape :math:`(1, H, S, D)`.
    self_k : nn.Module
        Key projection applied to ``x_mid``.
    self_v : nn.Module
        Value projection applied to ``x_mid``.
    scale : float
        Attention scale factor.

    Returns
    -------
    torch.Tensor
        Self-attended output of shape :math:`(B, H, N, D)`.
    """
    G = q_global.to(dtype=x_mid.dtype).expand(x_mid.shape[0], -1, -1, -1)
    k = self_k(x_mid)
    v = self_v(x_mid)
    z = F.scaled_dot_product_attention(G, k, v, scale=scale)
    return F.scaled_dot_product_attention(k, G, z, scale=scale)


def _flare_self_attention_te(
    x_mid: Float[torch.Tensor, "B H N D"],
    q_global: nn.Parameter,
    self_k: nn.Module,
    self_v: nn.Module,
    attn_fn: nn.Module,
    heads: int,
) -> Float[torch.Tensor, "B H N D"]:
    r"""FLARE two-pass self-attention kernel on the Transformer Engine backend.

    Same computation as :func:`_flare_self_attention`, but the two attention
    passes run through a Transformer Engine ``DotProductAttention`` module.  Both
    passes are treated as cross-attention because the global-query and token
    sequences have different lengths.  ``DotProductAttention`` consumes ``bshd``
    inputs and returns the head dimensions flattened, so each pass is reshaped
    back to ``bshd``/``bhnd`` around the call.

    Parameters
    ----------
    x_mid : torch.Tensor
        Projected input of shape :math:`(B, H, N, D)`.
    q_global : nn.Parameter
        Learned global queries of shape :math:`(1, H, S, D)`.
    self_k : nn.Module
        Key projection applied to ``x_mid``.
    self_v : nn.Module
        Value projection applied to ``x_mid``.
    attn_fn : nn.Module
        Transformer Engine ``DotProductAttention`` module configured with
        ``qkv_format="bshd"`` and ``attention_type="cross"``.
    heads : int
        Number of attention heads :math:`H`, used to un-flatten the attention
        output.

    Returns
    -------
    torch.Tensor
        Self-attended output of shape :math:`(B, H, N, D)`.
    """
    G = q_global.to(dtype=x_mid.dtype).expand(x_mid.shape[0], -1, -1, -1)
    G = rearrange(G, "b h s d -> b s h d")
    k = rearrange(self_k(x_mid), "b h n d -> b n h d")
    v = rearrange(self_v(x_mid), "b h n d -> b n h d")
    z = attn_fn(G, k, v)
    z = rearrange(z, "b s (h d) -> b s h d", h=heads)
    y = attn_fn(k, G, z)
    return rearrange(y, "b n (h d) -> b h n d", h=heads)


# ---------------------------------------------------------------------------
# FLARE / FLARE++ mixers
# ---------------------------------------------------------------------------


def _check_dims(dim: int, heads: int, dim_head: int | None) -> int:
    """Validate the mixer sizes and return the per-head width."""
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if heads <= 0:
        raise ValueError(f"heads must be positive, got {heads}")
    if dim_head is None:
        if dim % heads != 0:
            raise ValueError(
                f"dim must be divisible by heads, got dim={dim}, heads={heads}"
            )
        dim_head = dim // heads
    if heads * dim_head != dim:
        raise ValueError(
            "FLARE mixers project dim -> dim, so heads * dim_head must equal dim; "
            f"got heads={heads}, dim_head={dim_head}, dim={dim}"
        )
    return dim_head


def _head_norm(
    dim_head: int, *, enabled: bool, use_rmsnorm: bool, elementwise_affine: bool
) -> nn.Module:
    r"""Per-head normalization: RMSNorm (``eps=1e-6``), LayerNorm, or identity."""
    if not enabled:
        return nn.Identity()
    if use_rmsnorm:
        return nn.RMSNorm(dim_head, eps=1e-6, elementwise_affine=elementwise_affine)
    return nn.LayerNorm(dim_head, elementwise_affine=elementwise_affine)


def _identity_noise_linear(dim: int) -> nn.Linear:
    r"""``Linear(dim, dim)`` initialized to identity plus small noise, zero bias.

    The weight is :math:`I + \epsilon` with :math:`\epsilon \sim
    \mathcal{N}_{[-2, 2]}(0, 0.02^2)` (truncated normal), so the projection
    starts close to the identity map. The layer is flagged with
    ``_skip_weight_init`` so model-level weight initializers leave it untouched.
    """
    proj = nn.Linear(dim, dim)
    with torch.no_grad():
        nn.init.trunc_normal_(proj.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
        proj.weight.add_(torch.eye(dim, dtype=proj.weight.dtype))
        proj.bias.zero_()
    proj._skip_weight_init = True  # type: ignore[attr-defined]
    return proj


def _init_latents(latents: nn.Parameter) -> None:
    r"""Initialize latent query tokens from :math:`\mathcal{N}(0, 0.02^2)`."""
    nn.init.normal_(latents, mean=0.0, std=0.02)


def _split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    return rearrange(x, "b n (h d) -> b h n d", h=heads)


class FLARE(nn.Module):
    r"""FLARE: Fast Low-rank Attention Routing Engine attention layer.

    Learned latent queries gather information from all tokens and scatter it
    back, so the cost is linear in the number of tokens:

    .. math::

        Z = \mathrm{softmax}(Q K^\top s) V, \qquad
        Y = \mathrm{softmax}(K Q^\top s) Z,

    with per-head latent queries :math:`Q \in \mathbb{R}^{H \times M \times D}`,
    keys/values :math:`K = x W_K`, :math:`V = x W_V` and scale
    :math:`s = D^{-1/2}`. :math:`W_K`, :math:`W_V` are ``dim x dim`` linear
    maps initialized to identity plus small noise. With ``qk_norm=True`` the
    queries and keys are normalized per head before attention.

    See the `FLARE paper <https://arxiv.org/abs/2508.12594>`_.

    Parameters
    ----------
    dim : int
        Number of input and output channels :math:`C`.
    heads : int, optional
        Number of attention heads :math:`H`. Default is 8.
    dim_head : int | None, optional
        Channels per head :math:`D`. Must satisfy ``heads * dim_head == dim``.
        ``None`` uses ``dim // heads``. Default is ``None``.
    dropout : float, optional
        Dropout after the output projection. Default is 0.0.
    n_global_queries : int, optional
        Number of latent queries :math:`M` per head. Default is 64.
    qk_norm : bool, optional
        Normalize latent queries and keys per head (affine). Default is ``False``.
    use_rmsnorm : bool, optional
        Use RMSNorm (``eps=1e-6``) for the per-head norms; ``False`` uses
        LayerNorm. Default is ``True``.

    Forward
    -------
    x : torch.Tensor
        Input of shape :math:`(B, N, C)`.

    Outputs
    -------
    torch.Tensor
        Output of shape :math:`(B, N, C)`.

    Notes
    -----
    :meth:`no_weight_decay` returns the names of this module's parameters that
    should be optimized without weight decay (the latent queries). Training
    code should use it to build optimizer parameter groups.

    Examples
    --------
    >>> import torch
    >>> flare = FLARE(dim=256, heads=8)
    >>> flare(torch.randn(2, 100, 256)).shape
    torch.Size([2, 100, 256])
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int | None = None,
        dropout: float = 0.0,
        n_global_queries: int = 64,
        qk_norm: bool = False,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        dim_head = _check_dims(dim, heads, dim_head)
        if n_global_queries <= 0:
            raise ValueError(
                f"n_global_queries must be positive, got {n_global_queries}"
            )
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.n_global_queries = n_global_queries
        self.scale = dim_head**-0.5

        self.latent_q = nn.Parameter(torch.empty(heads, n_global_queries, dim_head))
        _init_latents(self.latent_q)

        self.q_norm = _head_norm(
            dim_head, enabled=qk_norm, use_rmsnorm=use_rmsnorm, elementwise_affine=True
        )
        self.k_norm = _head_norm(
            dim_head, enabled=qk_norm, use_rmsnorm=use_rmsnorm, elementwise_affine=True
        )
        self.k_proj = _identity_noise_linear(dim)
        self.v_proj = _identity_noise_linear(dim)
        self.out_proj = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def no_weight_decay(self) -> set[str]:
        r"""Names of this module's parameters to train without weight decay."""
        return {"latent_q"}

    def forward(self, x: Float[torch.Tensor, "B N C"]) -> Float[torch.Tensor, "B N C"]:
        r"""Apply FLARE attention to ``x`` of shape :math:`(B, N, C)`."""
        if not torch.compiler.is_compiling() and x.ndim != 3:
            raise ValueError(
                f"Expected a 3D input tensor (B, N, C), got shape {tuple(x.shape)}"
            )
        k = self.k_norm(_split_heads(self.k_proj(x), self.heads))
        v = _split_heads(self.v_proj(x), self.heads)
        q = self.q_norm(self.latent_q).to(dtype=k.dtype)
        q = q.unsqueeze(0).expand(x.shape[0], -1, -1, -1)

        z = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        y = F.scaled_dot_product_attention(k, q, z, scale=self.scale)
        y = rearrange(y, "b h n d -> b n (h d)")
        return self.out_dropout(self.out_proj(y))


class FLAREPlusPlus(nn.Module):
    r"""FLARE++ attention with input-conditioned routing queries.

    FLARE++ builds its routing queries from the input, then routes like
    :class:`FLARE`:

    1. Dynamic queries: learned seeds :math:`Q_0` attend over the tokens,
       :math:`Q_{dyn} = \mathrm{softmax}(\bar Q_0 \bar K_0^\top s) V_0`, with
       :math:`K_0 = x W_{K_0}` and :math:`V_0 = x W_{V_0}` (shared with
       :math:`K_0` when ``share_k0_v0=True``). :math:`Q_0` is normalized
       (affine), :math:`K_0` and :math:`V_0` are normalized without affine.
    2. Gated anchor: :math:`Q = Q_{fix} + \sigma(g_h) Q_{dyn}` with learned
       fixed queries :math:`Q_{fix}` (normalized without affine when
       ``q_fixed_norm=True``) and one gate logit :math:`g_h` per head.
    3. Gather/scatter as in FLARE, with keys :math:`K = x W_K` (normalized when
       ``k_norm=True``) and values :math:`V = x W_V`.

    All projections are ``dim x dim`` linear maps initialized to identity plus
    small noise; the attention scale is :math:`s = D^{-1/2}`.

    See the `FLARE++ paper <https://arxiv.org/abs/2608.11519>`_.

    Parameters
    ----------
    dim : int
        Number of input and output channels :math:`C`.
    heads : int, optional
        Number of attention heads :math:`H`. Default is 8.
    dim_head : int | None, optional
        Channels per head :math:`D`. Must satisfy ``heads * dim_head == dim``.
        ``None`` uses ``dim // heads``. Default is ``None``.
    dropout : float, optional
        Dropout after the output projection. Default is 0.0.
    n_global_queries : int, optional
        Number of routing queries :math:`M` per head. Default is 64.
    k_norm : bool, optional
        Normalize routing keys :math:`K` per head (affine). Default is ``True``.
    share_k0_v0 : bool, optional
        Use the query-synthesis keys as its values. Default is ``True``.
    q_fixed_norm : bool, optional
        Normalize the fixed queries (no affine). Default is ``True``.
    gate_logit_init : float, optional
        Initial gate logit for every head. Default is 0.25.
    use_rmsnorm : bool, optional
        Use RMSNorm (``eps=1e-6``) for the per-head norms; ``False`` uses
        LayerNorm. Default is ``True``.

    Forward
    -------
    x : torch.Tensor
        Input of shape :math:`(B, N, C)`.

    Outputs
    -------
    torch.Tensor
        Output of shape :math:`(B, N, C)`.

    Notes
    -----
    :meth:`no_weight_decay` returns the names of this module's parameters that
    should be optimized without weight decay (seeds, fixed queries and gate
    logits).

    Examples
    --------
    >>> import torch
    >>> attention = FLAREPlusPlus(dim=256, heads=8)
    >>> attention(torch.randn(2, 100, 256)).shape
    torch.Size([2, 100, 256])
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int | None = None,
        dropout: float = 0.0,
        n_global_queries: int = 64,
        k_norm: bool = True,
        share_k0_v0: bool = True,
        q_fixed_norm: bool = True,
        gate_logit_init: float = 0.25,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        dim_head = _check_dims(dim, heads, dim_head)
        if n_global_queries <= 0:
            raise ValueError(
                f"n_global_queries must be positive, got {n_global_queries}"
            )
        if not math.isfinite(gate_logit_init):
            raise ValueError(f"gate_logit_init must be finite, got {gate_logit_init}")
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.n_global_queries = n_global_queries
        self.share_k0_v0 = bool(share_k0_v0)
        self.scale = dim_head**-0.5

        shape = (heads, n_global_queries, dim_head)
        self.latent_q0 = nn.Parameter(torch.empty(shape))
        _init_latents(self.latent_q0)
        self.latent_q_fixed = nn.Parameter(torch.empty(shape))
        _init_latents(self.latent_q_fixed)
        self.gate_logit = nn.Parameter(torch.full((heads,), float(gate_logit_init)))

        def norm(enabled: bool, affine: bool) -> nn.Module:
            return _head_norm(
                dim_head,
                enabled=enabled,
                use_rmsnorm=use_rmsnorm,
                elementwise_affine=affine,
            )

        self.q0_norm = norm(True, True)
        self.k0_norm = norm(True, False)
        self.v0_norm = norm(True, False)
        self.k_norm = norm(k_norm, True)
        self.q_fixed_norm = norm(q_fixed_norm, False)

        self.k0_proj = _identity_noise_linear(dim)
        self.v0_proj = None if self.share_k0_v0 else _identity_noise_linear(dim)
        self.k_proj = _identity_noise_linear(dim)
        self.v_proj = _identity_noise_linear(dim)
        self.out_proj = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def no_weight_decay(self) -> set[str]:
        r"""Names of this module's parameters to train without weight decay."""
        return {"latent_q0", "latent_q_fixed", "gate_logit"}

    def forward(self, x: Float[torch.Tensor, "B N C"]) -> Float[torch.Tensor, "B N C"]:
        r"""Apply FLARE++ attention to ``x`` of shape :math:`(B, N, C)`."""
        if not torch.compiler.is_compiling():
            if x.ndim != 3:
                raise ValueError(
                    f"Expected a 3D input tensor (B, N, C), got shape {tuple(x.shape)}"
                )
            # The query-synthesis pass normalizes over all tokens; token-sharded
            # inputs are not supported yet.
            if hasattr(x, "redistribute"):
                raise NotImplementedError(
                    "FLAREPlusPlus does not yet support token-sharded inputs; "
                    "use replicated inputs with data parallelism."
                )
        batch, heads = x.shape[0], self.heads

        k0 = self.k0_norm(_split_heads(self.k0_proj(x), heads))
        if self.v0_proj is None:
            v0 = k0
        else:
            v0 = self.v0_norm(_split_heads(self.v0_proj(x), heads))
        k = self.k_norm(_split_heads(self.k_proj(x), heads))
        v = _split_heads(self.v_proj(x), heads)

        q0 = self.q0_norm(self.latent_q0).to(dtype=k0.dtype)
        q0 = q0.unsqueeze(0).expand(batch, -1, -1, -1)
        q_dynamic = F.scaled_dot_product_attention(q0, k0, v0, scale=self.scale)

        # Anchored mix in fp32: fixed queries plus the gated dynamic queries.
        q_fixed = self.q_fixed_norm(self.latent_q_fixed).float()
        gate = torch.sigmoid(self.gate_logit.float()).view(1, heads, 1, 1)
        q = (q_fixed.unsqueeze(0) + gate * q_dynamic.float()).to(dtype=k.dtype)

        z = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        y = F.scaled_dot_product_attention(k, q, z, scale=self.scale)
        y = rearrange(y, "b h n d -> b n (h d)")
        return self.out_dropout(self.out_proj(y))
