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
the number of tokens. Their attention passes run on the shared kernels below:
``F.scaled_dot_product_attention`` on ``(B, H, S, D)`` tensors, which also
dispatches to :class:`~physicsnemo.domain_parallel.ShardTensor` for
token-sharded inputs, or a Transformer Engine ``DotProductAttention`` module on
``(B, S, H, D)`` tensors when ``use_te=True``. The ``_flare_self_attention*``
kernels are shared with :class:`~physicsnemo.nn.module.gale.GALE`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float

from physicsnemo.core.version_check import OptionalImport

from .physics_attention import _project_input

te = OptionalImport("transformer_engine.pytorch")


# ---------------------------------------------------------------------------
# Attention kernels (SDPA: (B, H, S, D); Transformer Engine: (B, S, H, D))
# ---------------------------------------------------------------------------


def _flare_attention(
    q: Float[torch.Tensor, "B H S_q D"],
    k: Float[torch.Tensor, "B H S_kv D"],
    v: Float[torch.Tensor, "B H S_kv D"],
    scale: float,
) -> Float[torch.Tensor, "B H S_q D"]:
    r"""One scaled dot-product attention pass on ``(B, H, S, D)`` tensors."""
    return F.scaled_dot_product_attention(q, k, v, scale=scale)


def _flare_attention_te(
    q: Float[torch.Tensor, "B S_q H D"],
    k: Float[torch.Tensor, "B S_kv H D"],
    v: Float[torch.Tensor, "B S_kv H D"],
    attn_fn: nn.Module,
    heads: int,
) -> Float[torch.Tensor, "B S_q H D"]:
    r"""One attention pass through a Transformer Engine ``DotProductAttention``.

    ``attn_fn`` must be configured with ``qkv_format="bshd"`` and
    ``attention_type="cross"`` (query and key sequence lengths differ). It
    returns the heads flattened, so the output is reshaped back to ``bshd``.
    """
    return rearrange(attn_fn(q, k, v), "b s (h d) -> b s h d", h=heads)


def _flare_gather_scatter(
    q: Float[torch.Tensor, "B H M D"],
    k: Float[torch.Tensor, "B H N D"],
    v: Float[torch.Tensor, "B H N D"],
    scale: float,
) -> Float[torch.Tensor, "B H N D"]:
    r"""FLARE two-pass kernel: ``M`` queries gather from ``N`` tokens, then scatter back.

    .. math::

        Z = \mathrm{softmax}(Q K^\top s) V, \qquad
        Y = \mathrm{softmax}(K Q^\top s) Z.
    """
    z = _flare_attention(q, k, v, scale)
    return _flare_attention(k, q, z, scale)


def _flare_gather_scatter_te(
    q: Float[torch.Tensor, "B M H D"],
    k: Float[torch.Tensor, "B N H D"],
    v: Float[torch.Tensor, "B N H D"],
    attn_fn: nn.Module,
    heads: int,
) -> Float[torch.Tensor, "B N H D"]:
    r"""Same as :func:`_flare_gather_scatter` on the Transformer Engine backend."""
    z = _flare_attention_te(q, k, v, attn_fn, heads)
    return _flare_attention_te(k, q, z, attn_fn, heads)


def _flare_self_attention(
    x_mid: Float[torch.Tensor, "B H N D"],
    q_global: nn.Parameter,
    self_k: nn.Module,
    self_v: nn.Module,
    scale: float,
) -> Float[torch.Tensor, "B H N D"]:
    r"""FLARE two-pass self-attention kernel with head-wise K/V projections.

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
    return _flare_gather_scatter(G, self_k(x_mid), self_v(x_mid), scale)


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
    y = _flare_gather_scatter_te(G, k, v, attn_fn, heads)
    return rearrange(y, "b n h d -> b h n d")


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


def _check_te(use_te: bool, name: str) -> None:
    if use_te and not te.available:
        raise ImportError(
            f"{name}(use_te=True) requires Transformer Engine. Install it with "
            "`pip install transformer-engine`, or set use_te=False."
        )


def _head_norm(
    dim_head: int,
    *,
    enabled: bool,
    use_rmsnorm: bool,
    elementwise_affine: bool,
    use_te: bool = False,
) -> nn.Module:
    r"""Per-head normalization: RMSNorm (``eps=1e-6``), LayerNorm, or identity.

    With ``use_te=True`` affine norms are Transformer Engine ``RMSNorm`` /
    ``LayerNorm``. Transformer Engine norms always carry a learnable weight, so
    non-affine norms stay PyTorch modules (they have no parameters).
    """
    if not enabled:
        return nn.Identity()
    if use_te and elementwise_affine:
        if use_rmsnorm:
            return te.RMSNorm(dim_head, eps=1e-6)
        return te.LayerNorm(dim_head)
    if use_rmsnorm:
        return nn.RMSNorm(dim_head, eps=1e-6, elementwise_affine=elementwise_affine)
    return nn.LayerNorm(dim_head, elementwise_affine=elementwise_affine)


def _identity_noise_linear(dim: int, use_te: bool = False) -> nn.Module:
    r"""``Linear(dim, dim)`` initialized to identity plus small noise, zero bias.

    The weight is :math:`I + \epsilon` with :math:`\epsilon \sim
    \mathcal{N}_{[-2, 2]}(0, 0.02^2)` (truncated normal), so the projection
    starts close to the identity map. The layer (``nn.Linear``, or
    Transformer Engine ``Linear`` with ``use_te=True``) is flagged with
    ``_skip_weight_init`` so model-level weight initializers leave it untouched.
    """
    proj = te.Linear(dim, dim) if use_te else nn.Linear(dim, dim)
    with torch.no_grad():
        nn.init.trunc_normal_(proj.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
        proj.weight.add_(
            torch.eye(dim, dtype=proj.weight.dtype, device=proj.weight.device)
        )
        proj.bias.zero_()
    proj._skip_weight_init = True  # type: ignore[attr-defined]
    return proj


def _init_latents(latents: nn.Parameter) -> None:
    r"""Initialize latent query tokens from :math:`\mathcal{N}(0, 0.02^2)`."""
    nn.init.normal_(latents, mean=0.0, std=0.02)


class _FLAREMixerBase(nn.Module):
    r"""Backend plumbing shared by :class:`FLARE` and :class:`FLAREPlusPlus`.

    Activations are kept in ``(B, H, S, D)`` layout for the SDPA kernels and
    in ``(B, S, H, D)`` layout (``bshd``) for the Transformer Engine kernels.
    """

    heads: int
    dim_head: int
    scale: float
    use_te: bool

    def _make_attn_fn(self) -> nn.Module:
        # Both FLARE passes are cross-attention (query and token sequence
        # lengths differ). Dropout lives in out_dropout, so TE and PyTorch use
        # the same dropout site.
        return te.DotProductAttention(
            num_attention_heads=self.heads,
            kv_channels=self.dim_head,
            attention_dropout=0.0,
            attn_mask_type="no_mask",
            attention_type="cross",
            qkv_format="bshd",
            softmax_scale=self.scale,
        )

    def _tokens(self, x: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        r"""Project ``(B, N, C)`` tokens and split heads into the backend layout."""
        x = _project_input(x, proj, self.heads, self.dim_head, "B N (H D) -> B N H D")
        return x if self.use_te else x.permute(0, 2, 1, 3)

    def _latents(self, q: torch.Tensor, batch: int) -> torch.Tensor:
        r"""Broadcast ``(H, M, D)`` latent queries over the batch in the backend layout."""
        if self.use_te:
            q = rearrange(q, "h m d -> m h d")
        return q.unsqueeze(0).expand(batch, -1, -1, -1)

    def _per_head(self, t: torch.Tensor) -> torch.Tensor:
        r"""View a per-head vector ``(H,)`` so it broadcasts over the backend layout."""
        return t.view(1, 1, -1, 1) if self.use_te else t.view(1, -1, 1, 1)

    def _attend(self, q, k, v) -> torch.Tensor:
        if self.use_te:
            return _flare_attention_te(q, k, v, self.attn_fn, self.heads)
        return _flare_attention(q, k, v, self.scale)

    def _gather_scatter(self, q, k, v) -> torch.Tensor:
        if self.use_te:
            return _flare_gather_scatter_te(q, k, v, self.attn_fn, self.heads)
        return _flare_gather_scatter(q, k, v, self.scale)

    def _merge_heads(self, y: torch.Tensor) -> torch.Tensor:
        if not self.use_te:
            y = y.permute(0, 2, 1, 3)  # (B, H, N, D) -> (B, N, H, D)
        return rearrange(y, "b n h d -> b n (h d)")


class FLARE(_FLAREMixerBase):
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
    use_te : bool, optional
        Use Transformer Engine linear layers, norms and ``DotProductAttention``
        (CUDA only). Default is ``False``.

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
        use_te: bool = False,
    ) -> None:
        super().__init__()
        _check_te(use_te, "FLARE")
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
        self.use_te = use_te

        self.latent_q = nn.Parameter(torch.empty(heads, n_global_queries, dim_head))
        _init_latents(self.latent_q)

        def norm() -> nn.Module:
            return _head_norm(
                dim_head,
                enabled=qk_norm,
                use_rmsnorm=use_rmsnorm,
                elementwise_affine=True,
                use_te=use_te,
            )

        self.q_norm = norm()
        self.k_norm = norm()
        self.k_proj = _identity_noise_linear(dim, use_te)
        self.v_proj = _identity_noise_linear(dim, use_te)
        self.out_proj = te.Linear(dim, dim) if use_te else nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)
        if use_te:
            self.attn_fn = self._make_attn_fn()

    def no_weight_decay(self) -> set[str]:
        r"""Names of this module's parameters to train without weight decay."""
        return {"latent_q"}

    def forward(self, x: Float[torch.Tensor, "B N C"]) -> Float[torch.Tensor, "B N C"]:
        r"""Apply FLARE attention to ``x`` of shape :math:`(B, N, C)`."""
        if not torch.compiler.is_compiling():
            if x.ndim != 3:
                raise ValueError(
                    f"Expected a 3D input tensor (B, N, C), got shape {tuple(x.shape)}"
                )
            # Token-sharded inputs are supported through the SDPA kernels'
            # ShardTensor dispatch; Transformer Engine attention is not
            # ShardTensor aware and would attend over the local shard only.
            if self.use_te and hasattr(x, "redistribute"):
                raise NotImplementedError(
                    "FLARE(use_te=True) does not support token-sharded inputs; "
                    "use use_te=False for domain parallelism."
                )
        k = self.k_norm(self._tokens(x, self.k_proj))
        v = self._tokens(x, self.v_proj)
        q = self._latents(self.q_norm(self.latent_q).to(dtype=k.dtype), x.shape[0])

        y = self._merge_heads(self._gather_scatter(q, k, v))
        return self.out_dropout(self.out_proj(y))


class FLAREPlusPlus(_FLAREMixerBase):
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
    use_te : bool, optional
        Use Transformer Engine linear layers, norms and ``DotProductAttention``
        (CUDA only). Default is ``False``.

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
        use_te: bool = False,
    ) -> None:
        super().__init__()
        _check_te(use_te, "FLAREPlusPlus")
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
        self.use_te = use_te

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
                use_te=use_te,
            )

        self.q0_norm = norm(True, True)
        self.k0_norm = norm(True, False)
        self.v0_norm = norm(True, False)
        self.k_norm = norm(k_norm, True)
        self.q_fixed_norm = norm(q_fixed_norm, False)

        self.k0_proj = _identity_noise_linear(dim, use_te)
        self.v0_proj = None if self.share_k0_v0 else _identity_noise_linear(dim, use_te)
        self.k_proj = _identity_noise_linear(dim, use_te)
        self.v_proj = _identity_noise_linear(dim, use_te)
        self.out_proj = te.Linear(dim, dim) if use_te else nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)
        if use_te:
            self.attn_fn = self._make_attn_fn()

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
        batch = x.shape[0]

        k0 = self.k0_norm(self._tokens(x, self.k0_proj))
        if self.v0_proj is None:
            v0 = k0
        else:
            v0 = self.v0_norm(self._tokens(x, self.v0_proj))
        k = self.k_norm(self._tokens(x, self.k_proj))
        v = self._tokens(x, self.v_proj)

        # Query synthesis: normalized seeds attend over the tokens.
        q0 = self._latents(self.q0_norm(self.latent_q0).to(dtype=k0.dtype), batch)
        q_dynamic = self._attend(q0, k0, v0)

        # Anchored mix in fp32: fixed queries plus the gated dynamic queries.
        q_fixed = self._latents(self.q_fixed_norm(self.latent_q_fixed).float(), 1)
        gate = self._per_head(torch.sigmoid(self.gate_logit.float()))
        q = (q_fixed + gate * q_dynamic.float()).to(dtype=k.dtype)

        y = self._merge_heads(self._gather_scatter(q, k, v))
        return self.out_dropout(self.out_proj(y))
