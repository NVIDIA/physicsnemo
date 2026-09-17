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

r"""Building blocks for the :class:`~physicsnemo.experimental.models.strata.StrataTransformer3D`
transformer.

These layers are specific to the StrataTransformer3D / Strata models and are kept in the
model package (rather than ``physicsnemo.nn``) per the self-contained-model
convention. The attention layer reuses
:func:`physicsnemo.nn.functional.na3d` for 3D neighborhood attention so that it
inherits NATTEN optional-dependency handling and ``ShardTensor`` dispatch.

StrataTransformer3D / Strata reuse the Diffusion-Transformer (DiT) architecture but are
deterministic regression models, not generative diffusion models — these blocks
carry no diffusion / timestep conditioning (:class:`StrataTransformer3DBlock` is a plain
pre-norm transformer block).
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float

from physicsnemo.nn import apply_rotary_pos_emb
from physicsnemo.nn.functional.natten import na3d as _na3d_func
from physicsnemo.nn.module.mlp_layers import Mlp

__all__ = [
    "Natten3DSelfAttention",
    "StrataTransformer3DBlock",
    "StrataPixel3DBlock",
    "PatchEmbed3D",
    "FinalLayer3D",
    "RopeTables",
    "DepthwiseConv",
]

# A pair of (cos, sin) RoPE lookup tables, as produced by
# ``build_axial_rope_cos_sin_2d_continuous``.
RopeTables = Tuple[torch.Tensor, torch.Tensor]


def _as_kernel_triple(
    attn_kernel: Union[int, Tuple[int, int, int]],
) -> Tuple[int, int, int]:
    r"""Normalize an attention-kernel spec to a ``(kd, kh, kw)`` triple.

    Parameters
    ----------
    attn_kernel : int | Tuple[int, int, int]
        Either a single window size applied to all three axes, or an explicit
        per-axis triple.

    Returns
    -------
    Tuple[int, int, int]
        The ``(depth, height, width)`` window sizes.
    """
    if isinstance(attn_kernel, int):
        return (attn_kernel, attn_kernel, attn_kernel)
    kernel = tuple(attn_kernel)
    if len(kernel) != 3:
        raise ValueError(
            f"attn_kernel tuple must have length 3 (kd, kh, kw); got {attn_kernel}"
        )
    return kernel  # type: ignore[return-value]


class Natten3DSelfAttention(nn.Module):
    r"""Multi-head self-attention over a 3D token grid.

    Supports three attention patterns selected at construction time:

    - **Full attention** (``attn_kernel == -1``): dense self-attention over all
      :math:`N = D \cdot H \cdot W` tokens via
      :func:`torch.nn.functional.scaled_dot_product_attention`.
    - **3D neighborhood attention** (``attn_kernel > 0``): windowed attention
      via :func:`physicsnemo.nn.functional.na3d` (NATTEN), with an integer or
      per-axis ``(kd, kh, kw)`` window and optional dilation.
    - **Depth-axis attention** (``do_depthwise_attention=True``): independent
      full attention along the vertical (depth) axis for each ``(h, w)`` column,
      cheaply capturing vertical structure.

    Optionally applies a 2D rotary position embedding to the queries and keys
    (via :func:`~physicsnemo.nn.apply_rotary_pos_emb`) and a sigmoid
    output gate.

    Parameters
    ----------
    dim : int
        Token embedding dimension. Must be divisible by ``num_heads``.
    num_heads : int, optional, default=8
        Number of attention heads.
    qkv_bias : bool, optional, default=False
        Whether the fused QKV projection uses a bias.
    qk_norm : bool, optional, default=False
        If ``True``, applies RMS normalization to the per-head queries and keys.
    qk_norm_affine : bool, optional, default=False
        Whether the QK RMS norms use a learnable affine scale.
    attn_drop_rate : float, optional, default=0.0
        Dropout probability applied to attention weights (training only).
    proj_drop_rate : float, optional, default=0.0
        Dropout probability applied after the output projection.
    attn_kernel : int | Tuple[int, int, int], optional, default=-1
        Neighborhood-attention window size; ``-1`` selects full attention.
        Ignored when ``do_depthwise_attention=True``.
    do_depthwise_attention : bool, optional, default=False
        If ``True``, attend only along the depth axis (per ``(h, w)`` column).
    na_dilation : int, optional, default=1
        Dilation factor for 3D neighborhood attention.
    gated_attention : bool, optional, default=False
        If ``True``, multiply the attention output by a learned sigmoid gate.
    na3d_backend : str, optional, default=None
        NATTEN backend passed to :func:`physicsnemo.nn.functional.na3d` (e.g.
        ``"cutlass-fna"``); ``None`` uses the NATTEN default.

    Forward
    -------
    x : torch.Tensor
        Input tokens of shape :math:`(B, N, C)` with :math:`N = D \cdot H \cdot W`.
    latent_dhw : Tuple[int, int, int], optional
        The ``(D, H, W)`` token-grid shape. Required for neighborhood and
        depth-axis attention.
    rope_tables : Tuple[torch.Tensor, torch.Tensor], optional
        Precomputed ``(cos, sin)`` RoPE tables to rotate queries / keys.

    Outputs
    -------
    torch.Tensor
        Output tokens of shape :math:`(B, N, C)`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        qk_norm_affine: bool = False,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_kernel: Union[int, Tuple[int, int, int]] = -1,
        do_depthwise_attention: bool = False,
        na_dilation: int = 1,
        gated_attention: bool = False,
        na3d_backend: Optional[str] = None,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attn_drop_rate = attn_drop_rate
        # Validate a per-axis kernel eagerly (length-3) so all entry points —
        # StrataTransformer3D, Strata, and direct use — fail at construction, not deep inside
        # NATTEN. The raw value is kept as given (int stays int).
        _as_kernel_triple(attn_kernel)
        self.attn_kernel = attn_kernel
        self.do_depthwise_attention = do_depthwise_attention
        self.na_dilation = na_dilation
        self.gated_attention = gated_attention
        self.na3d_backend = na3d_backend

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = (
            nn.RMSNorm(self.head_dim, elementwise_affine=qk_norm_affine, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            nn.RMSNorm(self.head_dim, elementwise_affine=qk_norm_affine, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = (
            nn.Dropout(proj_drop_rate) if proj_drop_rate > 0.0 else nn.Identity()
        )
        self.gate_proj = nn.Linear(dim, dim) if gated_attention else nn.Identity()

    def forward(
        self,
        x: Float[torch.Tensor, "batch tokens dim"],
        latent_dhw: Optional[Tuple[int, int, int]] = None,
        rope_tables: Optional[RopeTables] = None,
    ) -> Float[torch.Tensor, "batch tokens dim"]:
        B, N, C = x.shape

        # Optional output gate computed from the (pre-attention) input tokens.
        if self.gated_attention:
            gate = torch.sigmoid(self.gate_proj(x))  # (B, N, C)

        # Fused QKV projection -> (B, heads, N, head_dim) for each of q, k, v.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Rotary position embedding (skipped for depth-axis attention by the caller).
        if rope_tables is not None:
            cos, sin = rope_tables
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        # RoPE / qk-norm may upcast; match v's dtype before the attention kernel.
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        dropout_p = self.attn_drop_rate if self.training else 0.0

        if self.do_depthwise_attention:
            # Independent full attention along the depth axis per (h, w) column.
            if not torch.compiler.is_compiling():
                if latent_dhw is None:
                    raise ValueError("depth-axis attention requires latent_dhw")
                if N != latent_dhw[0] * latent_dhw[1] * latent_dhw[2]:
                    raise ValueError(
                        f"Expected N == D*H*W for latent_dhw={latent_dhw}, got N={N}"
                    )
            d, h, w = latent_dhw
            q, k, v = (
                rearrange(
                    t, "b head (d hh ww) c -> (b hh ww) head d c", d=d, hh=h, ww=w
                )
                for t in (q, k, v)
            )
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, scale=self.scale
            )
            out = rearrange(
                out, "(b hh ww) head d c -> b (d hh ww) (head c)", b=B, hh=h, ww=w
            )
        elif self.attn_kernel == -1:
            # Dense self-attention over the whole token sequence.
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, scale=self.scale
            )
            out = out.transpose(1, 2).reshape(B, N, C)
        else:
            # 3D neighborhood (windowed) attention via NATTEN.
            if not torch.compiler.is_compiling():
                if latent_dhw is None:
                    raise ValueError("neighborhood attention requires latent_dhw")
                if N != latent_dhw[0] * latent_dhw[1] * latent_dhw[2]:
                    raise ValueError(
                        f"Expected N == D*H*W for latent_dhw={latent_dhw}, got N={N}"
                    )
            d, h, w = latent_dhw
            q, k, v = (
                rearrange(t, "b head (d h w) c -> b d h w head c", d=d, h=h, w=w)
                for t in (q, k, v)
            )
            out = _na3d_func(
                q,
                k,
                v,
                _as_kernel_triple(self.attn_kernel),
                dilation=self.na_dilation,
                is_causal=False,
                backend=self.na3d_backend,
            )
            out = rearrange(out, "b d h w head c -> b (d h w) (head c)")

        if self.gated_attention:
            out = out * gate

        return self.proj_drop(self.proj(out))


class StrataTransformer3DBlock(nn.Module):
    r"""Pre-norm transformer block for StrataTransformer3D.

    Applies, with residual connections, a :class:`Natten3DSelfAttention`
    sub-layer followed by an MLP sub-layer (reusing
    :class:`physicsnemo.nn.Mlp`). Layer norms are non-affine, matching the
    standard DiT block. Unlike the diffusion DiT block, no adaLN conditioning is
    used (StrataTransformer3D is a deterministic field-to-field model).

    Parameters
    ----------
    dim : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float, optional, default=4.0
        Ratio of MLP hidden dimension to ``dim``.
    qkv_bias : bool, optional, default=True
        Whether the attention QKV projection uses a bias.
    qk_norm : bool, optional, default=False
        Whether to RMS-normalize queries and keys.
    qk_norm_affine : bool, optional, default=False
        Whether the QK RMS norms use a learnable affine scale.
    mlp_drop_rate : float, optional, default=0.0
        Dropout probability inside the MLP and attention output projection.
    attn_drop_rate : float, optional, default=0.0
        Dropout probability on attention weights.
    attn_kernel : int | Tuple[int, int, int], optional, default=-1
        Neighborhood-attention window; ``-1`` selects full attention.
    do_depthwise_attention : bool, optional, default=False
        If ``True``, this block attends only along the depth axis.
    na_dilation : int, optional, default=1
        Dilation factor for 3D neighborhood attention.
    gated_attention : bool, optional, default=False
        Whether to apply a learned sigmoid gate to the attention output.
    na3d_backend : str, optional, default=None
        NATTEN backend forwarded to :class:`Natten3DSelfAttention`.

    Forward
    -------
    x : torch.Tensor
        Input tokens of shape :math:`(B, N, C)`.
    latent_dhw : Tuple[int, int, int], optional
        The ``(D, H, W)`` token-grid shape.
    rope_tables : Tuple[torch.Tensor, torch.Tensor], optional
        Precomputed ``(cos, sin)`` RoPE tables.

    Outputs
    -------
    torch.Tensor
        Output tokens of shape :math:`(B, N, C)`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        qk_norm_affine: bool = False,
        mlp_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        attn_kernel: Union[int, Tuple[int, int, int]] = -1,
        do_depthwise_attention: bool = False,
        na_dilation: int = 1,
        gated_attention: bool = False,
        na3d_backend: Optional[str] = None,
    ):
        super().__init__()
        self.do_depthwise_attention = do_depthwise_attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Natten3DSelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            qk_norm_affine=qk_norm_affine,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=mlp_drop_rate,
            attn_kernel=attn_kernel,
            do_depthwise_attention=do_depthwise_attention,
            na_dilation=na_dilation,
            gated_attention=gated_attention,
            na3d_backend=na3d_backend,
        )
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_layer=nn.GELU,
            drop=mlp_drop_rate,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch tokens dim"],
        latent_dhw: Optional[Tuple[int, int, int]] = None,
        rope_tables: Optional[RopeTables] = None,
    ) -> Float[torch.Tensor, "batch tokens dim"]:
        # Self-attention sub-layer (pre-norm, residual).
        x = x + self.attn(self.norm1(x), latent_dhw=latent_dhw, rope_tables=rope_tables)
        # MLP sub-layer (pre-norm, residual).
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed3D(nn.Module):
    r"""Patchify a 3D field with a strided 3D convolution.

    Splits a :math:`(B, C, D, H, W)` field into non-overlapping patches and
    linearly embeds each patch, producing a :math:`(B, E, D', H', W')` feature
    map where ``D' = D / p_d`` etc.

    Parameters
    ----------
    depth : int
        Input depth :math:`D` (number of vertical levels).
    height : int
        Input height :math:`H`.
    width : int
        Input width :math:`W`.
    patch_size : int | Tuple[int, int, int], optional, default=16
        Patch size, either isotropic or per-axis ``(p_d, p_h, p_w)``.
    in_chans : int, optional, default=3
        Number of input channels.
    embed_dim : int, optional, default=768
        Output embedding dimension :math:`E`.

    Forward
    -------
    x : torch.Tensor
        Input field of shape :math:`(B, C, D, H, W)`.

    Outputs
    -------
    torch.Tensor
        Patch embeddings of shape :math:`(B, E, D', H', W')`.
    """

    def __init__(
        self,
        depth: int,
        height: int,
        width: int,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        pd, ph, pw = patch_size
        if depth % pd != 0:
            raise ValueError(
                f"Depth ({depth}) must be divisible by vertical patch size ({pd})"
            )
        if height % ph != 0:
            raise ValueError(
                f"Height ({height}) must be divisible by horizontal patch size ({ph})"
            )
        if width % pw != 0:
            raise ValueError(
                f"Width ({width}) must be divisible by horizontal patch size ({pw})"
            )

        self.depth = depth
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.num_patches = (depth // pd) * (height // ph) * (width // pw)
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True
        )

    def forward(
        self, x: Float[torch.Tensor, "batch in_chans depth height width"]
    ) -> Float[torch.Tensor, "batch embed_dim depth_p height_p width_p"]:
        return self.proj(x)


class FinalLayer3D(nn.Module):
    r"""Final projection head: fp32 layer norm followed by a linear decoder.

    Normalizes the token features and linearly maps each token to ``out_features``
    output channels. The norm and linear run in fp32 (autocast disabled) for
    numerical stability of the output head. Shared by the patch-level decoder
    (``out_features = p_d * p_h * p_w * C_out``, later unpatchified to a field) and
    the pixel-level decoder (``out_features = C_out``).

    Parameters
    ----------
    hidden_size : int
        Token embedding dimension.
    out_features : int
        Number of output channels produced per token.

    Forward
    -------
    x : torch.Tensor
        Input tokens of shape :math:`(B, N, \text{hidden\_size})`.

    Outputs
    -------
    torch.Tensor
        Per-token outputs of shape :math:`(B, N, \text{out\_features})`.
    """

    def __init__(self, hidden_size: int, out_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_features)

    def forward(
        self, x: Float[torch.Tensor, "batch tokens hidden_size"]
    ) -> Float[torch.Tensor, "batch tokens out_features"]:
        # Force the output head to fp32 regardless of any outer autocast context.
        # Match the linear input to the weight dtype so the model also works when
        # cast wholesale to bf16/half (model.bfloat16()); under fp32 weights (the
        # common autocast path) the cast is a no-op and the result is unchanged.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.norm(x.float())
            x = self.linear(x.to(self.linear.weight.dtype))
        return x


# ---------------------------------------------------------------------------
# Depthwise 2D convolution with a torch.vmap fallback for very large tensors.
#
# torch.nn.functional.conv2d has an internal element-count limit that a
# depthwise convolution over a high-resolution pixel grid (Strata's
# "bilinear_dw" adaptive-layer-norm path) can exceed; DepthwiseConv optionally
# chunks the convolution with torch.vmap to stay under that limit.
# ---------------------------------------------------------------------------


def _apply_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    stride,
    padding,
    dilation,
    padding_mode,
) -> torch.Tensor:
    r"""Apply a single-sample 2D convolution (helper for :func:`torch.vmap`).

    Parameters
    ----------
    x : torch.Tensor
        Single-sample input of shape :math:`(C, H, W)`.
    weight : torch.Tensor
        Convolution weight of shape :math:`(C, 1, k_h, k_w)`.
    bias : torch.Tensor
        Bias of shape :math:`(C,)`.
    stride, padding, dilation : tuple
        Standard convolution parameters.
    padding_mode : str
        Padding mode; non-``"zeros"`` modes are applied explicitly with
        :func:`torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Convolved single-sample output of shape :math:`(C, H', W')`.
    """
    x = x.unsqueeze(0)
    w = weight.unsqueeze(0)
    bias = bias.unsqueeze(0)
    if padding_mode != "zeros":
        pad_h, pad_w = padding
        x = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=padding_mode)
        padding = (0, 0)
    return torch.nn.functional.conv2d(
        x, w, bias=bias, stride=stride, padding=padding, dilation=dilation
    )[0]


def _build_chunked_depthwise_conv(conv: nn.Conv2d, chunk_size: int = 4):
    r"""Build a chunked ``torch.vmap`` depthwise convolution callable.

    The returned callable takes ``(x, weight, bias)`` explicitly and captures
    only the static convolution configuration (stride / padding / dilation /
    padding mode) -- **not** the module or its parameters. The forward pass
    threads ``self.weight`` / ``self.bias`` in live, so the callable stays
    correct across ``deepcopy`` (e.g. EMA / ``AveragedModel``) and ``.to(device)``.

    Parameters
    ----------
    conv : torch.nn.Conv2d
        A depthwise convolution (``groups == out_channels``); used only to read
        its static configuration.
    chunk_size : int, optional, default=4
        Channel chunk size for the inner :func:`torch.vmap`.

    Returns
    -------
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
        A function mapping ``(x, weight, bias)`` with
        :math:`x \in (B, C, H, W)` to :math:`(B, C, H', W')`.
    """
    if conv.groups != conv.out_channels:
        raise ValueError("only works with depthwise convolution")

    # Inner vmap over channels (chunked); outer vmap over the batch dim with the
    # weight / bias broadcast (``in_dims=(0, None, None)``). This is equivalent
    # to mapping ``func(x[i], weight, bias)`` over the batch, but keeps weight
    # and bias as explicit arguments rather than closed-over module state.
    per_sample = torch.vmap(
        partial(
            _apply_conv2d,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            padding_mode=conv.padding_mode,
        ),
        chunk_size=chunk_size,
    )
    return torch.vmap(per_sample, in_dims=(0, None, None))


class DepthwiseConv(torch.nn.Conv2d):
    r"""Depthwise 2D convolution that chunks large inputs via ``torch.vmap``.

    A :class:`torch.nn.Conv2d` with ``groups == channels``. When ``chunk_size``
    is provided, the forward pass uses a chunked :func:`torch.vmap`
    implementation (see :func:`_build_chunked_depthwise_conv`) that avoids the
    conv2d element-count limit for very large tensors; otherwise it falls back
    to the standard convolution and warns if a single chunk would exceed the
    limit.

    Parameters
    ----------
    channels : int
        Number of input / output channels (the convolution is depthwise).
    *args
        Positional arguments forwarded to :class:`torch.nn.Conv2d` (e.g.
        ``kernel_size``).
    chunk_size : int, optional, default=None
        Channel chunk size for the ``torch.vmap`` path. If ``None``, the
        standard convolution is used.
    **kwargs
        Keyword arguments forwarded to :class:`torch.nn.Conv2d`. A ``groups``
        argument is rejected (the convolution is always depthwise).

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, H, W)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C, H', W')`.
    """

    def __init__(self, channels: int, *args, chunk_size: int | None = None, **kwargs):
        if "groups" in kwargs:
            raise ValueError("DepthwiseConv does not accept a groups argument")
        super().__init__(channels, channels, *args, **kwargs, groups=channels)
        self.chunk_size = chunk_size
        # Build the chunked callable once from the static conv configuration. It
        # does not capture ``self`` or the parameters, so ``forward`` can thread
        # ``self.weight`` / ``self.bias`` in live -- this keeps the module correct
        # after ``deepcopy`` (EMA / ``AveragedModel``) and ``.to(device)``, unlike
        # binding a ``self``-capturing closure to ``self.forward``.
        self._chunked_conv = (
            _build_chunked_depthwise_conv(self, chunk_size)
            if chunk_size is not None
            else None
        )

    def forward(self, x, *args, **kwargs):
        if self._chunked_conv is not None:
            # Chunked path: read parameters live so a deep-copied module uses its
            # own (possibly relocated / re-initialized) weights, not the source's.
            bias = self.bias
            if bias is None:
                bias = torch.zeros(
                    self.out_channels, device=self.weight.device, dtype=self.weight.dtype
                )
            return self._chunked_conv(x, self.weight, bias)

        # Standard (non-chunked) path: warn if conv2d would exceed its 32-bit
        # indexing limit (INT_MAX elements), then defer to nn.Conv2d. The limit is
        # element-count, not bytes -- it trips identically for fp32 and bf16.
        if x.numel() > torch.iinfo(torch.int32).max:
            warnings.warn(
                f"conv2d input has {x.numel()} elements (> 2**31 - 1, the 32-bit "
                f"indexing limit), so it will raise a RuntimeError "
                f"(canUse32BitIndexMath). Set the chunk_size option to enable "
                f"chunking, which keeps each conv2d call under the limit.",
                stacklevel=2,
            )

        return super().forward(x, *args, **kwargs)


class StrataPixel3DBlock(nn.Module):
    r"""Pixel-pathway transformer block with pixel-wise adaptive layer norm.

    Like a DiT block, but the adaptive-layer-norm (AdaLN) modulation parameters
    (``shift``, ``scale``, ``gate`` for both the attention and MLP sub-layers)
    vary *per pixel* and are derived from the backbone conditioning tokens
    ``backbone_cond``. This is a **regression** conditioning: unlike the original DiT's
    adaLN-Zero (which modulates on the diffusion timestep / noise level), here
    the modulation is a learned function of the backbone stage's features —
    there is no diffusion timestep or noise. Two derivation modes are supported
    (``adaln_mode``):

    - ``"pixel_proj"``: project each backbone token to
      ``pixels_per_patch * 6 * dim`` values and scatter them to the pixels of
      that patch.
    - ``"bilinear_dw"``: trilinearly upsample the backbone tokens to pixel
      resolution (over depth and the horizontal plane), smooth horizontal patch
      seams with a per-level depthwise :math:`5\times5` convolution,
      RMS-normalize, apply GeLU, then project to ``6 * dim``.

    Modulation projections are zero-initialized so the pixel pathway starts as
    an identity residual mapping.

    The pixel-wise AdaLN is an independent reimplementation of the conditioning
    in PixelDiT (`arXiv:2511.20645 <https://arxiv.org/abs/2511.20645>`_); the
    ``"bilinear_dw"`` mode is an original addition beyond that work.

    Parameters
    ----------
    dim : int
        Pixel-pathway embedding dimension.
    cond_dim : int
        Backbone-pathway (conditioning) embedding dimension.
    pixels_per_patch : int
        Number of pixels per backbone patch (:math:`p_d \cdot p_h \cdot p_w`);
        used by the ``"pixel_proj"`` mode.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float, optional, default=4.0
        Ratio of MLP hidden dimension to ``dim``.
    qkv_bias : bool, optional, default=True
        Whether attention QKV projections use a bias.
    qk_norm : bool, optional, default=False
        Whether to RMS-normalize attention queries and keys.
    qk_norm_affine : bool, optional, default=False
        Whether the QK RMS norms use a learnable affine scale.
    mlp_drop_rate : float, optional, default=0.0
        Dropout probability inside the MLP and attention output projection.
    attn_drop_rate : float, optional, default=0.0
        Dropout probability on attention weights.
    attn_kernel : int | Tuple[int, int, int], optional, default=-1
        Neighborhood-attention window; ``-1`` selects full attention.
    na_dilation : int, optional, default=1
        Dilation for neighborhood attention.
    gated_attention : bool, optional, default=False
        Whether attention outputs are multiplied by a learned sigmoid gate.
    na3d_backend : str, optional, default=None
        NATTEN backend forwarded to :class:`Natten3DSelfAttention`.
    adaln_mode : Literal["pixel_proj", "bilinear_dw"], optional, default="pixel_proj"
        How per-pixel modulation parameters are produced from ``backbone_cond``.
    chunk_size_grouped_conv : int, optional, default=2
        ``torch.vmap`` chunk size for the depthwise conv (``"bilinear_dw"`` only).
    use_chunked_depthwise_conv : bool, optional, default=True
        Whether the ``"bilinear_dw"`` depthwise conv uses the chunked
        :class:`DepthwiseConv`; if ``False``, a plain grouped
        :class:`torch.nn.Conv2d` is used.

    Forward
    -------
    x : torch.Tensor
        Pixel tokens of shape :math:`(B, D \cdot H \cdot W, \text{dim})`.
    backbone_cond : torch.Tensor
        Backbone conditioning tokens of shape :math:`(B, N_s, \text{cond\_dim})`.
    pixel_dhw : Tuple[int, int, int]
        Full pixel-grid shape :math:`(D, H, W)`.
    backbone_dhw : Tuple[int, int, int]
        Backbone patch-grid shape :math:`(s_d, s_h, s_w)`.
    s_cond_bilinear : torch.Tensor, optional
        Shared bilinear-upsampled conditioning (``"bilinear_dw"`` mode only),
        as produced by :meth:`precompute_bilinear_cond`.
    rope_tables : Tuple[torch.Tensor, torch.Tensor], optional
        Precomputed ``(cos, sin)`` RoPE tables.

    Outputs
    -------
    torch.Tensor
        Pixel tokens of shape :math:`(B, D \cdot H \cdot W, \text{dim})`.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        pixels_per_patch: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        qk_norm_affine: bool = False,
        mlp_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        attn_kernel: Union[int, Tuple[int, int, int]] = -1,
        na_dilation: int = 1,
        gated_attention: bool = False,
        na3d_backend: Optional[str] = None,
        adaln_mode: Literal["pixel_proj", "bilinear_dw"] = "pixel_proj",
        chunk_size_grouped_conv: int = 2,
        use_chunked_depthwise_conv: bool = True,
    ):
        super().__init__()
        if adaln_mode not in ("pixel_proj", "bilinear_dw"):
            raise ValueError(
                f"adaln_mode must be 'pixel_proj' or 'bilinear_dw'; got {adaln_mode!r}"
            )
        self.adaln_mode = adaln_mode
        self.num_adaln_params = 6  # shift1, scale1, gate1, shift2, scale2, gate2

        self.attn = Natten3DSelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            qk_norm_affine=qk_norm_affine,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=mlp_drop_rate,
            attn_kernel=attn_kernel,
            do_depthwise_attention=False,
            na_dilation=na_dilation,
            gated_attention=gated_attention,
            na3d_backend=na3d_backend,
        )
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_layer=nn.GELU,
            drop=mlp_drop_rate,
        )

        if adaln_mode == "bilinear_dw":
            # Bilinear-upsample -> depthwise 5x5 smoothing -> RMSNorm -> GeLU ->
            # zero-init projection, all at pixel resolution.
            if use_chunked_depthwise_conv:
                self.adaln_bilinear_dw_conv = DepthwiseConv(
                    cond_dim,
                    chunk_size=chunk_size_grouped_conv,
                    kernel_size=(5, 5),
                    padding=(2, 2),
                    bias=True,
                    padding_mode="replicate",
                )
            else:
                self.adaln_bilinear_dw_conv = nn.Conv2d(
                    cond_dim,
                    cond_dim,
                    kernel_size=(5, 5),
                    padding=(2, 2),
                    groups=cond_dim,
                    bias=True,
                    padding_mode="replicate",
                )
            # Identity-init: smoothing starts as a no-op (center tap = 1).
            nn.init.zeros_(self.adaln_bilinear_dw_conv.weight)
            self.adaln_bilinear_dw_conv.weight.data[:, 0, 2, 2] = 1.0
            nn.init.zeros_(self.adaln_bilinear_dw_conv.bias)
            self.adaln_bilinear_dw_norm = nn.RMSNorm(cond_dim, elementwise_affine=False)
            self.adaln_bilinear_dw_proj = nn.Linear(
                cond_dim, self.num_adaln_params * dim
            )
            nn.init.zeros_(self.adaln_bilinear_dw_proj.weight)
            nn.init.zeros_(self.adaln_bilinear_dw_proj.bias)
        else:
            self.pixels_per_patch = pixels_per_patch
            self.adaln_pixel_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, pixels_per_patch * self.num_adaln_params * dim),
            )
            nn.init.zeros_(self.adaln_pixel_proj[-1].weight)
            nn.init.zeros_(self.adaln_pixel_proj[-1].bias)

    def reset_adaln_zero(self) -> None:
        r"""Re-zero the AdaLN projection so the block starts as an identity residual.

        The block zero-inits its AdaLN projection at construction; call this after
        a blanket Xavier pass (e.g. in :meth:`Strata.initialize_weights`) that
        would otherwise clobber it. The depthwise smoothing conv keeps its
        identity init (a Conv2d, untouched by an ``nn.Linear`` Xavier pass).
        """
        if hasattr(self, "adaln_bilinear_dw_proj"):
            nn.init.zeros_(self.adaln_bilinear_dw_proj.weight)
            nn.init.zeros_(self.adaln_bilinear_dw_proj.bias)
        if hasattr(self, "adaln_pixel_proj"):
            nn.init.zeros_(self.adaln_pixel_proj[-1].weight)
            nn.init.zeros_(self.adaln_pixel_proj[-1].bias)

    @staticmethod
    def _expand_cond_to_pixels(
        adaln_raw: torch.Tensor,
        pixel_dhw: Tuple[int, int, int],
        backbone_dhw: Tuple[int, int, int],
    ) -> torch.Tensor:
        r"""Scatter per-patch modulation values to per-pixel order.

        Parameters
        ----------
        adaln_raw : torch.Tensor
            Per-patch modulation of shape
            :math:`(B, N_s, p_{ppp} \cdot 6 \cdot \text{dim})`.
        pixel_dhw : Tuple[int, int, int]
            Full pixel-grid shape :math:`(D, H, W)`.
        backbone_dhw : Tuple[int, int, int]
            Backbone patch-grid shape :math:`(s_d, s_h, s_w)`.

        Returns
        -------
        torch.Tensor
            Per-pixel modulation of shape :math:`(B, D \cdot H \cdot W, 6 \cdot \text{dim})`.
        """
        d, h, w = pixel_dhw
        sd, sh, sw = backbone_dhw
        pv, ph, pw = d // sd, h // sh, w // sw
        return rearrange(
            adaln_raw,
            "b (sd sh sw) (pv ph pw m) -> b (sd pv) (sh ph) (sw pw) m",
            sd=sd,
            sh=sh,
            sw=sw,
            pv=pv,
            ph=ph,
            pw=pw,
        ).reshape(adaln_raw.shape[0], d * h * w, -1)

    @staticmethod
    def precompute_bilinear_cond(
        backbone_cond: Float[torch.Tensor, "batch backbone_tokens cond_dim"],
        pixel_dhw: Tuple[int, int, int],
        backbone_dhw: Tuple[int, int, int],
    ) -> Float[torch.Tensor, "batch depth cond_dim height width"]:
        r"""Upsample backbone tokens to pixel resolution.

        Shared across all ``"bilinear_dw"`` blocks so the (expensive) upsample is
        computed once per forward pass. When the pixel depth equals the backbone
        depth (:math:`s_d = D`, i.e. no depth upsampling) it uses a per-level **2D
        bilinear** upsample — bit-identical to the original 2D-only path, so the
        common case (backbone vertical patch size 1, or any matching pixel/backbone
        depth) is numerically unchanged. This branch is deliberate, not just an
        optimization: a 3D trilinear upsample at :math:`s_d = D` agrees with the
        per-level 2D bilinear only to ~1e-7 (float), **not** exactly, so unifying on
        trilinear would silently shift the numerics of every matching-depth model.
        Only when the depth must actually be upsampled (:math:`s_d < D`) does it use
        a **3D trilinear** upsample over :math:`(D, H, W)`.

        Parameters
        ----------
        backbone_cond : torch.Tensor
            Backbone tokens of shape :math:`(B, N_s, C)`.
        pixel_dhw : Tuple[int, int, int]
            Full pixel-grid shape :math:`(D, H, W)`.
        backbone_dhw : Tuple[int, int, int]
            Backbone patch-grid shape :math:`(s_d, s_h, s_w)`.

        Returns
        -------
        torch.Tensor
            Upsampled conditioning of shape :math:`(B, D, C, H, W)`.
        """
        d, h, w = pixel_dhw  # full pixel-grid depth / height / width
        sd, sh, sw = backbone_dhw
        if d == sd:
            # Depth not upsampled (pixel depth == backbone depth): per-level 2D
            # bilinear. Bit-identical to the original 2D path (backward compatible).
            s_sp = rearrange(
                backbone_cond, "b (sd sh sw) c -> (b sd) c sh sw", sd=sd, sh=sh, sw=sw
            )
            s_pix = F.interpolate(
                s_sp, size=(h, w), mode="bilinear", align_corners=False
            )
            return rearrange(s_pix, "(b sd) c h w -> b sd c h w", sd=sd)
        # Depth upsampled (pixel depth > backbone depth): 3D trilinear over (D,H,W).
        s_sp = rearrange(backbone_cond, "b (sd sh sw) c -> b c sd sh sw", sd=sd, sh=sh, sw=sw)
        s_pix = F.interpolate(
            s_sp, size=(d, h, w), mode="trilinear", align_corners=False
        )
        return rearrange(s_pix, "b c d h w -> b d c h w")

    def _compute_bilinear_dw_adaln_params(
        self, s_cond_bilinear: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        r"""Compute the six per-pixel modulation tensors from upsampled conditioning.

        Parameters
        ----------
        s_cond_bilinear : torch.Tensor
            Trilinearly-upsampled conditioning of shape :math:`(B, D, C, H, W)`
            (see :meth:`precompute_bilinear_cond`).

        Returns
        -------
        Tuple[torch.Tensor, ...]
            The six modulation tensors ``(shift1, scale1, gate1, shift2, scale2,
            gate2)``, each of shape :math:`(B, D \cdot H \cdot W, \text{dim})`.
        """
        bsz, d, _, _, _ = s_cond_bilinear.shape
        s_pix = rearrange(s_cond_bilinear, "b d c h w -> (b d) c h w")
        s_pix = self.adaln_bilinear_dw_conv(s_pix)
        # Channel-last RMSNorm, then back to channel-first.
        s_pix = self.adaln_bilinear_dw_norm(s_pix.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )
        s_pix = rearrange(s_pix, "(b d) c h w -> b d c h w", b=bsz, d=d)
        s_pix_seq = rearrange(s_pix, "b d c h w -> b (d h w) c")
        s_pix_seq = F.gelu(s_pix_seq)
        adaln_params = self.adaln_bilinear_dw_proj(s_pix_seq)
        adaln_params = rearrange(
            adaln_params, "b n (six c) -> b n six c", six=self.num_adaln_params
        )
        return adaln_params.unbind(dim=-2)

    def forward(
        self,
        x: Float[torch.Tensor, "batch pixels dim"],
        backbone_cond: Float[torch.Tensor, "batch patches cond_dim"],
        pixel_dhw: Tuple[int, int, int],
        backbone_dhw: Tuple[int, int, int],
        s_cond_bilinear: Optional[torch.Tensor] = None,
        rope_tables: Optional[RopeTables] = None,
    ) -> Float[torch.Tensor, "batch pixels dim"]:
        # Derive the six per-pixel modulation tensors from the conditioning.
        if self.adaln_mode == "bilinear_dw":
            if s_cond_bilinear is None:
                raise ValueError(
                    "s_cond_bilinear is required when adaln_mode='bilinear_dw'"
                )
            shift1, scale1, gate1, shift2, scale2, gate2 = (
                self._compute_bilinear_dw_adaln_params(s_cond_bilinear)
            )
        else:
            adaln_raw = self.adaln_pixel_proj(backbone_cond)  # (B, N_s, ppp*6*dim)
            adaln_params = self._expand_cond_to_pixels(
                adaln_raw, pixel_dhw, backbone_dhw
            )  # (B, D*H*W, 6*dim)
            shift1, scale1, gate1, shift2, scale2, gate2 = adaln_params.chunk(
                self.num_adaln_params, dim=-1
            )

        # Attention sub-layer with modulated pre-norm and gated residual.
        y = self.norm1(x) * (1 + scale1) + shift1
        z = self.attn(y, latent_dhw=pixel_dhw, rope_tables=rope_tables)
        x = x + gate1 * z

        # MLP sub-layer with modulated pre-norm and gated residual.
        y = self.norm2(x) * (1 + scale2) + shift2
        z = self.mlp(y)
        x = x + gate2 * z
        return x
