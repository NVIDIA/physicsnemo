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

r"""Building blocks for the :class:`~physicsnemo.experimental.models.strata.DiT3D`
transformer.

These layers are specific to the DiT3D / PixelDiT models and are kept in the
model package (rather than ``physicsnemo.nn``) per the self-contained-model
convention. The attention layer reuses
:func:`physicsnemo.nn.functional.na3d` for 3D neighborhood attention so that it
inherits NATTEN optional-dependency handling and ``ShardTensor`` dispatch.

DiT3D / PixelDiT reuse the Diffusion-Transformer (DiT) architecture but are
deterministic regression models, not generative diffusion models — these blocks
carry no diffusion / timestep conditioning (:class:`DiT3DBlock` is a plain
pre-norm transformer block).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

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
    "DiT3DBlock",
    "PatchEmbed3D",
    "FinalLayer3D",
]

# A pair of (cos, sin) RoPE lookup tables, as produced by
# ``StereographicRotaryPositionEmbedding2D.build_tables``.
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
        # DiT3D, PixelDiT, and direct use — fail at construction, not deep inside
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


class DiT3DBlock(nn.Module):
    r"""Pre-norm transformer block for DiT3D.

    Applies, with residual connections, a :class:`Natten3DSelfAttention`
    sub-layer followed by an MLP sub-layer (reusing
    :class:`physicsnemo.nn.Mlp`). Layer norms are non-affine, matching the
    standard DiT block. Unlike the diffusion DiT block, no adaLN conditioning is
    used (DiT3D is a deterministic field-to-field model).

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
    r"""Final projection head: fp32 layer norm followed by a linear patch decoder.

    Normalizes the token features and linearly maps each token to the flattened
    pixel block it represents (:math:`p_d \cdot p_h \cdot p_w \cdot C_{out}`
    channels). The norm and linear run in fp32 (autocast disabled) for numerical
    stability of the output head.

    Parameters
    ----------
    hidden_size : int
        Token embedding dimension.
    patch_size : Tuple[int, int, int]
        The ``(p_d, p_h, p_w)`` patch size used by the tokenizer.
    out_chans : int
        Number of output field channels :math:`C_{out}`.

    Forward
    -------
    x : torch.Tensor
        Input tokens of shape :math:`(B, N, \text{hidden\_size})`.

    Outputs
    -------
    torch.Tensor
        Per-token patch pixels of shape
        :math:`(B, N, p_d \cdot p_h \cdot p_w \cdot C_{out})`.
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: Tuple[int, int, int],
        out_chans: int,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * out_chans
        )

    def forward(
        self, x: Float[torch.Tensor, "batch tokens hidden_size"]
    ) -> Float[torch.Tensor, "batch tokens patch_pixels"]:
        # Force the output head to fp32 regardless of any outer autocast context.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.norm(x.float())
            x = self.linear(x)
        return x
