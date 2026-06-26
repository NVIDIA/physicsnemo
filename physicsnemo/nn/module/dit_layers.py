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
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from timm.layers import RmsNorm
from timm.layers.attention import Attention

from physicsnemo.core import Module
from physicsnemo.core.version_check import OptionalImport
from physicsnemo.nn.functional.natten import na2d as _na2d_func
from physicsnemo.nn.module.drop import DropPath
from physicsnemo.nn.module.hpx.tokenizer import (
    HEALPixPatchDetokenizer,
    HEALPixPatchTokenizer,
)
from physicsnemo.nn.module.mlp_layers import Mlp
from physicsnemo.nn.module.rope import (
    apply_rotary_pos_emb,
    build_axial_rope_cos_sin_2d,
)
from physicsnemo.nn.module.utils import PatchEmbed2D

te = OptionalImport("transformer_engine.pytorch")
apex_normalization = OptionalImport("apex.normalization")

TE_AVAILABLE = te.available
APEX_AVAILABLE = apex_normalization.available


def get_layer_norm(
    hidden_size: int,
    layernorm_backend: Literal["apex", "torch"],
    elementwise_affine: bool = False,
    eps: float = 1e-6,
) -> nn.Module:
    r"""Construct a LayerNorm module based on the selected backend.

    Parameters
    ----------
    hidden_size : int
        Normalized feature dimension.
    layernorm_backend : Literal["apex", "torch"]
        Implementation selector.
    elementwise_affine : bool, optional, default=False
        Whether to learn per-element affine parameters.
    eps : float, optional, default=1e-6
        Numerical stability epsilon.

    Returns
    -------
    nn.Module
        A configured LayerNorm module from Apex or Torch. The returned module is a subclass of ``nn.Module``
        and expects a tensor of shape :math:`(B, L, D)` as input, returning a normalized tensor of the same shape.
    """
    if layernorm_backend == "apex":
        if not APEX_AVAILABLE:
            raise ImportError(
                "Apex is not available. Please install Apex to use FusedLayerNorm or choose 'torch'."
            )
        return apex_normalization.FusedLayerNorm(
            hidden_size, elementwise_affine=elementwise_affine, eps=eps
        )
    if layernorm_backend == "torch":
        return nn.LayerNorm(hidden_size, elementwise_affine=elementwise_affine, eps=eps)
    raise ValueError("layernorm_backend must be one of 'apex' or 'torch'.")


def get_attention(
    hidden_size: int,
    num_heads: int,
    attention_backend: Literal[
        "transformer_engine", "timm", "natten2d", "natten2d_rope"
    ],
    attn_drop_rate: float = 0.0,
    proj_drop_rate: float = 0.0,
    **attn_kwargs: Any,
) -> Module:
    r"""Construct a pre-defined attention module for DiT.

    Parameters
    ----------
    hidden_size : int
        The embedding dimension.
    num_heads : int
        Number of attention heads.
    attention_backend : Literal["transformer_engine", "timm", "natten2d", "natten2d_rope"]
        One of ``"timm"``, ``"transformer_engine"``, ``"natten2d"``, or ``"natten2d_rope"`` to select between pre-defined attention modules. ``"natten2d_rope"`` is :class:`Natten2DSelfAttention` with axial 2D rotary position embeddings (see :class:`RopeNatten2DSelfAttention`) and requires ``latent_hw`` in ``attn_kwargs``.
    attn_drop_rate : float, optional, default=0.0
        The dropout rate for the attention operation.
    proj_drop_rate : float, optional, default=0.0
        The dropout rate for the projection operation.
    **attn_kwargs : Any
        Additional keyword arguments for the attention module.

    Returns
    -------
    Module
        A module whose forward accepts :math:`(B, L, D)` and returns :math:`(B, L, D)`.
    """
    if attention_backend == "timm":
        return TimmSelfAttention(
            hidden_size,
            num_heads,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            **attn_kwargs,
        )
    if attention_backend == "transformer_engine":
        return TESelfAttention(
            hidden_size,
            num_heads,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            **attn_kwargs,
        )
    if attention_backend == "natten2d":
        return Natten2DSelfAttention(
            hidden_size,
            num_heads,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            **attn_kwargs,
        )
    if attention_backend == "natten2d_rope":
        return RopeNatten2DSelfAttention(
            hidden_size,
            num_heads,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            **attn_kwargs,
        )
    raise ValueError(
        "attention_backend must be one of 'timm', 'transformer_engine', 'natten2d', 'natten2d_rope' if using pre-defined attention modules."
    )


class AttentionModuleBase(Module, ABC):
    r"""Abstract base class for attention modules used in DiTBlock.

    Implementations must define a forward method that accepts a single tensor of shape
    :math:`(B, L, D)` and returns a tensor of the same shape.
    Subclasses must implement the forward method, and may add additional input arguments
    as needed.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, L, D)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, L, D)`.
    """

    @abstractmethod
    def forward(
        self, x: Float[torch.Tensor, "batch sequence hidden_size"]
    ) -> Float[torch.Tensor, "batch sequence hidden_size"]:
        pass


class TimmSelfAttention(AttentionModuleBase):
    r"""Self-attention module using the timm library implementation.

    Expects an input tensor of shape :math:`(B, L, D)` and returns a tensor of the same shape.
    Under the hood, timm uses :func:`torch.nn.functional.scaled_dot_product_attention` for the attention operation.

    Parameters
    ----------
    hidden_size : int
        The embedding dimension.
    num_heads : int
        Number of attention heads.
    attn_drop_rate : float, optional, default=0.0
        The dropout rate for the attention operation.
    proj_drop_rate : float, optional, default=0.0
        The dropout rate for the projection operation.
    qk_norm_type : Literal["RMSNorm", "LayerNorm"] or None, optional
        QK normalization type. Options: ``"RMSNorm"``, ``"LayerNorm"``, or ``None``.
    qk_norm_affine : bool, optional, default=True
        Whether QK normalization layers should use learnable affine parameters.
    is_causal : bool, optional, default=False
        Whether to apply a causal self-attention mask, so that each position
        attends only to itself and earlier positions in the sequence. Uses
        timm's native causal path (``torch.nn.functional.scaled_dot_product_attention``
        with ``is_causal=True``), which builds the lower-triangular mask
        internally for the current sequence length.
    **kwargs : Any
        Additional keyword arguments for the timm attention module.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, L, D)`.
    attn_mask : torch.Tensor, optional
        The attention mask to apply (passed to timm's Attention module).
        If ``None``, no mask is applied. Mutually exclusive with
        ``is_causal=True``; passing both raises a ``ValueError``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, L, D)`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        qk_norm_type: Literal["RMSNorm", "LayerNorm"] | None = None,
        qk_norm_affine: bool = True,
        is_causal: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.is_causal = is_causal

        # Translate qk_norm_type to timm's qk_norm and norm_layer
        if qk_norm_type == "RMSNorm":
            kwargs["qk_norm"] = True
            kwargs["norm_layer"] = partial(RmsNorm, affine=qk_norm_affine)
        elif qk_norm_type == "LayerNorm":
            kwargs["qk_norm"] = True
            kwargs["norm_layer"] = partial(
                nn.LayerNorm, elementwise_affine=qk_norm_affine
            )

        self.attn_op = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=proj_drop_rate,
            qkv_bias=True,
            **kwargs,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch sequence hidden_size"],
        attn_mask: Optional[Float[torch.Tensor, "..."]] = None,
    ) -> Float[torch.Tensor, "batch sequence hidden_size"]:
        if attn_mask is not None and self.is_causal:
            raise ValueError(
                "attn_mask cannot be combined with is_causal=True in TimmSelfAttention; "
                "they are mutually exclusive."
            )

        return self.attn_op(x, attn_mask=attn_mask, is_causal=self.is_causal)


class TESelfAttention(AttentionModuleBase):
    r"""Self-attention module using the transformer_engine library implementation.

    Expects an input tensor of shape :math:`(B, L, D)` and returns a tensor of the same shape.

    Parameters
    ----------
    hidden_size : int
        The embedding dimension.
    num_heads : int
        Number of attention heads.
    attn_drop_rate : float, optional, default=0.0
        The dropout rate for the attention operation.
    proj_drop_rate : float, optional, default=0.0
        The dropout rate for the projection operation.
    qkv_format : str, optional, default="bshd"
        Dimension format for Q/K/V tensors. Use ``"bshd"`` for batch-first layout, ``"sbhd"`` for sequence-first layout.
    **kwargs : Any
        Additional keyword arguments for the transformer_engine attention module.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, L, D)`.
    attn_mask : torch.Tensor, optional
        The attention mask to apply (passed to transformer_engine's MultiheadAttention).
        If ``None``, no mask is applied.
    mask_type : str, optional, default="no_mask"
        The type of mask (passed to transformer_engine's MultiheadAttention).
        If no mask is provided, ``"no_mask"`` is used.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, L, D)`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        qkv_format: str = "bshd",
        **kwargs: Any,
    ):
        super().__init__()
        if not TE_AVAILABLE:
            raise ImportError(
                "Transformer Engine is not installed. Please install it with `pip install transformer-engine`."
            )

        if "qk_norm_affine" in kwargs and not kwargs["qk_norm_affine"]:
            warnings.warn(
                "Transformer Engine does not support disabling affine parameters for QK norm. "
                "Ignoring qk_norm_affine=False and using affine parameters.",
                UserWarning,
                stacklevel=2,
            )
        kwargs.pop("qk_norm_affine", None)
        self.attn_op = te.MultiheadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            attention_dropout=attn_drop_rate,
            qkv_format=qkv_format,
            **kwargs,
        )
        # TE doesn't support proj_drop natively, so we add it manually after the attention output
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(
        self,
        x: Float[torch.Tensor, "batch sequence hidden_size"],
        attn_mask: Optional[Float[torch.Tensor, "..."]] = None,
        mask_type: Optional[str] = "no_mask",
    ) -> Float[torch.Tensor, "batch sequence hidden_size"]:
        if attn_mask is not None:
            mask_type = "arbitrary"
        out = self.attn_op(x, attention_mask=attn_mask, attn_mask_type=mask_type)
        return self.proj_drop(out)


class Natten2DSelfAttention(AttentionModuleBase):
    r"""
    Self-attention module that performs 2D neighborhood attention using NATTEN.

    Expects an input tensor of shape :math:`(B, L, D)` and returns a tensor of the same shape
    (reshapes sequence to 2D internally for the attention operation).

    Parameters
    ----------
    hidden_size : int
        The embedding dimension.
    num_heads : int
        Number of attention heads.
    attn_kernel : int, optional, default=3
        The kernel size for the NATTEN neighborhood attention.
    qkv_bias : bool, optional, default=True
        Whether to use bias in the QKV projection.
    qk_norm : bool, optional, default=False
        Whether to use layer normalization on the query and key. When ``True``, the ``norm_layer`` backend is used (e.g. ``"apex"`` or ``"torch"``).
    attn_drop_rate : float, optional, default=0.0
        The dropout rate for the attention operation.
    proj_drop_rate : float, optional, default=0.0
        The dropout rate for the output projection.
    norm_layer : Literal["apex", "torch"], optional, default="torch"
        The layer normalization backend for QK norm when ``qk_norm=True``. When used inside :class:`~physicsnemo.nn.module.dit_layers.DiTBlock` with ``attention_backend="natten2d"``, this is set from the block's ``layernorm_backend``.
    na2d_kwargs : Dict[str, Any], optional, default=None
        Optional keyword arguments forwarded to :func:`physicsnemo.nn.functional.na2d` for performance tuning (e.g. ``dilation``, ``is_causal``, ``scale``). If ``None``, an empty dict is used.
    use_mask_token : bool, optional, default=False
        If ``True``, allocate a per-block learned ``mask_token`` parameter of shape :math:`(1, 1, D)`. Spatial tokens flagged by the ``invalid_token_mask`` passed to ``forward`` are replaced by this learned token immediately before the QKV projection, so the neighborhood window mixes in a single learned feature instead of corrupted (e.g. NaN-padded) signal. Initialized to zero so the first forward is numerically identical to the unmasked path.

    References
    ----------
    - `Neighborhood Attention Transformer <https://arxiv.org/abs/2204.07143>`_
    - `NATTEN <https://natten.org/>`_

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, L, D)`.
    latent_hw : Tuple[int, int]
        The height and width of the 2D latent space for reshaping. Sequence length must equal ``latent_hw[0] * latent_hw[1]``.
    invalid_token_mask : torch.Tensor, optional
        Boolean (or float) mask of shape :math:`(L,)` (shared across the batch) or :math:`(B, L)` (per-sample), ``True`` (or ``1``) at spatial token positions to overwrite with the learned ``mask_token`` before QKV. The per-sample form supports dynamic, batch-dimension-variable masking. Ignored when ``use_mask_token=False`` or when ``None``. Invalid positions in ``x`` must be finite (sanitize NaNs beforehand); see :meth:`_apply_mask_token`.

    Returns
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, L, D)`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.nn.module.dit_layers import Natten2DSelfAttention
    >>> attn = Natten2DSelfAttention(hidden_size=64, num_heads=4, attn_kernel=3).cuda()
    >>> x = torch.randn(2, 16, 64, device="cuda")
    >>> out = attn(x, latent_hw=(4, 4))
    >>> out.shape
    torch.Size([2, 16, 64])
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_kernel: int = 3,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        norm_layer: Literal["apex", "torch"] = "torch",
        na2d_kwargs: Optional[Dict[str, Any]] = None,
        use_mask_token: bool = False,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop_rate = proj_drop_rate
        self.norm_layer = norm_layer
        self.attn_kernel = attn_kernel
        self.na2d_kwargs = na2d_kwargs if na2d_kwargs is not None else {}

        # Per-block learned token used to overwrite invalid spatial tokens
        # immediately before QKV. Init to zero so the first forward matches the
        # unmasked path and only diverges as gradients shape the token.
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, hidden_size)) if use_mask_token else None
        )

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        if qk_norm:
            self.q_norm = get_layer_norm(self.head_dim, norm_layer)
            self.k_norm = get_layer_norm(self.head_dim, norm_layer)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.proj = nn.Linear(hidden_size, hidden_size)

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def _apply_mask_token(
        self,
        x: Float[torch.Tensor, "batch sequence hidden_size"],
        invalid_token_mask: Optional[
            Union[
                Float[torch.Tensor, " sequence"],
                Float[torch.Tensor, "batch sequence"],
            ]
        ],
    ) -> Float[torch.Tensor, "batch sequence hidden_size"]:
        r"""Replace invalid spatial tokens with the learned ``mask_token``.

        Implemented as ``x * (1 - alpha) + mask_token * alpha`` rather than
        :func:`torch.where`. For a boolean mask (``alpha`` in :math:`\{0, 1\}`)
        the result is bit-identical to ``torch.where`` but uses only elementwise
        multiply/add. This keeps every operand on the same tensor type, which is
        what makes it safe under domain-parallel sharding: when ``mask_token`` is
        a replicated DTensor and ``x``/``invalid_token_mask`` are sharded
        tensors, the arithmetic stays entirely within DTensor dispatch and never
        triggers a mixed plain-tensor / DTensor op.

        .. note::

            Because the splice multiplies ``x`` by ``(1 - alpha)``, an invalid
            token whose value is non-finite (e.g. ``NaN``) is **not** sanitized
            (``NaN * 0 == NaN``). Callers that flag NaN-padded regions must
            therefore replace those values (e.g. via :func:`torch.nan_to_num`)
            *before* the forward pass, so ``x`` is finite at the masked
            positions.

        The mask may be shared across the batch (shape :math:`(L,)`) or
        per-sample (shape :math:`(B, L)`); the latter enables dynamic,
        batch-dimension-variable masking.
        """
        if self.mask_token is None or invalid_token_mask is None:
            return x
        alpha = invalid_token_mask.to(dtype=x.dtype)
        # (L,) -> (1, L, 1) broadcasts across the batch; (B, L) -> (B, L, 1)
        # applies a distinct pattern per sample.
        alpha = alpha.view(1, -1, 1) if alpha.ndim == 1 else alpha.unsqueeze(-1)
        return x * (1.0 - alpha) + self.mask_token.to(x.dtype) * alpha

    def forward(
        self,
        x: Float[torch.Tensor, "batch sequence hidden_size"],
        latent_hw: Tuple[int, int],
        invalid_token_mask: Optional[
            Union[
                Float[torch.Tensor, " sequence"],
                Float[torch.Tensor, "batch sequence"],
            ]
        ] = None,
    ) -> Float[torch.Tensor, "batch sequence hidden_size"]:
        B, N, C = x.shape
        h, w = latent_hw

        if not torch.compiler.is_compiling() and N != h * w:
            raise ValueError(
                f"Sequence length must be {h * w} based on latent_hw={latent_hw}, but got {N}"
            )

        # Overwrite invalid spatial tokens with the learned mask token before QKV.
        x = self._apply_mask_token(x, invalid_token_mask)

        # Project to query, key, value and split into heads
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Windowed neighborhood self-attention
        q, k, v = map(
            lambda x: rearrange(x, "b head (h w) c -> b h w head c", h=h),
            [q, k, v],
        )
        x = _na2d_func(q, k, v, kernel_size=self.attn_kernel, **self.na2d_kwargs)
        x = self.attn_drop(x)
        x = rearrange(x, "b h w head c -> b (h w) (head c)")

        x = self.proj_drop(self.proj(x))
        return x


class RopeNatten2DSelfAttention(Natten2DSelfAttention):
    r"""NATTEN 2D self-attention with axial 2D rotary position embeddings (RoPE) on Q, K.

    Subclass of :class:`Natten2DSelfAttention` that replaces an additive
    positional embedding with relative rotary embeddings applied to the query
    and key inside each attention block. Position is encoded purely as a
    relative rotation between query and key, making attention output
    translation-equivariant within each NATTEN window. When this backend is
    used, the model should disable any additive positional embedding to avoid
    double-counting the position signal.

    The cos/sin tables are precomputed at construction so that
    :func:`torch.compile` sees a stable forward graph. They are stored as
    ``persistent=False`` buffers because they are deterministically rebuilt from
    ``(latent_hw, head_dim, rope_theta)``; for domain-parallel training they are
    built at the global spatial size and sharded along height by
    ``distribute_module``, so each rank holds the rows with globally-correct
    frequencies and no explicit rank offset is needed in model code.

    Parameters
    ----------
    latent_hw : Tuple[int, int]
        Spatial size :math:`(h, w)` of the token grid used to precompute the
        cos/sin tables. For inference at a different shape, :meth:`forward`
        rebuilds the tables in place (off the ``torch.compile`` path).
    rope_theta : float, optional, default=10000.0
        Base used for the RoPE frequency schedule.
    *args, **kwargs
        Forwarded to :class:`Natten2DSelfAttention` (e.g. ``attn_kernel``,
        ``qk_norm``, ``use_mask_token``).

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, L, D)`.
    latent_hw : Tuple[int, int]
        The height and width of the 2D latent space for reshaping.
    invalid_token_mask : torch.Tensor, optional
        See :class:`Natten2DSelfAttention`.

    Returns
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, L, D)`.
    """

    def __init__(
        self,
        *args: Any,
        latent_hw: Tuple[int, int],
        rope_theta: float = 10000.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if self.head_dim % 4 != 0:
            raise ValueError(
                f"head_dim={self.head_dim} must be divisible by 4 for axial 2D RoPE."
            )
        self.rope_theta = float(rope_theta)
        self._latent_hw: Tuple[int, int] = (int(latent_hw[0]), int(latent_hw[1]))

        cos, sin = build_axial_rope_cos_sin_2d(
            *self._latent_hw, self.head_dim, theta=self.rope_theta
        )
        # persistent=False: not in state_dict (rebuilt deterministically at
        # __init__ from latent_hw + head_dim + theta), so checkpoints stay lean.
        self.register_buffer("rope_cos", cos, persistent=False)  # (h, w, head_dim)
        self.register_buffer("rope_sin", sin, persistent=False)  # (h, w, head_dim)

    def _rebuild_for_shape(self, h: int, w: int) -> None:
        r"""Rebuild the RoPE buffers for a new latent shape.

        Reached only when :meth:`forward` is called with a ``latent_hw`` that
        differs from construction (e.g. variable-resolution inference); not part
        of the training-time hot path.
        """
        target_dtype = self.rope_cos.dtype
        target_device = self.rope_cos.device
        cos, sin = build_axial_rope_cos_sin_2d(
            h, w, self.head_dim, theta=self.rope_theta, device=target_device
        )
        self.register_buffer("rope_cos", cos.to(dtype=target_dtype), persistent=False)
        self.register_buffer("rope_sin", sin.to(dtype=target_dtype), persistent=False)
        self._latent_hw = (int(h), int(w))

    def forward(
        self,
        x: Float[torch.Tensor, "batch sequence hidden_size"],
        latent_hw: Tuple[int, int],
        invalid_token_mask: Optional[
            Union[
                Float[torch.Tensor, " sequence"],
                Float[torch.Tensor, "batch sequence"],
            ]
        ] = None,
    ) -> Float[torch.Tensor, "batch sequence hidden_size"]:
        B, N, C = x.shape
        h, w = int(latent_hw[0]), int(latent_hw[1])
        if not torch.compiler.is_compiling() and N != h * w:
            raise ValueError(
                f"Sequence length must be {h * w} based on latent_hw={latent_hw}, but got {N}"
            )
        if (h, w) != self._latent_hw:
            self._rebuild_for_shape(h, w)

        # Overwrite invalid spatial tokens with the learned mask token before QKV.
        x = self._apply_mask_token(x, invalid_token_mask)

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Reshape Q, K to spatial layout for RoPE indexing.
        q_2d = q.reshape(B, self.num_heads, h, w, self.head_dim)
        k_2d = k.reshape(B, self.num_heads, h, w, self.head_dim)

        # cos/sin: (h, w, head_dim) -> (1, 1, h, w, head_dim) broadcasts over
        # batch and head axes. apply_rotary_pos_emb rotates in fp32 internally.
        cos = self.rope_cos.unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin.unsqueeze(0).unsqueeze(0)
        q_rot = apply_rotary_pos_emb(q_2d, cos, sin)
        k_rot = apply_rotary_pos_emb(k_2d, cos, sin)

        # NATTEN expects (B, h, w, num_heads, head_dim).
        q_nat = q_rot.permute(0, 2, 3, 1, 4).contiguous()
        k_nat = k_rot.permute(0, 2, 3, 1, 4).contiguous()
        v_nat = rearrange(v, "b head (h w) c -> b h w head c", h=h)

        x = _na2d_func(
            q_nat, k_nat, v_nat, kernel_size=self.attn_kernel, **self.na2d_kwargs
        )
        x = self.attn_drop(x)
        x = rearrange(x, "b h w head c -> b (h w) (head c)")

        x = self.proj_drop(self.proj(x))
        return x


class PerSampleDropout(nn.Module):
    r"""Dropout module supporting scalar or per-sample probabilities.

    Per-sample dropout uses a different dropout probability for each sample in the batch.

    Parameters
    ----------
    inplace : bool, optional, default=False
        Whether to perform the dropout in place.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, L, D)`.
    p : float or torch.Tensor, optional
        The dropout probability. If ``None``, no dropout is applied.
        If a scalar, the same probability is applied to all samples.
        If a tensor of shape :math:`(B,)`, per-sample dropout is applied.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, L, D)`.
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(
        self,
        x: Float[torch.Tensor, "batch sequence hidden_size"],
        p: Optional[float | Float[torch.Tensor, " batch"]] = None,
    ) -> Float[torch.Tensor, "batch sequence hidden_size"]:
        if (not self.training) or p is None:
            return x

        # Standard dropout if p is scalar-like
        if isinstance(p, (float, int)):
            drop_p = float(p)
            if drop_p <= 0.0:
                return x
            return F.dropout(x, p=drop_p, training=True, inplace=self.inplace)

        if not torch.is_tensor(p):
            raise TypeError("p must be a float, int, or torch.Tensor")

        if p.ndim == 0:
            drop_p = float(p.item())
            if drop_p <= 0.0:
                return x
            return F.dropout(x, p=drop_p, training=True, inplace=self.inplace)

        # Per-sample dropout path: p expected shape [B]
        batch_size = x.size(0)
        if p.numel() != batch_size:
            raise ValueError(
                f"Per-sample dropout expects p with numel == batch size ({batch_size}), got shape {tuple(p.shape)}"
            )

        # Broadcast keep probability across non-batch dims
        shape = [batch_size] + [1] * (x.ndim - 1)
        p = p.view(shape).to(device=x.device, dtype=x.dtype)
        keep_prob = (1.0 - p).clamp(min=1e-6)

        mask = (torch.rand_like(x) < keep_prob).to(x.dtype) / keep_prob
        if self.inplace:
            return x.mul_(mask)
        return x * mask


class DiTBlock(nn.Module):
    r"""A Diffusion Transformer (DiT) block with adaptive layer norm zero (adaLN-Zero) conditioning.

    Parameters
    ----------
    hidden_size : int
        The dimensionality of the input and output.
    num_heads : int
        The number of attention heads.
    attention_backend : Literal["timm", "transformer_engine", "natten2d"] or Module
        Either the name of a pre-defined attention implementation, or a user-provided Module implementing
        the :math:`(B, L, D) \rightarrow (B, L, D)` interface. Options: ``"timm"`` (see
        :class:`~physicsnemo.nn.module.dit_layers.TimmSelfAttention`), ``"transformer_engine"``
        (see :class:`~physicsnemo.nn.module.dit_layers.TESelfAttention`), ``"natten2d"``
        (see :class:`~physicsnemo.nn.module.dit_layers.Natten2DSelfAttention`). The expected
        interface is :class:`~physicsnemo.nn.module.dit_layers.AttentionModuleBase`. Default ``"timm"``.
    layernorm_backend : Literal["apex", "torch"], optional, default="torch"
        The layer normalization implementation.
    norm_eps : float, optional, default=1e-6
        Epsilon for layer normalization.
    mlp_ratio : float, optional, default=4.0
        The ratio for the MLP's hidden dimension.
    intermediate_dropout : bool, optional, default=False
        Whether to apply :class:`~physicsnemo.nn.module.dit_layers.PerSampleDropout` before attention.
    attn_drop_rate : float, optional, default=0.0
        The dropout rate for the attention operation.
    proj_drop_rate : float, optional, default=0.0
        The dropout rate for the projection operation.
    mlp_drop_rate : float, optional, default=0.0
        The dropout rate for the MLP operation.
    final_mlp_dropout : bool, optional, default=True
        Whether to apply final MLP dropout.
    drop_path : float, optional, default=0.0
        DropPath (stochastic depth) rate.
    condition_embed_dim : int, optional
        Input dimension of the adaptive layer norm (AdaLN) modulation. If ``None``, defaults to ``hidden_size``.
        Should match the output dimension of the conditioning embedder.
    **attn_kwargs : Any
        Additional keyword arguments for the attention module.

    Notes
    -----
    The attention module configured by ``attention_backend`` is not expected to be cross-compatible in terms of
    state_dict keys. The layer norm module configured by ``layernorm_backend`` is expected to be cross-compatible
    (models trained with ``torch`` layernorms can be loaded with ``apex`` layernorms and vice versa).

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, L, D)`.
    c : torch.Tensor
        Conditioning tensor of shape :math:`(B, D)`.
    attn_kwargs : Dict[str, Any], optional
        Additional keyword arguments for the attention module.
    p_dropout : float or torch.Tensor, optional
        Dropout probability for intermediate dropout. If ``None``, no dropout. If scalar, same for all samples.
        If tensor of shape :math:`(B,)`, per-sample dropout.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, L, D)`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_backend: Union[
            Literal["transformer_engine", "timm", "natten2d", "natten2d_rope"], Module
        ] = "timm",
        layernorm_backend: Literal["apex", "torch"] = "torch",
        norm_eps: float = 1e-6,
        mlp_ratio: float = 4.0,
        intermediate_dropout: bool = False,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        mlp_drop_rate: float = 0.0,
        final_mlp_dropout: bool = True,
        drop_path: float = 0.0,
        condition_embed_dim: Optional[int] = None,
        **attn_kwargs: Any,
    ):
        super().__init__()

        if isinstance(attention_backend, Module):
            self.attention = attention_backend
        else:
            attn_kwargs_final = dict(attn_kwargs)
            # Ensure the NATTEN attention modules use the same LayerNorm backend as the block when qk_norm is used
            if attention_backend in ("natten2d", "natten2d_rope"):
                attn_kwargs_final.setdefault("norm_layer", layernorm_backend)
            self.attention = get_attention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                attention_backend=attention_backend,
                attn_drop_rate=attn_drop_rate,
                proj_drop_rate=proj_drop_rate,
                **attn_kwargs_final,
            )

        self.pre_attention_norm = get_layer_norm(
            hidden_size, layernorm_backend, elementwise_affine=False, eps=norm_eps
        )
        self.pre_mlp_norm = get_layer_norm(
            hidden_size, layernorm_backend, elementwise_affine=False, eps=norm_eps
        )

        # Optional dropout/per-sample dropout module applied before attention
        if intermediate_dropout:
            self.interdrop = PerSampleDropout()
        else:
            self.interdrop = None

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=mlp_drop_rate,
            final_dropout=final_mlp_dropout,
        )
        modulation_input_dim = (
            hidden_size if condition_embed_dim is None else condition_embed_dim
        )
        self.adaptive_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(modulation_input_dim, 6 * hidden_size, bias=True)
        )
        self.modulation = lambda x, scale, shift: x * (
            1 + scale.unsqueeze(1)
        ) + shift.unsqueeze(1)

        self.drop_path = DropPath(drop_path)

    def initialize_weights(self):
        r"""Zero-initialize the adaptive modulation linear layer (adaLN-Zero).

        Parameters
        ----------
        None
            Uses ``self`` (module state).

        Returns
        -------
        None
            Modifies parameters in-place.
        """
        # Zero out the adaptive modulation weights
        nn.init.constant_(self.adaptive_modulation[-1].weight, 0)
        nn.init.constant_(self.adaptive_modulation[-1].bias, 0)

    def forward(
        self,
        x: Float[torch.Tensor, "batch sequence hidden_size"],
        c: Float[torch.Tensor, "batch condition_embed_dim"],
        attn_kwargs: Optional[Dict[str, Any]] = None,
        p_dropout: Optional[float | Float[torch.Tensor, " batch"]] = None,
    ) -> Float[torch.Tensor, "batch sequence hidden_size"]:
        (
            attention_shift,
            attention_scale,
            attention_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.adaptive_modulation(c).chunk(6, dim=1)

        # Attention block
        modulated_attn_input = self.modulation(
            self.pre_attention_norm(x), attention_scale, attention_shift
        )

        if self.interdrop is not None:
            # Apply intermediate dropout (supports scalar or per-sample p) if enabled
            modulated_attn_input = self.interdrop(modulated_attn_input, p_dropout)
        elif p_dropout is not None:
            raise ValueError(
                "p_dropout passed to DiTBlock but intermediate_dropout is disabled"
            )

        attention_output = self.attention(
            modulated_attn_input,
            **(attn_kwargs or {}),
        )
        x = torch.addcmul(
            x, self.drop_path(attention_gate.unsqueeze(1)), attention_output
        )

        # Feed-forward block
        modulated_mlp_input = self.modulation(
            self.pre_mlp_norm(x), mlp_scale, mlp_shift
        )
        mlp_output = self.linear(modulated_mlp_input)
        x = torch.addcmul(x, self.drop_path(mlp_gate.unsqueeze(1)), mlp_output)

        return x


class ProjLayer(nn.Module):
    r"""The penultimate layer of the DiT model, which projects the transformer output to a final embedding space.

    Parameters
    ----------
    hidden_size : int
        The dimensionality of the input from the transformer blocks.
    emb_channels : int
        The number of embedding channels for final projection.
    layernorm_backend : Literal["apex", "torch"], optional, default="torch"
        The layer normalization implementation.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, L, D)`.
    c : torch.Tensor
        Conditioning tensor of shape :math:`(B, D)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, L, \text{emb\_channels})`.
    """

    def __init__(
        self,
        hidden_size: int,
        emb_channels: int,
        layernorm_backend: Literal["apex", "torch"] = "torch",
    ):
        super().__init__()
        if layernorm_backend == "apex" and not APEX_AVAILABLE:
            raise ImportError(
                "Apex is not available. Please install Apex to use ProjLayer with FusedLayerNorm.\
                Or use 'torch' as layernorm_backend."
            )
        self.proj_layer_norm = get_layer_norm(
            hidden_size, layernorm_backend, elementwise_affine=False, eps=1e-6
        )
        self.output_projection = nn.Linear(hidden_size, emb_channels, bias=True)
        self.adaptive_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.modulation = lambda x, scale, shift: x * (
            1 + scale.unsqueeze(1)
        ) + shift.unsqueeze(1)

    def forward(
        self,
        x: Float[torch.Tensor, "batch sequence hidden_size"],
        c: Float[torch.Tensor, "batch hidden_size"],
    ) -> Float[torch.Tensor, "batch sequence emb_channels"]:
        shift, scale = self.adaptive_modulation(c).chunk(2, dim=1)
        modulated_output = self.modulation(self.proj_layer_norm(x), scale, shift)
        projected_output = self.output_projection(modulated_output)
        return projected_output


# -------------------------------------------------------------------------------------
# Tokenization / De-tokenization interfaces
# -------------------------------------------------------------------------------------


class TokenizerModuleBase(Module, ABC):
    r"""Abstract base class for tokenizers used by DiT.

    Must implement a forward method and an initialize_weights method.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, *\text{spatial\_dims})`. ``spatial_dims`` is determined by ``input_size``.

    Outputs
    -------
    torch.Tensor
        Token sequence of shape :math:`(B, L, D)`, where :math:`L` is the sequence length (number of patches for 2D DiT).
    """

    @abstractmethod
    def forward(
        self, x: Float[torch.Tensor, "batch channels *spatial_dims"]
    ) -> Float[torch.Tensor, "batch sequence token_dim"]:
        pass

    @abstractmethod
    def initialize_weights(self):
        """Initialize the weights of the tokenizer."""
        pass


class PatchEmbed2DTokenizer(TokenizerModuleBase):
    r"""Standard ViT-style tokenizer using PatchEmbed2D followed by a learnable positional embedding.

    Produces tokens of shape :math:`(B, L, D)` from images of shape :math:`(B, C, H, W)`, where :math:`L` is the number
    of patches and :math:`D` is ``hidden_size``.

    Parameters
    ----------
    input_size : Tuple[int, int]
        The size of the input image.
    patch_size : Tuple[int, int]
        The size of the patch.
    in_channels : int
        The number of input channels.
    hidden_size : int
        The size of the transformer latent space to project to.
    pos_embed : str, optional, default="learnable"
        The type of positional embedding. ``"learnable"`` uses a learnable embedding; otherwise no positional embedding.
    **tokenizer_kwargs : Any
        Additional keyword arguments for the tokenizer module.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, H, W)`.

    Outputs
    -------
    torch.Tensor
        Token sequence of shape :math:`(B, L, D)`, where :math:`L = (H / \text{patch}[0]) \times (W / \text{patch}[1])`.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        in_channels: int,
        hidden_size: int,
        pos_embed: str = "learnable",
        **tokenizer_kwargs: Any,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        self.h_patches = self.input_size[0] // self.patch_size[0]
        self.w_patches = self.input_size[1] // self.patch_size[1]
        self.num_patches = self.h_patches * self.w_patches

        self.x_embedder = PatchEmbed2D(
            self.input_size,
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            **tokenizer_kwargs,
        )
        # Learnable positional embedding per token
        if pos_embed == "learnable":
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, self.hidden_size), requires_grad=True
            )
        else:
            self.pos_embed = 0.0

    def initialize_weights(self):
        """Initialize the weights of the tokenizer."""
        # Initialize the tokenizer patch embedding projection (a Conv2D layer).
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize the learnable positional embedding with a normal distribution.
        if isinstance(self.pos_embed, nn.Parameter):
            nn.init.normal_(self.pos_embed, std=0.02)

    def forward(
        self, x: Float[torch.Tensor, "batch channels height width"]
    ) -> Float[torch.Tensor, "batch sequence hidden_size"]:
        # (B, D, Hp, Wp)
        x_emb = self.x_embedder(x)
        # (B, L, D) + positional embedding
        tokens = x_emb.flatten(2).transpose(1, 2) + self.pos_embed
        return tokens


def get_tokenizer(
    input_size: Tuple[int],
    patch_size: Tuple[int],
    in_channels: int,
    hidden_size: int,
    tokenizer: Literal["patch_embed_2d", "hpx_patch_embed"] = "patch_embed_2d",
    **tokenizer_kwargs: Any,
) -> Union[TokenizerModuleBase, nn.Module]:
    r"""Construct a tokenizer module.

    Returns a module whose forward accepts :math:`(B, C, *\text{spatial\_dims})` and returns :math:`(B, L, D)`.
    ``spatial_dims`` is determined by ``input_size``.

    Parameters
    ----------
    input_size : Tuple[int]
        The size of the input image (or tuple for multi-dimensional domain).
        Ignored by ``"hpx_patch_embed"``.
    patch_size : Tuple[int]
        The size of the patch (or tuple for multi-dimensional patch).
        Ignored by ``"hpx_patch_embed"``.
    in_channels : int
        The number of input channels.
    hidden_size : int
        The size of the transformer latent space to project to.
    tokenizer : Literal["patch_embed_2d", "hpx_patch_embed"], optional, default="patch_embed_2d"
        The tokenizer to use.
        - ``"patch_embed_2d"`` -- Uses a standard PatchEmbed2D to project the input image to a
          sequence of tokens based on ``input_size`` and ``patch_size``.
        - ``"hpx_patch_embed"`` -- Uses the HEALPix patch tokenizer from ``physicsnemo.nn.module.hpx.tokenizer``.
          Requires ``earth2grid`` and determines the patch size from ``level_fine`` and ``level_coarse`` in
          ``tokenizer_kwargs``.

    **tokenizer_kwargs : Any
        Additional keyword arguments for the tokenizer module.

    Returns
    -------
    TokenizerModuleBase or nn.Module
        The tokenizer module.
    """
    if tokenizer == "patch_embed_2d":
        return PatchEmbed2DTokenizer(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            **tokenizer_kwargs,
        )
    if tokenizer == "hpx_patch_embed":
        return HEALPixPatchTokenizer(
            in_channels=in_channels,
            hidden_size=hidden_size,
            **tokenizer_kwargs,
        )
    raise ValueError("tokenizer must be 'patch_embed_2d' or 'hpx_patch_embed'.")


class DetokenizerModuleBase(Module, ABC):
    r"""Abstract base class for detokenizers used by DiT.

    Must implement a forward method and an initialize_weights method.

    Forward
    -------
    x_tokens : torch.Tensor
        Token sequence of shape :math:`(B, L, D_{in})`.
    c : torch.Tensor
        Conditioning tensor of shape :math:`(B, D)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, *\text{spatial\_dims})`. ``spatial_dims`` is determined by ``input_size``.
    """

    @abstractmethod
    def forward(
        self,
        x_tokens: Float[torch.Tensor, "batch sequence token_dim"],
        c: Float[torch.Tensor, "batch condition_dim"],
    ) -> Float[torch.Tensor, "batch out_channels *spatial_dims"]:
        pass

    @abstractmethod
    def initialize_weights(self):
        """Initialize the weights of the detokenizer."""
        pass


class ProjReshape2DDetokenizer(DetokenizerModuleBase):
    r"""Standard DiT-style detokenizer that applies the DiT ProjLayer and reshapes the sequence to an image.

    Output image shape is :math:`(B, C_{out}, H, W)`.

    Parameters
    ----------
    input_size : Tuple[int, int]
        The size of the input image.
    patch_size : Tuple[int, int]
        The size of the patch.
    out_channels : int
        The number of output channels.
    hidden_size : int
        The size of the transformer latent space to project from.
    layernorm_backend : Literal["apex", "torch"], optional, default="torch"
        The layer normalization implementation.

    Forward
    -------
    x_tokens : torch.Tensor
        Token sequence of shape :math:`(B, L, D_{in})`.
    c : torch.Tensor
        Conditioning tensor of shape :math:`(B, D)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, *\text{spatial\_dims})`. ``spatial_dims`` is determined by ``input_size``.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        out_channels: int,
        hidden_size: int,
        layernorm_backend: Literal["apex", "torch"] = "torch",
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.h_patches = self.input_size[0] // self.patch_size[0]
        self.w_patches = self.input_size[1] // self.patch_size[1]

        self.proj_layer = ProjLayer(
            hidden_size=self.hidden_size,
            emb_channels=self.patch_size[0] * self.patch_size[1] * self.out_channels,
            layernorm_backend=layernorm_backend,
        )

    def initialize_weights(self):
        """Initialize the weights of the detokenizer."""
        # Zero out the adaptive modulation and output projection weights
        nn.init.constant_(self.proj_layer.adaptive_modulation[-1].weight, 0)
        nn.init.constant_(self.proj_layer.adaptive_modulation[-1].bias, 0)
        nn.init.constant_(self.proj_layer.output_projection.weight, 0)
        nn.init.constant_(self.proj_layer.output_projection.bias, 0)

    def forward(
        self,
        x_tokens: Float[torch.Tensor, "batch sequence hidden_size"],
        c: Float[torch.Tensor, "batch hidden_size"],
    ) -> Float[torch.Tensor, "batch out_channels height width"]:
        # Project tokens to per-patch pixel embeddings
        x = self.proj_layer(x_tokens, c)  # (B, L, p0*p1*C_out)

        # Reshape back to image
        x = x.reshape(
            shape=(
                x.shape[0],
                self.h_patches,
                self.w_patches,
                self.patch_size[0],
                self.patch_size[1],
                self.out_channels,
            )
        )
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(
            shape=(
                x.shape[0],
                self.out_channels,
                self.h_patches * self.patch_size[0],
                self.w_patches * self.patch_size[1],
            )
        )
        return x


class ConvDetokenizer(DetokenizerModuleBase):
    r"""Detokenizer with a residual convolutional smoothing head to suppress patch-seam artifacts.

    Wraps :class:`ProjReshape2DDetokenizer` and adds a small
    residual convolutional head over the full :math:`(B, C_{out}, H, W)` output
    image.  The base detokenizer maps each token independently to its
    ``patch × patch`` output block, so it cannot couple information across patch
    boundaries; this can produce checkerboard artifacts on output channels with
    high spatial frequency content.  The conv head lets information flow across
    patch seams and can smooth those artifacts away.

    The final conv layer is zero-initialized, so at construction the residual
    contribution is exactly zero and this module is numerically identical to
    :class:`ProjReshape2DDetokenizer`.

    Use this as a drop-in replacement for
    :class:`ProjReshape2DDetokenizer` (or ``detokenizer="proj_reshape_2d_conv"``
    in :func:`get_detokenizer`) when checkerboard artifacts are visible in the
    predicted image.  Because the head starts at zero, it can be swapped in
    for a run already trained with ``proj_reshape_2d`` without disturbing
    training dynamics.  The head is fully convolutional, so it can support
    variable-resolution inference.

    Parameters
    ----------
    input_size : Tuple[int, int]
        The size of the input image.
    patch_size : Tuple[int, int]
        The size of the patch.
    out_channels : int
        The number of output channels.
    hidden_size : int
        The size of the transformer latent space to project from.
    layernorm_backend : Literal["apex", "torch"], optional, default="torch"
        The layer normalization implementation.
    conv_layers : int, optional, default=2
        Number of conv layers in the smoothing head.
    conv_hidden : int, optional, default=64
        Number of feature maps in intermediate conv layers.
    conv_kernel : int, optional, default=3
        Spatial kernel size (same-padding is applied).

    Forward
    -------
    x_tokens : torch.Tensor
        Token sequence of shape :math:`(B, L, D_{in})`.
    c : torch.Tensor
        Conditioning tensor of shape :math:`(B, D)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, H, W)`.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        out_channels: int,
        hidden_size: int,
        layernorm_backend: Literal["apex", "torch"] = "torch",
        conv_layers: int = 2,
        conv_hidden: int = 64,
        conv_kernel: int = 3,
    ):
        super().__init__()
        self.proj = ProjReshape2DDetokenizer(
            input_size=input_size,
            patch_size=patch_size,
            out_channels=out_channels,
            hidden_size=hidden_size,
            layernorm_backend=layernorm_backend,
        )
        if conv_layers < 1:
            raise ValueError(f"conv_layers must be >= 1; got {conv_layers}")
        pad = conv_kernel // 2
        layers: list[nn.Module] = []
        c_in = out_channels
        n = int(conv_layers)
        for i in range(n):
            last = i == n - 1
            c_out = out_channels if last else conv_hidden
            layers.append(nn.Conv2d(c_in, c_out, conv_kernel, padding=pad))
            if not last:
                layers.append(nn.GELU())
            c_in = c_out
        self.conv_head = nn.Sequential(*layers)

    def initialize_weights(self) -> None:
        r"""Initialize weights; zero the last conv so the residual starts at zero.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Modifies module parameters in-place.
        """
        self.proj.initialize_weights()
        convs = [m for m in self.conv_head if isinstance(m, nn.Conv2d)]
        nn.init.zeros_(convs[-1].weight)
        if convs[-1].bias is not None:
            nn.init.zeros_(convs[-1].bias)

    def forward(
        self,
        x_tokens: Float[torch.Tensor, "batch sequence hidden_size"],
        c: Float[torch.Tensor, "batch hidden_size"],
    ) -> Float[torch.Tensor, "batch out_channels height width"]:
        img = self.proj(x_tokens, c)
        return img + self.conv_head(img)


def get_detokenizer(
    input_size: Union[int, Tuple[int]],
    patch_size: Union[int, Tuple[int]],
    out_channels: int,
    hidden_size: int,
    detokenizer: Literal[
        "proj_reshape_2d", "proj_reshape_2d_conv", "hpx_patch_detokenizer"
    ] = "proj_reshape_2d",
    **detokenizer_kwargs: Any,
) -> Union[DetokenizerModuleBase, nn.Module]:
    r"""Construct a detokenizer module.

    Returns a module whose forward accepts :math:`(B, L, D)` and :math:`(B, D)` and returns
    :math:`(B, C_{out}, *\text{spatial\_dims})`. ``spatial_dims`` is determined by ``input_size``.

    Parameters
    ----------
    input_size : Union[int, Tuple[int]]
        The size of the input image (int for square 2D, tuple for multi-dimensional).
        Ignored by ``"hpx_patch_detokenizer"``.
    patch_size : Union[int, Tuple[int]]
        The size of the patch (int for square 2D, tuple for multi-dimensional).
        Ignored by ``"hpx_patch_detokenizer"``.
    out_channels : int
        The number of output channels.
    hidden_size : int
        The size of the transformer latent space to project from.
    detokenizer : Literal["proj_reshape_2d", "proj_reshape_2d_conv", "hpx_patch_detokenizer"], optional, default="proj_reshape_2d"
        The detokenizer to use.
        - ``"proj_reshape_2d"`` -- Uses a standard DiT ProjLayer and reshapes the sequence back to an
          image based on ``input_size`` and ``patch_size``.
        - ``"proj_reshape_2d_conv"`` -- Same as ``"proj_reshape_2d"`` followed by a zero-initialized
          residual conv smoothing head (:class:`ConvDetokenizer`). Reduces checkerboard artifacts on
          spiky output channels. Accepts ``conv_layers``, ``conv_hidden``, and ``conv_kernel`` in
          ``detokenizer_kwargs``.
        - ``"hpx_patch_detokenizer"`` -- HEALPix patch detokenizer from
          ``physicsnemo.nn.module.hpx.tokenizer``. Requires ``earth2grid`` and determines the patch size
          from ``level_coarse`` and ``level_fine`` in ``detokenizer_kwargs``.

    **detokenizer_kwargs : Any
        Additional keyword arguments forwarded to the detokenizer constructor.

    Returns
    -------
    DetokenizerModuleBase or nn.Module
        The detokenizer module.
    """
    if detokenizer == "proj_reshape_2d":
        return ProjReshape2DDetokenizer(
            input_size=input_size,
            patch_size=patch_size,
            out_channels=out_channels,
            hidden_size=hidden_size,
            **detokenizer_kwargs,
        )
    if detokenizer == "proj_reshape_2d_conv":
        return ConvDetokenizer(
            input_size=input_size,
            patch_size=patch_size,
            out_channels=out_channels,
            hidden_size=hidden_size,
            **detokenizer_kwargs,
        )
    if detokenizer == "hpx_patch_detokenizer":
        return HEALPixPatchDetokenizer(
            hidden_size=hidden_size,
            out_channels=out_channels,
            **detokenizer_kwargs,
        )
    raise ValueError(
        "detokenizer must be 'proj_reshape_2d', 'proj_reshape_2d_conv', or 'hpx_patch_detokenizer'."
    )
