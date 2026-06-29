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

r"""StrataTransformer3D: a 3D transformer backbone for regression on spherical fields.

StrataTransformer3D reuses the Diffusion-Transformer (DiT) *architecture* but is a
**deterministic regression** model, not a generative diffusion model: it has no
diffusion / denoising process and none of the original DiT's diffusion
conditioning (no noise, timestep, class-label, or text conditioning). The "DiT"
in the name refers to the architecture lineage only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.nn import build_axial_rope_cos_sin_2d_continuous

from .coords import build_axial_token_coords, build_stereographic_token_coords
from .layers import (
    FinalLayer3D,
    PatchEmbed3D,
    RopeTables,
    StrataTransformer3DBlock,
)

__all__ = ["StrataTransformer3D"]


@dataclass
class StrataTransformer3DMetaData(ModelMetaData):
    r"""Metadata for :class:`StrataTransformer3D` (see :class:`~physicsnemo.core.meta.ModelMetaData`)."""

    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class StrataTransformer3D(Module):
    r"""3D transformer (DiT architecture) for deterministic regression on spherical fields.

    StrataTransformer3D tokenizes a :math:`(B, C, D, H, W)` field with a 3D patch embedding,
    processes the tokens with a stack of pre-norm transformer blocks, and
    decodes them back to a :math:`(B, C_{out}, D, H, W)` field.

    It is a **conceptual 3D analog** of :class:`~physicsnemo.models.dit.DiT` — it
    follows the same DiT-style template (patch-embed -> pre-norm transformer ->
    linear decode) but is an **independent reimplementation**, not a wrapper or a
    reuse of that class's components. It defines its own 3D blocks
    (:class:`StrataTransformer3DBlock`, :class:`Natten3DSelfAttention`, :class:`PatchEmbed3D`,
    :class:`FinalLayer3D`) and shares only low-level primitives with the rest of
    the library: :func:`physicsnemo.nn.functional.na3d`, :class:`physicsnemo.nn.Mlp`,
    and the RoPE helpers. (The 2D :class:`~physicsnemo.models.dit.DiT` is built from
    a tokenizer/detokenizer registry and an adaLN conditioning embedder, none of
    which fit a 3D, unconditioned, neighborhood-attention model.)

    The name refers to the *architecture family*, not a training objective:
    despite "DiT" (Diffusion Transformer), this is a **deterministic regression**
    model, **not** generative — it carries none of the original DiT's diffusion
    conditioning (no noise, timestep, adaLN, class-label, or text conditioning),
    and is a plain pre-norm transformer mapping an input field to an output field.

    Geometry is decoupled from construction: latitude / longitude are supplied
    at ``forward`` time (only when ``rope_mode="stereographic"``) rather than
    built from a grid, so the model has no hard dependency on a specific
    spherical grid library.

    Parameters
    ----------
    in_channels : int
        Number of input field channels :math:`C`.
    out_channels : int, optional, default=None
        Number of output channels :math:`C_{out}`. If ``None``, set to ``in_channels``.
    input_shape : Tuple[int, int, int], optional, default=(16, 64, 64)
        The ``(D, H, W)`` shape of the input field (depth / height / width).
    patch_size : int | Tuple[int, int, int], optional, default=1
        Patch size, isotropic or per-axis ``(p_d, p_h, p_w)``. Each axis of
        ``input_shape`` must be divisible by the corresponding patch size.
    embed_dim : int, optional, default=768
        Token embedding dimension. Must be divisible by ``num_heads``.
    num_heads : int, optional, default=8
        Number of attention heads.
    num_layers : int, optional, default=12
        Number of transformer blocks.
    mlp_ratio : float, optional, default=4.0
        Ratio of MLP hidden dimension to ``embed_dim``.
    qkv_bias : bool, optional, default=True
        Whether attention QKV projections use a bias.
    qk_norm : bool, optional, default=False
        Whether to RMS-normalize attention queries and keys.
    qk_norm_affine : bool, optional, default=False
        Whether the QK RMS norms use a learnable affine scale.
    attn_kernel : int | Tuple[int, int, int], optional, default=3
        3D neighborhood-attention window size, isotropic or per-axis. Use ``-1``
        for full (dense) attention, which runs on CPU without NATTEN.
    na_dilation : int, optional, default=1
        Dilation factor for neighborhood attention.
    do_interleaved_dilation : bool, optional, default=False
        If ``True``, apply ``na_dilation`` on every fourth block (``i % 4 == 2``)
        and dilation 1 elsewhere.
    do_alt_depthwise_attn : bool, optional, default=False
        If ``True``, every odd-indexed block attends only along the depth axis.
    gated_attention : bool, optional, default=False
        Whether attention outputs are multiplied by a learned sigmoid gate.
    na3d_backend : str, optional, default=None
        NATTEN backend for neighborhood attention (e.g. ``"cutlass-fna"``).
    rope_mode : Literal["none", "axial", "stereographic"], optional, default="none"
        Rotary position embedding mode. ``"axial"`` uses integer row/column
        token indices; ``"stereographic"`` uses stereographic projection of the
        ``forward`` latitude / longitude (see
        :func:`~physicsnemo.experimental.nn.build_axial_rope_cos_sin_2d_continuous`).
    rope_base : float, optional, default=100.0
        Base of the RoPE frequency progression.
    rope_length_scale : float, optional, default=1.0
        Divisor applied to stereographic coordinates to normalize the per-token
        length scale. Ignored unless ``rope_mode="stereographic"``.
    activation_checkpointing : bool | float, optional, default=False
        Activation checkpointing of transformer blocks during training. ``True``
        / ``1.0`` checkpoints all blocks, ``0.0`` / ``False`` none, and a value
        in ``(0, 1)`` checkpoints that leading fraction of blocks.
    bf16_mixed : bool, optional, default=False
        If ``True``, run the transformer blocks under ``bfloat16`` autocast on
        CUDA. The output head always runs in fp32.
    include_head : bool, optional, default=True
        If ``True``, build the output head (``final_layer``) and decode to a field.
        If ``False``, omit the head so :meth:`forward` returns post-block tokens;
        used when the backbone is a feature trunk (e.g. inside ``Strata``).

    Forward
    -------
    x : torch.Tensor
        Input field of shape :math:`(B, C, D, H, W)`.
    pos : torch.Tensor, optional
        Latitude / longitude in radians of shape :math:`(B, 2, H, W)` (channel 0
        latitude, channel 1 longitude). Required when ``rope_mode="stereographic"``,
        ignored otherwise.

    Outputs
    -------
    torch.Tensor
        With ``include_head=True`` (default), the decoded field of shape
        :math:`(B, C_{out}, D, H, W)`. With ``include_head=False`` there is no
        output head, so the post-block tokens of shape :math:`(B, N, E)` are
        returned instead (equivalently :meth:`forward_tokens`).

    References
    ----------
    - `Scalable Diffusion Models with Transformers <https://arxiv.org/abs/2212.09748>`_
      (origin of the DiT architecture; StrataTransformer3D reuses the architecture only, not
      the diffusion training or conditioning)
    - `Neighborhood Attention Transformer <https://arxiv.org/abs/2204.07143>`_

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.strata import StrataTransformer3D
    >>> model = StrataTransformer3D(
    ...     in_channels=4,
    ...     input_shape=(4, 8, 8),
    ...     patch_size=(1, 2, 2),
    ...     embed_dim=32,
    ...     num_heads=4,
    ...     num_layers=2,
    ...     attn_kernel=-1,
    ... )
    >>> x = torch.randn(2, 4, 4, 8, 8)
    >>> out = model(x)
    >>> out.shape
    torch.Size([2, 4, 4, 8, 8])
    """

    __model_checkpoint_version__ = "1.0.0"

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        input_shape: Tuple[int, int, int] = (16, 64, 64),
        patch_size: Union[int, Tuple[int, int, int]] = 1,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        qk_norm_affine: bool = False,
        attn_kernel: Union[int, Tuple[int, int, int]] = 3,
        na_dilation: int = 1,
        do_interleaved_dilation: bool = False,
        do_alt_depthwise_attn: bool = False,
        gated_attention: bool = False,
        na3d_backend: Optional[str] = None,
        rope_mode: Literal["none", "axial", "stereographic"] = "none",
        rope_base: float = 100.0,
        rope_length_scale: float = 1.0,
        activation_checkpointing: Union[bool, float] = False,
        bf16_mixed: bool = False,
        include_head: bool = True,
    ):
        super().__init__(meta=StrataTransformer3DMetaData())

        ### Input validation
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if rope_mode not in ("none", "axial", "stereographic"):
            raise ValueError(
                f"rope_mode must be 'none', 'axial', or 'stereographic'; got {rope_mode!r}"
            )
        head_dim = embed_dim // num_heads
        if rope_mode != "none" and head_dim % 4 != 0:
            raise ValueError(
                f"RoPE requires head_dim (embed_dim // num_heads = {head_dim}) "
                f"divisible by 4"
            )
        depth, height, width = input_shape
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        patch_size = tuple(patch_size)
        pd, ph, pw = patch_size

        # Public attributes (also read by Strata when composing StrataTransformer3D).
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.input_shape = (depth, height, width)
        self.depth = depth
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.patch_size_vert = pd
        self.patch_size_horiz = ph
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.attn_kernel = attn_kernel
        self.rope_mode = rope_mode
        self.rope_base = rope_base
        self.rope_length_scale = rope_length_scale
        self.bf16_mixed = bf16_mixed
        self._activation_checkpointing_ratio = self._parse_checkpointing_param(
            activation_checkpointing
        )

        # Tokenizer: 3D patch embedding.
        self.patch_embed = PatchEmbed3D(
            depth=depth,
            height=height,
            width=width,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
        )

        # Transformer blocks. Odd blocks optionally attend along depth only; every
        # fourth block optionally uses dilated neighborhood attention.
        self.blocks = nn.ModuleList(
            [
                StrataTransformer3DBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    qk_norm_affine=qk_norm_affine,
                    attn_kernel=attn_kernel,
                    do_depthwise_attention=(i % 2 == 1) and do_alt_depthwise_attn,
                    na_dilation=(
                        na_dilation if (do_interleaved_dilation and i % 4 == 2) else 1
                    ),
                    gated_attention=gated_attention,
                    na3d_backend=na3d_backend,
                )
                for i in range(num_layers)
            ]
        )

        # Rotary position embedding. Both modes feed continuous (x, y) token
        # coordinates to ``build_axial_rope_cos_sin_2d_continuous``; the mode only
        # selects the coordinates. "axial" uses a static integer grid (cos/sin
        # precomputed once and cached as buffers below); "stereographic" projects
        # ``pos`` (lat/lon) per forward (see ``coords.py``).
        self.head_dim = head_dim
        if rope_mode == "axial":
            d, h, w = depth // pd, height // ph, width // pw
            coords = build_axial_token_coords(d, h, w)
            cos, sin = build_axial_rope_cos_sin_2d_continuous(
                coords[:, 0], coords[:, 1], self.head_dim, theta=self.rope_base
            )
            self.register_buffer("_rope_cos", cos, persistent=False)
            self.register_buffer("_rope_sin", sin, persistent=False)

        # The output head is optional: Strata builds the backbone headless and
        # consumes its tokens via ``forward_tokens``, so it skips these parameters.
        self.final_layer = (
            FinalLayer3D(embed_dim, pd * ph * pw * self.out_channels)
            if include_head
            else None
        )
        self.initialize_weights()

    @staticmethod
    def _parse_checkpointing_param(
        activation_checkpointing: Union[bool, float],
    ) -> float:
        r"""Parse the activation-checkpointing argument into a ratio in ``[0, 1]``.

        Parameters
        ----------
        activation_checkpointing : bool | float
            ``True`` / ``1.0`` for all blocks, ``False`` / ``0.0`` for none, or a
            fraction in ``(0, 1)``.

        Returns
        -------
        float
            The fraction of (leading) blocks to checkpoint.
        """
        # bool is a subclass of int, so it must be handled before the numeric path.
        if isinstance(activation_checkpointing, bool):
            return 1.0 if activation_checkpointing else 0.0
        if not isinstance(activation_checkpointing, (int, float)):
            raise TypeError(
                "activation_checkpointing must be bool or numeric, got "
                f"{type(activation_checkpointing).__name__}"
            )
        ratio = float(activation_checkpointing)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(
                f"activation_checkpointing must be bool or a float in [0, 1], got {ratio}"
            )
        return ratio

    def _should_checkpoint_block(self, block_idx: int) -> bool:
        r"""Return whether the block at ``block_idx`` should be checkpointed.

        Parameters
        ----------
        block_idx : int
            Zero-based block index.

        Returns
        -------
        bool
            ``True`` if this block should use activation checkpointing.
        """
        if not self.training:
            return False
        ratio = self._activation_checkpointing_ratio
        if ratio <= 0.0:
            return False
        if ratio >= 1.0:
            return True
        return block_idx < round(ratio * len(self.blocks))

    def initialize_weights(self) -> None:
        r"""Initialize parameters with DiT-style initialization.

        Applies Xavier-uniform initialization to all linear layers, a fan-based
        Xavier init to the patch-embedding convolution, and zero-initializes the
        output head so the model starts as a near-identity residual mapping.

        Returns
        -------
        None
            Modifies parameters in place.
        """

        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        self.apply(_basic_init)

        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0.0)

        if self.final_layer is not None:
            nn.init.constant_(self.final_layer.linear.weight, 0.0)
            nn.init.constant_(self.final_layer.linear.bias, 0.0)

    def set_tile_size(self, height: int, width: int) -> None:
        r"""Reconfigure the expected spatial tile size for a new resolution.

        Use this before running inference on a tile larger or smaller than the
        configured one. It updates the expected input shape (``height`` /
        ``width`` / ``input_shape``) for **every** ``rope_mode``, so the forward
        shape check and :meth:`_unpatchify` accept the new tile. For
        ``rope_mode="axial"`` it *additionally* rebuilds the cached ``(cos, sin)``
        RoPE buffers; the ``stereographic`` and ``none`` modes build their tables
        per forward (from the supplied ``pos``, or not at all), so updating the
        expected shape is all that is required for them.

        Parameters
        ----------
        height : int
            New input height.
        width : int
            New input width.

        Returns
        -------
        None
            Updates internal state in place.

        Raises
        ------
        ValueError
            If ``height`` or ``width`` is not divisible by the corresponding
            horizontal patch size (mirrors the constructor's check, so re-tiling
            cannot silently truncate the input via the patch-embed conv).
        """
        ph, pw = self.patch_size[1], self.patch_size[2]
        if height % ph != 0:
            raise ValueError(
                f"Height ({height}) must be divisible by horizontal patch size ({ph})"
            )
        if width % pw != 0:
            raise ValueError(
                f"Width ({width}) must be divisible by horizontal patch size ({pw})"
            )
        self.height = height
        self.width = width
        self.input_shape = (self.depth, height, width)
        if self.rope_mode != "axial":
            # stereographic / none build their RoPE per forward, so the expected
            # shape updated above is all that re-tiling needs.
            return
        d = self.depth // self.patch_size_vert
        h = height // self.patch_size[1]
        w = width // self.patch_size[2]
        coords = build_axial_token_coords(d, h, w)
        cos, sin = build_axial_rope_cos_sin_2d_continuous(
            coords[:, 0], coords[:, 1], self.head_dim, theta=self.rope_base
        )
        self.register_buffer(
            "_rope_cos", cos.to(self._rope_cos.device), persistent=False
        )
        self.register_buffer(
            "_rope_sin", sin.to(self._rope_sin.device), persistent=False
        )

    def _build_rope_tables(
        self,
        pos: Optional[torch.Tensor],
        latent_dhw: Tuple[int, int, int],
    ) -> Optional[RopeTables]:
        r"""Build the ``(cos, sin)`` RoPE tables for the current forward pass.

        Parameters
        ----------
        pos : torch.Tensor, optional
            Latitude / longitude of shape :math:`(B, 2, H, W)`; required for
            stereographic mode.
        latent_dhw : Tuple[int, int, int]
            The ``(d, h, w)`` token-grid shape.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor], optional
            The ``(cos, sin)`` tables, or ``None`` when ``rope_mode="none"``.
        """
        if self.rope_mode == "none":
            return None
        if self.rope_mode == "axial":
            return (self._rope_cos, self._rope_sin)
        # Stereographic: coordinates depend on the supplied geometry.
        d, _, _ = latent_dhw
        coords = build_stereographic_token_coords(
            pos,
            (self.patch_size[1], self.patch_size[2]),
            d_patch=d,
            length_scale=self.rope_length_scale,
        )  # (B, N, 2)
        cos, sin = build_axial_rope_cos_sin_2d_continuous(
            coords[..., 0], coords[..., 1], self.head_dim, theta=self.rope_base
        )  # (B, N, head_dim)
        # Insert a heads axis so the tables broadcast over (B, heads, N, head_dim).
        return cos.unsqueeze(1), sin.unsqueeze(1)

    def _prepare_tokens(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        r"""Patchify a field into a token sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input field of shape :math:`(B, C, D, H, W)`.

        Returns
        -------
        Tuple[torch.Tensor, Tuple[int, int, int]]
            The token tensor of shape :math:`(B, N, E)` and the ``(d, h, w)``
            token-grid shape.
        """
        x = self.patch_embed(x)  # (B, E, d, h, w)
        d, h, w = x.shape[-3:]
        x = rearrange(x, "b e d h w -> b (d h w) e")
        return x, (d, h, w)

    def forward_tokens(
        self,
        x: Float[torch.Tensor, "batch in_channels depth height width"],
        pos: Optional[Float[torch.Tensor, "batch 2 height width"]] = None,
    ) -> Tuple[Float[torch.Tensor, "batch tokens embed_dim"], Tuple[int, int, int]]:
        r"""Run the tokenizer and all transformer blocks, returning raw tokens.

        This is the shared trunk used both by :meth:`forward` (which then applies
        the output head) and by
        :class:`~physicsnemo.experimental.models.strata.Strata` (which uses the
        tokens as backbone conditioning).

        Parameters
        ----------
        x : torch.Tensor
            Input field of shape :math:`(B, C, D, H, W)`.
        pos : torch.Tensor, optional
            Latitude / longitude of shape :math:`(B, 2, H, W)`; required for
            stereographic RoPE.

        Returns
        -------
        Tuple[torch.Tensor, Tuple[int, int, int]]
            The post-block tokens of shape :math:`(B, N, E)` and the ``(d, h, w)``
            token-grid shape.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if x.ndim != 5:
                raise ValueError(
                    f"Expected 5D input (B, C, D, H, W), got tensor of shape "
                    f"{tuple(x.shape)}"
                )
            _, c, d_in, h_in, w_in = x.shape
            if c != self.in_channels:
                raise ValueError(f"Expected {self.in_channels} input channels, got {c}")
            if (d_in, h_in, w_in) != (self.depth, self.height, self.width):
                raise ValueError(
                    f"Expected spatial shape {(self.depth, self.height, self.width)}, "
                    f"got {(d_in, h_in, w_in)}"
                )
            if self.rope_mode == "stereographic":
                if pos is None:
                    raise ValueError(
                        "pos (lat/lon of shape (B, 2, H, W)) is required when "
                        "rope_mode='stereographic'"
                    )
                if pos.shape != (x.shape[0], 2, h_in, w_in):
                    raise ValueError(
                        f"Expected pos of shape {(x.shape[0], 2, h_in, w_in)}, got "
                        f"{tuple(pos.shape)}"
                    )

        x, latent_dhw = self._prepare_tokens(x)
        rope_tables = self._build_rope_tables(pos, latent_dhw)

        # Run blocks under optional bf16 autocast (CUDA only) and activation
        # checkpointing. RoPE is skipped for depth-axis blocks.
        autocast_enabled = self.bf16_mixed and x.is_cuda
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            for i, block in enumerate(self.blocks):
                block_rope = None if block.do_depthwise_attention else rope_tables
                if self._should_checkpoint_block(i):

                    def _run(inp, b=block, r=block_rope):
                        return b(inp, latent_dhw=latent_dhw, rope_tables=r)

                    x = activation_checkpoint(_run, x, use_reentrant=False)
                else:
                    x = block(x, latent_dhw=latent_dhw, rope_tables=block_rope)
        return x, latent_dhw

    def _unpatchify(
        self,
        x: torch.Tensor,
        depth: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        r"""Reassemble per-token patch pixels into a field.

        Parameters
        ----------
        x : torch.Tensor
            Per-token patch pixels of shape
            :math:`(B, N, p_d \cdot p_h \cdot p_w \cdot C_{out})`.
        depth : int, optional
            Output depth; defaults to the configured depth.
        height : int, optional
            Output height; defaults to the configured height.
        width : int, optional
            Output width; defaults to the configured width.

        Returns
        -------
        torch.Tensor
            Field of shape :math:`(B, C_{out}, D, H, W)`.
        """
        depth = depth or self.depth
        height = height or self.height
        width = width or self.width
        pd, ph, pw = self.patch_size
        return rearrange(
            x,
            "b (d h w) (P_d P_h P_w C) -> b C (d P_d) (h P_h) (w P_w)",
            P_d=pd,
            P_h=ph,
            P_w=pw,
            C=self.out_channels,
            d=depth // pd,
            h=height // ph,
            w=width // pw,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch in_channels depth height width"],
        pos: Optional[Float[torch.Tensor, "batch 2 height width"]] = None,
    ) -> torch.Tensor:
        r"""Run the model, returning a decoded field (or features when headless).

        With ``include_head=True`` (default) returns the decoded field of shape
        :math:`(B, C_{out}, D, H, W)`. With ``include_head=False`` there is no
        output head, so the post-block tokens of shape :math:`(B, N, E)` are
        returned instead (equivalently use :meth:`forward_tokens`).
        """
        tokens, _ = self.forward_tokens(x, pos)
        if self.final_layer is None:
            return tokens
        x = self.final_layer(tokens)
        # forward_tokens already validated the input shape == (depth, height,
        # width), so use the configured dims (avoids unpacking x before validation).
        return self._unpatchify(x, self.depth, self.height, self.width)
