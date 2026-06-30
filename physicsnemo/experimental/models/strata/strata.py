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

r"""Strata: a two-stage 3D transformer regression model with pixel-wise conditioning.

Stage 1 (backbone) is a :class:`~physicsnemo.experimental.models.strata.StrataTransformer3D`
operating on coarse patches; its output tokens ``backbone_cond`` condition stage 2
(pixel), which runs at full :math:`1\times1\times1`-patch resolution and injects
the conditioning through pixel-wise adaptive layer norm, adapting the
pixel-wise-AdaLN idea from PixelDiT (`arXiv:2511.20645
<https://arxiv.org/abs/2511.20645>`_). This is an adaptation, not a faithful
port: a deterministic regression model (no diffusion / timestep / label
conditioning), an independent reimplementation of the AdaLN, with an original
conditioning path. See :class:`Strata` for the full attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

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
    StrataPixel3DBlock,
    StrataTransformer3DBlock,
)
from .transformer import StrataTransformer3D

__all__ = ["Strata"]


@dataclass
class StrataMetaData(ModelMetaData):
    r"""Metadata for :class:`Strata` (see :class:`~physicsnemo.core.meta.ModelMetaData`)."""

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


class Strata(Module):
    r"""Two-stage 3D transformer regression model with pixel-wise conditioning.

    Composes a :class:`~physicsnemo.experimental.models.strata.StrataTransformer3D` backbone
    stage (built from ``backbone_config``) with a pixel-resolution stage. The
    backbone stage's tokens condition every pixel block via pixel-wise adaptive
    layer norm (see :class:`StrataPixel3DBlock`). The backbone stage's unused output
    head is dropped at construction (only its
    :meth:`~physicsnemo.experimental.models.strata.StrataTransformer3D.forward_tokens` trunk is
    used), so its forward is unavailable once wrapped.

    Like :class:`StrataTransformer3D`, this is an **independent reimplementation** of the
    Diffusion-Transformer (DiT) architecture (not a reuse of
    :class:`~physicsnemo.models.dit.DiT`'s components), used as a **deterministic
    regression** model — **not** a generative diffusion model. The "DiT" name is
    especially apt here because the pixel stage keeps DiT's *defining* feature,
    **adaptive-layer-norm (adaLN) conditioning** (see :class:`StrataPixel3DBlock`); only
    the conditioning *signal* differs — the backbone stage's learned features,
    rather than a diffusion timestep / noise level. There is no diffusion process
    and no class-label or text conditioning (see Notes).

    Parameters
    ----------
    backbone_config : Dict[str, Any], optional, default=None
        Keyword arguments forwarded to :class:`StrataTransformer3D` to build the backbone
        stage. Its ``in_channels``, ``input_shape``, ``patch_size``,
        ``out_channels``, and ``embed_dim`` determine the pixel-pathway layout.
        ``None`` is coerced to an empty dict, i.e. a default-configured
        :class:`StrataTransformer3D` backbone stage.
    embed_dim_pixel : int, optional, default=128
        Pixel-pathway embedding dimension.
    num_layers_pixel : int, optional, default=4
        Number of pixel-pathway blocks.
    num_heads_pixel : int, optional, default=None
        Number of pixel-pathway attention heads. If ``None``, set to
        ``max(1, embed_dim_pixel // 64)``.
    mlp_ratio_pixel : float, optional, default=4.0
        Ratio of pixel MLP hidden dimension to ``embed_dim_pixel``.
    attn_kernel_pixel : int | Tuple[int, int, int], optional, default=3
        Pixel-pathway neighborhood-attention window; ``-1`` selects full attention.
    gated_attention_pixel : bool, optional, default=False
        Whether pixel attention uses a learned sigmoid gate.
    qk_norm_pixel : bool, optional, default=False
        Whether to RMS-normalize pixel attention queries and keys.
    qk_norm_affine_pixel : bool, optional, default=False
        Whether the pixel QK RMS norms use a learnable affine scale.
    na3d_backend_pixel : str, optional, default=None
        NATTEN backend for the pixel pathway.
    adaln_mode : Literal["pixel_proj", "bilinear_dw"], optional, default="pixel_proj"
        Pixel-wise modulation derivation mode (see :class:`StrataPixel3DBlock`).
        ``"bilinear_dw"`` trilinearly upsamples the backbone conditioning over
        depth and the horizontal plane, so any vertical patch size is supported.
    first_block_only_adaln : bool, optional, default=False
        If ``True``, only the first pixel block injects conditioning (adaptive
        layer norm); the rest are plain :class:`StrataTransformer3DBlock` blocks.
    use_chunked_depthwise_conv : bool, optional, default=True
        Whether ``"bilinear_dw"`` blocks use the chunked :class:`DepthwiseConv`.
    chunk_size_grouped_conv : int, optional, default=2
        ``torch.vmap`` chunk size for the ``"bilinear_dw"`` depthwise conv.
    rope_mode_pixel : Literal["none", "axial", "stereographic"], optional, default="none"
        Pixel-pathway RoPE mode. ``"stereographic"`` builds pixel-resolution
        coordinates via :func:`~physicsnemo.experimental.models.strata.coords.build_stereographic_token_coords`
        and requires ``pos`` at ``forward``. Note: unlike
        :meth:`~physicsnemo.experimental.models.strata.StrataTransformer3D.set_tile_size`, the
        pixel stage has no re-tiling hook, so an ``"axial"`` pixel RoPE is fixed to
        the construction-time grid (``stereographic`` is recomputed per forward and
        is unaffected).
    rope_base_pixel : float, optional, default=100.0
        Base of the pixel RoPE frequency progression.
    rope_length_scale_pixel : float, optional, default=1.0
        Stereographic coordinate normalization for the pixel pathway.
    bf16_mixed_pixel : bool, optional, default=False
        If ``True``, run the pixel blocks under ``bfloat16`` autocast on CUDA.
    activation_checkpointing_pixel : bool | float, optional, default=False
        Activation checkpointing of the pixel blocks during training. ``True`` /
        ``1.0`` checkpoints all pixel blocks, ``0.0`` / ``False`` none, and a value
        in ``(0, 1)`` checkpoints that leading fraction. Only engages in train mode.

    Forward
    -------
    x : torch.Tensor
        Input field of shape :math:`(B, C, D, H, W)`.
    pos : torch.Tensor, optional
        Latitude / longitude in radians of shape :math:`(B, 2, H, W)`. Required
        when either stage uses stereographic RoPE.

    Outputs
    -------
    torch.Tensor
        Output field of shape :math:`(B, C_{out}, D, H, W)`.

    Notes
    -----
    The pixel-wise adaptive-layer-norm conditioning is adapted from PixelDiT
    (see References). This is an **adaptation, not a faithful reimplementation**:

    - It is a deterministic regression model (e.g. weather emulation), not a
      diffusion model: there is no noise / timestep conditioning and no
      class-label or text conditioning.
    - The pixel-wise AdaLN is an independent reimplementation written from the
      paper (predating the public PixelDiT release), not ported code.
    - The ``"bilinear_dw"`` conditioning path (bilinear upsample + depthwise
      convolution) is an original addition beyond the paper.

    References
    ----------
    - `PixelDiT: Pixel Diffusion Transformers for Image Generation <https://arxiv.org/abs/2511.20645>`_

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.strata import Strata
    >>> model = Strata(
    ...     backbone_config=dict(
    ...         in_channels=4,
    ...         input_shape=(4, 8, 8),
    ...         patch_size=(1, 2, 2),
    ...         embed_dim=32,
    ...         num_heads=4,
    ...         num_layers=2,
    ...         attn_kernel=-1,
    ...     ),
    ...     embed_dim_pixel=16,
    ...     num_layers_pixel=2,
    ...     num_heads_pixel=2,
    ...     attn_kernel_pixel=-1,
    ... )
    >>> x = torch.randn(2, 4, 4, 8, 8)
    >>> model(x).shape
    torch.Size([2, 4, 4, 8, 8])
    """

    __model_checkpoint_version__ = "1.0.0"

    def __init__(
        self,
        backbone_config: Optional[Dict[str, Any]] = None,
        embed_dim_pixel: int = 128,
        num_layers_pixel: int = 4,
        num_heads_pixel: Optional[int] = None,
        mlp_ratio_pixel: float = 4.0,
        attn_kernel_pixel: Union[int, Tuple[int, int, int]] = 3,
        gated_attention_pixel: bool = False,
        qk_norm_pixel: bool = False,
        qk_norm_affine_pixel: bool = False,
        na3d_backend_pixel: Optional[str] = None,
        adaln_mode: Literal["pixel_proj", "bilinear_dw"] = "pixel_proj",
        first_block_only_adaln: bool = False,
        use_chunked_depthwise_conv: bool = True,
        chunk_size_grouped_conv: int = 2,
        rope_mode_pixel: Literal["none", "axial", "stereographic"] = "none",
        rope_base_pixel: float = 100.0,
        rope_length_scale_pixel: float = 1.0,
        bf16_mixed_pixel: bool = False,
        activation_checkpointing_pixel: Union[bool, float] = False,
    ):
        super().__init__(meta=StrataMetaData())

        ### Stage 1: backbone StrataTransformer3D, built from the provided config.
        backbone_config = dict(backbone_config or {})
        # Build the backbone stage as a headless feature trunk: only its
        # forward_tokens output is used here, so force include_head=False to avoid
        # creating unused output-head parameters (no DDP unused-parameter issues).
        backbone_config["include_head"] = False
        self.backbone = StrataTransformer3D(**backbone_config)

        depth, height, width = (
            self.backbone.depth,
            self.backbone.height,
            self.backbone.width,
        )
        in_channels = self.backbone.in_channels
        out_channels = self.backbone.out_channels
        cond_dim = self.backbone.embed_dim
        pd, ph, pw = self.backbone.patch_size
        pixels_per_patch = pd * ph * pw

        num_heads_pixel = (
            num_heads_pixel
            if num_heads_pixel is not None
            else max(1, embed_dim_pixel // 64)
        )

        ### Input validation
        if embed_dim_pixel % num_heads_pixel != 0:
            raise ValueError(
                f"embed_dim_pixel ({embed_dim_pixel}) must be divisible by "
                f"num_heads_pixel ({num_heads_pixel})"
            )
        head_dim_pixel = embed_dim_pixel // num_heads_pixel
        if rope_mode_pixel not in ("none", "axial", "stereographic"):
            raise ValueError(
                f"rope_mode_pixel must be 'none', 'axial', or 'stereographic'; "
                f"got {rope_mode_pixel!r}"
            )
        if rope_mode_pixel != "none" and head_dim_pixel % 4 != 0:
            raise ValueError(
                f"Pixel RoPE requires head_dim_pixel "
                f"(embed_dim_pixel // num_heads_pixel = {head_dim_pixel}) divisible by 4"
            )
        if first_block_only_adaln and num_layers_pixel < 1:
            raise ValueError(
                "first_block_only_adaln requires num_layers_pixel >= 1; "
                f"got num_layers_pixel={num_layers_pixel}"
            )

        # Public attributes.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.height = height
        self.width = width
        self.embed_dim_pixel = embed_dim_pixel
        self.num_layers_pixel = num_layers_pixel
        self.num_heads_pixel = num_heads_pixel
        self.adaln_mode = adaln_mode
        self.first_block_only_adaln = first_block_only_adaln
        self.rope_mode_pixel = rope_mode_pixel
        self.rope_length_scale_pixel = rope_length_scale_pixel
        self.rope_base_pixel = rope_base_pixel
        self.head_dim_pixel = head_dim_pixel
        self.bf16_mixed_pixel = bf16_mixed_pixel
        # Reuse StrataTransformer3D's pure parser (a staticmethod) so the ratio semantics match
        # the backbone stage.
        self._activation_checkpointing_ratio_pixel = StrataTransformer3D._parse_checkpointing_param(
            activation_checkpointing_pixel
        )

        ### Stage 2: pixel pathway at 1x1x1-patch resolution.
        self.pixel_patch_embed = PatchEmbed3D(
            depth=depth,
            height=height,
            width=width,
            patch_size=(1, 1, 1),
            in_chans=in_channels,
            embed_dim=embed_dim_pixel,
        )

        def _make_adaln_block() -> StrataPixel3DBlock:
            return StrataPixel3DBlock(
                dim=embed_dim_pixel,
                cond_dim=cond_dim,
                pixels_per_patch=pixels_per_patch,
                num_heads=num_heads_pixel,
                mlp_ratio=mlp_ratio_pixel,
                qk_norm=qk_norm_pixel,
                qk_norm_affine=qk_norm_affine_pixel,
                attn_kernel=attn_kernel_pixel,
                gated_attention=gated_attention_pixel,
                na3d_backend=na3d_backend_pixel,
                adaln_mode=adaln_mode,
                use_chunked_depthwise_conv=use_chunked_depthwise_conv,
                chunk_size_grouped_conv=chunk_size_grouped_conv,
            )

        def _make_plain_block() -> StrataTransformer3DBlock:
            return StrataTransformer3DBlock(
                dim=embed_dim_pixel,
                num_heads=num_heads_pixel,
                mlp_ratio=mlp_ratio_pixel,
                qk_norm=qk_norm_pixel,
                qk_norm_affine=qk_norm_affine_pixel,
                attn_kernel=attn_kernel_pixel,
                gated_attention=gated_attention_pixel,
                na3d_backend=na3d_backend_pixel,
            )

        if first_block_only_adaln:
            blocks = [_make_adaln_block()] + [
                _make_plain_block() for _ in range(num_layers_pixel - 1)
            ]
        else:
            blocks = [_make_adaln_block() for _ in range(num_layers_pixel)]
        self.pixel_blocks = nn.ModuleList(blocks)

        self.pixel_final_layer = FinalLayer3D(embed_dim_pixel, out_channels)

        # Pixel-pathway RoPE. Axial coords are static; stereographic coords are
        # computed per forward via the shared geometry helpers in ``coords.py``
        # (no dependency on the backbone stage's RoPE module).
        if rope_mode_pixel == "axial":
            # Pixel patches are 1x1x1, so the token grid is (depth, height, width).
            coords = build_axial_token_coords(depth, height, width)
            cos, sin = build_axial_rope_cos_sin_2d_continuous(
                coords[:, 0],
                coords[:, 1],
                self.head_dim_pixel,
                theta=self.rope_base_pixel,
            )
            self.register_buffer("_rope_cos_pixel", cos, persistent=False)
            self.register_buffer("_rope_sin_pixel", sin, persistent=False)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        r"""Initialize the pixel-stage parameters DiT-style.

        The backbone stage self-initializes. For the pixel stage this mirrors
        :meth:`StrataTransformer3D.initialize_weights`: Xavier-uniform on the
        transformer blocks' linears, a fan-based Xavier on the pixel
        patch-embedding conv, and a zero-initialized output head. The AdaLN
        projections (zeroed at block construction) are restored after the blanket
        Xavier pass so each conditioning block still starts as an identity
        residual.

        Returns
        -------
        None
            Modifies parameters in place.
        """

        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Xavier the pixel blocks' attention / MLP linears, then re-zero the AdaLN
        # projections that the blanket pass clobbered (preserving AdaLN-zero).
        self.pixel_blocks.apply(_basic_init)
        for block in self.pixel_blocks:
            if isinstance(block, StrataPixel3DBlock):
                block.reset_adaln_zero()

        w = self.pixel_patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.pixel_patch_embed.proj.bias, 0.0)

        nn.init.constant_(self.pixel_final_layer.linear.weight, 0.0)
        nn.init.constant_(self.pixel_final_layer.linear.bias, 0.0)

    def _should_checkpoint_pixel_block(self, block_idx: int) -> bool:
        r"""Return whether the pixel block at ``block_idx`` should be checkpointed.

        Activation checkpointing only engages in training mode. Mirrors
        :meth:`~physicsnemo.experimental.models.strata.StrataTransformer3D._should_checkpoint_block`.

        Parameters
        ----------
        block_idx : int
            Zero-based pixel-block index.

        Returns
        -------
        bool
            ``True`` if this pixel block should use activation checkpointing.
        """
        if not self.training:
            return False
        ratio = self._activation_checkpointing_ratio_pixel
        if ratio <= 0.0:
            return False
        if ratio >= 1.0:
            return True
        return block_idx < round(ratio * len(self.pixel_blocks))

    def _build_pixel_rope_tables(
        self, pos: Optional[torch.Tensor]
    ) -> Optional[RopeTables]:
        r"""Build the pixel-pathway ``(cos, sin)`` RoPE tables for this forward pass.

        Parameters
        ----------
        pos : torch.Tensor, optional
            Latitude / longitude of shape :math:`(B, 2, H, W)`; required for
            stereographic mode.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor], optional
            The ``(cos, sin)`` tables, or ``None`` when ``rope_mode_pixel="none"``.
        """
        if self.rope_mode_pixel == "none":
            return None
        if self.rope_mode_pixel == "axial":
            return (self._rope_cos_pixel, self._rope_sin_pixel)
        # Stereographic: pixel resolution -> horizontal patch (1, 1), tiled over depth.
        coords = build_stereographic_token_coords(
            pos, (1, 1), d_patch=self.depth, length_scale=self.rope_length_scale_pixel
        )  # (B, N, 2)
        cos, sin = build_axial_rope_cos_sin_2d_continuous(
            coords[..., 0],
            coords[..., 1],
            self.head_dim_pixel,
            theta=self.rope_base_pixel,
        )
        # Insert a heads axis so the tables broadcast over (B, heads, N, head_dim).
        return cos.unsqueeze(1), sin.unsqueeze(1)

    def forward(
        self,
        x: Float[torch.Tensor, "batch in_channels depth height width"],
        pos: Optional[Float[torch.Tensor, "batch 2 height width"]] = None,
    ) -> Float[torch.Tensor, "batch out_channels depth height width"]:
        ### Input validation
        if not torch.compiler.is_compiling():
            if x.ndim != 5:
                raise ValueError(
                    f"Expected 5D input (B, C, D, H, W), got tensor of shape "
                    f"{tuple(x.shape)}"
                )
            if self.rope_mode_pixel == "stereographic" and pos is None:
                raise ValueError(
                    "pos (lat/lon of shape (B, 2, H, W)) is required when "
                    "rope_mode_pixel='stereographic'"
                )
            # Validate pos shape here too: the backbone only checks it when its
            # own rope_mode='stereographic', so a backbone='none'/'axial' +
            # pixel='stereographic' config would otherwise hit a cryptic rearrange
            # error in build_stereographic_token_coords.
            if pos is not None and pos.shape != (x.shape[0], 2, x.shape[3], x.shape[4]):
                raise ValueError(
                    f"Expected pos of shape "
                    f"{(x.shape[0], 2, x.shape[3], x.shape[4])}, got {tuple(pos.shape)}"
                )

        _, _, dd, hh, ww = x.shape

        # Stage 1: backbone conditioning tokens (input validation handled here).
        backbone_cond, backbone_dhw = self.backbone.forward_tokens(x, pos)

        # Stage 2: pixel tokens at full resolution.
        pixel_dhw = (dd, hh, ww)
        x_pix = rearrange(self.pixel_patch_embed(x), "b c d h w -> b (d h w) c")

        # Precompute the shared bilinear conditioning once if any block needs it.
        use_shared_bilinear = any(
            isinstance(b, StrataPixel3DBlock) and b.adaln_mode == "bilinear_dw"
            for b in self.pixel_blocks
        )
        s_cond_bilinear = (
            StrataPixel3DBlock.precompute_bilinear_cond(backbone_cond, pixel_dhw, backbone_dhw)
            if use_shared_bilinear
            else None
        )

        rope_tables = self._build_pixel_rope_tables(pos)

        autocast_enabled = self.bf16_mixed_pixel and x.is_cuda
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            for i, block in enumerate(self.pixel_blocks):
                # The two block types take different keyword signatures; wrap the
                # call in a closure so activation checkpointing can drive either.
                # The grad-bearing conditioning tensors are passed as explicit
                # arguments (rather than closed over) so the checkpoint recompute's
                # gradient path is auditable; the static (d, h, w) tuples and the
                # loop-invariant rope_tables stay captured.
                if isinstance(block, StrataPixel3DBlock):

                    def _run(inp, cond, cond_bilinear, b=block):
                        return b(
                            inp,
                            backbone_cond=cond,
                            pixel_dhw=pixel_dhw,
                            backbone_dhw=backbone_dhw,
                            s_cond_bilinear=cond_bilinear,
                            rope_tables=rope_tables,
                        )

                    run_args = (x_pix, backbone_cond, s_cond_bilinear)
                else:

                    def _run(inp, b=block):
                        return b(inp, latent_dhw=pixel_dhw, rope_tables=rope_tables)

                    run_args = (x_pix,)

                if self._should_checkpoint_pixel_block(i):
                    x_pix = activation_checkpoint(_run, *run_args, use_reentrant=False)
                else:
                    x_pix = _run(*run_args)
            x_pix = self.pixel_final_layer(x_pix)

        return rearrange(x_pix, "b (d h w) c -> b c d h w", d=dd, h=hh, w=ww)
