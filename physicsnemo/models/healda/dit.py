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
# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/dit_transformer_2d.py
"""
Adapted from the DiT code in the huggingface diffusers library

NVIDIA Modifications
- simplify code and remove unused layer normalizations
- pass emb directly to the AdaLayerNormZero
- incorporate the noise, class label, position, and calendar embeddings from the other cBottle code
"""

import dataclasses
import math
from functools import partial
from typing import Any, Dict, Optional, Tuple

import earth2grid.healpix
import einops
import torch
import torch.distributed.fsdp
import torch.utils.checkpoint
from diffusers.models.attention import Attention, FeedForward
from torch import nn

from . import profiling
from .config import ModelSensorConfig, ObsConfig, SensorEmbedderConfig
from .domain import HealPixDomain
from .embedding import EmbedNoiseLabels
from .healpix_layers import HPXPatchDecode, HPXPatchEmbed, Subdomain
from .obs_embedding.decoder import ObsDecoder
from .obs_embedding.point_embed import (
    MultiSensorObsEmbedding,
)
from .sharding import shard_t, shard_x
from .types import UnifiedObservation


@dataclasses.dataclass
class Output:
    """DiT model forward pass output."""

    out: torch.Tensor
    obs: torch.Tensor | None = None


class DropPath(torch.nn.Module):
    """
    Stochastic Depth (DropPath)
    """

    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        # broadcast mask over non-batch dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p:.3f})"


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, emb_channels: int, bias=True):
        super().__init__()
        # TODO silu unused. Is this a bug? --noah 9/5/25
        self.silu = nn.SiLU()
        self.linear = nn.Linear(emb_channels, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        def unsqueeze(gate):
            if gate.ndim != 2:
                raise ValueError(f"Expected gate.ndim == 2, got {gate.ndim}")
            return gate[:, None, None, :]

        emb = self.linear(emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
            6, dim=1
        )
        x = self.norm(x) * (1 + unsqueeze(scale_msa)) + unsqueeze(shift_msa)

        return (
            x,
            unsqueeze(gate_msa),
            unsqueeze(shift_mlp),
            unsqueeze(scale_mlp),
            unsqueeze(gate_mlp),
        )


class AdaLayerNormTemporalAttn(nn.Module):
    r"""Ada Layernorm which is only use for the temporal attn
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Could be fused with AdaLayerNormZero for a slight computational gain

    """

    def __init__(self, embedding_dim: int, emb_channels: int, bias=True):
        super().__init__()
        # TODO silu unused. Is this a bug? --noah 9/5/25
        self.silu = nn.SiLU()
        self.linear = nn.Linear(emb_channels, 3 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(emb)
        shift, scale, gate = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale[:, None, None]) + shift[:, None, None]
        return x, gate[:, None, None]


class RotaryPositionEmbedding(torch.nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.

    This class provides rotary position embeddings that can be applied to query and key tensors
    in attention mechanisms to encode positional information.
    """

    def __init__(self, head_dim, base: int = 10000, max_seq_len: int = 24):
        """
        Initialize the Rotary Position Embedding.

        Args:
            head_dim (int): Dimension of each attention head
            base (int): Base for frequency calculation (default: 10000)
            max_seq_len (int): Maximum sequence length to precompute
        """
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len

        # Precompute frequencies for efficiency
        self._precompute_freqs()

    def _precompute_freqs(self):
        """Precompute frequency matrices for all possible sequence lengths."""
        # Create position indices up to max_seq_len
        position = torch.arange(self.max_seq_len).float()

        # Create frequency indices for pairs (head_dim//2 pairs)
        dim_indices = torch.arange(self.head_dim // 2).float()
        dim_indices = dim_indices[None, :]  # [1, head_dim//2]

        # Calculate frequencies
        freqs = 1.0 / (
            self.base ** (2 * dim_indices / self.head_dim)
        )  # [1, head_dim//2]
        freqs = position[:, None] * freqs  # [max_seq_len, head_dim//2]

        # Generate cos and sin
        self.register_buffer("freqs_cos", torch.cos(freqs))
        self.register_buffer("freqs_sin", torch.sin(freqs))

    @torch.compile
    def forward(self, x):
        """
        Apply rotary position embedding to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, t, x, heads, head_dim]

        Returns:
            torch.Tensor: Tensor with rotary position embedding applied
        """
        seq_len = x.shape[1]

        # Ensure we don't exceed precomputed length
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        # Get the relevant frequency matrices
        freqs_cos = self.freqs_cos[:seq_len]  # [seq_len, head_dim]
        freqs_sin = self.freqs_sin[:seq_len]  # [seq_len, head_dim]

        return self._apply_rotary_pos_emb(x, freqs_cos, freqs_sin)

    def _apply_rotary_pos_emb(self, x, freqs_cos, freqs_sin):
        """Apply rotary position embedding to input tensor x."""
        # x: [b, t, x, heads, head_dim
        # freqs_cos, freqs_sin: [t, head_dim//2]

        # Split x into even and odd indices along the head_dim
        # Each pair of dimensions gets rotated together
        x1, x2 = x[..., 0::2], x[..., 1::2]

        cos = freqs_cos[None, :, None, None, :]
        sin = freqs_sin[None, :, None, None, :]

        # Apply rotation - each pair shares the same angle
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # Interleave the results back
        out = torch.stack([out1, out2], dim=-1)
        out = out.reshape(x.shape)

        return out


# NVIDIA authored class
# TODO refactor to another module where owernship is more clear
def mask_causal_(attn):
    """causal mask of a [b tq tk x h] shaped attention mask

    attn = 0 if tq < tk
    """
    x, h = attn.shape[-2:]
    attn = einops.rearrange(attn, "b q k x h -> (b x h) q k")
    attn = attn.tril_()
    return einops.rearrange(attn, "(b x h) q k -> b q k x h", x=x, h=h)


class TemporalAttention(torch.nn.Module):
    """Multi-head attention over time dimension with optional RoPE."""

    def __init__(
        self,
        *,
        embed_dim,
        num_heads,
        use_rope=True,
        rope_base=100,
        max_seq_len=100,
    ) -> None:
        super().__init__()
        self.qkv = torch.nn.Linear(embed_dim, embed_dim * 3)
        self.proj = torch.nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.head_dim = embed_dim // num_heads

        # Initialize RoPE if enabled
        if self.use_rope:
            self.rope = RotaryPositionEmbedding(
                head_dim=self.head_dim, base=rope_base, max_seq_len=max_seq_len
            )
        else:
            self.rope = None

    @torch.compile
    def forward(self, x, is_causal: bool = False):
        qkv = self.qkv(x)
        q, k, v = einops.rearrange(
            qkv,
            "b t x (n heads c) -> n b t x heads c",
            n=3,
            heads=self.num_heads,
        )

        # Apply RoPE to queries and keys if enabled
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # q - q time dim
        # k - k time dim
        attn = torch.einsum(
            "b q x h c, b k x h c -> b q k x h", q, k / math.sqrt(k.shape[1])
        )

        if is_causal:
            attn = mask_causal_(attn)

        w = attn
        w = w.softmax(2)
        out = einops.einsum(attn, v, "b q k x h, b k x h c -> b q x h c")
        out = einops.rearrange(out, "b t x h c -> b t x (h c)")
        out = self.proj(out)
        return out


class SpatialAttention(Attention):
    """Multi-head attention over spatial dimension."""

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        **cross_attention_kwargs,
    ):
        b, t, x, c = hidden_states.shape
        hidden_states = hidden_states.reshape(b * t, x, c)
        out = super().forward(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        return out.reshape(b, t, x, c)


class SpatioTemporalAttention(Attention):
    """Multi-head attention over flattened space-time dimension."""

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        **cross_attention_kwargs,
    ):
        b, t, x, c = hidden_states.shape
        hidden_states = hidden_states.reshape(b, t * x, c)
        out = super().forward(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        return out.reshape(b, t, x, c)


class TransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        drop_path: float = 0.0,
        temporal_attention: bool = False,  # TODO change to default of False
        qk_rms_norm: bool = False,
        *,
        emb_channels: int,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.double_self_attention = double_self_attention
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.only_cross_attention = only_cross_attention

        # We keep these boolean flags for backward-compatibility.
        self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = AdaLayerNormZero(dim, emb_channels=emb_channels)

        self.temporal_attn = None
        if temporal_attention:
            attn_cls = SpatialAttention
            self.temporal_attn_norm = AdaLayerNormTemporalAttn(dim, emb_channels)
            self.temporal_attn = TemporalAttention(
                embed_dim=dim, num_heads=num_attention_heads
            )
        else:
            attn_cls = SpatioTemporalAttention

        self.attn1 = attn_cls(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            qk_norm="rms_norm" if qk_rms_norm else None,
            elementwise_affine=not qk_rms_norm,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=(
                    cross_attention_dim if not double_self_attention else None
                ),
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.attn2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        self.drop_path = DropPath(drop_path)

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            # self.fuser = GatedSelfAttentionDense(
            #     dim, cross_attention_dim, num_attention_heads, attention_head_dim
            # )
            raise NotImplementedError()

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        self._parallel_group = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        obs_data,
        obs_parent_id,
        input_t_sharded: bool,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        loop=None,
        is_causal: bool = False,  # if temporal attn is causal
        checkpoint_ff: bool = False,
    ) -> torch.Tensor:
        if not input_t_sharded and self._parallel_group is not None:
            hidden_states = shard_t(hidden_states, self._parallel_group)

        # 0. Self-Attention
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb
        )

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        with profiling.nvtx_range("attn1", enabled=False):
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=(
                    encoder_hidden_states if self.only_cross_attention else None
                ),
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

            # TODO there is a bug here when batch size > 1
            hidden_states = torch.addcmul(
                hidden_states, self.drop_path(gate_msa), attn_output
            )

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = self.drop_path(attn_output) + hidden_states

        # temporal attention
        if self.temporal_attn is not None:
            with profiling.nvtx_range("temporal_attn", enabled=False):
                if self._parallel_group is not None:
                    hidden_states = shard_x(hidden_states, self._parallel_group)
                    input_t_sharded = False

                norm_hidden_states, gate = self.temporal_attn_norm(hidden_states, emb)
                attn_output = self.temporal_attn(norm_hidden_states, is_causal)
                hidden_states = torch.addcmul(
                    hidden_states, self.drop_path(gate), attn_output
                )

        # 4. Feed-forward
        # i2vgen doesn't have this norm ?????
        with profiling.nvtx_range("norm3", enabled=False):
            norm_hidden_states = self.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp

        with profiling.nvtx_range("ff", enabled=False):
            if checkpoint_ff:
                ff_output = torch.utils.checkpoint.checkpoint(
                    self.ff, norm_hidden_states, use_reentrant=False
                )
            else:
                ff_output = self.ff(norm_hidden_states)
            hidden_states = torch.addcmul(
                hidden_states, self.drop_path(gate_mlp), ff_output
            )
        return hidden_states, input_t_sharded

    def set_parallel_group(self, group):
        self._parallel_group = group


class DiT(torch.nn.Module):
    r"""
    A 2D Transformer model as introduced in DiT (https://huggingface.co/papers/2212.09748).

    Parameters:
        num_attention_heads (int, optional, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (int, optional, defaults to 72): The number of channels in each head.
        in_channels (int, defaults to 4): The number of channels in the input.
        out_channels (int, optional):
            The number of channels in the output. Specify this parameter if the output channel number differs from the
            input.
        num_layers (int, optional, defaults to 28): The number of layers of Transformer blocks to use.
        dropout (float, optional, defaults to 0.0): The dropout probability to use within the Transformer blocks.
        norm_num_groups (int, optional, defaults to 32):
            Number of groups for group normalization within Transformer blocks.
        attention_bias (bool, optional, defaults to True):
            Configure if the Transformer blocks' attention should contain a bias parameter.
        sample_size (int, defaults to 32):
            The width of the latent images. This parameter is fixed during training.
        patch_size (int, defaults to 2):
            Size of the patches the model processes, relevant for architectures working on non-sequential data.
        activation_fn (str, optional, defaults to "gelu-approximate"):
            Activation function to use in feed-forward networks within Transformer blocks.
        upcast_attention (bool, optional, defaults to False):
            If true, upcasts the attention mechanism dimensions for potentially improved performance.
        norm_type (str, optional, defaults to "ada_norm_zero"):
            Specifies the type of normalization used, can be 'ada_norm_zero'.
        norm_elementwise_affine (bool, optional, defaults to False):
            If true, enables element-wise affine parameters in the normalization layers.
        norm_eps (float, optional, defaults to 1e-5):
            A small constant added to the denominator in normalization layers to prevent division by zero.
    """

    pixel_order = earth2grid.healpix.HEALPIX_PAD_XY

    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _supports_gradient_checkpointing = True
    _supports_group_offloading = False

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        num_layers: int = 28,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_size: int = 32,
        activation_fn: str = "gelu-approximate",
        qk_rms_norm: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        # hpx grid info
        level_in: int = 6,
        level_model: int = 4,
        time_length: int = 1,
        label_dim: int = 0,
        label_dropout: float = 0.0,
        legacy_label_bias: bool = False,
        obs_config: Optional[ObsConfig] = None,
        drop_path: float = 0.0,
        group_norm_eps: float = 1e-6,
        use_gains: bool = False,
        temporal_attention: bool = False,
        obs_decoder: bool = False,
        dense_decoder: bool = True,
        # FLAGS
        embed_v2: bool = False,
        embed_v2_meta_dim: int = 28,
        embed_v2_n_embed=1024,
        embed_v2_in_level=7,
        sensor_embedder_config: Optional[
            SensorEmbedderConfig
        ] = None,  # Per-sensor observation embedder config
        sensors: Optional[dict[str, "ModelSensorConfig"]] = None,  # Sensor configs
        compile_dit: bool = False,  # Enable torch.compile for _forward_DiT
        allow_nans_condition: bool = False,
        emb_channels: int | None = None,
        noise_channels: int | None = None,
        gradient_checkpointing: int = 0,
        as_vit: bool = False,
    ):
        """
        Args:
            gradient_checkpointing:
                0: no gradient checkpointing
                1: gradient checkpointing for ff
                >1: gradient checkpoint for all blocks
            as_vit: If True, skip noise/label conditioning entirely.
                Sets emb_channels=0 internally, making all AdaLN linear layers bias-only
        """
        super().__init__()
        self._level_in = level_in
        self.temporal_attention = temporal_attention
        self.compile_dit = compile_dit
        self.as_vit = as_vit
        if level_in < level_model:
            raise ValueError(
                f"level_in must be >= level_model, got {level_in} < {level_model}"
            )
        patch_size = 2 ** (level_in - level_model)
        self.level_model = level_model

        self.time_length = time_length

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = gradient_checkpointing

        self.embed_v2_patch = None
        if embed_v2 and sensor_embedder_config is not None:
            if sensors is None:
                raise ValueError(
                    "sensors is required when sensor_embedder_config is provided"
                )

            self.embed_v2_patch = MultiSensorObsEmbedding(
                sensor_embedder_config=sensor_embedder_config,
                sensors=sensors,
                hpx_level=level_in,
            )
            pos_embed_in_channels = in_channels + sensor_embedder_config.fusion_dim
        else:
            self.patch_size = patch_size
            pos_embed_in_channels = in_channels

        self.pos_embed = HPXPatchEmbed(
            in_channels=pos_embed_in_channels,
            out_channels=self.inner_dim,
            level_fine=level_in,
            level_coarse=level_model,
            use_gains=use_gains,
            allow_nans=allow_nans_condition,
        )

        if as_vit:
            emb_channels = 0
            self.noise_embed = None
        else:
            if emb_channels is None:
                emb_channels = 4 * self.inner_dim
            self.noise_embed = EmbedNoiseLabels(
                emb_channels,
                label_dim,
                noise_channels=noise_channels
                if noise_channels is not None
                else self.inner_dim,
                label_dropout=label_dropout,
                legacy_label_bias=legacy_label_bias,
            )

        # 2. Initialize the position embedding and transformer blocks.
        self.height = sample_size
        self.width = sample_size

        drop_path_schedule = [
            drop_path * i / max(1, num_layers - 1) for i in range(num_layers)
        ]
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    emb_channels=emb_channels,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    qk_rms_norm=qk_rms_norm,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    drop_path=drop_path_schedule[i],
                    temporal_attention=temporal_attention,
                )
                for i in range(num_layers)
            ]
        )

        # 3. Output blocks.
        self.proj_out_1 = nn.Linear(emb_channels, 2 * self.inner_dim)
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)

        if dense_decoder:
            self.patch_decode = HPXPatchDecode(
                in_channels=self.inner_dim,
                out_channels=out_channels,
                level_fine=level_in,
                level_coarse=level_model,
            )
        else:
            self.patch_decode = None

        if obs_decoder:
            self.decode_obs = ObsDecoder(
                self.inner_dim,
                metadata_dim=embed_v2_meta_dim,
                hpx_fine_level=8,
                hpx_in_level=level_model,
                max_embed_id=1024,
                obs_dim=1,
            )
        else:
            self.decode_obs = None

        if self.inner_dim % 4 != 0:
            raise ValueError(self.inner_dim)

        self._parallel_group = None

    @property
    def grid(self):
        return earth2grid.healpix.Grid(
            level=self._level_in, pixel_order=self.pixel_order
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def domain(self):
        return HealPixDomain(self.grid)

    def load_from_checkpoint(self, checkpoint):
        """Load from checkpoint adjusting settings for fine tuning"""
        state_dict = checkpoint.read_model_state_dict()
        config = checkpoint.read_model_config()

        # old checkpoints before I added fsdp training
        # were saved with label_dim = 0
        if self.noise_embed is not None and self.noise_embed.map_label is None:
            state_dict.pop("noise_embed.map_label.weight", None)
            state_dict.pop("noise_embed.map_label.bias", None)

        if self.temporal_attention and (not config.dit_temporal_attention):
            my_state_dict = self.state_dict()
            import re

            for key in my_state_dict:
                if not re.match(r"transformer_blocks\..*\.temporal_attn", key):
                    continue

                if key in state_dict:
                    continue

                state_dict[key] = my_state_dict[key]

        self.load_state_dict(state_dict)

    @profiling.nvtx
    def forward(
        self,
        hidden_states: torch.Tensor,
        noise_labels: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        day_of_year: torch.Tensor | None = None,
        second_of_day: torch.Tensor | None = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        unified_obs: UnifiedObservation | None = None,
        timestamp=None,
        is_causal: bool = False,
        subdomain: Subdomain | None = None,
        level_localize: int | None = None,
    ):
        """
        The [`DiTTransformer2DModel`] forward method.

        Args:
            hidden_states: network input. shaped [b, c, t, x]
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            timestamp: [b t] shaped tensor. contains integer timestamp for each frame.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        if self.embed_v2_patch is not None:
            if unified_obs is None:
                raise ValueError("Need to provide observation inputs.")

            obs_emb = self.embed_v2_patch(unified_obs)
            hidden_states = torch.cat([hidden_states, obs_emb], dim=1)

        # 1. Input
        obs_input = obs_parent_id = None
        hidden_states = self.pos_embed(
            hidden_states,
            second_of_day=second_of_day,
            day_of_year=day_of_year,
            subdomain=subdomain,
        )

        if level_localize is not None:
            hidden_states = hidden_states.movedim(-2, -1)
            hidden_states = earth2grid.healpix.reorder(
                hidden_states, self.pixel_order, earth2grid.healpix.NEST
            )
            hidden_states = einops.rearrange(
                hidden_states,
                "b t c (n x) -> (b n) t x c",
                n=12 * 4 ** (self.level_model - level_localize),
            )

        # 2. Blocks (compiled)
        def _forward_blocks(
            hidden_states, obs_input, obs_parent_id, blocks, gradient_checkpointing
        ):
            # In vit mode, emb is ignored - Adapative scaling layers just return bias
            if self.as_vit:
                b = hidden_states.shape[0]
                emb = torch.empty(
                    b, 0, device=hidden_states.device, dtype=hidden_states.dtype
                )
            else:
                if noise_labels is None:
                    raise ValueError("noise_labels is required when as_vit=False")
                emb = self.noise_embed(noise_labels, class_labels)

            input_t_sharded = True

            for block in blocks:
                args = (hidden_states, emb, obs_input, obs_parent_id)
                if gradient_checkpointing > 1:
                    hidden_states, input_t_sharded = torch.utils.checkpoint.checkpoint(
                        partial(
                            block,
                            input_t_sharded=input_t_sharded,
                        ),
                        *args,
                        use_reentrant=False,
                        is_causal=is_causal,
                    )

                else:
                    hidden_states, input_t_sharded = block(
                        *args,
                        input_t_sharded=input_t_sharded,
                        is_causal=is_causal,
                        checkpoint_ff=gradient_checkpointing == 1,
                    )

            # the output should be sharded in time
            if self._parallel_group is not None and not input_t_sharded:
                hidden_states = shard_t(hidden_states, self._parallel_group)

            # 3. Output
            shift, scale = self.proj_out_1(emb).chunk(2, dim=1)
            hidden_states = (
                self.norm_out(hidden_states) * (1 + scale[:, None, None])
                + shift[:, None, None]
            )

            return hidden_states

        if self.compile_dit:
            _forward_blocks = torch.compile(_forward_blocks)

        hidden_states = _forward_blocks(
            hidden_states,
            obs_input,
            obs_parent_id,
            self.transformer_blocks,
            self.gradient_checkpointing,
        )

        if level_localize:
            hidden_states = einops.rearrange(
                hidden_states,
                "(b n) t x c -> b t c (n x)",
                n=12 * 4 ** (self.level_model - level_localize),
            )
            hidden_states = earth2grid.healpix.reorder(
                hidden_states, earth2grid.healpix.NEST, self.pixel_order
            )
            hidden_states = hidden_states.movedim(-1, -2)

        # Coarsen subdomain to model level before passing to decoder
        subdomain_coarse = None
        if subdomain is not None:
            subdomain_coarse = subdomain.coarsen(self._level_in - self.level_model)

        # 4. Decode (not compiled - includes HEALPix operations)
        return Output(
            out=self._decode_dense(hidden_states, subdomain_coarse),
            obs=self._decode_obs(hidden_states, unified_obs, subdomain_coarse),
        )

    @profiling.nvtx
    def _decode_dense(
        self,
        hidden_states: torch.Tensor,
        subdomain: Subdomain | None,
    ) -> torch.Tensor:
        if self.patch_decode is None:
            return torch.empty([])

        return self.patch_decode(hidden_states, subdomain)

    def _decode_obs(
        self,
        hidden_states: torch.Tensor,
        unified_obs: UnifiedObservation | None,
        subdomain: Subdomain | None = None,
    ) -> None | torch.Tensor:
        if self.decode_obs is None:
            return None

        if unified_obs is None:
            raise ValueError("Need to provide unified_obs object for decoder.")

        platform = unified_obs.int_metadata[:, unified_obs.bucket_index.platform]
        pix = unified_obs.int_metadata[:, unified_obs.bucket_index.pix]
        obs_type = unified_obs.int_metadata[:, unified_obs.bucket_index.obs_type]
        channel = unified_obs.int_metadata[:, unified_obs.bucket_index.global_channel]

        return self.decode_obs(
            latent=hidden_states,
            batch_idx=unified_obs.batch_idx,
            metadata=unified_obs.float_metadata,
            pix=pix,
            platform=platform,
            obs_type=obs_type,
            channel=channel,
            hpx_level=unified_obs.hpx_level,
            subdomain=subdomain,
        )

    def set_parallel_group(self, group):
        """Set the parallel group

        Assumes that the x-dimension of the input is sharded across
        ``group``. If temporal attention is active, then an all_to_all will
        collect the frames on a single rank before hand.

        """
        self._parallel_group = group
        for block in self.transformer_blocks:
            block.set_parallel_group(group)

    def fully_shard(self, mesh: torch.distributed.device_mesh.DeviceMesh | None = None):
        """

        Args:
            mesh: This data parallel mesh defines the sharding and device. If
                1D, then parameters are fully sharded across the 1D mesh (FSDP) with (Shard(0),)
                placement. If 2D, then parameters are sharded across the 1st dim and replicated
                across the 0th dim (HSDP) with (Replicate(), Shard(0)) placement. The mesh’s
                device type gives the device type used for communication; if a CUDA or CUDA-like
                device type, then we use the current device.

        """
        for block in self.transformer_blocks:
            torch.distributed.fsdp.fully_shard(block, mesh=mesh)

        torch.distributed.fsdp.fully_shard(self, mesh=mesh)
