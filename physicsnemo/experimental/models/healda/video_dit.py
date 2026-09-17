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
"""Diffusion Transformer over ``(B, C, T, X)`` inputs with an explicit time axis."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from jaxtyping import Float
from torch.distributed.device_mesh import DeviceMesh

from physicsnemo.core import Module
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.nn import (
    ConditioningEmbedder,
    ConditioningEmbedderType,
    RotaryEmbedding1DTables,
    get_conditioning_embedder,
)
from physicsnemo.nn.module.dit_layers import get_attention, get_layer_norm
from physicsnemo.nn.module.drop import DropPath
from physicsnemo.nn.module.mlp_layers import Mlp

from .adaln import AdaLNModulation, gated_residual, modulate
from .attention_layers import TemporalAttention
from .sharding import shard_t, shard_t_shardtensor, shard_x, shard_x_shardtensor


@dataclass
class MetaData(ModelMetaData):
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


class VideoDiTBlock(nn.Module):
    r"""A DiT block over :math:`(B, T, X, C)` with optional temporal and cross-attention.

    Spatial attention runs per frame (time folded into batch); the optional
    temporal and cross-attention sub-layers each add a gated residual branch.

    Parameters
    ----------
    hidden_size : int
        Token / channel dimension :math:`C`.
    num_heads : int
        Number of spatial- and temporal-attention heads.
    condition_embed_dim : int
        Dimension of the conditioning embedding feeding the adaLN modulations.
    attention_backend : Literal["timm", "transformer_engine", "natten2d", "natten2d_rope"] or Module, optional, default="timm"
        Spatial-attention backend name or a pre-instantiated attention module.
    layernorm_backend : Literal["apex", "torch"], optional, default="torch"
        LayerNorm backend for all adaLN-Zero pre-norms.
    mlp_ratio : float, optional, default=4.0
        MLP hidden-dim multiplier.
    norm_eps : float, optional, default=1e-6
        Epsilon for the affine-free layer norms.
    attn_drop_rate : float, optional, default=0.0
        Spatial-attention dropout rate.
    proj_drop_rate : float, optional, default=0.0
        Spatial-attention output-projection dropout rate.
    mlp_drop_rate : float, optional, default=0.0
        Dropout rate inside the MLP.
    final_mlp_dropout : bool, optional, default=True
        Whether to apply the final MLP dropout.
    drop_path : float, optional, default=0.0
        Stochastic-depth rate applied to every residual branch.
    temporal_attention : bool, optional, default=False
        Add a gated temporal-attention sub-layer.
    use_rope : bool, optional, default=True
        Apply rotary position embeddings in the temporal-attention sub-layer.
        Ignored unless ``temporal_attention=True``.
    temporal_kwargs : Dict[str, Any], optional, default=None
        Extra arguments for :class:`~physicsnemo.experimental.models.healda.attention_layers.TemporalAttention`
        (e.g. ``linear_attention``, ``causal_window``).
    cross_attention : Callable[..., Module], optional, default=None
        Factory building this block's cross-attention module
        (:class:`~physicsnemo.experimental.models.healda.attention_layers.CrossAttentionModuleBase`).
        When set, adds a gated cross-attention sub-layer consuming the opaque
        ``cross_attn_kwargs`` passed to :meth:`forward`.
    is_causal : bool, optional, default=False
        Causal masking for temporal attention, fixed at construction.
    adaln_zero_init : bool, optional, default=True
        Forwarded to every :class:`~physicsnemo.experimental.models.healda.adaln.AdaLNModulation` ``zero_init``.
    attn_kwargs : Dict[str, Any], optional, default=None
        Extra arguments for the spatial-attention backend constructor.

    Forward
    -------
    hidden_states : torch.Tensor
        Latents of shape :math:`(B, T, X, C)` (t-sharded under context
        parallelism).
    c : torch.Tensor
        Conditioning embedding of shape :math:`(B, D_c)`.
    attn_kwargs : Dict[str, Any], optional
        Forwarded to the spatial-attention backend forward.
    cross_attn_kwargs : Dict[str, Any], optional
        Forwarded to the injected ``cross_attention`` module.
    temporal_attn_kwargs : Dict[str, Any], optional
        Forwarded to :class:`~physicsnemo.experimental.models.healda.attention_layers.TemporalAttention`'s
        forward (e.g. ``rope_cos``/``rope_sin`` from a shared
        :class:`~physicsnemo.nn.module.rope.RotaryEmbedding1DTables` provider).

    Outputs
    -------
    torch.Tensor
        Updated latents of shape :math:`(B, T, X, C)` in the same layout.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        condition_embed_dim: int,
        attention_backend: Union[str, Module] = "timm",
        layernorm_backend: str = "torch",
        mlp_ratio: float = 4.0,
        norm_eps: float = 1e-6,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        mlp_drop_rate: float = 0.0,
        final_mlp_dropout: bool = True,
        drop_path: float = 0.0,
        temporal_attention: bool = False,
        use_rope: bool = True,
        temporal_kwargs: Optional[Dict[str, Any]] = None,
        cross_attention: Optional[Callable[..., Module]] = None,
        is_causal: bool = False,
        adaln_zero_init: bool = True,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.condition_embed_dim = condition_embed_dim

        # Spatial self-attention backend (name -> built here, or injected module),
        # named ``attention`` to match DiTBlock for checkpoint translatability.
        if isinstance(attention_backend, Module):
            self.attention = attention_backend
        else:
            attn_kwargs_final = dict(attn_kwargs or {})
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
        # n_blocks=2: one projection drives both spatial attention and the MLP
        # (the shared-projection DiT adaLN layout).
        self.norm1_modulation = AdaLNModulation(
            hidden_size,
            condition_embed_dim,
            n_blocks=2,
            zero_init=adaln_zero_init,
        )
        self.attn_norm = get_layer_norm(
            hidden_size,
            layernorm_backend,
            elementwise_affine=False,
            eps=norm_eps,
        )

        self.linear = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=mlp_drop_rate,
            final_dropout=final_mlp_dropout,
        )
        self.mlp_norm = get_layer_norm(
            hidden_size,
            layernorm_backend,
            elementwise_affine=False,
            eps=norm_eps,
        )

        # Optional gated temporal-attention sub-layer (own one-block projection + norm).
        self.temporal_attention = None
        self.temporal_attn_modulation = None
        self.temporal_attn_norm = None
        if temporal_attention:
            self.temporal_attention = TemporalAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                use_rope=use_rope,
                is_causal=is_causal,
                **(temporal_kwargs or {}),
            )
            self.temporal_attn_modulation = AdaLNModulation(
                hidden_size,
                condition_embed_dim,
                zero_init=adaln_zero_init,
            )
            self.temporal_attn_norm = get_layer_norm(
                hidden_size,
                layernorm_backend,
                elementwise_affine=False,
                eps=norm_eps,
            )

        self.cross_attention = (
            cross_attention() if cross_attention is not None else None
        )
        self.cross_attn_modulation = None
        self.cross_attn_norm = None
        if self.cross_attention is not None:
            self.cross_attn_modulation = AdaLNModulation(
                hidden_size,
                condition_embed_dim,
                zero_init=adaln_zero_init,
            )
            self.cross_attn_norm = get_layer_norm(
                hidden_size,
                layernorm_backend,
                elementwise_affine=False,
                eps=norm_eps,
            )

        self.drop_path = DropPath(drop_path)

        # Context-parallel reshard config (set via set_context_parallel).
        self._reshard_mode: Optional[str] = None
        self._reshard_target = None

    def initialize_weights(self) -> None:
        r"""Zero-init every adaLN-Zero modulation (when their ``zero_init`` is set).

        Returns
        -------
        None
            Delegates to each :class:`~physicsnemo.experimental.models.healda.adaln.AdaLNModulation`.
        """
        for mod in (
            self.norm1_modulation,
            self.temporal_attn_modulation,
            self.cross_attn_modulation,
        ):
            if mod is not None:
                mod.initialize_weights()

    def set_context_parallel(self, mode: Optional[str], target=None) -> None:
        r"""Configure the temporal time<->space reshard.

        Parameters
        ----------
        mode : str or None
            One of ``None`` (no resharding), ``"all_to_all"`` (manual collective
            over a ``ProcessGroup``), or ``"shardtensor"``
            (``ShardTensor.redistribute`` over a 1D mesh).
        target : ProcessGroup or DeviceMesh, optional, default=None
            The process group (``all_to_all``) or device mesh (``shardtensor``).

        Returns
        -------
        None
        """
        if mode not in (None, "all_to_all", "shardtensor"):
            raise ValueError(f"unknown reshard mode {mode!r}")
        if mode == "all_to_all" and not isinstance(target, dist.ProcessGroup):
            raise TypeError(
                "mode='all_to_all' requires target to be a ProcessGroup, got "
                f"{type(target).__name__}"
            )
        if mode == "shardtensor" and not isinstance(target, DeviceMesh):
            raise TypeError(
                "mode='shardtensor' requires target to be a DeviceMesh, got "
                f"{type(target).__name__}"
            )
        self._reshard_mode = mode
        self._reshard_target = target

    def _to_space_sharded(self, x: torch.Tensor) -> torch.Tensor:
        if self._reshard_mode == "all_to_all":
            return shard_x(x, self._reshard_target)
        if self._reshard_mode == "shardtensor":
            return shard_x_shardtensor(x, self._reshard_target)
        return x

    def _to_time_sharded(self, x: torch.Tensor) -> torch.Tensor:
        if self._reshard_mode == "all_to_all":
            return shard_t(x, self._reshard_target)
        if self._reshard_mode == "shardtensor":
            return shard_t_shardtensor(x, self._reshard_target)
        return x

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch time space hidden_size"],
        c: Float[torch.Tensor, "batch condition_embed_dim"],
        attn_kwargs: Optional[Dict[str, Any]] = None,
        cross_attn_kwargs: Optional[Dict[str, Any]] = None,
        temporal_attn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[torch.Tensor, "batch time space hidden_size"]:
        b, t, x, ch = hidden_states.shape
        if not torch.compiler.is_compiling():
            if hidden_states.ndim != 4:
                raise ValueError(
                    f"Expected hidden_states of shape (B, T, X, C), got {hidden_states.ndim}D "
                    f"tensor with shape {tuple(hidden_states.shape)}"
                )
            if c.shape != (b, self.condition_embed_dim):
                raise ValueError(
                    f"Expected conditioning of shape ({b}, {self.condition_embed_dim}), got "
                    f"tensor with shape {tuple(c.shape)}"
                )
            if self.cross_attention is not None and not cross_attn_kwargs:
                raise ValueError(
                    "cross_attention was provided at construction but no "
                    "cross_attn_kwargs was passed to forward"
                )

        (
            attn_shift,
            attn_scale,
            attn_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.norm1_modulation(c)

        # Spatial self-attention per frame (time folded into batch).
        normed = modulate(self.attn_norm(hidden_states), attn_shift, attn_scale)
        attn_out = self.attention(
            normed.reshape(b * t, x, ch), **(attn_kwargs or {})
        ).reshape(b, t, x, ch)
        hidden_states = gated_residual(
            hidden_states, attn_out, attn_gate, self.drop_path
        )

        # Cross-attention to the opaque injected context.
        if self.cross_attention is not None:
            shift, scale, gate = self.cross_attn_modulation(c)
            normed = modulate(self.cross_attn_norm(hidden_states), shift, scale)
            cross_out = self.cross_attention(normed, **cross_attn_kwargs)
            hidden_states = gated_residual(
                hidden_states, cross_out, gate, self.drop_path
            )

        # Temporal attention across time, with t<->x reshard around it.
        if self.temporal_attention is not None:
            hidden_states = self._to_space_sharded(hidden_states)
            shift, scale, gate = self.temporal_attn_modulation(c)
            normed = modulate(self.temporal_attn_norm(hidden_states), shift, scale)
            temporal_out = self.temporal_attention(
                normed, **(temporal_attn_kwargs or {})
            )
            hidden_states = gated_residual(
                hidden_states, temporal_out, gate, self.drop_path
            )
            hidden_states = self._to_time_sharded(hidden_states)

        # Feed-forward block, modulated by norm1_modulation's MLP shift/scale/gate.
        mlp_in = modulate(self.mlp_norm(hidden_states), mlp_shift, mlp_scale)
        hidden_states = gated_residual(
            hidden_states, self.linear(mlp_in), mlp_gate, self.drop_path
        )
        return hidden_states


class VideoDiT(Module):
    r"""Diffusion Transformer over :math:`(B, C, T, X)` inputs with an explicit time axis.

    The tokenizer and detokenizer are arbitrary modules that define the grid (e.g.
    HEALPix patch (de)tokenizers); the backbone operates on a flat token sequence.

    Parameters
    ----------
    tokenizer : torch.nn.Module
        Maps :math:`(B, C, T, X)` to a token sequence :math:`(B, T, X', D)`,
        defining the grid (e.g. a HEALPix patch tokenizer).
    detokenizer : torch.nn.Module
        Maps tokens :math:`(B, T, X', D)` and the conditioning embedding back to
        :math:`(B, C_{out}, T, X)`.
    hidden_size : int
        Transformer token dimension.
    num_heads : int
        Number of spatial-attention heads.
    num_layers : int
        Number of :class:`VideoDiTBlock` blocks.
    emb_channels : int, optional, default=None
        EDM conditioning-embedding dimension. Defaults to ``4 * hidden_size``.
    noise_channels : int, optional, default=None
        EDM noise positional-embedding dimension. Defaults to ``hidden_size``.
    condition_dim : int, optional, default=0
        Conditioning input dimension (0 = noise-only).
    temporal_attention : bool, optional, default=False
        Enable factorized temporal attention in every block.
    use_rope : bool, optional, default=True
        Apply rotary position embeddings along the time axis. When ``True``,
        builds a single :class:`~physicsnemo.nn.module.rope.RotaryEmbedding1DTables`
        provider shared by every block, instead of building tables per block.
    rope_base : int, optional, default=100
        Base frequency :math:`\theta` for the RoPE sinusoidal schedule.
    max_seq_len : int, optional, default=100
        Maximum temporal sequence length for the RoPE table pre-computation.
    temporal_kwargs : Dict[str, Any], optional, default=None
        Extra keyword arguments for the temporal-attention layers (e.g.
        ``linear_attention``, ``causal_window``).
    cross_attention : Callable[..., Module], optional, default=None
        Factory called once per block to build its cross-attention module
        (:class:`~physicsnemo.experimental.models.healda.attention_layers.CrossAttentionModuleBase`).
    is_causal : bool, optional, default=False
        Causal masking for temporal attention, fixed at construction.
    attention_backend : str or Module, optional, default="timm"
        Spatial-attention backend for the blocks.
    layernorm_backend : Literal["apex", "torch"], optional, default="torch"
        LayerNorm backend for the blocks' adaLN-Zero pre-norms.
    mlp_ratio : float, optional, default=4.0
        Block MLP hidden-dim multiplier.
    drop_path : float, optional, default=0.0
        Scalar drop-path used to build a linear schedule across blocks when
        ``drop_path_rates`` is ``None``.
    drop_path_rates : List[float], optional, default=None
        Explicit per-block drop-path rates; must have length ``num_layers``. When
        ``None``, the linear schedule from ``drop_path`` is used.
    conditioning_embedder : Literal["dit", "edm", "zero"] or ConditioningEmbedder, optional, default="edm"
        Conditioning embedder type or a pre-instantiated embedder. It must emit a
        pre-activation embedding (adaLN-Zero applies the ``SiLU``).
    conditioning_embedder_kwargs : Dict[str, Any], optional, default=None
        Extra keyword arguments for the conditioning embedder.
    dit_initialization : bool, optional, default=True
        If ``True``, apply DiT-style initialization (Xavier on linears, then
        delegate to the tokenizer, detokenizer, and blocks).
    adaln_zero_init : bool, optional, default=True
        Forwarded to every block's :class:`~physicsnemo.experimental.models.healda.adaln.AdaLNModulation` ``zero_init``.
    attn_kwargs : Dict[str, Any], optional, default=None
        Extra keyword arguments for the spatial-attention backend constructor.
    block_kwargs : Dict[str, Any], optional, default=None
        Extra keyword arguments forwarded to every block.

    Forward
    -------
    x : torch.Tensor
        Field sequence of shape :math:`(B, C, T, X)`.
    t : torch.Tensor
        Diffusion timestep (noise level) tensor of shape :math:`(B,)`.
    condition : torch.Tensor, optional
        Conditioning input of shape :math:`(B, \text{condition\_dim})`.
    attn_kwargs : Dict[str, Any], optional
        Forwarded to every block's spatial-attention backend forward.
    cross_attn_kwargs : Dict[str, Any], optional
        Forwarded to every block's injected cross-attention module.
    temporal_attn_kwargs : Dict[str, Any], optional
        Forwarded to every block's :class:`~physicsnemo.experimental.models.healda.attention_layers.TemporalAttention`
        forward. RoPE tables are appended automatically when RoPE
        is active.
    tokenizer_kwargs : Dict[str, Any], optional
        Extra keyword arguments forwarded to the tokenizer's forward.

    Outputs
    -------
    torch.Tensor
        Field sequence of shape :math:`(B, C_{out}, T, X)`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.healda import VideoDiT
    >>> from physicsnemo.nn.module.hpx.tokenizer import (
    ...     HEALPixPatchDetokenizer, HEALPixPatchTokenizer,
    ... )
    >>> hidden, level_fine, level_coarse = 64, 2, 1
    >>> tokenizer = HEALPixPatchTokenizer(
    ...     in_channels=2, hidden_size=hidden, level_fine=level_fine,
    ...     level_coarse=level_coarse, separate_time_axis=True,
    ... )
    >>> detokenizer = HEALPixPatchDetokenizer(
    ...     hidden_size=hidden, out_channels=2, level_coarse=level_coarse,
    ...     level_fine=level_fine, condition_dim=4 * hidden,
    ... )
    >>> model = VideoDiT(tokenizer, detokenizer, hidden_size=hidden, num_heads=2, num_layers=2)
    >>> b, t, npix = 1, 2, 12 * 4**level_fine
    >>> out = model(
    ...     torch.randn(b, 2, t, npix),
    ...     torch.zeros(b),
    ...     tokenizer_kwargs={
    ...         "second_of_day": torch.rand(b, t) * 86400.0,
    ...         "day_of_year": torch.rand(b, t) * 365.0,
    ...     },
    ... )
    >>> out.shape
    torch.Size([1, 2, 2, 192])
    """

    def __init__(
        self,
        tokenizer: nn.Module,
        detokenizer: nn.Module,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        *,
        emb_channels: Optional[int] = None,
        noise_channels: Optional[int] = None,
        condition_dim: int = 0,
        temporal_attention: bool = False,
        use_rope: bool = True,
        rope_base: int = 100,
        max_seq_len: int = 100,
        temporal_kwargs: Optional[Dict[str, Any]] = None,
        cross_attention: Optional[Callable[..., Module]] = None,
        is_causal: bool = False,
        attention_backend: Union[str, Module] = "timm",
        layernorm_backend: str = "torch",
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        drop_path_rates: Optional[List[float]] = None,
        conditioning_embedder: Union[str, ConditioningEmbedder] = "edm",
        conditioning_embedder_kwargs: Optional[Dict[str, Any]] = None,
        dit_initialization: bool = True,
        adaln_zero_init: bool = True,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        block_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(meta=MetaData())
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.hidden_size = hidden_size
        self.condition_dim = condition_dim

        if isinstance(conditioning_embedder, str):
            embedder_type = ConditioningEmbedderType[conditioning_embedder.upper()]
            embedder_kwargs = dict(conditioning_embedder_kwargs or {})
            if embedder_type is ConditioningEmbedderType.EDM:
                embedder_kwargs.setdefault(
                    "emb_channels", emb_channels or 4 * hidden_size
                )
                embedder_kwargs.setdefault(
                    "noise_channels", noise_channels or hidden_size
                )
            self.conditioning_embedder = get_conditioning_embedder(
                embedder_type,
                hidden_size=hidden_size,
                condition_dim=condition_dim,
                amp_mode=self.meta.amp_gpu,
                **embedder_kwargs,
            )
        elif isinstance(conditioning_embedder, ConditioningEmbedder):
            self.conditioning_embedder = conditioning_embedder
        else:
            raise TypeError(
                "conditioning_embedder must be a name in {'dit', 'edm', 'zero'} "
                "or a ConditioningEmbedder instance"
            )
        cond_dim = self.conditioning_embedder.output_dim

        self.temporal_rope = None
        if temporal_attention and use_rope:
            self.temporal_rope = RotaryEmbedding1DTables(
                head_dim=hidden_size // num_heads,
                max_seq_len=max_seq_len,
                theta=rope_base,
            )

        if drop_path_rates is None:
            drop_path_rates = [
                drop_path * i / max(1, num_layers - 1) for i in range(num_layers)
            ]
        elif len(drop_path_rates) != num_layers:
            raise ValueError(
                f"drop_path_rates length ({len(drop_path_rates)}) must match "
                f"num_layers ({num_layers})"
            )

        self.blocks = nn.ModuleList(
            [
                VideoDiTBlock(
                    hidden_size,
                    num_heads,
                    condition_embed_dim=cond_dim,
                    attention_backend=attention_backend,
                    layernorm_backend=layernorm_backend,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_rates[i],
                    temporal_attention=temporal_attention,
                    use_rope=use_rope,
                    temporal_kwargs=temporal_kwargs,
                    cross_attention=cross_attention,
                    is_causal=is_causal,
                    adaln_zero_init=adaln_zero_init,
                    attn_kwargs=attn_kwargs,
                    **(block_kwargs or {}),
                )
                for i in range(num_layers)
            ]
        )

        if dit_initialization:
            self.initialize_weights()

    def initialize_weights(self) -> None:
        r"""Apply DiT-style initialization.

        Applies Xavier uniform to all linear layers, then delegates to the
        tokenizer, detokenizer (when they expose ``initialize_weights``), and each
        block.

        Returns
        -------
        None
            Modifies module parameters in-place.
        """

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        for module in (self.tokenizer, self.detokenizer):
            if hasattr(module, "initialize_weights"):
                module.initialize_weights()
        for block in self.blocks:
            block.initialize_weights()

    def set_context_parallel(self, mode: Optional[str], target=None) -> None:
        r"""Configure the temporal time<->space reshard on every block.

        Parameters
        ----------
        mode : str or None
            ``None`` (no resharding), ``"all_to_all"`` (manual collective over a
            ``ProcessGroup``), or ``"shardtensor"`` (``ShardTensor.redistribute``
            over a 1D mesh).
        target : ProcessGroup or DeviceMesh, optional, default=None
            The process group (``all_to_all``) or device mesh (``shardtensor``).
        """
        for block in self.blocks:
            block.set_context_parallel(mode, target)

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels time space"],
        t: Float[torch.Tensor, " batch"],
        condition: Optional[Float[torch.Tensor, "batch condition_dim"]] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        cross_attn_kwargs: Optional[Dict[str, Any]] = None,
        temporal_attn_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[torch.Tensor, "batch out_channels time space"]:
        if not torch.compiler.is_compiling():
            if x.ndim != 4:
                raise ValueError(
                    f"Expected 4D input (B, C, T, X), got {x.ndim}D tensor with shape "
                    f"{tuple(x.shape)}"
                )
            b = x.shape[0]
            if t.ndim != 1 or t.shape[0] != b:
                raise ValueError(
                    f"Expected t of shape ({b},), got tensor with shape "
                    f"{tuple(t.shape)}"
                )
            if condition is not None:
                if condition.ndim != 2 or condition.shape != (b, self.condition_dim):
                    raise ValueError(
                        f"Expected condition of shape ({b}, {self.condition_dim}), got "
                        f"tensor with shape {tuple(condition.shape)}"
                    )

        # Tokenize: (B, C, T, X) -> (B, T, X', hidden)
        h = self.tokenizer(x, **(tokenizer_kwargs or {}))
        if not torch.compiler.is_compiling() and h.ndim != 4:
            raise ValueError(
                f"tokenizer must emit (B, T, X, hidden) for VideoDiT; got {h.ndim}D "
                "(use a tokenizer with separate_time_axis=True)."
            )

        emb = self.conditioning_embedder(t, condition=condition)
        block_temporal_kwargs = dict(temporal_attn_kwargs or {})
        if self.temporal_rope is not None:
            rope_cos, rope_sin = self.temporal_rope(seq_len=h.shape[1])
            block_temporal_kwargs["rope_cos"] = rope_cos
            block_temporal_kwargs["rope_sin"] = rope_sin
        for block in self.blocks:
            h = block(
                h,
                emb,
                attn_kwargs=attn_kwargs,
                cross_attn_kwargs=cross_attn_kwargs,
                temporal_attn_kwargs=block_temporal_kwargs,
            )

        # De-tokenize: (B, T, X', hidden) -> (B, C, T, X)
        return self.detokenizer(h, emb)
