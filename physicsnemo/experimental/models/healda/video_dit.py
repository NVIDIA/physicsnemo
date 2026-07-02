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
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.nn import (
    ConditioningEmbedder,
    ConditioningEmbedderType,
    get_conditioning_embedder,
)

from .video_dit_block import VideoDiTBlock


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
        Number of :class:`.video_dit_block.VideoDiTBlock` blocks.
    emb_channels : int, optional, default=None
        EDM conditioning-embedding dimension. Defaults to ``4 * hidden_size``.
    noise_channels : int, optional, default=None
        EDM noise positional-embedding dimension. Defaults to ``hidden_size``.
    condition_dim : int, optional, default=0
        Conditioning input dimension (0 = noise-only).
    temporal_attention : bool, optional, default=False
        Enable factorized temporal attention in every block.
    temporal_kwargs : Dict[str, Any], optional, default=None
        Extra keyword arguments for the temporal-attention layers.
    cross_attention : Callable[..., Module], optional, default=None
        Factory called once per block to build its cross-attention module
        (:class:`~physicsnemo.experimental.models.healda.cross_attention.CrossAttentionModuleBase`).
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
        Forwarded to every block's :class:`.adaln.AdaLNModulation` ``zero_init``.
    attn_kwargs : Dict[str, Any], optional, default=None
        Extra keyword arguments for the spatial-attention backend constructor.
    block_kwargs : Dict[str, Any], optional, default=None
        Extra keyword arguments forwarded to every block.

    Forward
    -------
    x : torch.Tensor
        Field sequence of shape :math:`(B, C, T, X)`.
    noise_labels : torch.Tensor
        Diffusion noise levels of shape :math:`(B,)`.
    condition : torch.Tensor, optional
        Conditioning input of shape :math:`(B, \text{condition\_dim})`.
    cross_attention_context : Any, optional
        Opaque per-call context consumed by the injected cross-attention module.
    tokenizer_kwargs : Dict[str, Any], optional
        Extra keyword arguments forwarded to the tokenizer's forward.

    Outputs
    -------
    torch.Tensor
        Field sequence of shape :math:`(B, C_{out}, T, X)`.
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
        noise_labels: Float[torch.Tensor, " batch"],
        condition: Optional[Float[torch.Tensor, "batch condition_dim"]] = None,
        cross_attention_context: Optional[Any] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[torch.Tensor, "batch out_channels time space"]:
        if not torch.compiler.is_compiling():
            if x.ndim != 4:
                raise ValueError(
                    f"Expected 4D input (B, C, T, X), got {x.ndim}D tensor with shape "
                    f"{tuple(x.shape)}"
                )
            b = x.shape[0]
            if noise_labels.ndim != 1 or noise_labels.shape[0] != b:
                raise ValueError(
                    f"Expected noise_labels of shape ({b},), got tensor with shape "
                    f"{tuple(noise_labels.shape)}"
                )
            if condition is not None:
                if condition.ndim != 2 or condition.shape != (b, self.condition_dim):
                    raise ValueError(
                        f"Expected condition of shape ({b}, {self.condition_dim}), got "
                        f"tensor with shape {tuple(condition.shape)}"
                    )

        # (B, C, T, X) -> (B, T, X', hidden)
        h = self.tokenizer(x, **(tokenizer_kwargs or {}))
        if not torch.compiler.is_compiling() and h.ndim != 4:
            raise ValueError(
                f"tokenizer must emit (B, T, X, hidden) for VideoDiT; got {h.ndim}D "
                "(use a tokenizer with separate_time_axis=True)."
            )

        emb = self.conditioning_embedder(noise_labels, condition=condition)
        for block in self.blocks:
            h = block(h, emb, cross_attention_context=cross_attention_context)

        return self.detokenizer(h, emb)
