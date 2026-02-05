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

from typing import Tuple, Union, Optional, Literal, Dict, Any
import torch
import torch.nn as nn

from physicsnemo.nn import PositionalEmbedding, Linear
from dataclasses import dataclass
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.models.dit import DiTBlock
from physicsnemo.experimental.models.dit.layers import get_tokenizer, get_detokenizer, TokenizerModuleBase, DetokenizerModuleBase


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


class DiT(Module):
    """
    Warning
    -----------
    This model is experimental and there may be changes in the future.
    
    The Diffusion Transformer (DiT) model.

    Parameters
    -----------
    input_size (Union[int, Tuple[int]]):
        Spatial dimensions of the input. If an integer is provided, the input is assumed to be on a square 2D domain.
        If a tuple is provided, the input is assumed to be on a multi-dimensional domain.
    in_channels (int):
        The number of input channels..
    patch_size (Union[int, Tuple[int]], optional):
        The size of each image patch. Defaults to (8,8). If an integer is provided, the patch_size is assumed to be a square 2D patch.
        If a tuple is provided, the patch_size is assumed to be a multi-dimensional patch.
    tokenizer (Union[Literal["patch_embed_2d"], Module], optional):
        The tokenizer to use. Defaults to 'patch_embed_2d'. You may provide:
        - A string in {"patch_embed_2d"} to select a built-in tokenizer. Built-in tokenizers include:
            - 'patch_embed_2d': Uses a standard PatchEmbed2D to project the input image to a sequence of tokens.
        - An instantiated PhysicsNeMo `Module` implementing the tokenizer interface defined in :class:`physicsnemo.experimental.models.dit.layers.TokenizerModuleBase`.
          The tokenizer module must be a subclass of :class:`physicsnemo.experimental.models.dit.layers.TokenizerModuleBase`, and
          define a forward method and an initialize_weights method, with the forward method accepting an input Tensor of shape (B, C, *spatial_dims) and returning (B, L, D).
    detokenizer (Union[Literal["proj_reshape_2d"], Module], optional):
        The detokenizer to use. Defaults to 'proj_reshape_2d'. You may provide:
        - A string in {"proj_reshape_2d"} to select a built-in detokenizer. Built-in tokenizers include:
            - 'proj_reshape_2d': Uses a standard project and reshape operation to convert the token sequence back to an image.
        - An instantiated PhysicsNeMo `Module` implementing the detokenizer interface defined in :class:`physicsnemo.experimental.models.dit.layers.DetokenizerModuleBase`.
          The detokenizer module must be a subclass of :class:`physicsnemo.experimental.models.dit.layers.DetokenizerModuleBase`, and
          define a forward method and an initialize_weights method, with the forward method accepting an input Tensor of shape (B, L, D) and (B, D) and returning (B, C, *spatial_dims).
    out_channels (Union[None, int], optional):
        The number of output channels. If None, it is `in_channels`. Defaults to None,
        which means the output will have the same number of channels as the input.
    hidden_size (int, optional):
        The dimensionality of the transformer embeddings. Defaults to 384.
    depth (int, optional):
        The number of transformer blocks. Defaults to 12.
    num_heads (int, optional):
        The number of attention heads. Defaults to 8.
    mlp_ratio (float, optional):
        The ratio of the MLP hidden dimension to the embedding dimension. Defaults to 4.0.
    attention_backend (Literal["timm", "transformer_engine", "natten2d"], optional):
        The attention backend to use. Defaults to 'timm'. You may provide:
        - A string in {"timm", "transformer_engine", "natten2d"} to select a built-in backend.
          See :class:`physicsnemo.experimental.models.dit.layers.DiTBlock` for a description of each built-in backend.
    layernorm_backend (Literal["apex", "torch"], optional):
        If 'apex', uses FusedLayerNorm from apex. If 'torch', uses LayerNorm from torch.nn. Defaults to 'apex'.
    condition_dim (int, optional):
        Dimensionality of conditioning. If None, the model is unconditional. Defaults to None.
    dit_initialization (bool, optional):
        If True, applies the DiT specific initialization. Defaults to True.
    tokenizer_kwargs (Dict[str, Any], optional):
        Additional keyword arguments for the tokenizer module.
    detokenizer_kwargs (Dict[str, Any], optional):
        Additional keyword arguments for the detokenizer module.
    block_kwargs (Dict[str, Any], optional):
        Additional keyword arguments for the DiTBlock modules.
    timestep_embed_kwargs (Dict[str, Any], optional):
        Additional keyword arguments to be passed to :class:`physicsnemo.nn.PositionalEmbedding`.
    attn_kwargs (Dict[str, Any], optional):
        Additional keyword arguments for the attention module constructor, if using a custom attention backend.
    force_tokenization_fp32 (bool, optional):
        If True, forces the tokenization and de-tokenization operations to be run in fp32. Defaults to False.
    
    Forward
    -------
    x (torch.Tensor):
        (N, C, *spatial_dims) tensor of spatial inputs. `spatial_dims` is determined by the input_size/dimensionality.
    t (torch.Tensor):
        (N,) tensor of diffusion timesteps.
    condition (Optional[torch.Tensor]):
        (N, d) tensor of conditions.
    p_dropout (Optional[float | torch.Tensor]):
        The dropout probability for the intermediate dropout module (pre-attention) in the DiTBlock. If None, no dropout will be applied.
        If a scalar, the same dropout probability will be applied to all samples in the batch.
        Otherwise, it should be a tensor of shape (B,) to apply per-sample dropout to each sample in a batch.

    Returns
    -------
    torch.Tensor:
        The output tensor of shape (N, out_channels, *spatial_dims). `spatial_dims` is determined by the input_size/dimensionality.
    
    Note
    -----
    Reference: Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers.
    In Proceedings of the IEEE/CVF international conference on computer vision (pp. 4195-4205).

    Example
    --------
    >>> model = DiT(
    ...     input_size=(32,64),
    ...     patch_size=4,
    ...     in_channels=3,
    ...     out_channels=3,
    ...     condition_dim=8,
    ... )
    >>> x = torch.randn(2, 3, 32, 64)     # [B, C, H, W]
    >>> t = torch.randint(0, 1000, (2,))  # [B]
    >>> condition = torch.randn(2, 8)    # [B, d]
    >>> output = model(x, t, condition)
    >>> output.size()
    torch.Size([2, 3, 32, 64])
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int]],
        in_channels: int,
        patch_size: Union[int, Tuple[int]] = (8, 8),
        tokenizer: Union[Literal["patch_embed_2d"], Module] = "patch_embed_2d",
        detokenizer: Union[Literal["proj_reshape_2d"], Module] = "proj_reshape_2d",
        out_channels: Optional[int] = None,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attention_backend: Literal["timm", "transformer_engine", "natten2d"] = "timm",
        layernorm_backend: Literal["apex", "torch"] = "torch",
        condition_dim: Optional[int] = None,
        dit_initialization: Optional[int] = True,
        tokenizer_kwargs: Dict[str, Any] = {},
        detokenizer_kwargs: Dict[str, Any] = {},
        block_kwargs: Dict[str, Any] = {},
        timestep_embed_kwargs: Dict[str, Any] = {},
        attn_kwargs: Dict[str, Any] = {},
        force_tokenization_fp32: bool = False,
    ):
        super().__init__(meta=MetaData())
        self.input_size = input_size if isinstance(input_size, (tuple, list)) else (input_size, input_size)
        self.in_channels = in_channels
        if out_channels:
            self.out_channels = out_channels
        else:
            self.out_channels = in_channels
        self.patch_size = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        self.num_heads = num_heads
        self.condition_dim = condition_dim
        if attention_backend == "natten2d":
            latent_hw = (
                self.input_size[0] // self.patch_size[0],
                self.input_size[1] // self.patch_size[1]
            )
            self.attn_kwargs_forward = {"latent_hw": latent_hw}
        else:
            self.attn_kwargs_forward = {}

        # Input validation
        if attention_backend not in ["timm", "transformer_engine", "natten2d"]:
            raise ValueError("attention_backend must be one of 'timm', 'transformer_engine', 'natten2d'")

        if layernorm_backend not in ["apex", "torch"]:
            raise ValueError("layernorm_backend must be one of 'apex', 'torch'")

        if isinstance(tokenizer, str) and tokenizer not in ["patch_embed_2d"]:
            raise ValueError("tokenizer must be 'patch_embed_2d'")

        if isinstance(detokenizer, str) and detokenizer not in ["proj_reshape_2d"]:
            raise ValueError("detokenizer must be 'proj_reshape_2d'")

        # Tokenizer module: accept string or pre-instantiated PhysicsNeMo Module
        if isinstance(tokenizer, str):
            self.tokenizer = get_tokenizer(
                input_size=self.input_size,
                patch_size=self.patch_size,
                in_channels=in_channels,
                hidden_size=hidden_size,
                tokenizer=tokenizer,
                **tokenizer_kwargs,
            )
        else:
            if not isinstance(tokenizer, TokenizerModuleBase):
                raise TypeError("tokenizer must be a string or a physicsnemo.core.Module instance subclassing physicsnemo.experimental.models.dit.layers.TokenizerModuleBase")
            self.tokenizer = tokenizer

        self.t_embedder = PositionalEmbedding(hidden_size, amp_mode=self.meta.amp_gpu, learnable=True, **timestep_embed_kwargs)
        self.cond_embedder = (
            Linear(
                in_features=condition_dim,
                out_features=hidden_size,
                bias=False,
                amp_mode=self.meta.amp_gpu,
                init_mode="kaiming_uniform",
                init_weight=0,
                init_bias=0,
            )
            if condition_dim
            else None
        )

        # Detokenizer module: accept string or pre-instantiated PhysicsNeMo Module
        if isinstance(detokenizer, str):
            self.detokenizer = get_detokenizer(
                input_size=self.input_size,
                patch_size=self.patch_size,
                out_channels=self.out_channels,
                hidden_size=hidden_size,
                layernorm_backend=layernorm_backend,
                detokenizer=detokenizer,
                **detokenizer_kwargs,
            )
        else:
            if not isinstance(detokenizer, DetokenizerModuleBase):
                raise TypeError("detokenizer must be a string or a physicsnemo.core.Module instance subclassing physicsnemo.experimental.models.dit.layers.DetokenizerModuleBase")
            self.detokenizer = detokenizer


        blocks = []
        for _ in range(depth):
            if isinstance(attention_backend, str):
                attn_module = attention_backend
            else:
                custom_attn_module_constructor = attention_backend.__class__
                attn_module = custom_attn_module_constructor(hidden_size=hidden_size, num_heads=num_heads, **attn_kwargs)
            
            blocks.append(
                DiTBlock(
                    hidden_size,
                    num_heads,
                    attention_backend=attn_module,
                    layernorm_backend=layernorm_backend,
                    mlp_ratio=mlp_ratio,
                    **block_kwargs,
                    **attn_kwargs,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if dit_initialization:
            self.initialize_weights()

        self.force_tokenization_fp32 = force_tokenization_fp32

    def initialize_weights(self):
        # Apply a basic Xavier uniform initialization to all linear layers.
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Delegate custom weight initialization to the tokenizer, detokenizer, and blocks
        self.tokenizer.initialize_weights()
        self.detokenizer.initialize_weights()
        for block in self.blocks:
            block.initialize_weights()
        
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        p_dropout: Optional[float | torch.Tensor] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # Tokenize: (B, C, H, W) -> (B, L, D)
        if self.force_tokenization_fp32:
            dtype = x.dtype
            x = x.to(torch.float32)
            with torch.autocast(device_type="cuda", enabled=False):
                x = self.tokenizer(x)
            x = x.to(dtype)
        else:
            x = self.tokenizer(x)

        t = self.t_embedder(t)  # (B, D)

        # Handle conditioning
        if self.cond_embedder is not None:
            if condition is None:
                # Fallback to using only timestep embedding if conditioning is not provided
                c = t
            else:
                condition_embedding = self.cond_embedder(condition)  # (B, D)
                c = t + condition_embedding  # (B, D)
        else:
            if condition is not None:
                raise ValueError("Conditioning was provided but DiT has no conditioning embedding module.")
            c = t  # (B, D)
        
        for block in self.blocks:
            x = block(x, c, p_dropout=p_dropout, attn_kwargs={**self.attn_kwargs_forward, **(attn_kwargs or {})})  # (B, L, D)

        # De-tokenize: (B, L, D) -> (B, C, H, W)
        if self.force_tokenization_fp32:
            dtype = x.dtype
            x = x.to(torch.float32)
            with torch.autocast(device_type="cuda", enabled=False):
                x = self.detokenizer(x, c)
            x = x.to(dtype)
        else:
            x = self.detokenizer(x, c)
        
        return x
