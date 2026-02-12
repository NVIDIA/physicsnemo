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
from dataclasses import dataclass
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.models.dit.layers import (
    DiTBlock,
    get_tokenizer,
    get_detokenizer,
    TokenizerModuleBase,
    DetokenizerModuleBase,
)
from physicsnemo.experimental.models.dit.conditioning_embedders import (
    ConditioningEmbedder,
    ConditioningEmbedderType,
    get_conditioning_embedder,
)

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
    r"""
    The Diffusion Transformer (DiT) model.

    .. warning::
        This model is experimental and there may be changes in the future.

    Parameters
    ----------
    input_size : Union[int, Tuple[int]]
        Spatial dimensions of the input. If an integer is provided, the input is assumed to be on a square 2D domain.
        If a tuple is provided, the input is assumed to be on a multi-dimensional domain.
    in_channels : int
        The number of input channels.
    patch_size : Union[int, Tuple[int]], optional, default=(8, 8)
        The size of each image patch. If an integer is provided, a square 2D patch is assumed.
        If a tuple is provided, a multi-dimensional patch is assumed.
    tokenizer : Union[Literal["patch_embed_2d"], Module], optional, default="patch_embed_2d"
        The tokenizer to use. Either a string in ``{"patch_embed_2d"}`` or an instantiated PhysicsNeMo :class:`~physicsnemo.core.Module` implementing
        :class:`~physicsnemo.experimental.models.dit.layers.TokenizerModuleBase`, with forward accepting input of shape :math:`(B, C, *\text{spatial\_dims})` and returning :math:`(B, L, D)`.
    detokenizer : Union[Literal["proj_reshape_2d"], Module], optional, default="proj_reshape_2d"
        The detokenizer to use. Either a string in ``{"proj_reshape_2d"}`` or an instantiated PhysicsNeMo :class:`~physicsnemo.core.Module` implementing
        :class:`~physicsnemo.experimental.models.dit.layers.DetokenizerModuleBase`, with forward accepting :math:`(B, L, D)` and :math:`(B, D)` and returning :math:`(B, C, *\text{spatial\_dims})`.
    out_channels : Union[None, int], optional, default=None
        The number of output channels. If ``None``, set to ``in_channels``.
    hidden_size : int, optional, default=384
        The dimensionality of the transformer embeddings.
    depth : int, optional, default=12
        The number of transformer blocks.
    num_heads : int, optional, default=8
        The number of attention heads.
    mlp_ratio : float, optional, default=4.0
        The ratio of the MLP hidden dimension to the embedding dimension.
    attention_backend : Literal["timm", "transformer_engine", "natten2d"], optional, default="timm"
        The attention backend to use. See :class:`~physicsnemo.experimental.models.dit.layers.DiTBlock` for a description of each built-in backend.
    layernorm_backend : Literal["apex", "torch"], optional, default="torch"
        If ``"apex"``, uses FusedLayerNorm from apex. If ``"torch"``, uses :class:`torch.nn.LayerNorm`. Also passed to :class:`~physicsnemo.experimental.models.dit.layers.Natten2DSelfAttention` when ``qk_norm=True``.
    condition_dim : int, optional, default=None
        Dimensionality of conditioning. If ``None``, the model is unconditional.
    dit_initialization : bool, optional, default=True
        If ``True``, applies DiT-specific initialization.
    conditioning_embedder : Literal["dit", "edm", "zero"] or ConditioningEmbedder, optional, default="dit"
        The conditioning embedder type or an instantiated :class:`~physicsnemo.experimental.models.dit.conditioning_embedders.ConditioningEmbedder`.
    conditioning_embedder_kwargs : Dict[str, Any], optional, default={}
        Additional keyword arguments for the conditioning embedder.
    tokenizer_kwargs : Dict[str, Any], optional, default={}
        Additional keyword arguments for the tokenizer module.
    detokenizer_kwargs : Dict[str, Any], optional, default={}
        Additional keyword arguments for the detokenizer module.
    block_kwargs : Dict[str, Any], optional, default={}
        Additional keyword arguments for the DiTBlock modules.
    attn_kwargs : Dict[str, Any], optional, default={}
        Additional keyword arguments for the attention module constructor (e.g. ``na2d_kwargs`` when using ``attention_backend="natten2d"``).
    drop_path_rates : list[float], optional, default=None
        DropPath (stochastic depth) rates, one per block. Must have length equal to ``depth``. If ``None``, no drop path is applied.
    force_tokenization_fp32 : bool, optional, default=False
        If ``True``, forces tokenization and de-tokenization to run in fp32.

    Forward
    -------
    x : torch.Tensor
        Spatial inputs of shape :math:`(N, C, *\text{spatial\_dims})`. ``spatial_dims`` is determined by ``input_size``.
    t : torch.Tensor
        Diffusion timesteps of shape :math:`(N,)`.
    condition : Optional[torch.Tensor]
        Conditions of shape :math:`(N, d)`.
    p_dropout : Optional[Union[float, torch.Tensor]], optional
        Dropout probability for the intermediate dropout (pre-attention) in each DiTBlock. If ``None``, no dropout. If a scalar, same for all samples; if a tensor, shape :math:`(B,)` for per-sample dropout.
    attn_kwargs : Dict[str, Any], optional
        Additional keyword arguments passed to the attention module's forward method.
    tokenizer_kwargs : Dict[str, Any], optional
        Additional keyword arguments passed to the tokenizer's forward method.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(N, \text{out\_channels}, *\text{spatial\_dims})`.

    Notes
    -----
    Reference: Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers.
    In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4195-4205).

    Examples
    --------
    >>> model = DiT(
    ...     input_size=(32, 64),
    ...     patch_size=4,
    ...     in_channels=3,
    ...     out_channels=3,
    ...     condition_dim=8,
    ... )
    >>> x = torch.randn(2, 3, 32, 64)
    >>> t = torch.randint(0, 1000, (2,))
    >>> condition = torch.randn(2, 8)
    >>> output = model(x, t, condition)
    >>> output.shape
    torch.Size([2, 3, 32, 64])
    """

    __model_checkpoint_version__ = "0.2.0"
    __supported_model_checkpoint_version__ = {
        "0.1.0": "Automatically converting legacy DiT checkpoint timestep / conditioning embedder arguments.",
    }

    @classmethod
    def _backward_compat_arg_mapper(
        cls, version: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        r"""
        Map arguments from legacy checkpoints to the current format.

        Parameters
        ----------
        version : str
            Version of the checkpoint being loaded.
        args : Dict[str, Any]
            Arguments dictionary from the checkpoint.

        Returns
        -------
        Dict[str, Any]
            Updated arguments dictionary compatible with the current version.
        """
        args = super()._backward_compat_arg_mapper(version, args)
        if version != "0.1.0":
            return args
        
        if "timestep_embed_kwargs" in args:
            args["conditioning_embedder_kwargs"] = args.pop("timestep_embed_kwargs")
        return args

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
        conditioning_embedder: Literal["dit","edm","zero"] | ConditioningEmbedder = "dit",
        dit_initialization: Optional[int] = True,
        conditioning_embedder_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
        detokenizer_kwargs: Dict[str, Any] = {},
        block_kwargs: Dict[str, Any] = {},
        attn_kwargs: Dict[str, Any] = {},
        drop_path_rates: list[float] | None = None,
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

        # Conditioning embedder: accept enum or pre-instantiated Module
        if isinstance(conditioning_embedder, str):
            self.conditioning_embedder = get_conditioning_embedder(
                ConditioningEmbedderType[conditioning_embedder.upper()],
                hidden_size=hidden_size,
                condition_dim=condition_dim or 0,
                amp_mode=self.meta.amp_gpu,
                **conditioning_embedder_kwargs,
            )
        else:
            if not isinstance(conditioning_embedder, ConditioningEmbedder):
                raise TypeError("conditioning_embedder must be a ConditioningEmbedderType or a Module implementing the ConditioningEmbedder protocol")
            self.conditioning_embedder = conditioning_embedder

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


        # Validate drop_path_rates
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth
        else:
            if len(drop_path_rates) != depth:
                raise ValueError(
                    f"drop_path_rates length ({len(drop_path_rates)}) must match DiT depth ({depth})"
                )

        blocks = []
        for i in range(depth):
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
                    drop_path=drop_path_rates[i],
                    condition_embed_dim=self.conditioning_embedder.output_dim,
                    **block_kwargs,
                    **attn_kwargs,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if dit_initialization:
            self.initialize_weights()

        self.force_tokenization_fp32 = force_tokenization_fp32
        self.register_load_state_dict_pre_hook(self._migrate_legacy_checkpoint)

    @staticmethod
    def _migrate_legacy_checkpoint(
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        r"""Remap legacy state_dict keys where timestep embedder was at root.

        Previous versions stored the timestep embedder at root
        (e.g. ``t_embedder.mlp.0.weight``). The current model nests it under
        ``conditioning_embedder`` (e.g. ``conditioning_embedder.t_embedder.mlp.0.weight``).
        This pre-hook rewrites those keys in-place so loading succeeds. It also
        drops the positional embedding ``freqs`` key, which is not part of the state_dict
        anymore due to the usage of ``persistent=False``.

        Parameters
        ----------
        module : torch.nn.Module
            The module being loaded (unused; required by ``register_load_state_dict_pre_hook``).
        state_dict : dict
            State dict being loaded; modified in-place.
        prefix : str
            Prefix for the module (unused).
        local_metadata : dict, optional
            Local metadata (unused).
        strict : bool
            Whether strict loading is requested (unused).
        missing_keys : list of str
            List of missing keys (unused).
        unexpected_keys : list of str
            List of unexpected keys (unused).
        error_msgs : list of str
            Error messages (unused).

        Returns
        -------
        None
            Modifies ``state_dict`` in-place; no return value.
        """
        legacy_prefix = "t_embedder."
        new_prefix = "conditioning_embedder.t_embedder."

        # Iterate over a snapshot of keys to avoid mutating dict while iterating
        for old_key in list(state_dict.keys()):
            if not old_key.startswith(legacy_prefix):
                continue
            new_key = new_prefix + old_key[len(legacy_prefix) :]
            if old_key == legacy_prefix+"freqs":
                del state_dict[old_key]
            elif new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)

    def initialize_weights(self):
        r"""Apply DiT-specific weight initialization.

        Applies Xavier uniform to linear layers, then delegates to tokenizer,
        detokenizer, and each block's ``initialize_weights``.

        Parameters
        ----------
        None
            Uses ``self`` (module state).

        Returns
        -------
        None
            Modifies module parameters in-place.
        """
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
        attn_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
    ) -> torch.Tensor:
        # Tokenize: (B, C, H, W) -> (B, L, D)
        if self.force_tokenization_fp32:
            dtype = x.dtype
            x = x.to(torch.float32)
            with torch.autocast(device_type="cuda", enabled=False):
                x = self.tokenizer(x, **tokenizer_kwargs)
            x = x.to(dtype)
        else:
            x = self.tokenizer(x, **tokenizer_kwargs)

        # Compute conditioning embedding
        c = self.conditioning_embedder(t, condition=condition)  # (B, D)
        
        for block in self.blocks:
            x = block(x, c, p_dropout=p_dropout, attn_kwargs={**self.attn_kwargs_forward, **attn_kwargs})  # (B, L, D)

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
