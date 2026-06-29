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
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn import (
    ConditioningEmbedder,
    ConditioningEmbedderType,
    DetokenizerModuleBase,
    DiTBlock,
    TokenizerModuleBase,
    get_conditioning_embedder,
    get_detokenizer,
    get_tokenizer,
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
    tokenizer : Union[Literal["patch_embed_2d", "hpx_patch_embed"], Module], optional, default="patch_embed_2d"
        The tokenizer to use. Either a string in ``{"patch_embed_2d", "hpx_patch_embed"}`` or an instantiated PhysicsNeMo :class:`~physicsnemo.core.Module` implementing
        :class:`~physicsnemo.nn.TokenizerModuleBase`, with forward accepting input of shape :math:`(B, C, *\text{spatial\_dims})` and returning :math:`(B, L, D)`.
    detokenizer : Union[Literal["proj_reshape_2d", "proj_reshape_2d_conv", "hpx_patch_detokenizer"], Module], optional, default="proj_reshape_2d"
        The detokenizer to use. Either a string in ``{"proj_reshape_2d", "proj_reshape_2d_conv", "hpx_patch_detokenizer"}`` or an instantiated PhysicsNeMo :class:`~physicsnemo.core.Module` implementing
        :class:`~physicsnemo.nn.DetokenizerModuleBase`, with forward accepting :math:`(B, L, D)` and :math:`(B, D)` and returning :math:`(B, C, *\text{spatial\_dims})`.
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
    attention_backend : Literal["timm", "transformer_engine", "natten2d", "natten2d_rope"], optional, default="timm"
        The attention backend to use. See :class:`~physicsnemo.nn.DiTBlock` for a description of each built-in backend. ``"natten2d_rope"`` applies axial 2D rotary position embeddings inside NATTEN; selecting it forces ``pos_embed="none"`` in the tokenizer (additive positional embedding is disabled to avoid double-counting position) and emits a warning if a conflicting ``pos_embed`` was explicitly passed.
    layernorm_backend : Literal["apex", "torch"], optional, default="torch"
        If ``"apex"``, uses FusedLayerNorm from apex. If ``"torch"``, uses :class:`torch.nn.LayerNorm`. Also passed to :class:`~physicsnemo.nn.Natten2DSelfAttention` when ``qk_norm=True``.
    condition_dim : int, optional, default=None
        Dimensionality of conditioning. If ``None``, the model is unconditional.
    dit_initialization : bool, optional, default=True
        If ``True``, applies DiT-specific initialization.
    conditioning_embedder : Literal["dit", "edm", "zero"] or ConditioningEmbedder, optional, default="dit"
        The conditioning embedder type or an instantiated :class:`~physicsnemo.nn.ConditioningEmbedder`.
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
    use_nan_mask_tokens : bool, optional, default=False
        If ``True``, every NATTEN block overwrites invalid spatial tokens with a per-block learned ``mask_token`` immediately before the QKV projection, so the neighborhood window mixes in a single learned feature instead of corrupted (e.g. NaN-padded) signal. Requires a NATTEN attention backend (``"natten2d"`` or ``"natten2d_rope"``). This only allocates the learned ``mask_token`` parameters; the invalid pattern itself is supplied dynamically per forward call via the ``invalid_mask`` argument (see :meth:`forward`). When no ``invalid_mask`` is passed, all tokens are treated as valid and behavior is identical to ``use_nan_mask_tokens=False``.

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
    invalid_mask : Optional[torch.Tensor], optional
        Per-sample boolean (or float) invalid-region mask of shape :math:`(N, *\text{spatial\_dims})` or :math:`(N, 1, *\text{spatial\_dims})`, ``True`` (or ``1``) at invalid pixels (e.g. NaN-padded / outside sensor coverage). It is max-pooled to patch (token) granularity and the flagged tokens are replaced by each NATTEN block's learned ``mask_token`` before attention. The pattern may differ per sample (dynamic, batch-variable masking) and per forward call. Requires ``use_nan_mask_tokens=True``. Because the splice does not sanitize non-finite values, invalid pixels in ``x`` must be finite (e.g. pass ``x`` through :func:`torch.nan_to_num` first). Under domain parallelism, pass ``invalid_mask`` as a ``ShardTensor`` sharded along height exactly like ``x``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(N, \text{out\_channels}, *\text{spatial\_dims})`.

    Notes
    -----
    Reference: Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers.
    In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4195-4205).

    Under domain parallelism (the model wrapped with ``distribute_module``), the
    spatial input ``x`` is a sharded ``ShardTensor`` while the model's buffers and
    parameters are ``DTensor``s. The non-spatial inputs ``t`` and ``condition``
    must therefore be passed as ``Replicate`` ``DTensor``s on the same mesh (rather
    than plain tensors), so they compose with the distributed buffers/parameters
    (e.g. the timestep embedder's ``freqs``).

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
        tokenizer: Union[
            Literal["patch_embed_2d", "hpx_patch_embed"], Module
        ] = "patch_embed_2d",
        detokenizer: Union[
            Literal["proj_reshape_2d", "proj_reshape_2d_conv", "hpx_patch_detokenizer"],
            Module,
        ] = "proj_reshape_2d",
        out_channels: Optional[int] = None,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attention_backend: Literal[
            "timm", "transformer_engine", "natten2d", "natten2d_rope"
        ] = "timm",
        layernorm_backend: Literal["apex", "torch"] = "torch",
        condition_dim: Optional[int] = None,
        conditioning_embedder: Literal["dit", "edm", "zero"]
        | ConditioningEmbedder = "dit",
        dit_initialization: Optional[int] = True,
        conditioning_embedder_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
        detokenizer_kwargs: Dict[str, Any] = {},
        block_kwargs: Dict[str, Any] = {},
        attn_kwargs: Dict[str, Any] = {},
        drop_path_rates: list[float] | None = None,
        force_tokenization_fp32: bool = False,
        use_nan_mask_tokens: bool = False,
    ):
        super().__init__(meta=MetaData())
        self.input_size = (
            input_size
            if isinstance(input_size, (tuple, list))
            else (input_size, input_size)
        )
        self.in_channels = in_channels
        if out_channels:
            self.out_channels = out_channels
        else:
            self.out_channels = in_channels
        self.patch_size = (
            patch_size
            if isinstance(patch_size, (tuple, list))
            else (patch_size, patch_size)
        )
        self.num_heads = num_heads
        self.condition_dim = condition_dim

        # Input validation
        if attention_backend not in [
            "timm",
            "transformer_engine",
            "natten2d",
            "natten2d_rope",
        ]:
            raise ValueError(
                "attention_backend must be one of 'timm', 'transformer_engine', 'natten2d', 'natten2d_rope'"
            )

        if layernorm_backend not in ["apex", "torch"]:
            raise ValueError("layernorm_backend must be one of 'apex', 'torch'")

        is_natten = attention_backend in ("natten2d", "natten2d_rope")

        # Latent (token) grid size, used by the NATTEN backends.
        self._latent_h = self.input_size[0] // self.patch_size[0]
        self._latent_w = self.input_size[1] // self.patch_size[1]
        latent_hw = (self._latent_h, self._latent_w)

        # Keyword arguments threaded into every attention module's forward.
        if is_natten:
            self.attn_kwargs_forward = {"latent_hw": latent_hw}
        else:
            self.attn_kwargs_forward = {}

        # NaN-mask-token handling: replace invalid spatial tokens with a learned
        # per-block mask token before NATTEN. Only valid with a NATTEN backend.
        self._use_nan_mask_tokens = use_nan_mask_tokens
        if use_nan_mask_tokens and not is_natten:
            raise ValueError(
                "use_nan_mask_tokens=True requires a NATTEN attention backend "
                "('natten2d' or 'natten2d_rope')"
            )

        # Constructor-time attention kwargs (copied so the caller's dict is not
        # mutated). RoPE attention needs the latent grid at construction so its
        # cos/sin tables can be precomputed; the mask-token backends need to
        # allocate their learned mask parameter.
        attn_kwargs = dict(attn_kwargs)
        if attention_backend == "natten2d_rope":
            attn_kwargs["latent_hw"] = latent_hw
        if use_nan_mask_tokens:
            attn_kwargs["use_mask_token"] = True

        # Using RoPE alongside an additive positional embedding double-counts
        # position. Force the (patch-based) tokenizer's pos_embed to "none",
        # warning if a conflicting value was explicitly requested.
        if attention_backend == "natten2d_rope" and tokenizer == "patch_embed_2d":
            tokenizer_kwargs = dict(tokenizer_kwargs)
            requested_pos_embed = tokenizer_kwargs.get("pos_embed", None)
            if requested_pos_embed not in (None, "none"):
                warnings.warn(
                    "attention_backend='natten2d_rope' uses rotary position "
                    "embeddings; overriding the requested "
                    f"pos_embed={requested_pos_embed!r} with 'none' to avoid "
                    "double-counting the positional signal.",
                    UserWarning,
                    stacklevel=2,
                )
            tokenizer_kwargs["pos_embed"] = "none"

        if isinstance(tokenizer, str) and tokenizer not in [
            "patch_embed_2d",
            "hpx_patch_embed",
        ]:
            raise ValueError("tokenizer must be 'patch_embed_2d' or 'hpx_patch_embed'")

        if isinstance(detokenizer, str) and detokenizer not in [
            "proj_reshape_2d",
            "proj_reshape_2d_conv",
            "hpx_patch_detokenizer",
        ]:
            raise ValueError(
                "detokenizer must be 'proj_reshape_2d', 'proj_reshape_2d_conv', or 'hpx_patch_detokenizer'"
            )

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
                raise TypeError(
                    "tokenizer must be a string or a physicsnemo.core.Module instance subclassing physicsnemo.nn.TokenizerModuleBase"
                )
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
                raise TypeError(
                    "conditioning_embedder must be a ConditioningEmbedderType or a Module implementing the ConditioningEmbedder protocol"
                )
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
                raise TypeError(
                    "detokenizer must be a string or a physicsnemo.core.Module instance subclassing physicsnemo.nn.DetokenizerModuleBase"
                )
            self.detokenizer = detokenizer

        # Validate drop_path_rates
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth
        else:
            if len(drop_path_rates) != depth:
                raise ValueError(
                    f"drop_path_rates length ({len(drop_path_rates)}) must match DiT depth ({depth})"
                )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    attention_backend=attention_backend,
                    layernorm_backend=layernorm_backend,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_rates[i],
                    condition_embed_dim=self.conditioning_embedder.output_dim,
                    **block_kwargs,
                    **attn_kwargs,
                )
                for i in range(depth)
            ]
        )

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
            if old_key == legacy_prefix + "freqs":
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

    def _pixel_mask_to_token_mask(
        self,
        invalid_mask: torch.Tensor,
    ) -> Float[torch.Tensor, "batch sequence"]:
        r"""Reduce a per-sample pixel-level invalid mask to token granularity.

        Aggregates an invalid-pixel mask of shape :math:`(B, H, W)` or
        :math:`(B, 1, H, W)` to a flattened patch-level mask of shape
        :math:`(B, L)` with ``L = h_lat * w_lat``: a patch (token) is marked
        invalid if *any* pixel in its ``patch_size`` block is invalid. The
        flattening order (row-major over ``(h_lat, w_lat)``) matches the
        tokenizer's ``flatten(2)`` token ordering, so the returned mask aligns
        positionally with the token sequence consumed by the NATTEN blocks.

        Implemented with :func:`torch.nn.functional.max_pool2d`, which is
        registered for ``ShardTensor``: with ``kernel_size == stride ==
        patch_size`` the pooling is non-overlapping and stays local under
        height-sharded domain parallelism, mirroring how the tokenizer's
        strided convolution produces the sharded token sequence.

        Parameters
        ----------
        invalid_mask : torch.Tensor
            Boolean/float mask, ``True`` (or ``>0``) at invalid pixels.

        Returns
        -------
        torch.Tensor
            Boolean token mask of shape :math:`(B, L)`.
        """
        if invalid_mask.ndim == len(self.input_size) + 1:
            # (B, *spatial) -> (B, 1, *spatial)
            invalid_mask = invalid_mask.unsqueeze(1)
        if invalid_mask.ndim != len(self.input_size) + 2 or invalid_mask.shape[1] != 1:
            raise ValueError(
                "invalid_mask must have shape (B, *spatial_dims) or "
                "(B, 1, *spatial_dims) matching the DiT spatial input; got "
                f"shape {tuple(invalid_mask.shape)}"
            )

        # Any invalid pixel within a patch -> invalid token. Pool in float so
        # the op is well-defined; threshold back to bool afterwards.
        patch_mask = F.max_pool2d(
            invalid_mask.to(torch.float32),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )  # (B, 1, h_lat, w_lat)
        # (B, 1, h_lat, w_lat) -> (B, L). Row-major flatten matches the
        # tokenizer; under ShardTensor this merges the height shard with the
        # (replicated) width axis, exactly as the static-buffer path did.
        return (patch_mask > 0).reshape(patch_mask.shape[0], -1)

    def forward(
        self,
        x: Float[torch.Tensor, "batch in_channels *spatial_dims"],
        t: Float[torch.Tensor, " batch"],
        condition: Optional[Float[torch.Tensor, "batch condition_dim"]] = None,
        p_dropout: Optional[float | Float[torch.Tensor, " batch"]] = None,
        attn_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
        invalid_mask: Optional[Float[torch.Tensor, " batch *spatial_dims"]] = None,
    ) -> Float[torch.Tensor, "batch out_channels *spatial_dims"]:
        if invalid_mask is not None and not self._use_nan_mask_tokens:
            raise ValueError(
                "invalid_mask was provided but the DiT was constructed with "
                "use_nan_mask_tokens=False, so no learned mask tokens were "
                "allocated. Rebuild the model with use_nan_mask_tokens=True to "
                "use dynamic invalid-region masking."
            )

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

        block_attn_kwargs = {**self.attn_kwargs_forward, **attn_kwargs}
        if invalid_mask is not None:
            # Reduce the (B, *spatial) pixel mask to a (B, L) token mask aligned
            # with the token sequence. Under domain parallelism invalid_mask is a
            # ShardTensor sharded along height like x, so the pooled token mask
            # is sharded along the sequence axis exactly like the tokens.
            block_attn_kwargs.setdefault(
                "invalid_token_mask", self._pixel_mask_to_token_mask(invalid_mask)
            )

        for block in self.blocks:
            x = block(
                x,
                c,
                p_dropout=p_dropout,
                attn_kwargs=block_attn_kwargs,
            )  # (B, L, D)

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
