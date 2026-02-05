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
from typing import Any, Dict, Literal, Union, Optional, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from importlib.metadata import version
from packaging.version import Version

# Import Attention from timm: location depends on timm version
try:
    timm_version = version("timm")
    timm_v1_0_16 = Version(timm_version) >= Version("1.0.16")
except Exception:
    timm_v1_0_16 = False

if timm_v1_0_16:
    from timm.layers.attention import Attention
else:
    from timm.models.vision_transformer import Attention

try:
    from transformer_engine.pytorch import MultiheadAttention
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False

try:
    from apex.normalization import FusedLayerNorm

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False

try:
    from natten.functional import na2d
    NATTEN_AVAILABLE = True
except ImportError:
    NATTEN_AVAILABLE = False

from physicsnemo.core import Module
from physicsnemo.nn import Mlp
from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.domain_parallel.shard_utils.natten_patches import partial_na2d
from physicsnemo.nn.module.utils import PatchEmbed2D


def get_layer_norm(
    hidden_size: int,
    layernorm_backend: Literal["apex", "torch"],
    elementwise_affine: bool = False,
    eps: float = 1e-6,
) -> nn.Module:
    """Construct a LayerNorm module based on the selected backend.

    Parameters
    ----------
    hidden_size: int
        Normalized feature dimension.
    layernorm_backend: Literal["apex", "torch"]
        Implementation selector.
    elementwise_affine: bool
        Whether to learn per-element affine parameters.
    eps: float
        Numerical stability epsilon.

    Returns
    -------
    nn.Module
        A configured LayerNorm module from Apex or Torch. The returned module is a subclass of nn.Module
        and expects a tensor shape (B, L, D) as input, returning a normalized tensor of the same shape.
    """
    if layernorm_backend == "apex":
        if not APEX_AVAILABLE:
            raise ImportError(
                "Apex is not available. Please install Apex to use FusedLayerNorm or choose 'torch'."
            )
        return FusedLayerNorm(hidden_size, elementwise_affine=elementwise_affine, eps=eps)
    if layernorm_backend == "torch":
        return nn.LayerNorm(hidden_size, elementwise_affine=elementwise_affine, eps=eps)
    raise ValueError("layernorm_backend must be one of 'apex' or 'torch'.")


def get_attention(
    hidden_size: int,
    num_heads: int,
    attention_backend: Literal["transformer_engine", "timm", "natten2d"],
    attn_drop_rate: float = 0.0,
    proj_drop_rate: float = 0.0,
    **attn_kwargs: Any,
) -> Module:
    """Construct a pre-defined attention module for DiT.

    Parameters
    ----------
    hidden_size: int
        The embedding dimension.
    num_heads: int
        Number of attention heads.
    attention_backend: str
        One of {"timm", "transformer_engine", "natten2d"} to select between pre-defined attention modules.
    attn_drop_rate: float
        The dropout rate for the attention operation.
    proj_drop_rate: float
        The dropout rate for the projection operation.
    **attn_kwargs: Any
        Additional keyword arguments for the attention module.

    Returns
    -------
    Module
        A module whose forward accepts (B, L, D) and returns (B, L, D).
    """
    if attention_backend == "timm":
        return TimmSelfAttention(hidden_size, num_heads, attn_drop_rate=attn_drop_rate, proj_drop_rate=proj_drop_rate, **attn_kwargs)
    if attention_backend == "transformer_engine":
        return TESelfAttention(hidden_size, num_heads, attn_drop_rate=attn_drop_rate, proj_drop_rate=proj_drop_rate, **attn_kwargs)
    if attention_backend == "natten2d":
        return Natten2DSelfAttention(hidden_size, num_heads, attn_drop_rate=attn_drop_rate, proj_drop_rate=proj_drop_rate, **attn_kwargs)
    raise ValueError("attention_backend must be one of 'timm', 'transformer_engine', 'natten2d' if using pre-defined attention modules.")


class AttentionModuleBase(Module, ABC):
    """Abstract base class for attention modules used in DiTBlock

    Implementations must define a forward method that accepts a single tensor of shape
    (batch, sequence_length, hidden_size) and returns a tensor of the same shape.
    Subclasses must implement the forward method, and may add additional input arguments
    as needed.

    Forward
    -------
    x: torch.Tensor
        Input tensor of shape (B, L, D).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, L, D).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class TimmSelfAttention(AttentionModuleBase):
    """Self-attention module that performs self-attention using the timm library implementation.
    Expects an input tensor of shape (B, L, D) and returns a tensor of the same shape. Under the hood,
    timm uses torch.nn.functional.scaled_dot_product_attention for the attention operation.

    Parameters
    ----------
    hidden_size: int
        The embedding dimension.
    num_heads: int
        Number of attention heads.
    attn_drop_rate: float
        The dropout rate for the attention operation.
    proj_drop_rate: float
        The dropout rate for the projection operation.
    **kwargs: Any
        Additional keyword arguments for the timm attention module.

    Forward
    -------
    x: torch.Tensor
        Input tensor of shape (B, L, D).
    attn_mask: Optional[torch.Tensor]
        The attention mask to apply to the input tensor (passed to timm's Attention module).
        If None, no mask will be applied. This is only supported for timm version 1.0.16 and higher.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, L, D).
    """
    def __init__(self, hidden_size: int, num_heads: int, attn_drop_rate: float = 0.0, proj_drop_rate: float = 0.0, **kwargs: Any):
        super().__init__()
        self.attn_op = Attention(dim=hidden_size, num_heads=num_heads, attn_drop=attn_drop_rate, proj_drop=proj_drop_rate, qkv_bias=True, **kwargs)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attn_mask is not None and not timm_v1_0_16:
            raise ValueError("attn_mask in TimmSelfAttention is only supported for timm version 1.0.16 and higher")
        
        if not timm_v1_0_16:
            return self.attn_op(x)
        else:
            return self.attn_op(x, attn_mask=attn_mask)


class TESelfAttention(AttentionModuleBase):
    """Self-attention module that performs self-attention using the transformer_engine library implementation.
    Expects an input tensor of shape (B, L, D) and 
    returns a tensor of the same shape.

    Parameters
    ----------
    hidden_size: int
        The embedding dimension.
    num_heads: int
        Number of attention heads.
    attn_drop_rate: float
        The dropout rate for the attention operation.
    proj_drop_rate: float
        The dropout rate for the projection operation.
    **kwargs: Any
        Additional keyword arguments for the transformer_engine attention module.

    Forward
    -------
    x: torch.Tensor
        Input tensor of shape (B, L, D).
    attn_mask: Optional[torch.Tensor]
        The attention mask to apply to the input tensor (passed to transformer_engine's MultiheadAttention module).
        If None, no mask will be applied.
    mask_type: Optional[str]
        The type of mask to apply to the input tensor (passed to transformer_engine's MultiheadAttention module).
        If no mask is provided, "no_mask" will be used.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, L, D).
    """
    def __init__(self, hidden_size: int, num_heads: int, attn_drop_rate: float = 0.0, proj_drop_rate: float = 0.0, **kwargs: Any):
        super().__init__()
        if not TE_AVAILABLE:
            raise ImportError(
                "Transformer Engine is not installed. Please install it with `pip install transformer-engine`."
            )

        if proj_drop_rate > 0:
            warnings.warn(
                "Transformer Engine MultiheadAttention does not support projection dropout (proj_drop_rate > 0). "
                "The specified proj_drop_rate will be ignored."
            )
        self.attn_op = MultiheadAttention(hidden_size=hidden_size, num_attention_heads=num_heads, attention_dropout=attn_drop_rate, **kwargs)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, mask_type: Optional[str] = "no_mask") -> torch.Tensor:
        if attn_mask is not None:
            mask_type = "arbitrary"
        return self.attn_op(x, attention_mask=attn_mask, attn_mask_type=mask_type)


class Natten2DSelfAttention(AttentionModuleBase):
    """Self-attention module that performs 2D neighborhood attention using NATTEN.
    Expects an input tensor of shape (B, L, D) and 
    returns a tensor of the same shape (reshapes sequence to 2D internally).

    Parameters
    ----------
    hidden_size: int
        The embedding dimension.
    num_heads: int
        Number of attention heads.
    attn_kernel: int
        The kernel size for the NATTEN neighborhood attention.
    qkv_bias: bool
        Whether to use bias in the qkv projection.
    qk_norm: bool
        Whether to use layer normalization on the query and key.
    proj_drop_rate: float
        The dropout rate for the projection operation.
    norm_layer: Literal["apex", "torch"]
        The layer normalization to use.

    References
    ----------
    - https://arxiv.org/abs/2204.07143
    - https://natten.org/

    Forward
    -------
    x: torch.Tensor
        Input tensor of shape (B, L, D).
    latent_hw: Tuple[int, int]
        The desired height and width of the 2D latent space, used for reshaping before applying attention.
        The total token sequence length must be latent_hw[0] * latent_hw[1].

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, L, D).
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
    ):
        super().__init__()
        if not NATTEN_AVAILABLE:
            raise ImportError(
                "Natten is not installed. Please install it into your environment."
            )

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop_rate = proj_drop_rate
        self.norm_layer = norm_layer
        self.attn_kernel = attn_kernel

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


    def forward(self, x: torch.Tensor, latent_hw: Tuple[int, int]) -> torch.Tensor:

        B, N, C = x.shape
        h, w = latent_hw

        if N != h * w:
            raise ValueError(f"Sequence length must be {h * w} based on latent_hw={latent_hw}, but got {N}")

        # Project to query, key, value and split into heads
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Windowed neighborhood self-attention
        q, k, v = map(
            lambda x: rearrange(x, "b head (h w) c -> b h w head c", h=h),
            [q, k, v],
        )
        if isinstance(q, ShardTensor):
            # Use automatic halo padding for sharded tensors
            x = partial_na2d(q, k, v, kernel_size=self.attn_kernel, base_func=na2d, dilation=1)
        else:
            x = na2d(q, k, v, kernel_size=self.attn_kernel)
        x = self.attn_drop(x)
        x = rearrange(x, "b h w head c -> b (h w) (head c)")

        x = self.proj_drop(self.proj(x))
        return x


class PerSampleDropout(nn.Module):
    """Dropout module supporting scalar or per-sample probabilities. Per-sample dropout uses a different 
    dropout probability for each sample in the batch.

    Parameters
    ----------
    inplace: bool
        Whether to perform the dropout in place.


    Forward
    -------
    x: torch.Tensor
        Input tensor of shape (B, L, D).
    p: Optional[float | torch.Tensor]
        The dropout probability for the intermediate dropout module. If None, no dropout will be applied.
        If a scalar, the same dropout probability will be applied to all samples.
        Otherwise, it should be a tensor of shape (Batch,) to apply per-sample dropout to each sample in a batch.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, L, D).
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor, p: Optional[float | torch.Tensor] = None) -> torch.Tensor:

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
    """A Diffusion Transformer (DiT) block with adaptive layer norm zero (adaLN-Zero) conditioning.

    Parameters
    -----------
    hidden_size (int):
        The dimensionality of the input and output.
    num_heads (int):
        The number of attention heads.
    attention_backend (Union[str, Module]):
        Either the name of a pre-defined attention implementation ('timm', 'transformer_engine', or 'natten2d'),
        or a user-provided Module implementing the same (B, L, D)->(B, L, D) interface for self-attention.
        Options:
            - 'timm' uses the self-attention module from timm. For timm version 1.0.16 and higher, passing an attention mask to the forward method is supported.
              Under the hood, timm uses torch.nn.functional.scaled_dot_product_attention. See physicsnemo.experimental.models.dit.layers.TimmSelfAttention for more details.
            - 'transformer_engine' uses the MultiheadAttention module from transformer_engine. This performs the same operation as timm, but uses an efficient fused implementation.
              See physicsnemo.experimental.models.dit.layers.TESelfAttention for more details.
            - 'natten2d' uses an attention module performing 2D neighborhood attention using NATTEN. See physicsnemo.experimental.models.dit.layers.Natten2DSelfAttention for more details.
        The expected interface for the attention module is defined in physicsnemo.experimental.models.dit.layers.AttentionModuleBase.
        Default is 'timm'.
    layernorm_backend (str):
        The layer normalization implementation ('apex' or 'torch'). Default is 'torch'.
    mlp_ratio (float):
        The ratio for the MLP's hidden dimension. Default is 4.0.
    intermediate_dropout (bool):
        Whether to apply intermediate dropout. If True, the PerSampleDropout module will be applied before the attention module; this
        module supports scalar or per-sample dropout depending on the type/shape of the p_dropout tensor passed to the forward method.
        Default is False.
    attn_drop_rate (float):
        The dropout rate for the attention operation. Default is 0.0.
    proj_drop_rate (float):
        The dropout rate for the projection operation. Default is 0.0.
    mlp_drop_rate (float):
        The dropout rate for the MLP operation. Default is 0.0.
    **attn_kwargs (Any):
        Additional keyword arguments for the attention module.

    Note
    -----
    The attention module configured by `attention_backend` is not expected to be cross-compatible in terms of state_dict keys, but the
    layer norm module configured by `layernorm_backend` is expected to be cross-compatible (models trained with `torch` layernorms can be loaded with `apex` layernorms and vice versa).
    
    Forward
    -------
    x (torch.Tensor):
        Input tensor of shape (B, L, D).
    c (torch.Tensor):
        Conditioning tensor of shape (B, D).
    attn_kwargs (Optional[Dict[str, Any]]):
        Additional keyword arguments for the attention module.
    p_dropout (Optional[float | torch.Tensor]):
        The dropout probability for the intermediate dropout module. If None, no dropout will be applied.
        If a scalar, the same dropout probability will be applied to all samples.
        Otherwise, it should be a tensor of shape (B,) to apply per-sample dropout to each sample in a batch.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, L, D).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_backend: Union[Literal["transformer_engine", "timm", "natten2d"], Module] = "timm",
        layernorm_backend: Literal["apex", "torch"] = "torch",
        mlp_ratio: float = 4.0,
        intermediate_dropout: bool = False,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        mlp_drop_rate: float = 0.0,
        **attn_kwargs: Any,
    ):
        super().__init__()

        if isinstance(attention_backend, Module):
            self.attention = attention_backend
        else:
            self.attention = get_attention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                attention_backend=attention_backend,
                attn_drop_rate=attn_drop_rate,
                proj_drop_rate=proj_drop_rate,
                **attn_kwargs,
            )

        self.pre_attention_norm = get_layer_norm(
            hidden_size, layernorm_backend, elementwise_affine=False, eps=1e-6
        )
        self.pre_mlp_norm = get_layer_norm(
            hidden_size, layernorm_backend, elementwise_affine=False, eps=1e-6
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
            drop=0,
        )
        self.adaptive_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.modulation = lambda x, scale, shift: x * (
            1 + scale.unsqueeze(1)
        ) + shift.unsqueeze(1)

    def initialize_weights(self):
        # Zero out the adaptive modulation weights
        nn.init.constant_(self.adaptive_modulation[-1].weight, 0)
        nn.init.constant_(self.adaptive_modulation[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        p_dropout: Optional[float | torch.Tensor] = None,
    ) -> torch.Tensor:

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
            raise ValueError("p_dropout passed to DiTBlock but intermediate_dropout is disabled")
        
        attention_output = self.attention(
            modulated_attn_input,
            **(attn_kwargs or {}),
        )
        x = x + attention_gate.unsqueeze(1) * attention_output

        # Feed-forward block
        modulated_mlp_input = self.modulation(
            self.pre_mlp_norm(x), mlp_scale, mlp_shift
        )
        mlp_output = self.linear(modulated_mlp_input)
        x = x + mlp_gate.unsqueeze(1) * mlp_output

        return x


class ProjLayer(nn.Module):
    """The penultimate layer of the DiT model, which projects the transformer output
    to a final embedding space.

    Parameters
    -----------
    hidden_size (int):
        The dimensionality of the input from the transformer blocks.
    emb_channels (int):
        The number of embedding channels for final projection.
    layernorm_backend (str):
        The layer normalization implementation ('apex' or 'torch'). Defaults to 'apex'.
    
    Forward
    -------
    x (torch.Tensor):
        Input tensor of shape (B, L, D).
    c (torch.Tensor):
        Conditioning tensor of shape (B, D).

    Outputs
    -------
    torch.Tensor: Output tensor of shape (B, L, D).
    """

    def __init__(
        self, hidden_size: int,
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

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaptive_modulation(c).chunk(2, dim=1)
        modulated_output = self.modulation(
            self.proj_layer_norm(x), scale, shift
        )
        projected_output = self.output_projection(modulated_output)
        return projected_output


# -------------------------------------------------------------------------------------
# Tokenization / De-tokenization interfaces
# -------------------------------------------------------------------------------------


class TokenizerModuleBase(Module, ABC):
    """Abstract base class for tokenizers used by DiT. Must implement a forward method and an initialize_weights method.

    Forward
    -------
    x: torch.Tensor
        Input tensor of shape (B, C, *spatial_dims). `spatial_dims` is determined by the input_size/dimensionality.

    Returns
    -------
    torch.Tensor
        Token sequence of shape (B, L, D), where L is the length of the token sequence (number of patches for a standard 2D DiT).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def initialize_weights(self):
        pass


class PatchEmbed2DTokenizer(TokenizerModuleBase):
    """Standard ViT-style tokenizer using `PatchEmbed2D` followed by a learnable positional embedding.

    Produces tokens of shape (B, L, D) from images (B, C, H, W), where L is the number of
    patches and D is `hidden_size`.

    Parameters
    ----------
    input_size: Tuple[int, int]
        The size of the input image.
    patch_size: Tuple[int, int]
        The size of the patch.
    in_channels: int
        The number of input channels.
    hidden_size: int
        The size of the transformer latent space to project to.
    pos_embed: str = "learnable",
        The type of positional embedding to use. Defaults to 'learnable'.
        Options:
            - 'learnable': Uses a learnable positional embedding.
            - Otherwise, uses no positional embedding.
    **tokenizer_kwargs: Any
        Additional keyword arguments for the tokenizer module.

    Forward
    -------
    x: torch.Tensor
        Input tensor of shape (B, C, H, W).

    Returns
    -------
    torch.Tensor
        Token sequence of shape (B, L, D), where L = (H // patch[0]) * (W // patch[1]).
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
            self.pos_embed = 0.

    def initialize_weights(self):
        # Initialize the tokenizer patch embedding projection (a Conv2D layer).
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)
        
        # Initialize the learnable positional embedding with a normal distribution.
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    tokenizer: Literal["patch_embed_2d"] = "patch_embed_2d",
    **tokenizer_kwargs: Any,
) -> TokenizerModuleBase:
    """Construct a tokenizer module.

    Returns a module whose forward accepts (B, C, *spatial_dims) and returns (B, L, D). 
    `spatial_dims` is determined by the input_size/dimensionality.

    Parameters
    ----------
    input_size: Tuple[int]
        The size of the input image. If an integer is provided, the input is assumed to be on a square 2D domain.
        If a tuple is provided, the input is assumed to be on a multi-dimensional domain.
    patch_size: Tuple[int]
        The size of the patch. If an integer is provided, the patch_size is assumed to be a square 2D patch.
        If a tuple is provided, the patch_size is assumed to be a multi-dimensional patch.
    in_channels: int
        The number of input channels.
    hidden_size: int
        The size of the transformer latent space to project to.
    tokenizer: Literal["patch_embed_2d"]
        The tokenizer to use. Defaults to 'patch_embed_2d'. Note tokenizers are dimensionality-specific; 
        your choice of `tokenizer` must match the dimensionality of your input data (i.e., the `input_size` and `patch_size`).
        Options:
            - 'patch_embed_2d': Uses a standard PatchEmbed2D to project the input image to a sequence of tokens.
    **tokenizer_kwargs: Any
        Additional keyword arguments for the tokenizer module.
    
    Returns
    -------
    TokenizerModuleBase
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
    raise ValueError("tokenizer must be 'patch_embed_2d', no other supported tokenizers are available yet.")


class DetokenizerModuleBase(Module, ABC):
    """Abstract base class for detokenizers used by DiT.

    Must implement a forward method and an initialize_weights method.

    Forward
    -------
    x_tokens: torch.Tensor
        Token sequence of shape (B, L, D_in).
    c: torch.Tensor
        Conditioning tensor of shape (B, D).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, C_out, *spatial_dims). `spatial_dims` is determined by the input_size/dimensionality.
    """

    @abstractmethod
    def forward(self, x_tokens: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def initialize_weights(self):
        pass


class ProjReshape2DDetokenizer(DetokenizerModuleBase):
    """Standard DiT-style detokenizer that applies the DiT `ProjLayer` and reshapes the sequence back
    to an image of shape (B, C_out, H, W).

    Parameters
    ----------
    input_size: Tuple[int, int]
        The size of the input image.
    patch_size: Tuple[int, int]
        The size of the patch.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The size of the transformer latent space to project to.
    layernorm_backend: Literal["apex", "torch"]
        The layer normalization implementation ('apex' or 'torch'). Defaults to 'apex'.

    Forward
    -------
    x_tokens: torch.Tensor
        Token sequence of shape (B, L, D_in).
    c: torch.Tensor
        Conditioning tensor of shape (B, D).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, C_out, *spatial_dims). `spatial_dims` is determined by the input_size/dimensionality.
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
        # Zero out the adaptive modulation and output projection weights
        nn.init.constant_(self.proj_layer.adaptive_modulation[-1].weight, 0)
        nn.init.constant_(self.proj_layer.adaptive_modulation[-1].bias, 0)
        nn.init.constant_(self.proj_layer.output_projection.weight, 0)
        nn.init.constant_(self.proj_layer.output_projection.bias, 0)
        

    def forward(self, x_tokens: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
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
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(
            shape=(
                x.shape[0],
                self.out_channels,
                self.h_patches * self.patch_size[0],
                self.w_patches * self.patch_size[1],
            )
        )
        return x



def get_detokenizer(
    input_size: Union[int, Tuple[int]],
    patch_size: Union[int, Tuple[int]],
    out_channels: int,
    hidden_size: int,
    detokenizer: Literal["proj_reshape_2d"] = "proj_reshape_2d",
    **detokenizer_kwargs: Any,
) -> DetokenizerModuleBase:
    """Construct a detokenizer module.

    Returns a module whose forward accepts (B, L, D) and (B, D) and returns (B, C_out, *spatial_dims). 
    `spatial_dims` is determined by the input_size/dimensionality.

    Parameters
    ----------
    input_size: Union[int, Tuple[int]]
        The size of the input image. If an integer is provided, the input is assumed to be on a square 2D domain.
        If a tuple is provided, the input is assumed to be on a multi-dimensional domain.
    patch_size: Union[int, Tuple[int]]
        The size of the patch. If an integer is provided, the patch_size is assumed to be a square 2D patch.
        If a tuple is provided, the patch_size is assumed to be a multi-dimensional patch.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The size of the transformer latent space to project to.
    detokenizer: Literal["proj_reshape_2d"]
        The detokenizer to use. Defaults to 'proj_reshape_2d'. Note detokenizers are dimensionality-specific; 
        your choice of `detokenizer` must match the dimensionality of your input data (i.e., the `input_size` and `patch_size`).
        Options:
            - 'proj_reshape_2d': Uses a standard DiT `ProjLayer` and reshapes the sequence back to an image.
    **detokenizer_kwargs: Any
        Additional keyword arguments for the detokenizer module.
    """
    if detokenizer == "proj_reshape_2d":
        return ProjReshape2DDetokenizer(
            input_size=input_size,
            patch_size=patch_size,
            out_channels=out_channels,
            hidden_size=hidden_size,
            **detokenizer_kwargs,
        )
    raise ValueError("detokenizer must be 'proj_reshape_2d', no other supported detokenizers are available yet.")
