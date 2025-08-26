# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Any, Literal
import torch
import torch.nn as nn

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

from physicsnemo.models.layers import Mlp


class DiTBlock(nn.Module):
    """
    Warning
    -----------
    This feature is experimental and subject to future API changes.

    A Diffusion Transformer (DiT) block with adaptive layer norm zero (adaLN-Zero) conditioning.

    Parameters
    -----------
    hidden_size (int):
        The dimensionality of the input and output.
    num_heads (int):
        The number of attention heads.
    attention_backbone (str):
        The attention implementation ('timm' or 'transformer_engine').
    layernorm_backbone (str):
        The layer normalization implementation ('apex' or 'torch').
    mlp_ratio (float):
        The ratio for the MLP's hidden dimension.
    **block_kwargs (Any):
        Additional keyword arguments for the attention layer.
    
    Forward
    -------
    x (torch.Tensor):
        Input tensor of shape (Batch, Sequence_Length, Hidden_Size).
    c (torch.Tensor):
        Conditioning tensor of shape (Batch, Hidden_Size).

    Outputs
    -------
    torch.Tensor: Output tensor of shape (Batch, Sequence_Length, Hidden_Size).

    Notes
    -------
    # TODO - Check if there will cause an error restoring in a different environment
    # User trains with apex enabled and uses FusedLayerNorm.
    # User saves the model.
    # User loads the model in a different deployment environment which doesn't have apex.
    # Will torch.nn.LayerNorm restore smoothly and correctly?
    # Similarly for TE vs timm attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_backbone: Literal["transformer_engine", "timm"] = "transformer_engine",
        layernorm_backbone: Literal["apex", "torch"] = "torch",
        mlp_ratio: float = 4.0,
        **block_kwargs: Any,
    ):
        """
        Initializes the DiTBlock.
        """
        super().__init__()
        if layernorm_backbone == "apex" and not APEX_AVAILABLE:
            raise ImportError(
                "Apex is not available. Please install Apex to use DiT with FusedLayerNorm.\
                    Or use 'torch' as layernorm_backbone."
            )
        if attention_backbone == "transformer_engine" and not TE_AVAILABLE:
            raise ImportError(
                "Transformer Engine is not installed. Please install it with `pip install transformer-engine`.\
                    Or use 'timm' as attention_backbone."
            )
        if attention_backbone == "transformer_engine":
            self.attention = MultiheadAttention(
                hidden_size=hidden_size, num_attention_heads=num_heads, **block_kwargs
            )
        elif attention_backbone == "timm":
            self.attention = Attention(
                dim=hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
            )
        else:
            raise ValueError(
                "attention_backbone must be one of 'timm' or 'transformer_engine'."
            )
        if layernorm_backbone == "apex":
            self.pre_attention_norm = FusedLayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
            self.pre_mlp_norm = FusedLayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
        elif layernorm_backbone == "torch":
            self.pre_attention_norm = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
            self.pre_mlp_norm = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
        else:
            raise ValueError(
                "layernorm_backbone must be one of 'apex' or 'torch'."
            )
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

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the DiTBlock.
        """
        (
            attention_shift,
            attention_scale,
            attention_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.adaptive_modulation(c).chunk(6, dim=1)

        # Attention block
        modulated_attention_input = self.modulation(
            self.pre_attention_norm(x), attention_scale, attention_shift
        )
        attention_output = self.attention(modulated_attention_input)
        x = x + attention_gate.unsqueeze(1) * attention_output

        # Feed-forward block
        modulated_mlp_input = self.modulation(
            self.pre_mlp_norm(x), mlp_scale, mlp_shift
        )
        mlp_output = self.linear(modulated_mlp_input)
        x = x + mlp_gate.unsqueeze(1) * mlp_output

        return x


class ProjLayer(nn.Module):
    """
    Warning
    -----------
    This feature is experimental and there may be changes in the future.
    
    The penultimate layer of the DiT model, which projects the transformer output
    to a final embedding space.

    Parameters
    -----------
    hidden_size (int):
        The dimensionality of the input from the transformer blocks.
    emb_channels (int):
        The number of embedding channels for final projection.
    layernorm_backbone (str):
        The layer normalization implementation ('apex' or 'torch'). Defaults to 'apex'.
    
    Forward
    -------
    x (torch.Tensor):
        Input tensor of shape (Batch, Sequence_Length, Hidden_Size).
    c (torch.Tensor):
        Conditioning tensor of shape (Batch, Hidden_Size).

    Outputs
    -------
    torch.Tensor: Output tensor of shape (Batch, Sequence_Length, Embed_Size).
    """

    def __init__(
        self, hidden_size: int,
        emb_channels: int,
        layernorm_backbone: Literal["apex", "torch"] = "torch",
    ):
        """
        Initializes the ProjLayer.
        """
        super().__init__()
        if layernorm_backbone == "apex" and not APEX_AVAILABLE:
            raise ImportError(
                "Apex is not available. Please install Apex to use ProjLayer with FusedLayerNorm.\
                Or use 'torch' as layernorm_backbone."
            )
        if layernorm_backbone == "apex":
            self.proj_layer_norm = FusedLayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
        elif layernorm_backbone == "torch":
            self.proj_layer_norm = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
        else:
            raise ValueError(
                "layernorm_backbone must be one of 'apex' or 'torch'."
            )
        self.output_projection = nn.Linear(hidden_size, emb_channels, bias=True)
        self.adaptive_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.modulation = lambda x, scale, shift: x * (
            1 + scale.unsqueeze(1)
        ) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the ProjLayer.
        """
        shift, scale = self.adaptive_modulation(c).chunk(2, dim=1)
        modulated_output = self.modulation(
            self.proj_layer_norm(x), scale, shift
        )
        projected_output = self.output_projection(modulated_output)
        return projected_output
