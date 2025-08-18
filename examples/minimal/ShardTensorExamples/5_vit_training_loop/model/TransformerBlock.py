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
 
import torch
from torch import nn

from . MultiHeadAttention import MultiHeadAttention
from . MLP import MLP

class TransformerBlock(nn.Module):
    """Standard transformer block with multi-head attention and MLP."""
    
    def __init__(self, 
                 dim: int, 
                 num_heads: int, 
                 mlp_ratio: float = 4., 
                 qkv_bias: bool = False,
                 norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, 
                      out_features=dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block with residual connections.
        
        Args:
            x: Input tensor of shape (B, N, C)
            
        Returns:
            Transformed tensor of shape (B, N, C)
        """
        # Attention block with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))
        return x
