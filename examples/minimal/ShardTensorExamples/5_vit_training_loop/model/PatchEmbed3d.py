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
import torch.nn as nn
from einops import rearrange

class PatchEmbedding3d(nn.Module):
    """Single patch embedding layer that tokenizes and embeds input 3D images."""
    
    def __init__(self, img_size: tuple[int], patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        for i in img_size:
            assert i % patch_size == 0, f"Image size {i} must be divisible by patch size {patch_size}"
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)
        
        # Single convolution that acts as both tokenizer and linear embedding
        self.conv = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embeddings.
        
        Args:
            x: Input tensor of shape (B, C, H, W, D)
            
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        x = self.conv(x)
        # Rearrange to apply LayerNorm correctly: BCHWD -> B(HWD)C
        x = rearrange(x, 'b c h w d -> b (h w d) c')
        x = self.norm(x)
        # Keep in BHWC format for efficient downstream processing
        x = nn.functional.relu(x)
        
        return x