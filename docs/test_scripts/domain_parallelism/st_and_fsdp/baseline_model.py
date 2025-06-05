import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any
import math
from einops import rearrange


class PatchEmbedding2d(nn.Module):
    """Single patch embedding layer that tokenizes and embeds input 2D images."""
    
    def __init__(self, img_size: Tuple[int], patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        for i in img_size:
            assert i % patch_size == 0, f"Image size {i} must be divisible by patch size {patch_size}"
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        # Single convolution that acts as both tokenizer and linear embedding
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embeddings.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        x = self.conv(x)
        # Rearrange to apply LayerNorm correctly: BCHW -> B(HW)C
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        # Keep in BHWC format for efficient downstream processing
        x = F.relu(x)
        
        return x

class PatchEmbedding3d(nn.Module):
    """Single patch embedding layer that tokenizes and embeds input 3D images."""
    
    def __init__(self, img_size: Tuple[int], patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768) -> None:
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
        x = F.relu(x)
        
        return x
    
    
class MultiHeadAttention(nn.Module):
    """Standard multi-head attention using PyTorch's scaled_dot_product_attention."""
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape (B, N, C)
            
        Returns:
            Attention output of shape (B, N, C)
        """
        B, N, C = x.shape
        # Project to Q, K, V and reshape for multi-head attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, num_heads, N, head_dim
        
        # Use PyTorch's optimized scaled dot product attention
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=False
        )
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer."""
    
    def __init__(self, 
                 in_features: int, 
                 hidden_features: int, 
                 out_features: int) -> None:
        
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # Two-layer MLP with activation
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP transformation.
        
        Args:
            x: Input tensor of shape (B, N, C)
            
        Returns:
            Transformed tensor of shape (B, N, out_features)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


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
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
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



class HybridViT(nn.Module):
    """
    Hybrid Vision Transformer with conv patch embedding and multiple transformer layers.
    
    Args:
        img_size: Input image size
        patch_size: Size of patches for tokenization
        in_channels: Number of input channels
        num_classes: Number of classes for classification
        embed_dim: Embedding dimension (same for all layers)
        num_heads: Number of attention heads for each stage
        depth: Number of transformer layers
        mlp_ratio: MLP ratios for each layer
        qkv_bias: Whether to use bias in QKV projections
    """
    
    def __init__(self, img_size: int = [256, 256], patch_size: int = 8, in_channels: int = 3, 
                 num_classes: int = 1000, embed_dim: int = 768,
                 num_heads: int = 6,
                 depth: int = 16, 
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True) -> None:
        super().__init__()
        
        if len(img_size) == 2:
            self.patch_embed = PatchEmbedding2d(
                img_size=img_size, 
                patch_size=patch_size, 
                in_channels=in_channels, 
                embed_dim=embed_dim
            )
        elif len(img_size) == 3:
            self.patch_embed = PatchEmbedding3d(
                img_size=img_size, 
                patch_size=patch_size, 
                in_channels=in_channels, 
                embed_dim=embed_dim
            )
        
        # Positional embeddings (for patches + CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        
        # Build transformer stages (all operating on same resolution)
        self.stages = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
            )
            for _ in range(depth)
        ])
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        

        
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features through all stages.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            CLS token features of shape (B, embed_dim)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # B, N, C
        
        # Add positional embeddings
        x = x + self.pos_embed
 
         # Apply transformer stages
        for stage in self.stages:
            x = stage(x)
            
        return x.mean(dim=(1,))  # Return the mean of all tokens
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass for classification.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Classification logits of shape (B, num_classes)
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x


