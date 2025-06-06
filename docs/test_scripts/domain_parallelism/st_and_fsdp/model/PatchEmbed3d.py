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