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
