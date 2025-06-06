import torch
from torch import nn

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
        x = nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=False
        )
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x
