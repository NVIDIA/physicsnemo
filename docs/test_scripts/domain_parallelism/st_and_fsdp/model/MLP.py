import torch
from torch import nn

class MLP(nn.Module):
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