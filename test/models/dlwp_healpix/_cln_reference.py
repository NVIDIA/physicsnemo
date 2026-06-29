# Reference (old) implementation of ConditionalLayerNorm for testing.
# This is a copy of the original code before optimization.

from typing import List

import torch as th

try:
    from apex.normalization import FusedLayerNorm
    _APEX_AVAILABLE = True
except ImportError:
    _APEX_AVAILABLE = False


class ConditionalLayerNormReference(th.nn.Module):
    def __init__(
        self,
        condition_shape: int,
        channel_depth: int,
        mlp_hidden_dims: List[int] = [128, 128],
        activation: th.nn.Module = None,
        eps: float = 1e-5,
        n_faces: int = 12,
        norm_op: str = "torch",
        init_cln_to_zero: bool = False,
        scale_center: float = 0.0,
    ):
        super().__init__()
        self.eps = eps
        self.condition_shape = condition_shape
        self.channel_depth = channel_depth
        self.hidden_dims = mlp_hidden_dims
        self.activation = activation if activation is not None else th.nn.Identity()
        self.gamma_mlp = self._make_mlp(self.condition_shape, self.hidden_dims, self.channel_depth, self.activation)
        self.beta_mlp = self._make_mlp(self.condition_shape, self.hidden_dims, self.channel_depth, self.activation)
        self.n_faces = n_faces
        self.scale_center = scale_center

        if init_cln_to_zero:
            self.gamma_mlp[-1].weight.data.zero_()
            self.beta_mlp[-1].weight.data.zero_()
            self.gamma_mlp[-1].bias.data.zero_()
            self.beta_mlp[-1].bias.data.zero_()

        if norm_op == "torch":
            self.norm = th.nn.LayerNorm(channel_depth, elementwise_affine=False)
        elif norm_op == "apex":
            if not _APEX_AVAILABLE:
                raise ImportError("Apex FusedLayerNorm requested but apex is not available")
            self.norm = FusedLayerNorm(channel_depth, elementwise_affine=False)

    def _make_mlp(self, in_dim: int, hidden_dims: List[int], out_dim: int, activation: th.nn.Module) -> th.nn.Sequential:
        layers = []
        for hdim in hidden_dims:
            layers.append(th.nn.Linear(in_dim, hdim))
            if activation:
                layers.append(activation)
            in_dim = hdim
        layers.append(th.nn.Linear(in_dim, out_dim))
        return th.nn.Sequential(*layers)

    def forward(self, x: th.Tensor, conditions: th.Tensor) -> th.Tensor:
        x = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x)

        gamma = self.scale_center + self.gamma_mlp(conditions)[:, None, None, :]
        beta = self.beta_mlp(conditions)[:, None, None, :]

        gamma = gamma.repeat_interleave(self.n_faces, dim=0)
        beta = beta.repeat_interleave(self.n_faces, dim=0)

        x = gamma * x_norm + beta
        return x.permute(0, 3, 1, 2)
