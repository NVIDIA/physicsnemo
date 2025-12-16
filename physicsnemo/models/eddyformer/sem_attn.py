from typing import Tuple
from torch import Tensor

import torch
import torch.nn as nn

from functools import partial

from ..module import Module
from ._datatype import SEM
from .sem_conv import SEMConv

class SEMAttn(Module):

    proj: nn.ModuleDict
    bias: nn.ParameterDict
    norm: nn.ModuleDict

    out: nn.Linear

    def __init__(self,
                 idim: int,
                 odim: int,
                 mode: Tuple[int],
                 num_heads: int,
                 heads_dim: int,
                 *,
                 conv: partial[SEMConv],
                 bias_init = torch.zeros):
        """
        """
        super().__init__()

        self.proj = nn.ModuleDict()
        self.bias = nn.ParameterDict()
        self.norm = nn.ModuleDict()

        for name in "QKV":
            self.proj[name] = conv(idim, (num_heads, heads_dim))

            if name in ["Q", "K"]:
                for n in range(len(mode)):
                    self.bias[f"{name}{n}"] = nn.Parameter(bias_init((num_heads, heads_dim)))
                    self.norm[f"{name}{n}"] = nn.LayerNorm(heads_dim)

        self.out = nn.Linear(num_heads * heads_dim * len(mode), odim)

    def project(self, ϕ: SEM, name: str) -> Tensor:
        """
        Project the features to attention space.
        """
        xs = []

        for n in range(ϕ.ndim):
            x = self.proj[name].factor(ϕ, n).nodal

            if name in ["Q", "K"]:
                x = x + self.bias[f"{name}{n}"]
                assert x.shape[-1] % 2 == 0

                f, g = torch.split(self.norm[f"{name}{n}"](x), x.shape[-1] // 2, dim=-1)
                k = ϕ.coords[..., None, [n]] * torch.arange(f.shape[-1], device=x.device)

                f, g = torch.cos(k) * f - torch.sin(k) * g, torch.sin(k) * f + torch.cos(k) * g
                x = torch.concatenate([torch.cos(k) + f, torch.sin(k) + g], dim=-1)

            xs.append(x.reshape(ϕ.mode + (-1, ) + x.shape[-2:]))
        return torch.concatenate(xs, dim=-1)

    def __call__(self, ϕ: SEM) -> SEM:
        """
        Self-attention on SEM features.
        """
        q, k, v = (self.project(ϕ, name) for name in "QKV")

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        attn = nn.functional.scaled_dot_product_attention(q, k, v)
        return ϕ.new(self.out(attn.reshape(*ϕ.mode, *ϕ.mesh, -1)))
