from typing import Tuple, Union, Optional
from torch import Tensor

import torch
import torch.nn as nn

from dataclasses import dataclass
from functools import partial

from ..module import Module
from ..meta import ModelMetaData
from ..layers.mlp_layers import Mlp

from ._datatype import SEM
from .sem_conv import SEMConv
from .sem_attn import SEMAttn

class EddyFormerConfig(Module):

    basis: str
    mesh: Tuple[int]
    mode: Tuple[int]

    # SGS STREAM
    kernel_size: Tuple[int]

    ffn_dim: int
    activation: str

    # LES STREAM
    mode_les: Tuple[int]
    kernel_size_les: Tuple[int]

    num_heads: int
    heads_dim: int

    def __init__(self, basis: str, mesh: Tuple[int], mode: Tuple[int],
                 kernel_size: Tuple[int], ffn_dim: int, activation: str,
                 mode_les: Tuple[int], kernel_size_les: Tuple[int], num_heads: int, heads_dim: int):
        """
        """
        super().__init__()

        self.basis = basis
        self.mesh = mesh
        self.mode = mode

        self.kernel_size = kernel_size
        self.ffn_dim = ffn_dim
        self.activation = activation

        self.mode_les = mode_les
        self.kernel_size_les = kernel_size_les
        self.num_heads = num_heads
        self.heads_dim = heads_dim

    @property
    def ffn(self) -> partial[Mlp]:
        return partial(Mlp,
            hidden_features=self.ffn_dim,
            act_layer=getattr(nn, self.activation),
        )

    @property
    def attn(self) -> partial[SEMAttn]:
        return partial(SEMAttn,
            mode=self.mode_les,
            num_heads=self.num_heads,
            heads_dim=self.heads_dim,
        )

    def conv(self, stream: str) -> partial[SEMConv]:
        return partial(SEMConv,
            kernel_mode=(mode:=self.mode if stream == "sgs" else self.mode_les),
            kernel_size=self.kernel_size if stream == "sgs" else self.kernel_size_les,
            T=tuple(map(SEM.basis(self.basis), mode)),
        )

# Layer

class EddyFormerLayer(Module):

    def __init__(self, hdim: int, cfg: EddyFormerConfig, *, layer_scale: float = 1e-7):
        """
        EddyFormer layer.
        """
        super().__init__()

        self.mode = cfg.mode
        self.mode_les = cfg.mode_les

        self.eps = nn.Parameter(torch.ones(hdim) * layer_scale)
        self.ffn_les, self.ffn_sgs = cfg.ffn(hdim), cfg.ffn(hdim)

        self.sem_conv_sgs = cfg.conv("sgs")(hdim, hdim)
        self.sem_conv_les = cfg.conv("les")(hdim, hdim)
        self.sem_attn = cfg.attn(hdim, hdim, conv=cfg.conv("les"))

    def __call__(self, les: SEM, sgs: SEM) -> Tuple[SEM, SEM]:
        """
        """
        les.nodal = les.nodal + self.sem_attn(les).nodal
        les.nodal = les.nodal + self.ffn_les(self.sem_conv_les(les).nodal)

        sgs.nodal = sgs.nodal + self.eps * les.to(self.mode).nodal
        sgs.nodal = sgs.nodal + self.ffn_sgs(self.sem_conv_sgs(sgs).nodal)

        return les, sgs

# Model

@dataclass
class MetaData(ModelMetaData):
    name: str = "EddyFormer"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp: bool = False
    # Inference
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False

class EddyFormer(Module):

    cfg: EddyFormerConfig

    lift_les: nn.Linear
    lift_sgs: nn.Linear

    layers: nn.ModuleList

    proj_les: Mlp
    proj_sgs: Mlp

    scale: Optional[nn.Parameter]

    def __init__(self,
                 idim: int,
                 odim: int,
                 hdim: int,
                 num_layers: int,
                 *,
                 use_scale: bool = True,
                 cfg: EddyFormerConfig):
        """
        EddyFormer model.
        """
        super().__init__(meta=MetaData())

        self.cfg = cfg
        self.ndim = len(cfg.mesh)

        self.lift_les = nn.Linear(idim + self.ndim, hdim)
        self.lift_sgs = nn.Linear(idim + self.ndim, hdim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = EddyFormerLayer(hdim, cfg)
            self.layers.append(layer)

        self.proj_les = cfg.ffn(hdim, out_features=odim)
        self.proj_sgs = cfg.ffn(hdim, out_features=odim)

        self.scale = nn.Parameter(torch.zeros(odim)) if use_scale else None

    def __call__(self, input: Union[SEM, Tensor], return_sem: bool = False) -> Union[SEM, Tensor]:
        """
        """
        if isinstance(input, Tensor):
            size = 2 * torch.pi * torch.ones(self.ndim, device=input.device)
            ϕ = SEM(self.cfg.basis, size, self.cfg.mesh, self.cfg.mode) \
               .from_grid(input, "lag8") # default interpolation method
        else:
            ϕ = input

        x = ϕ.grid.to(ϕ.nodal)
        for n, mesh in enumerate(ϕ.mesh):
          x = x.unsqueeze(dim:=self.ndim + n)
          x = torch.repeat_interleave(x, mesh, dim)
        x = torch.concatenate(torch.broadcast_tensors(ϕ.nodal, x), dim=-1)

        sgs = ϕ.new(x)
        les = sgs.to(self.cfg.mode_les)

        sgs.nodal = self.lift_sgs(sgs.nodal)
        les.nodal = self.lift_les(les.nodal)

        for layer in self.layers:
            les, sgs = layer(les, sgs)

        sgs.nodal = self.proj_sgs(sgs.nodal)
        les.nodal = self.proj_les(les.nodal)

        scale = self.scale if self.scale is not None else 1.
        out = ϕ.new(les.to(ϕ.mode).nodal + scale * sgs.nodal)

        if not return_sem:
            out = out.eval(input.shape[:-1])
        return out
