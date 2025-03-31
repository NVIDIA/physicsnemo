# ignore_header_test
# ruff: noqa: E402
""""""
"""
Transolver model. This code was modified from, https://github.com/thuml/Transolver

The following license is provided from their source,

MIT License

Copyright (c) 2024 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from timm.layers import trunc_normal_

import physicsnemo  # noqa: F401 for docs

from ..meta import ModelMetaData
from ..module import Module
from .Embedding import timestep_embedding
from .Physics_Attention import Physics_Attention_Structured_Mesh_2D

ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(n_hidden, n_hidden), act())
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        # print(x.shape)
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        H=85,
        W=85,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Structured_Mesh_2D(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            H=H,
            W=W,
        )

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(
        self,
        space_dim=1,
        n_layers=5,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        Time_Input=False,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=False,
        H=85,
        W=85,
    ):
        super().__init__()
        self.__name__ = "Transolver_2D"
        self.H = H
        self.W = W
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.pos = self.get_grid()
            self.preprocess = MLP(
                fun_dim + self.ref * self.ref,
                n_hidden * 2,
                n_hidden,
                n_layers=0,
                res=False,
                act=act,
            )
        else:
            self.preprocess = MLP(
                fun_dim + space_dim,
                n_hidden * 2,
                n_hidden,
                n_layers=0,
                res=False,
                act=act,
            )

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if Time_Input:
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden)
            )

        self.blocks = nn.ModuleList(
            [
                Transolver_block(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    H=H,
                    W=W,
                    last_layer=(_ == n_layers - 1),
                )
                for _ in range(n_layers)
            ]
        )
        self.initialize_weights()
        self.placeholder = nn.Parameter(
            (1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float)
        )

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):
        size_x, size_y = self.H, self.W
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1)  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1)  # B H W 8 8 2

        pos = (
            torch.sqrt(
                torch.sum(
                    (grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :])
                    ** 2,
                    dim=-1,
                )
            )
            .reshape(batchsize, size_x, size_y, self.ref * self.ref)
            .contiguous()
        )
        return pos

    def forward(self, x, fx, T=None):
        if self.unified_pos:
            x = (
                self.pos.repeat(x.shape[0], 1, 1, 1)
                .reshape(x.shape[0], self.H * self.W, self.ref * self.ref)
                .to(x.device)
            )
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)

        return fx


@dataclass
class MetaData(ModelMetaData):
    name: str = "Transolver"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = False
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class Transolver(Module):
    """Transformer-based solver for PDEs.

    Note
    ----
    Transolver is a model specifically designed for structured 2D mesh data.

    Parameters
    ----------
    space_dim : int
        The spatial dimension of the input data.
    n_layers : int
        The number of transformer layers.
    n_hidden : int
        The hidden dimension of the transformer.
    dropout : float
        The dropout rate.
    n_head : int
        The number of attention heads.
    Time_Input : bool
        Whether to include time embeddings.
    act : str
        The activation function.
    mlp_ratio : int
        The ratio of hidden dimension in the MLP.
    fun_dim : int
        The dimension of the function.
    out_dim : int
        The output dimension.
    slice_num : int
        The number of slices in the structured attention.
    ref : int
        The reference dimension.
    unified_pos : bool
        Whether to use unified positional embeddings.
    H : int
        The height of the mesh.
    W : int
        The width of the mesh.
    """

    def __init__(
        self,
        space_dim: int,
        n_layers: int,
        n_hidden: int,
        dropout: float,
        n_head: int,
        Time_Input: bool,
        act: str,
        mlp_ratio: int,
        fun_dim: int,
        out_dim: int,
        slice_num: int,
        ref: int,
        unified_pos: bool,
        H: int,
        W: int,
    ) -> None:
        super().__init__(meta=MetaData())
        self.H = H
        self.W = W
        self.model = Model(
            space_dim=space_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            Time_Input=Time_Input,
            act=act,
            mlp_ratio=mlp_ratio,
            fun_dim=fun_dim,
            out_dim=out_dim,
            slice_num=slice_num,
            ref=ref,
            unified_pos=unified_pos,
            H=H,
            W=W,
        )

    def forward(
        self, x: torch.Tensor, fx: torch.Tensor = None, T: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        fx : torch.Tensor
            The function tensor.
        T : torch.Tensor
            The time tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.

        """
        y = self.model(x, fx, T)
        y = y.reshape(x.shape[0], self.H, self.W, -1)
        return y
