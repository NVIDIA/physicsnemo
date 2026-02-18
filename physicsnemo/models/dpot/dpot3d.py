# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

Tensor = torch.Tensor

_ACTIVATIONS = {
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "silu": nn.SiLU(),
}


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation '{name}'. Available: {list(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[key]


class AFNO3DLayer(nn.Module):
    r"""Adaptive Fourier Neural Operator 3D spectral mixing layer.

    Applies block-diagonal linear transforms in the 3D Fourier domain.

    Parameters
    ----------
    width : int
        Channel dimension (divisible by ``num_blocks``).
    num_blocks : int
        Number of block partitions for per-frequency weight matrices.
    modes : int
        Number of spatial (x & y) low-frequency modes to keep.
    temporal_modes : int
        Number of low-frequency modes to keep along z (temporal/depth) dimension in rFFT domain.
    hidden_size_factor : int
        Expansion factor for intermediate spectral channels.
    activation : str
        Activation function name.
    channel_first : bool
        If True expects (B,C,X,Y,Z), else (B,X,Y,Z,C).
    sparsity_threshold : float
        Lambda for optional soft-shrink (kept for experimentation, disabled by default).

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, X, Y, Z)` if ``channel_first`` else
        :math:`(B, X, Y, Z, C)`.

    Returns
    -------
    torch.Tensor
        Mixed tensor with the same layout and shape as the input.
    """

    def __init__(
        self,
        width: int = 32,
        num_blocks: int = 8,
        modes: int = 32,
        temporal_modes: int = 8,
        hidden_size_factor: int = 1,
        activation: str = "gelu",
        channel_first: bool = True,
        sparsity_threshold: float = 0.01,
    ) -> None:
        super().__init__()
        if width % num_blocks != 0:
            raise ValueError("width must be divisible by num_blocks")
        self.width = width
        self.num_blocks = num_blocks
        self.block_size = width // num_blocks
        self.modes = modes
        self.temporal_modes = temporal_modes
        self.hidden_size_factor = hidden_size_factor
        self.channel_first = channel_first
        self.sparsity_threshold = sparsity_threshold
        self.act = get_activation(activation)

        scale = 1.0 / (self.block_size * self.block_size * hidden_size_factor)
        self.w1 = nn.Parameter(
            scale
            * torch.rand(
                2, num_blocks, self.block_size, self.block_size * hidden_size_factor
            )
        )
        self.b1 = nn.Parameter(
            scale * torch.rand(2, num_blocks, self.block_size * hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            scale
            * torch.rand(
                2, num_blocks, self.block_size * hidden_size_factor, self.block_size
            )
        )
        self.b2 = nn.Parameter(scale * torch.rand(2, num_blocks, self.block_size))

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        if self.channel_first:
            b, c, hx, hy, hz = x.shape
            x = x.permute(0, 2, 3, 4, 1)  # (B,X,Y,Z,C)
        else:
            b, hx, hy, hz, c = x.shape
        residual = x

        # rFFT over spatial+depth dims (X,Y,Z). Output: (B,X,Y,Z//2+1,C)
        x_ft = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")
        x_ft = x_ft.view(b, hx, hy, x_ft.shape[3], self.num_blocks, self.block_size)

        kx = min(self.modes, hx)
        ky = min(self.modes, hy)
        kz = min(self.temporal_modes, x_ft.shape[3])

        o1_real = torch.zeros(
            b,
            hx,
            hy,
            x_ft.shape[3],
            self.num_blocks,
            self.block_size * self.hidden_size_factor,
            device=x.device,
        )
        o1_imag = torch.zeros_like(o1_real)
        o2_real = torch.zeros(
            b, hx, hy, x_ft.shape[3], self.num_blocks, self.block_size, device=x.device
        )
        o2_imag = torch.zeros_like(o2_real)

        # First complex linear + activation
        sel = (slice(None), slice(0, kx), slice(0, ky), slice(0, kz))
        o1_real[sel] = self.act(
            torch.einsum("...bi,bio->...bo", x_ft[sel].real, self.w1[0])
            - torch.einsum("...bi,bio->...bo", x_ft[sel].imag, self.w1[1])
            + self.b1[0]
        )
        o1_imag[sel] = self.act(
            torch.einsum("...bi,bio->...bo", x_ft[sel].imag, self.w1[0])
            + torch.einsum("...bi,bio->...bo", x_ft[sel].real, self.w1[1])
            + self.b1[1]
        )

        # Second complex linear
        o2_real[sel] = (
            torch.einsum("...bi,bio->...bo", o1_real[sel], self.w2[0])
            - torch.einsum("...bi,bio->...bo", o1_imag[sel], self.w2[1])
            + self.b2[0]
        )
        o2_imag[sel] = (
            torch.einsum("...bi,bio->...bo", o1_imag[sel], self.w2[0])
            + torch.einsum("...bi,bio->...bo", o1_real[sel], self.w2[1])
            + self.b2[1]
        )

        x_mix = torch.view_as_complex(torch.stack([o2_real, o2_imag], dim=-1))
        x_mix = x_mix.view(b, hx, hy, x_mix.shape[3], c)
        x_out = torch.fft.irfftn(x_mix, s=(hx, hy, hz), dim=(1, 2, 3), norm="ortho")
        x_out = x_out + residual
        if self.channel_first:
            x_out = x_out.permute(0, 4, 1, 2, 3)
        return x_out


class ConvMlp3D(nn.Module):
    r"""3D Convolutional MLP.

    Parameters
    ----------
    width : int
        Channel dimension of the input/output.
    mlp_ratio : float
        Hidden expansion ratio for the MLP.
    activation : str
        Activation function name.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, X, Y, Z)`.

    Returns
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C, X, Y, Z)`.
    """

    def __init__(self, width: int, mlp_ratio: float, activation: str) -> None:
        super().__init__()
        hidden = int(width * mlp_ratio)
        self.act = get_activation(activation)
        self.fc1 = nn.Conv3d(width, hidden, kernel_size=1)
        self.fc2 = nn.Conv3d(hidden, width, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block3D(nn.Module):
    r"""3D AFNO Block: spectral mixing + Conv MLP + (optional) double skip.

    Parameters
    ----------
    width : int
        Channel dimension of features.
    num_blocks : int
        Number of AFNO block partitions.
    mlp_ratio : float
        Hidden expansion ratio in the Conv MLP.
    modes : int
        Number of low-frequency spatial modes (x, y).
    temporal_modes : int
        Number of low-frequency modes along the z (depth/time) axis.
    activation : str, optional, default="gelu"
        Activation function name.
    double_skip : bool, optional, default=True
        If ``True``, applies an extra residual connection between sublayers.
    norm_groups : int, optional, default=8
        GroupNorm groups.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, X, Y, Z)`.

    Returns
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C, X, Y, Z)`.
    """

    def __init__(
        self,
        width: int,
        num_blocks: int,
        mlp_ratio: float,
        modes: int,
        temporal_modes: int,
        activation: str = "gelu",
        double_skip: bool = True,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(norm_groups, width)
        self.filter = AFNO3DLayer(
            width=width,
            num_blocks=num_blocks,
            modes=modes,
            temporal_modes=temporal_modes,
            channel_first=True,
            activation=activation,
        )
        self.norm2 = nn.GroupNorm(norm_groups, width)
        self.mlp = ConvMlp3D(width=width, mlp_ratio=mlp_ratio, activation=activation)
        self.double_skip = double_skip

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        r = x
        x = self.norm1(x)
        x = self.filter(x)
        if self.double_skip:
            x = x + r
            r = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + r
        return x


class PatchEmbed3D(nn.Module):
    r"""3D patch embedding (voxel embedding).

    Parameters
    ----------
    inp_shape : int
        Spatial size per dimension (cube assumed).
    patch_size : int
        Patch size per dimension.
    in_chans : int
        Input channels.
    embed_dim : int
        Intermediate embedding dimension.
    out_dim : int
        Output embedding dimension.
    activation : str
        Activation name.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, X, Y, Z)`.

    Returns
    -------
    torch.Tensor
        Patch-embedded tensor of shape :math:`(B, C_{out}, X', Y', Z')`
        where :math:`(X', Y', Z')` are the downsampled spatial sizes.
    """

    def __init__(
        self,
        inp_shape: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        out_dim: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.inp_shape = (
            (inp_shape, inp_shape, inp_shape)
            if isinstance(inp_shape, int)
            else inp_shape
        )
        self.patch_size = (
            (patch_size, patch_size, patch_size)
            if isinstance(patch_size, int)
            else patch_size
        )
        self.out_size = (
            self.inp_shape[0] // self.patch_size[0],
            self.inp_shape[1] // self.patch_size[1],
            self.inp_shape[2] // self.patch_size[2],
        )
        self.act = get_activation(activation)
        self.proj = nn.Sequential(
            nn.Conv3d(
                in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
            ),
            self.act,
            nn.Conv3d(embed_dim, out_dim, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        b, c, hx, hy, hz = x.shape
        if (hx, hy, hz) != self.inp_shape:
            raise ValueError(
                f"Input size ({hx}*{hy}*{hz}) does not match model ({self.inp_shape})."
            )
        return self.proj(x)


class TimeAggregator(nn.Module):
    r"""Temporal aggregator.

    Parameters
    ----------
    in_channels : int
        Number of spatial feature channels.
    in_timesteps : int
        Number of timesteps to aggregate over.
    embed_dim : int
        Target embedding dimension after aggregation.
    mode : Literal["mlp", "exp_mlp"], optional, default="exp_mlp"
        Aggregation strategy across time. Allowed values are ``"mlp"`` and ``"exp_mlp"``.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, X, Y, Z, T, C)`.

    Returns
    -------
    torch.Tensor
        Aggregated tensor of shape :math:`(B, X, Y, Z, C)`.
    """

    def __init__(
        self,
        in_channels: int,
        in_timesteps: int,
        embed_dim: int,
        mode: Literal["mlp", "exp_mlp"] = "exp_mlp",
    ) -> None:
        super().__init__()
        self.mode = mode
        scale = 1.0 / (in_timesteps * embed_dim**0.5)
        self.w = nn.Parameter(scale * torch.randn(in_timesteps, embed_dim, embed_dim))
        if mode == "exp_mlp":
            gamma = torch.linspace(-10, 10, embed_dim)
            self.gamma = nn.Parameter(2 ** gamma.unsqueeze(0))
        elif mode != "mlp":
            raise ValueError(f"Unsupported TimeAggregator mode: {mode}")

    def forward(self, x: Tensor) -> Tensor:  # x: (B,H,W,D,T,C)
        if self.mode == "mlp":
            x = torch.einsum("tij,...ti->...j", self.w, x)
        else:  # exp_mlp
            t = torch.linspace(0, 1, x.shape[-2], device=x.device).unsqueeze(-1)
            t_embed = torch.cos(t @ self.gamma)
            x = torch.einsum("tij,...ti->...j", self.w, x * t_embed)
        return x


@dataclass
class DPOT3DMeta:
    name: str = "DPOTNet3D"
    jit: bool = False
    amp: bool = True
    cuda_graphs: bool = False


class DPOTNet3D(nn.Module):
    r"""3D AFNO-based spatio-temporal predictor.

    Parameters
    ----------
    inp_shape : int or tuple[int, int, int]
        Cubic spatial dimension or per-dimension sizes.
    patch_size : int or tuple[int, int, int]
        Patch size per dimension.
    mixing_type : str
        Currently only ``"afno"``.
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    in_timesteps : int
        Input temporal length.
    out_timesteps : int
        Output temporal length.
    num_blocks : int
        Block partitions for spectral weights.
    embed_dim : int
        Embedding dimension.
    out_layer_dim : int
        Hidden dim in reconstruction head.
    depth : int
        Number of AFNO blocks.
    modes : int
        Spatial Fourier modes kept (x,y).
    temporal_modes : int
        Fourier modes kept along the z (depth) FFT axis.
    mlp_ratio : float
        Conv MLP hidden ratio.
    n_classes : int
        Number of classes for classification head.
    normalize : bool
        Use adaptive instance normalization.
    activation : str
        Activation name.
    time_agg : Literal["mlp", "exp_mlp"], optional, default="exp_mlp"
        Temporal aggregation mode. Allowed values are ``"mlp"`` and ``"exp_mlp"``.

    Forward
    -------
    x : torch.Tensor
        Tensor of shape :math:`(B, X, Y, Z, T, C_{in})`.

    Returns
    -------
    torch.Tensor
        Tensor of shape :math:`(B, X, Y, Z, T_{out}, C_{out})`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.dpot.dpot3d import DPOTNet3D
    >>> x = torch.rand(2, 32, 32, 32, 6, 3)  # (B,X,Y,Z,T,C)
    >>> net = DPOTNet3D(
    ...     inp_shape=(32, 32, 32),
    ...     patch_size=(8, 8, 8),
    ...     in_channels=3,
    ...     out_channels=3,
    ...     in_timesteps=6,
    ...     out_timesteps=2,
    ...     embed_dim=96,
    ...     depth=4,
    ...     num_blocks=4,
    ...     modes=16,
    ...     temporal_modes=6,
    ...     mlp_ratio=1.5,
    ...     normalize=True,
    ... )
    >>> y = net(x)
    >>> tuple(y.shape)
    (2, 32, 32, 32, 2, 3)
    """

    def __init__(
        self,
        inp_shape: int = 224,
        patch_size: int = 16,
        mixing_type: Literal["afno"] = "afno",
        in_channels: int = 1,
        out_channels: int = 3,
        in_timesteps: int = 1,
        out_timesteps: int = 1,
        num_blocks: int = 4,
        embed_dim: int = 768,
        out_layer_dim: int = 32,
        depth: int = 12,
        modes: int = 32,
        temporal_modes: int = 8,
        mlp_ratio: float = 1.0,
        normalize: bool = False,
        activation: str = "gelu",
        time_agg: Literal["mlp", "exp_mlp"] = "exp_mlp",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if mixing_type != "afno":
            raise ValueError("Currently only 'afno' mixing_type is supported.")
        self.meta = DPOT3DMeta()
        self.inp_shape = (
            (inp_shape, inp_shape, inp_shape)
            if isinstance(inp_shape, int)
            else inp_shape
        )
        self.patch_size = (
            (patch_size, patch_size, patch_size)
            if isinstance(patch_size, int)
            else patch_size
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.embed_dim = embed_dim
        self.normalize = normalize
        self.activation = activation

        # Coordinate grid adds 4 channels (x,y,z,t)
        self.patch_embed = PatchEmbed3D(
            inp_shape=inp_shape,
            patch_size=patch_size,
            in_chans=in_channels + 4,
            embed_dim=out_channels * max(self.patch_size) + 4,
            out_dim=embed_dim,
            activation=activation,
        )
        sx, sy, sz = self.patch_embed.out_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, sx, sy, sz))

        self.time_agg_layer = TimeAggregator(
            in_channels=in_channels,
            in_timesteps=in_timesteps,
            embed_dim=embed_dim,
            mode=time_agg,
        )

        if normalize:
            self.scale_mu = nn.Linear(2 * in_channels, embed_dim)
            self.scale_sigma = nn.Linear(2 * in_channels, embed_dim)

        self.blocks = nn.ModuleList(
            [
                Block3D(
                    width=embed_dim,
                    num_blocks=num_blocks,
                    mlp_ratio=mlp_ratio,
                    modes=modes,
                    temporal_modes=temporal_modes,
                    activation=activation,
                    double_skip=False,
                    norm_groups=norm_groups,
                )
                for _ in range(depth)
            ]
        )

        self.act = get_activation(activation)

        self.reconstruct = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=embed_dim,
                out_channels=out_layer_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            self.act,
            nn.Conv3d(out_layer_dim, out_layer_dim, kernel_size=1),
            self.act,
            nn.Conv3d(out_layer_dim, out_channels * out_timesteps, kernel_size=1),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _build_grid(x: Tensor) -> Tensor:
        # x: (B,X,Y,Z,T,C)
        b, xdim, ydim, zdim, tdim, _ = x.shape
        gx = (
            torch.linspace(0, 1, xdim, device=x.device)
            .view(1, xdim, 1, 1, 1, 1)
            .repeat(b, 1, ydim, zdim, tdim, 1)
        )
        gy = (
            torch.linspace(0, 1, ydim, device=x.device)
            .view(1, 1, ydim, 1, 1, 1)
            .repeat(b, xdim, 1, zdim, tdim, 1)
        )
        gz = (
            torch.linspace(0, 1, zdim, device=x.device)
            .view(1, 1, 1, zdim, 1, 1)
            .repeat(b, xdim, ydim, 1, tdim, 1)
        )
        gt = (
            torch.linspace(0, 1, tdim, device=x.device)
            .view(1, 1, 1, 1, tdim, 1)
            .repeat(b, xdim, ydim, zdim, 1, 1)
        )
        return torch.cat([gx, gy, gz, gt], dim=-1)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        b, xx, yy, zz, tt, cc = x.shape
        if tt != self.in_timesteps or cc != self.in_channels:
            raise ValueError(
                f"Input timesteps/channels mismatch: got T={tt},C={cc}; expected T={self.in_timesteps},C={self.in_channels}"
            )

        if self.normalize:
            mu = x.mean(dim=(1, 2, 3, 4), keepdim=True)
            sigma = x.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-6
            x_n = (x - mu) / sigma
        else:
            mu = sigma = None
            x_n = x

        grid = self._build_grid(x_n)
        x_feat = torch.cat([x_n, grid], dim=-1)  # (B,X,Y,Z,T,C+4)
        x_feat = rearrange(x_feat, "b x y z t c -> (b t) c x y z")
        x_feat = self.patch_embed(x_feat) + self.pos_embed  # (B*T,E,Sx,Sy,Sz)
        x_feat = rearrange(x_feat, "(b t) e sx sy sz -> b sx sy sz t e", b=b, t=tt)

        # Temporal aggregation -> (B,Sx,Sy,Sz,E)
        x_feat = self.time_agg_layer(x_feat)
        x_feat = rearrange(x_feat, "b sx sy sz e -> b e sx sy sz")

        if self.normalize:
            scale_mu = (
                self.scale_mu(torch.cat([mu, sigma], dim=-1))
                .squeeze(-2)
                .permute(0, 4, 1, 2, 3)
            )
            scale_sigma = (
                self.scale_sigma(torch.cat([mu, sigma], dim=-1))
                .squeeze(-2)
                .permute(0, 4, 1, 2, 3)
            )
            x_feat = scale_sigma * x_feat + scale_mu

        for blk in self.blocks:
            x_feat = blk(x_feat)

        y = self.reconstruct(x_feat)  # (B,C_out*T_out,X,Y,Z)
        y = y.permute(0, 2, 3, 4, 1)
        y = y.view(b, xx, yy, zz, self.out_timesteps, self.out_channels)

        if self.normalize:
            y = y * sigma + mu
        return y

    def extra_repr(self) -> str:  # noqa: D401
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"in_timesteps={self.in_timesteps}, out_timesteps={self.out_timesteps}, "
            f"embed_dim={self.embed_dim}, depth={len(self.blocks)}"
        )


def resize_pos_embed(pos_embed: Tensor, new_pos_embed: Tensor) -> Tensor:
    if pos_embed.shape == new_pos_embed.shape:
        return pos_embed
    _, _, hx_new, hy_new, hz_new = new_pos_embed.shape
    _, _, hx_old, hy_old, hz_old = pos_embed.shape
    if (hx_old, hy_old, hz_old) == (hx_new, hy_new, hz_new):
        return pos_embed
    pos_grid = F.interpolate(
        pos_embed, size=(hx_new, hy_new, hz_new), mode="trilinear", align_corners=False
    )
    return pos_grid


def checkpoint_filter_fn(state_dict: dict, model: DPOTNet3D) -> dict:
    out = {}
    for k, v in state_dict.items():
        if k == "pos_embed" and v.shape != model.pos_embed.shape:
            v = resize_pos_embed(v, model.pos_embed)
        out[k] = v
    return out
