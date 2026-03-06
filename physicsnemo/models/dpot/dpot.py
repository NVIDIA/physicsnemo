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

from physicsnemo.core.module import Module

Tensor = torch.Tensor

# ---------------------------------------------------------------------------
# Activation factory
# ---------------------------------------------------------------------------
_ACTIVATIONS = {
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),  # normalized key
    "silu": nn.SiLU(),
}


def get_activation(name: str) -> nn.Module:
    """Return activation module by (case-insensitive) name.

    Parameters
    ----------
    name : str
        Activation name.
    """
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation '{name}'. Available: {list(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[key]


# ---------------------------------------------------------------------------
# AFNO2D Spectral Layer
# ---------------------------------------------------------------------------
class DPOT2DLayer(nn.Module):
    r"""Adaptive Fourier Neural Operator 2D spectral mixing layer.

    Parameters
    ----------
    width : int
        Channel dimension (must be divisible by ``num_blocks``).
    num_blocks : int, optional, default=8
        Number of block-diagonal partitions in frequency mixing weights.
    sparsity_threshold : float, optional, default=0.01
        Lambda for optional soft-shrinkage (currently disabled, kept for ablation).
    modes : int, optional, default=32
        Number of (low-frequency) Fourier modes to keep along each spatial dimension.
    hidden_size_factor : int, optional, default=1
        Expansion factor for intermediate spectral channel dimension.
    channel_first : bool, optional, default=False
        If ``True`` expects input as :math:`(B, C, H, W)` else :math:`(B, H, W, C)`.
    activation : str, optional, default="gelu"
        Activation name.
    """

    def __init__(
        self,
        width: int = 32,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        modes: int = 32,
        hidden_size_factor: int = 1,
        channel_first: bool = False,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if width % num_blocks != 0:
            raise ValueError(
                f"width {width} must be divisible by num_blocks {num_blocks}"
            )

        self.width = width
        self.num_blocks = num_blocks
        self.block_size = width // num_blocks
        self.sparsity_threshold = sparsity_threshold
        self.modes = modes
        self.hidden_size_factor = hidden_size_factor
        self.channel_first = channel_first
        self.act = get_activation(activation)

        # Initialization (ref: AFNO baseline uses small normal / trunc normal). Using scalable uniform here.
        scale = 1.0 / (self.block_size * self.block_size * self.hidden_size_factor)

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
        r"""Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape :math:`(B, C, H, W)` or :math:`(B, H, W, C)`
            depending on ``channel_first``.
        """
        if not torch.compiler.is_compiling():
            if x.ndim != 4:
                raise ValueError(f"Expected 4D tensor, got shape {tuple(x.shape)}")
        if self.channel_first:
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1)  # -> (B, H, W, C)
        else:
            b, h, w, c = x.shape

        residual = x

        # rFFT2 over spatial dims (H, W)
        x_ft = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")  # (B,H,W//2+1,C)
        x_ft = x_ft.view(b, h, x_ft.shape[2], self.num_blocks, self.block_size)

        kept = min(self.modes, h, w // 2 + 1)

        o1_real = torch.zeros(
            b,
            h,
            x_ft.shape[2],
            self.num_blocks,
            self.block_size * self.hidden_size_factor,
            device=x.device,
        )
        o1_imag = torch.zeros_like(o1_real)
        o2_real = torch.zeros(
            b, h, x_ft.shape[2], self.num_blocks, self.block_size, device=x.device
        )
        o2_imag = torch.zeros_like(o2_real)

        # First linear (complex) with activation
        o1_real[:, :kept, :kept] = self.act(
            torch.einsum("...bi,bio->...bo", x_ft[:, :kept, :kept].real, self.w1[0])
            - torch.einsum("...bi,bio->...bo", x_ft[:, :kept, :kept].imag, self.w1[1])
            + self.b1[0]
        )
        o1_imag[:, :kept, :kept] = self.act(
            torch.einsum("...bi,bio->...bo", x_ft[:, :kept, :kept].imag, self.w1[0])
            + torch.einsum("...bi,bio->...bo", x_ft[:, :kept, :kept].real, self.w1[1])
            + self.b1[1]
        )

        # Second linear (complex)
        o2_real[:, :kept, :kept] = (
            torch.einsum("...bi,bio->...bo", o1_real[:, :kept, :kept], self.w2[0])
            - torch.einsum("...bi,bio->...bo", o1_imag[:, :kept, :kept], self.w2[1])
            + self.b2[0]
        )
        o2_imag[:, :kept, :kept] = (
            torch.einsum("...bi,bio->...bo", o1_imag[:, :kept, :kept], self.w2[0])
            + torch.einsum("...bi,bio->...bo", o1_real[:, :kept, :kept], self.w2[1])
            + self.b2[1]
        )

        x_mix = torch.view_as_complex(torch.stack([o2_real, o2_imag], dim=-1))
        x_mix = x_mix.view(b, h, x_mix.shape[2], c)
        x_out = torch.fft.irfft2(x_mix, s=(h, w), dim=(1, 2), norm="ortho")

        x_out = x_out + residual
        if self.channel_first:
            x_out = x_out.permute(0, 3, 1, 2)
        return x_out


# ---------------------------------------------------------------------------
# MLP (Conv style inside block)
# ---------------------------------------------------------------------------
class ConvMlp(nn.Module):
    r"""1x1 Convolutional MLP used inside each mixing Block."""

    def __init__(
        self, width: int, mlp_ratio: float = 1.0, activation: str = "gelu"
    ) -> None:
        super().__init__()
        hidden = int(width * mlp_ratio)
        self.act = get_activation(activation)
        self.fc1 = nn.Conv2d(width, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, width, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------
class Block(nn.Module):
    r"""AFNO Block: Spectral mixing + (conv) MLP with residual connections.

    Parameters
    ----------
    width : int
        Channel dimension.
    num_blocks : int
        Block diagonal partitions for spectral layer.
    mlp_ratio : float
        Hidden ratio for MLP.
    modes : int
        Number of spectral modes kept.
    activation : str, optional, default="gelu"
        Activation name.
    double_skip : bool, optional, default=True
        If ``True`` apply two residual merges (as in reference AFNO design).
    """

    def __init__(
        self,
        width: int,
        num_blocks: int,
        mlp_ratio: float,
        modes: int,
        activation: str = "gelu",
        double_skip: bool = True,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(norm_groups, width)
        self.filter = DPOT2DLayer(
            width=width,
            num_blocks=num_blocks,
            modes=modes,
            channel_first=True,
            activation=activation,
        )
        self.norm2 = nn.GroupNorm(norm_groups, width)
        self.mlp = ConvMlp(width=width, mlp_ratio=mlp_ratio, activation=activation)
        self.double_skip = double_skip

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        residual = x
        x = self.norm1(x)
        x = self.filter(x)
        if self.double_skip:
            x = x + residual
            residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    r"""Patch embedding with two-step conv (patchify + projection).

    Parameters
    ----------
    inp_shape : int
        Input image spatial size (assumed square).
    patch_size : int
        Patch size (assumed square).
    in_chans : int
        Number of input channels.
    embed_dim : int
        Intermediate embedding dimension.
    out_dim : int
        Output embedding dimension.
    activation : str, optional, default="gelu"
        Activation name.
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
            (inp_shape, inp_shape) if isinstance(inp_shape, int) else inp_shape
        )
        self.patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.out_size = (
            self.inp_shape[0] // self.patch_size[0],
            self.inp_shape[1] // self.patch_size[1],
        )
        self.act = get_activation(activation)
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
            ),
            self.act,
            nn.Conv2d(embed_dim, out_dim, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        b, c, h, w = x.shape
        if (h, w) != self.inp_shape:
            raise ValueError(
                f"Input image size ({h}*{w}) doesn't match model ({self.inp_shape[0]}*{self.inp_shape[1]})."
            )
        return self.proj(x)


# ---------------------------------------------------------------------------
# Temporal Aggregator
# ---------------------------------------------------------------------------
class TimeAggregator(nn.Module):
    r"""Temporal aggregation over input time dimension using learned per-time MLP.

    Parameters
    ----------
    in_channels : int
        Number of feature channels of the raw input (spatial channels only).
    in_timesteps : int
        Number of input timesteps.
    embed_dim : int
        Target embedding dimension after aggregation.
    mode : Literal["mlp", "exp_mlp"], optional, default="exp_mlp"
        Aggregation strategy. Allowed values are ``"mlp"`` and ``"exp_mlp"``.

    Forward
    -------
    x : Tensor
        Input tensor of shape :math:`(B, H, W, T, C)`, where :math:`T` is the
        time dimension to aggregate and :math:`C` is the feature dimension.

    Returns
    -------
    Tensor
        Aggregated tensor of shape :math:`(B, H, W, C)`.
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
            self.gamma = nn.Parameter(2 ** gamma.unsqueeze(0))  # (1, C)
        elif mode != "mlp":
            raise ValueError("Unsupported TimeAggregator mode: {mode}")

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        if not torch.compiler.is_compiling():
            if x.ndim != 5:
                raise ValueError(f"Expected 5D tensor, got shape {tuple(x.shape)}")
        if self.mode == "mlp":
            x = torch.einsum("tij,...ti->...j", self.w, x)
        else:  # exp_mlp
            t = torch.linspace(0, 1, x.shape[-2], device=x.device).unsqueeze(
                -1
            )  # (T,1)
            t_embed = torch.cos(t @ self.gamma)  # (T, C)
            x = torch.einsum("tij,...ti->...j", self.w, x * t_embed)
        return x


@dataclass
class DPOTMeta:
    name: str = "DPOTNet"
    jit: bool = False
    amp: bool = True
    cuda_graphs: bool = False
    # Extend with additional model-zoo metadata fields as needed.


class DPOTNet(Module):
    r"""DPOTNet with AFNO spectral mixing.

    Parameters
    ----------
    inp_shape : int
        Spatial input size (square images assumed).
    patch_size : int
        Patch size.
    mixing_type : str
        Currently only ``"afno"`` supported (reserved for future mixers).
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    in_timesteps : int
        Number of input timesteps.
    out_timesteps : int
        Number of output timesteps.
    num_blocks : int
        Block diagonal partitions for AFNO spectral weights.
    embed_dim : int
        Embedding dimension.
    out_layer_dim : int
        Intermediate dimension in reconstruction head.
    depth : int
        Number of AFNO blocks.
    modes : int
        Number of Fourier modes kept.
    mlp_ratio : float
        MLP ratio in Block.
    n_classes : int
        Number of classes for auxiliary classification head.
    normalize : bool
        If ``True`` apply adaptive instance normalization based on input stats.
    activation : str
        Activation name.
    time_agg : Literal["mlp", "exp_mlp"], optional, default="exp_mlp"
        Temporal aggregation mode. Allowed values are ``"mlp"`` and ``"exp_mlp"``.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, H, W, T, C)`.

    Outputs
    -------
    torch.Tensor
        Prediction tensor of shape :math:`(B, H, W, T_{out}, C_{out})`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.dpot.dpot import DPOTNet
    >>> x = torch.rand(4, 20, 20, 6, 3)  # (B,H,W,T,C)
    >>> net = DPOTNet(
    ...     inp_shape=(20, 20),
    ...     patch_size=(5, 10),
    ...     in_channels=3,
    ...     out_channels=3,
    ...     in_timesteps=6,
    ...     out_timesteps=1,
    ...     embed_dim=32,
    ...     normalize=True,
    ...     depth=4,
    ...     num_blocks=4,
    ... )
    >>> y = net(x)
    >>> tuple(y.shape)
    (4, 20, 20, 1, 3)
    """

    def __init__(
        self,
        inp_shape: int = 224,
        patch_size: int = 16,
        mixing_type: Literal["afno"] = "afno",
        in_channels: int = 1,
        out_channels: int = 4,
        in_timesteps: int = 1,
        out_timesteps: int = 1,
        num_blocks: int = 4,
        embed_dim: int = 768,
        out_layer_dim: int = 32,
        depth: int = 12,
        modes: int = 32,
        mlp_ratio: float = 1.0,
        normalize: bool = False,
        activation: str = "gelu",
        time_agg: Literal["mlp", "exp_mlp"] = "exp_mlp",
        norm_groups: int = 8,
    ) -> None:
        super().__init__(meta=DPOTMeta())

        if mixing_type != "afno":
            raise ValueError("Currently only 'afno' mixing_type is supported.")

        self.inp_shape = (
            (inp_shape, inp_shape) if isinstance(inp_shape, int) else inp_shape
        )
        self.patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.normalize = normalize
        self.activation = activation
        self.embed_dim = embed_dim

        # Patch embedding expects (in_channels + 3) because of coordinate grid appended.
        self.patch_embed = PatchEmbed(
            inp_shape=inp_shape,
            patch_size=patch_size,
            in_chans=in_channels + 3,
            embed_dim=out_channels * max(self.patch_size) + 3,
            out_dim=embed_dim,
            activation=activation,
        )
        h_patches, w_patches = self.patch_embed.out_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, h_patches, w_patches))

        # Adaptive normalization affine layers (AdaIN style)
        if normalize:
            self.scale_mu = nn.Linear(2 * in_channels, embed_dim)
            self.scale_sigma = nn.Linear(2 * in_channels, embed_dim)

        # Temporal aggregator operates after spatial embedding
        self.time_agg_layer = TimeAggregator(
            in_channels=in_channels,
            in_timesteps=in_timesteps,
            embed_dim=embed_dim,
            mode=time_agg,
        )

        # AFNO Blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    width=embed_dim,
                    num_blocks=num_blocks,
                    mlp_ratio=mlp_ratio,
                    modes=modes,
                    activation=activation,
                    double_skip=False,
                    norm_groups=norm_groups,
                )
                for _ in range(depth)
            ]
        )

        self.act = get_activation(activation)

        self.reconstruct = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=embed_dim,
                out_channels=out_layer_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            self.act,
            nn.Conv2d(out_layer_dim, out_layer_dim, kernel_size=1),
            self.act,
            nn.Conv2d(out_layer_dim, out_channels * out_timesteps, kernel_size=1),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _build_grid(x: Tensor) -> Tensor:
        b, h, w, t, _ = x.shape
        grid_x = (
            torch.linspace(0, 1, h, device=x.device)
            .view(1, h, 1, 1, 1)
            .repeat(b, 1, w, t, 1)
        )
        grid_y = (
            torch.linspace(0, 1, w, device=x.device)
            .view(1, 1, w, 1, 1)
            .repeat(b, h, 1, t, 1)
        )
        grid_t = (
            torch.linspace(0, 1, t, device=x.device)
            .view(1, 1, 1, t, 1)
            .repeat(b, h, w, 1, 1)
        )
        return torch.cat([grid_x, grid_y, grid_t], dim=-1)  # (B,H,W,T,3)

    def forward(self, x: Tensor) -> Tensor:
        b, h, w, t, c = x.shape
        if t != self.in_timesteps or c != self.in_channels:
            raise ValueError(
                f"Input has shape T={t}, C={c}; expected T={self.in_timesteps}, C={self.in_channels}."
            )

        if self.normalize:
            mu = x.mean(dim=(1, 2, 3), keepdim=True)
            sigma = x.std(dim=(1, 2, 3), keepdim=True) + 1e-6
            x_norm = (x - mu) / sigma
        else:
            mu = sigma = None
            x_norm = x

        grid = self._build_grid(x_norm)
        x_feat = torch.cat([x_norm, grid], dim=-1)  # (B,H,W,T,C+3)
        x_feat = rearrange(x_feat, "b h w t c -> (b t) c h w")
        x_feat = self.patch_embed(x_feat) + self.pos_embed  # (B*T, E, H', W')
        x_feat = rearrange(x_feat, "(b t) c hp wp -> b hp wp t c", b=b, t=t)

        # Temporal aggregation -> (B, Hp, Wp, E)
        x_feat = self.time_agg_layer(x_feat)
        x_feat = rearrange(x_feat, "b hp wp c -> b c hp wp")

        if self.normalize:
            scale_mu = (
                self.scale_mu(torch.cat([mu, sigma], dim=-1))
                .squeeze(-2)
                .permute(0, 3, 1, 2)
            )
            scale_sigma = (
                self.scale_sigma(torch.cat([mu, sigma], dim=-1))
                .squeeze(-2)
                .permute(0, 3, 1, 2)
            )
            x_feat = scale_sigma * x_feat + scale_mu

        for blk in self.blocks:
            x_feat = blk(x_feat)

        y = self.reconstruct(x_feat)  # (B, out_channels*out_timesteps, H, W)
        y = y.permute(0, 2, 3, 1)
        y = y.view(b, h, w, self.out_timesteps, self.out_channels)

        if self.normalize:
            y = y * sigma + mu
        return y

    # ---------------------------------------------------------------- repr
    def extra_repr(self) -> str:  # noqa: D401
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"in_timesteps={self.in_timesteps}, out_timesteps={self.out_timesteps}, "
            f"embed_dim={self.embed_dim}, depth={len(self.blocks)}"
        )


# ---------------------------------------------------------------------------
# Utility functions for checkpoint compatibility
# ---------------------------------------------------------------------------


def resize_pos_embed(pos_embed: Tensor, new_pos_embed: Tensor) -> Tensor:
    """Resize positional embedding using bilinear interpolation.

    Parameters
    ----------
    pos_embed : Tensor
        Old positional embedding.
    new_pos_embed : Tensor
        Target positional embedding (shape reference).
    """
    if pos_embed.shape == new_pos_embed.shape:
        return pos_embed
    _, _, h_new, w_new = new_pos_embed.shape
    pos_grid = pos_embed
    _, _, h_old, w_old = pos_grid.shape
    if (h_old, w_old) == (h_new, w_new):
        return pos_grid
    pos_grid = F.interpolate(
        pos_grid, size=(h_new, w_new), mode="bilinear", align_corners=False
    )
    return pos_grid


def checkpoint_filter_fn(state_dict: dict, model: DPOTNet) -> dict:
    """Adapt legacy checkpoints to current parameter shapes."""
    out = {}
    for k, v in state_dict.items():
        if k == "pos_embed" and v.shape != model.pos_embed.shape:
            v = resize_pos_embed(v, model.pos_embed)
        out[k] = v
    return out
