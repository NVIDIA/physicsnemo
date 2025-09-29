# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import math
from dataclasses import dataclass
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.layers import Mlp

from physicsnemo.models.diffusion.song_unet import SongUNetPosEmbd
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module


class AttentionPool(nn.Module):
    r"""Cross-attention pooling from variable-length inputs to fixed-length
    outputs.

    Parameters
    ----------
    num_channels : int
        Number of input/output channels :math:`C`.
    out_length : int
        Desired fixed output length :math:`L_{out}`.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, L_{in}, C)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, L_{out}, C)`.
    """

    def __init__(self, num_channels: int, out_length: int):
        super().__init__()
        self.num_channels = num_channels
        self.out_length = out_length

        # Learned queries (one per output slot)
        self.query_tokens = nn.Parameter(
            torch.randn(out_length, num_channels) / math.sqrt(num_channels)
        )

        self.kv_proj = nn.Linear(num_channels, 2 * num_channels)
        nn.init.xavier_uniform_(self.kv_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, L_in, C = x.shape

        # Validate inputs
        if C != self.num_channels:
            raise ValueError(
                f"x last dim must match num_channels: {C} != {self.num_channels}"
            )

        # Build batch of learned queries
        q = self.query_tokens.unsqueeze(0).expand(
            B, self.out_length, self.num_channels
        )  # (B, L_out, C)

        kv = self.kv_proj(x)  # (B, L_in, 2C)
        k, v = torch.chunk(kv, 2, dim=2)  # (B, L_in, C), (B, L_in, C)
        y = F.scaled_dot_product_attention(
            q.unsqueeze(1),  # (B, 1, L_out, C)
            k.unsqueeze(1),  # (B, 1, L_in, C)
            v.unsqueeze(1),  # (B, 1, L_in, C)
        )  # (B, 1, L_out, C)

        return y.squeeze(1)  # (B, L_out, C)


class GlobalFilterBlock1D(nn.Module):
    r"""
    Global filter layer that applies a learnable global filter to the input
    tensor, followed by a layer norm and a MLP.

    Parameters
    ----------
    num_channels : int
        Number of input/output channels :math:`C`.
    length : int
        Length of the input tensor :math:`L`.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, L, C)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, L, C)`.
    """

    def __init__(self, num_channels: int, length: int):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(length // 2 + 1, num_channels, 2, dtype=torch.float32)
            / num_channels
        )
        self.length = length
        self.num_channels = num_channels
        self.layer_norm = nn.LayerNorm(num_channels)
        self.mlp = Mlp(num_channels, 4 * num_channels, num_channels)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, C)
    ) -> torch.Tensor:
        B, L, C = x.shape

        if L != self.length or C != self.num_channels:
            raise ValueError(
                f"x shape mismatch: expected {(B, self.length, self.num_channels)}, "
                f"but got {x.shape}"
            )

        y = torch.fft.rfft(x, dim=1, norm="ortho")  # (B, L, C)
        weight = torch.view_as_complex(self.complex_weight)  # (L, C)
        y = y * weight
        y = torch.fft.irfft(y, n=self.length, dim=1, norm="ortho")  # (B, L, C)

        y = self.layer_norm(y)
        y = self.mlp(y)

        y += x

        return y


class TimeSignalEncoder(nn.Module):
    r"""
    Encodes a 2D image consisting of multiple individual time signals into a
    fixed-length embedding. Combines a lifting network with a series of
    ``GlobalFilterBlock1D`` layers.

    Parameters
    ----------
    in_channels : int
        Number of channels :math:`C_{in}` in the input image.
    out_channels : int
        Number of channels :math:`C_{out}` in the output image.
    hidden_channels : int
        Number of hidden channels :math:`C_{h}` in the encoder.
    in_length : int
        Length :math:`L_{in}` of the input image.
    out_length : int
        Length :math:`L_{out}` of the output embedding.
    num_encoder_blocks : int, optional, default=4
        Number of encoder blocks.

    Forward
    -------
    y : torch.Tensor
        Two-dimensional image of shape :math:`(B, C_{in}, L_{in}, W)`. Each column
        along ``dim=2`` is assumed to correspond to a an individuel time signal
        of length :math:`L_{in}` (and with :math:`C_{in}` channels).
        The input tensor consists of :math:`W` such signals combined into a
        single two-dimensional image.

    Outputs
    -------
    torch.Tensor
        Two-dimensional image of shape :math:`(B, C_{out}, L_{out}, W)`. Each
        time-signal is embedded into a fixed-length embedding of length
        :math:`L_{out}`. The embedded signals are then recombined into a
        single two-dimensional image.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        in_length: int,
        out_length: int,
        num_encoder_blocks: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.in_length = in_length
        self.out_length = out_length
        self.num_encoder_blocks = num_encoder_blocks

        # Lifting network
        self.lift_network = nn.Sequential(
            nn.Linear(in_channels + 1, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
        )

        # Encoder blocks
        self.encoder = nn.ModuleList(
            [
                GlobalFilterBlock1D(hidden_channels, in_length)
                for _ in range(self.num_encoder_blocks)
            ]
        )

        # Output/decoder blocks
        self.attn_pool = AttentionPool(self.hidden_channels, self.out_length)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            Mlp(hidden_channels, 4 * hidden_channels, out_channels),
        )

    def forward(
        self,
        y: torch.Tensor,  # (B, C_in, T, W)
    ) -> torch.Tensor:
        B, C_in, T, W = y.shape

        # Validate inputs
        if y.shape != (B, self.in_channels, self.in_length, W):
            raise ValueError(
                f"y shape mismatch: expected "
                f"{(B, self.in_channels, self.in_length, W)}, but got {y.shape}"
            )
        # Consider all time signals as batched elements
        y = rearrange(y, "b c t w -> (b w) t c")  # (B * W, T, C_in)

        # Add time steps information
        time_steps = torch.linspace(0, 1, T, device=y.device, dtype=y.dtype)[
            None, :, None
        ].expand(B * W, T, 1)  # (B * W, T, 1)
        y = torch.cat((y, time_steps), dim=2)  # (B * W, T, C_in + 1)

        # Apply lifting network to lift hidden dim
        y = self.lift_network(y)  # (B * W, T, C_hidden)

        # Apply encoder blocks
        for enc in self.encoder:
            y = enc(y)  # (B * W, T, C_hidden)

        # Apply pooling to target length and output head
        y = self.attn_pool(y)  # (B * W, L_out, C_hidden)
        y = self.output_head(y)  # (B * W, L_out, C_out)

        # Reshape back to original shape
        y = rearrange(y, "(b w) l c -> b c l w", b=B, w=W)  # (B, C_out, T, W)

        return y


@dataclass
class DiffusionFWINetMetaData(ModelMetaData):
    """
    Metadata for the DiffusionFWINet model.
    """

    name: str = "DiffusionFWINet"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    # Inference
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class DiffusionFWINet(Module):
    r"""
    DiffusionFWINet is a conditional diffusion model designed to denoise a
    latent state vector :math:`x` that corresponds to a velocity model,
    represented by a 2D image where the channel dimension is the individual
    variables that we are seeking to estimate (e.g. wave velocities :math:`V_P`,
    :math:`V_S`, density :math:`\rho`, etc.). The model is conditioned on
    seismic observations :math:`\mathbf{Y}` (e.g. particle velocities :math:`u_x` and
    :math:`u_z`).

    The conditioning is performed by a time-signal encoder that encodes individual
    signals from the seismic observations into fixed-length embeddings. This
    time signal encoder is based on a temporal Global Filter network. The
    embeddings are then channel-wise concatenated to the latent state vector of
    the diffusion model, and finally passed to a UNet that denoises the latent
    state vector.

    Parameters
    ----------
    x_resolution: List[int]
        Resolution of thge latent state vector :math:`x`. For a 2D velocity model, this
        should be of the form :math:`(H, W)`, where :math:`H` and :math:`W` are
        the depth and width of the velocity model, respectively.
    x_channels: int
        Number of channels :math:`C_{\mathbf{x}}` in the latent state vector
        :math:`\mathbf{x}`.
        This should be equal to the number of variables that we are seeking to
        estimate (e.g. `x_channels=3` for :math:`V_P`, :math:`V_S`, and
        :math:`\rho`).
    y_resolution: List[int]
        Resolution of the seismic observations :math:`\mathbf{Y}`. For a 2D velocity
        model, this should be of the form :math:`(T, W)`, where :math:`T` is the
        number of timesteps in each time signal and :math:`W` is the number of
        receivers.

        *Note:* the number of timesteps is the same for all time
        signals and the number of receivers is the same as the width :math:`W` of
        the velocity model.
    y_channels: int
        Number of channels in the seismic observations :math:`\mathbf{Y}`. This should be
        equal to :math:`S \times V`, where :math:`S` is the number of sources and
        :math:`V` is the number of variables observed (e.g. :math:`V=2` for particle
        velocities :math:`u_x` and :math:`u_z`).
    encoder_hidden_channels: int, optional, default=128
        Number of hidden channels in the time signal encoder.
    N_grid_channels: int, optional, default=20
        Number of learnable positional embedding channels in the UNet.
    model_channels: int, optional, default=128
        Base multiplier for the number of channels in the UNet.
    channel_mult: List[int], optional, default=[1, 2, 2, 2, 2]
        Multipliers for the number of channels at each level in the UNet.
    num_blocks: int, optional, default=4
        Number of blocks at each level in the UNet.

    Forward
    -------
    x: torch.Tensor
        Latent state vector :math:`\mathbf{x}` of shape :math:`(B,
        C_{\mathbf{x}}, H, W)`.
    y: torch.Tensor
        Seismic observations :math:`\mathbf{Y}` of shape :math:`(B,
        C_{\mathbf{Y}}, T, W)`.
    sigma: torch.Tensor
        Diffusion noise level, of shape :math:`(B,)`.

    Outputs
    -------
    torch.Tensor
        Denoised latent state vector :math:`\mathbf{x}` of shape :math:`(B,
        C_{\mathbf{x}}, H, W)`.

    Notes
    -----
    This model uses :class:`physicsnemo.models.diffusion.song_unet.SongUNetPosEmbd` as its
    diffusion UNet. For more details on the diffusion model parameters, refer
    to its documentation.
    """

    def __init__(
        self,
        x_resolution: List[int],
        x_channels: int,
        y_resolution: List[int],
        y_channels: int,
        encoder_hidden_channels: int = 128,
        num_encoder_blocks: int = 4,
        N_grid_channels: int = 20,
        model_channels: int = 128,
        channel_mult: List[int] = [1, 2, 2, 2, 2],
        num_blocks: int = 4,
        **unet_kwargs: Any,
    ):
        super().__init__()
        self.meta = DiffusionFWINetMetaData()
        self.x_resolution = tuple(x_resolution)
        self.x_channels = x_channels
        self.y_resolution = tuple(y_resolution)
        self.y_channels = y_channels
        self._grid_to_receivers_channels_ratio = 4

        # Seismic data encoder
        self.time_signal_encoder = TimeSignalEncoder(
            in_channels=(
                y_channels + self._grid_to_receivers_channels_ratio * N_grid_channels
            ),
            out_channels=encoder_hidden_channels // 2,
            hidden_channels=encoder_hidden_channels,
            in_length=self.y_resolution[0],
            out_length=self.x_resolution[0],
            num_encoder_blocks=num_encoder_blocks,
        )

        # Default settings for attention in the UNet
        self._attn_default_threshold = 16
        _unet_resolutions = [self.x_resolution[0]]
        for _ in channel_mult[1:]:
            _unet_resolutions.append(_unet_resolutions[-1] // 2)
        attn_resolutions = unet_kwargs.get(
            "attn_resolutions",
            [r for r in _unet_resolutions if r <= self._attn_default_threshold],
        )

        # Denoising UNet
        self.unet = SongUNetPosEmbd(
            img_resolution=self.x_resolution,
            in_channels=(
                x_channels + self.time_signal_encoder.out_channels + N_grid_channels
            ),
            out_channels=x_channels,
            label_dim=0,
            augment_dim=0,
            model_channels=model_channels,
            channel_mult=channel_mult,
            attn_resolutions=attn_resolutions,
            N_grid_channels=N_grid_channels,
            gridtype="learnable",
            **unet_kwargs,
        )

        # Grid to receivers transform
        self.grid_to_receivers = AttentionPool(
            num_channels=N_grid_channels,
            out_length=4,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        B, C_x, H, W = x.shape
        _, C_y, T, _ = y.shape

        # Validate inputs
        if y.shape != (B, self.y_channels) + self.y_resolution:
            raise ValueError(
                f"y shape mismatch: expected "
                f"{(B, self.y_channels) + self.y_resolution}, but got {y.shape}"
            )
        if x.shape != (B, self.x_channels) + self.x_resolution:
            raise ValueError(
                f"x shape mismatch: expected "
                f"{(B, self.x_channels) + self.x_resolution}, but got {x.shape}"
            )
        if sigma.shape != (B,):
            raise ValueError(f"t shape mismatch: expected {(B,)}, but got {t.shape}")

        # Embed grid coordinates and concatenate to seismic data
        pos_embd = self.unet.pos_embd  # (N_grid, H, W)
        if x.dtype != pos_embd.dtype:
            pos_embd = pos_embd.to(x.dtype)
        pos_emb = pos_embd.permute(2, 1, 0)  # (W, H, N_grid)
        grid_embed = self.grid_to_receivers(pos_emb)  # (W, Cg, N_grid)
        grid_embed = rearrange(grid_embed, "w g n -> (g n) w")  # (Cg * N_grid, W)
        grid_embed = grid_embed[None, :, None, :].expand(
            B, -1, T, -1
        )  # (B, Cg * N_grid, T, W)
        y = torch.cat((y, grid_embed), dim=1)  # (B, C_y + Cg * N_grid, T, W)

        # Encode seismic data
        y = self.time_signal_encoder(y)  # (B, C_hidden//2, H, W)

        # Concatenate latent state vector and seismic data
        x = torch.cat((x, y), dim=1)  # (B, C_x + C_hidden//2, H, W)

        # Denoise
        x = self.unet(x=x, noise_labels=sigma, class_labels=None)  # (B, C_x, H, W)

        return x
