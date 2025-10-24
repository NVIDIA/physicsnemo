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

"""
Model architectures used in the paper Diffusion models beat gans on image synthesis".
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import silu

from ..diffusion import (
    Conv2d,
    GroupNorm,
    Linear,
    PositionalEmbedding,
    UNetBlock,
)
from ..meta import ModelMetaData
from ..module import Module


@dataclass
class MetaData(ModelMetaData):
    name: str = "TopoDiff"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    # Inference
    onnx: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class TopoDiff(Module):
    """
    Reimplementation of the ADM architecture, a U-Net variant, with optional
    self-attention.

    This model supports conditional and unconditional setups, as well as several
    options for various internal architectural choices such as encoder and decoder
    type, embedding type, etc., making it flexible and adaptable to different tasks
    and configurations.

    Parameters:
    -----------
    img_resolution : int
        The resolution of the input/output image.
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels in the output image.
    label_dim : int, optional
        Number of class labels; 0 indicates an unconditional model. By default 0.
    augment_dim : int, optional
        Dimensionality of augmentation labels; 0 means no augmentation. By default 0.
    model_channels : int, optional
        Base multiplier for the number of channels across the network, by default 192.
    channel_mult : List[int], optional
        Per-resolution multipliers for the number of channels. By default [1,2,3,4].
    channel_mult_emb : int, optional
        Multiplier for the dimensionality of the embedding vector. By default 4.
    num_blocks : int, optional
        Number of residual blocks per resolution. By default 3.
    attn_resolutions : List[int], optional
        Resolutions at which self-attention layers are applied. By default [32, 16, 8].
    dropout : float, optional
        Dropout probability applied to intermediate activations. By default 0.10.
    label_dropout : float, optional
       Dropout probability of class labels for classifier-free guidance. By default 0.0.

    Note:
    -----
    Reference: Dhariwal, P. and Nichol, A., 2021. Diffusion models beat gans on image
    synthesis. Advances in neural information processing systems, 34, pp.8780-8794.

    Note:
    -----
    Equivalent to the original implementation by Dhariwal and Nichol, available at
    https://github.com/openai/guided-diffusion
    """

    def __init__(
        self,
        img_resolution: int,
        in_channels: int,
        out_channels: int,
        label_dim: int = 0,
        augment_dim: int = 0,
        model_channels: int = 128,
        channel_mult: List[int] = [1, 1, 1, 1],
        channel_mult_emb: int = 4,
        num_blocks: int = 2,
        attn_resolutions: List[int] = [16, 8],
        dropout: float = 0.10,
        label_dropout: float = 0.0,
    ):
        super().__init__(meta=MetaData())
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(
            init_mode="kaiming_uniform",
            init_weight=np.sqrt(1 / 3),
            init_bias=np.sqrt(1 / 3),
        )
        init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)
        block_kwargs = dict(
            emb_channels=emb_channels,
            channels_per_head=64,
            dropout=dropout,
            init=init,
            init_zero=init_zero,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = (
            Linear(
                in_features=augment_dim,
                out_features=model_channels,
                bias=False,
                **init_zero,
            )
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(
            in_features=model_channels, out_features=emb_channels, **init
        )
        self.map_layer1 = Linear(
            in_features=emb_channels, out_features=emb_channels, **init
        )
        self.map_label = (
            Linear(
                in_features=label_dim,
                out_features=emb_channels,
                bias=False,
                init_mode="kaiming_normal",
                init_weight=np.sqrt(label_dim),
            )
            if label_dim
            else None
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(
            in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
        )

    def forward(self, x, cons, timesteps):
        # Mapping.
        emb = self.map_noise(timesteps)

        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        emb = silu(emb)

        x = torch.cat([x, cons], dim=1)
        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x


class UNetEncoder(Module):
    """
    U-Net encoder for TopoDiff.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int = 128,
        num_res_blocks: int = 4,
        channel_mult: tuple = (1, 1, 1, 1),
        channel_mult_emb: int = 4,
        attention_resolutions: tuple = (16, 8),
        dropout=0,
        output_prob=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.output_prob = output_prob

        ch = int(model_channels * channel_mult[0])
        self.conv = Conv2d(in_channels=in_channels, out_channels=ch, kernel=3)

        emb_channels = model_channels * channel_mult_emb
        self.time_embed = nn.Sequential(
            Linear(in_features=model_channels, out_features=emb_channels),
            nn.SiLU(),
            Linear(in_features=emb_channels, out_features=emb_channels),
        )

        ds = 1
        self.encoder = nn.ModuleList()
        for level, mult in enumerate(channel_mult):
            attention = ds in attention_resolutions
            for i in range(num_res_blocks):
                down = i == num_res_blocks - 1 and level != len(channel_mult) - 1

                layer = UNetBlock(
                    in_channels=ch,
                    out_channels=int(mult * model_channels),
                    emb_channels=emb_channels,
                    down=down,
                    attention=attention,
                )

                self.encoder.append(layer)
                ch = int(mult * model_channels)
            ds *= 2

        self.middle = nn.ModuleList(
            [
                UNetBlock(
                    in_channels=ch,
                    out_channels=ch,
                    emb_channels=emb_channels,
                    attention=True,
                ),
                UNetBlock(
                    in_channels=ch,
                    out_channels=ch,
                    emb_channels=emb_channels,
                    attention=False,
                ),
            ]
        )

        self.out = nn.Sequential(
            Linear(in_features=ch, out_features=2048),
            GroupNorm(num_channels=2048),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=2048, out_features=self.out_channels),
        )

        if self.output_prob:
            self.out.append(nn.Sigmoid())

    def forward(self, x, time_steps):
        """
        param x: an [N x C x H x W] Tensor of inputs
        param time_steps: a 1-D batch of timesteps
        return: an [N x K] tensor of products
        """
        emb = self.time_embed(self.map_noise(time_steps))

        h = self.conv(x)

        for m in self.encoder:
            h = m(h, emb)

        for m in self.middle:
            h = m(h, emb)
        return self.out(h.mean(dim=(2, 3)))
