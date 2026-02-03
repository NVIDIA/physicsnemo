# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import torch

from physicsnemo.core.module import Module
from physicsnemo.nn import PositionalEmbedding


class FrequencyEmbedding(Module):
    """Periodic Embedding.

    Useful for inputs defined on the circle [0, 2pi)
    """

    def __init__(self, num_channels):
        super().__init__()
        self.register_buffer(
            "freqs", torch.arange(1, num_channels + 1), persistent=False
        )

    def forward(self, x):
        freqs = self.freqs[None, :, None, None]
        x = x[:, None, :, :]
        x = x * (2 * math.pi * freqs).to(x.dtype)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class CalendarEmbedding(Module):
    """Time embedding assuming 365.25 day years

    Args:
        day_of_year: (n, t)
        second_of_day: (n, t)
    Returns:
        (n, embed_channels * 4, t, x)

    """

    def __init__(self, lon, embed_channels: int, include_legacy_bug: bool = False):
        """
        Args:
            include_legacy_bug: Provided for backwards compatibility
                with existing checkpoints. If True, use the incorrect formula
                for local_time (hour - lon) instead of the correct formula (hour + lon)
        """
        super().__init__()
        self.register_buffer("lon", lon, persistent=False)
        self.embed_channels = embed_channels
        self.embed_second = FrequencyEmbedding(embed_channels)
        self.embed_day = FrequencyEmbedding(embed_channels)
        self.out_channels = embed_channels * 4
        self.include_legacy_bug = include_legacy_bug

    def forward(self, day_of_year, second_of_day):
        if second_of_day.shape != day_of_year.shape:
            raise ValueError()

        if self.include_legacy_bug:
            local_time = (second_of_day.unsqueeze(2) - self.lon * 86400 // 360) % 86400
        else:
            local_time = (second_of_day.unsqueeze(2) + self.lon * 86400 // 360) % 86400

        a = self.embed_second(local_time / 86400)
        doy = day_of_year.unsqueeze(2)
        b = self.embed_day((doy / 365.25) % 1)
        a, b = torch.broadcast_tensors(a, b)
        return torch.concat([a, b], dim=1)  # (n c x)


class EmbedNoiseLabels(Module):
    """Embedding layer for noise levels and class labels."""

    def __init__(
        self,
        emb_channels,
        label_dim,
        noise_channels,
        label_dropout=None,
        legacy_label_bias: bool = False,
    ):
        super().__init__()
        self.label_dropout = label_dropout
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)

        # legacy_label_bias: for loading old checkpoints that had Linear(0, noise_channels)
        # which contributed a trained bias even with label_dim=0
        self.map_label = None
        if label_dim != 0 or legacy_label_bias:
            self.map_label = torch.nn.Linear(label_dim, noise_channels)

        self.map_layer0 = torch.nn.Linear(
            in_features=noise_channels, out_features=emb_channels
        )
        self.map_layer1 = torch.nn.Linear(
            in_features=emb_channels, out_features=emb_channels
        )

    def forward(self, noise_labels, class_labels):
        emb = self.map_noise(noise_labels)
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        )  # swap sin/cos

        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (
                    torch.rand([noise_labels.shape[0], 1], device=tmp.device)
                    >= self.label_dropout
                ).to(tmp.dtype)
            emb = emb + self.map_label(tmp * math.sqrt(self.map_label.in_features))

        emb = torch.nn.functional.silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)  # No SiLU - consumers (AdaLN) add SiLU before modulation linear
        return emb
