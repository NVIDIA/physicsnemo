# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .mlp_layers import Mlp


def fourier_encode(coords: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Vectorized Fourier feature encoding

    Args:
        coords: Tensor containing coordinates, of shape (batch_size, D)
        freqs: Tensor containing frequencies, of shape (F,) (num frequencies)

    Returns:
        Tensor containing Fourier features, of shape (batch_size, D * 2 * F)
    """

    D = coords.shape[-1]
    F = freqs.shape[0]

    freqs = freqs[None, None, :, None]  # reshape to [*, F, 1] for broadcasting

    coords = coords.unsqueeze(-2)  # [*, 1, D]
    scaled = (coords * freqs).reshape(*coords.shape[:-2], D * F)  # [*, D, F]
    features = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)  # [*, D, 2F]

    return features.reshape(*coords.shape[:-2], D * 2 * F)  # [*, D * 2F]


class FourierMLP(nn.Module):
    """
    This is an MLP that will, optionally, fourier encode the input features.

    The encoded features are concatenated to the original inputs, and then
    processed with an MLP.

    Args:
        input_features: The number of input features to the MLP.
        base_layer: The number of neurons in the hidden layer of the MLP.
        fourier_features: Whether to fourier encode the input features.
        num_modes: The number of modes to use for the fourier encoding.
        activation: The activation function to use in the MLP.

    """

    def __init__(
        self,
        input_features: int,
        base_layer: int,
        fourier_features: bool,
        num_modes: int,
        activation: nn.Module | str,
    ):
        super().__init__()
        self.fourier_features = fourier_features

        # self.num_modes = model_parameters.num_modes

        if self.fourier_features:
            input_features_calculated = input_features + input_features * num_modes * 2
            self.register_buffer(
                "freqs", torch.exp(torch.linspace(0, math.pi, num_modes))
            )
        else:
            input_features_calculated = input_features

        self.mlp = Mlp(
            in_features=input_features_calculated,
            hidden_features=[
                base_layer,
                base_layer,
            ],
            out_features=base_layer,
            act_layer=activation,
            drop=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fourier_features:
            x = torch.cat((x, fourier_encode(x, self.freqs)), dim=-1)

        return self.mlp(x)


class FourierLayer(nn.Module):
    """Fourier layer used in the Fourier feature network"""

    def __init__(
        self,
        in_features: int,
        frequencies,
    ) -> None:
        super().__init__()

        # To do: Need more robust way for these params
        if isinstance(frequencies[0], str):
            if "gaussian" in frequencies[0]:
                nr_freq = frequencies[2]
                np_f = (
                    np.random.normal(0, 1, size=(nr_freq, in_features)) * frequencies[1]
                )
            else:
                nr_freq = len(frequencies[1])
                np_f = []
                if "full" in frequencies[0]:
                    np_f_i = np.meshgrid(
                        *[np.array(frequencies[1]) for _ in range(in_features)],
                        indexing="ij",
                    )
                    np_f.append(
                        np.reshape(
                            np.stack(np_f_i, axis=-1),
                            (nr_freq**in_features, in_features),
                        )
                    )
                if "axis" in frequencies[0]:
                    np_f_i = np.zeros((nr_freq, in_features, in_features))
                    for i in range(in_features):
                        np_f_i[:, i, i] = np.reshape(
                            np.array(frequencies[1]), (nr_freq)
                        )
                    np_f.append(
                        np.reshape(np_f_i, (nr_freq * in_features, in_features))
                    )
                if "diagonal" in frequencies[0]:
                    np_f_i = np.reshape(np.array(frequencies[1]), (nr_freq, 1, 1))
                    np_f_i = np.tile(np_f_i, (1, in_features, in_features))
                    np_f_i = np.reshape(np_f_i, (nr_freq * in_features, in_features))
                    np_f.append(np_f_i)
                np_f = np.concatenate(np_f, axis=-2)

        else:
            np_f = frequencies  # [nr_freq, in_features]

        frequencies = torch.tensor(np_f, dtype=torch.get_default_dtype())
        frequencies = frequencies.t().contiguous()
        self.register_buffer("frequencies", frequencies)

    def out_features(self) -> int:
        return int(self.frequencies.size(1) * 2)

    def forward(self, x: Tensor) -> Tensor:
        x_hat = torch.matmul(x, self.frequencies)
        x_sin = torch.sin(2.0 * math.pi * x_hat)
        x_cos = torch.cos(2.0 * math.pi * x_hat)
        x_i = torch.cat([x_sin, x_cos], dim=-1)
        return x_i


class FourierFilter(nn.Module):
    """Fourier filter used in the multiplicative filter network"""

    def __init__(
        self,
        in_features: int,
        layer_size: int,
        nr_layers: int,
        input_scale: float,
    ) -> None:
        super().__init__()

        self.weight_scale = input_scale / math.sqrt(nr_layers + 1)

        self.frequency = nn.Parameter(torch.empty(in_features, layer_size))
        # The shape of phase tensor was supposed to be [1, layer_size], but it has issue
        # with batched tensor in FuncArch.
        # We could just rely on broadcast here.
        self.phase = nn.Parameter(torch.empty(layer_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets parameters"""
        nn.init.xavier_uniform_(self.frequency)
        nn.init.uniform_(self.phase, -math.pi, math.pi)

    def forward(self, x: Tensor) -> Tensor:
        frequency = self.weight_scale * self.frequency

        x_i = torch.sin(torch.matmul(x, 2.0 * math.pi * frequency) + self.phase)
        return x_i


class GaborFilter(nn.Module):
    """Gabor filter used in the multiplicative filter network"""

    def __init__(
        self,
        in_features: int,
        layer_size: int,
        nr_layers: int,
        input_scale: float,
        alpha: float,
        beta: float,
    ) -> None:
        super().__init__()

        self.layer_size = layer_size
        self.alpha = alpha
        self.beta = beta

        self.weight_scale = input_scale / math.sqrt(nr_layers + 1)

        self.frequency = nn.Parameter(torch.empty(in_features, layer_size))
        self.phase = nn.Parameter(torch.empty(layer_size))
        self.mu = nn.Parameter(torch.empty(in_features, layer_size))
        self.gamma = nn.Parameter(torch.empty(layer_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets parameters"""
        nn.init.xavier_uniform_(self.frequency)
        nn.init.uniform_(self.phase, -math.pi, math.pi)
        nn.init.uniform_(self.mu, -1.0, 1.0)
        with torch.no_grad():
            self.gamma.copy_(
                torch.from_numpy(
                    np.random.gamma(self.alpha, 1.0 / self.beta, (self.layer_size)),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        frequency = self.weight_scale * (self.frequency * self.gamma.sqrt())

        x_c = x.unsqueeze(-1)
        x_c = x_c - self.mu
        # The norm dim changed from 1 to -2 to be compatible with BatchedTensor
        x_c = torch.square(x_c.norm(p=2, dim=-2))
        x_c = torch.exp(-0.5 * x_c * self.gamma)
        x_i = x_c * torch.sin(torch.matmul(x, 2.0 * math.pi * frequency) + self.phase)
        return x_i
