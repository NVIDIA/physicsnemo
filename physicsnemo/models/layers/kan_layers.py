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

import numpy as np
import torch
from torch import nn


class KolmogorovArnoldNetwork(nn.Module):
    """
    Kolmogorov–Arnold Network (KAN) layer using Fourier-based function approximation.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    output_dim : int
        Dimensionality of the output features.
    num_harmonics : int, optional
        Number of Fourier harmonics to use (default: 5).
    add_bias : bool, optional
        Whether to include an additive bias term (default: True).
    """

    def __init__(self, input_dim, output_dim, num_harmonics=5, add_bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_harmonics = num_harmonics
        self.add_bias = add_bias

        # Initialize Fourier coefficients (cosine and sine) with scaling for stability.
        # Shape: [2, output_dim, input_dim, num_harmonics]
        self.fourier_coeffs = nn.Parameter(
            torch.randn(2, output_dim, input_dim, num_harmonics)
            / (np.sqrt(input_dim) * np.sqrt(num_harmonics))
        )

        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the KAN layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        batch_size = x.size(0)
        # Reshape input to (batch_size, input_dim, 1) for harmonic multiplication.
        x = x.view(batch_size, self.input_dim, 1)
        # Create harmonic multipliers (from 1 to num_harmonics).
        k = torch.arange(1, self.num_harmonics + 1, device=x.device).view(
            1, 1, self.num_harmonics
        )
        # Compute cosine and sine components.
        cos_terms = torch.cos(k * x)
        sin_terms = torch.sin(k * x)
        # Perform Fourier expansion using Einstein summation for efficiency.
        y_cos = torch.einsum("bij,oij->bo", cos_terms, self.fourier_coeffs[0])
        y_sin = torch.einsum("bij,oij->bo", sin_terms, self.fourier_coeffs[1])
        y = y_cos + y_sin
        if self.add_bias:
            y = y + self.bias
        return y
