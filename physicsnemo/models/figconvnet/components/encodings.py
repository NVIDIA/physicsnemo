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

r"""Positional encoding components for FIGConvNet.

This module provides positional encoding implementations used to encode
spatial coordinates into high-dimensional feature representations.

The main classes are:

- :class:`SinusoidalEncoding`: Sinusoidal positional encoding using Fourier features
"""

# ruff: noqa: S101
import numpy as np
import torch
import torch.nn as nn


class SinusoidalEncoding(nn.Module):
    r"""Sinusoidal positional encoding using Fourier features.

    This module transforms input coordinates into a higher-dimensional space
    using sine and cosine functions at multiple frequencies. This encoding
    helps neural networks learn high-frequency patterns from low-dimensional
    spatial inputs.

    The encoding for a scalar input :math:`x` is computed as:

    .. math::

        \text{enc}(x) = [\cos(2\pi f_1 x), \sin(2\pi f_1 x), ...,
                         \cos(2\pi f_n x), \sin(2\pi f_n x)]

    where :math:`f_i = 2^i / \text{data\_range}` are the frequencies.

    Parameters
    ----------
    num_channels : int
        Total number of output channels. Must be even since each frequency
        contributes both sine and cosine components.
    data_range : float, optional, default=2.0
        Range of input data, used to normalize frequencies.

    Attributes
    ----------
    num_channels : int
        Number of output channels.
    data_range : float
        Data range for frequency normalization.

    Forward
    -------
    x : torch.Tensor
        Input tensor of any shape. The encoding is applied to the last
        dimension and expands it by ``num_channels``.

    Outputs
    -------
    torch.Tensor
        Encoded tensor with shape ``(*x.shape, num_channels)`` flattened
        over the last two dimensions.

    Examples
    --------
    >>> import torch
    >>> encoder = SinusoidalEncoding(num_channels=32, data_range=2.0)
    >>> x = torch.randn(4, 100, 3)  # (batch, points, xyz)
    >>> encoded = encoder(x)
    >>> encoded.shape
    torch.Size([4, 100, 96])

    Note
    ----
    The encoding is applied independently to each element along the last
    dimension of the input. For 3D coordinates, this results in
    ``3 * num_channels`` output features.

    See Also
    --------
    :class:`~physicsnemo.models.figconvnet.components.mlp.MLP`
    """

    def __init__(self, num_channels: int, data_range: float = 2.0):
        super().__init__()
        if num_channels % 2 != 0:
            raise ValueError(
                f"num_channels must be even for sin/cos, got {num_channels}"
            )
        self.num_channels = num_channels
        self.data_range = data_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply sinusoidal encoding to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of arbitrary shape. Encoding is applied to the
            last dimension.

        Returns
        -------
        torch.Tensor
            Encoded tensor with expanded last dimension.
        """
        # Compute frequencies: 2^0, 2^1, ..., 2^(num_channels/2 - 1)
        freqs = 2 ** torch.arange(
            start=0, end=self.num_channels // 2, device=x.device
        ).to(x.dtype)

        # Scale frequencies by 2*pi / data_range
        freqs = (2 * np.pi / self.data_range) * freqs

        # Expand x to broadcast with frequencies
        x = x.unsqueeze(-1)  # (..., D, 1)

        # Broadcast freqs to match x dimensions
        freqs = freqs.reshape((1,) * (len(x.shape) - 1) + freqs.shape)

        # Compute sinusoidal encoding
        x = x * freqs  # (..., D, num_channels // 2)

        # Concatenate cos and sin, then flatten last two dimensions
        x = torch.cat([x.cos(), x.sin()], dim=-1).flatten(start_dim=-2)

        return x
