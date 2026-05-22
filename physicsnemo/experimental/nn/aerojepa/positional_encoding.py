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

"""Fourier positional encoding for INR-style coordinate queries.

Provides :class:`FourierPositionalEncoding`, a deterministic log-frequency
sinusoidal encoding used by the AeroJEPA decoder to lift continuous query
coordinates into a high-dimensional feature space before they are fed into
the implicit field decoder. Distinct from
:class:`physicsnemo.nn.FourierEmbedding`, which uses random Gaussian
frequencies on scalar timesteps.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from jaxtyping import Float


class FourierPositionalEncoding(nn.Module):
    r"""Deterministic log-frequency Fourier positional encoding.

    Maps each coordinate vector to a higher-dimensional feature by
    concatenating its raw value (optionally) with ``sin`` and ``cos`` of
    the vector scaled by log-spaced powers of pi. The encoding has no
    learned parameters.

    Parameters
    ----------
    in_dim : int, optional
        Dimension of the input coordinates. Default 3.
    num_bands : int, optional
        Number of frequency bands. Default 10.
    include_input : bool, optional
        Whether to prepend the raw input to the encoded features.
        Default ``True``.

    Shape
    -----
    - Input: ``(*, in_dim)`` where ``*`` is any number of leading
      dimensions (batched or unbatched).
    - Output: ``(*, out_dim)`` with
      ``out_dim = in_dim * include_input + 2 * in_dim * num_bands``.

    Examples
    --------
    >>> import torch
    >>> pe = FourierPositionalEncoding(in_dim=3, num_bands=4)
    >>> pe.out_dim
    27
    >>> y = pe(torch.zeros(5, 3))
    >>> y.shape
    torch.Size([5, 27])
    """

    def __init__(
        self,
        in_dim: int = 3,
        num_bands: int = 10,
        include_input: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_bands = num_bands
        self.include_input = include_input

    @property
    def out_dim(self) -> int:
        r"""Output feature dimension."""
        base = self.in_dim if self.include_input else 0
        return base + 2 * self.in_dim * self.num_bands

    def forward(
        self,
        x: Float[torch.Tensor, "... D_in"],
    ) -> Float[torch.Tensor, "... D_out"]:
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected last dim {self.in_dim}, got {x.shape[-1]}"
            )

        out = [x] if self.include_input else []
        for i in range(self.num_bands):
            freq = 2.0**i * math.pi
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)
