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

"""Tests for the AeroJEPA Fourier positional encoding layer."""

import math

import pytest
import torch

from physicsnemo.experimental.nn.aerojepa import FourierPositionalEncoding


def test_constructor_default_attributes():
    """Default constructor uses ``in_dim=3, num_bands=10, include_input=True``."""
    pe = FourierPositionalEncoding()
    assert pe.in_dim == 3
    assert pe.num_bands == 10
    assert pe.include_input is True
    # No learnable parameters.
    assert sum(p.numel() for p in pe.parameters()) == 0


@pytest.mark.parametrize(
    "in_dim,num_bands,include_input,expected",
    [
        (3, 4, True, 3 + 2 * 3 * 4),
        (3, 4, False, 2 * 3 * 4),
        (2, 6, True, 2 + 2 * 2 * 6),
        (1, 1, True, 1 + 2 * 1 * 1),
    ],
)
def test_out_dim_formula(in_dim, num_bands, include_input, expected):
    """``out_dim`` follows ``D * include_input + 2 * D * num_bands``."""
    pe = FourierPositionalEncoding(
        in_dim=in_dim, num_bands=num_bands, include_input=include_input
    )
    assert pe.out_dim == expected


def test_forward_shape_unbatched(device):
    """Forward preserves all leading dims, replacing the last with ``out_dim``."""
    pe = FourierPositionalEncoding(in_dim=3, num_bands=4).to(device)
    x = torch.randn(7, 3, device=device)
    y = pe(x)
    assert y.shape == (7, pe.out_dim)
    assert y.dtype == x.dtype


def test_forward_shape_batched(device):
    """Variadic leading dims survive the encoding."""
    pe = FourierPositionalEncoding(in_dim=2, num_bands=3).to(device)
    x = torch.randn(2, 5, 2, device=device)
    y = pe(x)
    assert y.shape == (2, 5, pe.out_dim)


def test_forward_at_zero(device):
    """At ``x=0``: raw input contributes zeros, ``cos(2^i pi * 0)=1`` blocks alternate."""
    pe = FourierPositionalEncoding(in_dim=3, num_bands=2, include_input=True).to(device)
    y = pe(torch.zeros(1, 3, device=device))
    # Layout: [x (3 zeros), sin0, cos0, sin1, cos1] each (·) is length 3.
    expected = torch.tensor(
        [[0.0, 0.0, 0.0] + [0.0, 0.0, 0.0, 1.0, 1.0, 1.0] * 2],
        device=device,
    )
    assert torch.allclose(y, expected)


def test_forward_frequencies(device):
    """Frequencies are ``2^i * pi`` for ``i`` in ``[0, num_bands)``."""
    pe = FourierPositionalEncoding(in_dim=1, num_bands=3, include_input=False).to(
        device
    )
    x = torch.tensor([[0.25]], device=device)
    y = pe(x)
    expected = torch.tensor(
        [
            [
                math.sin(math.pi * 0.25),
                math.cos(math.pi * 0.25),
                math.sin(2.0 * math.pi * 0.25),
                math.cos(2.0 * math.pi * 0.25),
                math.sin(4.0 * math.pi * 0.25),
                math.cos(4.0 * math.pi * 0.25),
            ]
        ],
        device=device,
    )
    assert torch.allclose(y, expected, atol=1e-6)


def test_forward_wrong_last_dim_raises():
    """Wrong input last dim is rejected with a clear ValueError."""
    pe = FourierPositionalEncoding(in_dim=3)
    with pytest.raises(ValueError, match="Expected last dim 3, got 4"):
        pe(torch.zeros(2, 4))
