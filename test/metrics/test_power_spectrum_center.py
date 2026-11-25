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

import math

import pytest
import torch

import physicsnemo.metrics.general.power_spectrum as ps


def single_mode(size, wavenumber, axis, device):
    """Square field holding one cosine that varies along the given spatial axis."""
    coords = torch.arange(size, device=device, dtype=torch.float32)
    wave = torch.cos(2 * math.pi * wavenumber * coords / size)
    if axis == -2:
        return wave.view(size, 1).repeat(1, size)
    return wave.view(1, size).repeat(size, 1)


@pytest.mark.parametrize("size", [32, 33])
def test_power_spectrum_treats_both_axes_alike(device, size):
    # The same cosine along the height and along the width has to give the same
    # azimuthally averaged spectrum. The width direction used w / 2 as the index
    # of the zero frequency, which is off by half a cell when w is odd.
    k_h, power_h = ps.power_spectrum(single_mode(size, 5, -2, device)[None, None])
    k_w, power_w = ps.power_spectrum(single_mode(size, 5, -1, device)[None, None])
    assert torch.allclose(k_h, k_w)
    assert torch.allclose(power_h, power_w, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("shape", [(32, 32), (33, 33), (32, 33)])
def test_power_spectrum_is_transpose_invariant(device, shape):
    torch.manual_seed(0)
    x = torch.randn(2, 3, *shape, device=device)
    k, power = ps.power_spectrum(x)
    k_t, power_t = ps.power_spectrum(x.transpose(-2, -1))
    assert torch.allclose(k, k_t)
    assert torch.allclose(power, power_t, rtol=1e-4, atol=1e-6)


def test_power_spectrum_single_mode_along_width_fills_one_bin(device):
    size, wavenumber = 33, 5
    x = single_mode(size, wavenumber, -1, device)[None, None]
    k, power = ps.power_spectrum(x)
    power = power[0, 0]
    # Both Fourier coefficients of a real cosine sit at total wavenumber 5, so
    # the whole power belongs to the one bin that contains k = 5.
    assert int((power > 1e-6 * power.max()).sum()) == 1
    bin_width = k[1] - k[0]
    assert abs(k[power.argmax()] - wavenumber) <= 0.5 * bin_width + 1e-5
