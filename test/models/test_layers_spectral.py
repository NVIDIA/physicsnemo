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

import pytest
import torch

from physicsnemo.models.layers.spectral_layers import (
    calc_latent_derivatives,
    fourier_derivatives,
)


@pytest.fixture
def simple_tensor():
    return torch.randn((1, 1, 4, 4))


@pytest.fixture
def domain_length():
    return [2.0, 2.0]


@pytest.fixture
def tensor_3d():
    return torch.randn((1, 2, 8))


@pytest.fixture
def tensor_4d():
    return torch.randn((1, 2, 8, 8))


@pytest.fixture
def tensor_5d():
    return torch.randn((1, 2, 8, 8, 8))


def test_basic_functionality(simple_tensor, domain_length):
    wx, wxx = fourier_derivatives(simple_tensor, domain_length)
    assert wx is not None
    assert wxx is not None


def test_output_shapes(simple_tensor, domain_length):
    wx, wxx = fourier_derivatives(simple_tensor, domain_length)
    assert wx.shape == (1, 2, 4, 4)  # 2 because of 2 dimensions in domain length
    assert wxx.shape == (1, 2, 4, 4)


def test_mismatched_dimensions(simple_tensor):
    wrong_domain_length = [2.0]
    with pytest.raises(ValueError, match=r"input shape doesn't match domain dims"):
        fourier_derivatives(simple_tensor, wrong_domain_length)


def test_basic_functionality_3d(tensor_3d, domain_length):
    dx_list, ddx_list = calc_latent_derivatives(tensor_3d, domain_length)
    assert all(d is not None for d in dx_list)
    assert all(dd is not None for dd in ddx_list)


def test_basic_functionality_4d(tensor_4d, domain_length):
    dx_list, ddx_list = calc_latent_derivatives(tensor_4d, domain_length)
    assert all(d is not None for d in dx_list)
    assert all(dd is not None for dd in ddx_list)


def test_output_shapes_3d(tensor_3d, domain_length):
    dx_list, ddx_list = calc_latent_derivatives(tensor_3d, domain_length)
    assert all(d.shape == (1, 2, 8) for d in dx_list)
    assert all(dd.shape == (1, 2, 8) for dd in ddx_list)


def test_output_shapes_4d(tensor_4d, domain_length):
    dx_list, ddx_list = calc_latent_derivatives(tensor_4d, domain_length)
    assert all(d.shape == (1, 2, 8, 8) for d in dx_list)
    assert all(dd.shape == (1, 2, 8, 8) for dd in ddx_list)
