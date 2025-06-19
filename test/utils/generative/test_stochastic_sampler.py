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

from typing import Callable, Optional

import pytest
import torch
from pytest_utils import import_or_fail
from torch import Tensor


# Mock network class
class MockNet:
    def __init__(self, sigma_min=0.1, sigma_max=1000):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def round_sigma(self, t: Tensor) -> Tensor:
        return t

    def __call__(
        self,
        x: Tensor,
        x_lr: Tensor,
        t: Tensor,
        class_labels: Optional[Tensor],
        global_index: Optional[Tensor] = None,
        embedding_selector: Optional[Callable] = None,
    ) -> Tensor:
        # Mock behavior: return input tensor for testing purposes
        return x * 0.9


# The test function for edm_sampler
@import_or_fail("cftime")
def test_stochastic_sampler(pytestconfig):

    from physicsnemo.utils.diffusion import stochastic_sampler

    net = MockNet()
    latents = torch.randn(2, 3, 448, 448)  # Mock latents
    img_lr = torch.randn(2, 3, 112, 112)  # Mock low-res image

    # Basic sampler functionality test
    result = stochastic_sampler(
        net=net,
        latents=latents,
        img_lr=img_lr,
        patching=None,
        mean_hr=None,
        num_steps=4,
        sigma_min=0.002,
        sigma_max=800,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    )

    assert result.shape == latents.shape, "Output shape does not match expected shape"

    # Test with mean_hr conditioning
    mean_hr = torch.randn(2, 3, 112, 112)
    result_mean_hr = stochastic_sampler(
        net=net,
        latents=latents,
        img_lr=img_lr,
        patching=None,
        mean_hr=mean_hr,
        num_steps=2,
        sigma_min=0.002,
        sigma_max=800,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    )

    assert (
        result_mean_hr.shape == latents.shape
    ), "Mean HR conditioned output shape does not match expected shape"

    # Test with different S_churn value
    result_churn = stochastic_sampler(
        net=net,
        latents=latents,
        img_lr=img_lr,
        patching=None,
        mean_hr=None,
        num_steps=3,
        sigma_min=0.002,
        sigma_max=800,
        rho=7,
        S_churn=0.1,  # Non-zero churn value
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    )

    assert (
        result_churn.shape == latents.shape
    ), "Churn output shape does not match expected shape"


# The test function for edm_sampler with rectangular domain and patching
@import_or_fail("cftime")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_stochastic_sampler_rectangle_patching(device, pytestconfig):
    from physicsnemo.utils.diffusion import stochastic_sampler
    from physicsnemo.utils.patching import GridPatching2D

    net = MockNet()

    img_shape_y, img_shape_x = 256, 64
    patch_shape_y, patch_shape_x = 16, 10

    latents = torch.randn(2, 3, img_shape_y, img_shape_x, device=device)  # Mock latents
    img_lr = torch.randn(
        2, 3, img_shape_y, img_shape_x, device=device
    )  # Mock low-res image

    # Test with patching
    patching = GridPatching2D(
        img_shape=(img_shape_y, img_shape_x),
        patch_shape=(patch_shape_y, patch_shape_x),
        overlap_pix=4,
        boundary_pix=2,
    )

    # Test with mean_hr conditioning
    mean_hr = torch.randn(2, 3, img_shape_y, img_shape_x, device=device)
    result_mean_hr = stochastic_sampler(
        net=net,
        latents=latents,
        img_lr=img_lr,
        patching=patching,
        mean_hr=mean_hr,
        num_steps=2,
        sigma_min=0.002,
        sigma_max=800,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    )

    assert (
        result_mean_hr.shape == latents.shape
    ), "Mean HR conditioned output shape does not match expected shape"


# Test that the stochastic sampler is differentiable with rectangular patching
# (tests differentiation through the patching and fusing)
@import_or_fail("cftime")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_stochastic_sampler_patching_differentiable(device, pytestconfig):
    from physicsnemo.utils.diffusion import stochastic_sampler
    from physicsnemo.utils.patching import GridPatching2D

    # Mock network class
    class MockNet:
        def __init__(self, sigma_min=0.1, sigma_max=1000):
            self.sigma_min = sigma_min
            self.sigma_max = sigma_max

        def round_sigma(self, t: Tensor) -> Tensor:
            return t

        def __call__(
            self,
            x: Tensor,
            x_lr: Tensor,
            t: Tensor,
            class_labels: Optional[Tensor],
            global_index: Optional[Tensor] = None,
            embedding_selector: Optional[Callable] = None,
        ) -> Tensor:
            # Mock behavior: return input tensor for testing purposes
            return x * 0.9 + x_lr[:, : x.shape[1], :, :] * 0.1

    net = MockNet()

    img_shape_y, img_shape_x = 256, 64
    patch_shape_y, patch_shape_x = 16, 10

    latents = torch.randn(2, 3, img_shape_y, img_shape_x, device=device)  # Mock latents
    img_lr = torch.randn(
        2, 3, img_shape_y, img_shape_x, device=device
    )  # Mock low-res image

    # Tensors with requires grad
    a = torch.randn(1, requires_grad=True, device=device)
    b = torch.randn(1, requires_grad=True, device=device)
    c = torch.randn(1, requires_grad=True, device=device)
    d = torch.randn(1, requires_grad=True, device=device)
    e = torch.randn(1, requires_grad=True, device=device)
    f = torch.randn(1, requires_grad=True, device=device)

    # Test with patching
    patching = GridPatching2D(
        img_shape=(img_shape_y, img_shape_x),
        patch_shape=(patch_shape_y, patch_shape_x),
        overlap_pix=4,
        boundary_pix=2,
    )

    # Test with mean_hr conditioning
    mean_hr = torch.randn(2, 3, img_shape_y, img_shape_x, device=device)
    result_mean_hr = stochastic_sampler(
        net=net,
        latents=a * latents + b,
        img_lr=c * img_lr + d,
        patching=patching,
        mean_hr=e * mean_hr + f,
        num_steps=2,
        sigma_min=0.002,
        sigma_max=800,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    )

    assert (
        result_mean_hr.shape == latents.shape
    ), "Mean HR conditioned output shape does not match expected shape"

    loss = result_mean_hr.sum()
    loss.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert c.grad is not None
    assert d.grad is not None
    assert e.grad is not None
    assert f.grad is not None
