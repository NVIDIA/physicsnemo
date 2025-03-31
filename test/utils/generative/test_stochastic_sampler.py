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

from typing import Optional

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
    ) -> Tensor:
        # Mock behavior: return input tensor for testing purposes
        return x * 0.9


# The test function for edm_sampler
@import_or_fail("cftime")
def test_stochastic_sampler(pytestconfig):

    from physicsnemo.utils.generative import stochastic_sampler

    net = MockNet()
    latents = torch.randn(2, 3, 448, 448)  # Mock latents
    img_lr = torch.randn(2, 3, 112, 112)  # Mock low-res image

    # Basic sampler functionality test
    result = stochastic_sampler(
        net=net,
        latents=latents,
        img_lr=img_lr,
        img_shape=448,
        patch_shape=448,
        overlap_pix=4,
        boundary_pix=2,
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
        img_shape=448,
        patch_shape=448,
        overlap_pix=4,
        boundary_pix=2,
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
        img_shape=448,
        patch_shape=448,
        overlap_pix=4,
        boundary_pix=2,
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


@import_or_fail("cftime")
def test_image_fuse_basic(pytestconfig):

    from physicsnemo.utils.generative import image_fuse

    # Basic test: No overlap, no boundary, one patch
    batch_size = 1
    img_shape_x = img_shape_y = 4
    patch_shape_x = patch_shape_y = 4
    overlap_pix = 0
    boundary_pix = 0

    input_tensor = torch.arange(1, 17).view(1, 1, 4, 4).cuda().float()
    fused_image = image_fuse(
        input_tensor,
        img_shape_x,
        img_shape_y,
        patch_shape_x,
        patch_shape_y,
        batch_size,
        overlap_pix,
        boundary_pix,
    )
    assert fused_image.shape == (batch_size, 1, img_shape_x, img_shape_y)
    expected_output = input_tensor
    assert torch.allclose(
        fused_image, expected_output, atol=1e-5
    ), "Output does not match expected output."


@import_or_fail("cftime")
def test_image_fuse_with_boundary(pytestconfig):

    from physicsnemo.utils.generative import image_fuse

    # Test with boundary pixels
    batch_size = 1
    img_shape_x = img_shape_y = 4
    patch_shape_x = patch_shape_y = 6
    overlap_pix = 0
    boundary_pix = 1

    input_tensor = torch.ones(1, 1, 6, 6).cuda().float()  # All ones for easy validation
    fused_image = image_fuse(
        input_tensor,
        img_shape_x,
        img_shape_y,
        patch_shape_x,
        patch_shape_y,
        batch_size,
        overlap_pix,
        boundary_pix,
    )
    assert fused_image.shape == (batch_size, 1, img_shape_x, img_shape_y)
    expected_output = (
        torch.ones(1, 1, 4, 4).cuda().float()
    )  # Expected output is just the inner 4x4 part
    assert torch.allclose(
        fused_image, expected_output, atol=1e-5
    ), "Output with boundary does not match expected output."


@import_or_fail("cftime")
def test_image_fuse_with_multiple_batches(pytestconfig):

    from physicsnemo.utils.generative import image_fuse

    # Test with multiple batches
    batch_size = 2
    img_shape_x = img_shape_y = 4
    patch_shape_x = patch_shape_y = 4
    overlap_pix = 0
    boundary_pix = 0

    input_tensor = (
        torch.cat(
            [
                torch.arange(1, 17).view(1, 1, 4, 4),
                torch.arange(17, 33).view(1, 1, 4, 4),
            ]
        )
        .cuda()
        .float()
    )
    input_tensor = input_tensor.repeat(2, 1, 1, 1)
    fused_image = image_fuse(
        input_tensor,
        img_shape_x,
        img_shape_y,
        patch_shape_x,
        patch_shape_y,
        batch_size,
        overlap_pix,
        boundary_pix,
    )
    assert fused_image.shape == (batch_size, 1, img_shape_x, img_shape_y)
    expected_output = (
        torch.cat(
            [
                torch.arange(1, 17).view(1, 1, 4, 4),
                torch.arange(17, 33).view(1, 1, 4, 4),
            ]
        )
        .cuda()
        .float()
    )
    assert torch.allclose(
        fused_image, expected_output, atol=1e-5
    ), "Output for multiple batches does not match expected output."


@import_or_fail("cftime")
def test_image_batching_basic(pytestconfig):

    from physicsnemo.utils.generative import image_batching

    # Test with no overlap, no boundary, no input_interp
    batch_size = 1
    img_shape_x = img_shape_y = 4
    patch_shape_x = patch_shape_y = 4
    overlap_pix = 0
    boundary_pix = 0

    input_tensor = torch.arange(1, 17).view(1, 1, 4, 4).cuda().float()
    batched_images = image_batching(
        input_tensor,
        img_shape_x,
        img_shape_y,
        patch_shape_x,
        patch_shape_y,
        batch_size,
        overlap_pix,
        boundary_pix,
    )
    assert batched_images.shape == (batch_size, 1, patch_shape_x, patch_shape_y)
    expected_output = input_tensor
    assert torch.allclose(
        batched_images, expected_output, atol=1e-5
    ), "Batched images do not match expected output."


@import_or_fail("cftime")
def test_image_batching_with_boundary(pytestconfig):
    # Test with boundary pixels, no overlap, no input_interp

    from physicsnemo.utils.generative import image_batching

    batch_size = 1
    img_shape_x = img_shape_y = 4
    patch_shape_x = patch_shape_y = 6
    overlap_pix = 0
    boundary_pix = 1

    input_tensor = torch.ones(1, 1, 4, 4).cuda().float()  # All ones for easy validation
    batched_images = image_batching(
        input_tensor,
        img_shape_x,
        img_shape_y,
        patch_shape_x,
        patch_shape_y,
        batch_size,
        overlap_pix,
        boundary_pix,
    )
    assert batched_images.shape == (1, 1, patch_shape_x, patch_shape_y)
    expected_output = torch.ones(1, 1, 6, 6).cuda().float()
    assert torch.allclose(
        batched_images, expected_output, atol=1e-5
    ), "Batched images with boundary do not match expected output."


@import_or_fail("cftime")
def test_image_batching_with_input_interp(pytestconfig):
    # Test with input_interp tensor

    from physicsnemo.utils.generative import image_batching

    batch_size = 1
    img_shape_x = img_shape_y = 4
    patch_shape_x = patch_shape_y = 4
    overlap_pix = 0
    boundary_pix = 0

    input_tensor = torch.arange(1, 17).view(1, 1, 4, 4).cuda().float()
    input_interp = torch.ones(1, 1, 4, 4).cuda().float()  # All ones for easy validation
    batched_images = image_batching(
        input_tensor,
        img_shape_x,
        img_shape_y,
        patch_shape_x,
        patch_shape_y,
        batch_size,
        overlap_pix,
        boundary_pix,
        input_interp=input_interp,
    )
    assert batched_images.shape == (batch_size, 2, patch_shape_x, patch_shape_y)
    expected_output = torch.cat((input_tensor, input_interp), dim=1)
    assert torch.allclose(
        batched_images, expected_output, atol=1e-5
    ), "Batched images with input_interp do not match expected output."
