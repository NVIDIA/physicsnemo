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
from einops import rearrange, repeat
from pytest_utils import import_or_fail


@import_or_fail("cftime")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_image_fuse_basic(pytestconfig, device):
    from physicsnemo.utils.patching import image_fuse

    # Basic test: No overlap, no boundary, one patch
    batch_size = 1
    for img_shape_y, img_shape_x in ((4, 4), (8, 4)):
        overlap_pix = 0
        boundary_pix = 0

        input_tensor = (
            torch.arange(1, img_shape_y * img_shape_x + 1)
            .view(1, 1, img_shape_y, img_shape_x)
            .to(device)
            .float()
        ).requires_grad_(True)
        fused_image = image_fuse(
            input_tensor,
            img_shape_y,
            img_shape_x,
            batch_size,
            overlap_pix,
            boundary_pix,
        )
        assert fused_image.shape == (batch_size, 1, img_shape_y, img_shape_x)
        expected_output = input_tensor
        assert torch.allclose(
            fused_image, expected_output, atol=1e-5
        ), "Output does not match expected output."

        # Make sure that image_fuse is differentiable
        loss = fused_image.sum()
        loss.backward()
        assert input_tensor.grad is not None


@import_or_fail("cftime")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_image_fuse_with_boundary(pytestconfig, device):
    from physicsnemo.utils.patching import image_fuse

    # Test with boundary pixels
    overlap_pix = 0
    boundary_pix = 1

    input_tensor = (torch.randn(1, 1, 8, 6).to(device).float()).requires_grad_(True)
    fused_image = image_fuse(
        input_tensor,
        img_shape_y=6,
        img_shape_x=4,
        batch_size=1,
        overlap_pix=overlap_pix,
        boundary_pix=boundary_pix,
    )
    assert fused_image.shape == (1, 1, 6, 4)
    expected_output = input_tensor[
        :, :, boundary_pix:-boundary_pix, boundary_pix:-boundary_pix
    ]
    assert torch.allclose(
        fused_image, expected_output, atol=1e-5
    ), "Output with boundary does not match expected output."

    # Make sure that image_fuse is differentiable
    loss = fused_image.sum()
    loss.backward()
    assert input_tensor.grad is not None


@import_or_fail("cftime")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_image_fuse_with_multiple_batches(pytestconfig, device):
    from physicsnemo.utils.patching import image_batching, image_fuse

    # Test with multiple batches
    batch_size = 2

    # Test cases: (img_shape_y, img_shape_x, patch_shape_y, patch_shape_x, overlap_pix, boundary_pix)
    test_cases = [
        (32, 32, 16, 16, 0, 0),  # Square image, no overlap/boundary
        (64, 32, 32, 16, 0, 0),  # Rectangular image, no overlap/boundary
        (48, 48, 16, 16, 4, 2),  # Square image, minimal overlap/boundary
        (64, 48, 32, 16, 6, 2),  # Rectangular, larger overlap/boundary
    ]

    for (
        img_shape_y,
        img_shape_x,
        patch_shape_y,
        patch_shape_x,
        overlap_pix,
        boundary_pix,
    ) in test_cases:
        # Create original test image
        original_image = (
            torch.rand(batch_size, 3, img_shape_y, img_shape_x).to(device).float()
        ).requires_grad_(True)

        # Apply image_batching to split the image into patches
        batched_images = image_batching(
            original_image, patch_shape_y, patch_shape_x, overlap_pix, boundary_pix
        )

        # Apply image_fuse to reconstruct the image from patches
        fused_image = image_fuse(
            batched_images,
            img_shape_y,
            img_shape_x,
            batch_size,
            overlap_pix,
            boundary_pix,
        )

        # Verify that image_fuse reverses image_batching
        assert torch.allclose(fused_image, original_image, atol=1e-5), (
            f"Failed on {device}: img=({img_shape_y},{img_shape_x}), "
            f"patch=({patch_shape_y},{patch_shape_x}), "
            f"overlap={overlap_pix}, boundary={boundary_pix}"
        )

        # Make sure that image_batching is differentiable
        loss = fused_image.sum()
        loss.backward()

        assert original_image.grad is not None


@import_or_fail("cftime")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_image_batching_basic(pytestconfig, device):
    from physicsnemo.utils.patching import image_batching

    # Test with no overlap, no boundary, no input_interp
    batch_size = 1
    patch_shape_x = patch_shape_y = 4
    overlap_pix = 0
    boundary_pix = 0

    input_tensor = (
        torch.arange(1, 17).view(1, 1, 4, 4).to(device).float()
    ).requires_grad_(True)
    batched_images = image_batching(
        input_tensor,
        patch_shape_y,
        patch_shape_x,
        overlap_pix,
        boundary_pix,
    )
    assert batched_images.shape == (batch_size, 1, patch_shape_y, patch_shape_x)
    expected_output = input_tensor
    assert torch.allclose(
        batched_images, expected_output, atol=1e-5
    ), "Batched images do not match expected output."

    # Make sure that image_batching is differentiable
    loss = batched_images.sum()
    loss.backward()
    assert input_tensor.grad is not None


@import_or_fail("cftime")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_image_batching_with_boundary(pytestconfig, device):
    from physicsnemo.utils.patching import image_batching

    # Test with boundary pixels, no overlap, no input_interp
    patch_shape_y = 8
    patch_shape_x = 6
    overlap_pix = 0
    boundary_pix = 1

    input_tensor = (torch.rand(1, 1, 6, 4).to(device).float()).requires_grad_(True)
    batched_images = image_batching(
        input_tensor,
        patch_shape_y,
        patch_shape_x,
        overlap_pix,
        boundary_pix,
    )
    # Create expected output using reflection padding
    expected_output = torch.nn.functional.pad(
        input_tensor,
        pad=(boundary_pix, boundary_pix, boundary_pix, boundary_pix),
        mode="reflect",
    )

    assert batched_images.shape == (1, 1, patch_shape_y, patch_shape_x)
    assert torch.allclose(
        batched_images, expected_output, atol=1e-5
    ), "Batched images with boundary do not match expected output."

    # Make sure that image_batching is differentiable
    loss = batched_images.sum()
    loss.backward()
    assert input_tensor.grad is not None


@import_or_fail("cftime")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_image_batching_with_input_interp(device, pytestconfig):
    from physicsnemo.utils.patching import image_batching

    # Test with input_interp tensor
    patch_shape_x = patch_shape_y = 4
    overlap_pix = 0
    boundary_pix = 0

    for img_shape_y, img_shape_x in ((4, 4), (16, 8)):
        img_size = img_shape_y * img_shape_x
        patch_num = (img_shape_y // patch_shape_y) * (img_shape_x // patch_shape_x)
        input_tensor = (
            torch.arange(1, img_size + 1)
            .view(1, 1, img_shape_y, img_shape_x)
            .to(device)
            .float()
        ).requires_grad_(True)
        input_interp = (
            torch.arange(-patch_shape_y * patch_shape_x, 0)
            .view(1, 1, patch_shape_y, patch_shape_x)
            .to(device)
            .float()
        ).requires_grad_(True)
        batched_images = image_batching(
            input_tensor,
            patch_shape_y,
            patch_shape_x,
            overlap_pix,
            boundary_pix,
            input_interp=input_interp,
        )
        assert batched_images.shape == (patch_num, 2, patch_shape_y, patch_shape_x)

        # Define expected_output using einops operations
        expected_output = torch.cat(
            (
                rearrange(
                    input_tensor,
                    "b c (nb_p_h p_h) (nb_p_w p_w) -> (b nb_p_w nb_p_h) c p_h p_w",
                    p_h=patch_shape_y,
                    p_w=patch_shape_x,
                ),
                repeat(
                    input_interp,
                    "b c p_h p_w -> (b nb_p_w nb_p_h) c p_h p_w",
                    nb_p_h=img_shape_y // patch_shape_y,
                    nb_p_w=img_shape_x // patch_shape_x,
                ),
            ),
            dim=1,
        )

        assert torch.allclose(
            batched_images, expected_output, atol=1e-5
        ), "Batched images with input_interp do not match expected output."

        # Make sure that image_batching is differentiable
        loss = batched_images.sum()
        loss.backward()
        assert input_interp.grad is not None
        assert input_tensor.grad is not None
