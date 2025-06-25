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

from functools import partial
from typing import Callable, Optional

import pytest
import torch
from pytest_utils import import_or_fail


# Mock network class
class MockNet:
    def __init__(self, sigma_min=0.1, sigma_max=1000):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def round_sigma(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def __call__(
        self,
        x: torch.Tensor,
        x_lr: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor],
        global_index: Optional[torch.Tensor] = None,
        embedding_selector: Optional[Callable] = None,
    ) -> torch.Tensor:
        # Mock behavior: return input tensor for testing purposes
        return x * 0.9


@import_or_fail("cftime")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_regression_step(device, pytestconfig):

    from physicsnemo.models.diffusion import UNet
    from physicsnemo.utils.corrdiff import regression_step

    # define the net
    mock_unet = UNet(
        img_resolution=[16, 16],
        img_in_channels=8,
        img_out_channels=2,
        N_grid_channels=4,
        embedding_type="zero",
    ).to(device)

    # Define the input parameters
    img_lr = torch.randn(1, 2, 16, 16).to(device)
    latents_shape = torch.Size([2, 4, 16, 16])

    # Call the function
    output = regression_step(mock_unet, img_lr, latents_shape)

    # Assertions
    assert output.shape == (2, 2, 16, 16), "Output shape mismatch"


@import_or_fail("cftime")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_diffusion_step(device, pytestconfig):

    from physicsnemo.models.diffusion import EDMPrecondSuperResolution
    from physicsnemo.utils.corrdiff import diffusion_step
    from physicsnemo.utils.diffusion import deterministic_sampler, stochastic_sampler

    # Define the preconditioner
    mock_precond = EDMPrecondSuperResolution(
        img_resolution=[16, 16], img_in_channels=8, img_out_channels=2
    ).to(device)

    # Define the input parameters
    img_lr = torch.randn(1, 4, 16, 16).to(device)

    # Define the sampler
    sampler_fn = partial(
        deterministic_sampler,
        num_steps=2,
    )

    # Call the function
    output = diffusion_step(
        net=mock_precond,
        sampler_fn=sampler_fn,
        img_shape=(16, 16),
        img_out_channels=2,
        rank_batches=[[0]],
        img_lr=img_lr,
        rank=0,
        device=device,
    )

    # Assertions
    assert output.shape == (1, 2, 16, 16), "Output shape mismatch"

    # Also test with stochastic sampler
    sampler_fn = partial(
        stochastic_sampler,
        num_steps=2,
    )

    # Call the function
    output = diffusion_step(
        net=mock_precond,
        sampler_fn=sampler_fn,
        img_shape=(16, 16),
        img_out_channels=2,
        rank_batches=[[0]],
        img_lr=img_lr,
        rank=0,
        device=device,
    )

    # Assertions
    assert output.shape == (1, 2, 16, 16), "Output shape mismatch"


@import_or_fail("cftime")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_diffusion_step_rectangle(device, pytestconfig):
    from physicsnemo.utils.corrdiff import diffusion_step
    from physicsnemo.utils.diffusion import stochastic_sampler
    from physicsnemo.utils.patching import GridPatching2D

    img_shape_y, img_shape_x = 32, 16
    seed_batch_size = 4

    mock_precond = MockNet()

    # Define the input parameters
    img_lr = (
        torch.randn(1, 4, img_shape_y, img_shape_x)
        .expand(seed_batch_size, -1, -1, -1)
        .to(device)
    )

    # Stochastic sampler without patching
    sampler_fn = partial(
        stochastic_sampler,
        num_steps=2,
    )

    # Call the function
    output = diffusion_step(
        net=mock_precond,
        sampler_fn=sampler_fn,
        img_shape=(img_shape_y, img_shape_x),
        img_out_channels=2,
        rank_batches=[list(range(seed_batch_size))],
        img_lr=img_lr,
        mean_hr=None,
        rank=0,
        device=device,
    )

    # Assertions
    assert output.shape == (
        seed_batch_size,
        2,
        img_shape_y,
        img_shape_x,
    ), "Output shape mismatch"

    # Test with mean_hr conditioning

    # Define the input parameters
    mean_hr = torch.randn(1, 2, img_shape_y, img_shape_x).to(device)
    img_lr = (
        torch.randn(1, 4, img_shape_y, img_shape_x)
        .expand(seed_batch_size, -1, -1, -1)
        .to(device)
    )

    # Stochastic sampler without patching
    sampler_fn = partial(
        stochastic_sampler,
        num_steps=2,
    )

    # Call the function
    output = diffusion_step(
        net=mock_precond,
        sampler_fn=sampler_fn,
        img_shape=(img_shape_y, img_shape_x),
        img_out_channels=2,
        rank_batches=[list(range(seed_batch_size))],
        img_lr=img_lr,
        mean_hr=mean_hr,
        rank=0,
        device=device,
    )

    # Assertions
    assert output.shape == (
        seed_batch_size,
        2,
        img_shape_y,
        img_shape_x,
    ), "Output shape mismatch"

    # Test with mean_hr conditioning and rectangular patching

    # Define the input parameters
    mean_hr = torch.randn(1, 2, img_shape_y, img_shape_x).to(device)
    img_lr = (
        torch.randn(1, 4, img_shape_y, img_shape_x)
        .expand(seed_batch_size, -1, -1, -1)
        .to(device)
    )

    # Define patching utility
    patching = GridPatching2D(
        img_shape=(img_shape_y, img_shape_x),
        patch_shape=(16, 10),
        overlap_pix=4,
        boundary_pix=2,
    )

    # Stochastic sampler with rectangular patching
    sampler_fn = partial(stochastic_sampler, num_steps=2, patching=patching)

    # Call the function
    output = diffusion_step(
        net=mock_precond,
        sampler_fn=sampler_fn,
        img_shape=(img_shape_y, img_shape_x),
        img_out_channels=2,
        rank_batches=[list(range(seed_batch_size))],
        img_lr=img_lr,
        mean_hr=mean_hr,
        rank=0,
        device=device,
    )

    # Assertions
    assert output.shape == (
        seed_batch_size,
        2,
        img_shape_y,
        img_shape_x,
    ), "Output shape mismatch"
