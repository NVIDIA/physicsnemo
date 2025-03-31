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

import pytest
import torch
from pytest_utils import import_or_fail


@import_or_fail("cftime")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_regression_step(device, pytestconfig):

    from physicsnemo.models.diffusion import UNet
    from physicsnemo.utils.corrdiff import regression_step

    # define the net
    mock_unet = UNet(
        img_channels=2,
        N_grid_channels=4,
        embedding_type="zero",
        img_in_channels=8,
        img_out_channels=2,
        img_resolution=[16, 16],
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

    from physicsnemo.models.diffusion import EDMPrecondSR
    from physicsnemo.utils.corrdiff import diffusion_step
    from physicsnemo.utils.generative import deterministic_sampler, stochastic_sampler

    # Define the preconditioner
    mock_precond = EDMPrecondSR(
        img_resolution=[16, 16],
        img_in_channels=8,
        img_out_channels=2,
        img_channels=0,
        scale_cond_input=False,
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
        seed_batch_size=1,
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
        seed_batch_size=1,
        img_shape=(16, 16),
        img_out_channels=2,
        rank_batches=[[0]],
        img_lr=img_lr,
        rank=0,
        device=device,
    )

    # Assertions
    assert output.shape == (1, 2, 16, 16), "Output shape mismatch"
