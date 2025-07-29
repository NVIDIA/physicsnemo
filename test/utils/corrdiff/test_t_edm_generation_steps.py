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
def test_diffusion_step(device, pytestconfig):
    from physicsnemo.experimental.models.diffusion.preconditioning import (
        tEDMPrecondSuperRes,
    )
    from physicsnemo.utils.corrdiff import diffusion_step
    from physicsnemo.utils.diffusion import deterministic_sampler, stochastic_sampler

    torch._dynamo.reset()

    # Define the preconditioner
    precond = tEDMPrecondSuperRes(
        img_resolution=[16, 16],
        img_in_channels=8,
        img_out_channels=2,
        nu=10,
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
        net=precond,
        sampler_fn=sampler_fn,
        img_shape=(16, 16),
        img_out_channels=2,
        rank_batches=[[0]],
        img_lr=img_lr,
        rank=0,
        device=device,
        distribution="student_t",
        nu=10,
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
        net=precond,
        sampler_fn=sampler_fn,
        img_shape=(16, 16),
        img_out_channels=2,
        rank_batches=[[0]],
        img_lr=img_lr,
        rank=0,
        device=device,
        distribution="student_t",
        nu=10,
    )

    # Assertions
    assert output.shape == (1, 2, 16, 16), "Output shape mismatch"
