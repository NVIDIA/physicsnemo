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
from pytest_utils import import_or_fail


# Mock a minimal net class for testing
class MockNet:
    def __init__(self, sigma_min=0.0, sigma_max=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x, img_lr, sigma, class_labels):
        return x  # Mock behavior of net

    def round_sigma(self, sigma):
        return torch.tensor(sigma)


# Define a fixture for the network
@pytest.fixture
def mock_net():
    return MockNet()


# Basic functionality test
@import_or_fail("cftime")
def test_deterministic_sampler_output_type_and_shape(mock_net, pytestconfig):

    from physicsnemo.utils.generative import deterministic_sampler

    latents = torch.randn(1, 3, 64, 64)
    img_lr = torch.randn(1, 3, 64, 64)
    output = deterministic_sampler(net=mock_net, latents=latents, img_lr=img_lr)
    assert isinstance(output, torch.Tensor)
    assert output.shape == latents.shape


# Test for parameter validation
@import_or_fail("cftime")
@pytest.mark.parametrize("solver", ["invalid_solver", "euler", "heun"])
def test_deterministic_sampler_solver_validation(mock_net, solver, pytestconfig):

    from physicsnemo.utils.generative import deterministic_sampler

    if solver == "invalid_solver":
        with pytest.raises(ValueError):
            deterministic_sampler(
                net=mock_net,
                latents=torch.randn(1, 3, 64, 64),
                img_lr=torch.randn(1, 3, 64, 64),
                solver=solver,
            )
    else:
        # No exception should be raised for valid solvers
        deterministic_sampler(
            net=mock_net,
            latents=torch.randn(1, 3, 64, 64),
            img_lr=torch.randn(1, 3, 64, 64),
            solver=solver,
        )


# Test for edge cases
@import_or_fail("cftime")
def test_deterministic_sampler_edge_cases(mock_net, pytestconfig):

    from physicsnemo.utils.generative import deterministic_sampler

    latents = torch.randn(1, 3, 64, 64)
    img_lr = torch.randn(1, 3, 64, 64)
    # Test with extreme rho values, zero noise levels, etc.
    output = deterministic_sampler(
        net=mock_net, latents=latents, img_lr=img_lr, rho=1000, sigma_min=0, sigma_max=0
    )
    assert isinstance(output, torch.Tensor)


# Test discretization
@import_or_fail("cftime")
@pytest.mark.parametrize("discretization", ["vp", "ve", "iddpm", "edm"])
def test_deterministic_sampler_discretization(mock_net, discretization, pytestconfig):

    from physicsnemo.utils.generative import deterministic_sampler

    latents = torch.randn(1, 3, 64, 64)
    img_lr = torch.randn(1, 3, 64, 64)
    output = deterministic_sampler(
        net=mock_net, latents=latents, img_lr=img_lr, discretization=discretization
    )
    assert isinstance(output, torch.Tensor)


# Test schedule
@import_or_fail("cftime")
@pytest.mark.parametrize("schedule", ["vp", "ve", "linear"])
def test_deterministic_sampler_schedule(mock_net, schedule, pytestconfig):

    from physicsnemo.utils.generative import deterministic_sampler

    latents = torch.randn(1, 3, 64, 64)
    img_lr = torch.randn(1, 3, 64, 64)
    output = deterministic_sampler(
        net=mock_net, latents=latents, img_lr=img_lr, schedule=schedule
    )
    assert isinstance(output, torch.Tensor)


# Test number of steps
@import_or_fail("cftime")
@pytest.mark.parametrize("num_steps", [1, 5, 18])
def test_deterministic_sampler_num_steps(mock_net, num_steps, pytestconfig):

    from physicsnemo.utils.generative import deterministic_sampler

    latents = torch.randn(1, 3, 64, 64)
    img_lr = torch.randn(1, 3, 64, 64)
    output = deterministic_sampler(
        net=mock_net, latents=latents, img_lr=img_lr, num_steps=num_steps
    )
    assert isinstance(output, torch.Tensor)


# Test sigma
@import_or_fail("cftime")
@pytest.mark.parametrize("sigma_min, sigma_max", [(0.001, 0.01), (1.0, 1.5)])
def test_deterministic_sampler_sigma_boundaries(
    mock_net, sigma_min, sigma_max, pytestconfig
):

    from physicsnemo.utils.generative import deterministic_sampler

    latents = torch.randn(1, 3, 64, 64)
    img_lr = torch.randn(1, 3, 64, 64)
    output = deterministic_sampler(
        net=mock_net,
        latents=latents,
        img_lr=img_lr,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    assert isinstance(output, torch.Tensor)


# Test error handling
@import_or_fail("cftime")
@pytest.mark.parametrize("scaling", ["invalid_scaling", "vp", "none"])
def test_deterministic_sampler_scaling_validation(mock_net, scaling, pytestconfig):

    from physicsnemo.utils.generative import deterministic_sampler

    latents = torch.randn(1, 3, 64, 64)
    img_lr = torch.randn(1, 3, 64, 64)
    if scaling == "invalid_scaling":
        with pytest.raises(ValueError):
            deterministic_sampler(
                net=mock_net, latents=latents, img_lr=img_lr, scaling=scaling
            )
    else:
        output = deterministic_sampler(
            net=mock_net, latents=latents, img_lr=img_lr, scaling=scaling
        )
        assert isinstance(output, torch.Tensor)
