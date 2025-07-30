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

from physicsnemo.models.diffusion.preconditioning import EDMPrecondSuperResolution


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
    from physicsnemo.utils.diffusion import deterministic_sampler

    latents = torch.randn(1, 3, 64, 64)
    img_lr = torch.randn(1, 3, 64, 64)
    output = deterministic_sampler(net=mock_net, latents=latents, img_lr=img_lr)
    assert isinstance(output, torch.Tensor)
    assert output.shape == latents.shape


# Test for parameter validation
@import_or_fail("cftime")
@pytest.mark.parametrize("solver", ["invalid_solver", "euler", "heun"])
def test_deterministic_sampler_solver_validation(mock_net, solver, pytestconfig):
    from physicsnemo.utils.diffusion import deterministic_sampler

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
    from physicsnemo.utils.diffusion import deterministic_sampler

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
    from physicsnemo.utils.diffusion import deterministic_sampler

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
    from physicsnemo.utils.diffusion import deterministic_sampler

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
    from physicsnemo.utils.diffusion import deterministic_sampler

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
    from physicsnemo.utils.diffusion import deterministic_sampler

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
    from physicsnemo.utils.diffusion import deterministic_sampler

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


# Test correctness with known ODE solution
@import_or_fail("cftime")
def test_deterministic_sampler_correctness(pytestconfig):
    from physicsnemo.utils.diffusion import deterministic_sampler

    # Create a simple network that implements our ODE: dx/dt = -x ==> x(t) = exp(-t)
    class SimpleODENet(torch.nn.Module):
        def __init__(self, sigma_min=0.002, sigma_max=4.0):
            super().__init__()
            self.sigma_min = sigma_min
            self.sigma_max = sigma_max

        def forward(self, x, img_lr, sigma, class_labels=None):
            # Simulating ODE dx/dt = -x, we need denoiser to return (t + 1) * x
            # See EDM paper eqs. 3 and 4, and note EDM uses sigma <==> t
            return (sigma + 1.0) * x

        def round_sigma(self, sigma):
            return torch.tensor(sigma)

    # Create network and initial condition
    net = SimpleODENet()
    x0 = (
        torch.exp(-net.sigma_max * torch.ones(1, 1, 1, 1)) / net.sigma_max
    )  # "Initial condition" x(sigma_max) = exp(-sigma_max)
    img_lr = torch.zeros(1, 1, 1, 1)  # Dummy conditioning input

    # Run the sampler
    x_final = deterministic_sampler(
        net=net,
        latents=x0,
        img_lr=img_lr,
        num_steps=100,
        sigma_min=net.sigma_min,
        sigma_max=net.sigma_max,
    )

    # Analytical solution of x(t) = exp(-t) at t=0 is 1
    analytical_solution = torch.ones_like(x_final)

    # Check with loose tolerance since we're using numerical integration
    assert torch.allclose(x_final, analytical_solution, rtol=1e-2, atol=1e-2), (
        f"Numerical solution {x_final.item():.6f} does not match analytical solution {analytical_solution.item():.6f}"
    )


def setup_model_learnable_embd(img_resolution, C_x, C_cond, global_lr=False, seed=0):
    """
    Create a model with similar architecture to CorrDiff (learnable positional
    embeddings, self-attention, learnable lead time embeddings).
    """
    # Smaller architecture variant with learnable positional embeddings
    # (similar to CorrDiff example)
    N_pos = 20
    lt_steps = 9
    lt_channels = 4
    attn_res = (
        img_resolution[0] // 4
        if isinstance(img_resolution, list) or isinstance(img_resolution, tuple)
        else img_resolution // 4
    )
    torch.manual_seed(seed)
    if global_lr:
        img_in_channels = C_x + N_pos + C_cond * 2 + lt_channels
    else:
        img_in_channels = C_x + N_pos + C_cond + lt_channels
    model = EDMPrecondSuperResolution(
        model_type="SongUNetPosLtEmbd",
        img_resolution=img_resolution,
        img_in_channels=img_in_channels,
        img_out_channels=C_x,
        model_channels=16,
        channel_mult=[1, 2, 2],
        channel_mult_emb=2,
        num_blocks=2,
        attn_resolutions=[attn_res],
        gridtype="learnable",
        N_grid_channels=N_pos,
        lead_time_steps=lt_steps,
        lead_time_channels=lt_channels,
    )
    return model


# The test function for patch-based deterministic_sampler
@import_or_fail("cftime")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_deterministic_sampler_full_domain_lead_time(device, pytestconfig):
    from physicsnemo.utils.diffusion import deterministic_sampler

    latents = torch.randn(1, 3, 16, 16, device=device)  # Mock latents
    img_lr = torch.randn(1, 3, 16, 16, device=device)  # Mock low-res image

    net = setup_model_learnable_embd((16, 16), C_x=3, C_cond=3)
    net.to(device)
    # Test with mean_hr conditioning
    mean_hr = torch.randn(1, 3, 16, 16, device=device)
    result_mean_hr = deterministic_sampler(
        net=net,
        latents=latents,
        img_lr=img_lr,
        patching=None,
        mean_hr=mean_hr,
        num_steps=2,
        lead_time_label=torch.tensor([1]),
    )

    assert result_mean_hr.shape == latents.shape, (
        "Mean HR conditioned output shape does not match expected shape"
    )


# The test function for edm_sampler with rectangular domain and patching
@import_or_fail("cftime")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_deterministic_sampler_rectangle_patching_lead_time(device, pytestconfig):
    from physicsnemo.utils.diffusion import deterministic_sampler
    from physicsnemo.utils.patching import GridPatching2D

    torch._dynamo.reset()
    img_shape_y, img_shape_x = 32, 32
    patch_shape_y, patch_shape_x = 32, 16

    net = setup_model_learnable_embd(
        (img_shape_y, img_shape_x), C_x=3, C_cond=3, global_lr=True
    )
    net.to(device)
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
    result_mean_hr = deterministic_sampler(
        net=net,
        latents=latents,
        img_lr=img_lr,
        patching=patching,
        mean_hr=mean_hr,
        num_steps=2,
        lead_time_label=torch.tensor([1]),
    )

    assert result_mean_hr.shape == latents.shape, (
        "Mean HR conditioned output shape does not match expected shape"
    )


# Test that the deterministic sampler is differentiable with rectangular patching
# (tests differentiation through the patching and fusing)
@import_or_fail("cftime")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_deterministic_sampler_patching_differentiable(device, pytestconfig):
    from physicsnemo.utils.diffusion import deterministic_sampler
    from physicsnemo.utils.patching import GridPatching2D

    torch._dynamo.reset()

    img_shape_y, img_shape_x = 32, 32
    patch_shape_y, patch_shape_x = 32, 16

    net = setup_model_learnable_embd(
        (img_shape_y, img_shape_x), C_x=3, C_cond=3, global_lr=True
    )

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
    net.to(device)
    # Test with mean_hr conditioning
    mean_hr = torch.randn(2, 3, img_shape_y, img_shape_x, device=device)
    result_mean_hr = deterministic_sampler(
        net=net,
        latents=a * latents + b,
        img_lr=c * img_lr + d,
        patching=patching,
        mean_hr=e * mean_hr + f,
        num_steps=2,
        lead_time_label=torch.tensor([1]),
    )

    assert result_mean_hr.shape == latents.shape, (
        "Mean HR conditioned output shape does not match expected shape"
    )

    loss = result_mean_hr.sum()
    loss.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert c.grad is not None
    assert d.grad is not None
    assert e.grad is not None
    assert f.grad is not None
