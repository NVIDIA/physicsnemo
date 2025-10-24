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

import os
import sys
from pathlib import Path

import pytest
import torch

import physicsnemo

script_path: str = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import common  # noqa: E402


def _create_diffusion_fwi_net(arch_type: str = "fwi_small", **kwargs):
    """
    Factory function to create DiffusionFWINet with different configurations.
    """
    # Import DiffusionFWINet from the examples directory
    sys.path.append(
        os.path.join(
            os.path.dirname(__file__),
            "../../../examples/geophysics/diffusion_fwi/utils",
        )
    )
    from nn import DiffusionFWINet  # noqa: E402

    # Small test configuration - reduced from training config for faster tests
    if arch_type == "fwi_small":
        return DiffusionFWINet(
            x_resolution=[32, 32],
            x_channels=3,
            y_resolution=[50, 32],
            y_channels=6,
            encoder_hidden_channels=32,
            num_encoder_blocks=2,
            N_grid_channels=2,
            model_channels=16,
            channel_mult=[1, 2, 2],
            num_blocks=2,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")


def _instantiate_diffusion_fwi_net(
    arch_type: str = "fwi_small", seed: int = 0, **kwargs
):
    """
    Helper function to instantiate DiffusionFWINet with reproducible random
    parameters.
    """
    model = _create_diffusion_fwi_net(arch_type=arch_type, **kwargs)
    gen: torch.Generator = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(
                torch.randn(
                    param.shape,
                    generator=gen,
                    dtype=param.dtype,
                )
            )
    return model


def generate_data(device: str):
    """
    Helper function to generate data for the test.
    """
    torch.manual_seed(0)
    B = 2
    # Data shapes matching the small model configuration
    x_shape = (B, 3, 32, 32)  # (B, x_channels, H, W)
    y_shape = (B, 6, 50, 32)  # (B, y_channels, T, W)
    sigma_shape = (B,)  # (B,)

    x: torch.Tensor = torch.randn(*x_shape).to(device)
    y: torch.Tensor = torch.randn(*y_shape).to(device)
    sigma: torch.Tensor = torch.randn(*sigma_shape).to(device)

    return x, y, sigma


@pytest.mark.parametrize(
    "device",
    [
        "cuda:0",
        "cpu",
    ],
    ids=["gpu", "cpu"],
)
@pytest.mark.parametrize(
    "arch_type",
    ["fwi_small"],
    ids=["small"],
)
def test_diffusion_fwi_net_instantiation(device, arch_type):
    """
    Test that DiffusionFWINet can be instantiated and check its attributes.
    """
    model = _instantiate_diffusion_fwi_net(arch_type=arch_type).to(device)

    # Check that the model is instantiated correctly
    if arch_type == "fwi_small":
        assert model.x_resolution == (32, 32)
        assert model.x_channels == 3
        assert model.y_resolution == (50, 32)
        assert model.y_channels == 6

    # Test forward pass with correct input shapes
    x, y, sigma = generate_data(device)
    out: torch.Tensor = model(x, y, sigma)

    # Check output shape
    assert out.shape == x.shape  # Should match input x shape
    assert out.dtype == x.dtype


@pytest.mark.parametrize(
    "device",
    [
        "cuda:0",
        "cpu",
    ],
    ids=["gpu", "cpu"],
)
@pytest.mark.parametrize(
    "arch_type",
    ["fwi_small"],
    ids=["small"],
)
def test_diffusion_fwi_net_non_regression(device, arch_type):
    """
    Test that DiffusionFWINet can be instantiated and compare the output with a
    reference output generated with v1.2.0.
    """
    run_id = f"{arch_type}_{'cpu' if device == 'cpu' else 'gpu'}"
    model = _instantiate_diffusion_fwi_net(arch_type=arch_type).to(device)

    x, y, sigma = generate_data(device)
    out: torch.Tensor = model(x, y, sigma)

    # NOTE: on GPU scaled_dot_product_attention gives large differences on
    # different hardware. Need to increase tolerances to make the test pass.
    if device == "cuda:0":
        atol, rtol = 5.0, 1e-3
    else:
        atol, rtol = 1e-3, 1e-3

    assert common.validate_accuracy(
        out,
        file_name=f"output_diffusion_fwi_net_{run_id}-v1.2.0.pth",
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    "device",
    [
        "cuda:0",
        "cpu",
    ],
    ids=["gpu", "cpu"],
)
@pytest.mark.parametrize(
    "arch_type",
    ["fwi_small"],
    ids=["small"],
)
def test_diffusion_fwi_net_non_regression_from_checkpoint(device, arch_type):
    """
    Tests simple loading and non-regression of a checkpoint generated with the
    DiffusionFWINet class.
    """
    run_id = f"{arch_type}_{'cpu' if device == 'cpu' else 'gpu'}"
    file_name: str = str(
        Path(__file__).parents[1].resolve()
        / Path("data")
        / Path(f"checkpoint_diffusion_fwi_net_{run_id}-v1.2.0.mdlus")
    )

    model: physicsnemo.Module = physicsnemo.Module.from_checkpoint(
        file_name=file_name,
    ).to(device)

    # Check that the model is instantiated correctly
    if arch_type == "fwi_small":
        assert model.x_resolution == (32, 32)
        assert model.x_channels == 3
        assert model.y_resolution == (50, 32)
        assert model.y_channels == 6

    x, y, sigma = generate_data(device)
    out: torch.Tensor = model(x, y, sigma)

    # NOTE: on GPU scaled_dot_product_attention gives large differences on
    # different hardware. Need to increase tolerances to make the test pass.
    if device == "cuda:0":
        atol, rtol = 5.0, 1e-3
    else:
        atol, rtol = 1e-3, 1e-3

    assert common.validate_accuracy(
        out,
        file_name=f"output_diffusion_fwi_net_{run_id}-v1.2.0.pth",
        atol=atol,
        rtol=rtol,
    )


# ---------------------------------------------------------------------------
#   FOR CHECKPOINT AND DATA GENERATION
# ---------------------------------------------------------------------------

# For checkpoint and data generation
# @pytest.mark.parametrize(
#     "device",
#     ["cpu", "cuda:0"],
#     ids=["cpu", "gpu"],
# )
# @pytest.mark.parametrize(
#     "arch_type",
#     ["fwi_small"],
#     ids=["small"],
# )
# def test_diffusion_fwi_net_generate_data(device, arch_type):
#     """
#     Function to generate data for the DiffusionFWINet tests.
#     """
#     run_id = f"{arch_type}_{'cpu' if device == 'cpu' else 'gpu'}"
#     model = _instantiate_diffusion_fwi_net(arch_type=arch_type).to(device)

#     # Check that the model is instantiated correctly
#     if arch_type == "fwi_small":
#         assert model.x_resolution == (32, 32)
#         assert model.x_channels == 3
#         assert model.y_resolution == (50, 32)
#         assert model.y_channels == 6

#     x, y, sigma = generate_data(device)
#     out: torch.Tensor = model(x, y, sigma)

#     # Save model checkpoint and reference output
#     model.save(f"checkpoint_diffusion_fwi_net_{run_id}-v1.2.0.mdlus")
#     torch.save(out, f"output_diffusion_fwi_net_{run_id}-v1.2.0.pth")
