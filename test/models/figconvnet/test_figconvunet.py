# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
from torch.testing import assert_close

import physicsnemo
from test import common
from test.conftest import requires_module

# Default test configuration
IN_C = 3
OUT_C = 1
KERNEL_SIZE = 9
HIDDEN_C = [IN_C, 4, 4, 4]
MLP_C = [8, 8]


def _create_model(**kwargs) -> physicsnemo.Module:
    """Helper to create FIGConvUNet model with default or custom args."""
    from physicsnemo.models.figconvnet.figconvunet import FIGConvUNet

    defaults = {
        "in_channels": IN_C,
        "out_channels": OUT_C,
        "kernel_size": KERNEL_SIZE,
        "hidden_channels": HIDDEN_C,
        "mlp_channels": MLP_C,
    }
    defaults.update(kwargs)
    return FIGConvUNet(**defaults)


@requires_module("torch_scatter")
@pytest.mark.parametrize(
    "config",
    ["default", "custom"],
    ids=["with_defaults", "with_custom_args"],
)
def test_figconvunet_constructor(config, device):
    """Test FIGConvUNet constructor and attributes (MOD-008a)."""
    torch.manual_seed(0)

    if config == "default":
        model = _create_model().to(device)
        # Verify default values
        assert model.in_channels == IN_C
        assert model.out_channels == OUT_C
        assert model.hidden_channels == HIDDEN_C
        assert model.num_levels == 3  # Default value
        assert model.use_scalar_output is True  # Default value
        assert model.has_input_features is False  # Default value
    else:
        custom_hidden_c = [IN_C, 8, 8, 8, 8]
        model = _create_model(
            in_channels=IN_C,
            out_channels=2,
            kernel_size=5,
            hidden_channels=custom_hidden_c,
            num_levels=4,
            use_scalar_output=False,
            has_input_features=False,
        ).to(device)
        # Verify custom values
        assert model.in_channels == IN_C
        assert model.out_channels == 2
        assert model.hidden_channels == custom_hidden_c
        assert model.num_levels == 4
        assert model.use_scalar_output is False
        assert model.has_input_features is False

    # Common checks for both configurations
    assert isinstance(model, physicsnemo.Module)


@requires_module("torch_scatter")
def test_figconvunet_eval(pytestconfig, device):
    """Test FIGConvUNet evaluation mode produces consistent results."""
    torch.manual_seed(0)

    model = _create_model().to(device)
    assert isinstance(model, physicsnemo.Module)
    model.eval()

    batch_size = 1
    num_vertices = 100
    vertices = torch.randn((batch_size, num_vertices, 3), device=device)
    p_pred, c_d_pred = model(vertices)
    # Basic checks.
    assert p_pred.shape == (batch_size, num_vertices, OUT_C)
    # assert c_d_pred > 0

    # Run forward the second time, should be no changes.
    p_pred2, c_d_pred2 = model(vertices)

    assert_close(p_pred, p_pred2)
    assert_close(c_d_pred, c_d_pred2)


@requires_module("torch_scatter")
def test_figconvunet_forward(pytestconfig, device):
    """Test FIGConvUNet forward pass against reference output (MOD-008b)."""
    torch.manual_seed(0)

    if device == "cpu":
        pytest.skip("FigConvUNet is not reproducible between CPU vs. GPU.")

    model = _create_model().to(device)
    model.eval()

    batch_size = 1
    num_vertices = 100
    vertices = torch.randn((batch_size, num_vertices, 3), device=device)

    assert common.validate_forward_accuracy(
        model, (vertices,), file_name="models/figconvnet/data/figconvunet_output.pth"
    )


@requires_module("torch_scatter")
def test_figconvunet_checkpoint(device):
    """Test FIGConvUNet checkpoint save/load (MOD-008c)."""
    torch.manual_seed(0)

    # Construct two separate FIGConvUNet models
    model_1 = _create_model().to(device)
    model_2 = _create_model().to(device)

    batch_size = 1
    num_vertices = 100
    vertices = torch.randn((batch_size, num_vertices, 3), device=device)

    assert common.validate_checkpoint(model_1, model_2, (vertices,))


@requires_module("torch_scatter")
def test_figconvunet_output_shapes(device):
    """Test FIGConvUNet output shapes with various configurations."""
    torch.manual_seed(0)

    # Test with scalar output enabled (default)
    model = _create_model(use_scalar_output=True).to(device)
    model.eval()

    batch_size = 2
    num_vertices = 50
    vertices = torch.randn((batch_size, num_vertices, 3), device=device)
    p_pred, c_d_pred = model(vertices)

    assert p_pred.shape == (batch_size, num_vertices, OUT_C)
    assert c_d_pred.shape == (batch_size, 1)

    # Test with scalar output disabled
    model_no_scalar = _create_model(use_scalar_output=False).to(device)
    model_no_scalar.eval()

    p_pred2, c_d_pred2 = model_no_scalar(vertices)
    assert p_pred2.shape == (batch_size, num_vertices, OUT_C)
    assert c_d_pred2 is None
