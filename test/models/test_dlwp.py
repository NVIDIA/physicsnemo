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

import random

import numpy as np
import pytest
import torch

from physicsnemo.models.dlwp import DLWP

from . import common


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_dlwp_forward(device):
    """Test DLWP forward pass"""
    torch.manual_seed(0)
    # Construct model
    model = DLWP(
        nr_input_channels=2,
        nr_output_channels=2,
        nr_initial_channels=64,
        activation_fn="leaky_relu",
        depth=2,
        clamp_activation=(None, 10.0),
    ).to(device)

    bsize = 4
    invar = torch.randn(bsize, 2, 6, 64, 64).to(device)
    assert common.validate_forward_accuracy(
        model, (invar,), file_name="dlwp_output.pth", atol=1e-3
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("nr_input_channels", [2])
@pytest.mark.parametrize("nr_output_channels", [2])
@pytest.mark.parametrize("nr_initial_channels", [32])
@pytest.mark.parametrize("depth", [2])
def test_dlwp_constructor(
    device, nr_input_channels, nr_output_channels, nr_initial_channels, depth
):
    """Test DLWP constructor options"""

    # Construct model
    model = DLWP(
        nr_input_channels=nr_input_channels,
        nr_output_channels=nr_output_channels,
        nr_initial_channels=nr_initial_channels,
        depth=depth,
    ).to(device)

    bsize = random.randint(1, 4)
    invar = torch.randn(bsize, nr_input_channels, 6, 128, 128).to(device)
    outvar = model(invar)
    assert outvar.shape == (bsize, nr_output_channels, *invar.shape[2:])


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_dlwp_optims(device):
    """Test DLWP optimizations"""

    def setup_model():
        """Setups up fresh DLWP model and inputs for each optim test"""
        model = DLWP(
            nr_input_channels=2,
            nr_output_channels=2,
        ).to(device)

        bsize = random.randint(1, 5)
        invar = torch.randn(bsize, 2, 6, 16, 16).to(device)

        return model, invar

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (invar,))

    # Check JIT
    model, invar = setup_model()
    assert common.validate_jit(model, (invar,))
    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (invar,))
    # Check Combo
    model, invar = setup_model()
    assert common.validate_combo_optims(model, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_dlwp_checkpoint(device):
    """Test DLWP checkpoint save/load"""
    # Construct DLWP models
    model_1 = DLWP(
        nr_input_channels=2,
        nr_output_channels=2,
    ).to(device)

    model_2 = DLWP(
        nr_input_channels=2,
        nr_output_channels=2,
    ).to(device)

    bsize = random.randint(1, 2)
    invar = torch.randn(bsize, 2, 6, 64, 64).to(device)

    assert common.validate_checkpoint(model_1, model_2, (invar,))


common.check_ort_version()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_dlwp_deploy(device):
    """Test DLWP deployment support"""
    # Construct DLWP model
    model = DLWP(
        nr_input_channels=2,
        nr_output_channels=2,
    ).to(device)

    bsize = random.randint(1, 2)
    invar = torch.randn(bsize, 2, 6, 64, 64).to(device)

    assert common.validate_onnx_export(model, (invar,))
    assert common.validate_onnx_runtime(model, (invar,))


def test_dlwp_implementation():
    """Test DLWP implementation compared to publication"""

    model = DLWP(16, 12, 64, depth=2)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    assert params == 2676376
