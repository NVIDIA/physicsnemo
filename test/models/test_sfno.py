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

import physicsnemo
from physicsnemo.registry import ModelRegistry

from . import common

IN_OUT_SHAPE = [32, 32]
INP_CHANS = 2


def _create_model() -> physicsnemo.Module:
    registry = ModelRegistry()
    sfno_type = registry.factory("SFNO")

    return sfno_type(
        inp_shape=IN_OUT_SHAPE,
        out_shape=IN_OUT_SHAPE,
        inp_chans=INP_CHANS,
        out_chans=1,
        embed_dim=16,
    )


@import_or_fail("makani")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_sfno_forward(pytestconfig, device):
    """Test SFNO forward pass."""

    device = torch.device(device)

    torch.manual_seed(0)

    model = _create_model().to(device)
    assert isinstance(model, physicsnemo.Module)

    bsize = 2
    invar = torch.randn(bsize, INP_CHANS, *IN_OUT_SHAPE).to(device)

    # Check output size.
    # Use different checkpoints for different device types due to
    # SFNO implementation differences CPU vs GPU.
    model_file_name = f"{model.meta.name}_{device.type}_output.pth"
    assert common.validate_forward_accuracy(
        model, (invar,), file_name=model_file_name, atol=0.01
    )


@import_or_fail("makani")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_sfno_checkpoint(pytestconfig, device):
    """Test SFNO checkpoint save/load."""

    torch.manual_seed(0)

    # Construct SFNO models.
    model_1 = _create_model().to(device)
    model_2 = _create_model().to(device)

    bsize = 2
    invar = torch.randn(bsize, INP_CHANS, *IN_OUT_SHAPE).to(device)

    assert common.validate_checkpoint(model_1, model_2, (invar,))
