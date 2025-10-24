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

import torch
from pytest_utils import import_or_fail
from torch.testing import assert_close

import physicsnemo

from . import common

IN_C = 3
OUT_C = 1
KERNEL_SIZE = 9
HIDDEN_C = [IN_C, 4, 4, 4]
MLP_C = [8, 8]


def _create_model() -> physicsnemo.Module:
    from physicsnemo.models.figconvnet.figconvunet import FIGConvUNet

    return FIGConvUNet(
        IN_C,
        OUT_C,
        KERNEL_SIZE,
        HIDDEN_C,
        mlp_channels=MLP_C,
    )


@import_or_fail("webdataset")
def test_figconvunet_eval(pytestconfig):
    # FIGConvUNet works only on GPUs due to Warp.
    device = torch.device("cuda:0")

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
    assert c_d_pred > 0

    # Run forward the second time, should be no changes.
    p_pred2, c_d_pred2 = model(vertices)

    assert_close(p_pred, p_pred2)
    assert_close(c_d_pred, c_d_pred2)


@import_or_fail("webdataset")
def test_figconvunet_forward(pytestconfig):
    # FIGConvUNet works only on GPUs due to Warp.
    device = torch.device("cuda:0")

    torch.manual_seed(0)

    model = _create_model().to(device)
    model.eval()

    batch_size = 1
    num_vertices = 100
    vertices = torch.randn((batch_size, num_vertices, 3), device=device)

    assert common.validate_forward_accuracy(model, (vertices,))
