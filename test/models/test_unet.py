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

import pytest
import torch
from pytest_utils import import_or_fail

from . import common


@import_or_fail(["transformer_engine"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_unet_forward(device, pytestconfig):
    """Test unet forward pass"""
    from physicsnemo.models.unet import UNet

    torch.manual_seed(0)
    # Construct unet3d model
    model = UNet(
        in_channels=1,
        out_channels=1,
        model_depth=3,
        feature_map_channels=[8, 8, 16, 16, 32, 32],
        num_conv_blocks=2,
    ).to(device)

    bsize = 2
    invar = torch.randn(bsize, 1, 16, 16, 16).to(device)
    assert common.validate_forward_accuracy(model, (invar,))


@import_or_fail(["transformer_engine"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_unet_constructor(device, pytestconfig):
    """Test unet constructor options"""
    from physicsnemo.models.unet import UNet

    # Define dictionary of constructor args
    for depth in [2, 3]:
        if depth == 2:
            feature_map_channels = [2, 2, 4, 4]
        else:
            feature_map_channels = [2, 2, 4, 4, 8, 8]
        for pooling_type in ["MaxPool3d", "AvgPool3d"]:
            args = {
                "in_channels": random.randint(1, 3),
                "out_channels": random.randint(1, 3),
                "model_depth": depth,
                "feature_map_channels": feature_map_channels,
                "pooling_type": pooling_type,
            }
            # Construct model
            model = UNet(**args).to(device)
            bsize = random.randint(1, 16)
            invar = torch.randn(bsize, args["in_channels"], 8, 8, 8).to(device)
            outvar = model(invar)
            assert outvar.shape == (bsize, args["out_channels"], *invar.shape[2:])


# @pytest.mark.parametrize("device", ["cuda:0", "cpu"])
# def test_unet_optims(device):
#     """Test unet optimizations"""

#     def setup_model():
#         """Set up fresh model and inputs for each optim test"""
#         # Construct unet3d model
#         model = UNet(
#             in_channels=1,
#             out_channels=1,
#             model_depth=3,
#             feature_map_channels=[4, 4, 8, 8, 16, 16],
#             num_conv_blocks=2,
#         ).to(device)

#         bsize = 4
#         invar = torch.randn(bsize, 1, 16, 16, 16).to(device)
#         return model, invar

#     # Ideally always check graphs first
#     model, invar = setup_model()
#     assert common.validate_cuda_graphs(model, (invar,))
#     # Check JIT
#     model, invar = setup_model()
#     assert common.validate_jit(model, (invar,))
#     # Check AMP
#     model, invar = setup_model()
#     assert common.validate_amp(model, (invar,))
#     # Check Combo
#     model, invar = setup_model()
#     assert common.validate_combo_optims(model, (invar,))


# @pytest.mark.parametrize("device", ["cuda:0", "cpu"])
# def test_unet_checkpoint(device):
#     """Test unet checkpoint save/load"""
#     # Construct unet3d model
#     model_1 = UNet(
#         in_channels=1,
#         out_channels=1,
#         model_depth=3,
#         feature_map_channels=[8, 8, 16, 16, 32, 32],
#         num_conv_blocks=2,
#     ).to(device)

#     model_2 = UNet(
#         in_channels=1,
#         out_channels=1,
#         model_depth=3,
#         feature_map_channels=[8, 8, 16, 16, 32, 32],
#         num_conv_blocks=2,
#     ).to(device)

#     bsize = random.randint(1, 4)
#     invar = torch.randn(bsize, 1, 16, 16, 16).to(device)
#     assert common.validate_checkpoint(model_1, model_2, (invar,))


# @pytest.mark.parametrize("device", ["cuda:0", "cpu"])
# def test_unet_deploy(device):
#     """Test unet deployment support"""
#     # Construct unet3d model
#     model = UNet(
#         in_channels=1,
#         out_channels=1,
#         model_depth=3,
#         feature_map_channels=[8, 8, 16, 16, 32, 32],
#         num_conv_blocks=2,
#     ).to(device)

#     bsize = random.randint(1, 8)
#     invar = torch.randn(bsize, 1, 32, 32, 32).to(device)
#     assert common.validate_onnx_export(model, (invar,))
#     assert common.validate_onnx_runtime(model, (invar,), 1e-2, 1e-2)
