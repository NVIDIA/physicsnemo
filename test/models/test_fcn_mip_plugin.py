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

import datetime
import json
import os
import shutil

import numpy as np
import pytest
import torch
from pytest_utils import import_or_fail, nfsdata_or_fail

from physicsnemo.models.dlwp import DLWP
from physicsnemo.utils.filesystem import Package


@pytest.fixture
def dlwp_data_dir():
    """Data dir for dlwp package"""

    path = "/data/nfs/modulus-data/plugin_data/dlwp/"
    return path


def _copy_directory(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            _copy_directory(s, d)
        else:
            shutil.copy2(s, d)


def save_ddp_checkpoint(model, check_point_path, del_device_buffer=False):
    """Save checkpoint with similar structure to the training checkpoints

    The keys are prefixed with "module."
    """
    model_state = {f"module.{k}": v for k, v in model.state_dict().items()}
    if del_device_buffer:
        # This buffer is not present in some trained model checkpoints
        del model_state["module.device_buffer"]
    checkpoint = {"model_state": model_state}
    torch.save(checkpoint, check_point_path)


def save_checkpoint(model, check_point_path, del_device_buffer=False):
    """Save checkpoint with similar structure to the training checkpoints"""
    model_state = model.state_dict()
    if del_device_buffer:
        # This buffer is not present in some trained model checkpoints
        del model_state["module.device_buffer"]
    torch.save(model_state, check_point_path)


# def save_untrained_sfno(path):
#     """Function to save untrained SFNO"""

#     config = {
#         "N_in_channels": 2,
#         "N_out_channels": 1,
#         "img_shape_x": 4,
#         "img_shape_y": 5,
#         "scale_factor": 1,
#         "num_layers": 2,
#         "num_blocks": 2,
#         "embed_dim": 2,
#         "nettype": "sfno",
#         "add_zenith": True,
#     }
#     from physicsnemo.utils.sfno.YParams import ParamsBase

#     params = ParamsBase()
#     params.update_params(config)

#     from physicsnemo.models.sfno.sfnonet import SphericalFourierNeuralOperatorNet

#     model = SphericalFourierNeuralOperatorNet(params)

#     config_path = path / "config.json"
#     with config_path.open("w") as f:
#         json.dump(params.to_dict(), f)

#     check_point_path = path / "weights.tar"
#     save_ddp_checkpoint(model, check_point_path, del_device_buffer=True)

#     url = f"file://{path.as_posix()}"
#     package = Package(url, seperator="/")
#     return package


# @nfsdata_or_fail
# @import_or_fail(["dgl", "ruamel.yaml", "tensorly", "torch_harmonics", "tltorch"])
# def test_sfno(tmp_path, pytestconfig):
#     """Test SFNO plugin"""

#     from physicsnemo.models.fcn_mip_plugin import sfno

#     package = save_untrained_sfno(tmp_path)

#     model = sfno(package, pretrained=True)
#     x = torch.ones(1, 1, model.model.h, model.model.w)
#     time = datetime.datetime(2018, 1, 1)
#     with torch.no_grad():
#         out = model(x, time=time)

#     assert out.shape == x.shape


def save_untrained_dlwp(path):
    """Function to save untrained DLWP"""

    config = {
        "nr_input_channels": 18,
        "nr_output_channels": 14,
    }
    model = DLWP(
        nr_input_channels=config["nr_input_channels"],
        nr_output_channels=config["nr_output_channels"],
    )

    config_path = path / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f)

    check_point_path = path / "weights.pt"
    save_checkpoint(model, check_point_path, del_device_buffer=False)

    url = f"file://{path.as_posix()}"
    package = Package(url, seperator="/")
    return package


@nfsdata_or_fail
@import_or_fail(["dgl", "ruamel.yaml", "tensorly", "torch_harmonics", "tltorch"])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dlwp(tmp_path, batch_size, device, pytestconfig):

    from physicsnemo.models.fcn_mip_plugin import dlwp

    package = save_untrained_dlwp(tmp_path)
    source_dir = "/data/nfs/modulus-data/plugin_data/dlwp/"
    _copy_directory(source_dir, tmp_path)

    model = dlwp(package, pretrained=True).to(device)
    x = torch.ones(batch_size, 2, 7, 721, 1440).to(device)
    time = datetime.datetime(2018, 1, 1)
    with torch.no_grad():
        out = model(x, time)
    assert out.shape == x.shape


@nfsdata_or_fail
@import_or_fail(["dgl", "ruamel.yaml", "tensorly", "torch_harmonics", "tltorch"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test__CozZenWrapper(batch_size, pytestconfig):
    """Test Cosine Zenith wrapper"""

    from physicsnemo.models.fcn_mip_plugin import _CosZenWrapper

    class Id(torch.nn.Module):
        def forward(self, x):
            return x

    model = Id()
    nx, ny = (3, 4)
    lat = np.arange(nx)
    lon = np.arange(ny)

    x = torch.ones((batch_size, 1, nx, ny))
    time = datetime.datetime(2018, 1, 1)
    wrapper = _CosZenWrapper(model, lon, lat)
    wrapper(x, time=time)
