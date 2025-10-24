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

from physicsnemo.models.module import Module


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_EDMPrecondSuperResolution_forward(device):
    b, c_target, x, y = 1, 3, 8, 8
    c_cond = 4

    from physicsnemo.experimental.models.diffusion.preconditioning import (
        tEDMPrecondSuperRes,
    )

    # Create an instance of the preconditioner
    model = tEDMPrecondSuperRes(
        img_resolution=x,
        img_in_channels=c_cond,
        img_out_channels=c_target,
        use_fp16=False,
        model_type="SongUNet",
        nu=10,
    ).to(device)

    latents = torch.ones((b, c_target, x, y)).to(device)
    img_lr = torch.arange(b * c_cond * x * y).reshape((b, c_cond, x, y)).to(device)
    sigma = torch.tensor([10.0]).to(device)

    # Forward pass
    output = model(
        x=latents,
        img_lr=img_lr,
        sigma=sigma,
    )

    # Assert the output shape is correct
    assert output.shape == (b, c_target, x, y)


@import_or_fail("termcolor")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_EDMPrecondSuperResolution_serialization(tmp_path, pytestconfig, device):
    from physicsnemo.experimental.models.diffusion.preconditioning import (
        tEDMPrecondSuperRes,
    )
    from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

    module = tEDMPrecondSuperRes(8, 1, 1, nu=10).to(device)
    model_path = tmp_path / "output.mdlus"
    module.save(model_path.as_posix())
    loaded = Module.from_checkpoint(model_path.as_posix())
    assert isinstance(loaded, tEDMPrecondSuperRes)
    save_checkpoint(path=tmp_path, models=module, epoch=1)
    epoch = load_checkpoint(path=tmp_path)
    assert epoch == 1
