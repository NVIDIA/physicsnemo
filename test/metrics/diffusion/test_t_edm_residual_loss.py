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

from physicsnemo.models.diffusion import UNet
from physicsnemo.utils.patching import RandomPatching2D


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_residualloss_initialization(device):

    from physicsnemo.experimental.metrics.diffusion import tEDMResidualLoss

    # Mock regression network
    regression_net = torch.nn.Linear(1, 1).to(device)

    # Test default parameters
    loss_func = tEDMResidualLoss(
        regression_net=regression_net,
    )
    assert loss_func.P_mean == 0.0
    assert loss_func.P_std == 1.2
    assert loss_func.sigma_data == 0.5
    assert loss_func.hr_mean_conditioning is False
    assert loss_func.nu == 10

    # Test custom parameters
    loss_func = tEDMResidualLoss(
        regression_net=regression_net,
        P_mean=1.0,
        P_std=2.0,
        sigma_data=0.3,
        hr_mean_conditioning=True,
        nu=5,
    )
    assert loss_func.P_mean == 1.0
    assert loss_func.P_std == 2.0
    assert loss_func.sigma_data == 0.3
    assert loss_func.hr_mean_conditioning is True
    assert loss_func.nu == 5


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_residualloss_call_method(device):
    from physicsnemo.experimental.metrics.diffusion import tEDMResidualLoss

    def fake_residual_net(
        x,
        img_lr,
        sigma,
        labels=None,
        global_index=None,
        embedding_selector=None,
        augment_labels=None,
    ):
        return torch.zeros_like(x)

    # Mock regression network that returns scaled input
    class DummyRegNet(torch.nn.Module):
        def forward(self, x, *args, **kwargs):
            return 0.9 * x

    regression_net = DummyRegNet()
    regression_net.to(device)
    loss_func = tEDMResidualLoss(regression_net=regression_net, nu=10)

    # Create test inputs
    batch_size = 2
    channels = 3
    img_clean = torch.randn(batch_size, channels, 32, 32).to(device)
    img_lr = torch.randn(batch_size, channels, 32, 32).to(device)

    # Test without patching
    loss_value = loss_func(fake_residual_net, img_clean, img_lr)
    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.shape == (batch_size, channels, 32, 32)

    # Test with patching
    patch_num = 4
    patch_shape = (16, 16)
    patching = RandomPatching2D(
        img_shape=(32, 32), patch_shape=patch_shape, patch_num=patch_num
    )
    loss_value_with_patching = loss_func(
        fake_residual_net, img_clean, img_lr, patching=patching
    )
    assert isinstance(loss_value_with_patching, torch.Tensor)
    # Shape should be (batch_size * patch_num, channels, patch_shape_y, patch_shape_x)
    expected_shape = (batch_size * patch_num, channels, patch_shape[0], patch_shape[1])
    assert loss_value_with_patching.shape == expected_shape

    # Tests with patching accumulation
    loss_func.y_mean = None
    patch_nums_iter = [4, 4, 4, 2]
    patch_shape = (16, 16)
    for patch_num in patch_nums_iter:
        patching = RandomPatching2D(
            img_shape=(32, 32), patch_shape=patch_shape, patch_num=patch_num
        )
        loss_value_with_patching = loss_func(
            fake_residual_net,
            img_clean,
            img_lr,
            patching=patching,
            use_patch_grad_acc=True,
        )
        assert isinstance(loss_value_with_patching, torch.Tensor)
        # Shape should be (batch_size * patch_num, channels, patch_shape_y, patch_shape_x)
        expected_shape = (
            batch_size * patch_num,
            channels,
            patch_shape[0],
            patch_shape[1],
        )
        assert loss_value_with_patching.shape == expected_shape

    # Test error on invalid patching object
    with pytest.raises(ValueError):
        loss_func(
            fake_residual_net, img_clean, img_lr, patching="invalid patching object"
        )


# More realistic test with a UNet model
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_call_method_residualloss_with_unet(device):

    from physicsnemo.experimental.metrics.diffusion import tEDMResidualLoss
    from physicsnemo.experimental.models.diffusion.preconditioning import (
        tEDMPrecondSuperRes,
    )

    res, inc, outc = 64, 2, 3
    N_pos = 2
    regression_model = UNet(
        img_resolution=res,
        img_in_channels=inc + N_pos,
        img_out_channels=outc,
        model_type="SongUNetPosEmbd",
        N_grid_channels=N_pos,
        gridtype="test",
    ).to(device)
    diffusion_model = tEDMPrecondSuperRes(
        img_resolution=res,
        img_in_channels=inc + N_pos,
        img_out_channels=outc,
        model_type="SongUNetPosEmbd",
        N_grid_channels=N_pos,
        gridtype="test",
        nu=10,
    ).to(device)

    img_clean = torch.ones([1, outc, res, res]).to(device)
    img_lr = torch.randn([1, inc, res, res]).to(device)

    # Without hr_mean_conditioning
    loss_func = tEDMResidualLoss(
        regression_net=regression_model, hr_mean_conditioning=False, nu=10
    )
    loss_value = loss_func(diffusion_model, img_clean, img_lr)
    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.shape == img_clean.shape


# Test with UNets and hr_mean_conditioning
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_call_method_residualloss_with_unet_hr_mean_conditioning(device):

    from physicsnemo.experimental.metrics.diffusion import tEDMResidualLoss
    from physicsnemo.experimental.models.diffusion.preconditioning import (
        tEDMPrecondSuperRes,
    )

    res, inc, outc = 64, 2, 3
    N_pos = 2
    regression_model = UNet(
        img_resolution=res,
        img_in_channels=inc + N_pos,
        img_out_channels=outc,
        model_type="SongUNetPosEmbd",
        N_grid_channels=N_pos,
        gridtype="test",
    ).to(device)
    diffusion_model = tEDMPrecondSuperRes(
        img_resolution=res,
        img_in_channels=inc + N_pos + outc,
        img_out_channels=outc,
        model_type="SongUNetPosEmbd",
        N_grid_channels=N_pos,
        gridtype="test",
        nu=10,
    ).to(device)

    img_clean = torch.ones([1, outc, res, res]).to(device)
    img_lr = torch.randn([1, inc, res, res]).to(device)

    # With hr_mean_conditioning
    loss_func = tEDMResidualLoss(
        regression_net=regression_model,
        hr_mean_conditioning=True,
        nu=10,
    )
    loss_value = loss_func(diffusion_model, img_clean, img_lr)
    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.shape == img_clean.shape


# Test with UNets, hr_mean_conditioning, and lead-time aware embedding
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_call_method_residualloss_with_lt_unet_hr_mean_conditioning(device):

    from physicsnemo.experimental.metrics.diffusion import tEDMResidualLoss
    from physicsnemo.experimental.models.diffusion.preconditioning import (
        tEDMPrecondSuperRes,
    )

    res, inc, outc = 64, 2, 3
    N_pos, lead_time_channels = 2, 4
    prob_channels = [0, 2]
    regression_model = UNet(
        img_resolution=res,
        img_in_channels=inc + N_pos + lead_time_channels,
        img_out_channels=outc,
        model_type="SongUNetPosLtEmbd",
        N_grid_channels=N_pos,
        gridtype="test",
        lead_time_channels=lead_time_channels,
        prob_channels=prob_channels,
    ).to(device)
    diffusion_model = tEDMPrecondSuperRes(
        img_resolution=res,
        img_in_channels=inc + outc + N_pos + lead_time_channels,
        img_out_channels=outc,
        model_type="SongUNetPosLtEmbd",
        N_grid_channels=N_pos,
        gridtype="test",
        lead_time_channels=lead_time_channels,
        prob_channels=prob_channels,
        nu=10,
    ).to(device)

    img_clean = torch.ones([1, outc, res, res]).to(device)
    img_lr = torch.randn([1, inc, res, res]).to(device)
    lead_time_label = torch.tensor(8).to(device)

    # With hr_mean_conditioning
    loss_func = tEDMResidualLoss(
        regression_net=regression_model,
        hr_mean_conditioning=True,
        nu=10,
    )
    loss_value = loss_func(
        diffusion_model, img_clean, img_lr, lead_time_label=lead_time_label
    )
    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.shape == img_clean.shape
