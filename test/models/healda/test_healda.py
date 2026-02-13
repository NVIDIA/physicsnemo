# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch

from physicsnemo.experimental.models.healda import HealDA
from test import common
from test.conftest import requires_module

from .test_point_embed import _build_flattened_obs


def _setup_healda(
    diffusion_conditioning=True, condition_dim=0, time_length=1, device=None
):
    """Create a small HealDA model and matching input args tuple.

    Returns ``(model, in_args)``
    """
    in_channels = 2
    out_channels = 8
    level_in = 4
    level_model = 3
    npix = 12 * 4**level_in
    nsensors = 2
    nchannel_per_sensor = [4, 3]
    nplatform_per_sensor = [3, 2]
    batch_size = 1
    meta_dim = 28

    model = HealDA(
        in_channels=in_channels,
        out_channels=out_channels,
        nchannel_per_sensor=nchannel_per_sensor,
        nplatform_per_sensor=nplatform_per_sensor,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        level_in=level_in,
        level_model=level_model,
        time_length=time_length,
        embed_dim=16,
        fusion_dim=32,
        meta_dim=meta_dim,
        diffusion_conditioning=diffusion_conditioning,
        condition_dim=condition_dim,
    )
    if device is not None:
        model = model.to(device)

    counts = [
        [[100] * time_length] for _ in range(batch_size) for _ in range(nsensors)
    ]  # (s=2, b=1, t=time_length)
    obs, float_metadata, pix, local_channel, local_platform, obs_type, offsets = (
        _build_flattened_obs(
            counts,
            nchannel_per_sensor=nchannel_per_sensor,
            nplatform_per_sensor=nplatform_per_sensor,
            npix=npix,
            meta_dim=meta_dim,
            device=device,
        )
    )

    in_args = (
        torch.randn(
            batch_size,
            in_channels,
            time_length,
            npix,
        ).to(device),
        torch.zeros(batch_size).to(device),
        obs,
        float_metadata,
        pix,
        local_channel,
        local_platform,
        obs_type,
        offsets,
        torch.ones(batch_size, time_length).to(device),
        torch.ones(batch_size, time_length).to(device),
    )
    if diffusion_conditioning:
        in_args = in_args + (torch.randn(batch_size, condition_dim).to(device),)

    return model, in_args


@requires_module("earth2grid")
@pytest.mark.parametrize(
    "diffusion_conditioning, condition_dim, ref_file",
    [
        (False, 0, "healda_zero_conditioning_output.pth"),
        (True, 16, "healda_conditional_output.pth"),
    ],
)
def test_healda_forward_accuracy(
    diffusion_conditioning,
    condition_dim,
    ref_file,
    device,
):
    """Test HealDA forward pass against a saved reference output."""
    torch.manual_seed(0)
    model, in_args = _setup_healda(
        diffusion_conditioning=diffusion_conditioning,
        condition_dim=condition_dim,
        device=device,
    )
    model.eval()

    assert common.validate_forward_accuracy(
        model,
        in_args,
        file_name=f"models/healda/data/{ref_file}",
    )


@requires_module("earth2grid")
def test_healda_checkpoint(device):
    """Test that checkpoint save/load reproduces identical outputs."""
    torch.manual_seed(0)
    model_1, _ = _setup_healda(device=device)
    model_1.eval()

    torch.manual_seed(42)
    model_2, _ = _setup_healda(device=device)
    model_2.eval()

    # Change weights on one model to ensure they are different initially
    with torch.no_grad():
        for param in model_2.parameters():
            param.add_(0.1)

    torch.manual_seed(0)
    _, in_args = _setup_healda(device=device)

    assert common.validate_checkpoint(model_1, model_2, in_args)


@requires_module("earth2grid")
@pytest.mark.parametrize("t", [1, 2])
def test_healda_time_length(t, device):
    """Test HealDA with different time lengths."""
    torch.manual_seed(0)
    model, in_args = _setup_healda(time_length=t, device=device)
    model.eval()

    x = in_args[0]
    out = model(*in_args)

    B, _, T, npix = x.shape
    assert out.shape == (B, model.out_channels, T, npix)
    assert torch.isfinite(out).all()
