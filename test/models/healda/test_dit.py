# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

from physicsnemo.experimental.models.healda import (
    HealDA,
    HPXPatchDetokenizer,
    HPXPatchTokenizer,
    ModelSensorConfig,
    SensorEmbedderConfig,
)

from .utils.obs_test_utils import create_unified_observation


def test_tokenizer():
    """Test HPXPatchTokenizer."""
    level_fine = 6
    level_coarse = 5
    hidden_size = 64
    n, t = 2, 1
    npix_fine = 12 * 4**level_fine
    npix_coarse = 12 * 4**level_coarse

    tokenizer = HPXPatchTokenizer(
        in_channels=10,
        hidden_size=hidden_size,
        level_fine=level_fine,
        level_coarse=level_coarse,
    )

    x = torch.randn(n, 10, t, npix_fine)
    doy = torch.ones([n, t])
    second = torch.ones([n, t])

    out = tokenizer(x, second_of_day=second, day_of_year=doy)
    # Output: (B, L, D) where L = T * npix_coarse
    assert out.shape == (n, t * npix_coarse, hidden_size)


def test_detokenizer():
    """Test HPXPatchDetokenizer."""
    level_fine = 6
    level_coarse = 5
    hidden_size = 64
    out_channels = 3
    n, t = 2, 1
    npix_fine = 12 * 4**level_fine
    npix_coarse = 12 * 4**level_coarse

    detokenizer = HPXPatchDetokenizer(
        hidden_size=hidden_size,
        out_channels=out_channels,
        level_coarse=level_coarse,
        level_fine=level_fine,
        time_length=t,
    )

    # Input: (B, L, D) where L = T * npix_coarse
    x = torch.randn(n, t * npix_coarse, hidden_size)
    c = torch.randn(n, hidden_size)  # conditioning vector

    out = detokenizer(x, c)
    # Output: (B, C_out, T, npix_fine)
    assert out.shape == (n, out_channels, t, npix_fine)


def test_detokenizer_vit_mode():
    """Test HPXPatchDetokenizer in VIT mode (c=zeros)."""
    level_fine = 6
    level_coarse = 5
    hidden_size = 64
    out_channels = 3
    n, t = 2, 1
    npix_fine = 12 * 4**level_fine
    npix_coarse = 12 * 4**level_coarse

    detokenizer = HPXPatchDetokenizer(
        hidden_size=hidden_size,
        out_channels=out_channels,
        level_coarse=level_coarse,
        level_fine=level_fine,
        time_length=t,
    )
    # Zero-init for VIT mode (scale=0, shift=0 -> identity modulation)
    detokenizer.initialize_weights()

    x = torch.randn(n, t * npix_coarse, hidden_size)

    # VIT mode: pass zeros for conditioning (with zero-init weights -> identity)
    c = torch.zeros(n, hidden_size)
    out = detokenizer(x, c)
    assert out.shape == (n, out_channels, t, npix_fine)


# ============================================================================
# HealDA Model Tests
# ============================================================================


@pytest.mark.parametrize("condition_dim", [None, 256])
def test_healda_forward(condition_dim, device):
    """Test HealDA model forward pass in VIT and Diffusion modes."""
    n, t = 1, 1
    level_in = 6
    level_model = 5
    in_channels = 3
    out_channels = 3
    npix = 12 * 4**level_in

    sensor_config = {
        "sensor_1": ModelSensorConfig(sensor_id=1, nchannel=4, platform_ids=(0, 1, 2)),
    }
    sensor_embedder_config = SensorEmbedderConfig(embed_dim=16, fusion_dim=32)

    model = HealDA(
        in_channels=in_channels,
        out_channels=out_channels,
        sensor_embedder_config=sensor_embedder_config,
        sensors=sensor_config,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        level_in=level_in,
        level_model=level_model,
        time_length=t,
        condition_dim=condition_dim,
        attention_backend="timm",  # Use timm for CPU testing
        layernorm_backend="torch",
    )
    model.to(device)

    # Create inputs
    x = torch.randn(n, in_channels, t, npix, device=device)
    timestep = torch.zeros(n, device=device)
    doy = torch.ones([n, t], device=device)
    second = torch.ones([n, t], device=device)

    # Create mock observation
    obs = create_unified_observation(
        nobs=100,
        batch_size=n,
        time_steps=t,
        meta_dim=28,
        hpx_level=level_in,
        device=device,
        sensor_config=sensor_config,
    )

    # Prepare conditioning args
    kwargs = {}
    if condition_dim is not None:
        kwargs["noise_labels"] = torch.ones(n, device=device)

    out = model(
        x,
        timestep,
        unified_obs=obs,
        day_of_year=doy,
        second_of_day=second,
        **kwargs,
    )

    # Verify output shape
    assert out.shape == (n, out_channels, t, npix)


def test_healda_backward(device):
    """Test HealDA model backward pass."""
    n, t = 1, 1
    level_in = 6
    level_model = 5
    in_channels = 3
    out_channels = 3
    npix = 12 * 4**level_in

    sensor_config = {
        "sensor_1": ModelSensorConfig(sensor_id=1, nchannel=4, platform_ids=(0,)),
    }
    sensor_embedder_config = SensorEmbedderConfig(embed_dim=16, fusion_dim=32)

    model = HealDA(
        in_channels=in_channels,
        out_channels=out_channels,
        sensor_embedder_config=sensor_embedder_config,
        sensors=sensor_config,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        level_in=level_in,
        level_model=level_model,
        time_length=t,
        condition_dim=None,  # VIT mode
        attention_backend="timm",
        layernorm_backend="torch",
    )
    model.to(device)

    x = torch.randn(n, in_channels, t, npix, device=device, requires_grad=True)
    timestep = torch.zeros(n, device=device)
    doy = torch.ones([n, t], device=device)
    second = torch.ones([n, t], device=device)

    obs = create_unified_observation(
        nobs=100,
        batch_size=n,
        time_steps=t,
        meta_dim=28,
        hpx_level=level_in,
        device=device,
        sensor_config=sensor_config,
    )

    out = model(
        x,
        timestep,
        unified_obs=obs,
        day_of_year=doy,
        second_of_day=second,
    )

    # Backward pass
    loss = out.sum()
    loss.backward()

    # Verify gradients exist
    assert x.grad is not None
    assert x.grad.shape == x.shape


@pytest.mark.parametrize("t", [1, 2])
def test_healda_time_length(t, device):
    """Test HealDA with different time lengths."""
    n = 1
    level_in = 6
    level_model = 5
    in_channels = 3
    out_channels = 3
    npix = 12 * 4**level_in

    sensor_config = {
        "sensor_1": ModelSensorConfig(sensor_id=1, nchannel=4, platform_ids=(0,)),
    }
    sensor_embedder_config = SensorEmbedderConfig(embed_dim=16, fusion_dim=32)

    model = HealDA(
        in_channels=in_channels,
        out_channels=out_channels,
        sensor_embedder_config=sensor_embedder_config,
        sensors=sensor_config,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        level_in=level_in,
        level_model=level_model,
        time_length=t,
        condition_dim=None,
        attention_backend="timm",
        layernorm_backend="torch",
    )
    model.to(device)

    x = torch.randn(n, in_channels, t, npix, device=device)
    timestep = torch.zeros(n, device=device)
    doy = torch.ones([n, t], device=device)
    second = torch.ones([n, t], device=device)

    obs = create_unified_observation(
        nobs=100,
        batch_size=n,
        time_steps=t,
        meta_dim=28,
        hpx_level=level_in,
        device=device,
        sensor_config=sensor_config,
    )

    out = model(
        x,
        timestep,
        unified_obs=obs,
        day_of_year=doy,
        second_of_day=second,
    )

    assert out.shape == (n, out_channels, t, npix)
