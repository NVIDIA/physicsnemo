# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from physicsnemo.models.healda import (
    DiT,
    HPXPatchDecode,
    HPXPatchEmbed,
    ModelSensorConfig,
    SensorEmbedderConfig,
    Subdomain,
    UnifiedObservation,
)

from .utils.obs_test_utils import create_unified_observation


def test_patch_decode():
    level_fine = 6
    level_coarse = 4
    c_out = 12
    decode = HPXPatchDecode(
        in_channels=5,
        out_channels=c_out,
        level_coarse=level_coarse,
        level_fine=level_fine,
    )
    x = torch.randn(1, 1, 12 * 16 * 16, 5)
    out = decode(x)
    assert out.shape == (1, c_out, 1, 12 * 4**level_fine)


def test_patch_embed():
    n = 1
    t = 1
    level_fine = 6
    level_coarse = 4
    c_out = 12
    embed = HPXPatchEmbed(
        in_channels=5,
        out_channels=c_out,
        level_coarse=level_coarse,
        level_fine=level_fine,
    )
    doy = torch.ones([n, t])
    second = torch.ones(
        [n, t],
    )
    x = torch.randn(n, 5, t, 12 * 4**level_fine)
    out = embed(x, second_of_day=second, day_of_year=doy)
    assert out.shape == (n, t, 12 * 4**level_coarse, c_out)


@pytest.mark.parametrize("checkpoint", [0, 1, 2])
def test_dit(checkpoint):
    torch.manual_seed(0)
    net = DiT(
        num_layers=1,
        level_in=4,
        level_model=4,
        in_channels=3,
        out_channels=3,
        label_dim=0,
    )
    net.gradient_checkpointing = checkpoint
    device = "cuda"
    net.to(device)

    noise_labels = torch.zeros([1], device=device)
    class_labels = torch.empty([1, 0], device=device).int()
    n, t = 1, 1
    img = torch.ones(n, 3, t, net.domain.numel(), device=device).to(
        memory_format=torch.channels_last
    )
    doy = torch.ones([n, t], device=device)
    second = torch.ones([n, t], device=device)

    out = net(
        img,
        noise_labels,
        class_labels=class_labels,
        day_of_year=doy,
        second_of_day=second,
    )
    assert out.out.shape == (n, 3, t, net.domain.numel())

    out.out.sum().backward()


def test_dit_localize():
    net = DiT(
        num_layers=1,
        level_in=4,
        num_attention_heads=2,
        level_model=4,
        in_channels=3,
        out_channels=3,
    )
    device = "cuda"
    net.to(device)

    noise_labels = torch.ones([1], device=device)
    class_labels = torch.empty([1, 0], device=device).int()
    n, t = 1, 1
    img = torch.ones(n, 3, t, net.domain.numel(), device=device).to(
        memory_format=torch.channels_last
    )
    doy = torch.ones([n, t], device=device)
    second = torch.ones([n, t], device=device)

    out = net(
        img,
        noise_labels,
        class_labels=class_labels,
        day_of_year=doy,
        second_of_day=second,
        level_localize=3,
    )
    assert out.out.shape == (n, 3, t, net.domain.numel())


def create_mock_unified_observation(
    nobs=507586, batch_size=2, device="cuda", time_length=2
):
    """Create a mock UnifiedObservation with realistic data shapes and ranges."""
    # Based on the debug script output:
    # - nobs=507586 observations total
    # - batch_dims=torch.Size([2, 1])
    # - float_metadata.shape[-1] = 28 (meta_dim)
    # - hpx_level = 7
    # - obs values in range roughly [-1, 1]
    # - time values are large integers (nanoseconds since epoch)
    # - int_metadata contains [sensor_id, pix, channel, platform, obs_type, global_channel_id]

    # Handle empty observation case
    if nobs == 0:
        offsets_3d = torch.zeros(
            (1, batch_size, time_length), dtype=torch.long, device=device
        )
        sensor_id_to_local = torch.tensor([0], dtype=torch.long, device=device)
        return UnifiedObservation(
            obs=torch.empty(0, device=device),
            time=torch.empty(0, dtype=torch.long, device=device),
            float_metadata=torch.empty((0, 28), device=device),
            int_metadata=torch.empty((0, 6), dtype=torch.long, device=device),
            offsets=offsets_3d,
            sensor_id_to_local=sensor_id_to_local,
            hpx_level=7,
        )

    # Observation values - roughly normal distribution around 0
    obs = torch.randn(nobs, device=device) * 0.5

    # Time values - nanoseconds since epoch (realistic range)
    base_time = 946674000000000000  # From debug output
    time = torch.randint(
        base_time,
        base_time + 24 * 3600 * 1000000000,  # 24 hour range in nanoseconds
        (nobs,),
        device=device,
        dtype=torch.long,
    )

    # Float metadata - 28 features with values roughly in [-1, 1] range
    float_metadata = torch.randn(nobs, 28, device=device) * 0.8

    # HEALPix pixel indices for level 7 (12 * 4^7 = 196608 pixels)
    max_pix = 12 * (4**7)
    pix = torch.randint(0, max_pix, (nobs,), device=device, dtype=torch.long)

    # sensor, pix, local_channel, platform, observation_type, global_channel_id = obs.int_metadata
    sensor_id = torch.zeros(nobs, dtype=torch.long, device=device)
    zero = torch.zeros(nobs, dtype=torch.long, device=device)
    int_metadata = torch.stack([sensor_id, pix, zero, zero, zero, zero], dim=1)

    # Offsets for batching - split observations between 2 batches
    # Create 3D offsets: (n_sensors, batch, time)
    # We have 1 sensor (id=0), so shape is (1, batch_size, time_length)
    nobs_per_frame = nobs // (time_length * batch_size)
    offsets_flat = (
        torch.arange(time_length * batch_size, device=device) * nobs_per_frame
    )
    offsets_2d = offsets_flat.reshape(batch_size, time_length)
    offsets_3d = offsets_2d.unsqueeze(0)  # Add sensor dimension: (1, batch, time)

    # Create sensor_id_to_local mapping for single sensor (id=0)
    sensor_id_to_local = torch.tensor([0], dtype=torch.long, device=device)

    return UnifiedObservation(
        obs=obs,
        time=time,
        float_metadata=float_metadata,
        int_metadata=int_metadata,
        offsets=offsets_3d,
        sensor_id_to_local=sensor_id_to_local,
        hpx_level=7,
    )


@pytest.mark.parametrize("t", [1, 2])
@pytest.mark.parametrize("autocast", [True, False])
def test_dit_with_observations(t, device, autocast):
    """Test DiT model forward pass with observation data."""
    n = 1

    # Create sensor_embedder_config
    sensor_config = {
        "sensor_1": ModelSensorConfig(
            sensor_id=1,
            nchannel=8,
            platform_ids=tuple(range(10)),
        ),
        "sensor_2": ModelSensorConfig(
            sensor_id=4,
            nchannel=4,
            platform_ids=tuple(range(5)),
        ),
    }

    sensor_embedder_config = SensorEmbedderConfig(
        embed_dim=16,
    )

    obs = create_unified_observation(
        nobs=1000,
        batch_size=n,
        time_steps=t,
        meta_dim=28,
        hpx_level=7,
        n_embed=1024,
        device=device,
        sensor_config=sensor_config,
    )

    # DiT model configuration
    npix = 12 * 4**6  # level 6 HEALPix grid

    model = DiT(
        embed_v2=True,
        embed_v2_meta_dim=28,  # matches static_metadata.shape[-1]
        embed_v2_n_embed=1024,
        embed_v2_in_level=obs.hpx_level,  # 7
        sensor_embedder_config=sensor_embedder_config,
        sensors=sensor_config,
        num_layers=2,
        level_in=6,
        time_length=t,
        level_model=4,
        in_channels=3,
        out_channels=3,
        temporal_attention=True,
    )
    model.to(device)

    # Input tensors from debug script
    noise_labels = torch.ones([n], device=device)
    class_labels = torch.empty([n, 0], device=device).int()
    img = torch.ones(n, 3, t, npix, device=device).to(memory_format=torch.channels_last)
    doy = torch.ones([n, t], device=device)
    second = torch.ones([n, t], device=device)

    # Forward pass - this should work without real data dependencies
    with torch.autocast(device, torch.bfloat16, enabled=autocast):
        out = model(
            img,
            noise_labels,
            class_labels=class_labels,
            day_of_year=doy,
            second_of_day=second,
            unified_obs=obs,
        )

    # Verify output shape
    assert out.out.shape == (n, 3, t, npix)

    # Verify observation processing worked (model should handle the obs without errors)
    assert obs.batch_dims == (n, t)
    assert obs.float_metadata.shape[-1] == 28
    assert obs.hpx_level == 7

    # Verify the correct embedding module was initialized based on encoder type
    assert model.embed_v2_patch is not None


def test_subdomain_dit():
    """Test DiT with subdomain argument for regional processing."""
    device = "cuda"
    level_in = 6
    level_model = 4

    # Create DiT model
    net = DiT(
        num_layers=1,
        level_in=level_in,
        level_model=level_model,
        num_attention_heads=2,
        in_channels=3,
        out_channels=3,
    )
    net.to(device)

    # Create subdomain - single face with size 32x32
    # This represents a regional domain instead of full global coverage
    subdomain = Subdomain(
        x=torch.tensor([[0]], device=device),
        y=torch.tensor([[0]], device=device),
        f=torch.tensor([[0]], device=device),
        n=32,
        level=level_in,
    )

    # Prepare inputs matching subdomain size
    n, t = 1, 1
    subdomain_npix = subdomain.n**2 * subdomain.num_faces  # 32*32*1 = 1024

    noise_labels = torch.ones([n], device=device)
    class_labels = torch.empty([n, 0], device=device).int()
    img = torch.ones(n, 3, t, subdomain_npix, device=device).to(
        memory_format=torch.channels_last
    )
    doy = torch.ones([n, t], device=device)
    second = torch.ones([n, t], device=device)

    # Forward pass with subdomain
    out = net(
        img,
        noise_labels,
        class_labels=class_labels,
        day_of_year=doy,
        second_of_day=second,
        subdomain=subdomain,
    )

    # Verify output shape matches subdomain size
    assert out.out.shape == (n, 3, t, subdomain_npix)


def test_dit_obs_decoder(device):
    """Test DiT with obs_decoder=True returns observation predictions."""
    n, t = 1, 1
    obs_level = 8  # ObsDecoder requires hpx_fine_level=8

    # Create unified observation
    obs = create_unified_observation(
        nobs=1000,
        batch_size=n,
        time_steps=t,
        meta_dim=28,
        hpx_level=obs_level,
        n_embed=1024,
        device=device,
    )

    # Create DiT with obs_decoder enabled
    model = DiT(
        obs_decoder=True,  # Enable obs decoder
        num_attention_heads=2,
        num_layers=1,
        level_in=6,
        level_model=4,
        in_channels=3,
        out_channels=3,
    )
    model.to(device)

    # Forward pass inputs
    noise_labels = torch.ones([n], device=device)
    class_labels = torch.empty([n, 0], device=device).int()
    img = torch.ones(n, 3, t, 12 * 4**6, device=device)
    doy = torch.ones([n, t], device=device)
    second = torch.ones([n, t], device=device)

    # Forward pass
    out = model(
        img,
        noise_labels,
        class_labels=class_labels,
        day_of_year=doy,
        second_of_day=second,
        unified_obs=obs,
        level_localize=3,
    )

    # Check that obs decoder output is present
    assert out.obs is not None
    assert out.obs.shape[0] == obs.obs.shape[0]  # Should match number of observations
