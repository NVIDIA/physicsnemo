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

from physicsnemo.models.healda import (
    ModelSensorConfig,
    MultiSensorObsEmbedding,
    SensorEmbedderConfig,
    UnifiedObservation,
)
from physicsnemo.models.healda.obs_embedding import SensorEmbedder

from .utils.obs_test_utils import create_unified_observation

# ============================================================================
# Test Utilities
# ============================================================================


def check_all_params_have_gradients(model: torch.nn.Module) -> tuple[bool, list[str]]:
    """
    Check that all parameters in a model have gradients.

    Returns:
        (all_have_grads, params_without_grads)
    """
    params_without_grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            params_without_grads.append(name)

    return len(params_without_grads) == 0, params_without_grads


# ============================================================================
# SensorEmbedder Tests
# ============================================================================


@pytest.mark.parametrize("nobs", [0, 100])
def test_sensor_embedder(nobs):
    """Test SensorEmbedder shapes, values, and gradients (includes empty obs for DDP compatibility)."""
    torch.manual_seed(42)

    sensor_embed_dim = 16
    output_dim = 32
    hpx_level = 5
    meta_dim = 8
    nchannel = 10
    nplatform = 5
    batch_size = 2

    embedder = SensorEmbedder(
        platform_ids=list(
            range(nplatform)
        ),  # Simple case: platform IDs = [0, 1, ..., nplatform-1]
        sensor_embed_dim=sensor_embed_dim,
        output_dim=output_dim,
        meta_dim=meta_dim,
        n_embed=100,
        hpx_level=hpx_level,
        nchannel=nchannel,
    )
    embedder.train()

    # Create observation
    obs = create_unified_observation(
        nobs=nobs,
        batch_size=batch_size,
        time_steps=1,
        hpx_level=hpx_level + 1,
        meta_dim=meta_dim,
        nchannel=nchannel,
        nplatform=nplatform,
        n_embed=100,
    )

    # Forward pass
    output = embedder(obs)

    # Check output shape and values
    npix = 12 * 4**hpx_level
    assert output.shape == (batch_size, 1, npix, output_dim)
    assert torch.isfinite(output).all()

    # Check gradients
    loss = output.sum()
    loss.backward()
    all_have_grads, missing = check_all_params_have_gradients(embedder)
    assert all_have_grads, f"Parameters without gradients (nobs={nobs}): {missing}"


def test_sensor_embedder_no_batching():
    """Test SensorEmbedder with offsets=None (no explicit batching)."""
    torch.manual_seed(42)

    sensor_embed_dim = 16
    output_dim = 128
    hpx_level = 5
    nobs = 50
    nplatform = 5

    embedder = SensorEmbedder(
        platform_ids=list(range(nplatform)),
        sensor_embed_dim=sensor_embed_dim,
        output_dim=output_dim,
        meta_dim=8,
        n_embed=100,
        hpx_level=hpx_level,
    )

    # Create observation without offsets
    obs = create_unified_observation(
        nobs=nobs,
        batch_size=1,
        time_steps=1,
        hpx_level=hpx_level + 1,
        meta_dim=8,
        n_embed=100,
    )
    obs = UnifiedObservation(
        obs=obs.obs,
        time=obs.time,
        float_metadata=obs.float_metadata,
        int_metadata=obs.int_metadata,
        offsets=None,  # No explicit batching
        hpx_level=obs.hpx_level,
    )

    # Forward pass
    with torch.no_grad():
        output = embedder(obs)

    # Check output shape (should be 2D: npix x output_dim)
    npix = 12 * 4**hpx_level
    assert output.shape == (npix, output_dim)


# ============================================================================
# MultiSensorObsEmbedding Tests
# ============================================================================


@pytest.mark.parametrize("num_sensors", [1, 2])
def test_multisensor_obs_embedding(num_sensors):
    """Test MultiSensorObsEmbedding with different sensor counts."""
    torch.manual_seed(42)

    sensor_embed_dim = 16
    fusion_dim = 32
    hpx_level = 5
    meta_dim = 8

    # Build sensor configs
    all_sensor_configs = {
        "test_sensor_0": ModelSensorConfig(
            sensor_id=0, nchannel=10, platform_ids=tuple(range(5))
        ),
        "test_sensor_1": ModelSensorConfig(
            sensor_id=1, nchannel=10, platform_ids=tuple(range(5))
        ),
    }
    sensor_config = dict(list(all_sensor_configs.items())[:num_sensors])

    sensor_embedder_config = SensorEmbedderConfig(
        embed_dim=sensor_embed_dim,
        meta_dim=meta_dim,
        fusion_dim=fusion_dim,
    )

    embedder = MultiSensorObsEmbedding(
        sensor_embedder_config=sensor_embedder_config,
        sensors=sensor_config,
        hpx_level=hpx_level,
    )

    # Create observation
    obs = create_unified_observation(
        nobs=100,
        batch_size=2,
        time_steps=1,
        hpx_level=hpx_level + 1,
        meta_dim=meta_dim,
        sensor_config=sensor_config,
        n_embed=100,
        ensure_all_sensors=True,  # Ensure all obs are assigned to defined sensors
    )

    # Forward pass
    with torch.no_grad():
        output = embedder(obs)

    # Check output shape and values
    npix = 12 * 4**hpx_level
    assert output.shape == (2, fusion_dim, 1, npix)
    assert torch.isfinite(output).all()
    assert not torch.allclose(output, torch.zeros_like(output))


@pytest.mark.parametrize("nobs", [0, 50])
def test_multisensor_gradients(nobs):
    """Test gradient flow through MultiSensorObsEmbedding (includes empty obs for DDP compatibility)."""
    torch.manual_seed(42)

    sensor_embed_dim = 16
    fusion_dim = 32
    hpx_level = 5
    meta_dim = 8

    sensor_config = {
        "test_sensor_0": ModelSensorConfig(
            sensor_id=0, nchannel=10, platform_ids=tuple(range(5))
        ),
        "test_sensor_1": ModelSensorConfig(
            sensor_id=1, nchannel=10, platform_ids=tuple(range(5))
        ),
    }

    sensor_embedder_config = SensorEmbedderConfig(
        embed_dim=sensor_embed_dim,
        meta_dim=meta_dim,
        fusion_dim=fusion_dim,
    )

    embedder = MultiSensorObsEmbedding(
        sensor_embedder_config=sensor_embedder_config,
        sensors=sensor_config,
        hpx_level=hpx_level,
    )
    embedder.train()
    embedder.zero_grad()

    obs = create_unified_observation(
        nobs=nobs,
        batch_size=2,
        time_steps=1,
        hpx_level=hpx_level + 1,
        meta_dim=meta_dim,
        sensor_config=sensor_config,
        n_embed=100,
        ensure_all_sensors=True,
    )

    # Forward + backward
    output = embedder(obs)
    assert torch.isfinite(output).all()
    loss = output.sum()
    loss.backward()

    # Check gradients (critical for DDP - must work with empty obs)
    all_have_grads, missing = check_all_params_have_gradients(embedder)
    assert all_have_grads, f"Parameters without gradients (nobs={nobs}): {missing}"
