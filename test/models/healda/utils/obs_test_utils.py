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
"""Test utilities for observation embedding tests."""

import torch

from physicsnemo.experimental.models.healda import ModelSensorConfig, UnifiedObservation


def create_unified_observation(
    nobs: int,
    batch_size: int = 1,
    time_steps: int = 1,
    meta_dim: int = 8,
    hpx_level: int = 6,
    nchannel: int = 10,
    nplatform: int = 5,
    n_embed: int = 5,
    device: str = "cpu",
    ensure_all_sensors: bool = False,
    sensor_config: dict[str, ModelSensorConfig] | None = None,
) -> UnifiedObservation:
    torch.manual_seed(0)

    # Extract sensor info
    if sensor_config is not None:
        sensors = [
            (cfg.sensor_id, cfg.nchannel, list(cfg.platform_ids))
            for cfg in sensor_config.values()
        ]
    else:
        sensors = [(0, nchannel, list(range(nplatform)))]

    sensor_ids = [s[0] for s in sensors]
    n_sensors = len(sensor_ids)
    npix = 12 * 4**hpx_level

    # Build sensor_id_to_local mapping
    max_sensor_id = max(sensor_ids) if sensor_ids else 0
    sensor_id_to_local = torch.full((max_sensor_id + 1,), -1, dtype=torch.long)
    for local_idx, sid in enumerate(sensor_ids):
        sensor_id_to_local[sid] = local_idx

    # Handle empty case
    if nobs == 0:
        return UnifiedObservation(
            obs=torch.empty(0, device=device),
            time=torch.empty(0, dtype=torch.long, device=device),
            float_metadata=torch.empty((0, meta_dim), device=device),
            int_metadata=torch.empty((0, 6), dtype=torch.long, device=device),
            offsets=torch.zeros(
                (n_sensors, batch_size, time_steps), dtype=torch.long, device=device
            ),
            sensor_id_to_local=sensor_id_to_local.to(device),
            hpx_level=hpx_level,
        )

    # Generate random observations
    def random_obs_for_sensor(sid):
        """Generate one observation for a given sensor."""
        _, nchan, plat_ids = next(s for s in sensors if s[0] == sid)
        return {
            "obs": torch.randn(1).item() * 0.5,
            "time": 946674000000000000 + torch.randint(0, 86400 * 10**9, (1,)).item(),
            "pix": torch.randint(0, npix, (1,)).item(),
            "platform": plat_ids[torch.randint(0, len(plat_ids), (1,)).item()],
            "channel": torch.randint(0, nchan, (1,)).item(),
            "embed_id": torch.randint(0, n_embed, (1,)).item(),
            "float_meta": torch.randn(meta_dim) * 0.8,
            "sensor_id": sid,
        }

    # Generate observations
    if ensure_all_sensors:
        # One per sensor first, then random
        observations = [random_obs_for_sensor(sid) for sid in sensor_ids]
        observations.extend(
            random_obs_for_sensor(
                sensor_ids[torch.randint(0, len(sensor_ids), (1,)).item()]
            )
            for _ in range(nobs - len(sensor_ids))
        )
    else:
        observations = [
            random_obs_for_sensor(
                sensor_ids[torch.randint(0, len(sensor_ids), (1,)).item()]
            )
            for _ in range(nobs)
        ]

    # Sort by sensor_id (required for per-sensor processing)
    observations.sort(key=lambda x: x["sensor_id"])

    # Build tensors
    obs = torch.tensor([o["obs"] for o in observations], dtype=torch.float32)
    time = torch.tensor([o["time"] for o in observations], dtype=torch.long)
    float_metadata = torch.stack([o["float_meta"] for o in observations])
    sensor_id_tensor = torch.tensor(
        [o["sensor_id"] for o in observations], dtype=torch.long
    )

    pix_tensor = torch.tensor([o["pix"] for o in observations], dtype=torch.long)
    channel_tensor = torch.tensor(
        [o["channel"] for o in observations], dtype=torch.long
    )
    platform_tensor = torch.tensor(
        [o["platform"] for o in observations], dtype=torch.long
    )

    idx = UnifiedObservation.bucket_index
    int_metadata = torch.zeros((nobs, 6), dtype=torch.long)
    int_metadata[:, idx.sensor] = sensor_id_tensor
    int_metadata[:, idx.pix] = pix_tensor
    int_metadata[:, idx.local_channel] = channel_tensor
    int_metadata[:, idx.platform] = platform_tensor
    int_metadata[:, idx.obs_type] = 0
    int_metadata[:, idx.global_channel] = channel_tensor

    # Build 3D offsets: cumulative end indices over (sensor, batch, time)
    # All observations go into the first window for simplicity
    offsets = torch.zeros((n_sensors, batch_size, time_steps), dtype=torch.long)
    cumulative = 0
    for s_local, sid in enumerate(sensor_ids):
        cumulative += (sensor_id_tensor == sid).sum().item()
        offsets[s_local, :, :] = cumulative

    return UnifiedObservation(
        obs=obs.to(device),
        time=time.to(device),
        float_metadata=float_metadata.to(device),
        int_metadata=int_metadata.to(device),
        offsets=offsets.to(device),
        sensor_id_to_local=sensor_id_to_local.to(device),
        hpx_level=hpx_level,
    )
