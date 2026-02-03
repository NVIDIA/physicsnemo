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

from physicsnemo.experimental.models.healda.types import (
    UnifiedObservation,
    split_by_sensor,
)


def make_realistic_obs(
    B: int = 2, T: int = 2, sensors: list[int] = [0, 1, 2]
) -> UnifiedObservation:
    """Create realistic cyclic observation data matching real UFS patterns.

    Sensors cycle: 0,1,2,0,1,2,... within each (b,t) window, then data is sorted globally by sensor_id.
    """
    S = len(sensors)

    # Generate observations: each window has 6 obs cycling through sensors
    all_obs = []
    for b in range(B):
        for t in range(T):
            for i in range(6):  # 6 obs per window
                sensor_id = sensors[i % S]
                all_obs.append(
                    (sensor_id, b, t, len(all_obs))
                )  # (sensor, batch, time, value)

    # Sort by sensor_id (as real data is)
    all_obs.sort(key=lambda x: (x[0], x[3]))  # Sort by sensor, then original index

    # Extract sorted data
    sensor_ids = torch.tensor([x[0] for x in all_obs], dtype=torch.long)
    values = torch.tensor([x[3] for x in all_obs], dtype=torch.float32)

    # Build 3D offsets: (S, B, T) cumulative ends
    # Count how many obs each sensor has in each (b,t) window
    offsets_3d = torch.zeros((S, B, T), dtype=torch.int32)
    idx = 0
    for s_local, s_id in enumerate(sensors):
        for b in range(B):
            for t in range(T):
                # Count obs for this sensor in this window
                count = sum(
                    1
                    for obs in all_obs
                    if obs[0] == s_id and obs[1] == b and obs[2] == t
                )
                idx += count
                offsets_3d[s_local, b, t] = idx

    # Create sensor_id_to_local map
    sensor_id_to_local = torch.full((max(sensors) + 1,), -1, dtype=torch.int32)
    for local_idx, s_id in enumerate(sensors):
        sensor_id_to_local[s_id] = local_idx

    # Build UnifiedObservation
    nobs = len(all_obs)
    return UnifiedObservation(
        obs=values.unsqueeze(1).expand(nobs, 3),  # (nobs, 3) features
        time=values,
        float_metadata=values.unsqueeze(1).expand(nobs, 5),
        int_metadata=torch.stack(
            [
                sensor_ids,
                torch.arange(nobs),
                torch.zeros(nobs),
                torch.zeros(nobs),
                torch.zeros(nobs),
                torch.zeros(nobs),
            ],
            dim=1,
        ),
        offsets=offsets_3d,
        sensor_id_to_local=sensor_id_to_local,
        hpx_level=6,
    )


def test_split_preserves_all_observations():
    """Critical test: verify no observations are lost during split."""
    obs = make_realistic_obs(B=2, T=2, sensors=[0, 1, 2])
    total_before = obs.obs.shape[0]

    split = split_by_sensor(obs, [0, 1, 2])

    # Count total after split
    total_after = sum(split[sid].obs.shape[0] for sid in [0, 1, 2])
    assert total_after == total_before, (
        f"LOST OBSERVATIONS: {total_before} → {total_after}"
    )

    # Each sensor appears 2 times per window (6 obs / 3 sensors), across B*T=4 windows = 8 total
    for sid in [0, 1, 2]:
        assert split[sid].obs.shape[0] == 8, f"Sensor {sid} should have 8 obs"


def test_split_content_correctness():
    """Verify split observations contain correct data for each sensor."""
    obs = make_realistic_obs(B=2, T=2, sensors=[0, 1, 2])
    split = split_by_sensor(obs, [0, 1, 2])

    # Verify each split contains only its sensor's data
    for sid in [0, 1, 2]:
        s_obs = split[sid]
        sensor_ids_in_split = s_obs.int_metadata[:, s_obs.bucket_index.sensor]

        # All observations must be for this sensor
        assert torch.all(sensor_ids_in_split == sid), (
            f"Sensor {sid} contains wrong sensor IDs: {sensor_ids_in_split.unique().tolist()}"
        )

        # Verify values match (obs tensor should match time for our test data)
        assert torch.allclose(s_obs.obs[:, 0], s_obs.time), "Data corruption detected"


def test_split_offsets_are_relative():
    """Verify split offsets are relative to each sensor's slice, not absolute."""
    obs = make_realistic_obs(B=1, T=2, sensors=[0, 1])
    split = split_by_sensor(obs, [0, 1])

    for sid in [0, 1]:
        s_obs = split[sid]
        # Last offset should equal obs count (not some large absolute index)
        assert s_obs.offsets[0, -1, -1].item() == s_obs.obs.shape[0], (
            f"Sensor {sid} offsets not relative"
        )


def test_split_empty_sensor():
    """Test handling of sensor with no data."""
    obs = make_realistic_obs(B=1, T=1, sensors=[0, 1])
    split = split_by_sensor(obs, [0, 1, 2])  # Request sensor 2 which doesn't exist

    assert split[2].obs.shape[0] == 0, "Empty sensor should have 0 observations"
    assert split[2].offsets.shape == (
        1,
        1,
        1,
    ), "Empty sensor should preserve batch structure"


def test_split_requires_offsets():
    """Test that split_by_sensor requires offsets."""
    obs = UnifiedObservation(
        obs=torch.randn(10, 3),
        time=torch.arange(10, dtype=torch.float32),
        float_metadata=torch.randn(10, 5),
        int_metadata=torch.zeros((10, 6), dtype=torch.long),
        offsets=None,  # No offsets
        sensor_id_to_local=None,
        hpx_level=6,
    )

    with pytest.raises(ValueError, match="offsets is required"):
        split_by_sensor(obs, [0, 1])


def test_offsets_monotonic_row_major():
    """Offsets must be nondecreasing in row-major (b,t) order for each sensor."""
    obs = make_realistic_obs(B=2, T=3, sensors=[0, 1, 2])
    S, B, T = obs.offsets.shape

    for s_local in range(S):
        flat = obs.offsets[s_local].reshape(-1)
        assert torch.all(flat[:-1] <= flat[1:]), (
            f"offsets for sensor {s_local} must be nondecreasing in row-major (b,t)"
        )


def test_split_handles_sparse_windows():
    """Sensor missing from some (b,t) windows; split must still work."""
    B, T = 2, 3
    sensors = [0, 4]

    # Sparse data: sensor 0 everywhere (2 obs/window), sensor 4 only in (b=1,t=2) with 3 obs
    all_obs = []
    for b in range(B):
        for t in range(T):
            all_obs.extend([(0, b, t)] * 2)  # sensor 0: 2 obs/window
    all_obs.extend([(4, 1, 2)] * 3)  # sensor 4: 3 obs only in (1,2)

    sensor_ids = torch.tensor([x[0] for x in all_obs], dtype=torch.long)
    nobs = len(all_obs)

    # Build offsets: sensor 0 cumulative, sensor 4 mostly zeros except (1,2)
    offsets_3d = torch.zeros((2, B, T), dtype=torch.int32)
    idx = 0
    for b in range(B):
        for t in range(T):
            idx += 2  # sensor 0 has 2 obs/window
            offsets_3d[0, b, t] = idx
    for b in range(B):
        for t in range(T):
            if b == 1 and t == 2:
                idx += 3  # sensor 4 only here
            offsets_3d[1, b, t] = idx

    sensor_id_to_local = torch.full((5,), -1, dtype=torch.int32)
    for local_idx, s_id in enumerate(sensors):
        sensor_id_to_local[s_id] = local_idx

    obs = UnifiedObservation(
        obs=torch.arange(nobs, dtype=torch.float32).unsqueeze(1).expand(nobs, 3),
        time=torch.arange(nobs, dtype=torch.float32),
        float_metadata=torch.arange(nobs, dtype=torch.float32)
        .unsqueeze(1)
        .expand(nobs, 5),
        int_metadata=torch.stack(
            [
                sensor_ids,
                torch.arange(nobs),
                torch.zeros(nobs),
                torch.zeros(nobs),
                torch.zeros(nobs),
                torch.zeros(nobs),
            ],
            dim=1,
        ),
        offsets=offsets_3d,
        sensor_id_to_local=sensor_id_to_local,
        hpx_level=6,
    )

    assert obs.batch_dims == (2, 3)

    split = split_by_sensor(obs, [0, 4, 99])

    # Sensor 0: 12 obs (2 per window * 6 windows)
    s0 = split[0]
    assert s0.obs.shape[0] == 12
    assert s0.offsets.shape == (1, 2, 3)
    assert s0.batch_dims == (2, 3)
    assert s0.offsets[0, -1, -1].item() == 12

    # Sensor 4: 3 obs (only in window (1,2))
    s4 = split[4]
    assert s4.obs.shape[0] == 3
    assert s4.offsets.shape == (1, 2, 3)
    assert s4.offsets[0, -1, -1].item() == 3
    assert s4.batch_dims == (2, 3)

    # Sensor 99: absent
    s99 = split[99]
    assert s99.obs.shape[0] == 0
    assert s99.offsets.shape == (1, 2, 3)
    assert torch.all(s99.offsets == 0)
