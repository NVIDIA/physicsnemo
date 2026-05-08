# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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
"""Tests for UnifiedObservation and split_by_sensor."""

import pytest
import torch

from physicsnemo.experimental.datapipes.healda.types import (
    UnifiedObservation,
    split_by_sensor,
)


def make_realistic_obs(
    B: int = 2, T: int = 2, sensors: list[int] = [0, 1, 2]
) -> UnifiedObservation:
    """Create realistic cyclic observation data matching real UFS patterns."""
    S = len(sensors)

    all_obs = []
    for b in range(B):
        for t in range(T):
            for i in range(6):
                sensor_id = sensors[i % S]
                all_obs.append((sensor_id, b, t, len(all_obs)))

    all_obs.sort(key=lambda x: (x[0], x[3]))

    values = torch.tensor([x[3] for x in all_obs], dtype=torch.float32)

    lengths_3d = torch.zeros((S, B, T), dtype=torch.int32)
    for s_local, s_id in enumerate(sensors):
        for b in range(B):
            for t in range(T):
                lengths_3d[s_local, b, t] = sum(
                    1
                    for obs in all_obs
                    if obs[0] == s_id and obs[1] == b and obs[2] == t
                )

    nobs = len(all_obs)
    return UnifiedObservation(
        obs=values.unsqueeze(1).expand(nobs, 3),
        time=values.long(),
        float_metadata=values.unsqueeze(1).expand(nobs, 5),
        pix=torch.arange(nobs, dtype=torch.long),
        local_channel=torch.zeros(nobs, dtype=torch.long),
        local_platform=torch.zeros(nobs, dtype=torch.long),
        obs_type=torch.zeros(nobs, dtype=torch.long),
        global_channel=torch.zeros(nobs, dtype=torch.long),
        global_platform=torch.zeros(nobs, dtype=torch.long),
        hpx_level=6,
        lengths=lengths_3d,
    )


def test_split_preserves_all_observations():
    obs = make_realistic_obs(B=2, T=2, sensors=[0, 1, 2])
    total_before = obs.obs.shape[0]

    split = split_by_sensor(obs, [0, 1, 2])

    total_after = sum(split[sid].obs.shape[0] for sid in [0, 1, 2])
    assert total_after == total_before

    for sid in [0, 1, 2]:
        assert split[sid].obs.shape[0] == 8


def test_split_content_correctness():
    obs = make_realistic_obs(B=2, T=2, sensors=[0, 1, 2])
    split = split_by_sensor(obs, [0, 1, 2])

    for sid in [0, 1, 2]:
        assert split[sid].obs.shape[0] == 8


def test_split_lengths_match_obs_count():
    obs = make_realistic_obs(B=1, T=2, sensors=[0, 1])
    split = split_by_sensor(obs, [0, 1])

    for sid in [0, 1]:
        s_obs = split[sid]
        assert s_obs.lengths.sum().item() == s_obs.obs.shape[0]


def test_split_empty_sensor():
    """Extra sensor_ids beyond the configured list produce empty sub-objects."""
    obs = make_realistic_obs(B=1, T=1, sensors=[0, 1])
    # target_sensor_ids must include the configured order first, extras appended
    split = split_by_sensor(obs, [0, 1, 2])

    assert split[2].obs.shape[0] == 0
    assert split[2].lengths.shape == (1, 1, 1)


def test_split_requires_lengths():
    obs = UnifiedObservation(
        obs=torch.randn(10, 3),
        time=torch.zeros(10, dtype=torch.long),
        float_metadata=torch.randn(10, 5),
        pix=torch.zeros(10, dtype=torch.long),
        local_channel=torch.zeros(10, dtype=torch.long),
        local_platform=torch.zeros(10, dtype=torch.long),
        obs_type=torch.zeros(10, dtype=torch.long),
        global_channel=torch.zeros(10, dtype=torch.long),
        hpx_level=6,
        lengths=None,
    )

    with pytest.raises(ValueError, match="lengths is required"):
        split_by_sensor(obs, [0, 1])


def test_lengths_nonnegative():
    obs = make_realistic_obs(B=2, T=3, sensors=[0, 1, 2])
    assert torch.all(obs.lengths >= 0)


def test_split_handles_sparse_windows():
    """Sensor missing from some (b,t) windows."""
    B, T = 2, 3

    all_obs = []
    for b in range(B):
        for t in range(T):
            all_obs.extend([(0, b, t)] * 2)
    all_obs.extend([(4, 1, 2)] * 3)

    nobs = len(all_obs)

    lengths_3d = torch.zeros((2, B, T), dtype=torch.int32)
    lengths_3d[0, :, :] = 2
    lengths_3d[1, 1, 2] = 3

    obs = UnifiedObservation(
        obs=torch.arange(nobs, dtype=torch.float32).unsqueeze(1).expand(nobs, 3),
        time=torch.zeros(nobs, dtype=torch.long),
        float_metadata=torch.arange(nobs, dtype=torch.float32)
        .unsqueeze(1)
        .expand(nobs, 5),
        pix=torch.arange(nobs, dtype=torch.long),
        local_channel=torch.zeros(nobs, dtype=torch.long),
        local_platform=torch.zeros(nobs, dtype=torch.long),
        obs_type=torch.zeros(nobs, dtype=torch.long),
        global_channel=torch.zeros(nobs, dtype=torch.long),
        global_platform=torch.zeros(nobs, dtype=torch.long),
        hpx_level=6,
        lengths=lengths_3d,
    )

    assert obs.batch_dims == (2, 3)

    # Positional: target_sensor_ids[0]=0 -> lengths[0], target_sensor_ids[1]=4 -> lengths[1]
    # Extra sensor 99 is beyond len(sizes) -> empty
    split = split_by_sensor(obs, [0, 4, 99])

    s0 = split[0]
    assert s0.obs.shape[0] == 12
    assert s0.lengths.shape == (1, 2, 3)
    assert s0.lengths.sum().item() == 12

    s4 = split[4]
    assert s4.obs.shape[0] == 3
    assert s4.lengths[0, 1, 2].item() == 3

    s99 = split[99]
    assert s99.obs.shape[0] == 0
    assert torch.all(s99.lengths == 0)


def test_split_global_platform_propagated():
    """global_platform is sliced correctly through split_by_sensor."""
    obs = make_realistic_obs(B=1, T=1, sensors=[0, 1])
    split = split_by_sensor(obs, [0, 1])
    for sid in [0, 1]:
        assert split[sid].global_platform is not None
        assert split[sid].global_platform.shape[0] == split[sid].obs.shape[0]
