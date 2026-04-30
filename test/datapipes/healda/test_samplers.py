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
"""Tests for RestartableDistributedSampler."""

from physicsnemo.experimental.datapipes.healda.samplers import (
    RestartableDistributedSampler,
)


def test_basic_iteration():
    """Sampler yields all indices for the rank and respects __len__."""
    dataset = list(range(100))
    sampler = RestartableDistributedSampler(dataset, rank=0, num_replicas=1, seed=42)
    sampler.set_epoch(0)

    indices = list(sampler)
    assert len(indices) == 100
    assert sorted(indices) == list(range(100))


def test_epoch_auto_advance():
    """Exhausting an epoch raises StopIteration, then next epoch starts."""
    dataset = list(range(20))
    sampler = RestartableDistributedSampler(dataset, rank=0, num_replicas=1, seed=7)
    sampler.set_epoch(0)

    epoch0 = list(sampler)
    assert len(epoch0) == 20

    # After StopIteration, epoch has advanced; next iteration gives a new epoch
    epoch1 = list(sampler)
    assert len(epoch1) == 20
    assert sorted(epoch1) == list(range(20))
    # Different permutation (with high probability)
    assert epoch0 != epoch1


def test_restart_resumes_correctly():
    """restart() resumes from exact checkpoint position."""
    dataset = list(range(50))
    sampler = RestartableDistributedSampler(dataset, rank=0, num_replicas=1, seed=42)
    sampler.set_epoch(0)

    # Consume first 20 indices
    _ = [next(sampler) for _ in range(20)]

    # Collect remaining
    remaining = []
    try:
        while True:
            remaining.append(next(sampler))
    except StopIteration:
        pass

    # Now restart at position 20 and verify we get the same remaining
    sampler.restart(epoch=0, iteration=20, seed=42)
    restarted_remaining = list(sampler)
    assert restarted_remaining == remaining


def test_reproducible():
    """Same seed/rank/epoch produces identical permutation."""
    dataset = list(range(100))
    s1 = RestartableDistributedSampler(dataset, rank=0, num_replicas=1, seed=42)
    s2 = RestartableDistributedSampler(dataset, rank=0, num_replicas=1, seed=42)
    s1.set_epoch(0)
    s2.set_epoch(0)
    assert list(s1) == list(s2)


def test_multi_replica_independent():
    """Different ranks receive disjoint slices of the shared permutation."""
    dataset = list(range(100))
    s0 = RestartableDistributedSampler(dataset, rank=0, num_replicas=2, seed=42)
    s1 = RestartableDistributedSampler(dataset, rank=1, num_replicas=2, seed=42)
    s0.set_epoch(0)
    s1.set_epoch(0)

    idx0 = list(s0)
    idx1 = list(s1)

    assert len(idx0) == 50
    assert len(idx1) == 50
    # Each rank gets a valid subset of indices
    assert all(0 <= i < 100 for i in idx0)
    assert all(0 <= i < 100 for i in idx1)
    # Ranks visit different indices (stride-partitioned shared permutation)
    assert idx0 != idx1
    # Together they cover every sample exactly once
    assert sorted(idx0 + idx1) == list(range(100))


def test_len_drops_remainder():
    """Length accounts for dropping remainder across replicas."""
    dataset = list(range(103))
    sampler = RestartableDistributedSampler(dataset, rank=0, num_replicas=4, seed=0)
    # 103 // 4 = 25, remainder 3 dropped
    assert len(sampler) == 25
