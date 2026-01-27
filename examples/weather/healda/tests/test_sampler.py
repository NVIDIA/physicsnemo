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
# ruff: noqa: S101
import itertools

from datasets import samplers


def test_InfiniteSequentialSampler():
    ds = list(range(10))
    sampler = samplers.InfiniteSequentialSampler(ds, shuffle_every=2)
    iterator = iter(sampler)
    i = next(iterator)
    ip1 = next(iterator)
    assert (i + 1) % 10 == ip1


def test_ChunkedRandomSampler():
    s = samplers.ChunkedDistributedSampler(list(range(100)), chunk_size=5)
    it = iter(s)
    visited = set()
    for chunk in range(20):
        last_i = 0
        for i in range(5):
            idx = next(it)
            if i > 0:
                assert idx - last_i == 1
            last_i = idx
            visited.add(idx)

    no_repeats = len(visited) == 100
    assert no_repeats


def test_ChunkedDistributedSampler_with_islice():
    """Test that ChunkedDistributedSampler works correctly with itertools.islice.

    This test verifies the fix for a bug where calling iter(sampler) multiple times
    would reset the iterator state, causing itertools.islice to restart from the
    beginning instead of continuing from where it left off.
    """
    # Create a dataset with 100 items and chunk size of 10
    dataset = list(range(100))
    sampler = samplers.ChunkedDistributedSampler(
        dataset, chunk_size=10, drop_last=False
    )

    # Test 1: Basic iteration works
    iterator = iter(sampler)
    first_10 = list(itertools.islice(iterator, 10))
    assert len(first_10) == 10
    assert first_10 == list(range(10)), f"Expected [0-9], got {first_10}"

    # Test 2: The key bug fix - calling iter(sampler) again should continue from where we left off
    # Before the fix, this would restart from the beginning
    # After the fix, this should continue from where we left off
    iterator2 = iter(sampler)
    next_10 = list(itertools.islice(iterator2, 10))

    # Verify we get the next 10 items (10-19), not the same as first_10
    expected_next_10 = list(range(10, 20))
    assert next_10 == expected_next_10, f"Expected {expected_next_10}, got {next_10}"
    assert first_10 != next_10, "Iterator state was reset - this is the bug!"

    # Test 3: Verify the items are sequential within chunks
    # Items should be consecutive within each chunk of 10
    for i in range(0, len(first_10), 10):
        chunk = first_10[i : i + 10]
        if len(chunk) > 1:
            for j in range(1, len(chunk)):
                assert chunk[j] - chunk[j - 1] == 1, (
                    f"Items not consecutive in chunk: {chunk}"
                )

    # Test 4: Verify we can continue iteration from the same iterator
    iterator3 = iter(sampler)
    first_5 = list(itertools.islice(iterator3, 5))
    next_5 = list(itertools.islice(iterator3, 5))

    # These should be consecutive (continuing from where iterator2 left off at 20)
    expected_first_5 = list(range(20, 25))
    expected_next_5 = list(range(25, 30))
    assert first_5 == expected_first_5, f"Expected {expected_first_5}, got {first_5}"
    assert next_5 == expected_next_5, f"Expected {expected_next_5}, got {next_5}"

    # Test 5: Verify we can exhaust the iterator and it resets properly
    iterator4 = iter(sampler)
    all_items = list(iterator4)
    # Should continue from where iterator3 left off (at 30)
    expected_remaining = list(range(30, 100))
    assert all_items == expected_remaining, (
        f"Expected {expected_remaining}, got {all_items}"
    )

    # After exhaustion, a new iterator should start from the beginning
    iterator5 = iter(sampler)
    first_item = next(iterator5)
    # The first item should be from chunk 0 (0-9 range)
    assert 0 <= first_item < 10, f"Expected first item to be 0-9, got {first_item}"


def test_shuffle_within_chunk():
    """Test shuffle_within_chunk randomizes samples within each chunk."""
    s = samplers.ChunkedDistributedSampler(
        list(range(100)),
        chunk_size=10,
        shuffle=False,  # Keep chunks in sequential order
        shuffle_within_chunk=True,
        seed=42,
    )

    indices = list(s)

    # All indices should be present
    assert sorted(indices) == list(range(100))

    # First chunk should have indices 0-9 but shuffled
    first_chunk = indices[:10]
    assert sorted(first_chunk) == list(range(10))
    assert first_chunk != list(range(10)), "Within-chunk shuffle should change order"


def test_shuffle_epoch_changes_chunks():
    """Test that epoch auto-increment causes different chunk order between epochs."""
    s = samplers.ChunkedDistributedSampler(
        list(range(100)),
        chunk_size=10,
        shuffle=True,
        shuffle_within_chunk=True,
        seed=42,
    )

    # First epoch - get first 10 indices
    epoch1_indices = list(s)
    epoch1_first_10 = sorted(epoch1_indices[:10])

    # Second epoch (auto-incremented) - get first 10 indices
    epoch2_indices = list(s)
    epoch2_first_10 = sorted(epoch2_indices[:10])

    # Both epochs should have all 100 indices
    assert sorted(epoch1_indices) == list(range(100))
    assert sorted(epoch2_indices) == list(range(100))

    # First 10 indices should correspond to different chunks across epochs
    assert epoch1_first_10 != epoch2_first_10, (
        "First 10 indices should correspond to different chunks across epochs"
    )


def test_restartable_distributed_sampler_iteration():
    """Test RestartableDistributedSampler iteration and epoch transitions."""
    dataset = list(range(100))
    sampler = samplers.RestartableDistributedSampler(
        dataset, rank=0, num_replicas=2, seed=42
    )
    sampler.set_epoch(0)

    # Test distributed splitting - rank 0 should get 50 items
    assert len(sampler) == 50

    # Iterate through one epoch
    indices_epoch0 = list(sampler)
    assert len(set(indices_epoch0)) == 50  # All unique

    # After exhaustion, should auto-transition to next epoch
    assert sampler.epoch == 1
