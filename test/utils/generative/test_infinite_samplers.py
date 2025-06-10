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
from torch.utils.data import Dataset


# Mock dataset for testing - memory efficient for any size
class MockDataset(Dataset):
    """Mock dataset that doesn't actually store data, just simulates any size."""

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx >= self.size:
            raise IndexError("Index out of range")
        return idx  # Just return the index as mock data


# ============================================================================
# InfiniteSampler Tests
# ============================================================================


def test_infinite_sampler_parameter_validation():
    """Test parameter validation in InfiniteSampler constructor."""
    from physicsnemo.utils.generative.utils import InfiniteSampler

    dataset = MockDataset(10)

    # Test empty dataset
    empty_dataset = MockDataset(0)
    with pytest.raises(ValueError, match="Dataset must contain at least one item"):
        InfiniteSampler(empty_dataset)

    # Test invalid num_replicas
    with pytest.raises(ValueError, match="num_replicas must be positive"):
        InfiniteSampler(dataset, num_replicas=0)

    with pytest.raises(ValueError, match="num_replicas must be positive"):
        InfiniteSampler(dataset, num_replicas=-1)

    # Test invalid rank
    with pytest.raises(
        ValueError, match="rank must be non-negative and less than num_replicas"
    ):
        InfiniteSampler(dataset, rank=-1, num_replicas=2)

    with pytest.raises(
        ValueError, match="rank must be non-negative and less than num_replicas"
    ):
        InfiniteSampler(dataset, rank=2, num_replicas=2)

    # Test invalid window_size
    with pytest.raises(ValueError, match="window_size must be between 0 and 1"):
        InfiniteSampler(dataset, window_size=-0.1)

    with pytest.raises(ValueError, match="window_size must be between 0 and 1"):
        InfiniteSampler(dataset, window_size=1.1)

    # Test valid parameters (should not raise)
    sampler = InfiniteSampler(
        dataset, rank=0, num_replicas=1, shuffle=True, seed=42, window_size=0.5
    )
    assert sampler is not None


def test_infinite_sampler_basic_iteration():
    """Test basic iteration behavior without shuffling."""
    from physicsnemo.utils.generative.utils import InfiniteSampler

    dataset = MockDataset(5)
    sampler = InfiniteSampler(dataset, shuffle=False, rank=0, num_replicas=1)

    # Get first few indices
    iterator = iter(sampler)
    indices = [next(iterator) for _ in range(15)]

    # Should cycle through 0,1,2,3,4 repeatedly
    expected = [0, 1, 2, 3, 4] * 3
    assert indices == expected


def test_infinite_sampler_distributed_behavior():
    """Test distributed sampling with multiple ranks."""
    from physicsnemo.utils.generative.utils import InfiniteSampler

    dataset = MockDataset(10)

    # Test with 2 replicas
    sampler_rank0 = InfiniteSampler(dataset, shuffle=False, rank=0, num_replicas=2)
    sampler_rank1 = InfiniteSampler(dataset, shuffle=False, rank=1, num_replicas=2)

    # Get indices for each rank
    iter0 = iter(sampler_rank0)
    iter1 = iter(sampler_rank1)

    indices_rank0 = [next(iter0) for _ in range(10)]
    indices_rank1 = [next(iter1) for _ in range(10)]

    # Rank 0 should get indices 0,2,4,6,8,0,2,4,6,8
    # Rank 1 should get indices 1,3,5,7,9,1,3,5,7,9
    expected_rank0 = [0, 2, 4, 6, 8, 0, 2, 4, 6, 8]
    expected_rank1 = [1, 3, 5, 7, 9, 1, 3, 5, 7, 9]

    assert indices_rank0 == expected_rank0
    assert indices_rank1 == expected_rank1

    # Ensure no overlap between ranks
    assert set(indices_rank0[:5]).isdisjoint(set(indices_rank1[:5]))


def test_infinite_sampler_shuffle_reproducibility():
    """Test that shuffling is reproducible with same seed."""
    from physicsnemo.utils.generative.utils import InfiniteSampler

    dataset = MockDataset(10)

    # Create two samplers with same seed
    sampler1 = InfiniteSampler(dataset, shuffle=True, seed=42, rank=0, num_replicas=1)
    sampler2 = InfiniteSampler(dataset, shuffle=True, seed=42, rank=0, num_replicas=1)

    # Get indices from both
    iter1 = iter(sampler1)
    iter2 = iter(sampler2)

    indices1 = [next(iter1) for _ in range(20)]
    indices2 = [next(iter2) for _ in range(20)]

    # Should be identical
    assert indices1 == indices2

    # Create sampler with different seed
    sampler3 = InfiniteSampler(dataset, shuffle=True, seed=123, rank=0, num_replicas=1)
    iter3 = iter(sampler3)
    indices3 = [next(iter3) for _ in range(20)]

    # Should be different from the first two
    assert indices1 != indices3


def test_infinite_sampler_window_size_effects():
    """Test different window sizes and their behavioral effects."""
    from physicsnemo.utils.generative.utils import InfiniteSampler

    dataset = MockDataset(10)

    # Test basic functionality - all window sizes should produce valid indices
    for window_size in [0.0, 0.1, 0.5, 1.0]:
        sampler = InfiniteSampler(
            dataset, shuffle=True, seed=42, window_size=window_size
        )
        iterator = iter(sampler)
        indices = [next(iterator) for _ in range(50)]

        # All indices should be in valid range
        assert all(0 <= idx < 10 for idx in indices)
        # Should see a reasonable variety of indices over longer sequence
        unique_indices = set(indices)
        assert (
            len(unique_indices) >= 6
        ), f"window_size={window_size} saw only {len(unique_indices)} unique indices"

    # Test reproducibility - same window_size + seed should be deterministic
    sampler1 = InfiniteSampler(dataset, shuffle=True, seed=123, window_size=0.5)
    sampler2 = InfiniteSampler(dataset, shuffle=True, seed=123, window_size=0.5)

    iter1 = iter(sampler1)
    iter2 = iter(sampler2)

    indices1 = [next(iter1) for _ in range(25)]
    indices2 = [next(iter2) for _ in range(25)]

    assert (
        indices1 == indices2
    ), "Same seed + window_size should produce identical sequences"


def test_infinite_sampler_edge_cases():
    """Test edge cases like single-item dataset."""
    from physicsnemo.utils.generative.utils import InfiniteSampler

    # Single item dataset
    single_dataset = MockDataset(1)
    sampler = InfiniteSampler(single_dataset, shuffle=True, seed=42)

    iterator = iter(sampler)
    indices = [next(iterator) for _ in range(10)]

    # Should always return 0
    assert all(idx == 0 for idx in indices)

    # Two item dataset with distributed sampling
    two_dataset = MockDataset(2)
    sampler_rank0 = InfiniteSampler(two_dataset, rank=0, num_replicas=2, shuffle=False)
    sampler_rank1 = InfiniteSampler(two_dataset, rank=1, num_replicas=2, shuffle=False)

    iter0 = iter(sampler_rank0)
    iter1 = iter(sampler_rank1)

    # Rank 0 should get index 0 repeatedly, rank 1 should get index 1 repeatedly
    indices0 = [next(iter0) for _ in range(5)]
    indices1 = [next(iter1) for _ in range(5)]

    assert all(idx == 0 for idx in indices0)
    assert all(idx == 1 for idx in indices1)


def test_infinite_sampler_attributes():
    """Test that sampler attributes are set correctly."""
    from physicsnemo.utils.generative.utils import InfiniteSampler

    dataset = MockDataset(10)
    sampler = InfiniteSampler(
        dataset=dataset, rank=2, num_replicas=4, shuffle=True, seed=123, window_size=0.8
    )

    assert sampler.dataset is dataset
    assert sampler.rank == 2
    assert sampler.num_replicas == 4
    assert sampler.shuffle is True
    assert sampler.seed == 123
    assert sampler.window_size == 0.8


# ============================================================================
# InfiniteHashSampler Tests
# ============================================================================


def test_infinite_hash_sampler_parameter_validation():
    """Test parameter validation in InfiniteHashSampler constructor."""
    from physicsnemo.utils.generative.utils import InfiniteHashSampler

    dataset = MockDataset(10)

    # Test empty dataset
    empty_dataset = MockDataset(0)
    with pytest.raises(ValueError, match="Dataset must contain at least one item"):
        InfiniteHashSampler(empty_dataset)

    # Test invalid num_replicas
    with pytest.raises(ValueError, match="num_replicas must be positive"):
        InfiniteHashSampler(dataset, num_replicas=0)

    with pytest.raises(ValueError, match="num_replicas must be positive"):
        InfiniteHashSampler(dataset, num_replicas=-1)

    # Test invalid rank
    with pytest.raises(
        ValueError, match="rank must be non-negative and less than num_replicas"
    ):
        InfiniteHashSampler(dataset, rank=-1, num_replicas=2)

    with pytest.raises(
        ValueError, match="rank must be non-negative and less than num_replicas"
    ):
        InfiniteHashSampler(dataset, rank=2, num_replicas=2)

    # Test valid parameters (should not raise)
    sampler = InfiniteHashSampler(
        dataset, rank=0, num_replicas=1, randomize=True, seed=42
    )
    assert sampler is not None


def test_infinite_hash_sampler_basic_sequential_iteration():
    """Test basic iteration behavior without randomization."""
    from physicsnemo.utils.generative.utils import InfiniteHashSampler

    dataset = MockDataset(5)
    sampler = InfiniteHashSampler(dataset, randomize=False, rank=0, num_replicas=1)

    # Get first few indices
    iterator = iter(sampler)
    indices = [next(iterator) for _ in range(15)]

    # Should cycle through 0,1,2,3,4 repeatedly
    expected = [0, 1, 2, 3, 4] * 3
    assert indices == expected


def test_infinite_hash_sampler_distributed_behavior():
    """Test distributed sampling with multiple ranks."""
    from physicsnemo.utils.generative.utils import InfiniteHashSampler

    dataset = MockDataset(10)

    # Test with 2 replicas, no randomization for predictable testing
    sampler_rank0 = InfiniteHashSampler(
        dataset, randomize=False, rank=0, num_replicas=2
    )
    sampler_rank1 = InfiniteHashSampler(
        dataset, randomize=False, rank=1, num_replicas=2
    )

    # Get indices for each rank
    iter0 = iter(sampler_rank0)
    iter1 = iter(sampler_rank1)

    indices_rank0 = [next(iter0) for _ in range(10)]
    indices_rank1 = [next(iter1) for _ in range(10)]

    # Fixed InfiniteHashSampler algorithm:
    # global_idx starts at rank, increments by num_replicas
    # sequential_idx = global_idx % dataset_size
    # Rank 0: global_idx = 0,2,4,6,8,10,12,14,16,18 → sequential_idx = 0,2,4,6,8,0,2,4,6,8
    # Rank 1: global_idx = 1,3,5,7,9,11,13,15,17,19 → sequential_idx = 1,3,5,7,9,1,3,5,7,9
    expected_rank0 = [0, 2, 4, 6, 8, 0, 2, 4, 6, 8]
    expected_rank1 = [1, 3, 5, 7, 9, 1, 3, 5, 7, 9]

    assert indices_rank0 == expected_rank0
    assert indices_rank1 == expected_rank1

    # Ensure no overlap between ranks (proper distributed sampling)
    assert set(indices_rank0[:5]).isdisjoint(set(indices_rank1[:5]))


def test_infinite_hash_sampler_randomization_reproducibility():
    """Test that hash-based randomization is reproducible with same seed."""
    from physicsnemo.utils.generative.utils import InfiniteHashSampler

    dataset = MockDataset(10)

    # Create two samplers with same seed
    sampler1 = InfiniteHashSampler(
        dataset, randomize=True, seed=42, rank=0, num_replicas=1
    )
    sampler2 = InfiniteHashSampler(
        dataset, randomize=True, seed=42, rank=0, num_replicas=1
    )

    # Get indices from both
    iter1 = iter(sampler1)
    iter2 = iter(sampler2)

    indices1 = [next(iter1) for _ in range(20)]
    indices2 = [next(iter2) for _ in range(20)]

    # Should be identical
    assert indices1 == indices2

    # Create sampler with different seed
    sampler3 = InfiniteHashSampler(
        dataset, randomize=True, seed=123, rank=0, num_replicas=1
    )
    iter3 = iter(sampler3)
    indices3 = [next(iter3) for _ in range(20)]

    # Should be different from the first two
    assert indices1 != indices3


def test_infinite_hash_sampler_hash_function_properties():
    """Test properties of the internal hash function."""
    from physicsnemo.utils.generative.utils import InfiniteHashSampler

    dataset = MockDataset(100)
    sampler = InfiniteHashSampler(dataset, randomize=True, seed=42)

    # Test hash function determinism
    hash_results = []
    for i in range(10):
        hash_val = sampler._hash_index(i)
        hash_results.append(hash_val)

    # Same inputs should produce same outputs
    for i in range(10):
        assert sampler._hash_index(i) == hash_results[i]

    # All hash results should be in valid range
    assert all(0 <= h < 100 for h in hash_results)

    # Hash function should distribute values (not all the same)
    assert len(set(hash_results)) > 5


def test_infinite_hash_sampler_hash_distribution():
    """Test that hash function provides reasonable distribution."""
    from physicsnemo.utils.generative.utils import InfiniteHashSampler

    dataset = MockDataset(100)
    sampler = InfiniteHashSampler(dataset, randomize=True, seed=42)

    # Test distribution over a range of inputs
    hash_results = [sampler._hash_index(i) for i in range(200)]

    # Should see good coverage of the output space
    unique_hashes = set(hash_results)
    coverage = len(unique_hashes) / 100

    # Should hit at least 80% of possible indices with 200 samples
    assert coverage >= 0.8, f"Hash function coverage too low: {coverage:.2%}"

    # Test that consecutive inputs don't produce consecutive outputs
    consecutive_pairs = sum(
        1
        for i in range(len(hash_results) - 1)
        if abs(hash_results[i] - hash_results[i + 1]) <= 1
    )
    total_pairs = len(hash_results) - 1
    consecutive_rate = consecutive_pairs / total_pairs

    # Should have low rate of consecutive outputs (good mixing)
    assert (
        consecutive_rate < 0.3
    ), f"Too many consecutive hash outputs: {consecutive_rate:.2%}"


def test_infinite_hash_sampler_different_seeds_different_patterns():
    """Test that different seeds produce different hash patterns."""
    from physicsnemo.utils.generative.utils import InfiniteHashSampler

    dataset = MockDataset(50)

    # Create samplers with different seeds
    seeds = [1, 42, 123, 999]
    patterns = []

    for seed in seeds:
        sampler = InfiniteHashSampler(dataset, randomize=True, seed=seed)
        iterator = iter(sampler)
        pattern = [next(iterator) for _ in range(25)]
        patterns.append(pattern)

    # All patterns should be different from each other
    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            differences = sum(a != b for a, b in zip(patterns[i], patterns[j]))
            assert (
                differences > 10
            ), f"Seed {seeds[i]} and {seeds[j]} produced too similar patterns"


def test_infinite_hash_sampler_distributed_randomization():
    """Test that distributed randomization produces different sequences per rank and is reproducible."""
    from physicsnemo.utils.generative.utils import InfiniteHashSampler

    dataset = MockDataset(20)

    # Test with 3 replicas, all with same seed but randomization enabled
    samplers = []
    for rank in range(3):
        sampler = InfiniteHashSampler(
            dataset, randomize=True, seed=42, rank=rank, num_replicas=3
        )
        samplers.append(sampler)

    # Get indices from each rank
    rank_indices = []
    for sampler in samplers:
        iterator = iter(sampler)
        indices = [next(iterator) for _ in range(15)]
        rank_indices.append(indices)

    # All should have valid indices
    for indices in rank_indices:
        assert all(0 <= idx < 20 for idx in indices)

    # Ranks should get DIFFERENT sequences (proper distributed sampling)
    diff_01 = sum(a != b for a, b in zip(rank_indices[0], rank_indices[1]))
    diff_02 = sum(a != b for a, b in zip(rank_indices[0], rank_indices[2]))
    diff_12 = sum(a != b for a, b in zip(rank_indices[1], rank_indices[2]))

    assert diff_01 > 10, f"Rank 0 and 1 too similar: {diff_01}/15 differences"
    assert diff_02 > 10, f"Rank 0 and 2 too similar: {diff_02}/15 differences"
    assert diff_12 > 10, f"Rank 1 and 2 too similar: {diff_12}/15 differences"

    # Verify reproducibility - create new samplers with same parameters
    new_samplers = []
    for rank in range(3):
        sampler = InfiniteHashSampler(
            dataset, randomize=True, seed=42, rank=rank, num_replicas=3
        )
        new_samplers.append(sampler)

    new_rank_indices = []
    for sampler in new_samplers:
        iterator = iter(sampler)
        indices = [next(iterator) for _ in range(15)]
        new_rank_indices.append(indices)

    # Should be identical (reproducibility)
    for old, new in zip(rank_indices, new_rank_indices):
        assert old == new


def test_infinite_hash_sampler_edge_cases():
    """Test edge cases like single-item dataset."""
    from physicsnemo.utils.generative.utils import InfiniteHashSampler

    # Single item dataset
    single_dataset = MockDataset(1)
    sampler = InfiniteHashSampler(single_dataset, randomize=True, seed=42)

    iterator = iter(sampler)
    indices = [next(iterator) for _ in range(10)]

    # Should always return 0 (since hash(anything) % 1 == 0)
    assert all(idx == 0 for idx in indices)

    # Two item dataset with distributed sampling
    two_dataset = MockDataset(2)
    sampler_rank0 = InfiniteHashSampler(
        two_dataset, rank=0, num_replicas=2, randomize=False
    )
    sampler_rank1 = InfiniteHashSampler(
        two_dataset, rank=1, num_replicas=2, randomize=False
    )

    iter0 = iter(sampler_rank0)
    iter1 = iter(sampler_rank1)

    # Fixed: Rank 0 gets indices 0,0,0... and Rank 1 gets indices 1,1,1...
    indices0 = [next(iter0) for _ in range(5)]
    indices1 = [next(iter1) for _ in range(5)]

    assert all(idx == 0 for idx in indices0)
    assert all(idx == 1 for idx in indices1)


def test_infinite_hash_sampler_very_large_dataset():
    """Test behavior with very large dataset sizes (the main use case)."""
    from physicsnemo.utils.generative.utils import InfiniteHashSampler

    # Use memory-efficient mock dataset for billion-scale testing
    large_dataset = MockDataset(int(1e9))  # 1 billion items, no memory overhead
    sampler = InfiniteHashSampler(large_dataset, randomize=True, seed=42)

    # Should handle large datasets without memory issues
    iterator = iter(sampler)
    indices = [next(iterator) for _ in range(100)]

    # All indices should be in valid range
    assert all(0 <= idx < int(1e9) for idx in indices)

    # Should see good distribution even with large dataset
    unique_indices = set(indices)
    assert len(unique_indices) >= 80  # Should see most indices as unique

    # Test with an even larger dataset to verify no memory issues
    huge_dataset = MockDataset(int(1e10))  # 10 billion items
    huge_sampler = InfiniteHashSampler(huge_dataset, randomize=True, seed=42)

    huge_iterator = iter(huge_sampler)
    huge_indices = [next(huge_iterator) for _ in range(50)]

    # All indices should be in valid range
    assert all(0 <= idx < int(1e10) for idx in huge_indices)

    # Verify hash function still works with very large dataset sizes
    test_indices = [0, int(1e6), int(1e9), int(5e9)]
    for test_idx in test_indices:
        hash_result = huge_sampler._hash_index(test_idx)
        assert 0 <= hash_result < int(1e10)


def test_infinite_hash_sampler_attributes():
    """Test that sampler attributes are set correctly."""
    from physicsnemo.utils.generative.utils import InfiniteHashSampler

    dataset = MockDataset(10)
    sampler = InfiniteHashSampler(
        dataset=dataset, rank=2, num_replicas=4, randomize=True, seed=123
    )

    assert sampler.dataset is dataset
    assert sampler.rank == 2
    assert sampler.num_replicas == 4
    assert sampler.randomize is True
    assert sampler.seed == 123
    assert sampler.dataset_size == 10


def test_infinite_hash_sampler_prime_multiplier_behavior():
    """Test the behavior around the large prime multiplier (2654435761)."""
    from physicsnemo.utils.generative.utils import InfiniteHashSampler

    # Test with dataset size larger than typical ranges
    large_dataset = MockDataset(3000000)  # 3 million
    sampler = InfiniteHashSampler(large_dataset, randomize=True, seed=42)

    # Should work correctly with large indices
    iterator = iter(sampler)
    indices = [next(iterator) for _ in range(50)]

    # All indices should be in valid range
    assert all(0 <= idx < 3000000 for idx in indices)

    # Test the hash function directly with large sequential indices
    hash_values = []
    test_ranges = [
        range(0, 10),  # Small numbers
        range(1000000, 1000010),  # Medium numbers
        range(2654435750, 2654435760),  # Around the prime value
    ]

    for test_range in test_ranges:
        for i in test_range:
            hash_val = sampler._hash_index(i)
            hash_values.append(hash_val)

    # Should distribute well across all ranges
    unique_hashes = len(set(hash_values))
    assert unique_hashes >= 25  # Most should be unique (30 total inputs)
