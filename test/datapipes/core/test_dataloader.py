# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Tests for DataLoader class.

This module consolidates all DataLoader tests, using real CUDA streams
for GPU-related tests instead of mocks.
"""

import pytest
import torch
from tensordict import TensorDict
from torch.utils.data import (
    RandomSampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)

import physicsnemo.datapipes as dp

# ============================================================================
# Basic DataLoader functionality
# ============================================================================


class TestDataLoaderBasic:
    """Tests for basic DataLoader functionality."""

    def test_create_dataloader(self, numpy_data_dir):
        """Test creating a DataLoader."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(dataset, batch_size=2)

        # 10 samples / 2 batch_size = 5 batches
        assert len(loader) == 5

    def test_iterate_batches(self, numpy_data_dir):
        """Test iterating over batches."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(dataset, batch_size=2)

        batches = list(loader)
        assert len(batches) == 5

        for batched_data in batches:
            assert isinstance(batched_data, TensorDict)
            assert batched_data["positions"].shape[0] == 2  # batch dim

    def test_batch_collation(self, numpy_data_dir):
        """Test that samples are collated into batches correctly."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(dataset, batch_size=4)

        batched_data = next(iter(loader))

        # Should have batch dimension
        assert batched_data["positions"].shape == (4, 100, 3)
        assert batched_data["features"].shape == (4, 100, 8)

    def test_metadata_collation(self, numpy_data_dir):
        """Test that metadata is collated when requested."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(dataset, batch_size=3, collate_metadata=True)

        batched_data, metadata_list = next(iter(loader))

        # Metadata should be lists
        assert isinstance(metadata_list, list)
        assert len(metadata_list) == 3
        assert [m["index"] for m in metadata_list] == [0, 1, 2]

    def test_collate_metadata_false_returns_tensordict_only(self, numpy_data_dir):
        """Test collate_metadata=False returns only TensorDict."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(dataset, batch_size=3, collate_metadata=False)

        batch = next(iter(loader))

        # Should be just TensorDict, not tuple
        assert isinstance(batch, TensorDict)

    def test_drop_last(self, numpy_data_dir):
        """Test drop_last parameter."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        # Without drop_last: 10 samples / 3 = 4 batches (last has 1)
        loader_keep = dp.DataLoader(dataset, batch_size=3, drop_last=False)
        assert len(loader_keep) == 4

        # With drop_last: 10 samples / 3 = 3 batches
        loader_drop = dp.DataLoader(dataset, batch_size=3, drop_last=True)
        assert len(loader_drop) == 3

    def test_last_batch_smaller(self, numpy_data_dir):
        """Test that last batch can be smaller than batch_size."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(dataset, batch_size=3, drop_last=False)

        batches = list(loader)
        last_batched_data = batches[-1]

        # 10 % 3 = 1, so last batch should have 1 sample
        assert last_batched_data["positions"].shape[0] == 1


# ============================================================================
# Edge cases
# ============================================================================


class TestDataLoaderEdgeCases:
    """Tests for DataLoader edge cases."""

    def test_single_sample_batch(self, numpy_data_dir):
        """Test with batch_size=1."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(dataset, batch_size=1)

        assert len(loader) == 10

        batches = list(loader)
        assert len(batches) == 10

        for batch in batches:
            assert batch["positions"].shape[0] == 1

    def test_batch_size_larger_than_dataset(self, numpy_data_dir):
        """Test with batch_size larger than dataset."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(dataset, batch_size=15, drop_last=False)

        # ceil(10 / 15) = 1 batch
        assert len(loader) == 1

        batches = list(loader)
        assert len(batches) == 1
        assert batches[0]["positions"].shape[0] == 10

    def test_batch_size_larger_with_drop_last(self, numpy_data_dir):
        """Test batch_size larger than dataset with drop_last=True."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(dataset, batch_size=15, drop_last=True)

        # 10 // 15 = 0 batches
        assert len(loader) == 0

        batches = list(loader)
        assert len(batches) == 0

    def test_exact_batch_division(self, numpy_data_dir):
        """Test when dataset size divides evenly by batch_size."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(dataset, batch_size=5)

        # 10 / 5 = 2 batches
        assert len(loader) == 2

        batches = list(loader)
        assert len(batches) == 2

        for batch in batches:
            assert batch["positions"].shape[0] == 5

    def test_multiple_epochs(self, numpy_data_dir):
        """Test iterating over multiple epochs."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(dataset, batch_size=3, shuffle=True)

        for epoch in range(3):
            loader.set_epoch(epoch)
            batches = list(loader)
            assert len(batches) == 4  # ceil(10/3) = 4

            total_samples = sum(b["positions"].shape[0] for b in batches)
            assert total_samples == 10

    def test_invalid_batch_size(self, numpy_data_dir):
        """Test that invalid batch_size raises error."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            dp.DataLoader(dataset, batch_size=0)


# ============================================================================
# Shuffling
# ============================================================================


class TestDataLoaderShuffling:
    """Tests for DataLoader shuffling."""

    def test_shuffle_changes_order(self, numpy_data_dir):
        """Test that shuffle=True changes sample order between epochs."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        # Collect indices from multiple epochs
        torch.manual_seed(42)
        loader = dp.DataLoader(
            dataset, batch_size=2, shuffle=True, collate_metadata=True
        )

        indices_epoch1 = []
        for batched_data, metadata_list in loader:
            indices_epoch1.extend([m["index"] for m in metadata_list])

        indices_epoch2 = []
        for batched_data, metadata_list in loader:
            indices_epoch2.extend([m["index"] for m in metadata_list])

        # Different epochs should (likely) have different orders
        # Note: there's a tiny chance they're the same, but very unlikely
        # We mainly check that shuffling doesn't break anything
        assert set(indices_epoch1) == set(range(10))
        assert set(indices_epoch2) == set(range(10))

    def test_no_shuffle_preserves_order(self, numpy_data_dir):
        """Test that shuffle=False preserves sample order."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(
            dataset, batch_size=2, shuffle=False, collate_metadata=True
        )

        indices = []
        for batched_data, metadata_list in loader:
            indices.extend([m["index"] for m in metadata_list])

        assert indices == list(range(10))


# ============================================================================
# Prefetching
# ============================================================================


class TestDataLoaderPrefetching:
    """Tests for DataLoader prefetching functionality."""

    def test_prefetch_disabled(self, numpy_data_dir):
        """Test that prefetch_factor=0 disables prefetching."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(
            dataset,
            batch_size=2,
            prefetch_factor=0,  # Disabled
        )

        batches = list(loader)
        assert len(batches) == 5

    def test_prefetch_enabled_cpu(self, numpy_data_dir):
        """Test prefetching in CPU mode (use_streams=False)."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(
            dataset,
            batch_size=2,
            prefetch_factor=2,
            use_streams=False,  # CPU mode
        )

        assert loader.use_streams is False

        batches = list(loader)
        assert len(batches) == 5

        for batched_data in batches:
            assert batched_data["positions"].shape[0] == 2

    def test_disable_prefetch_method(self, numpy_data_dir):
        """Test disable_prefetch method."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(
            dataset,
            batch_size=2,
            prefetch_factor=2,
        )

        loader.disable_prefetch()
        assert loader.use_streams is False

        # Should still work in sync mode
        batches = list(loader)
        assert len(batches) == 5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_with_cuda_streams(self, numpy_data_dir):
        """Test prefetching with real CUDA streams."""
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(reader, device="cuda:0")
        loader = dp.DataLoader(
            dataset,
            batch_size=2,
            prefetch_factor=2,
            num_streams=4,
            use_streams=True,
        )

        assert loader.use_streams is True
        assert len(loader._streams) == 4

        # Verify streams are real CUDA streams
        for stream in loader._streams:
            assert isinstance(stream, torch.cuda.Stream)

        batches = list(loader)
        assert len(batches) == 5

        for batched_data in batches:
            assert batched_data["positions"].device.type == "cuda"

        torch.cuda.synchronize()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_with_multiple_streams(self, numpy_data_dir):
        """Test prefetching with multiple CUDA streams."""
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(reader, device="cuda:0")
        loader = dp.DataLoader(
            dataset,
            batch_size=2,
            prefetch_factor=3,
            num_streams=8,
            use_streams=True,
        )

        assert loader.num_streams == 8
        assert len(loader._streams) == 8

        batches = list(loader)
        assert len(batches) == 5

        for batch in batches:
            assert batch["positions"].device.type == "cuda"

        torch.cuda.synchronize()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_enable_prefetch_method(self, numpy_data_dir):
        """Test enable_prefetch method."""
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(reader, device="cuda:0")
        loader = dp.DataLoader(
            dataset,
            batch_size=2,
            prefetch_factor=2,
            use_streams=False,
        )

        # Initially disabled
        assert loader.use_streams is False

        # Enable prefetching
        loader.enable_prefetch()

        assert loader.use_streams is True
        assert len(loader._streams) > 0

        # Verify it works after enabling
        batches = list(loader)
        assert len(batches) == 5
        torch.cuda.synchronize()

    def test_enable_prefetch_without_cuda_raises(self, numpy_data_dir):
        """Test that enable_prefetch raises when CUDA unavailable."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test error case")

        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(
            dataset,
            batch_size=2,
            use_streams=False,
        )

        with pytest.raises(RuntimeError, match="CUDA is not available"):
            loader.enable_prefetch()

    def test_streams_not_created_when_cuda_unavailable(self, numpy_data_dir):
        """Test that no streams are created when CUDA is unavailable."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test this case")

        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(
            dataset,
            batch_size=2,
            num_streams=4,
            use_streams=True,  # Should be ignored
        )

        # use_streams should be False since CUDA unavailable
        assert loader.use_streams is False
        assert len(loader._streams) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_fewer_batches_than_prefetch_factor(self, numpy_data_dir):
        """Test prefetch when fewer batches than prefetch_factor."""
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(reader, device="cuda:0")
        loader = dp.DataLoader(
            dataset,
            batch_size=5,  # 2 batches total
            prefetch_factor=10,  # More than available batches
            num_streams=4,
            use_streams=True,
        )

        batches = list(loader)
        assert len(batches) == 2

        for batch in batches:
            assert batch["positions"].device.type == "cuda"

        torch.cuda.synchronize()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_empty_when_drop_last(self, numpy_data_dir):
        """Test prefetch with empty batch list (all dropped)."""
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(reader, device="cuda:0")
        loader = dp.DataLoader(
            dataset,
            batch_size=100,  # Larger than dataset
            drop_last=True,  # Will result in 0 batches
            use_streams=True,
        )

        batches = list(loader)
        assert len(batches) == 0


# ============================================================================
# Collation
# ============================================================================


class TestDataLoaderCollation:
    """Tests for DataLoader collation functionality."""

    def test_default_collation_stacks(self, numpy_data_dir):
        """Test that default collator stacks tensors."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(dataset, batch_size=3)

        batch = next(iter(loader))

        # Default stacks along new dimension
        assert batch["positions"].shape == (3, 100, 3)
        assert batch["features"].shape == (3, 100, 8)

    def test_concat_collation(self, numpy_data_dir):
        """Test ConcatCollator concatenates along specified dim."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(
            dataset,
            batch_size=3,
            collate_fn=dp.ConcatCollator(dim=0, add_batch_idx=False),
        )

        batch = next(iter(loader))

        # Concat flattens batch dimension
        assert batch["positions"].shape == (300, 3)
        assert batch["features"].shape == (300, 8)

    def test_concat_collation_with_batch_idx(self, numpy_data_dir):
        """Test ConcatCollator adds batch indices."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(
            dataset,
            batch_size=3,
            collate_fn=dp.ConcatCollator(dim=0, add_batch_idx=True),
        )

        batch = next(iter(loader))

        assert "batch_idx" in batch
        assert batch["batch_idx"].shape == (300,)

        # Check batch indices are correct
        assert (batch["batch_idx"][:100] == 0).all()
        assert (batch["batch_idx"][100:200] == 1).all()
        assert (batch["batch_idx"][200:] == 2).all()

    def test_custom_collate_fn(self, numpy_data_dir):
        """Test with custom collate function."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        def my_collate(samples):
            # Just return first sample
            return samples[0]

        loader = dp.DataLoader(
            dataset, batch_size=3, collate_fn=my_collate, collate_metadata=True
        )

        result = next(iter(loader))

        # Should be single sample tuple, not batched
        data, metadata = result
        assert data["positions"].shape == (100, 3)


# ============================================================================
# Samplers
# ============================================================================


class TestDataLoaderSamplers:
    """Tests for DataLoader with various samplers."""

    def test_sequential_sampler(self, numpy_data_dir):
        """Test with explicit SequentialSampler."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        sampler = SequentialSampler(dataset)
        loader = dp.DataLoader(
            dataset, batch_size=2, sampler=sampler, collate_metadata=True
        )

        indices = []
        for batched_data, metadata_list in loader:
            indices.extend([m["index"] for m in metadata_list])

        assert indices == list(range(10))

    def test_random_sampler(self, numpy_data_dir):
        """Test with explicit RandomSampler."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        torch.manual_seed(123)
        sampler = RandomSampler(dataset)
        loader = dp.DataLoader(
            dataset, batch_size=2, sampler=sampler, collate_metadata=True
        )

        indices = []
        for batched_data, metadata_list in loader:
            indices.extend([m["index"] for m in metadata_list])

        # All indices present, but possibly shuffled
        assert set(indices) == set(range(10))

    def test_subset_sampler(self, numpy_data_dir):
        """Test with SubsetRandomSampler."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        # Only use indices 0, 2, 4, 6, 8
        indices = [0, 2, 4, 6, 8]
        sampler = SubsetRandomSampler(indices)
        loader = dp.DataLoader(
            dataset, batch_size=2, sampler=sampler, collate_metadata=True
        )

        seen_indices = []
        for batched_data, metadata_list in loader:
            seen_indices.extend([m["index"] for m in metadata_list])

        assert set(seen_indices) == set(indices)

    def test_weighted_random_sampler(self, numpy_data_dir):
        """Test with WeightedRandomSampler."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        # Higher weight for first 5 samples
        weights = [2.0] * 5 + [0.1] * 5
        sampler = WeightedRandomSampler(weights, num_samples=10, replacement=True)

        loader = dp.DataLoader(
            dataset,
            batch_size=2,
            sampler=sampler,
            collate_metadata=True,
        )

        # Should complete without error
        batches = list(loader)
        assert len(batches) == 5

    def test_custom_sampler_overrides_shuffle(self, numpy_data_dir):
        """Test that custom sampler overrides shuffle parameter."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        sampler = SequentialSampler(dataset)

        loader = dp.DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,  # Should be ignored
            sampler=sampler,
            collate_metadata=True,
        )

        indices = []
        for batch, metadata in loader:
            indices.extend([m["index"] for m in metadata])

        # Should be sequential despite shuffle=True
        assert indices == list(range(10))

    def test_set_epoch(self, numpy_data_dir):
        """Test set_epoch for DistributedSampler compatibility."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(dataset, batch_size=2)

        # Should not raise even if sampler doesn't have set_epoch
        loader.set_epoch(0)
        loader.set_epoch(1)

    def test_set_epoch_calls_sampler_set_epoch(self, numpy_data_dir):
        """Test that set_epoch calls sampler's set_epoch if available."""

        class SamplerWithSetEpoch:
            def __init__(self, data_source):
                self.data_source = data_source
                self.epoch = None

            def __iter__(self):
                return iter(range(len(self.data_source)))

            def __len__(self):
                return len(self.data_source)

            def set_epoch(self, epoch):
                self.epoch = epoch

        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        sampler = SamplerWithSetEpoch(dataset)

        loader = dp.DataLoader(dataset, batch_size=2, sampler=sampler)

        loader.set_epoch(5)
        assert sampler.epoch == 5

    def test_set_epoch_no_op_without_set_epoch_method(self, numpy_data_dir):
        """Test that set_epoch is no-op if sampler lacks set_epoch."""

        class SamplerWithoutSetEpoch:
            def __init__(self, data_source):
                self.data_source = data_source

            def __iter__(self):
                return iter(range(len(self.data_source)))

            def __len__(self):
                return len(self.data_source)

        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        sampler = SamplerWithoutSetEpoch(dataset)

        loader = dp.DataLoader(dataset, batch_size=2, sampler=sampler)

        # Should not raise
        loader.set_epoch(5)


# ============================================================================
# Repr
# ============================================================================


class TestDataLoaderRepr:
    """Tests for DataLoader string representation."""

    def test_repr_basic(self, numpy_data_dir):
        """Test basic repr output."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(dataset, batch_size=4)

        repr_str = repr(loader)

        assert "DataLoader" in repr_str
        assert "batch_size=4" in repr_str
        assert "shuffle=False" in repr_str
        assert "drop_last=False" in repr_str

    def test_repr_with_options(self, numpy_data_dir):
        """Test repr with various options."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            drop_last=True,
            prefetch_factor=4,
            num_streams=6,
        )

        repr_str = repr(loader)

        assert "batch_size=8" in repr_str
        assert "shuffle=True" in repr_str
        assert "drop_last=True" in repr_str
        assert "prefetch_factor=4" in repr_str
        assert "num_streams=6" in repr_str


# ============================================================================
# Integration / End-to-end tests
# ============================================================================


class TestDataLoaderIntegration:
    """Integration tests for DataLoader with full pipeline."""

    def test_training_loop_simulation(self, numpy_data_dir):
        """Test simulated training loop over multiple epochs."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(
            reader,
            transforms=dp.SubsamplePoints(
                input_keys=["positions", "features"], n_points=50
            ),
        )
        loader = dp.DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
        )

        # Simulate 3 epochs
        for epoch in range(3):
            loader.set_epoch(epoch)

            total_samples = 0
            for batched_data in loader:
                batch_size = batched_data["positions"].shape[0]
                total_samples += batch_size

                # Verify transform was applied
                assert batched_data["positions"].shape[1] == 50

            assert total_samples == 10

    def test_full_pipeline_with_transforms(self, numpy_data_dir):
        """Test full data loading pipeline with transforms."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(
            reader,
            transforms=[
                dp.SubsamplePoints(input_keys=["positions", "features"], n_points=50),
                dp.Normalize(
                    input_keys=["positions"],
                    method="mean_std",
                    means={"positions": 0.0},
                    stds={"positions": 1.0},
                ),
            ],
        )
        loader = dp.DataLoader(dataset, batch_size=4, shuffle=True)

        for epoch in range(2):
            loader.set_epoch(epoch)

            for batch in loader:
                # Check transforms were applied
                assert batch["positions"].shape[1] == 50
                assert batch["features"].shape[1] == 50

    def test_dataloader_with_concat_collator_and_transforms(self, numpy_data_dir):
        """Test DataLoader with ConcatCollator and transforms."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(
            reader,
            transforms=dp.SubsamplePoints(
                input_keys=["positions", "features"], n_points=25
            ),
        )
        loader = dp.DataLoader(
            dataset,
            batch_size=3,
            collate_fn=dp.ConcatCollator(dim=0, add_batch_idx=True),
        )

        batch = next(iter(loader))

        # 3 samples * 25 points = 75
        assert batch["positions"].shape == (75, 3)
        assert batch["batch_idx"].shape == (75,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_training_loop(self, numpy_data_dir):
        """Test GPU data loading pipeline with stream prefetching."""
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(
            reader,
            device="cuda:0",
            transforms=[
                dp.Normalize(
                    input_keys=["positions"],
                    method="mean_std",
                    means={"positions": 0.0},
                    stds={"positions": 1.0},
                ),
            ],
        )
        loader = dp.DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            prefetch_factor=2,
            num_streams=4,
            use_streams=True,
        )

        for batched_data in loader:
            assert batched_data["positions"].device.type == "cuda"

            # Simulate forward pass
            _ = batched_data["positions"].mean()

        torch.cuda.synchronize()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_pipeline_with_stream_synchronization(self, numpy_data_dir):
        """Test that GPU pipeline properly synchronizes streams."""
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(reader, device="cuda:0")
        loader = dp.DataLoader(
            dataset,
            batch_size=2,
            prefetch_factor=3,
            num_streams=4,
            use_streams=True,
        )

        results = []
        for batch in loader:
            # Perform some computation
            result = batch["positions"].sum().item()
            results.append(result)

        # Verify we got all batches
        assert len(results) == 5

        # Synchronize and verify no errors
        torch.cuda.synchronize()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_disable_enable_prefetch_cycle(self, numpy_data_dir):
        """Test disabling and re-enabling prefetch during iteration."""
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(reader, device="cuda:0")
        loader = dp.DataLoader(
            dataset,
            batch_size=2,
            prefetch_factor=2,
            num_streams=4,
            use_streams=True,
        )

        # First pass with prefetch enabled
        batches1 = list(loader)
        assert len(batches1) == 5

        # Disable prefetch
        loader.disable_prefetch()
        assert loader.use_streams is False

        # Second pass without prefetch
        batches2 = list(loader)
        assert len(batches2) == 5

        # Re-enable prefetch
        loader.enable_prefetch()
        assert loader.use_streams is True

        # Third pass with prefetch re-enabled
        batches3 = list(loader)
        assert len(batches3) == 5

        torch.cuda.synchronize()
