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

"""Tests for Dataset class.

This module consolidates all Dataset tests, using real CUDA streams
for GPU-related tests instead of mocks.
"""

import pytest
import torch
from tensordict import TensorDict

import physicsnemo.datapipes as dp

# ============================================================================
# Basic Dataset functionality
# ============================================================================


class TestDatasetBasic:
    """Tests for basic Dataset functionality."""

    def test_create_dataset(self, numpy_data_dir):
        """Test creating a Dataset."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        assert len(dataset) == 10

    def test_dataset_get_sample(self, numpy_data_dir):
        """Test getting a sample from the Dataset."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        data, metadata = dataset[0]
        assert isinstance(data, TensorDict)
        assert "positions" in data

    def test_dataset_iteration(self, numpy_data_dir):
        """Test iterating over the Dataset."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        samples = list(dataset)
        assert len(samples) == 10

    def test_dataset_field_names(self, numpy_data_dir):
        """Test field_names property."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        assert "positions" in dataset.field_names
        assert "features" in dataset.field_names

    def test_dataset_context_manager(self, numpy_data_dir):
        """Test Dataset as context manager."""
        reader = dp.NumpyReader(numpy_data_dir)
        with dp.Dataset(reader) as dataset:
            data, metadata = dataset[0]
            assert "positions" in data


# ============================================================================
# Device Handling
# ============================================================================


class TestDatasetDeviceHandling:
    """Tests for Dataset device handling."""

    def test_device_as_torch_device(self, numpy_data_dir):
        """Test passing device as torch.device object."""
        reader = dp.NumpyReader(numpy_data_dir)
        device = torch.device("cpu")

        dataset = dp.Dataset(reader, device=device)

        assert dataset.target_device == device

    def test_device_as_string(self, numpy_data_dir):
        """Test passing device as string."""
        reader = dp.NumpyReader(numpy_data_dir)

        dataset = dp.Dataset(reader, device="cpu")

        assert dataset.target_device == torch.device("cpu")

    def test_device_none(self, numpy_data_dir):
        """Test passing device as None."""
        reader = dp.NumpyReader(numpy_data_dir)

        dataset = dp.Dataset(reader, device=None)

        assert dataset.target_device is None

    def test_device_auto_without_cuda(self, numpy_data_dir):
        """Test device='auto' falls back to cpu when CUDA not available."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test CPU fallback")

        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader, device="auto")

        assert dataset.target_device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_auto_with_cuda(self, numpy_data_dir):
        """Test device='auto' uses CUDA when available."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader, device="auto")

        # Should be cuda:0 or another CUDA device
        assert dataset.target_device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dataset_to_cuda_device(self, numpy_data_dir):
        """Test Dataset transfers data to CUDA device."""
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(reader, device="cuda:0")

        data, metadata = dataset[0]
        assert data["positions"].device.type == "cuda"

        torch.cuda.synchronize()


# ============================================================================
# Transform Handling
# ============================================================================


class TestDatasetTransforms:
    """Tests for Dataset transform handling."""

    def test_dataset_single_transform(self, numpy_data_dir):
        """Test Dataset with a single transform."""
        reader = dp.NumpyReader(numpy_data_dir)
        norm = dp.Normalize(
            input_keys=["positions"],
            method="mean_std",
            means={"positions": 0.0},
            stds={"positions": 1.0},
        )
        dataset = dp.Dataset(reader, transforms=norm)

        data, metadata = dataset[0]
        assert "positions" in data

    def test_dataset_transform_list(self, numpy_data_dir):
        """Test Dataset with a list of transforms."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(
            reader,
            transforms=[
                dp.SubsamplePoints(input_keys=["positions", "features"], n_points=50),
            ],
        )

        data, metadata = dataset[0]
        assert data["positions"].shape[0] == 50
        assert data["features"].shape[0] == 50

    def test_dataset_multiple_transforms_creates_compose(self, numpy_data_dir):
        """Test that multiple transforms creates a Compose."""
        reader = dp.NumpyReader(numpy_data_dir)

        transforms = [
            dp.SubsamplePoints(input_keys=["positions", "features"], n_points=50),
            dp.Normalize(
                input_keys=["positions"],
                method="mean_std",
                means={"positions": 0.0},
                stds={"positions": 1.0},
            ),
        ]

        dataset = dp.Dataset(reader, transforms=transforms)

        # Should have created a Compose
        assert isinstance(dataset.transforms, dp.Compose)
        assert len(dataset.transforms) == 2

        # Verify it works
        data, _ = dataset[0]
        assert data["positions"].shape[0] == 50

    def test_dataset_single_transform_in_list(self, numpy_data_dir):
        """Test that single transform in list is used directly."""
        reader = dp.NumpyReader(numpy_data_dir)

        transform = dp.SubsamplePoints(
            input_keys=["positions", "features"], n_points=50
        )

        dataset = dp.Dataset(reader, transforms=[transform])

        # Should use the transform directly, not Compose
        assert dataset.transforms is transform

    def test_dataset_compose_transforms(self, numpy_data_dir):
        """Test Dataset with explicit Compose."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(
            reader,
            transforms=dp.Compose(
                [
                    dp.SubsamplePoints(
                        input_keys=["positions", "features"], n_points=50
                    ),
                    dp.Normalize(
                        input_keys=["positions"],
                        method="mean_std",
                        means={"positions": 0.0},
                        stds={"positions": 1.0},
                    ),
                ]
            ),
        )

        data, metadata = dataset[0]
        assert data["positions"].shape[0] == 50

    def test_dataset_empty_transforms_list(self, numpy_data_dir):
        """Test Dataset with empty transforms list."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader, transforms=[])

        data, metadata = dataset[0]
        # Should work, no transforms applied
        assert "positions" in data
        assert dataset.transforms is None

    def test_transforms_moved_to_device(self, numpy_data_dir):
        """Test that transforms are moved to target device."""
        reader = dp.NumpyReader(numpy_data_dir)

        transform = dp.Normalize(
            input_keys=["positions"],
            method="mean_std",
            means={"positions": torch.tensor(0.0)},
            stds={"positions": torch.tensor(1.0)},
        )

        dataset = dp.Dataset(reader, transforms=transform, device="cpu")

        # Transform should have been moved to CPU
        assert dataset.transforms.device == torch.device("cpu")


# ============================================================================
# Prefetching
# ============================================================================


class TestDatasetPrefetching:
    """Tests for Dataset prefetching functionality."""

    def test_prefetch_single(self, numpy_data_dir):
        """Test prefetching a single sample."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        # Prefetch index 0
        dataset.prefetch(0)

        # Should have 1 prefetch in flight (may complete quickly)
        assert dataset.prefetch_count >= 0

        # Get should use prefetched result
        data, metadata = dataset[0]
        assert "positions" in data

    def test_prefetch_batch(self, numpy_data_dir):
        """Test prefetching multiple samples."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        # Prefetch multiple indices
        dataset.prefetch_batch([0, 1, 2, 3])

        # Get samples
        for i in range(4):
            data, metadata = dataset[i]
            assert metadata["index"] == i

    def test_prefetch_non_prefetched_index(self, numpy_data_dir):
        """Test getting a non-prefetched index loads synchronously."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        # Prefetch index 0
        dataset.prefetch(0)

        # Get non-prefetched index (should load synchronously)
        data, metadata = dataset[5]
        assert metadata["index"] == 5

    def test_prefetch_skips_if_already_in_flight(self, numpy_data_dir):
        """Test that prefetch skips if index is already being fetched."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        # Prefetch same index twice
        dataset.prefetch(0)
        initial_count = dataset.prefetch_count

        dataset.prefetch(0)  # Should be a no-op
        assert dataset.prefetch_count == initial_count

        # Still should be able to get the data
        data, metadata = dataset[0]
        assert metadata["index"] == 0

    def test_prefetch_with_transforms(self, numpy_data_dir):
        """Test prefetching with transforms applied."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(
            reader,
            transforms=dp.SubsamplePoints(
                input_keys=["positions", "features"], n_points=50
            ),
        )

        dataset.prefetch(0)
        data, metadata = dataset[0]

        # Transform should have been applied
        assert data["positions"].shape[0] == 50

    def test_prefetch_then_getitem_workflow(self, numpy_data_dir):
        """Test the typical prefetch then getitem workflow."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        # Prefetch several samples
        for i in range(5):
            dataset.prefetch(i)

        # Retrieve them
        for i in range(5):
            data, metadata = dataset[i]
            assert metadata["index"] == i

        # Prefetch count should be 0 after retrieving all
        assert dataset.prefetch_count == 0


# ============================================================================
# Prefetch with CUDA streams
# ============================================================================


class TestDatasetPrefetchWithStreams:
    """Tests for Dataset prefetching with real CUDA streams."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_with_stream(self, numpy_data_dir):
        """Test prefetching with a CUDA stream."""
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(reader, device="cuda:0")

        stream = torch.cuda.Stream()
        dataset.prefetch(0, stream=stream)

        data, metadata = dataset[0]
        assert data["positions"].device.type == "cuda"

        torch.cuda.synchronize()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_batch_with_streams(self, numpy_data_dir):
        """Test prefetch_batch with multiple CUDA streams."""
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(reader, device="cuda:0")

        streams = [torch.cuda.Stream() for _ in range(4)]
        dataset.prefetch_batch([0, 1, 2, 3], streams=streams)

        for i in range(4):
            data, metadata = dataset[i]
            assert data["positions"].device.type == "cuda"

        torch.cuda.synchronize()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_batch_with_stream_cycling(self, numpy_data_dir):
        """Test prefetch_batch cycles through streams correctly."""
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(reader, device="cuda:0")

        # Use fewer streams than indices to test cycling
        streams = [torch.cuda.Stream() for _ in range(2)]
        dataset.prefetch_batch([0, 1, 2, 3, 4], streams=streams)

        # All samples should be retrievable
        for i in range(5):
            data, metadata = dataset[i]
            assert metadata["index"] == i
            assert data["positions"].device.type == "cuda"

        torch.cuda.synchronize()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_with_stream_and_transforms(self, numpy_data_dir):
        """Test prefetching with CUDA stream and transforms."""
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        transform = dp.SubsamplePoints(
            input_keys=["positions", "features"], n_points=50
        )
        dataset = dp.Dataset(reader, transforms=transform, device="cuda:0")

        stream = torch.cuda.Stream()
        dataset.prefetch(0, stream=stream)

        data, metadata = dataset[0]

        assert data["positions"].shape[0] == 50
        assert data["positions"].device.type == "cuda"

        torch.cuda.synchronize()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_multiple_streams_parallel(self, numpy_data_dir):
        """Test that multiple streams can work in parallel."""
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(reader, device="cuda:0")

        # Create multiple streams
        num_streams = 4
        streams = [torch.cuda.Stream() for _ in range(num_streams)]

        # Prefetch with different streams
        for i in range(num_streams):
            dataset.prefetch(i, stream=streams[i])

        # Retrieve all results
        results = []
        for i in range(num_streams):
            data, metadata = dataset[i]
            results.append((data, metadata))

        # Verify all results are correct
        for i, (data, metadata) in enumerate(results):
            assert metadata["index"] == i
            assert data["positions"].device.type == "cuda"

        torch.cuda.synchronize()


# ============================================================================
# Cancel Prefetch
# ============================================================================


class TestDatasetCancelPrefetch:
    """Tests for Dataset cancel_prefetch functionality."""

    def test_prefetch_cancel_all(self, numpy_data_dir):
        """Test canceling all prefetches."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        dataset.prefetch_batch([0, 1, 2, 3])
        dataset.cancel_prefetch()

        # Prefetch count should be 0 after cancel
        assert dataset.prefetch_count == 0

    def test_prefetch_cancel_specific(self, numpy_data_dir):
        """Test canceling a specific prefetch."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        dataset.prefetch(0)
        dataset.prefetch(1)
        dataset.cancel_prefetch(0)

        # Should still be able to get both indices
        # (index 0 synchronously, index 1 from prefetch)
        data0, metadata0 = dataset[0]
        data1, metadata1 = dataset[1]

        assert metadata0["index"] == 0
        assert metadata1["index"] == 1

    def test_prefetch_cancel_nonexistent_index(self, numpy_data_dir):
        """Test canceling a prefetch index that doesn't exist."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        dataset.prefetch(0)

        # Should not raise
        dataset.cancel_prefetch(99)

        # Original should still work
        data, metadata = dataset[0]
        assert metadata["index"] == 0


# ============================================================================
# Close and Cleanup
# ============================================================================


class TestDatasetClose:
    """Tests for Dataset close and cleanup functionality."""

    def test_close_stops_prefetch(self, numpy_data_dir):
        """Test that close stops prefetching."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        dataset.prefetch_batch([0, 1, 2, 3])
        dataset.close()

        # Should not raise, prefetch should be stopped
        assert dataset.prefetch_count == 0

    def test_close_shuts_down_executor(self, numpy_data_dir):
        """Test that close shuts down the executor."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        # Trigger executor creation
        dataset.prefetch(0)
        assert dataset._executor is not None

        dataset.close()
        assert dataset._executor is None

    def test_close_without_executor(self, numpy_data_dir):
        """Test that close works when executor was never created."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        # Never use prefetch, so no executor
        assert dataset._executor is None

        # Should not raise
        dataset.close()

    def test_context_manager_cleans_up(self, numpy_data_dir):
        """Test that context manager properly cleans up resources."""
        reader = dp.NumpyReader(numpy_data_dir)

        with dp.Dataset(reader) as dataset:
            # Start some prefetches
            dataset.prefetch(0)
            dataset.prefetch(1)
            _ = dataset._ensure_executor()

        # After context exit, executor should be shut down
        assert dataset._executor is None
        assert dataset.prefetch_count == 0


# ============================================================================
# Error Handling
# ============================================================================


class TestDatasetErrors:
    """Tests for Dataset error handling."""

    def test_invalid_reader_type(self):
        """Test that invalid reader type raises error."""
        with pytest.raises(TypeError, match="must be a Reader"):
            dp.Dataset("not a reader")

    def test_invalid_transforms_type(self, numpy_data_dir):
        """Test that invalid transforms type raises error."""
        reader = dp.NumpyReader(numpy_data_dir)

        with pytest.raises(TypeError, match="must be Transform"):
            dp.Dataset(reader, transforms="not a transform")

    def test_invalid_transforms_type_dict(self, numpy_data_dir):
        """Test that dict as transforms raises error."""
        reader = dp.NumpyReader(numpy_data_dir)

        with pytest.raises(TypeError, match="must be Transform"):
            dp.Dataset(reader, transforms={"invalid": "type"})


# ============================================================================
# Repr
# ============================================================================


class TestDatasetRepr:
    """Tests for Dataset string representation."""

    def test_dataset_repr(self, numpy_data_dir):
        """Test basic repr output."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)

        repr_str = repr(dataset)
        assert "Dataset" in repr_str
        assert "NumpyReader" in repr_str

    def test_dataset_repr_with_transforms(self, numpy_data_dir):
        """Test repr with transforms."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(
            reader,
            transforms=dp.Normalize(
                input_keys=["positions"],
                method="mean_std",
                means={"positions": 0.0},
                stds={"positions": 1.0},
            ),
        )

        repr_str = repr(dataset)
        assert "Normalize" in repr_str


# ============================================================================
# Integration Tests
# ============================================================================


class TestDatasetIntegration:
    """Integration tests for Dataset."""

    def test_full_pipeline_with_device_and_transforms(self, numpy_data_dir):
        """Test full pipeline with device transfer and transforms."""
        reader = dp.NumpyReader(numpy_data_dir)

        transforms = [
            dp.SubsamplePoints(input_keys=["positions", "features"], n_points=25),
            dp.Normalize(
                input_keys=["positions"],
                method="mean_std",
                means={"positions": 0.0},
                stds={"positions": 1.0},
            ),
        ]

        dataset = dp.Dataset(reader, transforms=transforms, device="cpu")

        data, metadata = dataset[0]

        assert data["positions"].shape[0] == 25
        assert data["positions"].device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_gpu_pipeline(self, numpy_data_dir):
        """Test full GPU pipeline with prefetching."""
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)

        transforms = [
            dp.SubsamplePoints(input_keys=["positions", "features"], n_points=50),
            dp.Normalize(
                input_keys=["positions"],
                method="mean_std",
                means={"positions": 0.0},
                stds={"positions": 1.0},
            ),
        ]

        dataset = dp.Dataset(reader, transforms=transforms, device="cuda:0")

        # Prefetch with streams
        streams = [torch.cuda.Stream() for _ in range(2)]
        dataset.prefetch_batch([0, 1, 2, 3], streams=streams)

        # Retrieve results
        for i in range(4):
            data, metadata = dataset[i]
            assert data["positions"].shape[0] == 50
            assert data["positions"].device.type == "cuda"

        torch.cuda.synchronize()

    def test_iterate_all_samples(self, numpy_data_dir):
        """Test iterating through all samples."""
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(
            reader,
            transforms=dp.SubsamplePoints(
                input_keys=["positions", "features"], n_points=50
            ),
        )

        count = 0
        for data, metadata in dataset:
            assert data["positions"].shape[0] == 50
            count += 1

        assert count == 10

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_iteration_with_transforms(self, numpy_data_dir):
        """Test GPU iteration with transforms applied."""
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(
            reader,
            transforms=dp.SubsamplePoints(
                input_keys=["positions", "features"], n_points=25
            ),
            device="cuda:0",
        )

        for i, (data, metadata) in enumerate(dataset):
            assert data["positions"].shape[0] == 25
            assert data["positions"].device.type == "cuda"

        torch.cuda.synchronize()
