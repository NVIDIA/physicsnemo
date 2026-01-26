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
import functools

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import datetime_utils
from datasets.merged_dataset import (
    TimeMergedDataset,
    TimeMergedMapStyle,
    _FrameIndexGenerator,
    _split_array_contiguous,
)


class MockLoader:
    def __init__(
        self,
        num_frames: int,
        target_channels: int,
        condition_channels: int,
        num_pixels: int,
        start_date: str,
    ):
        # Create data where each frame's values equal its index
        # Shape: (num_frames, num_channels, 1, num_pixels)
        self.data = (
            torch.arange(num_frames)
            .float()
            .view(num_frames, 1, 1, 1)
            .expand(-1, target_channels + condition_channels, -1, num_pixels)
        )
        self.target_channels = target_channels
        self.start_time = pd.Timestamp(start_date)

    async def sel_time(self, times):
        # Convert timestamps to indices (hours since start)
        indices = [(t - self.start_time).total_seconds() / 3600 for t in times]
        indices = [int(i) for i in indices]
        return {
            "target": self.data[indices, : self.target_channels],  # (C, 1, X)
            "condition": self.data[indices, self.target_channels :],
        }


def temporal_stack(timestamp, frames):
    out = {key: torch.cat([f[key] for f in frames], dim=1) for key in frames[0].keys()}
    out["timestamp"] = datetime_utils.as_timestamp(timestamp)
    return out


@pytest.mark.parametrize(
    "time_length,frame_step,window_stride",
    [
        (1, 1, 2),  # Image
        (1, 1, 1),  # Image
        (4, 3, 2),  # Video
        (6, 4, 1),  # Video
    ],
)
def test_time_merged_dataset(time_length, frame_step, window_stride):
    # Setup small dataset where the total length is chunk_size + padding
    # This way we can verify both in-chunk and between-chunk behavior
    chunk_size = time_length * frame_step * window_stride + 10
    num_frames = (
        chunk_size + (time_length - 1) * frame_step
    )  # Enough frames for one full chunk
    target_channels = 5
    condition_channels = 1
    num_pixels = 16

    start_date = "2025-01-01"
    times = pd.date_range(start_date, periods=num_frames, freq="h")

    loader = MockLoader(
        num_frames, target_channels, condition_channels, num_pixels, start_date
    )

    transform = functools.partial(
        temporal_stack,
    )

    # Test with shuffle=False
    dataset = TimeMergedDataset(
        times=times,
        time_loaders=[loader],
        transform=transform,
        time_length=time_length,
        frame_step=frame_step,
        window_stride=window_stride,
        chunk_size=chunk_size,
        shuffle=False,
        infinite=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
    )

    # Collect all samples from the dataset
    samples = []
    for sample in dataloader:
        samples.append(sample)

    frames_per_window = (time_length - 1) * frame_step + 1
    valid_length = num_frames - frames_per_window + 1
    expected_windows = (valid_length + window_stride - 1) // window_stride
    assert len(samples) == expected_windows, (
        f"Expected {expected_windows} windows, got {len(samples)}"
    )

    for sample in samples:
        assert sample["target"].shape == (1, target_channels, time_length, num_pixels)
        assert sample["condition"].shape == (
            1,
            condition_channels,
            time_length,
            num_pixels,
        )

    # Get start indices of each window and verify they're window_stride apart
    start_indices = []
    for sample in samples:
        start_idx = int(sample["target"][0, 0, 0, 0].item())
        start_indices.append(start_idx)

    expected_starts = list(range(0, chunk_size * window_stride, window_stride))[
        :expected_windows
    ]
    assert start_indices == expected_starts, (
        "Without shuffle, windows should be in exact sequential order"
    )

    # Test with shuffle=True
    dataset = TimeMergedDataset(
        times=times,
        time_loaders=[loader],
        transform=transform,
        time_length=time_length,
        frame_step=frame_step,
        window_stride=window_stride,
        chunk_size=chunk_size,
        shuffle=True,
        infinite=True,  # Need infinite for multiple chunks
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
    )

    # Get two chunks worth of data and verify they are in a different order
    chunk1_indices = []
    chunk2_indices = []
    dataloader_iter = iter(dataloader)

    for _ in range(chunk_size // window_stride):
        sample = next(dataloader_iter)
        start_idx = int(sample["target"][0, 0, 0, 0].item())
        chunk1_indices.append(start_idx)

    for _ in range(chunk_size // window_stride):
        sample = next(dataloader_iter)
        start_idx = int(sample["target"][0, 0, 0, 0].item())
        chunk2_indices.append(start_idx)

    assert sorted(chunk1_indices) == sorted(chunk2_indices), (
        "Chunks should contain same indices"
    )

    if len(chunk1_indices) != 1:
        assert chunk1_indices != chunk2_indices, (
            "Chunks should be in different orders when shuffled"
        )


@pytest.mark.parametrize(
    "time_length,frame_step",
    [
        (1, 1),  # Image case
        (4, 2),  # Video case
    ],
)
def test_time_merged_map_style_dataset(time_length, frame_step):
    num_frames = 100
    target_channels = 5
    condition_channels = 1
    num_pixels = 16

    start_date = "2025-01-01"
    times = pd.date_range(start_date, periods=num_frames, freq="h")

    loader = MockLoader(
        num_frames, target_channels, condition_channels, num_pixels, start_date
    )

    transform = functools.partial(
        temporal_stack,
    )

    dataset = TimeMergedMapStyle(
        times=times,
        time_loaders=[loader],
        time_length=time_length,
        frame_step=frame_step,
        transform=transform,
    )

    # Test length
    frames_per_window = (time_length - 1) * frame_step + 1
    expected_length = num_frames - frames_per_window + 1
    assert len(dataset) == expected_length

    # Test first window
    sample = dataset[0]
    assert sample["target"].shape == (target_channels, time_length, num_pixels)
    assert sample["condition"].shape == (condition_channels, time_length, num_pixels)

    # Check values
    expected_indices = list(
        range(0, time_length * frame_step, frame_step)
    )  # [0] for image, [0,2,4,6] for video
    for t, expected_idx in enumerate(expected_indices):
        assert torch.all(sample["target"][:, t] == expected_idx)
        assert torch.all(sample["condition"][:, t] == expected_idx)

    # Test last window
    last_idx = len(dataset) - 1
    last_sample = dataset[last_idx]
    assert last_sample["target"].shape == (target_channels, time_length, num_pixels)
    assert last_sample["condition"].shape == (
        condition_channels,
        time_length,
        num_pixels,
    )

    # Test out of bounds
    try:
        dataset[len(dataset)]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass


def test_map_style_caching():
    start_date = "2025-01-01"
    num_frames = 16
    times = pd.date_range(start_date, periods=num_frames, freq="h")
    num_frames = 100
    target_channels = 5
    condition_channels = 1
    num_pixels = 16

    loader = MockLoader(
        num_frames, target_channels, condition_channels, num_pixels, start_date
    )

    transform = functools.partial(
        temporal_stack,
    )

    dataset = TimeMergedMapStyle(
        times=times,
        time_loaders=[loader],
        time_length=1,
        frame_step=1,
        cache_chunk_size=8,
        transform=transform,
    )

    for i in range(10):
        out = dataset[i]
        time = out["timestamp"].astype("datetime64[s]")
        assert time == times[i]

    assert dataset._cache_data is not None


def test_split_array_contiguous():
    input = np.arange(10)
    (output,) = _split_array_contiguous(input)
    assert np.all(input == output)

    input = np.array([0, 1, 2, 5, 6])
    (out1, out2) = _split_array_contiguous(input)
    assert out1.tolist() == [0, 1, 2]
    assert out2.tolist() == [5, 6]


def test_map_style_getitems():
    start_date = "2025-01-01"
    num_frames = 16
    times = pd.date_range(start_date, periods=num_frames, freq="h")
    num_frames = 100
    target_channels = 5
    condition_channels = 1
    num_pixels = 16
    batch_size = 3

    loader = MockLoader(
        num_frames, target_channels, condition_channels, num_pixels, start_date
    )

    transform = functools.partial(
        temporal_stack,
    )

    class BatchTransform:
        def __init__(self) -> None:
            self._called = False

        def __call__(self, times, frames):
            self._called = True
            return torch.ones(len(times), len(times[0]))

    batch_transform = BatchTransform()

    dataset = TimeMergedMapStyle(
        times=times,
        time_loaders=[loader],
        time_length=2,
        frame_step=1,
        cache_chunk_size=8,
        transform=transform,
        batch_transform=batch_transform,
    )

    loader = torch.utils.data.DataLoader(
        dataset, collate_fn=lambda x: x, batch_size=batch_size
    )
    out = next(iter(loader))
    assert out.shape == (batch_size, dataset.time_length)
    assert batch_transform._called


def test_frame_index_generator():
    """Test the _FrameIndexGenerator functionality."""
    # Test basic functionality
    times = np.arange(100)  # Simple integer array [0, 1, 2, ..., 99]
    generator = _FrameIndexGenerator(
        times=times, time_length=3, frame_step=2, model_rank=0, model_world_size=1
    )

    start_indices = torch.tensor([0, 10])
    frame_idxs = generator.generate_frame_indices(start_indices)

    expected = [[0, 2, 4], [10, 12, 14]]
    assert frame_idxs == expected

    # Test with model rank slicing
    generator_multi_rank = _FrameIndexGenerator(
        times=times, time_length=4, frame_step=1, model_rank=1, model_world_size=2
    )

    start_indices = torch.tensor([5])
    frame_idxs = generator_multi_rank.generate_frame_indices(start_indices)

    # Expected: [7, 8] for rank 1 of 2 with time_length=4
    # Full range would be [5, 6, 7, 8], rank 1 gets second half: [7, 8]
    expected = [7, 8]
    assert frame_idxs[0] == expected


def test_frame_index_generator_multiple_segments():
    """Test _FrameIndexGenerator with multiple segments."""
    times = np.concatenate(
        [
            np.arange(0, 10),  # [0, 1, 2, ..., 9]
            np.arange(20, 35),  # [20, 21, 22, ..., 34]
        ]
    )

    generator = _FrameIndexGenerator(
        times=times, time_length=3, frame_step=1, model_rank=0, model_world_size=1
    )

    # [7, 8, 9] is the last valid sample in the first segment
    assert times[generator._map_logical_to_physical(0)] == 0
    assert times[generator._map_logical_to_physical(1)] == 1
    assert times[generator._map_logical_to_physical(7)] == 7
    assert times[generator._map_logical_to_physical(8)] == 20

    assert all(times[generator.generate_frame_indices([7])[0]] == [7, 8, 9])
    assert all(times[generator.generate_frame_indices([8])[0]] == [20, 21, 22])
