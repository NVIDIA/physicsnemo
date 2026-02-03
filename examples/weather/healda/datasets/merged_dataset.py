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
import asyncio
import logging
import math
from typing import Callable

import numpy as np
import pandas as pd
import torch
from zarr.core.sync import sync

from utils import profiling

from . import datetime_utils

logger = logging.getLogger(__name__)


class _FrameIndexGenerator:
    """Handles frame index generation with striding, permuting, and model rank slicing."""

    def __init__(
        self,
        times,
        time_length: int,
        frame_step: int,
        model_rank: int,
        model_world_size: int,
    ):
        """
        Args:
            times: Time array to split into contiguous segments
            time_length: Number of frames per window
            frame_step: Step size between frames
            model_rank: Model rank for distributed training
            model_world_size: Total number of model ranks
        """
        self.time_length = time_length
        self.frame_step = frame_step
        self.model_rank = model_rank
        self.model_world_size = model_world_size

        # Split times into contiguous segments and compute sizes
        self.segments = _split_array_contiguous(times)
        self.sizes = [len(segment) for segment in self.segments]

        self.total_samples = sum(self.sizes)

        # Calculate valid lengths for each segment
        frames_per_window = (time_length - 1) * frame_step + 1
        self.segment_valid_lengths = []
        for segment in self.segments:
            segment_valid_length = len(segment) - frames_per_window + 1
            if segment_valid_length > 0:
                self.segment_valid_lengths.append(segment_valid_length)
            else:
                self.segment_valid_lengths.append(0)

        # Precompute cumulative sizes for efficient mapping
        self.cumulative_valid_sizes = [0] + list(np.cumsum(self.segment_valid_lengths))
        self.cumulative_sizes = [0] + list(np.cumsum(self.sizes))
        self.valid_length = sum(self.segment_valid_lengths)

    def generate_frame_indices(self, sample_indices: torch.Tensor) -> list[int]:
        """Generate frame indices from sample indices with striding and model rank slicing.

        Args:
            sample_indices: Tensor of logical sample indices for each sample in the batch

        Returns:
            List of frame indices to load
        """
        frame_idxs = []
        for sample_idx in sample_indices:
            # Map logical sample index to physical frame index
            physical_idx = self._map_logical_to_physical(sample_idx)

            # Create frame range with striding
            frames = list(
                range(
                    physical_idx,
                    physical_idx + self.time_length * self.frame_step,
                    self.frame_step,
                )
            )
            # Apply model rank slicing
            n = self.time_length // self.model_world_size
            frames = frames[self.model_rank * n : (self.model_rank + 1) * n]
            frame_idxs.append(frames)
        return frame_idxs

    def _map_logical_to_physical(self, logical_idx: int) -> int:
        """Map logical sample index to physical frame index across segments."""
        if logical_idx >= self.total_samples:
            raise IndexError(
                f"Sample index {logical_idx} out of bounds for {self.total_samples} samples"
            )

        # Find which segment this logical index belongs to
        segment_idx = 0
        for i, cum_size in enumerate(self.cumulative_valid_sizes[1:], 1):
            if logical_idx < cum_size:
                segment_idx = i - 1
                break

        # Calculate offset within the segment
        segment_start = self.cumulative_sizes[segment_idx]
        offset_within_segment = logical_idx - self.cumulative_valid_sizes[segment_idx]

        # Return the physical frame index in the original times array
        return segment_start + offset_within_segment

    def get_valid_length(self) -> int:
        """Get the total valid length across all segments."""
        return self.valid_length


def _validate_times(times, world_size, time_length, frame_step):
    if len(times) < world_size:
        raise ValueError(f"Not enough times provided. Received {len(times)=}.")

    if time_length == 1 and frame_step != 1:
        raise ValueError("Frame_step must be 1 for image setting")


class _MergedLoader:
    def __init__(self, loaders) -> None:
        self._loaders = loaders

    async def sel_time(self, time) -> dict[str, np.ndarray]:
        # Standardize time to np.ndarray of np.datetime64
        arrays = await asyncio.gather(
            *[loader.sel_time(time) for loader in self._loaders]
        )
        data = {}
        for d in arrays:
            data.update(d)
        return data


def _split(x, rank, world_size, drop_extra=True):
    n = len(x)
    base = n // world_size
    rem = n % world_size

    if drop_extra:
        samples_per_rank = base
        x = x[: base * world_size]
        start = rank * base
    else:
        # give the first rem ranks one extra sample
        if rank < rem:
            samples_per_rank = base + 1
            start = rank * samples_per_rank
        else:
            samples_per_rank = base
            start = rem * (base + 1) + (rank - rem) * base

    return x[start : start + samples_per_rank]


class TimeMergedDataset(torch.utils.data.IterableDataset):
    """Merge several loader objects in time and apply transforms.

    This is used to join several datasets along time, and grab data in a chunked manner.

    ``time_loaders`` is a list of objects with this interface::

        class Loader:

            async def sel_time(self, times) -> dict[str, np.ndarray]:
                pass

    ``chunk_size`` should ideally be larger than the chunking of each dataset.

    ``transform`` is a function that prepares the raw loaded data for the model::

        def transform(
            times: list[pd.Timestamp],
            data: list[dict[str, np.ndarray]]
        ) -> dict[str, Any]

    When `time_length = 1` and `frame_step = 1`, this collapses to the image case.
    """

    def __init__(
        self,
        times,
        # for performance times should be in sequence
        *,
        time_loaders,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        chunk_size: int = 48,
        transform: Callable,
        infinite: bool = True,
        time_length: int = 1,
        frame_step: int = 1,
        window_stride: int = 1,
    ):
        _validate_times(times, world_size, time_length, frame_step)

        frames_per_window = (time_length - 1) * frame_step + 1
        self._loader = _MergedLoader(time_loaders)
        self.rank = rank
        self.world_size = world_size
        self.set_times(times)  # Shard times across ranks

        if len(self._times) < chunk_size:
            raise ValueError(
                f"Sharded times too small for chunk size. Need {chunk_size} "
                f"frames but only got {len(self._times)}"
            )

        self.shuffle = shuffle
        self.transform = transform
        self.chunk_size = chunk_size
        self.infinite = infinite

        self.time_length = time_length
        self.frame_step = frame_step
        self.window_stride = window_stride

        self._generator = None

        max_valid_idx = len(times) - self.chunk_size
        self.max_valid_chunk_idx = max_valid_idx // self.chunk_size

        self.overlap = frames_per_window - 1

    @property
    def times(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self._times)

    def set_times(self, times):
        self._times = _split(datetime_utils.as_numpy(times), self.rank, self.world_size)

    def _load_chunk(self, chunk: int):
        return sync(self._loader.sel_time(self._times_for_chunk(chunk)))

    def _times_for_chunk(self, chunk: int) -> np.ndarray:
        return self._times[
            chunk * self.chunk_size : (chunk + 1) * self.chunk_size + self.overlap
        ]

    def __iter__(self):
        if self.infinite:
            while True:
                yield from self._iter()
        else:
            yield from self._iter()

    def __len__(self):
        return len(self._times)

    def _generator_shuffle(self, arr, worker_info=None):
        if self._generator is None:
            if worker_info:
                seed = worker_info.seed
            else:
                seed = np.random.randint(0, 2**31) + self.rank

            self._generator = np.random.default_rng(seed=(seed % 2**32))
        self._generator.shuffle(arr)

    def _iter(self):
        num_chunks = math.ceil(len(self._times) / self.chunk_size)
        chunk_idxs = np.arange(num_chunks)

        info = torch.utils.data.get_worker_info()
        num_workers = 1 if info is None else info.num_workers
        worker_id = 0 if info is None else info.id

        # Shard chunks across the data workers. Shard before shuffle so that all workers
        # have the same sharding pattern and each is assigned a unique set of chunks
        chunk_idxs = _split(chunk_idxs, worker_id, num_workers, drop_extra=False)

        if self.shuffle:
            self._generator_shuffle(chunk_idxs, info)

        for chunk_idx in chunk_idxs:
            if chunk_idx > self.max_valid_chunk_idx:
                continue

            arr = self._load_chunk(chunk_idx)
            times_for_chunk = self._times_for_chunk(chunk_idx)

            max_window_start = (
                len(times_for_chunk) - (self.time_length - 1) * self.frame_step
            )

            window_starts = np.arange(0, max_window_start, self.window_stride)
            if self.shuffle:
                self._generator_shuffle(window_starts, info)

            for i, start_idx in enumerate(window_starts):
                frame_idxs = range(
                    start_idx,
                    start_idx + self.time_length * self.frame_step,
                    self.frame_step,
                )

                frames = []
                timestamps = []
                for idx in frame_idxs:
                    time = times_for_chunk[idx]
                    arr_i = {k: v[idx] for k, v in arr.items()}
                    timestamp = pd.Timestamp(time)
                    cftimestamp = datetime_utils.as_cftime(timestamp)
                    frames.append(arr_i)
                    timestamps.append(cftimestamp)

                window_tensor = self.transform(timestamps, frames)

                yield window_tensor


class TimeMergedMapStyle(torch.utils.data.Dataset):
    """Map-style dataset wrapping time-loaders with transform and caching support.

    Applies transforms either across all batches together or on individual frames.
    Supports model parallelism to shard data across the time axis for ranks within
    the model group.
    """

    def __init__(
        self,
        times,
        *,
        time_loaders,
        time_length: int = 1,
        frame_step: int = 1,
        transform: Callable,
        cache_chunk_size: int = 0,
        model_rank=0,
        model_world_size=1,
        batch_transform=None,
    ):
        """
        Args:
            cache_chunk_size: if nonzero, then cache data in this chunk size, so that
                data cn be accessed efficiently in sequence.
            batch_size: if provided

        """
        _validate_times(times, model_world_size, time_length, frame_step)
        self.times = times
        self.transform = transform
        self.batch_transform = batch_transform
        self.time_length = time_length
        self.frame_step = frame_step
        self.model_rank = model_rank
        self.model_world_size = model_world_size
        self._loader = _MergedLoader(time_loaders)

        self._frame_indexer = _FrameIndexGenerator(
            times, time_length, frame_step, model_rank, model_world_size
        )

        # Get valid length from frame indexer
        self.valid_length = self._frame_indexer.get_valid_length()
        if self.valid_length <= 0:
            frames_per_window = (self.time_length - 1) * self.frame_step + 1
            raise ValueError(
                f"Dataset too small for window length. Need {frames_per_window} "
                f"frames but segments have lengths {self._frame_indexer.sizes}"
            )
        self.cache_chunk_size = cache_chunk_size
        self._cache_id = None  # cache id is based on idx//chunk_size
        self._cache_start = 0
        self._cache_end = 0
        self._cache_data = None

    def __len__(self):
        return self._frame_indexer.get_valid_length()

    def _load(self, frame_idxs):
        if not self.cache_chunk_size:
            window_times = self.times[list(frame_idxs)]
            window_data = sync(self._loader.sel_time(window_times))
            return [
                {k: v[i] for k, v in window_data.items()}
                for i in range(len(window_times))
            ]

        frames = []
        # TODO this caching logic only works well if frame_idxs is sequential.
        # otherwise it will results in many cache misses
        for i in frame_idxs:
            cache_id = i // self.cache_chunk_size
            if self._cache_id != cache_id:
                self._cache_start = cache_id * self.cache_chunk_size
                self._cache_end = min(
                    self._cache_start + self.cache_chunk_size, len(self.times)
                )
                window_times = self.times[self._cache_start : self._cache_end]
                window_data = sync(self._loader.sel_time(window_times))
                self._cache_data = [
                    {k: v[i] for k, v in window_data.items()}
                    for i in range(len(window_times))
                ]

                self._cache_id = cache_id
            frames.append(self._cache_data[i - self._cache_start])
        return frames

    def __getitem__(self, idx):
        result = self.__getitems__([idx])
        if self.batch_transform:
            return result
        else:
            return result[0]

    def _batch_transform(self, times, frames):
        if self.batch_transform:
            return self.batch_transform(times, frames)
        elif self.transform:
            output = []
            for sample_times, sample_frames in zip(times, frames):
                output.append(self.transform(sample_times, sample_frames))
            return output

    @profiling.nvtx
    def _get_times_and_frames(self, idx):
        if min(idx) < 0 or max(idx) >= self.valid_length:
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {self.valid_length}"
            )

        batch_size = len(idx)

        frame_idxs = self._frame_indexer.generate_frame_indices(idx)
        flat_frame_idxs = sum(frame_idxs, start=[])

        frames = self._load(flat_frame_idxs)
        window_times = self.times[flat_frame_idxs]

        timestamps = []
        for i, time in enumerate(window_times):
            timestamp = pd.Timestamp(time)
            cftimestamp = datetime_utils.as_cftime(timestamp)
            timestamps.append(cftimestamp)

        def reshape(list):
            n = len(list) // batch_size
            return [[list[n * i + j] for j in range(n)] for i in range(batch_size)]

        timestamps = reshape(timestamps)
        frames = reshape(frames)

        return timestamps, frames

    @profiling.nvtx
    def __getitems__(self, idx):
        timestamps, frames = self._get_times_and_frames(idx)
        return self._batch_transform(timestamps, frames)


def _split_array_contiguous(x):
    d = x[1] - x[0]
    segments = []
    start = 0
    for i in range(1, x.size):
        if (x[i] - x[i - 1]) != d:
            segments.append(x[start:i])
            start = i

    if start < x.size:
        segments.append(x[start:])

    return segments
