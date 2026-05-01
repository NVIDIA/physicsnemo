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
"""Temporal indexing infrastructure for HealDA datasets.

Provides frame-level indexing with temporal windowing, striding, and
model-parallel rank slicing.  Used by ``ObsERA5Dataset`` (and potentially
other map-style datasets) to convert a flat sample index into the set of
physical frame indices needed for a single training window.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Segment splitting
# ---------------------------------------------------------------------------


def split_array_contiguous(x):
    """Split *x* into sub-arrays at points where the step size changes.

    This detects gaps in a time array (e.g. year boundaries, missing data)
    and returns a list of contiguous segments.
    """
    if x.size <= 1:
        return [x] if x.size == 1 else []

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


# ---------------------------------------------------------------------------
# Frame index generator
# ---------------------------------------------------------------------------


class FrameIndexGenerator:
    """Generate frame indices with striding, permutation, and model-rank slicing.

    Given a 1-D time array (possibly with gaps), this class:

    1. Splits the array into contiguous segments.
    2. Computes the number of valid sliding windows per segment.
    3. Maps a logical sample index to the corresponding physical frame indices,
       applying temporal striding and model-rank slicing.

    Args:
        times: 1-D array of timestamps (used only for contiguity detection).
        time_length: Number of frames per window.
        frame_step: Step size between consecutive frames in a window.
        model_rank: This rank's index for model-parallel time slicing.
        model_world_size: Total number of model-parallel ranks.
    """

    def __init__(
        self,
        times,
        time_length: int,
        frame_step: int,
        model_rank: int,
        model_world_size: int,
    ):
        self.time_length = time_length
        self.frame_step = frame_step
        self.model_rank = model_rank
        self.model_world_size = model_world_size

        self.segments = split_array_contiguous(times)
        self.sizes = [len(segment) for segment in self.segments]
        self.total_samples = sum(self.sizes)

        frames_per_window = (time_length - 1) * frame_step + 1
        self.segment_valid_lengths = []
        for segment in self.segments:
            valid = len(segment) - frames_per_window + 1
            self.segment_valid_lengths.append(max(valid, 0))

        self.cumulative_valid_sizes = [0] + list(
            np.cumsum(self.segment_valid_lengths)
        )
        self.cumulative_sizes = [0] + list(np.cumsum(self.sizes))
        self.valid_length = sum(self.segment_valid_lengths)

    def generate_frame_indices(self, sample_indices: torch.Tensor) -> list[list[int]]:
        """Generate frame indices from sample indices.

        For each logical sample index, returns the physical frame indices after
        applying striding and model-rank slicing.

        Args:
            sample_indices: Tensor of logical sample indices.

        Returns:
            List of frame index lists, one per sample.
        """
        frame_idxs = []
        for sample_idx in sample_indices:
            physical_idx = self._map_logical_to_physical(sample_idx)
            frames = list(
                range(
                    physical_idx,
                    physical_idx + self.time_length * self.frame_step,
                    self.frame_step,
                )
            )
            # Model-parallel rank slicing
            n = self.time_length // self.model_world_size
            frames = frames[self.model_rank * n : (self.model_rank + 1) * n]
            frame_idxs.append(frames)
        return frame_idxs

    def _map_logical_to_physical(self, logical_idx: int) -> int:
        """Map a logical sample index to a physical frame index across segments."""
        if logical_idx >= self.valid_length:
            raise IndexError(
                f"Sample index {logical_idx} out of bounds "
                f"for {self.valid_length} valid samples"
            )

        segment_idx = 0
        for i, cum_size in enumerate(self.cumulative_valid_sizes[1:], 1):
            if logical_idx < cum_size:
                segment_idx = i - 1
                break

        segment_start = self.cumulative_sizes[segment_idx]
        offset_within_segment = logical_idx - self.cumulative_valid_sizes[segment_idx]
        return segment_start + offset_within_segment

    def get_valid_length(self) -> int:
        """Total number of valid sample windows across all segments."""
        return self.valid_length


# ---------------------------------------------------------------------------
# Multi-coordinate index
# ---------------------------------------------------------------------------


class MultiCoordIndex:
    """Map a flat integer index to multi-dimensional xarray coordinates.

    Combines arbitrary sample dimensions (e.g. ensemble members) with a
    temporal frame dimension managed by a ``FrameIndexGenerator``.

    Args:
        sample_dims: Names of the non-temporal sample dimensions.
        sample_sizes: Sizes of each sample dimension.
        frame_dim: Name of the temporal dimension.
        frame_indexer: A ``FrameIndexGenerator`` instance.
    """

    def __init__(self, sample_dims, sample_sizes, frame_dim, frame_indexer):
        self.sample_dims = sample_dims
        self.sample_sizes = sample_sizes
        self.sequence_dim = frame_dim
        self._frame_indexer = frame_indexer

    def __len__(self):
        n = 1
        for s in self.sample_sizes:
            n *= s
        n *= self._frame_indexer.get_valid_length()
        return n

    def __getitem__(self, i):
        shape = [*self.sample_sizes, self._frame_indexer.get_valid_length()]
        *index, seq_index = np.unravel_index(i, shape)
        frames = self._frame_indexer.generate_frame_indices([seq_index])[0]
        coords = dict(zip(self.sample_dims, index))
        coords[self.sequence_dim] = frames
        return coords


def get_flat_indexer(
    ds,
    sample_dims,
    frame_dim,
    time_length,
    frame_step,
    model_rank,
    model_world_size,
):
    """Create a ``MultiCoordIndex`` for an xarray Dataset.

    Args:
        ds: xarray Dataset with the relevant dimensions.
        sample_dims: List of non-temporal dimension names.
        frame_dim: Name of the temporal dimension.
        time_length: Frames per window.
        frame_step: Step between frames.
        model_rank: Model-parallel rank.
        model_world_size: Model-parallel world size.
    """
    times = ds[frame_dim].values
    frame_indexer = FrameIndexGenerator(
        times=times,
        time_length=time_length,
        frame_step=frame_step,
        model_rank=model_rank,
        model_world_size=model_world_size,
    )
    sample_sizes = [ds.sizes[dim] for dim in sample_dims]
    return MultiCoordIndex(sample_dims, sample_sizes, frame_dim, frame_indexer)
