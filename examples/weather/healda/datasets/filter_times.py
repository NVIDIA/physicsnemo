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
import pandas as pd
import numpy as np

from healda.datasets.base import DatasetMetadata


def _get_coarsest_freq(freq1: str, freq2: str) -> str:
    """Get the coarsest (largest time step) frequency between two pandas frequency strings."""
    td1 = pd.Timedelta(freq1)
    td2 = pd.Timedelta(freq2)
    return freq1 if td1 >= td2 else freq2


def get_chunk_aligned_times(
    base_metadata: DatasetMetadata,
    obs_metadata: DatasetMetadata,
    chunk_size: int = 24,
    dropouts: list[tuple[str, str]] = [
        ("2019-01-01", "2020-06-30"),
        ("2022-06-01", "2022-09-30"),
    ],
) -> pd.DatetimeIndex:
    """
    Get chunk-aligned times for a dataset filtered by dropouts. Ensures that starting from the first time
    returned, times come in full chunks of `chunk_size` and are aligned to the chunk boundaries.


    """

    # get max of obs start and base start
    desired_start = max(
        pd.Timestamp(obs_metadata.start), pd.Timestamp(base_metadata.start)
    )
    desired_end = min(pd.Timestamp(obs_metadata.end), pd.Timestamp(base_metadata.end))

    coarsest_freq = _get_coarsest_freq(base_metadata.freq, obs_metadata.freq)

    # Get chunk-aligned start
    base_start = pd.Timestamp(base_metadata.start)
    aligned_start = _find_next_chunk_boundary(
        desired_start, base_start, chunk_size, base_metadata.freq
    )

    # Filter out dropouts
    available_times = _generate_available_times(
        aligned_start, desired_end, coarsest_freq, dropouts
    )

    segments = _extract_chunk_aligned_segments(
        available_times, base_start, chunk_size, base_metadata.freq, obs_metadata.freq
    )
    if not segments:
        raise ValueError("No valid chunk-aligned segments found")

    stacked_times = pd.DatetimeIndex(np.concatenate([seg.values for seg in segments]))

    # validate present in base metadata and obs metadata
    base_valid_times = pd.date_range(
        base_metadata.start, base_metadata.end, freq=base_metadata.freq
    )
    obs_valid_times = pd.date_range(
        obs_metadata.start, obs_metadata.end, freq=obs_metadata.freq
    )

    if not (
        np.all(stacked_times.isin(base_valid_times))
        and np.all(stacked_times.isin(obs_valid_times))
    ):
        raise RuntimeError(
            "Some times in stacked_times are not in base_valid_times or obs_valid_times"
        )

    return stacked_times


def _find_next_chunk_boundary(
    desired_start: pd.Timestamp, dataset_start: pd.Timestamp, chunk_size: int, freq: str
) -> pd.Timestamp:
    """Find the next chunk boundary at or after desired_start."""
    freq_td = pd.Timedelta(freq)

    # Number of timestamps from dataset start
    n_timestamps = int((desired_start - dataset_start) / freq_td)

    # Round up to next chunk boundary if needed
    if n_timestamps % chunk_size != 0:
        n_timestamps = ((n_timestamps // chunk_size) + 1) * chunk_size

    return dataset_start + (n_timestamps * freq_td)


def _generate_available_times(
    start: pd.Timestamp, end: pd.Timestamp, freq: str, dropouts: list[tuple[str, str]]
) -> pd.DatetimeIndex:
    """Generate times excluding dropout periods."""
    # Generate all theoretical times
    all_times = pd.date_range(start, end, freq=freq)

    # Create availability mask
    available_mask = np.ones(len(all_times), dtype=bool)

    # Apply dropout masks
    for dropout_start_str, dropout_end_str in dropouts:
        dropout_start = pd.Timestamp(dropout_start_str)
        dropout_end = pd.Timestamp(dropout_end_str)
        dropout_mask = (all_times >= dropout_start) & (all_times <= dropout_end)
        available_mask &= ~dropout_mask

    return all_times[available_mask]


def _extract_chunk_aligned_segments(
    available_times: pd.DatetimeIndex,
    dataset_start: pd.Timestamp,
    chunk_size: int,
    ds_freq: str,
    obs_freq: str,
) -> list[pd.DatetimeIndex]:
    """Extract contiguous segments that are chunk-aligned."""
    if len(available_times) == 0:
        return []

    segments = []
    freq_td = pd.Timedelta(ds_freq)
    freq_obs_td = pd.Timedelta(obs_freq)

    # Find contiguous regions
    time_series = available_times.to_series()
    time_diffs = time_series.diff()
    gap_threshold = freq_obs_td * 2  # Allow one missing timestamp

    # Identify break points - need to get positional indices, not timestamps
    break_indices = [0]
    # Find where gaps occur and get their positional indices
    gap_mask = time_diffs > gap_threshold
    if gap_mask.any():
        # Get positional indices where gaps occur
        gap_positions = np.where(gap_mask)[0]
        break_indices.extend(gap_positions.tolist())

    break_indices.append(len(available_times))
    # Process each contiguous region
    for start_idx, end_idx in zip(break_indices[:-1], break_indices[1:]):
        segment = available_times[start_idx:end_idx]

        if len(segment) < chunk_size:
            continue

        # Align segment start to chunk boundary
        segment_start = segment[0]
        offset = int((segment_start - dataset_start) / freq_td)
        skip_count = (chunk_size - (offset % chunk_size)) % chunk_size

        if skip_count > 0:
            segment = segment[skip_count:]

        # Trim to chunk-aligned length
        aligned_length = (len(segment) // chunk_size) * chunk_size
        if aligned_length >= chunk_size:
            segment = segment[:aligned_length]
            segments.append(segment)

    return segments
