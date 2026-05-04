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
"""Tests for frame indexing and temporal windowing."""

import numpy as np
import torch

from physicsnemo.experimental.datapipes.healda.indexing import (
    FrameIndexGenerator,
    split_array_contiguous,
)


def test_split_array_contiguous_single_segment():
    arr = np.arange(10)
    (output,) = split_array_contiguous(arr)
    assert np.all(arr == output)


def test_split_array_contiguous_two_segments():
    arr = np.array([0, 1, 2, 5, 6])
    out1, out2 = split_array_contiguous(arr)
    assert out1.tolist() == [0, 1, 2]
    assert out2.tolist() == [5, 6]


def test_frame_index_generator_basic():
    """Test basic frame index generation with striding."""
    times = np.arange(100)
    generator = FrameIndexGenerator(
        times=times, time_length=3, frame_step=2, model_rank=0, model_world_size=1
    )

    start_indices = torch.tensor([0, 10])
    frame_idxs = generator.generate_frame_indices(start_indices)

    expected = [[0, 2, 4], [10, 12, 14]]
    assert frame_idxs == expected


def test_frame_index_generator_model_rank_slicing():
    """Test model-parallel rank slicing of frame indices."""
    times = np.arange(100)
    generator = FrameIndexGenerator(
        times=times, time_length=4, frame_step=1, model_rank=1, model_world_size=2
    )

    start_indices = torch.tensor([5])
    frame_idxs = generator.generate_frame_indices(start_indices)

    # Full range: [5, 6, 7, 8], rank 1 gets second half: [7, 8]
    assert frame_idxs[0] == [7, 8]


def test_frame_index_generator_multiple_segments():
    """Test frame index generation across non-contiguous segments."""
    times = np.concatenate(
        [
            np.arange(0, 10),  # [0, 1, ..., 9]
            np.arange(20, 35),  # [20, 21, ..., 34]
        ]
    )

    generator = FrameIndexGenerator(
        times=times, time_length=3, frame_step=1, model_rank=0, model_world_size=1
    )

    # Verify mapping across segment boundary
    assert times[generator._map_logical_to_physical(0)] == 0
    assert times[generator._map_logical_to_physical(1)] == 1
    assert times[generator._map_logical_to_physical(7)] == 7
    assert times[generator._map_logical_to_physical(8)] == 20

    assert all(times[generator.generate_frame_indices([7])[0]] == [7, 8, 9])
    assert all(times[generator.generate_frame_indices([8])[0]] == [20, 21, 22])


def test_frame_index_generator_valid_length():
    """Test valid length computation."""
    times = np.arange(20)
    generator = FrameIndexGenerator(
        times=times, time_length=3, frame_step=2, model_rank=0, model_world_size=1
    )
    # frames_per_window = (3-1)*2 + 1 = 5
    # valid_length = 20 - 5 + 1 = 16
    assert generator.get_valid_length() == 16
