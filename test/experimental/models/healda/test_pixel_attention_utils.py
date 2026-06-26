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
"""CPU tests for the pixel-attention packing utilities."""

import torch

from physicsnemo.experimental.models.healda.pixel_attention_utils import (
    build_pixel_group_map,
    counts_to_cu_seqlens,
    sort_and_pack,
)


def test_counts_to_cu_seqlens():
    counts = torch.tensor([5, 0, 3, 4], dtype=torch.int64)
    cu = counts_to_cu_seqlens(counts)
    assert cu.dtype == torch.int32
    assert cu.tolist() == [0, 5, 5, 8, 12]


def test_sort_and_pack_groups_by_pixel():
    # Three pixels, observations interleaved; sorting must group each pixel's
    # source indices contiguously (order within a pixel is unconstrained).
    flat_idx = torch.tensor([2, 0, 1, 2, 0, 2], dtype=torch.int32)
    sorted_order, counts = sort_and_pack(flat_idx, total_pixels=3)
    assert counts.tolist() == [2, 1, 3]
    grouped = flat_idx[sorted_order.long()]
    assert grouped.tolist() == [0, 0, 1, 2, 2, 2]


def test_build_pixel_group_map_pairs_small_pixels():
    # Docstring example: counts [5, 0, 3, 4, 200], nonzero median 4, thresh 8 ->
    # large=[4], small=[0, 2, 3] -> programs [[4], [0, 2], [3]].
    cu = counts_to_cu_seqlens(torch.tensor([5, 0, 3, 4, 200], dtype=torch.int64))
    gm = build_pixel_group_map(cu)
    assert gm.program_ptr.tolist() == [0, 1, 3, 4]
    assert gm.program_pixels.tolist() == [4, 0, 2, 3]


def test_build_pixel_group_map_empty():
    cu = torch.zeros(6, dtype=torch.int32)
    gm = build_pixel_group_map(cu)
    assert gm.program_ptr.tolist() == [0]
    assert gm.program_pixels.numel() == 0
