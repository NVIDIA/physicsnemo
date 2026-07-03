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

from physicsnemo.experimental.models.healda.obs_context import (
    build_pixel_group_map,
    counts_to_cu_seqlens,
    prepare_obs_context,
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


def test_prepare_obs_context_sorts_and_builds_group_map():
    obs = torch.tensor([1.0, 2.0, 3.0])
    float_metadata = torch.tensor([[1.0], [2.0], [3.0]])
    obs_type = torch.tensor([10, 20, 30])
    channel = torch.tensor([11, 21, 31])
    platform = torch.tensor([12, 22, 32])
    flat_idx = torch.tensor([2, 0, 2], dtype=torch.int32)

    context = prepare_obs_context(
        obs=obs,
        float_metadata=float_metadata,
        obs_type=obs_type,
        channel=channel,
        platform=platform,
        flat_idx=flat_idx,
        total_pixels=4,
    )

    assert torch.equal(context.obs, torch.tensor([2.0, 1.0, 3.0]))
    assert torch.equal(
        context.float_metadata.squeeze(-1), torch.tensor([2.0, 1.0, 3.0])
    )
    assert torch.equal(context.obs_type, torch.tensor([20, 10, 30]))
    assert torch.equal(
        context.cu_seqlens_k, torch.tensor([0, 1, 1, 3, 3], dtype=torch.int32)
    )
    assert context.max_seqlen_k == 2
    assert context.group_map is not None


def test_prepare_obs_context_empty_observations():
    context = prepare_obs_context(
        obs=torch.empty(0),
        float_metadata=torch.empty(0, 2),
        obs_type=torch.empty(0, dtype=torch.long),
        channel=torch.empty(0, dtype=torch.long),
        platform=torch.empty(0, dtype=torch.long),
        flat_idx=torch.empty(0, dtype=torch.int32),
        total_pixels=3,
    )

    assert context.obs.numel() == 0
    assert torch.equal(context.cu_seqlens_k, torch.zeros(4, dtype=torch.int32))
    assert context.max_seqlen_k == 0
    assert context.group_map is not None
    assert context.group_map.program_pixels.numel() == 0
