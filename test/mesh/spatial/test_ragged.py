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

"""Tests for _ragged_arange segmented tensor utility."""

import pytest
import torch
from torch._dynamo.utils import counters

from physicsnemo.mesh.spatial._ragged import _ragged_arange


@pytest.mark.parametrize(
    "starts, counts",
    [
        pytest.param([0, 5, 10], [3, 2, 4], id="basic"),
        pytest.param([7], [5], id="single_segment"),
        pytest.param([0, 1, 2, 3], [1, 1, 1, 1], id="all_ones"),
        pytest.param([0, 10], [10, 3], id="unequal"),
        pytest.param([100, 200, 300], [1, 1, 1], id="large_starts"),
        pytest.param([10, 20, 30], [2, 0, 3], id="zero_middle"),
        pytest.param([10, 20, 30], [0, 0, 3], id="zero_leading"),
        pytest.param([10, 20, 30], [3, 0, 0], id="zero_trailing"),
        pytest.param([10, 20, 30, 40, 50], [2, 0, 0, 3, 0], id="zero_interleaved"),
        pytest.param([10, 20, 30], [0, 0, 0], id="zero_all"),
        pytest.param([10], [0], id="zero_single"),
        pytest.param([10, 20, 30], [1, 0, 1], id="zero_between_units"),
        pytest.param([], [], id="empty"),
    ],
)
def test_ragged_arange_correctness(starts: list[int], counts: list[int]):
    """Verify positions and seg_ids match the naive per-segment arange."""
    starts_t = torch.tensor(starts)
    counts_t = torch.tensor(counts)

    positions, seg_ids = _ragged_arange(starts_t, counts_t)

    # Build expected output the obvious way
    pos_parts = [torch.arange(s, s + c) for s, c in zip(starts, counts)]
    seg_parts = [torch.full((c,), i, dtype=torch.long) for i, c in enumerate(counts)]
    expected_pos = (
        torch.cat(pos_parts) if pos_parts else torch.empty(0, dtype=torch.long)
    )
    expected_seg = (
        torch.cat(seg_parts) if seg_parts else torch.empty(0, dtype=torch.long)
    )

    assert torch.equal(positions, expected_pos)
    assert torch.equal(seg_ids, expected_seg)


def test_ragged_arange_explicit_total():
    """When total is passed, it should be used instead of counts.sum()."""
    starts = torch.tensor([0, 5, 10])
    counts = torch.tensor([3, 2, 4])

    pos1, seg1 = _ragged_arange(starts, counts)
    pos2, seg2 = _ragged_arange(starts, counts, total=9)

    assert torch.equal(pos1, pos2)
    assert torch.equal(seg1, seg2)


@pytest.mark.parametrize(
    "starts, counts",
    [
        pytest.param([0, 5, 10], [3, 2, 4], id="no_zeros"),
        pytest.param([10, 20, 30], [2, 0, 3], id="with_zeros"),
    ],
)
def test_ragged_arange_no_graph_break_with_explicit_total(
    starts: list[int],
    counts: list[int],
):
    """searchsorted implementation + explicit total should produce zero graph breaks."""

    def fn(starts_t, counts_t, total_holder):
        pos, seg = _ragged_arange(starts_t, counts_t, total=total_holder.shape[0])
        return pos.sum() + seg.sum()

    starts_t = torch.tensor(starts)
    counts_t = torch.tensor(counts)
    total_holder = torch.empty(int(counts_t.sum()))

    counters.clear()
    compiled = torch.compile(fn, dynamic=True, backend="eager")
    compiled(starts_t, counts_t, total_holder)

    n_breaks = (
        sum(counters["graph_break"].values()) if counters.get("graph_break") else 0
    )
    assert n_breaks == 0, f"Expected 0 graph breaks, got {n_breaks}"
