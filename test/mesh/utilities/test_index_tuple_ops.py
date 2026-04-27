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

import torch

from physicsnemo.mesh.utilities._index_tuple_ops import unique_index_tuples


def _assert_matches_torch_unique(rows: torch.Tensor, index_bound: int) -> None:
    expected = torch.unique(rows, dim=0, return_inverse=True, return_counts=True)
    actual = unique_index_tuples(
        rows,
        index_bound=index_bound,
        return_inverse=True,
        return_counts=True,
    )

    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


def test_unique_index_tuples_matches_torch_unique_for_edges() -> None:
    rows = torch.tensor(
        [
            [0, 1],
            [0, 2],
            [0, 1],
            [2, 3],
            [1, 3],
            [2, 3],
        ],
        dtype=torch.long,
    )
    _assert_matches_torch_unique(rows, index_bound=4)


def test_unique_index_tuples_matches_torch_unique_for_faces() -> None:
    rows = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 2],
            [2, 3, 4],
            [1, 2, 4],
        ],
        dtype=torch.long,
    )
    _assert_matches_torch_unique(rows, index_bound=5)


def test_unique_index_tuples_handles_empty_rows() -> None:
    rows = torch.empty((0, 2), dtype=torch.long)
    _assert_matches_torch_unique(rows, index_bound=1)


def test_unique_index_tuples_falls_back_when_packing_would_overflow() -> None:
    rows = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [1, 2, 3, 4]])
    _assert_matches_torch_unique(rows, index_bound=10_000_000)
