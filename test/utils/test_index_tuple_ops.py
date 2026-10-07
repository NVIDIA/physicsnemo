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

"""Tests for bounded integer tuple operations."""

import itertools

import pytest
import torch

from physicsnemo.utils._index_tuple_ops import (
    _SORTING_NETWORKS,
    _sort_by_network,
    sort_index_tuples,
    unique_index_tuples,
)


def _assert_matches_torch_unique(
    rows: torch.Tensor,
    index_bound: int,
    *,
    return_inverse: bool = True,
    return_counts: bool = True,
) -> None:
    expected = torch.unique(
        rows,
        dim=0,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )
    actual = unique_index_tuples(
        rows,
        index_bound=index_bound,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
            assert torch.equal(actual_tensor, expected_tensor)
    else:
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)


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
    _assert_matches_torch_unique(rows, index_bound=0)


def test_unique_index_tuples_falls_back_when_packing_would_overflow() -> None:
    rows = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [1, 2, 3, 4]])
    _assert_matches_torch_unique(rows, index_bound=10_000_000)


@pytest.mark.parametrize("n_columns", [1, 2, 3, 4])
@pytest.mark.parametrize("index_bound", [5, 17])
def test_unique_index_tuples_randomized_equivalence(
    n_columns: int,
    index_bound: int,
) -> None:
    generator = torch.Generator().manual_seed(1234 + n_columns + index_bound)
    rows = torch.randint(
        low=0,
        high=index_bound,
        size=(200, n_columns),
        generator=generator,
    )
    _assert_matches_torch_unique(rows, index_bound=index_bound)


@pytest.mark.parametrize(
    ("return_inverse", "return_counts"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_unique_index_tuples_return_modes(
    return_inverse: bool,
    return_counts: bool,
) -> None:
    rows = torch.tensor([[0, 2], [0, 2], [1, 3], [2, 4], [1, 3]])
    _assert_matches_torch_unique(
        rows,
        index_bound=5,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )


def test_unique_index_tuples_handles_noncontiguous_rows() -> None:
    base = torch.tensor(
        [
            [0, 99, 1],
            [0, 88, 1],
            [2, 77, 3],
            [0, 66, 1],
        ],
        dtype=torch.long,
    )
    rows = base[:, ::2]
    assert not rows.is_contiguous()
    _assert_matches_torch_unique(rows, index_bound=4)


def test_unique_index_tuples_handles_single_row_and_all_duplicates() -> None:
    _assert_matches_torch_unique(torch.tensor([[3, 1, 4]]), index_bound=5)
    _assert_matches_torch_unique(torch.tensor([[2, 2], [2, 2], [2, 2]]), index_bound=3)


def test_unique_index_tuples_rejects_floating_rows() -> None:
    with pytest.raises(TypeError, match="integer"):
        unique_index_tuples(torch.ones(3, 2), index_bound=2)


def test_unique_index_tuples_rejects_non_2d_rows() -> None:
    with pytest.raises(ValueError, match="2D"):
        unique_index_tuples(torch.ones(2, 2, 2, dtype=torch.long), index_bound=2)


def test_unique_index_tuples_rejects_invalid_index_bound() -> None:
    with pytest.raises(ValueError, match="index_bound"):
        unique_index_tuples(torch.ones(2, 2, dtype=torch.long), index_bound=0)


@pytest.mark.parametrize("n_columns", [1, 2, 3, 4])
def test_sorting_networks_sort_every_binary_input(n_columns: int) -> None:
    """By the 0-1 principle, a network that sorts every 0/1 input sorts every input."""
    rows = torch.tensor(list(itertools.product([0, 1], repeat=n_columns)))
    actual = _sort_by_network(rows, _SORTING_NETWORKS[n_columns])
    assert torch.equal(actual, torch.sort(rows, dim=-1).values)


@pytest.mark.parametrize("n_columns", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_sorting_networks_match_torch_sort(n_columns: int, dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(4321 + n_columns)
    # A small index range forces repeated entries within rows
    rows = torch.randint(
        low=0, high=4, size=(300, n_columns), generator=generator, dtype=dtype
    )
    actual = _sort_by_network(rows, _SORTING_NETWORKS[n_columns])
    assert actual.dtype == dtype
    assert torch.equal(actual, torch.sort(rows, dim=-1).values)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize("n_rows", [1_000, 300_000])
@pytest.mark.parametrize("n_columns", [2, 3, 4, 5])
def test_sort_index_tuples_matches_torch_sort(
    device: str, n_rows: int, n_columns: int
) -> None:
    """Both dispatch paths (torch.sort and the sorting network) sort correctly."""
    generator = torch.Generator().manual_seed(8765 + n_columns)
    rows = torch.randint(0, 50, size=(n_rows, n_columns), generator=generator).to(
        device
    )
    assert torch.equal(sort_index_tuples(rows), torch.sort(rows, dim=-1).values)


def test_sort_index_tuples_handles_batched_noncontiguous_and_empty_rows() -> None:
    generator = torch.Generator().manual_seed(8765)
    rows = torch.randint(0, 100, size=(7, 5, 6), generator=generator)[..., ::2]
    assert not rows.is_contiguous()
    assert torch.equal(
        _sort_by_network(rows, _SORTING_NETWORKS[3]), torch.sort(rows, dim=-1).values
    )

    empty = torch.empty((0, 3), dtype=torch.long)
    assert sort_index_tuples(empty).shape == (0, 3)
