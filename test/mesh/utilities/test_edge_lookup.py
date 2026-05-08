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

from physicsnemo.mesh.utilities._edge_lookup import find_edges_in_reference


def test_find_edges_with_explicit_bound_matches_implicit_bound() -> None:
    reference = torch.tensor([[0, 10], [2, 7], [4, 8]], dtype=torch.long)
    query = torch.tensor([[7, 2], [4, 8], [9, 10]], dtype=torch.long)

    implicit_indices, implicit_matches = find_edges_in_reference(reference, query)
    explicit_indices, explicit_matches = find_edges_in_reference(
        reference,
        query,
        index_bound=11,
    )

    assert torch.equal(explicit_indices, implicit_indices)
    assert torch.equal(explicit_matches, implicit_matches)


def test_find_edges_matches_reversed_query_edges() -> None:
    reference = torch.tensor([[1, 4], [2, 5], [6, 7]], dtype=torch.long)
    query = torch.tensor([[5, 2], [7, 6], [4, 1]], dtype=torch.long)

    indices, matches = find_edges_in_reference(reference, query, index_bound=8)

    assert matches.tolist() == [True, True, True]
    assert indices.tolist() == [1, 2, 0]


def test_find_edges_reports_unmatched_edges() -> None:
    reference = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    query = torch.tensor([[1, 0], [3, 4]], dtype=torch.long)

    indices, matches = find_edges_in_reference(reference, query, index_bound=5)

    assert matches.tolist() == [True, False]
    assert indices[0].item() == 0


def test_find_edges_handles_empty_reference_and_query() -> None:
    reference = torch.empty((0, 2), dtype=torch.long)
    query = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    indices, matches = find_edges_in_reference(reference, query, index_bound=3)

    assert indices.shape == (2,)
    assert not matches.any()

    indices, matches = find_edges_in_reference(
        torch.tensor([[0, 1]], dtype=torch.long),
        torch.empty((0, 2), dtype=torch.long),
        index_bound=2,
    )
    assert indices.shape == (0,)
    assert matches.shape == (0,)


def test_find_edges_uses_sparse_strict_index_bound() -> None:
    reference = torch.tensor([[10, 20], [30, 40]], dtype=torch.long)
    query = torch.tensor([[40, 30], [20, 10]], dtype=torch.long)

    indices, matches = find_edges_in_reference(reference, query, index_bound=100)

    assert matches.tolist() == [True, True]
    assert indices.tolist() == [1, 0]
