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

"""Tests for the union-find in ``vectorized_connected_components``.

Callers rely on each label being the smallest index in its component.
"""

import pytest
import torch

from physicsnemo.mesh.utilities._duplicate_detection import (
    vectorized_connected_components,
)

# Fail, rather than warn, if the union-find does not converge.
pytestmark = pytest.mark.filterwarnings("error:Union-find did not converge")


def _labels(pairs: list[list[int]], n_elements: int, device: str) -> list[int]:
    pairs = torch.tensor(pairs, dtype=torch.long, device=device).reshape(-1, 2)
    return vectorized_connected_components(pairs, n_elements).tolist()


def _sequential_labels(pairs: list[list[int]], n_elements: int) -> list[int]:
    """Smallest index in the component of each element, by a sequential union-find."""
    parent = list(range(n_elements))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        root_a, root_b = find(a), find(b)
        parent[max(root_a, root_b)] = min(root_a, root_b)
    return [find(i) for i in range(n_elements)]


def test_no_pairs_leave_each_element_alone(device) -> None:
    assert _labels([], 4, device) == [0, 1, 2, 3]


def test_disconnected_groups_take_their_smallest_index(device) -> None:
    # Groups {1, 6}, {3, 7, 9} and {5, 8}. Elements 0, 2 and 4 are alone.
    labels = _labels([[7, 3], [8, 5], [3, 9], [6, 1]], 10, device)
    assert labels == [0, 1, 2, 3, 4, 5, 1, 3, 5, 3]


def test_repeated_and_self_pairs_change_nothing(device) -> None:
    pairs = [[4, 2], [6, 4]]
    noisy = [[4, 2], [2, 4], [5, 5], [6, 4], [4, 2], [0, 0], [6, 4]]
    assert _labels(pairs, 7, device) == [0, 1, 2, 3, 2, 5, 2]
    assert _labels(noisy, 7, device) == [0, 1, 2, 3, 2, 5, 2]


def test_shuffled_chain_takes_its_smallest_index(device) -> None:
    # A chain through elements 5 to 999 in random order. Elements 0 to 4 are alone.
    chain = 5 + torch.randperm(995, generator=torch.Generator().manual_seed(0))
    pairs = torch.stack([chain[:-1], chain[1:]], dim=1)

    labels = vectorized_connected_components(pairs.to(device), 1000)

    assert labels.tolist() == [0, 1, 2, 3, 4] + [5] * 995


@pytest.mark.parametrize("n_pairs", [10, 100, 1000])
def test_random_graphs_match_sequential_union_find(n_pairs, device) -> None:
    """From mostly isolated elements to one large component, on 200 elements."""
    generator = torch.Generator().manual_seed(n_pairs)
    pairs = torch.randint(0, 200, (n_pairs, 2), generator=generator)

    labels = vectorized_connected_components(pairs.to(device), 200)

    assert labels.tolist() == _sequential_labels(pairs.tolist(), 200)
