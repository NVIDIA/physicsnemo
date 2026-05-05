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

from physicsnemo.mesh.neighbors._adjacency import build_adjacency_from_pairs


def test_build_adjacency_asymmetric_source_target_spaces() -> None:
    sources = torch.tensor([0, 0, 2, 2, 2, 4], dtype=torch.long)
    targets = torch.tensor([3, 1, 4, 0, 2, 1], dtype=torch.long)

    adjacency = build_adjacency_from_pairs(
        source_indices=sources,
        target_indices=targets,
        n_sources=5,
        n_targets=7,
    )

    assert adjacency.to_list() == [[1, 3], [], [0, 2, 4], [], [1]]


def test_build_adjacency_preserves_duplicate_pairs() -> None:
    sources = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    targets = torch.tensor([2, 2, 1, 0], dtype=torch.long)

    adjacency = build_adjacency_from_pairs(
        source_indices=sources,
        target_indices=targets,
        n_sources=2,
        n_targets=3,
    )

    assert adjacency.to_list() == [[1, 2, 2], [0]]


def test_build_adjacency_empty_pairs_with_explicit_target_bound() -> None:
    adjacency = build_adjacency_from_pairs(
        source_indices=torch.empty(0, dtype=torch.long),
        target_indices=torch.empty(0, dtype=torch.long),
        n_sources=3,
        n_targets=10,
    )

    assert adjacency.to_list() == [[], [], []]


def test_build_adjacency_large_target_bound_fallback_branch() -> None:
    sources = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    targets = torch.tensor([3, 2, 1, 0], dtype=torch.long)

    adjacency = build_adjacency_from_pairs(
        source_indices=sources,
        target_indices=targets,
        n_sources=2,
        n_targets=1 << 62,
    )

    assert adjacency.to_list() == [[0, 2], [1, 3]]
