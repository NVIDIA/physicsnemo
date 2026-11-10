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

"""Indexing an Adjacency selects sources and keeps their neighbor lists."""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.neighbors import Adjacency
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from test.mesh.mesh.test_slicing_sync import _cuda_sync_budget


@pytest.mark.parametrize("cells_dtype", [torch.int32, torch.int64])
def test_indexing_returns_the_neighbor_lists_of_the_sources(device, cells_dtype):
    sphere = sphere_icosahedral.load(subdivisions=2, device=device)
    mesh = Mesh(points=sphere.points, cells=sphere.cells.to(cells_dtype))
    adjacency = mesh.get_point_to_points_adjacency()
    sources = torch.tensor([7, 0, 161, 7], device=device)  # out of order, repeated

    selected = adjacency[sources]

    neighbor_lists = adjacency.to_list()
    assert selected.to_list() == [neighbor_lists[i] for i in sources.tolist()]
    assert selected.indices.dtype == cells_dtype
    assert selected.offsets.dtype == torch.int64
    assert selected.device == adjacency.device


def test_indexing_takes_the_selections_of_a_tensor(device):
    # Neighbor lists [[0, 1, 2], [3], [], [4, 5]]
    adjacency = Adjacency(
        offsets=torch.tensor([0, 3, 4, 4, 6], device=device),
        indices=torch.tensor([0, 1, 2, 3, 4, 5], device=device),
    )
    mask = torch.tensor([True, False, False, True], device=device)

    assert adjacency[[2, 0]].to_list() == [[], [0, 1, 2]]
    assert adjacency[(2, 0)].to_list() == [[], [0, 1, 2]]
    assert adjacency[1:].to_list() == [[3], [], [4, 5]]
    assert adjacency[mask].to_list() == [[0, 1, 2], [4, 5]]
    assert adjacency[(True, False, False, True)].to_list() == [[0, 1, 2], [4, 5]]
    for nothing in (torch.empty(0, dtype=torch.long, device=device), [], ()):
        empty = adjacency[nothing]
        assert empty.n_sources == 0
        assert empty.indices.dtype == adjacency.indices.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_indexing_sync_budget():
    """Only the output size and the validation of the result synchronize."""
    mesh = sphere_icosahedral.load(subdivisions=3, device="cuda")
    adjacency = mesh.get_point_to_points_adjacency()
    sources = torch.randperm(mesh.n_points, device="cuda")[:100]
    adjacency[sources]  # warm up
    torch.cuda.synchronize()

    with _cuda_sync_budget(2):
        adjacency[sources]
    torch.cuda.synchronize()
