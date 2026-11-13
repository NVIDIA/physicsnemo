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
from physicsnemo.mesh.primitives.curves import circle_2d
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.mesh.primitives.volumes import cube_volume
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


def _sphere_with_an_isolated_point(device):
    """Triangles with int32 cells, and one point that no cell uses."""
    sphere = sphere_icosahedral.load(subdivisions=2, device=device)
    isolated = torch.full((1, 3), 5.0, device=device)
    return Mesh(points=torch.cat([sphere.points, isolated]), cells=sphere.cells.int())


def _tetrahedra(device):
    return cube_volume.load(subdivisions=3, device=device)


def _curve(device):
    return circle_2d.load(device=device)


def _selections(n_sources, device):
    """One selection of each kind that indexing an Adjacency takes."""
    mask = torch.zeros(n_sources, dtype=torch.bool, device=device)
    mask[::3] = True
    return [
        torch.randint(n_sources, (20,), device=device),  # out of order, repeated
        [n_sources - 1, 0, -1],
        (n_sources - 1, 0),
        mask,
        slice(1, None, 4),
        slice(5, 2),  # empty
        torch.empty(0, dtype=torch.long, device=device),
        [],
    ]


def _topology_keys(mesh):
    topology = mesh._cache.get("topology", None)
    return set() if topology is None else set(topology.keys())


@pytest.mark.parametrize("load", [_sphere_with_an_isolated_point, _tetrahedra, _curve])
@pytest.mark.parametrize(
    "kind", ["point_to_points", "point_to_cells", "cell_to_cells", "cell_to_points"]
)
def test_indices_gives_the_lists_of_the_full_adjacency(device, load, kind):
    torch.manual_seed(0)
    mesh = load(device)
    n_sources = mesh.n_points if kind.startswith("point") else mesh.n_cells
    codimensions = (
        range(1, mesh.n_manifold_dims + 1) if kind == "cell_to_cells" else [None]
    )
    for codimension in codimensions:
        kwargs = {} if codimension is None else {"adjacency_codimension": codimension}
        full = getattr(mesh, f"get_{kind}_adjacency")(**kwargs)
        for indices in _selections(n_sources, device):
            want = full[indices]
            fresh = Mesh(points=mesh.points, cells=mesh.cells)  # nothing cached
            # Built from the cells near the sources, then indexed from the cache
            for source in (fresh, mesh):
                got = getattr(source, f"get_{kind}_adjacency")(
                    indices=indices, **kwargs
                )
                assert torch.equal(got.offsets, want.offsets)
                assert torch.equal(got.indices, want.indices)
                assert got.indices.dtype == want.indices.dtype
            if kind != "cell_to_points":  # the cells array itself, cached in full
                assert not _topology_keys(fresh)


def test_invalid_selections_raise(device):
    mesh = _tetrahedra(device)
    adjacency = mesh.get_point_to_points_adjacency()
    short_mask = torch.ones(3, dtype=torch.bool, device=device)
    matrix = torch.zeros((2, 2), dtype=torch.long, device=device)
    for bad, error in (
        (short_mask, IndexError),
        (matrix, IndexError),
        (slice(None, None, -1), ValueError),  # as for a tensor
    ):
        with pytest.raises(error):
            adjacency[bad]
        fresh = Mesh(points=mesh.points, cells=mesh.cells)
        with pytest.raises(error):
            fresh.get_point_to_points_adjacency(indices=bad)
