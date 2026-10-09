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

"""Adjacency queries agree with brute-force references, and stay sync-light on CUDA."""

import itertools
from collections import defaultdict

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.neighbors import (
    get_cell_to_cells_adjacency,
    get_point_to_cells_adjacency,
    get_point_to_points_adjacency,
)
from physicsnemo.mesh.neighbors._adjacency import build_adjacency_from_pairs
from test.mesh.mesh.test_slicing_sync import _cuda_sync_budget


def _random_mesh(n_points: int, n_cells: int, n_verts: int, seed: int) -> Mesh:
    """Random simplices over few points: many shared and non-manifold facets."""
    generator = torch.Generator().manual_seed(seed)
    cells = torch.stack(
        [
            torch.randperm(n_points, generator=generator)[:n_verts]
            for _ in range(n_cells)
        ]
    )
    points = torch.randn(n_points, 3, generator=generator)
    return Mesh(points=points, cells=cells)


def _book_mesh() -> Mesh:
    """Four triangles on one edge, plus a duplicate of the first one."""
    points = torch.randn(6, 3, generator=torch.Generator().manual_seed(0))
    cells = torch.tensor([[0, 1, 2], [0, 1, 3], [1, 0, 4], [0, 5, 1], [2, 0, 1]])
    return Mesh(points=points, cells=cells)


MESHES = {
    "segments": lambda: _random_mesh(12, 30, 2, seed=1),
    "triangles": lambda: _random_mesh(15, 60, 3, seed=2),
    "tetrahedra": lambda: _random_mesh(12, 50, 4, seed=3),
    "book": _book_mesh,
}


def _reference_cell_to_cells(
    cells: list[list[int]], codimension: int
) -> list[list[int]]:
    """Cells sharing a facet of ``len(cell) - codimension`` vertices, by brute force."""
    facet_size = len(cells[0]) - codimension
    cells_of_facet = defaultdict(set)
    for c, cell in enumerate(cells):
        for facet in itertools.combinations(sorted(cell), facet_size):
            cells_of_facet[facet].add(c)
    neighbors = [set() for _ in cells]
    for sharing in cells_of_facet.values():
        for c in sharing:
            neighbors[c] |= sharing - {c}
    return [sorted(n) for n in neighbors]


@pytest.mark.parametrize("name", MESHES)
def test_cell_to_cells_matches_brute_force(name, device):
    mesh = MESHES[name]().to(device)
    cells = mesh.cells.tolist()
    for codimension in range(1, mesh.n_manifold_dims + 1):
        adjacency = get_cell_to_cells_adjacency(mesh, adjacency_codimension=codimension)
        assert adjacency.device == mesh.cells.device
        assert adjacency.to_list() == _reference_cell_to_cells(cells, codimension)


@pytest.mark.parametrize("name", MESHES)
def test_point_adjacencies_match_brute_force(name, device):
    mesh = MESHES[name]().to(device)
    cells = mesh.cells.tolist()
    expected_cells = [
        sorted(c for c, cell in enumerate(cells) if p in cell)
        for p in range(mesh.n_points)
    ]
    expected_points = [
        sorted({q for cell in cells if p in cell for q in cell} - {p})
        for p in range(mesh.n_points)
    ]
    point_to_cells = get_point_to_cells_adjacency(mesh)
    point_to_points = get_point_to_points_adjacency(mesh)

    assert [sorted(n) for n in point_to_cells.to_list()] == expected_cells
    assert [sorted(n) for n in point_to_points.to_list()] == expected_points
    assert point_to_cells.device == point_to_points.device == mesh.points.device


def test_build_adjacency_offsets_skip_sources_without_pairs(device):
    """Sources with no pairs, including the first and last, get empty ranges."""
    sources = torch.tensor([4, 1, 4, 1, 2], device=device)
    targets = torch.tensor([0, 3, 2, 1, 0], device=device)
    adjacency = build_adjacency_from_pairs(sources, targets, n_sources=6)

    assert adjacency.offsets.tolist() == [0, 0, 2, 3, 3, 5, 5]
    assert adjacency.to_list() == [[], [1, 3], [0], [], [0, 2], []]
    assert adjacency.device == sources.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cached_adjacency_is_reused_without_syncs():
    """A cached adjacency is not rebuilt (and revalidated) on later queries."""
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=3, device="cuda")
    queries = [
        mesh.get_cell_to_cells_adjacency,
        mesh.get_point_to_points_adjacency,
        mesh.get_point_to_cells_adjacency,
        mesh.get_cell_to_points_adjacency,
    ]
    first = [query() for query in queries]
    torch.cuda.synchronize()

    with _cuda_sync_budget(0):
        again = [query() for query in queries]
    for a, b in zip(first, again):
        assert torch.equal(a.offsets, b.offsets) and torch.equal(a.indices, b.indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "query, max_syncs",
    [
        (lambda m: get_cell_to_cells_adjacency(m), 5),
        (lambda m: get_point_to_points_adjacency(m), 3),
        (lambda m: get_point_to_cells_adjacency(m), 1),
    ],
)
def test_adjacency_sync_budget(query, max_syncs):
    """Only data-dependent output sizes may synchronize."""
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=3, device="cuda")
    query(mesh)  # warm up
    torch.cuda.synchronize()

    with _cuda_sync_budget(max_syncs):
        query(mesh)
    torch.cuda.synchronize()
