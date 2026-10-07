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

"""Mesh operations keep their results while dropping hidden device synchronizations.

Typical hidden synchronizations: indexing a CUDA tensor with a Python list,
blocking copies of small host tables, boolean-mask indexing, and reading
counts back one ``.item()`` at a time.
"""

import itertools
import warnings

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.boundaries import extract_candidate_facets
from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_1
from physicsnemo.mesh.primitives.planar import structured_grid
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.mesh.primitives.volumes import cube_volume
from physicsnemo.mesh.repair import remove_degenerate_cells
from physicsnemo.mesh.repair._cleaning import remove_duplicate_cells
from physicsnemo.mesh.smoothing import smooth_laplacian
from physicsnemo.mesh.validation.quality import _compute_simplex_altitudes
from test.mesh.mesh.test_slicing_sync import _cuda_sync_budget

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required to detect synchronizations"
)


def _count_cuda_syncs(fn) -> int:
    """Run ``fn`` and count the CUDA synchronizations that it makes."""
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    return sum("synchronizing CUDA operation" in str(w.message) for w in caught)


MESHES = {
    "triangles_2d": lambda device: structured_grid.load(n_x=8, n_y=7, device=device),
    "triangles_3d": lambda device: sphere_icosahedral.load(
        subdivisions=2, device=device
    ),
    "tetrahedra": lambda device: cube_volume.load(subdivisions=3, device=device),
}


### Results


def test_merge_offsets_cells_and_keeps_data(device):
    meshes = [
        structured_grid.load(x_min=i, x_max=i + 1, n_x=3 + i, n_y=4, device=device)
        for i in range(3)
    ]
    for i, mesh in enumerate(meshes):
        mesh.cell_data["id"] = torch.full((mesh.n_cells,), i, device=device)

    merged = Mesh.merge(meshes)

    offsets = list(itertools.accumulate((m.n_points for m in meshes), initial=0))
    expected = torch.cat([m.cells + o for m, o in zip(meshes, offsets)])
    torch.testing.assert_close(merged.cells, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        merged.points, torch.cat([m.points for m in meshes]), rtol=0, atol=0
    )
    assert merged.cell_data["id"].tolist() == sum(
        ([i] * m.n_cells for i, m in enumerate(meshes)), []
    )


def test_remove_duplicate_cells_keeps_first_occurrences_in_order(device):
    cells = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [2, 0, 1], [1, 2, 3], [5, 3, 4], [0, 1, 2], [6, 7, 8]],
        device=device,
    )
    data = TensorDict({"id": torch.arange(7, device=device)}, batch_size=[7])

    unique_cells, unique_data = remove_duplicate_cells(cells, data, index_bound=9)

    assert unique_data["id"].tolist() == [0, 1, 3, 6]
    torch.testing.assert_close(unique_cells, cells[[0, 1, 3, 6]], rtol=0, atol=0)


def test_remove_degenerate_cells_counts_and_keeps(device):
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [1.0, 1.0]], device=device
    )
    cells = torch.tensor(
        [[0, 1, 2], [1, 1, 2], [0, 1, 3], [1, 3, 4], [2, 2, 2]], device=device
    )
    mesh = Mesh(points=points, cells=cells)
    mesh.cell_data["id"] = torch.arange(5, device=device)

    cleaned, stats = remove_degenerate_cells(mesh)

    # [0, 1, 3] is collinear (zero area); the repeated-vertex cells also have zero area
    assert stats == {
        "n_zero_area_cells": 3,
        "n_duplicate_vertex_cells": 2,
        "n_cells_original": 5,
        "n_cells_final": 2,
    }
    assert cleaned.cell_data["id"].tolist() == [0, 3]


@pytest.mark.parametrize("n_manifold_dims", [2, 3, 4])
def test_simplex_altitudes_match_volume_over_facet(n_manifold_dims, device):
    """Altitude opposite vertex k is n * volume / (measure of the facet without k)."""
    generator = torch.Generator().manual_seed(n_manifold_dims)
    n_verts = n_manifold_dims + 1
    points = torch.randn(10 * n_verts, n_manifold_dims, generator=generator).double()
    mesh = Mesh(
        points=points.to(device),
        cells=torch.arange(10 * n_verts, device=device).reshape(10, n_verts),
    )
    altitudes = _compute_simplex_altitudes(mesh, mesh.cell_areas)

    for k in range(n_verts):
        others = [i for i in range(n_verts) if i != k]
        facet = Mesh(points=mesh.points, cells=mesh.cells[:, others])
        torch.testing.assert_close(
            altitudes[:, k], n_manifold_dims * mesh.cell_areas / facet.cell_areas
        )


def test_exterior_derivative_1_matches_explicit_edge_orientation(device):
    mesh = MESHES["triangles_3d"](device)
    edges = mesh.get_facet_mesh(manifold_codimension=1).cells
    edge_values = torch.randn(len(edges), device=device, dtype=mesh.points.dtype)

    face_values, faces = exterior_derivative_1(mesh, edge_values, edges)

    # Circulation of the edge values around each face, edge by edge
    lookup = {tuple(e): i for i, e in enumerate(edges.tolist())}
    expected = []
    for face in faces.tolist():
        total = 0.0
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            sign = 1.0 if a < b else -1.0
            total += sign * edge_values[lookup[(min(a, b), max(a, b))]].item()
        expected.append(total)
    torch.testing.assert_close(
        face_values, torch.tensor(expected, device=device, dtype=face_values.dtype)
    )


def test_smoothing_keeps_boundary_vertices_fixed(device):
    mesh = MESHES["triangles_2d"](device)
    generator = torch.Generator().manual_seed(0)
    noise = 0.01 * torch.randn(mesh.points.shape, generator=generator).to(device)
    mesh = Mesh(points=mesh.points + noise, cells=mesh.cells)
    boundary = torch.unique(mesh.get_boundary_mesh().cells)

    smoothed = smooth_laplacian(mesh, n_iter=5, relaxation_factor=0.2)

    torch.testing.assert_close(
        smoothed.points[boundary], mesh.points[boundary], rtol=0, atol=0
    )
    assert not torch.equal(smoothed.points, mesh.points)


### Synchronizations


@requires_cuda
@pytest.mark.parametrize("name", MESHES)
def test_cell_areas_and_candidate_facets_are_sync_free(name):
    mesh = MESHES[name]("cuda")
    fresh = Mesh(points=mesh.points, cells=mesh.cells)
    extract_candidate_facets(mesh.cells)  # warm up lazy CUDA initialization
    _ = Mesh(points=mesh.points, cells=mesh.cells).cell_areas
    torch.cuda.synchronize()

    with _cuda_sync_budget(0):
        _ = fresh.cell_areas
        for codimension in range(1, mesh.n_manifold_dims + 1):
            extract_candidate_facets(mesh.cells, manifold_codimension=codimension)
    torch.cuda.synchronize()


@requires_cuda
def test_merge_is_sync_free():
    meshes = [MESHES["triangles_3d"]("cuda") for _ in range(3)]
    for mesh in meshes:
        mesh.point_data["t"] = torch.zeros(mesh.n_points, device="cuda")
    Mesh.merge(meshes)
    torch.cuda.synchronize()

    with _cuda_sync_budget(0):
        Mesh.merge(meshes)
    torch.cuda.synchronize()


@requires_cuda
def test_remove_degenerate_cells_reads_counts_once():
    mesh = MESHES["triangles_3d"]("cuda")
    _ = mesh.cell_areas
    remove_degenerate_cells(mesh)
    assert _count_cuda_syncs(lambda: remove_degenerate_cells(mesh)) <= 1


@requires_cuda
def test_smoothing_synchronizations_do_not_grow_with_iterations():
    mesh = MESHES["triangles_3d"]("cuda")
    smooth_laplacian(mesh, n_iter=1)
    one = _count_cuda_syncs(lambda: smooth_laplacian(mesh, n_iter=1))
    ten = _count_cuda_syncs(lambda: smooth_laplacian(mesh, n_iter=10))
    assert ten == one


@requires_cuda
@pytest.mark.parametrize(
    "subdivision, max_syncs", [("linear", 1), ("loop", 12), ("butterfly", 13)]
)
def test_subdivision_synchronizations_are_bounded(subdivision, max_syncs):
    """Subdivision synchronizes only for data-dependent sizes, not for small tables."""
    mesh = MESHES["triangles_3d"]("cuda")
    mesh.subdivide(levels=1, filter=subdivision)
    n_syncs = _count_cuda_syncs(lambda: mesh.subdivide(levels=1, filter=subdivision))
    assert n_syncs <= max_syncs
