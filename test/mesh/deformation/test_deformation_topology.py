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

"""Tests for mesh deformation-energy topology preprocessing."""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.utilities._topology import (
    _validate_closed_oriented_triangle_surface,
    _validate_indexed_simplex_topology,
    extract_triangle_hinges,
)


def _two_triangle_mesh() -> Mesh:
    return Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        cells=torch.tensor([[0, 1, 2], [0, 2, 3]]),
    )


def _tetrahedron_surface(*, inward: bool = False) -> Mesh:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    outward_cells = torch.tensor([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    cells = outward_cells[:, (0, 2, 1)] if inward else outward_cells
    return Mesh(points=points, cells=cells)


def test_extract_triangle_hinges_is_canonical_and_omits_boundary_edges():
    mesh = _two_triangle_mesh()

    hinges = extract_triangle_hinges(mesh)

    torch.testing.assert_close(hinges, torch.tensor([[0, 2, 1, 3]]))
    assert bool((hinges[:, 0] < hinges[:, 1]).all())


def test_extract_triangle_hinges_returns_cached_tensors():
    mesh = _two_triangle_mesh()

    first_hinges = extract_triangle_hinges(mesh)
    second_hinges = extract_triangle_hinges(mesh)

    assert first_hinges.data_ptr() == second_hinges.data_ptr()


def test_extract_triangle_hinges_caches_int64_for_int32_connectivity():
    source = _two_triangle_mesh()
    mesh = Mesh(points=source.points, cells=source.cells.to(torch.int32))

    first_hinges = extract_triangle_hinges(mesh)
    second_hinges = extract_triangle_hinges(mesh)

    assert first_hinges.dtype == torch.int64
    assert first_hinges.data_ptr() == second_hinges.data_ptr()


def test_triangle_hinge_cache_is_preserved_by_geometric_transforms():
    mesh = _two_triangle_mesh()
    hinges = extract_triangle_hinges(mesh)

    translated = mesh.translate([2.0, -3.0])
    translated_hinges = extract_triangle_hinges(translated)

    assert translated_hinges.data_ptr() == hinges.data_ptr()


def test_triangle_hinges_omit_all_edges_of_single_triangle():
    mesh = Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cells=torch.tensor([[0, 1, 2]]),
    )

    hinges = extract_triangle_hinges(mesh)

    assert hinges.shape == (0, 4)


def test_triangle_hinges_reject_nonmanifold_edge():
    mesh = Mesh(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        cells=torch.tensor([[0, 1, 2], [1, 0, 3], [0, 1, 4]]),
    )

    with pytest.raises(ValueError, match="at most two triangles"):
        extract_triangle_hinges(mesh)


def test_triangle_hinges_reject_duplicate_face():
    mesh = Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cells=torch.tensor([[0, 1, 2], [2, 1, 0]]),
    )

    with pytest.raises(ValueError, match="duplicate face"):
        extract_triangle_hinges(mesh)


@pytest.mark.parametrize("inward", [False, True])
def test_closed_surface_validation_accepts_either_global_orientation(inward):
    mesh = _tetrahedron_surface(inward=inward)

    _validate_closed_oriented_triangle_surface(mesh)

    assert mesh._cache.get(("topology", "closed_oriented_triangle_surface")) is not None


def test_closed_surface_validation_rejects_boundary():
    mesh = _tetrahedron_surface()
    mesh = Mesh(points=mesh.points, cells=mesh.cells[:-1])

    with pytest.raises(ValueError, match="every undirected edge"):
        _validate_closed_oriented_triangle_surface(mesh)


def test_closed_surface_validation_rejects_inconsistent_face_orientation():
    mesh = _tetrahedron_surface()
    cells = mesh.cells.clone()
    cells[0] = cells[0, (0, 2, 1)]
    mesh = Mesh(points=mesh.points, cells=cells)

    with pytest.raises(ValueError, match="consistently oriented"):
        _validate_closed_oriented_triangle_surface(mesh)


def test_closed_surface_validation_rejects_duplicate_faces():
    mesh = _tetrahedron_surface()
    cells = torch.cat((mesh.cells, mesh.cells[:1]), dim=0)
    mesh = Mesh(points=mesh.points, cells=cells)

    with pytest.raises(ValueError, match="duplicate face"):
        _validate_closed_oriented_triangle_surface(mesh)


def test_closed_surface_validation_rejects_disconnected_components():
    first = _tetrahedron_surface()
    second = _tetrahedron_surface(inward=True)
    points = torch.cat((first.points, second.points + 3.0), dim=0)
    cells = torch.cat((first.cells, second.cells + first.n_points), dim=0)
    mesh = Mesh(points=points, cells=cells)

    with pytest.raises(ValueError, match="edge-connected|disconnected"):
        _validate_closed_oriented_triangle_surface(mesh)


@pytest.mark.parametrize(
    "cells, match",
    [
        (torch.tensor([[0, 1, 4]]), "outside the valid range"),
        (torch.tensor([[0, 1, 1]]), "repeated vertex"),
    ],
)
def test_triangle_hinge_validation_rejects_invalid_connectivity(cells, match):
    mesh = Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cells=cells,
    )

    with pytest.raises(ValueError, match=match):
        extract_triangle_hinges(mesh)


@pytest.mark.parametrize("dtype", [torch.bool, torch.complex64])
def test_deformation_topology_rejects_noninteger_connectivity(dtype):
    cells = torch.tensor([[0, 1]], dtype=dtype)
    mesh = Mesh(points=torch.tensor([[0.0], [1.0]]), cells=cells)

    with pytest.raises(TypeError, match="integer dtype"):
        _validate_indexed_simplex_topology(mesh)
