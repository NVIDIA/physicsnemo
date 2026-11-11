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

"""Mesh topology agrees between the two dispatch paths of ``sort_index_tuples``.

Large CUDA inputs are sorted by a sorting network and everything else by
``torch.sort``, so the same mesh on CPU and on CUDA exercises both paths.
"""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0
from physicsnemo.mesh.primitives.planar import structured_grid
from physicsnemo.mesh.primitives.volumes import cube_volume
from physicsnemo.mesh.repair._cleaning import remove_duplicate_cells
from physicsnemo.mesh.subdivision.butterfly import compute_butterfly_weights_2d
from physicsnemo.mesh.utilities._edge_lookup import find_edges_in_reference
from physicsnemo.mesh.utilities._topology import extract_unique_edges
from physicsnemo.mesh.validation.validate import check_duplicate_cell_vertices
from physicsnemo.utils._index_tuple_ops import _SORTING_NETWORK_MIN_NUMEL

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for the network path"
)


def _triangles() -> Mesh:
    """A planar grid of ~180K triangles: enough entries for the network path."""
    return structured_grid.load(n_x=301, n_y=301)


def _tetrahedra() -> Mesh:
    """A cube of ~160K tetrahedra."""
    return cube_volume.load(subdivisions=30)


def _assert_same(cpu: torch.Tensor, cuda: torch.Tensor) -> None:
    torch.testing.assert_close(cuda.cpu(), cpu, rtol=0, atol=0)


@pytest.mark.parametrize("make_mesh", [_triangles, _tetrahedra])
def test_topology_matches_between_sort_paths(make_mesh):
    cpu = make_mesh()
    cuda = cpu.to("cuda")
    assert cuda.cells.numel() >= _SORTING_NETWORK_MIN_NUMEL

    for codimension in range(1, cpu.n_manifold_dims + 1):
        _assert_same(
            cpu.get_facet_mesh(manifold_codimension=codimension).cells,
            cuda.get_facet_mesh(manifold_codimension=codimension).cells,
        )
    _assert_same(cpu.get_boundary_mesh().cells, cuda.get_boundary_mesh().cells)

    for actual, expected in zip(extract_unique_edges(cuda), extract_unique_edges(cpu)):
        _assert_same(expected, actual)

    values = cpu.points[:, 0]
    for actual, expected in zip(
        exterior_derivative_0(cuda, values.cuda()), exterior_derivative_0(cpu, values)
    ):
        _assert_same(expected, actual)

    edges, _ = extract_unique_edges(cpu)
    queries = torch.cat([edges[::3].flip(1), edges[:5] + cpu.n_points])
    indices_cpu, matches_cpu = find_edges_in_reference(edges, queries)
    indices_cuda, matches_cuda = find_edges_in_reference(edges.cuda(), queries.cuda())
    _assert_same(matches_cpu, matches_cuda)
    _assert_same(indices_cpu[matches_cpu], indices_cuda[matches_cuda])

    # Every 7th cell again, with its vertices rotated: same vertex set
    cells = torch.cat([cpu.cells, cpu.cells[::7].roll(1, dims=1)])
    cell_data = TensorDict({"id": torch.arange(len(cells))}, batch_size=[len(cells)])
    unique_cpu, data_cpu = remove_duplicate_cells(cells, cell_data, cpu.n_points)
    unique_cuda, data_cuda = remove_duplicate_cells(
        cells.cuda(), cell_data.cuda(), cpu.n_points
    )
    _assert_same(unique_cpu, unique_cuda)
    _assert_same(data_cpu["id"], data_cuda["id"])
    assert len(unique_cpu) == cpu.n_cells

    degenerate = Mesh(points=cpu.points, cells=cpu.cells.clone())
    degenerate.cells[::5, 1] = degenerate.cells[::5, 0]
    n_cpu, invalid_cpu = check_duplicate_cell_vertices(degenerate)
    n_cuda, invalid_cuda = check_duplicate_cell_vertices(degenerate.to("cuda"))
    assert n_cpu == n_cuda == len(range(0, cpu.n_cells, 5))
    _assert_same(invalid_cpu, invalid_cuda)


def test_butterfly_weights_match_between_sort_paths():
    cpu = structured_grid.load(n_x=301, n_y=301)
    cpu = Mesh(
        points=torch.cat([cpu.points, (cpu.points**2).sum(-1, keepdim=True)], dim=-1),
        cells=cpu.cells,
    )
    cuda = cpu.to("cuda")
    edges, _ = extract_unique_edges(cpu)
    expected = compute_butterfly_weights_2d(cpu, edges)
    actual = compute_butterfly_weights_2d(cuda, edges.cuda())
    torch.testing.assert_close(actual.cpu(), expected)
