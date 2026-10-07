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

"""Loop and butterfly edge rules against per-edge references, and their syncs."""

import warnings
from collections import defaultdict

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.planar import structured_grid
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.mesh.subdivision.butterfly import compute_butterfly_weights_2d
from physicsnemo.mesh.subdivision.loop import compute_loop_edge_positions_2d
from physicsnemo.mesh.utilities._topology import extract_unique_edges


def _triangles_of_edges(cells: list[list[int]]) -> dict:
    triangles = defaultdict(list)
    for t, cell in enumerate(cells):
        for i, j in ((0, 1), (1, 2), (0, 2)):
            triangles[tuple(sorted((cell[i], cell[j])))].append(t)
    return triangles


def _opposite(cell: list[int], edge) -> int:
    return next(v for v in cell if v not in edge)


def _reference_loop(mesh: Mesh, edges: torch.Tensor) -> torch.Tensor:
    """Loop's edge rule, edge by edge."""
    cells = mesh.cells.tolist()
    triangles = _triangles_of_edges(cells)
    p = mesh.points
    rows = []
    for edge in edges.tolist():
        adjacent = triangles[tuple(edge)]
        if len(adjacent) == 2:
            a, b = (_opposite(cells[t], edge) for t in adjacent)
            rows.append(3 / 8 * (p[edge[0]] + p[edge[1]]) + 1 / 8 * (p[a] + p[b]))
        else:
            rows.append((p[edge[0]] + p[edge[1]]) / 2)
    return torch.stack(rows)


def _reference_butterfly(mesh: Mesh, edges: torch.Tensor) -> torch.Tensor:
    """The 8-point butterfly stencil, edge by edge, dropping missing wings."""
    cells = mesh.cells.tolist()
    triangles = _triangles_of_edges(cells)
    p = mesh.points

    def wing(v, w, known):
        others = [t for t in triangles[tuple(sorted((v, w)))] if t != known]
        return p[_opposite(cells[others[0]], (v, w))] if others else 0 * p[v]

    rows = []
    for v0, v1 in edges.tolist():
        adjacent = triangles[(v0, v1)]
        if len(adjacent) != 2:
            rows.append((p[v0] + p[v1]) / 2)
            continue
        t0, t1 = adjacent
        a, b = _opposite(cells[t0], (v0, v1)), _opposite(cells[t1], (v0, v1))
        wings = wing(v0, a, t0) + wing(v1, a, t0) + wing(v0, b, t1) + wing(v1, b, t1)
        rows.append((p[v0] + p[v1]) / 2 + (p[a] + p[b]) / 8 - wings / 16)
    return torch.stack(rows)


def _lifted_grid(device) -> Mesh:
    """An open, curved surface: a paraboloid over a planar grid."""
    grid = structured_grid.load(n_x=7, n_y=6, device=device)
    z = (grid.points**2).sum(-1, keepdim=True)
    return Mesh(points=torch.cat([grid.points, z], dim=-1), cells=grid.cells)


def _with_strays(mesh: Mesh) -> Mesh:
    """Add a lone triangle and a third triangle on one edge (non-manifold)."""
    n = mesh.n_points
    extra_points = torch.tensor(
        [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0], [0.3, 0.3, 3.0]],
        device=mesh.points.device,
    )
    a, b = mesh.cells[0, 0].item(), mesh.cells[0, 1].item()
    extra_cells = torch.tensor(
        [[n, n + 1, n + 2], [a, b, n + 3]], device=mesh.cells.device
    )
    return Mesh(
        points=torch.cat([mesh.points, extra_points]),
        cells=torch.cat([mesh.cells, extra_cells]),
    )


MESHES = {
    "closed": lambda d: sphere_icosahedral.load(subdivisions=2, device=d),
    "open": _lifted_grid,
}


@pytest.mark.parametrize("name", MESHES)
@pytest.mark.parametrize("strays", [False, True])
def test_loop_edge_rule_matches_reference(name, strays, device):
    mesh = MESHES[name](device)
    if strays:
        mesh = _with_strays(mesh)
    edges, _ = extract_unique_edges(mesh)
    torch.testing.assert_close(
        compute_loop_edge_positions_2d(mesh, edges), _reference_loop(mesh, edges)
    )


@pytest.mark.parametrize("name", MESHES)
def test_butterfly_edge_rule_matches_reference(name, device):
    mesh = MESHES[name](device)
    edges, _ = extract_unique_edges(mesh)
    torch.testing.assert_close(
        compute_butterfly_weights_2d(mesh, edges), _reference_butterfly(mesh, edges)
    )


def test_edge_rules_on_boundary_only_meshes(device):
    """A lone triangle has only boundary edges: every rule is the midpoint."""
    mesh = Mesh(
        points=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device
        ),
        cells=torch.tensor([[0, 1, 2]], device=device),
    )
    edges, _ = extract_unique_edges(mesh)
    midpoints = mesh.points[edges].mean(dim=1)
    torch.testing.assert_close(compute_loop_edge_positions_2d(mesh, edges), midpoints)
    torch.testing.assert_close(compute_butterfly_weights_2d(mesh, edges), midpoints)


def _count_cuda_syncs(fn) -> int:
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "rule, max_syncs",
    [(compute_loop_edge_positions_2d, 1), (compute_butterfly_weights_2d, 2)],
)
def test_edge_rules_synchronize_only_for_unique_edges(rule, max_syncs):
    """Only building the edge tables synchronizes, not selecting edges by kind."""
    mesh = _with_strays(_lifted_grid("cuda"))
    edges, _ = extract_unique_edges(mesh)
    rule(mesh, edges)
    assert _count_cuda_syncs(lambda: rule(mesh, edges)) <= max_syncs
