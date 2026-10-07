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

"""Tests that the containment ``tolerance`` does not depend on the mesh's length unit."""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.planar import structured_grid
from physicsnemo.mesh.primitives.surfaces import torus
from physicsnemo.mesh.sampling import (
    find_all_containing_cells,
    find_containing_cells,
    sample_data_at_points,
)
from physicsnemo.mesh.sampling.sample_data import (
    compute_barycentric_coordinates_pairwise,
)


def _scaled(mesh: Mesh, scale: float) -> Mesh:
    """Return ``mesh`` with its points multiplied by ``scale`` and data kept."""
    return Mesh(
        points=mesh.points * scale,
        cells=mesh.cells,
        point_data=mesh.point_data.clone(),
        cell_data=mesh.cell_data.clone(),
    )


def _contains_source_cell(
    mesh: Mesh, query_points: torch.Tensor, source_cells: torch.Tensor
) -> torch.Tensor:
    """Whether each query point is found in the cell it was sampled from."""
    query_idx, cell_idx = find_all_containing_cells(
        mesh, query_points
    ).expand_to_pairs()
    found = torch.zeros(len(query_points), dtype=torch.bool, device=query_points.device)
    found[query_idx[cell_idx == source_cells[query_idx]]] = True
    return found


def test_power_of_two_rescaling_gives_identical_results(device):
    """Rescaling mesh and points by a power of two leaves every output unchanged."""
    torch.manual_seed(0)
    base = torus.load(n_major=16, n_minor=8, device=device)
    base = Mesh(
        points=base.points + torch.tensor([3.0, -1.0, 0.5], device=device),
        cells=base.cells,
    )
    base.point_data["f"] = torch.randn(base.n_points, device=device)
    base.cell_data["id"] = torch.arange(
        base.n_cells, dtype=torch.float32, device=device
    )

    source_cells = torch.randint(0, base.n_cells, (500,), device=device)
    on_surface = base.sample_random_points_on_cells(source_cells)
    off_surface = on_surface[:100] + 1e-2 * base.cell_normals[source_cells[:100]]
    query = torch.cat([on_surface, off_surface])

    outputs = {}
    for scale in (2.0**-10, 1.0, 2.0**10):
        mesh = _scaled(base, scale)
        q = query * scale
        cell_idx, bary = find_containing_cells(mesh, q)
        adjacency = find_all_containing_cells(mesh, q)
        outputs[scale] = (
            cell_idx,
            bary,
            adjacency.offsets,
            adjacency.indices,
            sample_data_at_points(mesh, q, data_source="points")["f"],
            sample_data_at_points(mesh, q, multiple_cells_strategy="nan")["id"],
        )

    ### Guard against a vacuous comparison: on-surface points are found and
    ### off-surface points are not.
    reference_cell_idx = outputs[1.0][0]
    assert (reference_cell_idx[:500] >= 0).all()
    assert (reference_cell_idx[500:] == -1).all()

    for scale in (2.0**-10, 2.0**10):
        for actual, expected in zip(outputs[scale], outputs[1.0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("scale", [1000.0, 1024.0])
def test_float32_surface_in_large_units_finds_on_surface_points(scale, device):
    """A float32 triangle surface in large units (e.g. millimetres) finds its points.

    Float32 rounding of on-surface points is about ``1.2e-7 * |x|``, so an
    absolute distance tolerance of ``1e-6`` rejected most points at
    ``|x| ~ 1e3``.
    """
    torch.manual_seed(0)
    base = torus.load(device=device)
    mesh = Mesh(points=base.points * scale, cells=base.cells)
    mesh.cell_data["id"] = torch.arange(
        mesh.n_cells, dtype=torch.float32, device=device
    )

    source_cells = torch.randint(0, mesh.n_cells, (2000,), device=device)
    query = mesh.sample_random_points_on_cells(source_cells)

    assert not sample_data_at_points(mesh, query)["id"].isnan().any()
    assert _contains_source_cell(mesh, query, source_cells).all()


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1000.0])
def test_codimension_zero_containment_is_the_barycentric_test(scale):
    """For planar meshes, containment is exactly ``all(bary >= -tolerance)``.

    The reconstruction error is zero in codimension 0, so the length scale
    only pads the BVH boxes and must not change which cells are found.
    """
    torch.manual_seed(0)
    base = structured_grid.load(n_x=9, n_y=9)
    mesh = Mesh(
        points=(base.points + torch.tensor([2.0, -0.5])) * scale, cells=base.cells
    )

    on_cells = mesh.sample_random_points_on_cells(
        torch.randint(0, mesh.n_cells, (200,))
    )
    outside = (torch.rand(50, 2) + torch.tensor([4.0, -0.5])) * scale
    query = torch.cat([on_cells, mesh.points, outside])

    ### Brute force over every (query, cell) pair with the same barycentric solver
    n_queries, n_cells = len(query), mesh.n_cells
    pair_query = torch.arange(n_queries).repeat_interleave(n_cells)
    pair_cell = torch.arange(n_cells).repeat(n_queries)
    bary, _ = compute_barycentric_coordinates_pairwise(
        query[pair_query], mesh.points[mesh.cells[pair_cell]]
    )
    expected = (bary >= -1e-6).all(dim=-1).view(n_queries, n_cells)

    query_idx, cell_idx = find_all_containing_cells(mesh, query).expand_to_pairs()
    actual = torch.zeros(n_queries, n_cells, dtype=torch.bool)
    actual[query_idx, cell_idx] = True

    assert torch.equal(actual, expected)
    assert expected[: len(on_cells)].any(dim=1).all()
    assert not expected[-len(outside) :].any()
