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

"""Tests for find_nearest_cells.

Validates that find_nearest_cells (backed by knn) produces correct
nearest-neighbor assignments by comparing against brute-force cdist.
"""

import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import plane
from physicsnemo.mesh.sampling.sample_data import find_nearest_cells


def _make_point_cloud_mesh(n: int = 100) -> Mesh:
    """Create a mesh of single-vertex cells (point cloud) on the unit sphere."""
    torch.manual_seed(0)
    points = torch.randn(n, 3, dtype=torch.float64)
    points = points / points.norm(dim=1, keepdim=True)
    cells = torch.arange(n).unsqueeze(1)
    return Mesh(points=points, cells=cells)


### knn matches brute-force ###


class TestKnnMatchesBruteForce:
    """Verify knn-based assignments match brute-force cdist on varied mesh topologies."""

    def test_triangle_mesh(self):
        """knn assignments match brute-force cdist on a regular triangle mesh."""
        m = plane.load(subdivisions=14)  # 392 triangles
        mesh = Mesh(points=m.points.double(), cells=m.cells)
        torch.manual_seed(1)
        query = torch.rand(200, 3, dtype=torch.float64)
        query[:, 2] = 0.0

        idx_knn, _ = find_nearest_cells(mesh, query)
        idx_brute = torch.cdist(query, mesh.cell_centroids).argmin(dim=1)

        assert torch.equal(idx_knn, idx_brute)

    def test_point_cloud_mesh(self):
        """knn assignments match brute-force cdist for a point cloud mesh."""
        seed_mesh = _make_point_cloud_mesh(500)
        torch.manual_seed(2)
        query = torch.randn(2000, 3, dtype=torch.float64)
        query = query / query.norm(dim=1, keepdim=True) * 1.05

        idx_knn, _ = find_nearest_cells(seed_mesh, query)
        idx_brute = torch.cdist(query, seed_mesh.cell_centroids).argmin(dim=1)

        assert torch.equal(idx_knn, idx_brute)

    def test_non_uniform_mesh(self):
        """knn handles a mesh with widely varying cell sizes."""
        _f = plane.load(subdivisions=19)  # dense region
        fine = Mesh(points=_f.points.double(), cells=_f.cells)
        coarse_pts = torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
                [4.0, 2.0, 0.0],
            ],
            dtype=torch.float64,
        )
        coarse_cells = torch.tensor([[0, 1, 2], [1, 3, 2]]) + fine.n_points
        combined_pts = torch.cat([fine.points, coarse_pts])
        combined_cells = torch.cat([fine.cells, coarse_cells])
        mesh = Mesh(points=combined_pts, cells=combined_cells)

        torch.manual_seed(3)
        query = torch.rand(300, 3, dtype=torch.float64)
        query[:, 0] *= 4.0
        query[:, 1] *= 2.0
        query[:, 2] = 0.0

        idx_knn, _ = find_nearest_cells(mesh, query)
        idx_brute = torch.cdist(query, mesh.cell_centroids).argmin(dim=1)

        assert torch.equal(idx_knn, idx_brute)


### Exact centroid queries ###


class TestExactCentroidQueries:
    """Verify that querying at exact cell centroids yields identity mapping at zero distance."""

    def test_query_at_centroid_gives_distance_zero(self):
        """Querying at a cell's own centroid should find that cell at distance 0."""
        m = plane.load(subdivisions=4)
        mesh = Mesh(points=m.points.double(), cells=m.cells)
        centroids = mesh.cell_centroids

        idx, projected = find_nearest_cells(mesh, centroids)

        assert torch.equal(idx, torch.arange(mesh.n_cells))
        dists = (projected - centroids).norm(dim=1)
        assert dists.max().item() < 1e-12

    def test_point_cloud_self_query(self):
        """Querying a point cloud with its own points should yield identity mapping."""
        seed_mesh = _make_point_cloud_mesh(200)

        idx, projected = find_nearest_cells(seed_mesh, seed_mesh.cell_centroids)

        assert torch.equal(idx, torch.arange(seed_mesh.n_cells))
        dists = (projected - seed_mesh.cell_centroids).norm(dim=1)
        assert dists.max().item() < 1e-12
