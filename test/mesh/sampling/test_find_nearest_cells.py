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

"""Tests for find_nearest_cells with BVH acceleration.

Validates that the BVH-accelerated path produces correct nearest-neighbor
assignments, including regression tests for previously-fixed bugs:
- premature resolution (resolving a query before the nearest cell is found)
- candidate truncation (max_candidates_per_point pruning BVH subtrees)
- point cloud meshes (single-vertex cells with zero-volume AABBs)
"""

import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import plane
from physicsnemo.mesh.sampling.sample_data import (
    _find_nearest_cells_brute,
    find_nearest_cells,
)
from physicsnemo.mesh.spatial import BVH


def _make_point_cloud_mesh(n: int = 100) -> Mesh:
    """Create a mesh of single-vertex cells (point cloud) on the unit sphere."""
    torch.manual_seed(0)
    points = torch.randn(n, 3, dtype=torch.float64)
    points = points / points.norm(dim=1, keepdim=True)
    cells = torch.arange(n).unsqueeze(1)
    return Mesh(points=points, cells=cells)


### BVH matches brute-force ###


class TestBvhMatchesBruteForce:
    def test_triangle_mesh(self):
        """BVH assignments match brute-force on a regular triangle mesh."""
        m = plane.load(subdivisions=14)  # 392 triangles
        mesh = Mesh(points=m.points.double(), cells=m.cells)
        torch.manual_seed(1)
        query = torch.rand(200, 3, dtype=torch.float64)
        query[:, 2] = 0.0

        bvh = BVH.from_mesh(mesh)
        idx_bvh, _ = find_nearest_cells(mesh, query, bvh=bvh)
        idx_brute = _find_nearest_cells_brute(
            query, mesh.cell_centroids, chunk_size=500
        )

        assert torch.equal(idx_bvh, idx_brute)

    def test_point_cloud_mesh(self):
        """BVH assignments match brute-force for a point cloud mesh.

        This is a regression test: the original _find_nearest_cells_bvh had
        bugs that caused ~13% wrong assignments for point cloud meshes due to
        premature resolution and candidate truncation.
        """
        seed_mesh = _make_point_cloud_mesh(500)
        torch.manual_seed(2)
        query = torch.randn(2000, 3, dtype=torch.float64)
        query = query / query.norm(dim=1, keepdim=True) * 1.05

        bvh = BVH.from_mesh(seed_mesh)
        idx_bvh, _ = find_nearest_cells(seed_mesh, query, bvh=bvh)
        idx_brute = _find_nearest_cells_brute(
            query, seed_mesh.cell_centroids, chunk_size=500
        )

        assert torch.equal(idx_bvh, idx_brute)

    def test_non_uniform_mesh(self):
        """BVH handles a mesh with widely varying cell sizes.

        Mixes a fine grid region with a coarse grid region to stress the
        expanding-radius resolution logic.
        """
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

        bvh = BVH.from_mesh(mesh)
        idx_bvh, _ = find_nearest_cells(mesh, query, bvh=bvh)
        idx_brute = _find_nearest_cells_brute(
            query, mesh.cell_centroids, chunk_size=500
        )

        assert torch.equal(idx_bvh, idx_brute)


### Exact centroid queries ###


class TestExactCentroidQueries:
    def test_query_at_centroid_gives_distance_zero(self):
        """Querying at a cell's own centroid should find that cell at distance 0."""
        m = plane.load(subdivisions=4)
        mesh = Mesh(points=m.points.double(), cells=m.cells)
        centroids = mesh.cell_centroids
        bvh = BVH.from_mesh(mesh)

        idx, projected = find_nearest_cells(mesh, centroids, bvh=bvh)

        assert torch.equal(idx, torch.arange(mesh.n_cells))
        dists = (projected - centroids).norm(dim=1)
        assert dists.max().item() < 1e-12

    def test_point_cloud_self_query(self):
        """Querying a point cloud with its own points should yield identity mapping."""
        seed_mesh = _make_point_cloud_mesh(200)
        bvh = BVH.from_mesh(seed_mesh)

        idx, projected = find_nearest_cells(
            seed_mesh, seed_mesh.cell_centroids, bvh=bvh
        )

        assert torch.equal(idx, torch.arange(seed_mesh.n_cells))
        dists = (projected - seed_mesh.cell_centroids).norm(dim=1)
        assert dists.max().item() < 1e-12


### Resolution completeness ###


class TestResolutionCompleteness:
    def test_all_queries_resolve_triangle_mesh(self):
        """No queries should fall through to brute-force on a well-behaved mesh."""
        m = plane.load(subdivisions=9)
        mesh = Mesh(points=m.points.double(), cells=m.cells)
        torch.manual_seed(4)
        query = torch.rand(500, 3, dtype=torch.float64)
        query[:, 2] = 0.0

        bvh = BVH.from_mesh(mesh)
        # Call the internal BVH function to check resolution directly
        from physicsnemo.mesh.sampling.sample_data import _find_nearest_cells_bvh

        _, resolved = _find_nearest_cells_bvh(
            query, mesh.cell_centroids, bvh, mesh.n_cells, mesh.n_spatial_dims
        )
        assert resolved.all()

    def test_all_queries_resolve_point_cloud(self):
        """Point cloud queries should all resolve via BVH (no brute-force fallback)."""
        seed_mesh = _make_point_cloud_mesh(300)
        torch.manual_seed(5)
        query = torch.randn(1000, 3, dtype=torch.float64)
        query = query / query.norm(dim=1, keepdim=True)

        bvh = BVH.from_mesh(seed_mesh)
        from physicsnemo.mesh.sampling.sample_data import _find_nearest_cells_bvh

        _, resolved = _find_nearest_cells_bvh(
            query,
            seed_mesh.cell_centroids,
            bvh,
            seed_mesh.n_cells,
            seed_mesh.n_spatial_dims,
        )
        assert resolved.all()


### Without BVH (brute-force path) ###


class TestBruteForce:
    def test_no_bvh_gives_correct_results(self):
        """find_nearest_cells without BVH should use brute-force and be correct."""
        m = plane.load(subdivisions=7)
        mesh = Mesh(points=m.points.double(), cells=m.cells)
        torch.manual_seed(6)
        query = torch.rand(100, 3, dtype=torch.float64)
        query[:, 2] = 0.0

        idx_no_bvh, _ = find_nearest_cells(mesh, query, bvh=None)

        bvh = BVH.from_mesh(mesh)
        idx_bvh, _ = find_nearest_cells(mesh, query, bvh=bvh)

        assert torch.equal(idx_no_bvh, idx_bvh)
