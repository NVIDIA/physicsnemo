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

"""Tests for Mesh graph and point cloud extraction methods."""

import torch
from tensordict import TensorDict

from physicsnemo.mesh import Mesh

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _single_triangle() -> Mesh:
    """A single triangle in 2D."""
    return Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
        cells=torch.tensor([[0, 1, 2]]),
    )


def _two_triangles() -> Mesh:
    """Two triangles sharing edge (1, 2)."""
    return Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]]),
        cells=torch.tensor([[0, 1, 2], [1, 3, 2]]),
    )


def _single_tet() -> Mesh:
    """A single tetrahedron in 3D."""
    return Mesh(
        points=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        cells=torch.tensor([[0, 1, 2, 3]]),
    )


def _two_tets() -> Mesh:
    """Two tetrahedra sharing a face (vertices 0, 1, 2)."""
    return Mesh(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        ),
        cells=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]]),
    )


def _triangle_with_data() -> Mesh:
    """Two triangles with both point_data and cell_data."""
    return Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]]),
        cells=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        point_data=TensorDict(
            {"temperature": torch.tensor([100.0, 200.0, 300.0, 400.0])},
            batch_size=[4],
        ),
        cell_data=TensorDict(
            {"pressure": torch.tensor([10.0, 20.0])},
            batch_size=[2],
        ),
    )


# ---------------------------------------------------------------------------
# to_edge_graph
# ---------------------------------------------------------------------------


class TestToEdgeGraph:
    """Extracts unique edges from cell connectivity into a 1D mesh, verifying correct
    edge counts for triangles and tets, point coordinate preservation, and valid
    index bounds."""

    def test_single_triangle(self):
        """A single triangle has 3 edges."""
        mesh = _single_triangle()
        graph = mesh.to_edge_graph()

        assert graph.n_manifold_dims == 1
        assert graph.cells.shape[1] == 2
        assert graph.n_cells == 3

    def test_two_triangles(self):
        """Two adjacent triangles have 5 unique edges."""
        mesh = _two_triangles()
        graph = mesh.to_edge_graph()

        assert graph.n_manifold_dims == 1
        assert graph.cells.shape[1] == 2
        assert graph.n_cells == 5

    def test_single_tet(self):
        """A single tetrahedron has 6 edges."""
        mesh = _single_tet()
        graph = mesh.to_edge_graph()

        assert graph.n_manifold_dims == 1
        assert graph.cells.shape[1] == 2
        assert graph.n_cells == 6

    def test_preserves_points(self):
        """Edge graph shares the same point coordinates."""
        mesh = _single_triangle()
        graph = mesh.to_edge_graph()

        assert graph.n_points == mesh.n_points
        assert torch.equal(graph.points, mesh.points)

    def test_preserves_spatial_dims(self):
        mesh = _single_tet()
        graph = mesh.to_edge_graph()
        assert graph.n_spatial_dims == 3

    def test_valid_cell_indices(self):
        mesh = _two_triangles()
        graph = mesh.to_edge_graph()
        assert graph.cells.min() >= 0
        assert graph.cells.max() < graph.n_points


# ---------------------------------------------------------------------------
# to_dual_graph
# ---------------------------------------------------------------------------


class TestToDualGraph:
    """Constructs a cell-adjacency graph with centroids as nodes and shared-face pairs
    as edges, verifying topology, centroid positions, cell_data-to-point_data
    remapping, and edge uniqueness."""

    def test_two_triangles_one_edge(self):
        """Two adjacent triangles produce 1 dual-graph edge."""
        mesh = _two_triangles()
        dual = mesh.to_dual_graph()

        assert dual.n_manifold_dims == 1
        assert dual.n_points == 2  # 2 cells -> 2 centroids
        assert dual.n_cells == 1  # 1 shared edge -> 1 dual edge

    def test_single_cell_no_edges(self):
        """A single cell has no neighbors, so dual graph has 0 edges."""
        mesh = _single_triangle()
        dual = mesh.to_dual_graph()

        assert dual.n_points == 1
        assert dual.n_cells == 0

    def test_two_tets_one_edge(self):
        """Two tets sharing a face produce 1 dual-graph edge."""
        mesh = _two_tets()
        dual = mesh.to_dual_graph()

        assert dual.n_points == 2
        assert dual.n_cells == 1

    def test_centroids_match(self):
        """Dual graph points should equal the parent mesh's cell centroids."""
        mesh = _two_triangles()
        dual = mesh.to_dual_graph()

        expected_centroids = mesh.cell_centroids
        torch.testing.assert_close(dual.points, expected_centroids)

    def test_cell_data_becomes_point_data(self):
        """Parent cell_data should become dual graph's point_data."""
        mesh = _triangle_with_data()
        dual = mesh.to_dual_graph()

        assert "pressure" in dual.point_data
        assert dual.point_data["pressure"].shape[0] == 2

    def test_valid_cell_indices(self):
        mesh = _two_tets()
        dual = mesh.to_dual_graph()
        if dual.n_cells > 0:
            assert dual.cells.min() >= 0
            assert dual.cells.max() < dual.n_points

    def test_no_duplicate_edges(self):
        """Each neighbor pair should appear exactly once."""
        mesh = _two_triangles()
        dual = mesh.to_dual_graph()

        if dual.n_cells > 0:
            sorted_cells, _ = dual.cells.sort(dim=1)
            unique_edges = torch.unique(sorted_cells, dim=0)
            assert unique_edges.shape[0] == dual.n_cells


# ---------------------------------------------------------------------------
# to_point_cloud
# ---------------------------------------------------------------------------


class TestToPointCloud:
    """Reduces a mesh to a 0D point cloud using either original vertices or cell
    centroids, verifying both modes, data preservation, and rejection of invalid
    point_source values."""

    def test_vertices_default(self):
        """Default point cloud from vertices."""
        mesh = _single_triangle()
        pc = mesh.to_point_cloud()

        assert pc.n_manifold_dims == 0
        assert pc.n_points == 3
        assert pc.n_cells == 0
        assert torch.equal(pc.points, mesh.points)

    def test_vertices_preserves_point_data(self):
        mesh = _triangle_with_data()
        pc = mesh.to_point_cloud(point_source="vertices")

        assert "temperature" in pc.point_data
        torch.testing.assert_close(
            pc.point_data["temperature"],
            mesh.point_data["temperature"],
        )

    def test_centroids(self):
        """Point cloud from cell centroids."""
        mesh = _two_triangles()
        pc = mesh.to_point_cloud(point_source="cell_centroids")

        assert pc.n_manifold_dims == 0
        assert pc.n_points == 2  # 2 cells -> 2 centroids
        assert pc.n_cells == 0
        torch.testing.assert_close(pc.points, mesh.cell_centroids)

    def test_centroids_maps_cell_data(self):
        """Cell data should become point data for centroid cloud."""
        mesh = _triangle_with_data()
        pc = mesh.to_point_cloud(point_source="cell_centroids")

        assert "pressure" in pc.point_data
        torch.testing.assert_close(
            pc.point_data["pressure"],
            mesh.cell_data["pressure"],
        )

    def test_invalid_point_source_raises(self):
        mesh = _single_triangle()
        try:
            mesh.to_point_cloud(point_source="invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_3d_mesh(self):
        """Point cloud works on 3D meshes."""
        mesh = _single_tet()
        pc = mesh.to_point_cloud()

        assert pc.n_manifold_dims == 0
        assert pc.n_points == 4
        assert pc.n_spatial_dims == 3
