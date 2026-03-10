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

"""Tests for facet extraction from simplicial meshes.

Tests validate facet (boundary) extraction across spatial dimensions, manifold
dimensions, and compute backends, with data aggregation strategies.
"""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh

### Helper Functions ###


def create_simple_mesh(n_spatial_dims: int, n_manifold_dims: int, device: str = "cpu"):
    """Create a simple mesh for testing."""
    if n_manifold_dims > n_spatial_dims:
        raise ValueError(
            f"Manifold dimension {n_manifold_dims} cannot exceed spatial dimension {n_spatial_dims}"
        )

    if n_manifold_dims == 0:
        if n_spatial_dims == 2:
            points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], device=device)
        elif n_spatial_dims == 3:
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]], device=device
            )
        else:
            raise ValueError(f"Unsupported {n_spatial_dims=}")
        cells = torch.arange(len(points), device=device, dtype=torch.int64).unsqueeze(1)
    elif n_manifold_dims == 1:
        if n_spatial_dims == 2:
            points = torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [1.5, 1.0], [0.5, 1.5]], device=device
            )
        elif n_spatial_dims == 3:
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
                device=device,
            )
        else:
            raise ValueError(f"Unsupported {n_spatial_dims=}")
        cells = torch.tensor([[0, 1], [1, 2], [2, 3]], device=device, dtype=torch.int64)
    elif n_manifold_dims == 2:
        if n_spatial_dims == 2:
            points = torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 0.5]], device=device
            )
        elif n_spatial_dims == 3:
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.5, 0.5, 0.5]],
                device=device,
            )
        else:
            raise ValueError(f"Unsupported {n_spatial_dims=}")
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], device=device, dtype=torch.int64)
    elif n_manifold_dims == 3:
        if n_spatial_dims == 3:
            points = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                device=device,
            )
            cells = torch.tensor(
                [[0, 1, 2, 3], [1, 2, 3, 4]], device=device, dtype=torch.int64
            )
        else:
            raise ValueError("3-simplices require 3D embedding space")
    else:
        raise ValueError(f"Unsupported {n_manifold_dims=}")

    return Mesh(points=points, cells=cells)


def assert_on_device(tensor: torch.Tensor, expected_device: str) -> None:
    """Assert tensor is on expected device."""
    actual_device = tensor.device.type
    assert actual_device == expected_device, (
        f"Device mismatch: tensor is on {actual_device!r}, expected {expected_device!r}"
    )


class TestBasicEdgeExtraction:
    """Test basic edge extraction functionality."""

    def test_single_triangle_to_edges(self):
        """A single triangle should produce 3 unique edges."""
        ### Create a simple triangle
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])

        mesh = Mesh(points=points, cells=cells)
        facet_mesh = mesh.get_facet_mesh()

        ### Should have 3 edges
        assert facet_mesh.n_cells == 3
        assert facet_mesh.n_manifold_dims == 1
        assert facet_mesh.n_spatial_dims == 2

        ### Edges should be canonical (sorted)
        expected_edges = torch.tensor([[0, 1], [0, 2], [1, 2]])
        assert torch.equal(
            torch.sort(facet_mesh.cells, dim=0)[0],
            expected_edges,
        )

    def test_two_triangles_shared_edge(self):
        """Two triangles sharing an edge should deduplicate that edge."""
        ### Create two triangles sharing edge [1, 2]
        points = torch.tensor(
            [
                [0.0, 0.0],  # 0
                [1.0, 0.0],  # 1
                [0.5, 1.0],  # 2
                [1.5, 0.5],  # 3
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],  # Triangle 1
                [1, 3, 2],  # Triangle 2 (shares edge [1, 2])
            ]
        )

        mesh = Mesh(points=points, cells=cells)
        facet_mesh = mesh.get_facet_mesh()

        ### Should have 5 unique edges, not 6
        # Triangle 1: [0,1], [0,2], [1,2]
        # Triangle 2: [1,2], [1,3], [2,3]
        # Unique: [0,1], [0,2], [1,2], [1,3], [2,3] = 5 edges
        assert facet_mesh.n_cells == 5

        expected_edges = torch.tensor(
            [
                [0, 1],
                [0, 2],
                [1, 2],
                [1, 3],
                [2, 3],
            ]
        )
        assert torch.equal(
            torch.sort(facet_mesh.cells, dim=0)[0],
            expected_edges,
        )

    def test_facet_mesh_to_points(self):
        """An edge mesh (1-simplices) should extract to 0-simplices."""
        ### Create a simple line segment mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ]
        )
        # Two connected line segments
        cells = torch.tensor(
            [
                [0, 1],
                [1, 2],
            ]
        )

        mesh = Mesh(points=points, cells=cells)
        facet_mesh = mesh.get_facet_mesh()

        ### Should extract unique vertices
        assert facet_mesh.n_manifold_dims == 0
        # Each edge produces 2 vertices, but vertex 1 is shared
        # So we get vertices: [0], [1], [1], [2] -> unique: [0], [1], [2]
        assert facet_mesh.n_cells == 3

        ### Check that we have the right vertices
        expected_vertices = torch.tensor([[0], [1], [2]])
        assert torch.equal(
            torch.sort(facet_mesh.cells, dim=0)[0],
            expected_vertices,
        )

    def test_point_cloud_raises_error(self):
        """A point cloud (0-simplices) should raise an error."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ]
        )
        # Point cloud: each "face" is a single vertex
        cells = torch.tensor([[0], [1], [2]])

        mesh = Mesh(points=points, cells=cells)

        with pytest.raises(
            ValueError, match="Would result in negative manifold dimension"
        ):
            mesh.get_facet_mesh()


class TestDataInheritance:
    """Test data inheritance from parent mesh to edge mesh."""

    @pytest.mark.parametrize(
        "data_aggregation",
        [
            pytest.param("mean", id="mean"),
            pytest.param("area_weighted", id="area_weighted"),
            pytest.param("inverse_distance", id="inverse_distance"),
        ],
    )
    def test_cell_data_inheritance(self, data_aggregation):
        """Test face data inheritance with different aggregation strategies."""
        ### Create two triangles with known geometry
        points = torch.tensor(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [2.0, 2.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],  # Triangle 1
                [1, 3, 2],  # Triangle 2 (shares edge [1, 2])
            ]
        )

        cell_data = {
            "value": torch.tensor([100.0, 300.0]),
        }

        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)
        facet_mesh = mesh.get_facet_mesh(
            data_source="cells", data_aggregation=data_aggregation
        )

        ### Shared edge [1, 2] should have aggregated value
        shared_edge_idx = torch.where(
            (facet_mesh.cells[:, 0] == 1) & (facet_mesh.cells[:, 1] == 2)
        )[0]
        assert len(shared_edge_idx) == 1

        ### Compute expected value based on aggregation method
        if data_aggregation == "mean":
            expected_value = torch.tensor(200.0)  # (100 + 300) / 2
        elif data_aggregation == "area_weighted":
            areas = mesh.cell_areas
            expected_value = (100.0 * areas[0] + 300.0 * areas[1]) / (
                areas[0] + areas[1]
            )
        elif data_aggregation == "inverse_distance":
            edge_centroid = (points[1] + points[2]) / 2
            tri1_centroid = points[cells[0]].mean(dim=0)
            tri2_centroid = points[cells[1]].mean(dim=0)
            dist1 = torch.norm(edge_centroid - tri1_centroid)
            dist2 = torch.norm(edge_centroid - tri2_centroid)
            w1, w2 = 1.0 / dist1, 1.0 / dist2
            expected_value = (100.0 * w1 + 300.0 * w2) / (w1 + w2)

        assert torch.isclose(
            facet_mesh.cell_data["value"][shared_edge_idx[0]],
            expected_value,
            rtol=1e-5,
        )

    def test_point_data_inheritance(self):
        """Test point data inheritance (averaging from boundary vertices)."""
        ### Create a triangle with point data
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])

        point_data = {
            "value": torch.tensor([0.0, 1.0, 2.0]),
        }

        mesh = Mesh(points=points, cells=cells, point_data=point_data)
        facet_mesh = mesh.get_facet_mesh(data_source="points")

        ### Each edge should have averaged value from its endpoints
        # Edge [0, 1]: (0.0 + 1.0) / 2 = 0.5
        # Edge [0, 2]: (0.0 + 2.0) / 2 = 1.0
        # Edge [1, 2]: (1.0 + 2.0) / 2 = 1.5
        expected_values = {(0, 1): 0.5, (0, 2): 1.0, (1, 2): 1.5}
        for (v0, v1), expected in expected_values.items():
            edge_idx = torch.where(
                (facet_mesh.cells[:, 0] == v0) & (facet_mesh.cells[:, 1] == v1)
            )[0]
            assert torch.isclose(
                facet_mesh.cell_data["value"][edge_idx[0]],
                torch.tensor(expected),
                rtol=1e-5,
            ), f"Edge [{v0}, {v1}] expected {expected}"

    def test_multidimensional_data_aggregation(self):
        """Test that multidimensional face data is aggregated correctly."""
        ### Create two triangles
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [1.5, 0.5],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )

        ### Multi-dimensional face data (e.g., velocity vectors)
        cell_data = {
            "velocity": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
        }

        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)
        facet_mesh = mesh.get_facet_mesh(data_source="cells", data_aggregation="mean")

        ### Shared edge should have averaged velocity
        shared_edge_idx = torch.where(
            (facet_mesh.cells[:, 0] == 1) & (facet_mesh.cells[:, 1] == 2)
        )[0]
        assert len(shared_edge_idx) == 1

        expected_velocity = torch.tensor([0.5, 0.5])
        assert torch.allclose(
            facet_mesh.cell_data["velocity"][shared_edge_idx[0]],
            expected_velocity,
            rtol=1e-5,
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_cell_data(self):
        """Edge extraction should work with no face data."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])

        mesh = Mesh(points=points, cells=cells)
        facet_mesh = mesh.get_facet_mesh()

        assert facet_mesh.n_cells == 3
        assert len(facet_mesh.cell_data.keys()) == 0

    def test_cached_properties_not_inherited(self):
        """Cached properties should not be inherited from parent mesh."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])

        mesh = Mesh(points=points, cells=cells)

        ### Access cached properties to populate them
        _ = mesh.cell_centroids
        _ = mesh.cell_areas

        ### Extract edge mesh
        facet_mesh = mesh.get_facet_mesh()

        ### Cached properties should not be inherited by facet mesh
        assert facet_mesh._cache.get(("cell", "centroids"), None) is None
        assert facet_mesh._cache.get(("cell", "areas"), None) is None


class TestRigorousAggregation:
    """Rigorous tests for data aggregation with exact value verification."""

    def test_three_triangles_sharing_edge(self):
        """Test aggregation when three cells share a single edge."""
        ### Create three triangles sharing edge [1, 2]
        points = torch.tensor(
            [
                [0.0, 0.0],  # 0
                [1.0, 0.0],  # 1
                [0.5, 1.0],  # 2
                [1.5, 0.5],  # 3
                [0.5, -1.0],  # 4
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],  # Triangle 1: shares edge [1,2]
                [1, 3, 2],  # Triangle 2: shares edge [1,2]
                [1, 2, 4],  # Triangle 3: shares edge [1,2]
            ]
        )

        cell_data = {
            "value": torch.tensor([10.0, 20.0, 30.0]),
        }

        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)
        facet_mesh = mesh.get_facet_mesh(data_source="cells", data_aggregation="mean")

        ### Edge [1, 2] should have mean of all three values
        shared_edge_idx = torch.where(
            (facet_mesh.cells[:, 0] == 1) & (facet_mesh.cells[:, 1] == 2)
        )[0]
        assert len(shared_edge_idx) == 1

        expected_mean = (10.0 + 20.0 + 30.0) / 3.0
        assert torch.isclose(
            facet_mesh.cell_data["value"][shared_edge_idx[0]],
            torch.tensor(expected_mean),
            rtol=1e-6,
        )

    def test_area_weighted_with_exact_areas(self):
        """Test area-weighted aggregation with manually computed areas."""
        ### Create two triangles with known areas that share an edge
        points2 = torch.tensor(
            [
                [0.0, 0.0],  # 0
                [2.0, 0.0],  # 1
                [0.0, 1.0],  # 2
                [2.0, 2.0],  # 3
            ]
        )
        cells2 = torch.tensor(
            [
                [0, 1, 2],  # Triangle 1: area = 1.0
                [1, 3, 2],  # Triangle 2: area = 2.0, shares edge [1,2]
            ]
        )

        cell_data2 = {
            "temperature": torch.tensor([100.0, 300.0]),
        }

        mesh2 = Mesh(points=points2, cells=cells2, cell_data=cell_data2)

        ### Verify areas
        areas2 = mesh2.cell_areas
        assert torch.isclose(areas2[0], torch.tensor(1.0), rtol=1e-5)
        assert torch.isclose(areas2[1], torch.tensor(2.0), rtol=1e-5)

        facet_mesh = mesh2.get_facet_mesh(
            data_source="cells", data_aggregation="area_weighted"
        )

        ### Edge [1, 2] is shared and should be area-weighted
        shared_edge_idx = torch.where(
            (facet_mesh.cells[:, 0] == 1) & (facet_mesh.cells[:, 1] == 2)
        )[0]

        # Expected: (100.0 * 1.0 + 300.0 * 2.0) / (1.0 + 2.0) = 700 / 3 = 233.333...
        expected_temp = (100.0 * 1.0 + 300.0 * 2.0) / (1.0 + 2.0)

        assert torch.isclose(
            facet_mesh.cell_data["temperature"][shared_edge_idx[0]],
            torch.tensor(expected_temp),
            rtol=1e-5,
        )

    def test_boundary_vs_interior_edges(self):
        """Test that boundary edges (1 parent) and interior edges (2+ parents) are correctly distinguished."""
        ### Create a simple quad made of two triangles
        points = torch.tensor(
            [
                [0.0, 0.0],  # 0
                [1.0, 0.0],  # 1
                [1.0, 1.0],  # 2
                [0.0, 1.0],  # 3
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],  # Lower triangle
                [0, 2, 3],  # Upper triangle
            ]
        )

        cell_data = {
            "id": torch.tensor([1.0, 2.0]),
        }

        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)
        facet_mesh = mesh.get_facet_mesh(data_source="cells", data_aggregation="mean")

        ### Should have 5 edges total
        assert facet_mesh.n_cells == 5

        ### Verify interior edge averages both IDs, boundary edge keeps single ID
        expected_ids = {
            (0, 2): 1.5,  # Interior: (1.0 + 2.0) / 2
            (0, 1): 1.0,  # Boundary: only face 1
        }
        for (v0, v1), expected_id in expected_ids.items():
            edge_idx = torch.where(
                (facet_mesh.cells[:, 0] == v0) & (facet_mesh.cells[:, 1] == v1)
            )[0]
            assert len(edge_idx) == 1
            assert torch.isclose(
                facet_mesh.cell_data["id"][edge_idx[0]],
                torch.tensor(expected_id),
                rtol=1e-6,
            ), f"Edge [{v0}, {v1}] expected id {expected_id}"

    def test_multidimensional_point_data(self):
        """Test point data inheritance with multidimensional data (e.g., vectors)."""
        ### Create triangle with 2D velocity data at each point
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])

        point_data = {
            "velocity": torch.tensor(
                [
                    [1.0, 0.0],  # Point 0
                    [0.0, 1.0],  # Point 1
                    [1.0, 1.0],  # Point 2
                ]
            ),
        }

        mesh = Mesh(points=points, cells=cells, point_data=point_data)
        facet_mesh = mesh.get_facet_mesh(data_source="points")

        ### Each edge should average velocities of its endpoint vertices
        expected_velocities = {
            (0, 1): torch.tensor([0.5, 0.5]),  # ([1,0] + [0,1]) / 2
            (1, 2): torch.tensor([0.5, 1.0]),  # ([0,1] + [1,1]) / 2
        }
        for (v0, v1), expected_vel in expected_velocities.items():
            edge_idx = torch.where(
                (facet_mesh.cells[:, 0] == v0) & (facet_mesh.cells[:, 1] == v1)
            )[0]
            assert torch.allclose(
                facet_mesh.cell_data["velocity"][edge_idx[0]],
                expected_vel,
                rtol=1e-6,
            ), f"Edge [{v0}, {v1}] velocity mismatch"

    def test_tet_to_triangles_exact_count(self):
        """Test that a single tet produces exactly 4 unique triangular cells."""
        ### Single tetrahedron
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2, 3]])

        mesh = Mesh(points=points, cells=cells)
        facet_mesh = mesh.get_facet_mesh()

        ### Should produce exactly 4 triangular cells
        assert facet_mesh.n_cells == 4
        assert facet_mesh.n_manifold_dims == 2

        ### Verify all 4 expected triangles are present
        expected_triangles = torch.tensor(
            [
                [0, 1, 2],  # Exclude vertex 3
                [0, 1, 3],  # Exclude vertex 2
                [0, 2, 3],  # Exclude vertex 1
                [1, 2, 3],  # Exclude vertex 0
            ]
        )

        # Sort both for comparison
        actual_sorted = torch.sort(facet_mesh.cells, dim=1)[0]
        actual_sorted = torch.sort(actual_sorted, dim=0)[0]
        expected_sorted = torch.sort(expected_triangles, dim=1)[0]
        expected_sorted = torch.sort(expected_sorted, dim=0)[0]

        assert torch.equal(actual_sorted, expected_sorted)

    def test_two_tets_sharing_triangle(self):
        """Test two tetrahedra sharing a triangular face."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # 0
                [1.0, 0.0, 0.0],  # 1
                [0.0, 1.0, 0.0],  # 2
                [0.0, 0.0, 1.0],  # 3
                [0.0, 0.0, -1.0],  # 4
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2, 3],  # Tet 1
                [0, 1, 2, 4],  # Tet 2 (shares triangle [0,1,2])
            ]
        )

        cell_data = {
            "tet_id": torch.tensor([1.0, 2.0]),
        }

        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)
        facet_mesh = mesh.get_facet_mesh(data_source="cells", data_aggregation="mean")

        ### Should have 7 unique triangular cells (4 + 4 - 1 shared)
        assert facet_mesh.n_cells == 7

        ### Shared triangle [0, 1, 2] should average both tet IDs
        shared_tri_idx = torch.where(
            (facet_mesh.cells[:, 0] == 0)
            & (facet_mesh.cells[:, 1] == 1)
            & (facet_mesh.cells[:, 2] == 2)
        )[0]
        assert len(shared_tri_idx) == 1
        assert torch.isclose(
            facet_mesh.cell_data["tet_id"][shared_tri_idx[0]],
            torch.tensor(1.5),  # (1.0 + 2.0) / 2
            rtol=1e-6,
        )

    def test_edge_canonical_ordering(self):
        """Test that edges are stored in canonical (sorted) order."""
        ### Create triangles with vertices in different orders
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ]
        )
        # Define same triangle with different vertex orderings
        cells = torch.tensor(
            [
                [0, 1, 2],  # Standard order
                [2, 1, 0],  # Reversed order
            ]
        )

        mesh = Mesh(points=points, cells=cells)
        facet_mesh = mesh.get_facet_mesh()

        ### All edges should be in canonical order (sorted)
        for i in range(facet_mesh.n_cells):
            edge = facet_mesh.cells[i]
            assert edge[0] <= edge[1], f"Edge {edge} is not in canonical order"

        ### Since both triangles are identical, should only get 3 unique edges
        assert facet_mesh.n_cells == 3


class TestNestedTensorDicts:
    """Test edge extraction with nested TensorDict data structures."""

    def test_deeply_nested_cell_data(self):
        """Test aggregation with deeply nested TensorDicts."""
        from tensordict import TensorDict

        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 0.5]])
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])

        ### Create deeply nested structure
        cell_data = TensorDict(
            {
                "level1": TensorDict(
                    {
                        "level2": TensorDict(
                            {
                                "value": torch.tensor([1.0, 3.0]),
                            },
                            batch_size=torch.Size([2]),
                        ),
                    },
                    batch_size=torch.Size([2]),
                ),
            },
            batch_size=torch.Size([2]),
        )

        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)
        facet_mesh = mesh.get_facet_mesh(data_source="cells", data_aggregation="mean")

        ### Verify deeply nested aggregation
        shared_edge_idx = torch.where(
            (facet_mesh.cells[:, 0] == 1) & (facet_mesh.cells[:, 1] == 2)
        )[0]

        assert torch.isclose(
            facet_mesh.cell_data["level1"]["level2"]["value"][shared_edge_idx[0]],
            torch.tensor(2.0),  # (1 + 3) / 2
            rtol=1e-6,
        )

    def test_nested_point_data(self):
        """Test point data aggregation with nested TensorDicts."""
        from tensordict import TensorDict

        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])

        ### Create nested TensorDict for point data
        point_data = TensorDict(
            {
                "velocity": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                "nested": TensorDict(
                    {
                        "density": torch.tensor([1.0, 2.0, 3.0]),
                    },
                    batch_size=torch.Size([3]),
                ),
            },
            batch_size=torch.Size([3]),
        )

        mesh = Mesh(points=points, cells=cells, point_data=point_data)
        facet_mesh = mesh.get_facet_mesh(data_source="points")

        ### Edge [0, 1] should average point data from vertices 0 and 1
        edge_01_idx = torch.where(
            (facet_mesh.cells[:, 0] == 0) & (facet_mesh.cells[:, 1] == 1)
        )[0]
        idx = edge_01_idx[0]

        # Velocity: ([1, 0] + [0, 1]) / 2 = [0.5, 0.5]
        assert torch.allclose(
            facet_mesh.cell_data["velocity"][idx], torch.tensor([0.5, 0.5]), rtol=1e-6
        )
        # Nested density: (1.0 + 2.0) / 2 = 1.5
        assert torch.isclose(
            facet_mesh.cell_data["nested"]["density"][idx], torch.tensor(1.5), rtol=1e-6
        )

    def test_nested_with_area_weighting(self):
        """Test nested TensorDicts with area-weighted aggregation."""
        from tensordict import TensorDict

        points = torch.tensor(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [2.0, 2.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],  # Triangle 1: area = 1.0
                [1, 3, 2],  # Triangle 2: area = 2.0
            ]
        )

        cell_data = TensorDict(
            {
                "nested": TensorDict(
                    {
                        "value": torch.tensor([100.0, 300.0]),
                    },
                    batch_size=torch.Size([2]),
                ),
            },
            batch_size=torch.Size([2]),
        )

        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)

        ### Verify areas match expectations
        assert torch.isclose(mesh.cell_areas[0], torch.tensor(1.0), rtol=1e-5)
        assert torch.isclose(mesh.cell_areas[1], torch.tensor(2.0), rtol=1e-5)

        facet_mesh = mesh.get_facet_mesh(
            data_source="cells", data_aggregation="area_weighted"
        )

        ### Shared edge [1, 2] with area weighting
        shared_edge_idx = torch.where(
            (facet_mesh.cells[:, 0] == 1) & (facet_mesh.cells[:, 1] == 2)
        )[0]
        # Expected: (100.0 * 1.0 + 300.0 * 2.0) / (1.0 + 2.0) = 700 / 3
        expected = (100.0 * 1.0 + 300.0 * 2.0) / (1.0 + 2.0)
        assert torch.isclose(
            facet_mesh.cell_data["nested"]["value"][shared_edge_idx[0]],
            torch.tensor(expected),
            rtol=1e-5,
        )

    def test_mixed_nested_and_flat_data(self):
        """Test aggregation with mix of flat and nested data."""
        from tensordict import TensorDict

        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 0.5]])
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])

        cell_data = TensorDict(
            {
                "flat_scalar": torch.tensor([10.0, 20.0]),
                "flat_vector": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "nested": TensorDict(
                    {
                        "a": torch.tensor([100.0, 200.0]),
                        "b": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                    },
                    batch_size=torch.Size([2]),
                ),
            },
            batch_size=torch.Size([2]),
        )

        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)
        facet_mesh = mesh.get_facet_mesh(data_source="cells", data_aggregation="mean")

        shared_edge_idx = torch.where(
            (facet_mesh.cells[:, 0] == 1) & (facet_mesh.cells[:, 1] == 2)
        )[0]
        idx = shared_edge_idx[0]

        ### Check all data types averaged correctly
        assert torch.isclose(
            facet_mesh.cell_data["flat_scalar"][idx], torch.tensor(15.0), rtol=1e-6
        )
        assert torch.allclose(
            facet_mesh.cell_data["flat_vector"][idx],
            torch.tensor([2.0, 3.0]),
            rtol=1e-6,
        )
        assert torch.isclose(
            facet_mesh.cell_data["nested"]["a"][idx], torch.tensor(150.0), rtol=1e-6
        )
        assert torch.allclose(
            facet_mesh.cell_data["nested"]["b"][idx],
            torch.tensor([6.0, 7.0]),
            rtol=1e-6,
        )


class TestHigherCodimension:
    """Test extraction of higher-codimension meshes."""

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims,codim,expected_manifold_dim,expected_n_cells,expected_cell_size",
        [
            pytest.param(2, 2, 2, 0, 4, 1, id="triangles_to_vertices"),
            pytest.param(3, 3, 2, 1, 6, 2, id="tets_to_edges"),
            pytest.param(3, 3, 3, 0, 4, 1, id="tets_to_vertices"),
        ],
    )
    def test_basic_higher_codimension(
        self,
        n_spatial_dims,
        n_manifold_dims,
        codim,
        expected_manifold_dim,
        expected_n_cells,
        expected_cell_size,
    ):
        """Test higher codimension extraction across mesh types."""
        if n_spatial_dims == 2:
            points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        else:
            points = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            cells = torch.tensor([[0, 1, 2, 3]])

        mesh = Mesh(points=points, cells=cells)
        facet_mesh = mesh.get_facet_mesh(manifold_codimension=codim)

        assert facet_mesh.n_manifold_dims == expected_manifold_dim
        assert facet_mesh.n_cells == expected_n_cells
        assert facet_mesh.cells.shape == (expected_n_cells, expected_cell_size)

        ### For tet→edges, verify all 6 edges are present
        if n_spatial_dims == 3 and codim == 2:
            expected_edges = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
            actual_edges = {tuple(edge.tolist()) for edge in facet_mesh.cells}
            assert actual_edges == expected_edges

        ### For triangles→vertices, verify sorted unique vertices
        if n_spatial_dims == 2 and codim == 2:
            expected_vertices = torch.tensor([[0], [1], [2], [3]])
            assert torch.equal(
                torch.sort(facet_mesh.cells, dim=0)[0],
                expected_vertices,
            )

    def test_codimension_too_large_raises_error(self):
        """Test that requesting too high a codimension raises an error."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])  # Triangle (n_manifold_dims = 2)

        mesh = Mesh(points=points, cells=cells)

        ### Codimension 3 would give manifold_dims = -1, should raise
        with pytest.raises(
            ValueError, match="Would result in negative manifold dimension"
        ):
            mesh.get_facet_mesh(manifold_codimension=3)

    def test_data_inheritance_with_codim2(self):
        """Test that data inheritance works correctly with higher codimension."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2, 3]])  # Single tetrahedron

        ### Add some cell data
        cell_data = {"pressure": torch.tensor([100.0])}

        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)
        edge_mesh = mesh.get_facet_mesh(
            manifold_codimension=2, data_source="cells", data_aggregation="mean"
        )

        ### All edges should inherit the same pressure value
        assert "pressure" in edge_mesh.cell_data
        assert torch.allclose(
            edge_mesh.cell_data["pressure"],
            torch.tensor([100.0] * 6),
        )

    def test_codim2_multiple_cells_shared_edge(self):
        """Test codimension 2 extraction with multiple tets sharing edges."""
        ### Create two tetrahedra sharing edge [1, 2]
        # First tet: [0, 1, 2, 3]
        # Second tet: [1, 2, 4, 5]
        # They share edge [1, 2]
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # 0
                [1.0, 0.0, 0.0],  # 1 - shared
                [0.5, 1.0, 0.0],  # 2 - shared
                [0.5, 0.5, 1.0],  # 3
                [1.5, 0.5, 0.5],  # 4
                [1.0, 1.0, 1.0],  # 5
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2, 3],  # First tetrahedron
                [1, 2, 4, 5],  # Second tetrahedron (shares edge [1,2])
            ]
        )

        ### Add different pressure values to each tet
        cell_data = {
            "pressure": torch.tensor([100.0, 200.0]),
            "temperature": torch.tensor([300.0, 500.0]),
        }

        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)
        edge_mesh = mesh.get_facet_mesh(
            manifold_codimension=2, data_source="cells", data_aggregation="mean"
        )

        ### First tet has C(4,2)=6 edges, second tet has 6 edges
        ### They share edge [1,2], so total unique edges = 6 + 6 - 1 = 11
        assert edge_mesh.n_cells == 11
        assert "pressure" in edge_mesh.cell_data
        assert "temperature" in edge_mesh.cell_data

        ### Verify pressure values for shared and boundary edges
        expected_pressures = {
            (1, 2): 150.0,  # Shared edge: (100 + 200) / 2
            (0, 1): 100.0,  # First tet only
            (4, 5): 200.0,  # Second tet only
        }
        for (v0, v1), expected_pressure in expected_pressures.items():
            edge_idx = torch.where(
                (edge_mesh.cells[:, 0] == v0) & (edge_mesh.cells[:, 1] == v1)
            )[0]
            assert len(edge_idx) == 1, f"Edge [{v0}, {v1}] should exist exactly once"
            assert torch.isclose(
                edge_mesh.cell_data["pressure"][edge_idx[0]],
                torch.tensor(expected_pressure),
                rtol=1e-5,
            ), f"Edge [{v0}, {v1}] pressure expected {expected_pressure}"

        ### Shared edge should also have aggregated temperature
        shared_edge_idx = torch.where(
            (edge_mesh.cells[:, 0] == 1) & (edge_mesh.cells[:, 1] == 2)
        )[0]
        assert torch.isclose(
            edge_mesh.cell_data["temperature"][shared_edge_idx[0]],
            torch.tensor(400.0),  # (300 + 500) / 2
            rtol=1e-5,
        )


class TestDifferentDevices:
    """Test edge extraction on different devices."""

    @pytest.mark.cuda
    def test_cuda_edge_extraction(self):
        """Test edge extraction on CUDA device (specific real-world case)."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            device="cuda",
        )
        cells = torch.tensor([[0, 1, 2]], device="cuda")

        mesh = Mesh(points=points, cells=cells)
        facet_mesh = mesh.get_facet_mesh()

        assert facet_mesh.points.device.type == "cuda"
        assert facet_mesh.cells.device.type == "cuda"
        assert facet_mesh.n_cells == 3


### Parametrized Tests for Exhaustive Dimensional Coverage ###


class TestFacetExtractionParametrized:
    """Parametrized tests for facet extraction across all dimensions and backends."""

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 1),  # Edges → Points in 2D
            (2, 2),  # Triangles → Edges in 2D
            (3, 1),  # Edges → Points in 3D
            (3, 2),  # Surfaces → Edges in 3D
            (3, 3),  # Volumes → Surfaces in 3D
        ],
    )
    def test_basic_facet_extraction_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test basic facet extraction and deduplication across all dimension combinations."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        facet_mesh = mesh.get_facet_mesh()

        # Verify dimensions
        assert facet_mesh.n_spatial_dims == n_spatial_dims, (
            f"Spatial dims should be preserved: {facet_mesh.n_spatial_dims=} != {n_spatial_dims=}"
        )
        assert facet_mesh.n_manifold_dims == n_manifold_dims - 1, (
            f"Manifold dims should decrease by 1: {facet_mesh.n_manifold_dims=} != {n_manifold_dims - 1=}"
        )

        # Verify device consistency
        assert_on_device(facet_mesh.points, device)
        assert_on_device(facet_mesh.cells, device)

        # Verify facets exist
        assert facet_mesh.n_cells > 0, "Should extract at least some facets"

        # Verify cell shape
        expected_verts_per_facet = n_manifold_dims
        assert facet_mesh.cells.shape[1] == expected_verts_per_facet, (
            f"Facets should have {expected_verts_per_facet} vertices, "
            f"got {facet_mesh.cells.shape[1]}"
        )

        # Verify deduplication (facets are unique)
        if mesh.n_cells >= 2:
            sorted_facets = torch.sort(facet_mesh.cells, dim=1)[0]
            unique_facets = torch.unique(sorted_facets, dim=0)
            assert unique_facets.shape[0] == sorted_facets.shape[0], (
                f"Found duplicate facets: {sorted_facets.shape[0]} facets, "
                f"but only {unique_facets.shape[0]} unique"
            )

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [(2, 2), (3, 2), (3, 3)],
    )
    @pytest.mark.parametrize(
        "data_aggregation",
        ["mean", "area_weighted", "inverse_distance"],
    )
    def test_data_aggregation_parametrized(
        self, n_spatial_dims, n_manifold_dims, data_aggregation, device
    ):
        """Test all data aggregation strategies across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        # Add some cell data
        cell_data_values = (
            torch.arange(mesh.n_cells, dtype=torch.float32, device=device) * 10.0
        )
        mesh.cell_data["value"] = cell_data_values

        facet_mesh = mesh.get_facet_mesh(
            data_source="cells",
            data_aggregation=data_aggregation,
        )

        # Verify data was aggregated
        assert "value" in facet_mesh.cell_data, (
            f"Cell data should be aggregated with {data_aggregation=}"
        )
        assert facet_mesh.cell_data["value"].shape[0] == facet_mesh.n_cells, (
            "Aggregated data should have one value per facet"
        )

        # Verify device consistency
        assert_on_device(facet_mesh.cell_data["value"], device)

        # Verify values are reasonable (should be within range of original data)
        min_original = cell_data_values.min()
        max_original = cell_data_values.max()
        min_facet = facet_mesh.cell_data["value"].min()
        max_facet = facet_mesh.cell_data["value"].max()

        assert min_facet >= min_original, (
            f"Facet min value should be >= original min: {min_facet=}, {min_original=}"
        )
        assert max_facet <= max_original, (
            f"Facet max value should be <= original max: {max_facet=}, {max_original=}"
        )

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
        ],
    )
    def test_global_data_preserved_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test that global data is preserved across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        # Add global data
        mesh.global_data["time"] = torch.tensor(42.0, device=device)
        mesh.global_data["iteration"] = torch.tensor(100, device=device)

        facet_mesh = mesh.get_facet_mesh()

        # Verify global data preserved
        assert "time" in facet_mesh.global_data
        assert "iteration" in facet_mesh.global_data
        assert torch.equal(
            facet_mesh.global_data["time"], torch.tensor(42.0, device=device)
        )
        assert torch.equal(
            facet_mesh.global_data["iteration"], torch.tensor(100, device=device)
        )

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [(2, 2), (3, 2), (3, 3)],
    )
    def test_data_inheritance_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test point data and multidimensional cell data inheritance across dimensions."""
        torch.manual_seed(42)
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        # Test point data aggregation
        point_values = torch.arange(mesh.n_points, dtype=torch.float32, device=device)
        mesh.point_data["point_id"] = point_values

        facet_mesh_pt = mesh.get_facet_mesh(data_source="points")
        assert "point_id" in facet_mesh_pt.cell_data, (
            "Point data should be aggregated to facet cell_data"
        )
        assert_on_device(facet_mesh_pt.cell_data["point_id"], device)

        # Test multidimensional cell data aggregation
        velocity = torch.randn(mesh.n_cells, n_spatial_dims, device=device)
        mesh.cell_data["velocity"] = velocity

        facet_mesh_cd = mesh.get_facet_mesh(
            data_source="cells",
            data_aggregation="mean",
        )

        assert "velocity" in facet_mesh_cd.cell_data
        assert facet_mesh_cd.cell_data["velocity"].shape == (
            facet_mesh_cd.n_cells,
            n_spatial_dims,
        ), f"Velocity shape mismatch: {facet_mesh_cd.cell_data['velocity'].shape=}"
        assert_on_device(facet_mesh_cd.cell_data["velocity"], device)
