"""Tests for projection operations (extrusion, embedding, spatial dimension changes)."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.projections import embed_in_spatial_dims, extrude


class TestExtrude:
    """Test suite for mesh extrusion functionality."""

    def test_extrude_point_to_edge_2d(self):
        """Test extruding a 0D point cloud to 1D edges in 2D space."""
        ### Create a simple point cloud (0D manifold in 2D space)
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0], [1], [2]], dtype=torch.int64)  # 0-simplices
        mesh = Mesh(points=points, cells=cells)

        assert mesh.n_manifold_dims == 0
        assert mesh.n_spatial_dims == 2
        assert mesh.n_cells == 3

        ### Extrude along [0, 1] direction
        extruded = extrude(mesh, vector=[0.0, 1.0])

        ### Verify dimensions
        assert extruded.n_manifold_dims == 1
        assert extruded.n_spatial_dims == 2
        assert extruded.n_points == 6  # 3 original + 3 extruded
        assert extruded.n_cells == 3  # 3 edges (1 per original point)

        ### Verify point positions
        expected_points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],  # Original
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 2.0],  # Extruded
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(extruded.points, expected_points)

        ### Verify cells (edges connecting original to extruded)
        # Each 0-simplex [i] becomes 1 edge [i', i] or [i, i']
        # According to our algorithm: child 0 has [v0', v0]
        expected_cells = torch.tensor([[3, 0], [4, 1], [5, 2]], dtype=torch.int64)
        assert torch.equal(extruded.cells, expected_cells)

    def test_extrude_edge_to_triangle_2d(self):
        """Test extruding a 1D edge to 2D triangles in 2D space."""
        ### Create a single edge (1D manifold in 2D space)
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)  # 1-simplex (edge)
        mesh = Mesh(points=points, cells=cells)

        assert mesh.n_manifold_dims == 1
        assert mesh.n_spatial_dims == 2

        ### Extrude along [0, 1] direction
        extruded = extrude(mesh, vector=[0.0, 1.0])

        ### Verify dimensions
        assert extruded.n_manifold_dims == 2
        assert extruded.n_spatial_dims == 2
        assert extruded.n_points == 4  # 2 original + 2 extruded
        assert extruded.n_cells == 2  # 2 triangles (N+1 = 2 per edge)

        ### Verify point positions
        expected_points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
        )
        assert torch.allclose(extruded.points, expected_points)

        ### Verify cells
        # Edge [0, 1] becomes 2 triangles:
        #   Child 0: [v0', v0, v1] = [2, 0, 1]
        #   Child 1: [v0', v1', v1] = [2, 3, 1]
        expected_cells = torch.tensor([[2, 0, 1], [2, 3, 1]], dtype=torch.int64)
        assert torch.equal(extruded.cells, expected_cells)

        ### Verify total area (should equal width * height)
        total_area = extruded.cell_areas.sum()
        expected_area = 1.0 * 1.0  # Rectangle area
        assert torch.allclose(total_area, torch.tensor(expected_area), atol=1e-6)

    def test_extrude_edge_to_triangle_3d(self):
        """Test extruding a 1D edge to 2D triangles in 3D space."""
        ### Create a single edge in 3D space
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### Extrude along [0, 0, 1] direction (default)
        extruded = extrude(mesh)

        ### Verify dimensions
        assert extruded.n_manifold_dims == 2
        assert extruded.n_spatial_dims == 3
        assert extruded.n_points == 4
        assert extruded.n_cells == 2

        ### Verify point positions (default vector is [0, 0, 1])
        expected_points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        assert torch.allclose(extruded.points, expected_points)

    def test_extrude_triangle_to_tetrahedron(self):
        """Test extruding a 2D triangle to 3D tetrahedra in 3D space."""
        ### Create a single triangle (2D manifold in 3D space)
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)  # 2-simplex (triangle)
        mesh = Mesh(points=points, cells=cells)

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3

        ### Extrude along [0, 0, 1] direction (default)
        extruded = extrude(mesh)

        ### Verify dimensions
        assert extruded.n_manifold_dims == 3
        assert extruded.n_spatial_dims == 3
        assert extruded.n_points == 6  # 3 original + 3 extruded
        assert extruded.n_cells == 3  # 3 tetrahedra (N+1 = 3 per triangle)

        ### Verify point positions
        expected_points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],  # Original
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],  # Extruded
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(extruded.points, expected_points)

        ### Verify cells
        # Triangle [0, 1, 2] becomes 3 tetrahedra:
        #   Child 0: [v0', v0, v1, v2] = [3, 0, 1, 2]
        #   Child 1: [v0', v1', v1, v2] = [3, 4, 1, 2]
        #   Child 2: [v0', v1', v2', v2] = [3, 4, 5, 2]
        expected_cells = torch.tensor(
            [[3, 0, 1, 2], [3, 4, 1, 2], [3, 4, 5, 2]], dtype=torch.int64
        )
        assert torch.equal(extruded.cells, expected_cells)

        ### Verify total volume
        # Original triangle has area 0.5, extruded by height 1.0 → volume = 0.5
        total_volume = extruded.cell_areas.sum()  # "areas" is generic for n-volumes
        expected_volume = 0.5
        assert torch.allclose(total_volume, torch.tensor(expected_volume), atol=1e-6)

    def test_extrude_custom_vector(self):
        """Test extrusion with custom vector."""
        ### Create a triangle
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### Extrude with custom vector
        custom_vector = torch.tensor([1.0, 1.0, 2.0])
        extruded = extrude(mesh, vector=custom_vector)

        ### Verify extruded points
        expected_extruded = points + custom_vector
        assert torch.allclose(
            extruded.points[3:],
            expected_extruded,  # Last 3 points are extruded
        )

    def test_extrude_insufficient_spatial_dims_raises_error(self):
        """Test that extrusion raises ValueError when spatial dims are insufficient."""
        ### Create a 2D mesh in 2D space (can't extrude to 3D without new dims)
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 2

        ### Should raise ValueError by default
        with pytest.raises(
            ValueError, match="Cannot extrude.*without increasing spatial dimensions"
        ):
            extrude(mesh)

        ### Should also raise with explicit vector in 2D
        with pytest.raises(
            ValueError, match="Cannot extrude.*without increasing spatial dimensions"
        ):
            extrude(mesh, vector=[0.0, 1.0])

    def test_extrude_allow_new_spatial_dims(self):
        """Test extrusion with allow_new_spatial_dims=True."""
        ### Create a 2D mesh in 2D space
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### Extrude with allow_new_spatial_dims=True
        extruded = extrude(mesh, allow_new_spatial_dims=True)

        ### Verify new spatial dimensions
        assert extruded.n_manifold_dims == 3
        assert extruded.n_spatial_dims == 3  # New dimension added
        assert extruded.n_points == 6
        assert extruded.n_cells == 3

        ### Verify that original points are padded with zeros
        # Original points should be padded: [x, y] → [x, y, 0]
        expected_original = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        assert torch.allclose(extruded.points[:3], expected_original)

        ### Extruded points should be [x, y, 1]
        expected_extruded = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float32
        )
        assert torch.allclose(extruded.points[3:], expected_extruded)

    def test_extrude_data_propagation_point_data(self):
        """Test that point_data is correctly duplicated during extrusion."""
        ### Create mesh with point data
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        point_data = TensorDict(
            {
                "temperature": torch.tensor([300.0, 400.0]),
                "velocity": torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
            },
            batch_size=[2],
        )
        mesh = Mesh(points=points, cells=cells, point_data=point_data)

        ### Extrude
        extruded = extrude(mesh, vector=[0.0, 1.0])

        ### Verify point_data is duplicated
        assert extruded.n_points == 4
        assert "temperature" in extruded.point_data
        assert "velocity" in extruded.point_data

        # First 2 points should have original data
        assert torch.allclose(
            extruded.point_data["temperature"][:2], torch.tensor([300.0, 400.0])
        )
        # Last 2 points should have duplicated data
        assert torch.allclose(
            extruded.point_data["temperature"][2:], torch.tensor([300.0, 400.0])
        )

        # Check vector data too
        assert torch.allclose(
            extruded.point_data["velocity"][:2], torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        )
        assert torch.allclose(
            extruded.point_data["velocity"][2:], torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        )

    def test_extrude_data_propagation_cell_data(self):
        """Test that cell_data is correctly replicated during extrusion."""
        ### Create mesh with cell data
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        cell_data = TensorDict(
            {"pressure": torch.tensor([101325.0]), "id": torch.tensor([42])},
            batch_size=[1],
        )
        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)

        ### Extrude (1D edge → 2D, creates 2 child cells per parent)
        extruded = extrude(mesh, vector=[0.0, 1.0])

        ### Verify cell_data is replicated
        assert extruded.n_cells == 2  # 1 edge becomes 2 triangles
        assert "pressure" in extruded.cell_data
        assert "id" in extruded.cell_data

        # Both child cells should have same data as parent
        assert torch.allclose(
            extruded.cell_data["pressure"], torch.tensor([101325.0, 101325.0])
        )
        assert torch.equal(extruded.cell_data["id"], torch.tensor([42, 42]))

    def test_extrude_multiple_cells(self):
        """Test extrusion with multiple parent cells."""
        ### Create two edges
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)  # Two edges
        cell_data = TensorDict(
            {"cell_id": torch.tensor([10, 20])},
            batch_size=[2],
        )
        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)

        ### Extrude
        extruded = extrude(mesh, vector=[0.0, 1.0])

        ### Verify dimensions
        assert extruded.n_cells == 4  # 2 edges × 2 children each = 4 triangles

        ### Verify cell_data replication maintains grouping
        # First 2 cells should have cell_id=10, next 2 should have cell_id=20
        expected_cell_ids = torch.tensor([10, 10, 20, 20])
        assert torch.equal(extruded.cell_data["cell_id"], expected_cell_ids)

    def test_extrude_empty_mesh(self):
        """Test extrusion of empty mesh (no cells)."""
        ### Create empty mesh
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        cells = torch.empty((0, 2), dtype=torch.int64)  # No cells
        mesh = Mesh(points=points, cells=cells)

        assert mesh.n_cells == 0

        ### Extrude
        extruded = extrude(mesh, vector=[0.0, 1.0])

        ### Verify: points are duplicated but no cells created
        assert extruded.n_points == 4  # 2 original + 2 extruded
        assert extruded.n_cells == 0  # Still no cells
        assert extruded.n_manifold_dims == 2  # Manifold dim still increases
        assert extruded.cells.shape == (0, 3)  # Shape is (0, n_vertices_per_cell)

    def test_extrude_capping_not_implemented(self):
        """Test that capping=True raises NotImplementedError."""
        ### Create simple mesh
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### Should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="Capping is not yet implemented"):
            extrude(mesh, capping=True)

    @pytest.mark.parametrize(
        "n_manifold_dims,n_spatial_dims",
        [
            (0, 1),  # Points in 1D → edges in 1D
            (0, 2),  # Points in 2D → edges in 2D
            (0, 3),  # Points in 3D → edges in 3D
            (1, 2),  # Edges in 2D → triangles in 2D
            (1, 3),  # Edges in 3D → triangles in 3D
            (2, 3),  # Triangles in 3D → tetrahedra in 3D
        ],
    )
    def test_extrude_various_dimensions(self, n_manifold_dims, n_spatial_dims):
        """Test extrusion across various manifold and spatial dimensions."""
        ### Create a simple mesh of the specified dimension
        n_vertices_per_cell = n_manifold_dims + 1

        # Create points: use identity-like pattern
        n_points = n_vertices_per_cell
        points = torch.zeros((n_points, n_spatial_dims), dtype=torch.float32)
        for i in range(min(n_points, n_spatial_dims)):
            points[i, i] = 1.0

        # Create a single cell
        cells = torch.arange(n_vertices_per_cell).unsqueeze(0)

        mesh = Mesh(points=points, cells=cells)

        ### Extrude with default vector
        extruded = extrude(mesh)

        ### Verify dimensions
        assert extruded.n_manifold_dims == n_manifold_dims + 1
        assert extruded.n_spatial_dims == n_spatial_dims
        assert extruded.n_points == 2 * n_points
        assert extruded.n_cells == n_manifold_dims + 1  # N+1 children per parent

        ### Verify all cells have positive volume/area
        assert (extruded.cell_areas > 0).all()

    def test_extrude_preserves_global_data(self):
        """Test that global_data is preserved during extrusion."""
        ### Create mesh with global data
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        global_data = TensorDict({"timestamp": torch.tensor(12345)}, batch_size=[])
        mesh = Mesh(points=points, cells=cells, global_data=global_data)

        ### Extrude
        extruded = extrude(mesh, vector=[0.0, 1.0])

        ### Verify global_data is preserved
        assert "timestamp" in extruded.global_data
        assert extruded.global_data["timestamp"] == 12345

    def test_extrude_cached_data_cleared(self):
        """Test that cached properties are not propagated."""
        ### Create mesh and trigger some cached computations
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Access some cached properties to populate cache
        _ = mesh.cell_centroids
        _ = mesh.cell_areas

        # Verify cache exists
        assert "_cache" in mesh.cell_data

        ### Extrude
        extruded = extrude(mesh)

        ### Verify cache is not in extruded mesh
        # The exclude("_cache") should prevent propagation
        assert (
            "_cache" not in extruded.cell_data or len(extruded.cell_data["_cache"]) == 0
        )

    def test_extrude_vector_as_list(self):
        """Test that vector can be provided as a list or tuple."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### Extrude with list
        extruded_list = extrude(mesh, vector=[0.5, 1.5])
        assert torch.allclose(
            extruded_list.points[2:], mesh.points + torch.tensor([0.5, 1.5])
        )

        ### Extrude with tuple
        extruded_tuple = extrude(mesh, vector=(0.5, 1.5))
        assert torch.allclose(
            extruded_tuple.points[2:], mesh.points + torch.tensor([0.5, 1.5])
        )

    def test_extrude_4d_to_5d(self):
        """Test high-dimensional extrusion: 3D manifold in 4D space → 4D manifold."""
        ### Create a 3-simplex (tetrahedron) in 4D space
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert mesh.n_manifold_dims == 3
        assert mesh.n_spatial_dims == 4

        ### Extrude (default vector is [0, 0, 0, 1])
        extruded = extrude(mesh)

        ### Verify dimensions
        assert extruded.n_manifold_dims == 4
        assert extruded.n_spatial_dims == 4
        assert extruded.n_points == 8  # 4 original + 4 extruded
        assert extruded.n_cells == 4  # 4 children (N+1 where N=3)

        ### Verify all cells have positive hypervolume
        assert (extruded.cell_areas > 0).all()

    def test_extrude_orientation_consistency(self):
        """Test that extrusion maintains consistent orientation."""
        ### Create a simple triangle with known orientation
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### Compute original normal (should point in +z direction)
        original_normal = mesh.cell_normals[0]
        assert original_normal[2] > 0  # Points upward

        ### Extrude upward
        extruded = extrude(mesh, vector=[0.0, 0.0, 1.0])

        ### All extruded tetrahedra should have positive volume
        # (negative volume would indicate inverted orientation)
        assert (extruded.cell_areas > 0).all()

    def test_extrude_with_zero_vector_raises_or_degenerates(self):
        """Test extrusion with zero vector creates degenerate cells."""
        ### Create simple mesh
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### Extrude with zero vector
        extruded = extrude(mesh, vector=[0.0, 0.0])

        ### Extruded points should be same as original
        assert torch.allclose(extruded.points[:2], extruded.points[2:])

        ### Cells should have zero area (degenerate)
        assert torch.allclose(extruded.cell_areas, torch.zeros(2))

    def test_extrude_vector_wrong_shape_raises_error(self):
        """Test that vector with wrong shape raises ValueError."""
        ### Create simple mesh
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### 2D vector (should be 1D)
        with pytest.raises(ValueError, match="Extrusion vector must be 1D"):
            extrude(mesh, vector=torch.tensor([[0.0, 1.0]]))

        ### 3D vector (should be 1D)
        with pytest.raises(ValueError, match="Extrusion vector must be 1D"):
            extrude(mesh, vector=torch.zeros((2, 2, 2)))

    def test_extrude_vector_too_many_dimensions_raises_error(self):
        """Test that vector with too many spatial dimensions raises ValueError."""
        ### Create simple mesh in 2D
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### Provide vector with 5 dimensions (mesh is 2D, target would be 3D max)
        with pytest.raises(ValueError, match="Extrusion vector has .* dimensions but"):
            extrude(mesh, vector=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))

    def test_extrude_vector_too_small_gets_padded(self):
        """Test that vector with too few dimensions gets padded."""
        ### Create mesh in 3D space
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### Provide 2D vector for 3D mesh (should be padded)
        extruded = extrude(mesh, vector=torch.tensor([1.0, 2.0]))

        ### Verify extruded points: original + [1.0, 2.0, 0.0] (padded)
        expected_extruded = mesh.points + torch.tensor([1.0, 2.0, 0.0])
        assert torch.allclose(extruded.points[2:], expected_extruded)


class TestEmbedInSpatialDims:
    """Test suite for spatial dimension embedding/projection functionality."""

    def test_embed_2d_to_3d(self):
        """Test embedding a 2D mesh in 2D space into 3D space."""
        ### Create 2D triangle in 2D space
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh_2d = Mesh(points=points, cells=cells)

        assert mesh_2d.n_spatial_dims == 2
        assert mesh_2d.n_manifold_dims == 2
        assert mesh_2d.codimension == 0

        ### Embed in 3D space
        mesh_3d = embed_in_spatial_dims(mesh_2d, target_n_spatial_dims=3)

        ### Verify dimensions
        assert mesh_3d.n_spatial_dims == 3
        assert mesh_3d.n_manifold_dims == 2  # Manifold dim unchanged
        assert mesh_3d.codimension == 1  # Now codimension-1!
        assert mesh_3d.n_points == 3
        assert mesh_3d.n_cells == 1

        ### Verify points are padded with zeros
        expected_points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        assert torch.allclose(mesh_3d.points, expected_points)

        ### Verify cells unchanged
        assert torch.equal(mesh_3d.cells, cells)

        ### Verify we can now compute normals (codimension-1)
        normals = mesh_3d.cell_normals
        assert normals.shape == (1, 3)
        # Normal should point in z-direction
        assert torch.allclose(normals[0, 2].abs(), torch.tensor(1.0))

    def test_project_3d_to_2d(self):
        """Test projecting a 2D mesh in 3D space down to 2D space."""
        ### Create 2D triangle in 3D space
        points = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh_3d = Mesh(points=points, cells=cells)

        assert mesh_3d.n_spatial_dims == 3
        assert mesh_3d.codimension == 1

        ### Project to 2D space
        mesh_2d = embed_in_spatial_dims(mesh_3d, target_n_spatial_dims=2)

        ### Verify dimensions
        assert mesh_2d.n_spatial_dims == 2
        assert mesh_2d.n_manifold_dims == 2
        assert mesh_2d.codimension == 0  # No longer codimension-1
        assert mesh_2d.n_points == 3
        assert mesh_2d.n_cells == 1

        ### Verify points are sliced (z-coordinate removed)
        expected_points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
        )
        assert torch.allclose(mesh_2d.points, expected_points)

        ### Verify cells unchanged
        assert torch.equal(mesh_2d.cells, cells)

    def test_embed_1d_curve_2d_to_3d(self):
        """Test embedding a 1D curve in 2D space into 3D space."""
        ### Create edge in 2D
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        mesh_2d = Mesh(points=points, cells=cells)

        assert mesh_2d.n_manifold_dims == 1
        assert mesh_2d.n_spatial_dims == 2
        assert mesh_2d.codimension == 1

        ### Embed in 3D
        mesh_3d = embed_in_spatial_dims(mesh_2d, target_n_spatial_dims=3)

        ### Verify dimensions
        assert mesh_3d.n_manifold_dims == 1
        assert mesh_3d.n_spatial_dims == 3
        assert mesh_3d.codimension == 2  # Higher codimension

        ### Verify points padded
        expected_points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32
        )
        assert torch.allclose(mesh_3d.points, expected_points)

    def test_embed_no_change_returns_same_mesh(self):
        """Test that embedding to current dimension returns unchanged mesh."""
        ### Create mesh
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### Embed to same dimension
        result = embed_in_spatial_dims(mesh, target_n_spatial_dims=3)

        ### Should be same object (no-op)
        assert result is mesh

    def test_embed_preserves_point_data(self):
        """Test that point_data is preserved during embedding."""
        ### Create mesh with point data
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        point_data = TensorDict(
            {
                "temperature": torch.tensor([300.0, 400.0]),
                "pressure": torch.tensor([101325.0, 101325.0]),
            },
            batch_size=[2],
        )
        mesh = Mesh(points=points, cells=cells, point_data=point_data)

        ### Embed in 3D
        embedded = embed_in_spatial_dims(mesh, target_n_spatial_dims=3)

        ### Verify point_data preserved
        assert "temperature" in embedded.point_data
        assert "pressure" in embedded.point_data
        assert torch.allclose(
            embedded.point_data["temperature"], torch.tensor([300.0, 400.0])
        )
        assert torch.allclose(
            embedded.point_data["pressure"], torch.tensor([101325.0, 101325.0])
        )

    def test_embed_preserves_cell_data(self):
        """Test that cell_data is preserved during embedding."""
        ### Create mesh with cell data
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        cell_data = TensorDict(
            {"region_id": torch.tensor([42]), "density": torch.tensor([1.225])},
            batch_size=[1],
        )
        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)

        ### Embed in 3D
        embedded = embed_in_spatial_dims(mesh, target_n_spatial_dims=3)

        ### Verify cell_data preserved
        assert "region_id" in embedded.cell_data
        assert "density" in embedded.cell_data
        assert embedded.cell_data["region_id"] == 42
        assert torch.allclose(embedded.cell_data["density"], torch.tensor([1.225]))

    def test_embed_preserves_global_data(self):
        """Test that global_data is preserved during embedding."""
        ### Create mesh with global data
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        global_data = TensorDict({"simulation_time": torch.tensor(1.5)}, batch_size=[])
        mesh = Mesh(points=points, cells=cells, global_data=global_data)

        ### Embed in 3D
        embedded = embed_in_spatial_dims(mesh, target_n_spatial_dims=3)

        ### Verify global_data preserved
        assert "simulation_time" in embedded.global_data
        assert torch.allclose(
            embedded.global_data["simulation_time"], torch.tensor(1.5)
        )

    def test_embed_clears_cached_properties(self):
        """Test that cached geometric properties are cleared."""
        ### Create mesh and trigger cache
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Populate cache by accessing properties
        _ = mesh.cell_centroids
        _ = mesh.cell_areas
        _ = mesh.cell_normals

        # Verify cache exists
        assert "_cache" in mesh.cell_data
        assert len(mesh.cell_data["_cache"]) > 0

        ### Embed in 4D
        embedded = embed_in_spatial_dims(mesh, target_n_spatial_dims=4)

        ### Verify cache is cleared
        # Cache should either not exist or be empty
        if "_cache" in embedded.cell_data:
            assert len(embedded.cell_data["_cache"]) == 0

    def test_embed_multiple_steps(self):
        """Test embedding through multiple dimension changes."""
        ### Start with 1D in 2D
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        mesh_2d = Mesh(points=points, cells=cells)

        ### Embed to 3D
        mesh_3d = embed_in_spatial_dims(mesh_2d, target_n_spatial_dims=3)
        assert mesh_3d.n_spatial_dims == 3
        assert torch.allclose(
            mesh_3d.points, torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )

        ### Embed to 4D
        mesh_4d = embed_in_spatial_dims(mesh_3d, target_n_spatial_dims=4)
        assert mesh_4d.n_spatial_dims == 4
        assert torch.allclose(
            mesh_4d.points, torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        )

        ### Project back to 2D
        mesh_2d_again = embed_in_spatial_dims(mesh_4d, target_n_spatial_dims=2)
        assert mesh_2d_again.n_spatial_dims == 2
        assert torch.allclose(mesh_2d_again.points, points)

    def test_embed_raises_on_invalid_target(self):
        """Test that invalid target dimensions raise appropriate errors."""
        ### Create mesh
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### Target < 1 should fail
        with pytest.raises(ValueError, match="target_n_spatial_dims must be >= 1"):
            embed_in_spatial_dims(mesh, target_n_spatial_dims=0)

        with pytest.raises(ValueError, match="target_n_spatial_dims must be >= 1"):
            embed_in_spatial_dims(mesh, target_n_spatial_dims=-1)

    def test_embed_raises_when_target_less_than_manifold_dims(self):
        """Test that we can't embed manifold in lower-dimensional space."""
        ### Create 2D mesh
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert mesh.n_manifold_dims == 2

        ### Can't project 2D manifold to 1D space
        with pytest.raises(ValueError, match="Cannot embed.*dimensional manifold"):
            embed_in_spatial_dims(mesh, target_n_spatial_dims=1)

    def test_embed_round_trip_preserves_topology(self):
        """Test that embedding up and projecting down preserves topology."""
        ### Create triangle mesh
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        cell_data = TensorDict({"id": torch.tensor([123])}, batch_size=[1])
        mesh_original = Mesh(points=points, cells=cells, cell_data=cell_data)

        # Compute original area
        original_area = mesh_original.cell_areas[0].item()

        ### Embed to 5D and back
        mesh_5d = embed_in_spatial_dims(mesh_original, target_n_spatial_dims=5)
        mesh_back = embed_in_spatial_dims(mesh_5d, target_n_spatial_dims=3)

        ### Verify topology preserved
        assert torch.equal(mesh_back.cells, cells)
        assert mesh_back.cell_data["id"] == 123

        ### Verify points are same
        assert torch.allclose(mesh_back.points, points)

        ### Verify area is same (intrinsic property)
        assert torch.allclose(mesh_back.cell_areas[0], torch.tensor(original_area))

    @pytest.mark.parametrize(
        "start_dims,target_dims",
        [
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 4),
            (3, 5),
            (4, 5),
            (5, 4),
            (5, 3),
            (5, 2),
            (4, 3),
            (4, 2),
            (3, 2),
        ],
    )
    def test_embed_various_dimension_changes(self, start_dims, target_dims):
        """Test embedding across various dimension combinations."""
        ### Create simple edge in start_dims space
        points = torch.zeros((2, start_dims), dtype=torch.float32)
        points[1, 0] = 1.0  # Edge along first axis
        cells = torch.tensor([[0, 1]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert mesh.n_spatial_dims == start_dims
        assert mesh.n_manifold_dims == 1

        ### Embed/project to target
        result = embed_in_spatial_dims(mesh, target_n_spatial_dims=target_dims)

        ### Verify dimensions
        assert result.n_spatial_dims == target_dims
        assert result.n_manifold_dims == 1  # Unchanged
        assert result.n_points == 2
        assert result.n_cells == 1

        ### Verify edge length preserved (intrinsic)
        edge_length = result.cell_areas[0]
        assert torch.allclose(edge_length, torch.tensor(1.0))

    def test_embed_point_cloud(self):
        """Test embedding a 0D point cloud."""
        ### Create point cloud (0D manifold in 2D space)
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0], [1], [2]], dtype=torch.int64)  # 0-simplices
        mesh = Mesh(points=points, cells=cells)

        assert mesh.n_manifold_dims == 0
        assert mesh.n_spatial_dims == 2

        ### Embed in 4D
        embedded = embed_in_spatial_dims(mesh, target_n_spatial_dims=4)

        ### Verify
        assert embedded.n_manifold_dims == 0
        assert embedded.n_spatial_dims == 4
        assert embedded.n_points == 3
        assert embedded.points.shape == (3, 4)

        # Last two coordinates should be zero
        assert torch.allclose(embedded.points[:, 2:], torch.zeros(3, 2))

    def test_embed_preserves_cell_topology(self):
        """Test that cell connectivity is completely unchanged."""
        ### Create mesh with specific cell pattern
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 3], [1, 2, 3]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### Embed
        embedded = embed_in_spatial_dims(mesh, target_n_spatial_dims=5)

        ### Verify cells exactly the same (not just values, but same object)
        assert embedded.cells is mesh.cells
        assert torch.equal(embedded.cells, cells)
