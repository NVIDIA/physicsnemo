"""Tests for basic example meshes."""

import pytest

from physicsnemo.mesh import primitives


class TestBasicPrimitives:
    """Test all basic example meshes."""

    @pytest.mark.parametrize(
        "example_name,expected_manifold_dims,expected_spatial_dims",
        [
            # Points
            ("single_point_2d", 0, 2),
            ("single_point_3d", 0, 3),
            ("three_points_2d", 0, 2),
            ("three_points_3d", 0, 3),
            # Edges
            ("single_edge_2d", 1, 2),
            ("single_edge_3d", 1, 3),
            ("three_edges_2d", 1, 2),
            ("three_edges_3d", 1, 3),
            # Triangles
            ("single_triangle_2d", 2, 2),
            ("single_triangle_3d", 2, 3),
            ("two_triangles_2d", 2, 2),
            ("two_triangles_3d", 2, 3),
            # Tetrahedra
            ("single_tetrahedron", 3, 3),
            ("two_tetrahedra", 3, 3),
        ],
    )
    def test_basic_mesh(
        self, example_name, expected_manifold_dims, expected_spatial_dims
    ):
        """Test that basic mesh loads with correct dimensions."""
        primitives_module = getattr(primitives.basic, example_name)
        mesh = primitives_module.load()

        assert mesh.n_manifold_dims == expected_manifold_dims
        assert mesh.n_spatial_dims == expected_spatial_dims
        assert mesh.n_points > 0
        assert mesh.n_cells > 0
        assert mesh.points.device.type == "cpu"

    @pytest.mark.parametrize(
        "example_name",
        [
            "single_point_2d",
            "single_triangle_2d",
            "single_tetrahedron",
        ],
    )
    def test_device_transfer_cpu(self, example_name):
        """Test that meshes can be loaded on CPU."""
        primitives_module = getattr(primitives.basic, example_name)
        mesh_cpu = primitives_module.load(device="cpu")
        assert mesh_cpu.points.device.type == "cpu"

    @pytest.mark.cuda
    @pytest.mark.parametrize(
        "example_name",
        [
            "single_point_2d",
            "single_triangle_2d",
            "single_tetrahedron",
        ],
    )
    def test_device_transfer_cuda(self, example_name):
        """Test that meshes can be loaded on CUDA."""
        primitives_module = getattr(primitives.basic, example_name)
        mesh_gpu = primitives_module.load(device="cuda")
        assert mesh_gpu.points.device.type == "cuda"
