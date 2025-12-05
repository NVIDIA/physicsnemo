"""Tests for PyVista dataset example meshes."""

import pytest

from physicsnemo.mesh import primitives


class TestPyVistaDatasetPrimitives:
    """Test all PyVista dataset wrappers."""

    @pytest.mark.parametrize(
        "example_name,expected_manifold_dims",
        [
            # Surface meshes (2D→3D)
            ("airplane", 2),
            ("bunny", 2),
            ("ant", 2),
            ("cow", 2),
            ("globe", 2),
            # Volume meshes (3D→3D)
            ("tetbeam", 3),
            ("hexbeam", 3),
        ],
    )
    def test_pyvista_dataset(self, example_name, expected_manifold_dims):
        """Test that PyVista dataset loads with correct dimensions."""
        primitives_module = getattr(primitives.pyvista_datasets, example_name)
        mesh = primitives_module.load()

        assert mesh.n_manifold_dims == expected_manifold_dims
        assert mesh.n_spatial_dims == 3
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_bunny_is_large(self):
        """Test that bunny has reasonable number of vertices."""
        bunny = primitives.pyvista_datasets.bunny.load()

        # Stanford bunny should have a good number of vertices
        assert bunny.n_points > 1000

    def test_tetbeam_is_tetrahedral(self):
        """Test that tetbeam contains tetrahedra."""
        tetbeam = primitives.pyvista_datasets.tetbeam.load()

        # Should be 3D volume mesh
        assert tetbeam.n_manifold_dims == 3
        # Each cell should have 4 vertices (tetrahedron)
        assert tetbeam.cells.shape[1] == 4

    @pytest.mark.parametrize("example_name", ["airplane", "bunny"])
    def test_device_transfer_cpu(self, example_name):
        """Test that PyVista datasets can be loaded on CPU."""
        primitives_module = getattr(primitives.pyvista_datasets, example_name)
        mesh_cpu = primitives_module.load(device="cpu")
        assert mesh_cpu.points.device.type == "cpu"

    @pytest.mark.cuda
    @pytest.mark.parametrize("example_name", ["airplane", "bunny"])
    def test_device_transfer_cuda(self, example_name):
        """Test that PyVista datasets can be loaded on CUDA."""
        primitives_module = getattr(primitives.pyvista_datasets, example_name)
        mesh_gpu = primitives_module.load(device="cuda")
        assert mesh_gpu.points.device.type == "cuda"
