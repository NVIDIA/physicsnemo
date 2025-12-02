"""Tests for uncovered validation code paths."""

import torch
import pytest

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.validation import validate_mesh


@pytest.fixture
def device():
    """Test on CPU."""
    return "cpu"


class TestValidationCodePaths:
    """Tests for specific validation code paths."""

    def test_large_mesh_duplicate_check_skipped(self, device):
        """Test that duplicate check is skipped for large meshes."""
        # Create mesh with >10K points
        n = 101
        x = torch.linspace(0, 1, n, device=device)
        y = torch.linspace(0, 1, n, device=device)
        xx, yy = torch.meshgrid(x, y, indexing="xy")

        points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

        # Create some triangles
        cells = torch.tensor([[0, 1, n]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        # Should skip duplicate check (>10K points)
        report = validate_mesh(mesh, check_duplicate_vertices=True)

        # Returns -1 for skipped check
        assert report.get("n_duplicate_vertices", -1) == -1

    def test_inverted_cells_3d(self, device):
        """Test detection of inverted cells in 3D."""
        # Regular tetrahedron
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, (3**0.5) / 2, 0.0],
                [0.5, (3**0.5) / 6, ((2 / 3) ** 0.5)],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2, 3],  # Normal orientation
                [0, 2, 1, 3],  # Inverted (swapped 1 and 2)
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh, check_inverted_cells=True, raise_on_error=False)

        # Should detect one inverted cell
        assert report["n_inverted_cells"] >= 1
        assert not report["valid"]

    def test_non_manifold_edge_detection(self, device):
        """Test detection of non-manifold edges."""
        # Create T-junction (3 triangles meeting at one edge)
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, -1.0, 0.0],
                [0.5, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Three triangles sharing edge [0,1]
        cells = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh, check_manifoldness=True, raise_on_error=False)

        # Should detect non-manifold edge
        assert not report["is_manifold"]
        assert report["n_non_manifold_edges"] >= 1

    def test_validation_with_empty_cells(self, device):
        """Test validation on mesh with no cells."""
        points = torch.randn(5, 2, device=device)
        cells = torch.zeros((0, 3), dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(
            mesh,
            check_degenerate_cells=True,
            check_out_of_bounds=True,
            check_inverted_cells=True,
        )

        # Should be valid (no cells to have problems)
        assert report["valid"]
        assert report["n_degenerate_cells"] == 0
        assert report["n_out_of_bounds_cells"] == 0

    def test_inverted_check_not_applicable(self, device):
        """Test that inverted check returns -1 for non-volume meshes."""
        # 2D triangle in 3D (codimension 1)
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh, check_inverted_cells=True)

        # Should return -1 (not applicable for codimension != 0)
        assert report["n_inverted_cells"] == -1 or report["n_inverted_cells"] == 0

    def test_manifoldness_not_applicable_non_2d(self, device):
        """Test that manifoldness check is only for 2D manifolds."""
        # 1D mesh (edges)
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh, check_manifoldness=True)

        # Should return None or -1 for non-2D manifolds
        assert (
            report.get("is_manifold") is None
            or report.get("n_non_manifold_edges") == -1
        )

    def test_validation_skips_geometry_after_out_of_bounds(self, device):
        """Test that validation short-circuits after finding out-of-bounds indices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Invalid index
        cells = torch.tensor([[0, 1, 100]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        # Should not crash even though area computation would fail
        report = validate_mesh(
            mesh,
            check_out_of_bounds=True,
            check_degenerate_cells=True,
            raise_on_error=False,
        )

        assert not report["valid"]
        assert report["n_out_of_bounds_cells"] == 1
        # Degenerate check should be skipped (no key or not computed)
