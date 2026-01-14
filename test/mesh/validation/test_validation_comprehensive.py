"""Comprehensive tests for validation module."""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.validation import (
    compute_mesh_statistics,
    compute_quality_metrics,
    validate_mesh,
)


@pytest.fixture
def device():
    """Test on CPU."""
    return "cpu"


class TestMeshValidation:
    """Tests for mesh validation."""

    def test_valid_mesh(self, device):
        """Test that valid mesh passes all checks."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh)

        assert report["valid"]
        assert report["n_degenerate_cells"] == 0
        assert report["n_out_of_bounds_cells"] == 0

    def test_out_of_bounds_indices(self, device):
        """Test detection of out-of-bounds cell indices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Cell references non-existent vertex 10
        cells = torch.tensor([[0, 1, 10]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh, check_out_of_bounds=True, raise_on_error=False)

        assert not report["valid"]
        assert report["n_out_of_bounds_cells"] == 1

    def test_degenerate_cells_detection(self, device):
        """Test detection of degenerate cells."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [2.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Second cell has duplicate vertex (degenerate)
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 1],  # Duplicate vertex 1
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh, check_degenerate_cells=True, raise_on_error=False)

        assert not report["valid"]
        assert report["n_degenerate_cells"] >= 1

    def test_duplicate_vertices_detection(self, device):
        """Test detection of duplicate vertices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.0, 0.0],  # Exact duplicate of vertex 0
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(
            mesh, check_duplicate_vertices=True, raise_on_error=False
        )

        assert not report["valid"]
        assert report["n_duplicate_vertices"] >= 1

    def test_raise_on_error(self, device):
        """Test that raise_on_error triggers exception."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 10]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        with pytest.raises(ValueError, match="out-of-bounds"):
            validate_mesh(mesh, check_out_of_bounds=True, raise_on_error=True)

    def test_manifoldness_check_2d(self, device):
        """Test manifoldness check for 2D meshes."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Two triangles sharing edge [0,1]
        cells = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 3],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh, check_manifoldness=True)

        # Should be manifold (each edge shared by at most 2 faces)
        assert report["is_manifold"]
        assert report["n_non_manifold_edges"] == 0

    def test_empty_mesh_validation(self, device):
        """Test validation of empty mesh."""
        points = torch.zeros((0, 2), device=device)
        cells = torch.zeros((0, 3), dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh)

        # Empty mesh should be valid
        assert report["valid"]


class TestQualityMetrics:
    """Tests for quality metrics computation."""

    def test_equilateral_triangle_quality(self, device):
        """Test that equilateral triangle has high quality score."""

        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, (3**0.5) / 2],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        metrics = compute_quality_metrics(mesh)

        assert "quality_score" in metrics.keys()
        assert "aspect_ratio" in metrics.keys()
        assert "edge_length_ratio" in metrics.keys()

        # Equilateral triangle should have high quality
        quality = metrics["quality_score"][0]
        assert quality > 0.7  # High quality (formula gives ~0.75 for equilateral)

        # Edge length ratio should be close to 1.0
        edge_ratio = metrics["edge_length_ratio"][0]
        assert edge_ratio < 1.1  # Nearly equal edges

    def test_degenerate_triangle_quality(self, device):
        """Test that degenerate triangle has low quality score."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [10.0, 0.0],  # Nearly collinear
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        metrics = compute_quality_metrics(mesh)

        quality = metrics["quality_score"][0]

        # Very elongated triangle should have low quality
        assert quality < 0.3

    def test_quality_metrics_angles(self, device):
        """Test that angles are computed for triangles."""

        # Right triangle
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        metrics = compute_quality_metrics(mesh)

        assert "min_angle" in metrics.keys()
        assert "max_angle" in metrics.keys()

        min_angle = metrics["min_angle"][0]
        max_angle = metrics["max_angle"][0]

        # Right triangle has angles: π/4, π/4, π/2
        assert min_angle > 0
        assert max_angle <= torch.pi

        # Max angle should be close to π/2
        assert torch.abs(max_angle - torch.pi / 2) < 0.1

    def test_empty_mesh_quality(self, device):
        """Test quality metrics on empty mesh."""
        points = torch.zeros((5, 2), device=device)
        cells = torch.zeros((0, 3), dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        metrics = compute_quality_metrics(mesh)

        # Should return empty TensorDict
        assert len(metrics) == 0 or metrics.shape[0] == 0


class TestMeshStatistics:
    """Tests for mesh statistics computation."""

    def test_basic_statistics(self, device):
        """Test basic mesh statistics."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [1.5, 0.5],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        stats = compute_mesh_statistics(mesh)

        assert stats["n_points"] == 4
        assert stats["n_cells"] == 2
        assert stats["n_manifold_dims"] == 2
        assert stats["n_spatial_dims"] == 2
        assert stats["n_degenerate_cells"] == 0
        assert stats["n_isolated_vertices"] == 0

    def test_statistics_with_isolated(self, device):
        """Test statistics with isolated vertices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [5.0, 5.0],  # Isolated
                [6.0, 6.0],  # Isolated
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        stats = compute_mesh_statistics(mesh)

        assert stats["n_isolated_vertices"] == 2

    def test_statistics_edge_lengths(self, device):
        """Test edge length statistics."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        stats = compute_mesh_statistics(mesh)

        assert "edge_length_stats" in stats
        min_len, mean_len, max_len, std_len = stats["edge_length_stats"]

        # All should be positive
        assert min_len > 0
        assert mean_len > 0
        assert max_len > 0

    def test_statistics_empty_mesh(self, device):
        """Test statistics on empty mesh."""
        points = torch.zeros((5, 2), device=device)
        cells = torch.zeros((0, 3), dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        stats = compute_mesh_statistics(mesh)

        assert stats["n_cells"] == 0
        assert stats["n_isolated_vertices"] == 5


class TestMeshAPIIntegration:
    """Test that Mesh class methods work correctly."""

    def test_mesh_validate_method(self, device):
        """Test mesh.validate() convenience method."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        report = mesh.validate()

        assert isinstance(report, dict)
        assert "valid" in report
        assert report["valid"]

    def test_mesh_quality_metrics_property(self, device):
        """Test mesh.quality_metrics property."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        metrics = mesh.quality_metrics

        assert "quality_score" in metrics.keys()
        assert metrics["quality_score"].shape == (1,)

    def test_mesh_statistics_property(self, device):
        """Test mesh.statistics property."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        stats = mesh.statistics

        assert isinstance(stats, dict)
        assert stats["n_points"] == 3
        assert stats["n_cells"] == 1

    def test_validation_with_all_checks(self, device):
        """Test validation with all checks enabled."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        report = mesh.validate(
            check_degenerate_cells=True,
            check_duplicate_vertices=True,
            check_out_of_bounds=True,
            check_manifoldness=True,
        )

        assert report["valid"]

    def test_validation_detects_negative_indices(self, device):
        """Test that negative cell indices are caught."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, -1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh, check_out_of_bounds=True, raise_on_error=False)

        assert not report["valid"]
        assert report["n_out_of_bounds_cells"] == 1


class TestQualityMetricsEdgeCases:
    """Edge case tests for quality metrics."""

    def test_single_cell_quality(self, device):
        """Test quality metrics on single cell."""

        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, (3**0.5) / 2],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        metrics = compute_quality_metrics(mesh)

        assert metrics.shape[0] == 1
        assert not torch.isnan(metrics["quality_score"][0])

    def test_multiple_cells_quality(self, device):
        """Test quality metrics on multiple cells."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [1.5, 0.5],
                [0.5, -0.5],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
                [0, 1, 4],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        metrics = compute_quality_metrics(mesh)

        assert metrics.shape[0] == 3
        assert torch.all(metrics["quality_score"] > 0)
        assert torch.all(metrics["quality_score"] <= 1.0)

    def test_3d_mesh_quality(self, device):
        """Test quality metrics on 3D tetrahedral mesh."""

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

        cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        metrics = compute_quality_metrics(mesh)

        # Should compute metrics even for tets (angles will be NaN)
        assert metrics.shape[0] == 1
        assert not torch.isnan(metrics["quality_score"][0])
        assert torch.isnan(metrics["min_angle"][0])  # Not defined for tets yet


class TestStatisticsVariations:
    """Test statistics computation with various mesh configurations."""

    def test_statistics_include_quality(self, device):
        """Test that statistics include quality metrics."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        stats = compute_mesh_statistics(mesh)

        assert "cell_area_stats" in stats
        assert "quality_score_stats" in stats
        assert "aspect_ratio_stats" in stats

    def test_statistics_large_mesh(self, device):
        """Test statistics on larger mesh."""
        # Create structured grid
        n = 10
        x = torch.linspace(0, 1, n, device=device)
        y = torch.linspace(0, 1, n, device=device)
        xx, yy = torch.meshgrid(x, y, indexing="xy")

        points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

        # Create triangles
        cells_list = []
        for i in range(n - 1):
            for j in range(n - 1):
                idx = i * n + j
                cells_list.append([idx, idx + 1, idx + n])
                cells_list.append([idx + 1, idx + n + 1, idx + n])

        cells = torch.tensor(cells_list, dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        stats = compute_mesh_statistics(mesh)

        assert stats["n_cells"] == 2 * (n - 1) * (n - 1)
        assert stats["n_isolated_vertices"] == 0
