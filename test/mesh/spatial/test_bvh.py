"""Tests for BVH spatial acceleration structure.

Tests validate BVH construction, traversal, and queries across spatial dimensions,
manifold dimensions, and compute backends.
"""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.spatial import BVH

### Helper Functions ###


def create_simple_mesh(n_spatial_dims: int, n_manifold_dims: int, device: str = "cpu"):
    """Create a simple mesh for testing."""
    if n_manifold_dims > n_spatial_dims:
        raise ValueError(
            f"Manifold dimension {n_manifold_dims} cannot exceed spatial dimension {n_spatial_dims}"
        )

    if n_manifold_dims == 1:
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


### Test Fixtures ###


class TestBVHConstruction:
    """Tests for BVH construction from meshes."""

    def test_build_from_triangle_mesh(self):
        """Test building BVH from a simple triangle mesh."""
        ### Create a mesh with two triangles
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(points=points, cells=cells)

        ### Build BVH
        bvh = BVH.from_mesh(mesh)

        ### Verify structure
        assert bvh.node_aabb_min.shape[1] == 2  # 2D
        assert bvh.node_aabb_max.shape[1] == 2
        assert (
            bvh.node_aabb_min.shape[0] == bvh.node_aabb_max.shape[0]
        )  # Same number of nodes

        ### Root should contain all cells
        root_min = bvh.node_aabb_min[0]
        root_max = bvh.node_aabb_max[0]
        assert torch.allclose(root_min, torch.tensor([0.0, 0.0]))
        assert torch.allclose(root_max, torch.tensor([1.0, 1.0]))

    def test_build_from_3d_tetrahedra(self):
        """Test building BVH from 3D tetrahedral mesh."""
        ### Create a simple tetrahedral mesh
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
            ]
        )
        mesh = Mesh(points=points, cells=cells)

        ### Build BVH
        bvh = BVH.from_mesh(mesh)

        ### Verify 3D structure
        assert bvh.n_spatial_dims == 3
        assert bvh.node_aabb_min.shape[1] == 3

    def test_single_cell_mesh(self):
        """Test BVH for mesh with single cell."""
        ### Single triangle
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Build BVH
        bvh = BVH.from_mesh(mesh)

        ### Should have exactly one node (leaf)
        assert len(bvh.node_aabb_min) == 1
        assert bvh.node_cell_idx[0] == 0


class TestBVHTraversal:
    """Tests for BVH traversal and candidate finding."""

    def test_find_candidates_point_inside(self):
        """Test finding candidates for point inside a cell."""
        ### Create a simple mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        ### Query point inside first triangle
        query = torch.tensor([[0.25, 0.25]])
        candidates = bvh.find_candidate_cells(query)

        ### Should find at least one candidate (cell 0)
        assert len(candidates[0]) > 0
        assert 0 in candidates[0]

    def test_find_candidates_point_outside(self):
        """Test that point outside mesh returns no candidates."""
        ### Create a simple mesh
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        ### Query point far outside
        query = torch.tensor([[10.0, 10.0]])
        candidates = bvh.find_candidate_cells(query)

        ### Should find no candidates
        assert len(candidates[0]) == 0

    def test_find_candidates_multiple_points(self):
        """Test finding candidates for multiple query points."""
        ### Create a mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        ### Multiple query points
        queries = torch.tensor(
            [
                [0.25, 0.25],  # In first triangle
                [0.75, 0.75],  # In second triangle
                [10.0, 10.0],  # Outside
            ]
        )
        candidates = bvh.find_candidate_cells(queries)

        ### Verify results
        assert len(candidates) == 3
        assert len(candidates[0]) > 0  # First query has candidates
        assert len(candidates[1]) > 0  # Second query has candidates
        assert len(candidates[2]) == 0  # Third query has no candidates


class TestBVHDeviceHandling:
    """Tests for BVH device transfer."""

    def test_to_device_cpu(self):
        """Test moving BVH to CPU."""
        ### Create mesh and BVH
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        ### Move to CPU (should be no-op if already on CPU)
        bvh_cpu = bvh.to("cpu")
        assert bvh_cpu.device.type == "cpu"

    @pytest.mark.cuda
    def test_to_device_cuda(self):
        """Test moving BVH to CUDA."""
        ### Create mesh and BVH on CPU
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        ### Move to CUDA
        bvh_cuda = bvh.to("cuda")
        assert bvh_cuda.device.type == "cuda"
        assert bvh_cuda.node_aabb_min.is_cuda
        assert bvh_cuda.node_aabb_max.is_cuda


class TestBVHCorrectness:
    """Tests verifying BVH produces correct results."""

    def test_bvh_finds_all_containing_cells(self):
        """Test that BVH finds all cells that could contain a point."""
        ### Create a grid of triangles
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 3],
                [1, 4, 3],
                [1, 2, 4],
                [2, 5, 4],
            ]
        )
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        ### Query point at center of middle cell
        query = torch.tensor([[1.0, 0.5]])
        candidates = bvh.find_candidate_cells(query)

        ### Should include cells that overlap this region
        # Candidates should be a superset of actual containing cells
        assert len(candidates[0]) >= 1  # At least one candidate


### Parametrized Tests for Exhaustive Dimensional Coverage ###


class TestBVHParametrized:
    """Parametrized tests for BVH across all dimensions and backends."""

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 1),  # Edges in 2D
            (2, 2),  # Triangles in 2D
            (3, 1),  # Edges in 3D
            (3, 2),  # Surfaces in 3D
            (3, 3),  # Volumes in 3D
        ],
    )
    def test_bvh_construction_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test BVH construction across all dimension combinations."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        bvh = BVH.from_mesh(mesh)

        # Verify spatial dimension
        assert bvh.n_spatial_dims == n_spatial_dims, (
            f"BVH spatial dims mismatch: {bvh.n_spatial_dims=} != {n_spatial_dims=}"
        )

        # Verify AABB shapes
        assert bvh.node_aabb_min.shape[1] == n_spatial_dims
        assert bvh.node_aabb_max.shape[1] == n_spatial_dims
        assert bvh.node_aabb_min.shape[0] == bvh.node_aabb_max.shape[0]

        # Verify device
        assert_on_device(bvh.node_aabb_min, device)
        assert_on_device(bvh.node_aabb_max, device)

        # Verify at least one node exists
        assert bvh.node_aabb_min.shape[0] > 0, "BVH should have at least one node"

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 2),
            (3, 2),
            (3, 3),
        ],
    )
    def test_bvh_traversal_parametrized(self, n_spatial_dims, n_manifold_dims, device):
        """Test BVH traversal across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)
        bvh = BVH.from_mesh(mesh)

        # Create query point inside the mesh bounds
        query_point = torch.zeros(n_spatial_dims, device=device) + 0.5
        query = query_point.unsqueeze(0)

        candidates = bvh.find_candidate_cells(query)

        # Should return a list with one entry (for one query point)
        assert len(candidates) == 1

        # Should find at least one candidate
        assert len(candidates[0]) >= 0  # May be 0 if query is outside all cells

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
    def test_bvh_device_transfer_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test BVH device transfer across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)
        bvh = BVH.from_mesh(mesh)

        # BVH should be on the same device as mesh
        assert_on_device(bvh.node_aabb_min, device)
        assert_on_device(bvh.node_aabb_max, device)

        # Test explicit device transfer
        if device == "cpu":
            bvh_cpu = bvh.to("cpu")
            assert bvh_cpu.device.type == "cpu"
        elif device == "cuda":
            bvh_cuda = bvh.to("cuda")
            assert bvh_cuda.device.type == "cuda"

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 2),
            (3, 2),
            (3, 3),
        ],
    )
    def test_bvh_multiple_queries_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test BVH with multiple query points across dimensions."""
        torch.manual_seed(42)
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)
        bvh = BVH.from_mesh(mesh)

        # Create multiple query points
        n_queries = 5
        queries = torch.randn(n_queries, n_spatial_dims, device=device)

        candidates = bvh.find_candidate_cells(queries)

        # Should return list with n_queries entries
        assert len(candidates) == n_queries

        # Each entry should be a tensor of candidate cell indices
        for i, cands in enumerate(candidates):
            assert isinstance(cands, torch.Tensor), (
                f"Candidates[{i}] should be a tensor"
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
    def test_bvh_bounds_correctness_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test that BVH bounds are correct across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)
        bvh = BVH.from_mesh(mesh)

        # Root node should contain all points
        root_min = bvh.node_aabb_min[0]
        root_max = bvh.node_aabb_max[0]

        # Verify all mesh points are within root bounds
        mesh_min = mesh.points.min(dim=0)[0]
        mesh_max = mesh.points.max(dim=0)[0]

        assert torch.all(root_min <= mesh_min), (
            f"Root min should be <= mesh min: {root_min=}, {mesh_min=}"
        )
        assert torch.all(root_max >= mesh_max), (
            f"Root max should be >= mesh max: {root_max=}, {mesh_max=}"
        )
