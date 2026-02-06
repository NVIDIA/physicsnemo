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

"""Tests for BVH spatial acceleration structure.

Tests validate BVH construction, traversal, and queries across spatial dimensions,
manifold dimensions, and compute backends.
"""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.spatial import BVH
from physicsnemo.mesh.spatial.bvh import _compute_morton_codes

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


### Morton Code Tests ###


class TestMortonCodes:
    """Tests for morton code computation."""

    def test_2d_codes_are_non_negative(self):
        """Morton codes must be non-negative for correct sorting."""
        points = torch.rand(1000, 2)
        codes = _compute_morton_codes(points)
        assert (codes >= 0).all()

    def test_3d_codes_are_non_negative(self):
        """Morton codes must be non-negative for correct sorting."""
        points = torch.rand(1000, 3)
        codes = _compute_morton_codes(points)
        assert (codes >= 0).all()

    def test_spatial_locality_2d(self):
        """Nearby 2D points should have nearby morton codes."""
        # Two clusters: one near origin, one far away
        cluster_a = torch.rand(50, 2) * 0.01
        cluster_b = torch.rand(50, 2) * 0.01 + 10.0
        points = torch.cat([cluster_a, cluster_b])

        codes = _compute_morton_codes(points)
        sorted_idx = codes.argsort()

        # After sorting, cluster_a and cluster_b should each be contiguous
        # (not interleaved). Check that the first 50 sorted indices are all
        # from the same cluster.
        first_half = sorted_idx[:50]
        first_from_a = (first_half < 50).sum()
        first_from_b = (first_half >= 50).sum()

        # One of the halves should be purely from one cluster
        assert first_from_a == 50 or first_from_b == 50

    def test_spatial_locality_3d(self):
        """Nearby 3D points should have nearby morton codes."""
        cluster_a = torch.rand(50, 3) * 0.01
        cluster_b = torch.rand(50, 3) * 0.01 + 10.0
        points = torch.cat([cluster_a, cluster_b])

        codes = _compute_morton_codes(points)
        sorted_idx = codes.argsort()

        first_half = sorted_idx[:50]
        first_from_a = (first_half < 50).sum()
        first_from_b = (first_half >= 50).sum()
        assert first_from_a == 50 or first_from_b == 50

    def test_generic_fallback_4d(self):
        """The generic bit-loop path works for D > 3."""
        points = torch.rand(100, 4)
        codes = _compute_morton_codes(points)
        assert codes.shape == (100,)
        assert (codes >= 0).all()

    def test_single_point(self):
        """Morton codes work for a single point."""
        codes = _compute_morton_codes(torch.tensor([[1.0, 2.0, 3.0]]))
        assert codes.shape == (1,)
        # Single point: all dimensions map to the same value, code = 0 or max
        # Since min == max, extent is clamped, coordinates map to max_val
        assert codes[0] >= 0

    def test_identical_points(self):
        """All identical points produce the same code."""
        points = torch.ones(10, 3) * 5.0
        codes = _compute_morton_codes(points)
        assert (codes == codes[0]).all()

    def test_rejects_integer_input(self):
        """Integer centroids should be rejected (would silently corrupt quantization)."""
        with pytest.raises(TypeError, match="floating-point"):
            _compute_morton_codes(torch.randint(0, 100, (10, 3)))

    def test_rejects_1d_input(self):
        """1D input should be rejected."""
        with pytest.raises(ValueError, match="2D"):
            _compute_morton_codes(torch.rand(10))


### Construction Tests ###


class TestBVHConstruction:
    """Tests for BVH construction from meshes."""

    def test_build_from_triangle_mesh(self):
        """Test building BVH from a simple triangle mesh."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        mesh = Mesh(points=points, cells=cells)

        bvh = BVH.from_mesh(mesh)

        assert bvh.node_aabb_min.shape[1] == 2
        assert bvh.node_aabb_max.shape[1] == 2
        assert bvh.node_aabb_min.shape[0] == bvh.node_aabb_max.shape[0]

        ### Root should contain all cells
        root_min = bvh.node_aabb_min[0]
        root_max = bvh.node_aabb_max[0]
        assert torch.allclose(root_min, torch.tensor([0.0, 0.0]))
        assert torch.allclose(root_max, torch.tensor([1.0, 1.0]))

    def test_build_from_3d_tetrahedra(self):
        """Test building BVH from 3D tetrahedral mesh."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        mesh = Mesh(points=points, cells=cells)

        bvh = BVH.from_mesh(mesh)

        assert bvh.n_spatial_dims == 3
        assert bvh.node_aabb_min.shape[1] == 3

    def test_single_cell_mesh(self):
        """Test BVH for mesh with single cell."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        bvh = BVH.from_mesh(mesh)

        ### Should have exactly one node (leaf)
        assert bvh.n_nodes == 1
        assert bvh.leaf_count[0] == 1
        assert bvh.sorted_cell_order[bvh.leaf_start[0]] == 0

    def test_leaf_size_parameter(self):
        """Test that leaf_size controls the number of cells per leaf."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [3.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [[0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5], [2, 3, 6], [3, 7, 6]]
        )
        mesh = Mesh(points=points, cells=cells)

        ### With leaf_size=1, each leaf has exactly 1 cell
        bvh_1 = BVH.from_mesh(mesh, leaf_size=1)
        leaf_mask_1 = bvh_1.leaf_count > 0
        assert (bvh_1.leaf_count[leaf_mask_1] == 1).all()

        ### With large leaf_size, fewer nodes are needed
        bvh_big = BVH.from_mesh(mesh, leaf_size=100)
        assert bvh_big.n_nodes == 1  # single leaf for 6 cells with leaf_size=100
        assert bvh_big.leaf_count[0] == 6

    def test_empty_mesh(self):
        """Test BVH for an empty mesh."""
        points = torch.zeros((0, 2))
        cells = torch.zeros((0, 3), dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        bvh = BVH.from_mesh(mesh)

        assert bvh.n_nodes == 0
        assert len(bvh.sorted_cell_order) == 0

    def test_leaf_size_validation(self):
        """leaf_size < 1 should raise ValueError."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        with pytest.raises(ValueError, match="leaf_size"):
            BVH.from_mesh(mesh, leaf_size=0)

    def test_all_cells_represented_in_sorted_order(self):
        """sorted_cell_order must be a permutation of [0, n_cells)."""
        points = torch.rand(20, 3)
        cells = torch.tensor(
            [[i, (i + 1) % 20, (i + 2) % 20, (i + 3) % 20] for i in range(10)]
        )
        mesh = Mesh(points=points, cells=cells)

        bvh = BVH.from_mesh(mesh)

        assert len(bvh.sorted_cell_order) == 10
        assert set(bvh.sorted_cell_order.tolist()) == set(range(10))

    def test_node_count_bound(self):
        """Node count should not exceed the theoretical upper bound."""
        points = torch.rand(50, 2)
        cells = torch.tensor([[i, (i + 1) % 50, (i + 2) % 50] for i in range(30)])
        mesh = Mesh(points=points, cells=cells)

        for leaf_size in [1, 4, 8, 16]:
            bvh = BVH.from_mesh(mesh, leaf_size=leaf_size)
            min_cells_per_leaf = max(1, (leaf_size + 1) // 2)
            max_leaves = (30 + min_cells_per_leaf - 1) // min_cells_per_leaf
            max_nodes = max(1, 2 * max_leaves - 1)
            assert bvh.n_nodes <= max_nodes, (
                f"Node count {bvh.n_nodes} exceeds bound {max_nodes} for {leaf_size=}"
            )


### Traversal Tests ###


class TestBVHTraversal:
    """Tests for BVH traversal and candidate finding."""

    def test_find_candidates_point_inside(self):
        """Test finding candidates for point inside a cell."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        query = torch.tensor([[0.25, 0.25]])
        candidates = bvh.find_candidate_cells(query)

        candidates_list = candidates.to_list()
        assert len(candidates_list[0]) > 0
        assert 0 in candidates_list[0]

    def test_find_candidates_point_outside(self):
        """Test that point outside mesh returns no candidates."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        query = torch.tensor([[10.0, 10.0]])
        candidates = bvh.find_candidate_cells(query)

        candidates_list = candidates.to_list()
        assert len(candidates_list[0]) == 0

    def test_find_candidates_multiple_points(self):
        """Test finding candidates for multiple query points."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        queries = torch.tensor([[0.25, 0.25], [0.75, 0.75], [10.0, 10.0]])
        candidates = bvh.find_candidate_cells(queries)

        candidates_list = candidates.to_list()
        assert len(candidates_list) == 3
        assert len(candidates_list[0]) > 0
        assert len(candidates_list[1]) > 0
        assert len(candidates_list[2]) == 0

    def test_find_candidates_empty_bvh(self):
        """Querying an empty BVH returns empty adjacency."""
        points = torch.zeros((0, 2))
        cells = torch.zeros((0, 3), dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        query = torch.tensor([[0.5, 0.5]])
        candidates = bvh.find_candidate_cells(query)

        assert candidates.n_sources == 1
        assert candidates.n_total_neighbors == 0

    def test_find_candidates_empty_query(self):
        """Querying with zero points returns empty adjacency."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        query = torch.zeros((0, 2))
        candidates = bvh.find_candidate_cells(query)

        assert candidates.n_sources == 0

    def test_rejects_integer_query_points(self):
        """Integer query points should be rejected."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        with pytest.raises(TypeError, match="floating-point"):
            bvh.find_candidate_cells(torch.tensor([[0, 0]]))

    def test_rejects_wrong_spatial_dims(self):
        """Query points with wrong number of spatial dims should be rejected."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        with pytest.raises(ValueError, match="spatial dims"):
            bvh.find_candidate_cells(torch.tensor([[0.5, 0.5, 0.5]]))

    def test_rejects_1d_query_points(self):
        """1D query points should be rejected."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        with pytest.raises(ValueError, match="2D"):
            bvh.find_candidate_cells(torch.tensor([0.5, 0.5]))

    def test_leaf_size_does_not_miss_candidates(self):
        """Different leaf sizes should find the same set of candidates.

        Larger leaf sizes produce more false-positive candidates (cells whose
        AABB doesn't contain the query) but must never miss a true candidate.
        """
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
        cells = torch.tensor([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
        mesh = Mesh(points=points, cells=cells)

        query = torch.tensor([[0.5, 0.5], [1.5, 0.5]])

        bvh_1 = BVH.from_mesh(mesh, leaf_size=1)
        bvh_8 = BVH.from_mesh(mesh, leaf_size=8)

        cands_1 = bvh_1.find_candidate_cells(query, max_candidates_per_point=None)
        cands_8 = bvh_8.find_candidate_cells(query, max_candidates_per_point=None)

        # leaf_size=8 candidates should be a superset of leaf_size=1 candidates
        for i in range(query.shape[0]):
            set_1 = set(cands_1.to_list()[i])
            set_8 = set(cands_8.to_list()[i])
            assert set_1.issubset(set_8), (
                f"Query {i}: leaf_size=1 found {set_1}, leaf_size=8 found {set_8}. "
                f"Missing: {set_1 - set_8}"
            )


### Device Handling Tests ###


class TestBVHDeviceHandling:
    """Tests for BVH device transfer."""

    def test_to_device_cpu(self):
        """Test moving BVH to CPU."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        bvh_cpu = bvh.to("cpu")
        assert bvh_cpu.device.type == "cpu"

    @pytest.mark.cuda
    def test_to_device_cuda(self):
        """Test moving BVH to CUDA."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        bvh_cuda = bvh.to("cuda")
        assert bvh_cuda.device.type == "cuda"
        assert bvh_cuda.node_aabb_min.is_cuda
        assert bvh_cuda.node_aabb_max.is_cuda


### Correctness Tests ###


class TestBVHCorrectness:
    """Tests verifying BVH produces correct results."""

    def test_bvh_finds_all_containing_cells(self):
        """Test that BVH finds all cells that could contain a point."""
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
        cells = torch.tensor([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        query = torch.tensor([[1.0, 0.5]])
        candidates = bvh.find_candidate_cells(query)

        candidates_list = candidates.to_list()
        assert len(candidates_list[0]) >= 1


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

        assert bvh.n_spatial_dims == n_spatial_dims, (
            f"BVH spatial dims mismatch: {bvh.n_spatial_dims=} != {n_spatial_dims=}"
        )
        assert bvh.node_aabb_min.shape[1] == n_spatial_dims
        assert bvh.node_aabb_max.shape[1] == n_spatial_dims
        assert bvh.node_aabb_min.shape[0] == bvh.node_aabb_max.shape[0]
        assert_on_device(bvh.node_aabb_min, device)
        assert_on_device(bvh.node_aabb_max, device)
        assert bvh.n_nodes > 0

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

        query_point = torch.zeros(n_spatial_dims, device=device) + 0.5
        query = query_point.unsqueeze(0)

        candidates = bvh.find_candidate_cells(query)

        assert candidates.n_sources == 1
        assert candidates.n_total_neighbors >= 0

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

        assert_on_device(bvh.node_aabb_min, device)
        assert_on_device(bvh.node_aabb_max, device)

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

        n_queries = 5
        queries = torch.randn(n_queries, n_spatial_dims, device=device)

        candidates = bvh.find_candidate_cells(queries)

        assert candidates.n_sources == n_queries
        assert isinstance(candidates.indices, torch.Tensor)

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

        root_min = bvh.node_aabb_min[0]
        root_max = bvh.node_aabb_max[0]

        mesh_min = mesh.points.min(dim=0)[0]
        mesh_max = mesh.points.max(dim=0)[0]

        assert torch.all(root_min <= mesh_min), (
            f"Root min should be <= mesh min: {root_min=}, {mesh_min=}"
        )
        assert torch.all(root_max >= mesh_max), (
            f"Root max should be >= mesh max: {root_max=}, {mesh_max=}"
        )

    @pytest.mark.parametrize("leaf_size", [1, 4, 8, 16])
    def test_bvh_leaf_size_parametrized(self, leaf_size, device):
        """Test BVH construction with various leaf sizes."""
        mesh = create_simple_mesh(n_spatial_dims=3, n_manifold_dims=3, device=device)
        bvh = BVH.from_mesh(mesh, leaf_size=leaf_size)

        assert bvh.n_nodes > 0

        # All leaf counts should be <= leaf_size
        leaf_mask = bvh.leaf_count > 0
        if leaf_mask.any():
            assert (bvh.leaf_count[leaf_mask] <= leaf_size).all()

        # Query should still work
        query = torch.tensor([[0.3, 0.3, 0.3]], device=device)
        candidates = bvh.find_candidate_cells(query)
        assert candidates.n_sources == 1
