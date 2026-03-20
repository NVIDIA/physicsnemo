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

"""Tests for partition_cells (discrete restricted Voronoi partition).

Validates assignment correctness, area/normal/centroid accumulation, edge
cases (single seed, empty clusters, non-surface meshes), input validation
(device/dtype mismatch), and verifies that the kNN-accelerated path produces
identical results to brute-force nearest neighbor across devices.
"""

import pytest
import torch
import torch.nn.functional as F

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import plane
from physicsnemo.mesh.remeshing import CellPartition, partition_cells

### Fixtures ###


@pytest.fixture()
def two_triangles_3d() -> Mesh:
    """Two coplanar triangles in 3D forming a unit square in the z=0 plane.

    Triangle 0: (0,0,0)-(1,0,0)-(0,1,0)  centroid=(1/3, 1/3, 0)  area=0.5
    Triangle 1: (1,0,0)-(1,1,0)-(0,1,0)  centroid=(2/3, 2/3, 0)  area=0.5
    """
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
    return Mesh(points=points, cells=cells)


@pytest.fixture()
def grid_mesh_3d() -> Mesh:
    """4x4 regular triangle grid on the z=0 plane (32 triangles)."""
    m = plane.load(subdivisions=4)
    return Mesh(points=m.points.double(), cells=m.cells)


def _brute_force_assignments(
    cell_centroids: torch.Tensor, seeds: torch.Tensor
) -> torch.Tensor:
    """Reference nearest-neighbor via full distance matrix (for correctness checks)."""
    return torch.cdist(cell_centroids, seeds).argmin(dim=1)


### Return type ###


class TestReturnType:
    """Verify partition_cells returns a CellPartition that supports structured unpacking."""

    def test_returns_cell_partition(self, two_triangles_3d: Mesh):
        seeds = torch.tensor([[0.3, 0.3, 0.0], [0.7, 0.7, 0.0]], dtype=torch.float64)
        result = partition_cells(two_triangles_3d, seeds)
        assert isinstance(result, CellPartition)

    def test_unpacking(self, two_triangles_3d: Mesh):
        seeds = torch.tensor([[0.3, 0.3, 0.0], [0.7, 0.7, 0.0]], dtype=torch.float64)
        assignments, areas, normals, centroids = partition_cells(
            two_triangles_3d, seeds
        )
        assert assignments.shape == (2,)
        assert areas.shape == (2,)
        assert normals.shape == (2, 3)
        assert centroids.shape == (2, 3)


### Assignment correctness ###


class TestAssignments:
    """Verify each cell is assigned to its nearest seed by centroid distance."""

    def test_two_triangles_two_seeds(self, two_triangles_3d: Mesh):
        """Each triangle should be assigned to its nearest seed."""
        seeds = torch.tensor(
            [[0.25, 0.25, 0.0], [0.75, 0.75, 0.0]], dtype=torch.float64
        )
        result = partition_cells(two_triangles_3d, seeds)
        assert result.assignments.tolist() == [0, 1]

    def test_matches_brute_force(self, grid_mesh_3d: Mesh):
        """BVH assignments must match brute-force on a non-trivial mesh."""
        torch.manual_seed(42)
        seeds = torch.rand(8, 3, dtype=torch.float64)
        seeds[:, 2] = 0.0  # keep seeds on z=0 plane

        result = partition_cells(grid_mesh_3d, seeds)
        expected = _brute_force_assignments(grid_mesh_3d.cell_centroids, seeds)
        assert torch.equal(result.assignments, expected)

    def test_single_seed_gets_all_cells(self, grid_mesh_3d: Mesh):
        """With one seed, every cell should be assigned to it."""
        seeds = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64)
        result = partition_cells(grid_mesh_3d, seeds)
        assert (result.assignments == 0).all()


### Area conservation ###


class TestAreaConservation:
    """Verify that the sum of cluster areas equals total mesh surface area."""

    def test_total_area_preserved(self, two_triangles_3d: Mesh):
        seeds = torch.tensor(
            [[0.25, 0.25, 0.0], [0.75, 0.75, 0.0]], dtype=torch.float64
        )
        result = partition_cells(two_triangles_3d, seeds)
        assert result.cluster_areas.sum().item() == pytest.approx(
            two_triangles_3d.cell_areas.sum().item()
        )

    def test_total_area_preserved_many_seeds(self, grid_mesh_3d: Mesh):
        torch.manual_seed(7)
        seeds = torch.rand(10, 3, dtype=torch.float64)
        seeds[:, 2] = 0.0
        result = partition_cells(grid_mesh_3d, seeds)
        assert result.cluster_areas.sum().item() == pytest.approx(
            grid_mesh_3d.cell_areas.sum().item()
        )

    def test_single_seed_gets_total_area(self, grid_mesh_3d: Mesh):
        seeds = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64)
        result = partition_cells(grid_mesh_3d, seeds)
        assert result.cluster_areas[0].item() == pytest.approx(
            grid_mesh_3d.cell_areas.sum().item()
        )

    def test_one_seed_per_cell_recovers_raw_areas(self, two_triangles_3d: Mesh):
        """When each cell gets its own seed, cluster areas = raw cell areas."""
        seeds = two_triangles_3d.cell_centroids.clone()
        result = partition_cells(two_triangles_3d, seeds)
        torch.testing.assert_close(
            result.cluster_areas,
            two_triangles_3d.cell_areas,
        )


### Normal accumulation ###


class TestNormals:
    """Verify area-weighted normal accumulation and unit-normalization per cluster."""

    def test_coplanar_normals_are_z_axis(self, two_triangles_3d: Mesh):
        """For coplanar triangles in z=0, all cluster normals point along z."""
        seeds = torch.tensor(
            [[0.25, 0.25, 0.0], [0.75, 0.75, 0.0]], dtype=torch.float64
        )
        result = partition_cells(two_triangles_3d, seeds)
        for i in range(2):
            # Normal should be (0, 0, ±1)
            assert abs(result.cluster_normals[i, 2].abs().item() - 1.0) < 1e-10
            assert abs(result.cluster_normals[i, 0].item()) < 1e-10
            assert abs(result.cluster_normals[i, 1].item()) < 1e-10

    def test_normals_are_unit_vectors(self, grid_mesh_3d: Mesh):
        torch.manual_seed(3)
        seeds = torch.rand(5, 3, dtype=torch.float64)
        seeds[:, 2] = 0.0
        result = partition_cells(grid_mesh_3d, seeds)
        norms = result.cluster_normals.norm(dim=1)
        # Non-empty clusters should have unit normals
        nonempty = result.cluster_areas > 0
        torch.testing.assert_close(norms[nonempty], torch.ones_like(norms[nonempty]))

    def test_manual_area_weighted_normal(self, two_triangles_3d: Mesh):
        """Verify normal accumulation matches manual computation."""
        # Single seed collects both triangles
        seeds = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64)
        result = partition_cells(two_triangles_3d, seeds)

        cell_normals = two_triangles_3d.cell_normals
        cell_areas = two_triangles_3d.cell_areas
        expected = F.normalize(
            (cell_normals * cell_areas.unsqueeze(-1)).sum(dim=0, keepdim=True),
            dim=-1,
        )
        torch.testing.assert_close(result.cluster_normals, expected)


### Centroid accumulation ###


class TestCentroids:
    """Verify area-weighted centroid computation per cluster."""

    def test_single_seed_centroid_is_area_weighted_mean(self, two_triangles_3d: Mesh):
        seeds = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64)
        result = partition_cells(two_triangles_3d, seeds)

        cell_centroids = two_triangles_3d.cell_centroids
        cell_areas = two_triangles_3d.cell_areas
        expected = (cell_centroids * cell_areas.unsqueeze(-1)).sum(0) / cell_areas.sum()
        torch.testing.assert_close(result.cluster_centroids[0], expected)

    def test_one_seed_per_cell_centroids_match(self, two_triangles_3d: Mesh):
        """When each cell gets its own seed, cluster centroids = cell centroids."""
        seeds = two_triangles_3d.cell_centroids.clone()
        result = partition_cells(two_triangles_3d, seeds)
        torch.testing.assert_close(
            result.cluster_centroids,
            two_triangles_3d.cell_centroids,
        )


### Empty clusters ###


class TestEmptyClusters:
    """Verify graceful handling of seeds that receive no cells."""

    def test_far_seed_gets_zero_area(self, two_triangles_3d: Mesh):
        """A seed far from all cells should get zero area."""
        seeds = torch.tensor(
            [[0.3, 0.3, 0.0], [0.7, 0.7, 0.0], [100.0, 100.0, 100.0]],
            dtype=torch.float64,
        )
        result = partition_cells(two_triangles_3d, seeds)
        assert result.cluster_areas[2].item() == 0.0

    def test_empty_cluster_centroid_is_seed(self, two_triangles_3d: Mesh):
        """Empty cluster centroid should fall back to the seed position."""
        far_seed = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
        seeds = torch.tensor(
            [[0.3, 0.3, 0.0], [0.7, 0.7, 0.0], far_seed.tolist()],
            dtype=torch.float64,
        )
        result = partition_cells(two_triangles_3d, seeds)
        torch.testing.assert_close(result.cluster_centroids[2], far_seed)

    def test_empty_cluster_normal_is_zero(self, two_triangles_3d: Mesh):
        seeds = torch.tensor(
            [[0.5, 0.5, 0.0], [100.0, 100.0, 100.0]],
            dtype=torch.float64,
        )
        result = partition_cells(two_triangles_3d, seeds)
        assert result.cluster_normals[1].norm().item() == 0.0


### Non-codimension-1 meshes ###


class TestNonSurfaceMesh:
    """Verify partitioning of meshes that are not codimension-1 surfaces."""

    def test_2d_triangles_give_zero_normals(self):
        """Triangles in 2D (codimension 0) should produce zero normals."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 0.5]],
            dtype=torch.float64,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        mesh = Mesh(points=points, cells=cells)
        seeds = torch.tensor([[0.4, 0.3], [1.0, 0.5]], dtype=torch.float64)

        result = partition_cells(mesh, seeds)

        assert result.cluster_normals.abs().max().item() == 0.0
        # Areas should still be conserved
        assert result.cluster_areas.sum().item() == pytest.approx(
            mesh.cell_areas.sum().item()
        )


### Larger-scale correctness ###


class TestScaling:
    """Verify correctness and area conservation on a larger mesh (722 triangles)."""

    def test_many_cells_matches_brute_force(self):
        """On a larger mesh, assignments match brute-force exactly."""
        torch.manual_seed(0)
        m = plane.load(subdivisions=19)  # 722 triangles
        mesh = Mesh(points=m.points.double(), cells=m.cells)

        seeds = torch.rand(50, 3, dtype=torch.float64)
        seeds[:, 2] = 0.0

        result = partition_cells(mesh, seeds)
        expected = _brute_force_assignments(mesh.cell_centroids, seeds)
        assert torch.equal(result.assignments, expected)

        # Area conservation
        assert result.cluster_areas.sum().item() == pytest.approx(
            mesh.cell_areas.sum().item()
        )


### Grid nearest-neighbor robustness ###


class TestGridRobustness:
    """Stress the grid-accelerated nearest neighbor with adversarial seed layouts."""

    def test_queries_outside_seed_bbox(self):
        """Queries far from all seeds must still find their true nearest seed.

        Exercises the brute-force fallback and guards against cell-ID hash
        collisions from out-of-range grid coordinates.
        """
        mesh_pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [50.0, 50.0, 50.0],
                [51.0, 50.0, 50.0],
                [50.0, 51.0, 50.0],
            ],
            dtype=torch.float64,
        )
        mesh = Mesh(points=mesh_pts, cells=torch.tensor([[0, 1, 2], [3, 4, 5]]))
        seeds = torch.tensor([[0.3, 0.3, 0.0], [0.7, 0.7, 0.0]], dtype=torch.float64)
        result = partition_cells(mesh, seeds)
        expected = _brute_force_assignments(mesh.cell_centroids, seeds)
        assert torch.equal(result.assignments, expected)

    def test_3d_seeds_not_coplanar(self):
        """Seeds scattered in full 3D, not confined to a plane."""
        torch.manual_seed(99)
        m = plane.load(subdivisions=10)
        mesh = Mesh(points=m.points.double(), cells=m.cells)
        seeds = torch.rand(20, 3, dtype=torch.float64) * 2 - 1  # [-1, 1]^3
        result = partition_cells(mesh, seeds)
        expected = _brute_force_assignments(mesh.cell_centroids, seeds)
        assert torch.equal(result.assignments, expected)

    def test_nonuniform_seed_spacing(self):
        """Tight seed cluster plus a distant outlier seed."""
        torch.manual_seed(11)
        m = plane.load(subdivisions=8)
        mesh = Mesh(points=m.points.double(), cells=m.cells)
        tight = torch.rand(15, 3, dtype=torch.float64) * 0.1
        tight[:, 2] = 0.0
        outlier = torch.tensor([[5.0, 5.0, 0.0]], dtype=torch.float64)
        seeds = torch.cat([tight, outlier])
        result = partition_cells(mesh, seeds)
        expected = _brute_force_assignments(mesh.cell_centroids, seeds)
        assert torch.equal(result.assignments, expected)

    def test_deterministic(self):
        """Two calls with the same inputs give identical results."""
        torch.manual_seed(5)
        m = plane.load(subdivisions=12)
        mesh = Mesh(points=m.points.double(), cells=m.cells)
        seeds = torch.rand(30, 3, dtype=torch.float64)
        seeds[:, 2] = 0.0
        r1 = partition_cells(mesh, seeds)
        r2 = partition_cells(mesh, seeds)
        assert torch.equal(r1.assignments, r2.assignments)


### Input validation ###


def assert_on_device(tensor: torch.Tensor, expected_device: str) -> None:
    """Assert tensor is on expected device."""
    actual_device = tensor.device.type
    assert actual_device == expected_device, (
        f"Device mismatch: tensor is on {actual_device!r}, expected {expected_device!r}"
    )


class TestInputValidation:
    """Verify that mismatched device or dtype between mesh and seeds raises ValueError."""

    def test_dtype_mismatch_raises(self):
        """Mismatched seeds/mesh dtypes must raise ValueError."""
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        mesh = Mesh(points=points, cells=cells)
        seeds = torch.tensor([[0.3, 0.3, 0.0], [0.7, 0.7, 0.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="dtype"):
            partition_cells(mesh, seeds)

    @pytest.mark.cuda
    def test_device_mismatch_raises(self):
        """Mismatched seeds/mesh devices must raise ValueError."""
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        mesh = Mesh(points=points, cells=cells)  # CPU
        seeds = torch.tensor(
            [[0.3, 0.3, 0.0], [0.7, 0.7, 0.0]], dtype=torch.float64, device="cuda"
        )
        with pytest.raises(ValueError, match="device"):
            partition_cells(mesh, seeds)


### Device compatibility ###


class TestDeviceCompat:
    """Verify partition_cells works correctly on all available devices."""

    def test_assignments_and_area_conservation(self, device):
        """Assignment correctness and area conservation on target device."""
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], device=device)
        mesh = Mesh(points=points, cells=cells)
        seeds = torch.tensor(
            [[0.25, 0.25, 0.0], [0.75, 0.75, 0.0]], dtype=torch.float64, device=device
        )

        result = partition_cells(mesh, seeds)

        assert result.assignments.tolist() == [0, 1]
        assert result.cluster_areas.sum().item() == pytest.approx(
            mesh.cell_areas.sum().item()
        )
        for field in result:
            assert_on_device(field, device)

    def test_normals_on_surface(self, device):
        """Coplanar surface normals point along z on target device."""
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], device=device)
        mesh = Mesh(points=points, cells=cells)
        seeds = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64, device=device)

        result = partition_cells(mesh, seeds)

        assert abs(result.cluster_normals[0, 2].abs().item() - 1.0) < 1e-10
