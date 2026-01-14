"""Tests verifying equivalence between hierarchical and non-hierarchical sampling."""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.sampling import sample_data as non_hierarchical
from physicsnemo.mesh.sampling import sample_data_hierarchical as hierarchical
from physicsnemo.mesh.spatial import BVH


class TestEquivalence2D:
    """Test equivalence for 2D meshes."""

    def test_cell_data_sampling_equivalence(self):
        """Verify hierarchical and non-hierarchical give same results for cell data."""
        ### Create a mesh with cell data
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [2.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
                [1, 4, 3],
                [4, 5, 3],
            ]
        )
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"temperature": torch.tensor([100.0, 200.0, 300.0, 400.0])},
        )

        ### Query points
        queries = torch.tensor(
            [
                [0.25, 0.25],  # In first cell
                [0.75, 0.75],  # In second cell
                [1.5, 0.5],  # In third cell
                [10.0, 10.0],  # Outside
            ]
        )

        ### Sample with both methods
        result_brute = non_hierarchical.sample_data_at_points(
            mesh, queries, data_source="cells"
        )
        result_hierarchical = hierarchical.sample_data_at_points(
            mesh, queries, data_source="cells"
        )

        ### Results should be identical
        for key in result_brute.keys():
            assert torch.allclose(
                result_brute[key],
                result_hierarchical[key],
                equal_nan=True,
            ), f"Mismatch for {key=}"

    def test_point_data_interpolation_equivalence(self):
        """Verify interpolation gives same results."""
        ### Create a mesh with point data
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
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": torch.tensor([0.0, 1.0, 2.0, 3.0])},
        )

        ### Query points
        queries = torch.tensor(
            [
                [0.25, 0.25],
                [0.75, 0.75],
                [0.5, 0.5],  # On shared edge
            ]
        )

        ### Sample with both methods
        result_brute = non_hierarchical.sample_data_at_points(
            mesh, queries, data_source="points"
        )
        result_hierarchical = hierarchical.sample_data_at_points(
            mesh, queries, data_source="points"
        )

        ### Results should be identical
        for key in result_brute.keys():
            assert torch.allclose(
                result_brute[key],
                result_hierarchical[key],
                equal_nan=True,
                atol=1e-6,
            ), f"Mismatch for {key=}"

    def test_multidimensional_data_equivalence(self):
        """Test equivalence for multi-dimensional data arrays."""
        ### Create mesh with vector data
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={
                "velocity": torch.tensor(
                    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
                )
            },
        )

        queries = torch.tensor([[0.25, 0.25], [0.75, 0.75]])

        ### Sample
        result_brute = non_hierarchical.sample_data_at_points(
            mesh, queries, data_source="points"
        )
        result_hierarchical = hierarchical.sample_data_at_points(
            mesh, queries, data_source="points"
        )

        ### Verify
        assert torch.allclose(
            result_brute["velocity"],
            result_hierarchical["velocity"],
            atol=1e-6,
        )


class TestEquivalence3D:
    """Test equivalence for 3D meshes."""

    def test_tetrahedral_mesh_equivalence(self):
        """Test on 3D tetrahedral mesh."""
        ### Create a tetrahedral mesh
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
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"pressure": torch.tensor([1000.0, 2000.0])},
            point_data={
                "temperature": torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0])
            },
        )

        ### Query points
        queries = torch.tensor(
            [
                [0.25, 0.25, 0.25],  # Inside first tet
                [0.5, 0.5, 0.5],  # Possibly in second tet
                [10.0, 10.0, 10.0],  # Outside
            ]
        )

        ### Test cell data
        result_brute_cells = non_hierarchical.sample_data_at_points(
            mesh, queries, data_source="cells"
        )
        result_hier_cells = hierarchical.sample_data_at_points(
            mesh, queries, data_source="cells"
        )
        assert torch.allclose(
            result_brute_cells["pressure"],
            result_hier_cells["pressure"],
            equal_nan=True,
        )

        ### Test point data
        result_brute_points = non_hierarchical.sample_data_at_points(
            mesh, queries, data_source="points"
        )
        result_hier_points = hierarchical.sample_data_at_points(
            mesh, queries, data_source="points"
        )
        assert torch.allclose(
            result_brute_points["temperature"],
            result_hier_points["temperature"],
            equal_nan=True,
            atol=1e-5,
        )


class TestEquivalenceMultipleCells:
    """Test equivalence for multiple cells strategy."""

    def test_mean_strategy_equivalence(self):
        """Test mean strategy gives same results."""
        ### Create overlapping cells (shared edge)
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.5, -1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 3],
            ]
        )
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"value": torch.tensor([100.0, 200.0])},
        )

        ### Query on shared edge
        queries = torch.tensor([[0.5, 0.0]])

        ### Sample with mean strategy
        result_brute = non_hierarchical.sample_data_at_points(
            mesh, queries, data_source="cells", multiple_cells_strategy="mean"
        )
        result_hierarchical = hierarchical.sample_data_at_points(
            mesh, queries, data_source="cells", multiple_cells_strategy="mean"
        )

        ### Should be equal
        assert torch.allclose(
            result_brute["value"],
            result_hierarchical["value"],
            equal_nan=True,
        )

    def test_nan_strategy_equivalence(self):
        """Test nan strategy gives same results."""
        ### Same setup as above
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.5, -1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 3],
            ]
        )
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"value": torch.tensor([100.0, 200.0])},
        )

        queries = torch.tensor([[0.5, 0.0], [0.25, 0.25]])

        ### Sample with nan strategy
        result_brute = non_hierarchical.sample_data_at_points(
            mesh, queries, data_source="cells", multiple_cells_strategy="nan"
        )
        result_hierarchical = hierarchical.sample_data_at_points(
            mesh, queries, data_source="cells", multiple_cells_strategy="nan"
        )

        ### Should be equal (both NaN or both valid)
        assert torch.allclose(
            result_brute["value"],
            result_hierarchical["value"],
            equal_nan=True,
        )


class TestEquivalenceLargeMesh:
    """Test equivalence on larger meshes."""

    def test_random_mesh_equivalence(self):
        """Test on randomly generated mesh."""
        ### Generate a structured grid mesh (more predictable than random triangles)
        torch.manual_seed(42)

        # Create a grid of points
        nx, ny = 5, 5
        x = torch.linspace(0, 10, nx)
        y = torch.linspace(0, 10, ny)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

        # Create triangles from grid
        cells_list = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                # Two triangles per grid cell
                idx = i * ny + j
                # Lower triangle
                cells_list.append([idx, idx + ny, idx + 1])
                # Upper triangle
                cells_list.append([idx + 1, idx + ny, idx + ny + 1])

        cells = torch.tensor(cells_list)

        # Add random data
        n_cells = cells.shape[0]
        n_points = points.shape[0]
        cell_data_vals = torch.rand(n_cells) * 100.0
        point_data_vals = torch.rand(n_points) * 100.0

        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"scalar": cell_data_vals},
            point_data={"scalar": point_data_vals},
        )

        ### Random query points
        n_queries = 20
        queries = torch.rand(n_queries, 2) * 10.0

        ### Sample both ways
        result_brute = non_hierarchical.sample_data_at_points(
            mesh, queries, data_source="cells"
        )
        result_hierarchical = hierarchical.sample_data_at_points(
            mesh, queries, data_source="cells"
        )

        ### Results should match
        assert torch.allclose(
            result_brute["scalar"],
            result_hierarchical["scalar"],
            equal_nan=True,
        )


@pytest.mark.cuda
class TestEquivalenceGPU:
    """Test equivalence on GPU."""

    def test_gpu_equivalence(self):
        """Test that GPU and CPU give same results."""
        ### Create mesh on CPU
        points_cpu = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells_cpu = torch.tensor([[0, 1, 2], [1, 3, 2]])
        mesh_cpu = Mesh(
            points=points_cpu,
            cells=cells_cpu,
            cell_data={"temp": torch.tensor([100.0, 200.0])},
        )
        queries_cpu = torch.tensor([[0.25, 0.25], [0.75, 0.75]])

        ### Move to GPU
        mesh_gpu = Mesh(
            points=points_cpu.cuda(),
            cells=cells_cpu.cuda(),
            cell_data={"temp": torch.tensor([100.0, 200.0]).cuda()},
        )
        queries_gpu = queries_cpu.cuda()

        ### Sample on both devices
        result_cpu = hierarchical.sample_data_at_points(mesh_cpu, queries_cpu)
        result_gpu = hierarchical.sample_data_at_points(mesh_gpu, queries_gpu)

        ### Results should match
        assert torch.allclose(
            result_cpu["temp"],
            result_gpu["temp"].cpu(),
        )

    @pytest.mark.cuda
    def test_bvh_on_gpu(self):
        """Test that BVH works on GPU."""
        ### Create mesh on GPU
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            device="cuda",
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], device="cuda")
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"temp": torch.tensor([100.0, 200.0], device="cuda")},
        )

        ### Build BVH on GPU
        bvh = BVH.from_mesh(mesh)
        assert bvh.device.type == "cuda"

        ### Query on GPU
        queries = torch.tensor([[0.25, 0.25]], device="cuda")
        result = hierarchical.sample_data_at_points(mesh, queries, bvh=bvh)

        assert result["temp"].device.type == "cuda"
        assert not torch.isnan(result["temp"][0])
