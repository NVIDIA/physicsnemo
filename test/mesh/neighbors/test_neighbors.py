"""Tests for neighbor and adjacency computation.

Tests validate physicsnemo.mesh adjacency computations against PyVista's VTK-based
implementations as ground truth, and verify correctness across spatial dimensions,
manifold dimensions, and compute backends.
"""

import pytest
import pyvista as pv
import torch

from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.mesh import Mesh

### Helper Functions (shared across tests) ###


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


def create_single_cell_mesh(
    n_spatial_dims: int, n_manifold_dims: int, device: str = "cpu"
):
    """Create a mesh with a single cell."""
    if n_manifold_dims > n_spatial_dims:
        raise ValueError(
            f"Manifold dimension {n_manifold_dims} cannot exceed spatial dimension {n_spatial_dims}"
        )

    if n_manifold_dims == 0:
        if n_spatial_dims == 2:
            points = torch.tensor([[0.5, 0.5]], device=device)
        elif n_spatial_dims == 3:
            points = torch.tensor([[0.5, 0.5, 0.5]], device=device)
        else:
            raise ValueError(f"Unsupported {n_spatial_dims=}")
        cells = torch.tensor([[0]], device=device, dtype=torch.int64)
    elif n_manifold_dims == 1:
        if n_spatial_dims == 2:
            points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device=device)
        elif n_spatial_dims == 3:
            points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device)
        else:
            raise ValueError(f"Unsupported {n_spatial_dims=}")
        cells = torch.tensor([[0, 1]], device=device, dtype=torch.int64)
    elif n_manifold_dims == 2:
        if n_spatial_dims == 2:
            points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], device=device)
        elif n_spatial_dims == 3:
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device
            )
        else:
            raise ValueError(f"Unsupported {n_spatial_dims=}")
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
    elif n_manifold_dims == 3:
        if n_spatial_dims == 3:
            points = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
            )
            cells = torch.tensor([[0, 1, 2, 3]], device=device, dtype=torch.int64)
        else:
            raise ValueError("3-simplices require 3D embedding space")
    else:
        raise ValueError(f"Unsupported {n_manifold_dims=}")

    return Mesh(points=points, cells=cells)


def assert_mesh_valid(mesh, strict: bool = True) -> None:
    """Assert that a mesh is valid and well-formed."""
    assert mesh.n_points > 0
    assert mesh.points.ndim == 2
    assert mesh.points.shape[1] == mesh.n_spatial_dims

    if mesh.n_cells > 0:
        assert mesh.cells.ndim == 2
        assert mesh.cells.shape[1] == mesh.n_manifold_dims + 1
        assert torch.all(mesh.cells >= 0)
        assert torch.all(mesh.cells < mesh.n_points)

    assert mesh.points.dtype in [torch.float32, torch.float64]
    assert mesh.cells.dtype == torch.int64
    assert mesh.points.device == mesh.cells.device

    if strict and mesh.n_cells > 0:
        for i in range(mesh.n_cells):
            cell_verts = mesh.cells[i]
            unique_verts = torch.unique(cell_verts)
            assert len(unique_verts) == len(cell_verts)


def assert_on_device(tensor: torch.Tensor, expected_device: str) -> None:
    """Assert tensor is on expected device."""
    actual_device = tensor.device.type
    assert actual_device == expected_device, (
        f"Device mismatch: tensor is on {actual_device!r}, expected {expected_device!r}"
    )


### Test Fixtures ###


@pytest.fixture
def airplane_mesh_pair(device):
    """2D manifold (triangular surface) in 3D space."""
    pv_mesh = pv.examples.load_airplane()
    tm_mesh = from_pyvista(pv_mesh)
    tm_mesh = Mesh(
        points=tm_mesh.points.to(device),
        cells=tm_mesh.cells.to(device),
        point_data=tm_mesh.point_data,
        cell_data=tm_mesh.cell_data,
    )
    return tm_mesh, pv_mesh


@pytest.fixture
def tetbeam_mesh_pair(device):
    """3D manifold (tetrahedral volume) in 3D space."""
    pv_mesh = pv.examples.load_tetbeam()
    tm_mesh = from_pyvista(pv_mesh)
    tm_mesh = Mesh(
        points=tm_mesh.points.to(device),
        cells=tm_mesh.cells.to(device),
        point_data=tm_mesh.point_data,
        cell_data=tm_mesh.cell_data,
    )
    return tm_mesh, pv_mesh


class TestPointToPointsAdjacency:
    """Test point-to-points (edge) adjacency computation."""

    ### Cross-validation against PyVista ###

    def test_airplane_point_neighbors(self, airplane_mesh_pair):
        """Validate point-to-points adjacency against PyVista for airplane mesh."""
        tm_mesh, pv_mesh = airplane_mesh_pair
        device = tm_mesh.points.device.type

        ### Compute adjacency using physicsnemo.mesh
        adj = tm_mesh.get_point_to_points_adjacency()
        assert_on_device(adj.offsets, device)
        assert_on_device(adj.indices, device)

        tm_neighbors = adj.to_list()

        ### Get ground truth from PyVista (requires Python loop)
        pv_neighbors = []
        for i in range(pv_mesh.n_points):
            neighbors = pv_mesh.point_neighbors(i)
            pv_neighbors.append(neighbors)

        ### Compare results (order-independent)
        assert len(tm_neighbors) == len(pv_neighbors), (
            f"Mismatch in number of points: physicsnemo.mesh={len(tm_neighbors)}, pyvista={len(pv_neighbors)}"
        )

        for i, (tm_nbrs, pv_nbrs) in enumerate(zip(tm_neighbors, pv_neighbors)):
            # Sort both for order-independent comparison
            tm_sorted = sorted(tm_nbrs)
            pv_sorted = sorted(pv_nbrs)
            assert tm_sorted == pv_sorted, (
                f"Point {i} neighbors mismatch:\n  physicsnemo.mesh: {tm_sorted}\n  pyvista:   {pv_sorted}"
            )

    def test_tetbeam_point_neighbors(self, tetbeam_mesh_pair):
        """Validate point-to-points adjacency against PyVista for tetbeam mesh."""
        tm_mesh, pv_mesh = tetbeam_mesh_pair
        device = tm_mesh.points.device.type

        ### Compute adjacency using physicsnemo.mesh
        adj = tm_mesh.get_point_to_points_adjacency()
        assert_on_device(adj.offsets, device)
        assert_on_device(adj.indices, device)

        tm_neighbors = adj.to_list()

        ### Get ground truth from PyVista (requires Python loop)
        pv_neighbors = []
        for i in range(pv_mesh.n_points):
            neighbors = pv_mesh.point_neighbors(i)
            pv_neighbors.append(neighbors)

        ### Compare results (order-independent)
        assert len(tm_neighbors) == len(pv_neighbors)

        for i, (tm_nbrs, pv_nbrs) in enumerate(zip(tm_neighbors, pv_neighbors)):
            tm_sorted = sorted(tm_nbrs)
            pv_sorted = sorted(pv_nbrs)
            assert tm_sorted == pv_sorted, (
                f"Point {i} neighbors mismatch:\n  physicsnemo.mesh: {tm_sorted}\n  pyvista:   {pv_sorted}"
            )

    ### Symmetry Tests on Real-World Meshes ###

    def test_symmetry_airplane(self, airplane_mesh_pair):
        """Verify point adjacency is symmetric on airplane mesh (complex real-world case)."""
        tm_mesh, _ = airplane_mesh_pair

        adj = tm_mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                # If j is a neighbor of i, then i must be a neighbor of j
                assert i in neighbors[j], (
                    f"Asymmetric adjacency: {i} neighbors {j}, but {j} doesn't neighbor {i}"
                )

    def test_symmetry_tetbeam(self, tetbeam_mesh_pair):
        """Verify point adjacency is symmetric on tetbeam mesh (complex real-world case)."""
        tm_mesh, _ = tetbeam_mesh_pair

        adj = tm_mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                assert i in neighbors[j], (
                    f"Asymmetric adjacency: {i} neighbors {j}, but {j} doesn't neighbor {i}"
                )

    ### Parametrized Tests on Synthetic Meshes (Exhaustive Dimensional Coverage) ###

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
    def test_symmetry_parametrized(self, n_spatial_dims, n_manifold_dims, device):
        """Verify point adjacency is symmetric across all dimension combinations (synthetic meshes)."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)
        assert_mesh_valid(mesh, strict=True)

        adj = mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()

        ### Verify symmetry: if A neighbors B, then B neighbors A
        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                assert i in neighbors[j], (
                    f"Asymmetric adjacency ({n_spatial_dims=}, {n_manifold_dims=}): "
                    f"{i} neighbors {j}, but {j} doesn't neighbor {i}"
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
    def test_no_self_loops_parametrized(self, n_spatial_dims, n_manifold_dims, device):
        """Verify no point is its own neighbor across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        adj = mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            assert i not in nbrs, (
                f"Point {i} is listed as its own neighbor ({n_spatial_dims=}, {n_manifold_dims=})"
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
    def test_no_duplicates_parametrized(self, n_spatial_dims, n_manifold_dims, device):
        """Verify each neighbor appears exactly once across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        adj = mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            assert len(nbrs) == len(set(nbrs)), (
                f"Point {i} has duplicate neighbors: {nbrs} "
                f"({n_spatial_dims=}, {n_manifold_dims=})"
            )

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", [(2, 1), (3, 2)])
    def test_single_cell_connectivity(self, n_spatial_dims, n_manifold_dims, device):
        """Test point-to-points for single cell across dimensions."""
        mesh = create_single_cell_mesh(n_spatial_dims, n_manifold_dims, device=device)

        adj = mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()

        ### All vertices in a single cell should be connected to each other
        n_verts = n_manifold_dims + 1
        assert len(neighbors) == n_verts

        for i, nbrs in enumerate(neighbors):
            # Each vertex should neighbor all others except itself
            expected_neighbors = set(range(n_verts)) - {i}
            actual_neighbors = set(nbrs)
            assert actual_neighbors == expected_neighbors, (
                f"Single cell connectivity mismatch at vertex {i}: "
                f"expected {sorted(expected_neighbors)}, got {sorted(actual_neighbors)}"
            )


class TestCellToCellsAdjacency:
    """Test cell-to-cells adjacency computation."""

    ### Cross-validation against PyVista ###

    def test_airplane_cell_neighbors(self, airplane_mesh_pair):
        """Validate cell-to-cells adjacency against PyVista for airplane mesh."""
        tm_mesh, pv_mesh = airplane_mesh_pair
        device = tm_mesh.points.device.type

        ### Compute adjacency using physicsnemo.mesh
        # For triangular mesh, codimension=1 means sharing an edge
        adj = tm_mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        assert_on_device(adj.offsets, device)
        assert_on_device(adj.indices, device)

        tm_neighbors = adj.to_list()

        ### Get ground truth from PyVista
        # For triangular meshes, codimension=1 (sharing an edge) corresponds to
        # PyVista's connections="edges"
        pv_neighbors = []
        for i in range(pv_mesh.n_cells):
            neighbors = pv_mesh.cell_neighbors(i, connections="edges")
            pv_neighbors.append(neighbors)

        ### Compare results (order-independent)
        assert len(tm_neighbors) == len(pv_neighbors), (
            f"Mismatch in number of cells: physicsnemo.mesh={len(tm_neighbors)}, pyvista={len(pv_neighbors)}"
        )

        for i, (tm_nbrs, pv_nbrs) in enumerate(zip(tm_neighbors, pv_neighbors)):
            tm_sorted = sorted(tm_nbrs)
            pv_sorted = sorted(pv_nbrs)
            assert tm_sorted == pv_sorted, (
                f"Cell {i} neighbors mismatch:\n  physicsnemo.mesh: {tm_sorted}\n  pyvista:   {pv_sorted}"
            )

    def test_tetbeam_cell_neighbors(self, tetbeam_mesh_pair):
        """Validate cell-to-cells adjacency against PyVista for tetbeam mesh."""
        tm_mesh, pv_mesh = tetbeam_mesh_pair
        device = tm_mesh.points.device.type

        ### Compute adjacency using physicsnemo.mesh
        # For tetrahedral mesh, codimension=1 means sharing a triangular face
        adj = tm_mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        assert_on_device(adj.offsets, device)
        assert_on_device(adj.indices, device)

        tm_neighbors = adj.to_list()

        ### Get ground truth from PyVista
        # For tetrahedral meshes, codimension=1 (sharing a face) corresponds to
        # PyVista's connections="faces"
        pv_neighbors = []
        for i in range(pv_mesh.n_cells):
            neighbors = pv_mesh.cell_neighbors(i, connections="faces")
            pv_neighbors.append(neighbors)

        ### Compare results
        assert len(tm_neighbors) == len(pv_neighbors)

        for i, (tm_nbrs, pv_nbrs) in enumerate(zip(tm_neighbors, pv_neighbors)):
            tm_sorted = sorted(tm_nbrs)
            pv_sorted = sorted(pv_nbrs)
            assert tm_sorted == pv_sorted, (
                f"Cell {i} neighbors mismatch:\n  physicsnemo.mesh: {tm_sorted}\n  pyvista:   {pv_sorted}"
            )

    ### Symmetry Tests on Real-World Meshes ###

    def test_symmetry_airplane(self, airplane_mesh_pair):
        """Verify cell adjacency is symmetric on airplane mesh (complex real-world case)."""
        tm_mesh, _ = airplane_mesh_pair

        adj = tm_mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                assert i in neighbors[j], (
                    f"Asymmetric adjacency: cell {i} neighbors cell {j}, "
                    f"but cell {j} doesn't neighbor cell {i}"
                )

    def test_symmetry_tetbeam(self, tetbeam_mesh_pair):
        """Verify cell adjacency is symmetric on tetbeam mesh (complex real-world case)."""
        tm_mesh, _ = tetbeam_mesh_pair

        adj = tm_mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                assert i in neighbors[j], (
                    f"Asymmetric adjacency: cell {i} neighbors cell {j}, "
                    f"but cell {j} doesn't neighbor cell {i}"
                )

    ### Parametrized Tests on Synthetic Meshes (Exhaustive Dimensional Coverage) ###

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
    def test_symmetry_parametrized(self, n_spatial_dims, n_manifold_dims, device):
        """Verify cell adjacency is symmetric across all dimension combinations (synthetic meshes)."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)
        assert_mesh_valid(mesh, strict=True)

        adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                assert i in neighbors[j], (
                    f"Asymmetric adjacency ({n_spatial_dims=}, {n_manifold_dims=}): "
                    f"cell {i} neighbors cell {j}, but cell {j} doesn't neighbor cell {i}"
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
    def test_no_self_loops_parametrized(self, n_spatial_dims, n_manifold_dims, device):
        """Verify no cell is its own neighbor across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            assert i not in nbrs, (
                f"Cell {i} is listed as its own neighbor ({n_spatial_dims=}, {n_manifold_dims=})"
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
    def test_no_duplicates_parametrized(self, n_spatial_dims, n_manifold_dims, device):
        """Verify each neighbor appears exactly once across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            assert len(nbrs) == len(set(nbrs)), (
                f"Cell {i} has duplicate neighbors: {nbrs} "
                f"({n_spatial_dims=}, {n_manifold_dims=})"
            )

    @pytest.mark.parametrize(
        "n_manifold_dims,adjacency_codim",
        [
            (1, 1),  # Edges sharing vertices
            (2, 1),  # Triangles sharing edges
            (2, 2),  # Triangles sharing vertices
            (3, 1),  # Tets sharing faces
            (3, 2),  # Tets sharing edges
            (3, 3),  # Tets sharing vertices
        ],
    )
    def test_different_codimensions(self, n_manifold_dims, adjacency_codim, device):
        """Test adjacency with different codimensions."""
        # Use 3D space for all to support up to 3D manifolds
        n_spatial_dims = 3
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=adjacency_codim)
        neighbors = adj.to_list()

        ### Higher codimension should give same or more neighbors
        ### (more permissive connectivity criterion)
        if adjacency_codim < n_manifold_dims:
            adj_lower = mesh.get_cell_to_cells_adjacency(
                adjacency_codimension=adjacency_codim + 1
            )
            neighbors_lower = adj_lower.to_list()

            for i in range(len(neighbors)):
                # Lower codimension should be subset of higher codimension
                set_codim = set(neighbors[i])
                set_lower = set(neighbors_lower[i])
                assert set_codim.issubset(set_lower) or set_codim == set_lower, (
                    f"Codimension {adjacency_codim} neighbors should be subset of "
                    f"codimension {adjacency_codim + 1} neighbors"
                )


class TestPointToCellsAdjacency:
    """Test point-to-cells (star) adjacency computation."""

    @pytest.fixture
    def simple_triangles(self, device):
        """Simple triangle mesh for basic testing."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            device=device,
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ],
            device=device,
            dtype=torch.int64,
        )
        return Mesh(points=points, cells=cells)

    def test_simple_triangle_star(self, simple_triangles):
        """Test star computation on simple triangle mesh."""
        mesh = simple_triangles
        device = mesh.points.device.type

        adj = mesh.get_point_to_cells_adjacency()
        assert_on_device(adj.offsets, device)
        assert_on_device(adj.indices, device)

        stars = adj.to_list()

        # Point 0 is in cell 0 only
        assert sorted(stars[0]) == [0]

        # Point 1 is in cells 0 and 1
        assert sorted(stars[1]) == [0, 1]

        # Point 2 is in cells 0 and 1
        assert sorted(stars[2]) == [0, 1]

        # Point 3 is in cell 1 only
        assert sorted(stars[3]) == [1]

    def test_airplane_consistency(self, airplane_mesh_pair):
        """Verify consistency of point-to-cells adjacency for airplane mesh."""
        tm_mesh, pv_mesh = airplane_mesh_pair

        adj = tm_mesh.get_point_to_cells_adjacency()
        stars = adj.to_list()

        ### Verify each cell's vertices have that cell in their star
        for cell_id in range(tm_mesh.n_cells):
            cell_vertices = tm_mesh.cells[cell_id].tolist()
            for vertex_id in cell_vertices:
                assert cell_id in stars[vertex_id], (
                    f"Cell {cell_id} contains vertex {vertex_id}, "
                    f"but vertex's star doesn't contain the cell"
                )

    def test_tetbeam_consistency(self, tetbeam_mesh_pair):
        """Verify consistency of point-to-cells adjacency for tetbeam mesh."""
        tm_mesh, pv_mesh = tetbeam_mesh_pair

        adj = tm_mesh.get_point_to_cells_adjacency()
        stars = adj.to_list()

        ### Verify each cell's vertices have that cell in their star
        for cell_id in range(tm_mesh.n_cells):
            cell_vertices = tm_mesh.cells[cell_id].tolist()
            for vertex_id in cell_vertices:
                assert cell_id in stars[vertex_id], (
                    f"Cell {cell_id} contains vertex {vertex_id}, "
                    f"but vertex's star doesn't contain the cell"
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
    def test_no_duplicates_parametrized(self, n_spatial_dims, n_manifold_dims, device):
        """Verify each cell appears exactly once in each point's star."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        adj = mesh.get_point_to_cells_adjacency()
        stars = adj.to_list()

        for i, cells in enumerate(stars):
            assert len(cells) == len(set(cells)), (
                f"Point {i} has duplicate cells in star: {cells} "
                f"({n_spatial_dims=}, {n_manifold_dims=})"
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
    def test_completeness_parametrized(self, n_spatial_dims, n_manifold_dims, device):
        """Verify all cell-point relationships are captured."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        adj = mesh.get_point_to_cells_adjacency()
        stars = adj.to_list()

        ### Check that every cell-vertex relationship is present
        for cell_id in range(mesh.n_cells):
            cell_verts = mesh.cells[cell_id].tolist()
            for vert_id in cell_verts:
                assert cell_id in stars[vert_id], (
                    f"Cell {cell_id} contains vertex {vert_id} but vertex's star "
                    f"doesn't contain the cell ({n_spatial_dims=}, {n_manifold_dims=})"
                )


class TestCellsToPointsAdjacency:
    """Test cells-to-points adjacency computation."""

    @pytest.fixture
    def simple_triangles(self, device):
        """Simple triangle mesh for basic testing."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            device=device,
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ],
            device=device,
            dtype=torch.int64,
        )
        return Mesh(points=points, cells=cells)

    def test_simple_triangle_vertices(self, simple_triangles):
        """Test cells-to-points on simple triangle mesh."""
        mesh = simple_triangles
        device = mesh.points.device.type

        adj = mesh.get_cells_to_points_adjacency()
        assert_on_device(adj.offsets, device)
        assert_on_device(adj.indices, device)

        vertices = adj.to_list()

        # Cell 0 has vertices [0, 1, 2]
        assert vertices[0] == [0, 1, 2]

        # Cell 1 has vertices [1, 3, 2]
        assert vertices[1] == [1, 3, 2]

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
    def test_matches_cells_array_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Verify cells-to-points matches the cells array across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        adj = mesh.get_cells_to_points_adjacency()
        vertices = adj.to_list()

        # Verify each cell's vertices match the cells array
        for i in range(mesh.n_cells):
            expected = mesh.cells[i].tolist()
            assert vertices[i] == expected, (
                f"Cell {i} vertices mismatch:\n"
                f"  adjacency: {vertices[i]}\n"
                f"  cells array: {expected}\n"
                f"  ({n_spatial_dims=}, {n_manifold_dims=})"
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
    def test_all_cells_same_size_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Verify all cells have the correct number of vertices."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        adj = mesh.get_cells_to_points_adjacency()
        vertices = adj.to_list()

        # All cells should have (n_manifold_dims + 1) vertices
        expected_size = n_manifold_dims + 1
        for i, verts in enumerate(vertices):
            assert len(verts) == expected_size, (
                f"Cell {i} has {len(verts)} vertices, expected {expected_size} "
                f"({n_spatial_dims=}, {n_manifold_dims=})"
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
    def test_inverse_of_point_to_cells_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Verify cells-to-points is inverse of point-to-cells."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        # Get both adjacencies
        cells_to_points = mesh.get_cells_to_points_adjacency().to_list()
        points_to_cells = mesh.get_point_to_cells_adjacency().to_list()

        # For each cell-point pair, verify the inverse relationship
        for cell_id, point_ids in enumerate(cells_to_points):
            for point_id in point_ids:
                # This point should have this cell in its star
                assert cell_id in points_to_cells[point_id], (
                    f"Cell {cell_id} contains point {point_id}, "
                    f"but point's star doesn't contain the cell "
                    f"({n_spatial_dims=}, {n_manifold_dims=})"
                )


class TestAdjacencyValidation:
    """Test Adjacency class validation."""

    def test_valid_adjacency(self, device):
        """Test that valid adjacencies pass validation."""
        from physicsnemo.mesh.neighbors import Adjacency

        # Empty adjacency
        adj = Adjacency(
            offsets=torch.tensor([0], device=device),
            indices=torch.tensor([], device=device),
        )
        assert adj.n_sources == 0

        # Single source with neighbors
        adj = Adjacency(
            offsets=torch.tensor([0, 3], device=device),
            indices=torch.tensor([1, 2, 3], device=device),
        )
        assert adj.n_sources == 1

        # Multiple sources with varying neighbor counts
        adj = Adjacency(
            offsets=torch.tensor([0, 2, 2, 5], device=device),
            indices=torch.tensor([10, 11, 12, 13, 14], device=device),
        )
        assert adj.n_sources == 3

    def test_invalid_empty_offsets(self, device):
        """Test that empty offsets array raises error."""
        from physicsnemo.mesh.neighbors import Adjacency

        with pytest.raises(ValueError, match="Offsets array must have length >= 1"):
            Adjacency(
                offsets=torch.tensor(
                    [], device=device
                ),  # Invalid: should be at least [0]
                indices=torch.tensor([], device=device),
            )

    def test_invalid_first_offset(self, device):
        """Test that non-zero first offset raises error."""
        from physicsnemo.mesh.neighbors import Adjacency

        with pytest.raises(ValueError, match="First offset must be 0"):
            Adjacency(
                offsets=torch.tensor([1, 3, 5], device=device),  # Should start at 0
                indices=torch.tensor([0, 1], device=device),
            )

    def test_invalid_last_offset(self, device):
        """Test that mismatched last offset raises error."""
        from physicsnemo.mesh.neighbors import Adjacency

        with pytest.raises(
            ValueError, match="Last offset must equal length of indices"
        ):
            Adjacency(
                offsets=torch.tensor([0, 2, 5], device=device),  # Says 5 indices
                indices=torch.tensor([0, 1, 2], device=device),  # But only 3 indices
            )

        with pytest.raises(
            ValueError, match="Last offset must equal length of indices"
        ):
            Adjacency(
                offsets=torch.tensor([0, 2], device=device),  # Says 2 indices
                indices=torch.tensor([0, 1, 2, 3], device=device),  # But has 4 indices
            )


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_mesh(self, device):
        """Test adjacency computation on empty mesh."""
        mesh = Mesh(
            points=torch.zeros(0, 3, device=device),
            cells=torch.zeros(0, 3, dtype=torch.int64, device=device),
        )

        # Point-to-points
        adj = mesh.get_point_to_points_adjacency()
        assert adj.n_sources == 0
        assert len(adj.indices) == 0
        assert_on_device(adj.offsets, device)

        # Point-to-cells
        adj = mesh.get_point_to_cells_adjacency()
        assert adj.n_sources == 0
        assert len(adj.indices) == 0

        # Cell-to-cells
        adj = mesh.get_cell_to_cells_adjacency()
        assert adj.n_sources == 0
        assert len(adj.indices) == 0

        # Cells-to-points
        adj = mesh.get_cells_to_points_adjacency()
        assert adj.n_sources == 0
        assert len(adj.indices) == 0

    def test_isolated_triangle(self, device):
        """Test single triangle (no cell neighbors)."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)

        mesh = Mesh(points=points, cells=cells)

        # Cell-to-cells: no neighbors
        adj = mesh.get_cell_to_cells_adjacency()
        neighbors = adj.to_list()
        assert neighbors == [[]]

        # Point-to-points: all connected
        adj = mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()
        assert sorted(neighbors[0]) == [1, 2]
        assert sorted(neighbors[1]) == [0, 2]
        assert sorted(neighbors[2]) == [0, 1]

    def test_isolated_points(self, device):
        """Test mesh with isolated points (not in any cells)."""
        # Create mesh with 5 points but only 1 triangle using points 0,1,2
        # Points 3 and 4 are isolated
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [2.0, 2.0],  # Isolated
                [3.0, 3.0],  # Isolated
            ],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)

        mesh = Mesh(points=points, cells=cells)

        # Point-to-cells: isolated points should have empty stars
        adj = mesh.get_point_to_cells_adjacency()
        stars = adj.to_list()
        assert len(stars[0]) > 0  # Point 0 is in cells
        assert len(stars[1]) > 0  # Point 1 is in cells
        assert len(stars[2]) > 0  # Point 2 is in cells
        assert len(stars[3]) == 0  # Point 3 is isolated
        assert len(stars[4]) == 0  # Point 4 is isolated

        # Point-to-points: isolated points should have no neighbors
        adj = mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()
        assert len(neighbors[3]) == 0
        assert len(neighbors[4]) == 0

    def test_single_point_mesh(self, device):
        """Test mesh with single point and no cells."""
        points = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        cells = torch.zeros((0, 3), dtype=torch.int64, device=device)

        mesh = Mesh(points=points, cells=cells)

        # Point-to-cells: single point with no cells
        adj = mesh.get_point_to_cells_adjacency()
        assert adj.n_sources == 1
        assert len(adj.indices) == 0
        assert adj.to_list() == [[]]

        # Point-to-points: single point with no neighbors
        adj = mesh.get_point_to_points_adjacency()
        assert adj.n_sources == 1
        assert len(adj.indices) == 0
        assert adj.to_list() == [[]]

    def test_1d_manifold_edges(self, device):
        """Test adjacency on 1D manifold (polyline/edges)."""
        # Create a simple polyline: 0--1--2--3
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            device=device,
        )
        cells = torch.tensor(
            [
                [0, 1],  # Edge 0
                [1, 2],  # Edge 1
                [2, 3],  # Edge 2
            ],
            device=device,
            dtype=torch.int64,
        )

        mesh = Mesh(points=points, cells=cells)

        # Cell-to-cells (codim 1 = sharing a vertex for edges)
        adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors = adj.to_list()

        # Edge 0 shares vertex 1 with edge 1
        assert sorted(neighbors[0]) == [1]
        # Edge 1 shares vertex 1 with edge 0, vertex 2 with edge 2
        assert sorted(neighbors[1]) == [0, 2]
        # Edge 2 shares vertex 2 with edge 1
        assert sorted(neighbors[2]) == [1]

        # Point-to-points should give the polyline connectivity
        adj = mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()
        assert sorted(neighbors[0]) == [1]
        assert sorted(neighbors[1]) == [0, 2]
        assert sorted(neighbors[2]) == [1, 3]
        assert sorted(neighbors[3]) == [2]

    def test_dtype_consistency(self, device):
        """Test that all adjacency indices use int64 dtype."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], device=device)
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)

        mesh = Mesh(points=points, cells=cells)

        # Check all adjacency types
        adjacencies = [
            mesh.get_point_to_points_adjacency(),
            mesh.get_point_to_cells_adjacency(),
            mesh.get_cell_to_cells_adjacency(),
            mesh.get_cells_to_points_adjacency(),
        ]

        for adj in adjacencies:
            assert adj.offsets.dtype == torch.int64, (
                f"Expected offsets dtype int64, got {adj.offsets.dtype}"
            )
            assert adj.indices.dtype == torch.int64, (
                f"Expected indices dtype int64, got {adj.indices.dtype}"
            )

    def test_neighbor_count_conservation(self, device):
        """Test conservation of neighbor relationships."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            device=device,
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ],
            device=device,
            dtype=torch.int64,
        )

        mesh = Mesh(points=points, cells=cells)

        # Point-to-points: total edges counted twice (bidirectional)
        adj = mesh.get_point_to_points_adjacency()
        total_bidirectional_edges = adj.n_total_neighbors
        # Should be even since each edge appears twice
        assert total_bidirectional_edges % 2 == 0

        # Cell-to-cells: total adjacencies counted twice (bidirectional)
        adj = mesh.get_cell_to_cells_adjacency()
        total_bidirectional_adjacencies = adj.n_total_neighbors
        # Should be even
        assert total_bidirectional_adjacencies % 2 == 0

        # Point-to-cells: sum should equal cells-to-points
        point_to_cells = mesh.get_point_to_cells_adjacency()
        cells_to_points = mesh.get_cells_to_points_adjacency()
        assert point_to_cells.n_total_neighbors == cells_to_points.n_total_neighbors

    def test_cross_adjacency_consistency(self, device):
        """Test consistency between different adjacency relationships."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            device=device,
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ],
            device=device,
            dtype=torch.int64,
        )

        mesh = Mesh(points=points, cells=cells)

        # Get all adjacencies
        point_to_points = mesh.get_point_to_points_adjacency().to_list()
        point_to_cells = mesh.get_point_to_cells_adjacency().to_list()
        cells_to_points = mesh.get_cells_to_points_adjacency().to_list()
        cell_to_cells = mesh.get_cell_to_cells_adjacency().to_list()

        # Consistency check 1: If points A and B are neighbors,
        # there must exist a cell containing both
        for point_a, neighbors in enumerate(point_to_points):
            for point_b in neighbors:
                # Find cells containing point_a
                cells_with_a = set(point_to_cells[point_a])
                # Find cells containing point_b
                cells_with_b = set(point_to_cells[point_b])
                # There must be at least one cell containing both
                shared_cells = cells_with_a & cells_with_b
                assert len(shared_cells) > 0, (
                    f"Points {point_a} and {point_b} are neighbors but share no cells"
                )

        # Consistency check 2: cells_to_points is inverse of point_to_cells
        for cell_id, point_ids in enumerate(cells_to_points):
            for point_id in point_ids:
                assert cell_id in point_to_cells[point_id], (
                    f"Cell {cell_id} contains point {point_id}, "
                    f"but point's star doesn't contain the cell"
                )

        # Consistency check 3: If cells A and B are neighbors (share edge),
        # they must share at least 2 vertices
        for cell_a, neighbors in enumerate(cell_to_cells):
            for cell_b in neighbors:
                vertices_a = set(cells_to_points[cell_a])
                vertices_b = set(cells_to_points[cell_b])
                shared_vertices = vertices_a & vertices_b
                # Sharing an edge means at least 2 shared vertices
                assert len(shared_vertices) >= 2, (
                    f"Cells {cell_a} and {cell_b} are neighbors but share "
                    f"{len(shared_vertices)} vertices (expected >= 2)"
                )


class TestDisjointMeshNeighborhood:
    """Test neighbor computation on disjoint meshes.

    Verifies that merging two spatially-separated meshes produces connectivity
    identical to computing connectivity separately, accounting for index offsets.
    """

    @pytest.fixture
    def sphere_pair(self, device):
        """Create two spheres with different resolutions, spatially separated."""
        from physicsnemo.mesh.primitives.surfaces.sphere_icosahedral import (
            load as load_sphere,
        )

        # Create sphere A with subdivision level 1
        sphere_a = load_sphere(radius=1.0, subdivisions=1, device=device)

        # Create sphere B with subdivision level 2 (different resolution)
        sphere_b_base = load_sphere(radius=1.0, subdivisions=2, device=device)

        # Translate sphere B far away to ensure disjoint (100 units in x-direction)
        translation = torch.tensor([100.0, 0.0, 0.0], device=device)
        sphere_b = Mesh(
            points=sphere_b_base.points + translation,
            cells=sphere_b_base.cells,
            point_data=sphere_b_base.point_data,
            cell_data=sphere_b_base.cell_data,
            global_data=sphere_b_base.global_data,
        )

        return sphere_a, sphere_b

    def test_point_to_points_disjoint(self, sphere_pair):
        """Verify point-to-points adjacency for disjoint meshes."""
        sphere_a, sphere_b = sphere_pair

        # Compute adjacency for individual meshes
        adj_a = sphere_a.get_point_to_points_adjacency()
        adj_b = sphere_b.get_point_to_points_adjacency()

        neighbors_a = adj_a.to_list()
        neighbors_b = adj_b.to_list()

        # Merge the meshes
        merged = Mesh.merge([sphere_a, sphere_b])
        adj_merged = merged.get_point_to_points_adjacency()
        neighbors_merged = adj_merged.to_list()

        # Validate merged connectivity
        n_points_a = sphere_a.n_points

        # Check sphere A's points in merged mesh (indices 0 to n_points_a-1)
        for i in range(n_points_a):
            expected = sorted(neighbors_a[i])
            actual = sorted(neighbors_merged[i])
            assert actual == expected, (
                f"Point {i} (sphere A) neighbors mismatch in merged mesh:\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )

        # Check sphere B's points in merged mesh (indices n_points_a onwards)
        for i in range(sphere_b.n_points):
            # Sphere B's neighbors should be offset by n_points_a
            expected = sorted([n + n_points_a for n in neighbors_b[i]])
            actual = sorted(neighbors_merged[i + n_points_a])
            assert actual == expected, (
                f"Point {i} (sphere B, index {i + n_points_a} in merged) neighbors mismatch:\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )

        # Verify no cross-mesh connections (critical for disjoint property)
        for i in range(n_points_a):
            for neighbor in neighbors_merged[i]:
                assert neighbor < n_points_a, (
                    f"Point {i} in sphere A has neighbor {neighbor} from sphere B (disjoint violation)"
                )

        for i in range(sphere_b.n_points):
            merged_idx = i + n_points_a
            for neighbor in neighbors_merged[merged_idx]:
                assert neighbor >= n_points_a, (
                    f"Point {merged_idx} in sphere B has neighbor {neighbor} from sphere A (disjoint violation)"
                )

    def test_cell_to_cells_disjoint(self, sphere_pair):
        """Verify cell-to-cells adjacency for disjoint meshes."""
        sphere_a, sphere_b = sphere_pair

        # Compute adjacency for individual meshes
        adj_a = sphere_a.get_cell_to_cells_adjacency(adjacency_codimension=1)
        adj_b = sphere_b.get_cell_to_cells_adjacency(adjacency_codimension=1)

        neighbors_a = adj_a.to_list()
        neighbors_b = adj_b.to_list()

        # Merge the meshes
        merged = Mesh.merge([sphere_a, sphere_b])
        adj_merged = merged.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors_merged = adj_merged.to_list()

        # Validate merged connectivity
        n_cells_a = sphere_a.n_cells

        # Check sphere A's cells in merged mesh
        for i in range(n_cells_a):
            expected = sorted(neighbors_a[i])
            actual = sorted(neighbors_merged[i])
            assert actual == expected, (
                f"Cell {i} (sphere A) neighbors mismatch in merged mesh:\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )

        # Check sphere B's cells in merged mesh
        for i in range(sphere_b.n_cells):
            # Sphere B's cell neighbors should be offset by n_cells_a
            expected = sorted([n + n_cells_a for n in neighbors_b[i]])
            actual = sorted(neighbors_merged[i + n_cells_a])
            assert actual == expected, (
                f"Cell {i} (sphere B, index {i + n_cells_a} in merged) neighbors mismatch:\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )

        # Verify no cross-mesh connections
        for i in range(n_cells_a):
            for neighbor in neighbors_merged[i]:
                assert neighbor < n_cells_a, (
                    f"Cell {i} in sphere A has neighbor {neighbor} from sphere B (disjoint violation)"
                )

        for i in range(sphere_b.n_cells):
            merged_idx = i + n_cells_a
            for neighbor in neighbors_merged[merged_idx]:
                assert neighbor >= n_cells_a, (
                    f"Cell {merged_idx} in sphere B has neighbor {neighbor} from sphere A (disjoint violation)"
                )

    def test_point_to_cells_disjoint(self, sphere_pair):
        """Verify point-to-cells adjacency for disjoint meshes."""
        sphere_a, sphere_b = sphere_pair

        # Compute adjacency for individual meshes
        adj_a = sphere_a.get_point_to_cells_adjacency()
        adj_b = sphere_b.get_point_to_cells_adjacency()

        stars_a = adj_a.to_list()
        stars_b = adj_b.to_list()

        # Merge the meshes
        merged = Mesh.merge([sphere_a, sphere_b])
        adj_merged = merged.get_point_to_cells_adjacency()
        stars_merged = adj_merged.to_list()

        # Validate merged connectivity
        n_points_a = sphere_a.n_points
        n_cells_a = sphere_a.n_cells

        # Check sphere A's points in merged mesh
        for i in range(n_points_a):
            expected = sorted(stars_a[i])
            actual = sorted(stars_merged[i])
            assert actual == expected, (
                f"Point {i} (sphere A) star mismatch in merged mesh:\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )

        # Check sphere B's points in merged mesh
        for i in range(sphere_b.n_points):
            # Sphere B's cell indices should be offset by n_cells_a
            expected = sorted([c + n_cells_a for c in stars_b[i]])
            actual = sorted(stars_merged[i + n_points_a])
            assert actual == expected, (
                f"Point {i} (sphere B, index {i + n_points_a} in merged) star mismatch:\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )

        # Verify no cross-mesh connections
        for i in range(n_points_a):
            for cell in stars_merged[i]:
                assert cell < n_cells_a, (
                    f"Point {i} in sphere A is in cell {cell} from sphere B (disjoint violation)"
                )

        for i in range(sphere_b.n_points):
            merged_idx = i + n_points_a
            for cell in stars_merged[merged_idx]:
                assert cell >= n_cells_a, (
                    f"Point {merged_idx} in sphere B is in cell {cell} from sphere A (disjoint violation)"
                )

    def test_cells_to_points_disjoint(self, sphere_pair):
        """Verify cells-to-points adjacency for disjoint meshes."""
        sphere_a, sphere_b = sphere_pair

        # Compute adjacency for individual meshes
        adj_a = sphere_a.get_cells_to_points_adjacency()
        adj_b = sphere_b.get_cells_to_points_adjacency()

        vertices_a = adj_a.to_list()
        vertices_b = adj_b.to_list()

        # Merge the meshes
        merged = Mesh.merge([sphere_a, sphere_b])
        adj_merged = merged.get_cells_to_points_adjacency()
        vertices_merged = adj_merged.to_list()

        # Validate merged connectivity
        n_points_a = sphere_a.n_points
        n_cells_a = sphere_a.n_cells

        # Check sphere A's cells in merged mesh
        for i in range(n_cells_a):
            expected = vertices_a[i]  # Order matters for cells-to-points
            actual = vertices_merged[i]
            assert actual == expected, (
                f"Cell {i} (sphere A) vertices mismatch in merged mesh:\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )

        # Check sphere B's cells in merged mesh
        for i in range(sphere_b.n_cells):
            # Sphere B's point indices should be offset by n_points_a
            expected = [v + n_points_a for v in vertices_b[i]]
            actual = vertices_merged[i + n_cells_a]
            assert actual == expected, (
                f"Cell {i} (sphere B, index {i + n_cells_a} in merged) vertices mismatch:\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )

        # Verify no cross-mesh vertex references
        for i in range(n_cells_a):
            for vertex in vertices_merged[i]:
                assert vertex < n_points_a, (
                    f"Cell {i} in sphere A references vertex {vertex} from sphere B (disjoint violation)"
                )

        for i in range(sphere_b.n_cells):
            merged_idx = i + n_cells_a
            for vertex in vertices_merged[merged_idx]:
                assert vertex >= n_points_a, (
                    f"Cell {merged_idx} in sphere B references vertex {vertex} from sphere A (disjoint violation)"
                )


class TestNeighborTransformationInvariance:
    """Test that neighbor computation is invariant under geometric transformations.

    Verifies that translation, rotation, and reflection preserve topological
    connectivity, as they should since these operations don't change mesh topology.
    """

    @pytest.fixture
    def sphere_mesh(self, device):
        """Create a sphere mesh for transformation testing."""
        from physicsnemo.mesh.primitives.surfaces.sphere_icosahedral import (
            load as load_sphere,
        )

        return load_sphere(radius=1.0, subdivisions=2, device=device)

    def _create_rotation_matrix(
        self, axis: torch.Tensor, angle_rad: float
    ) -> torch.Tensor:
        """Create a 3D rotation matrix using Rodrigues' rotation formula.

        Args:
            axis: Rotation axis (will be normalized), shape (3,)
            angle_rad: Rotation angle in radians

        Returns:
            Rotation matrix, shape (3, 3)
        """
        # Normalize axis
        axis = axis / torch.norm(axis)
        x, y, z = axis[0], axis[1], axis[2]

        c = torch.cos(torch.tensor(angle_rad, device=axis.device))
        s = torch.sin(torch.tensor(angle_rad, device=axis.device))
        t = 1 - c

        # Rodrigues' rotation matrix
        rotation = torch.tensor(
            [
                [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
            ],
            device=axis.device,
            dtype=axis.dtype,
        )

        return rotation

    def _create_reflection_matrix(self, normal: torch.Tensor) -> torch.Tensor:
        """Create a 3D reflection matrix across a plane.

        Args:
            normal: Plane normal vector (will be normalized), shape (3,)

        Returns:
            Reflection matrix, shape (3, 3)
        """
        # Normalize normal
        n = normal / torch.norm(normal)

        # Householder reflection: I - 2*n*n^T
        reflection = torch.eye(3, device=n.device, dtype=n.dtype) - 2 * torch.outer(
            n, n
        )

        return reflection

    def test_translation_invariance_point_to_points(self, sphere_mesh):
        """Verify point-to-points adjacency is invariant under translation."""
        original = sphere_mesh

        # Compute adjacency for original mesh
        adj_original = original.get_point_to_points_adjacency()
        neighbors_original = adj_original.to_list()

        # Translate by arbitrary vector
        translation = torch.tensor([10.0, -5.0, 7.5], device=original.points.device)
        translated = Mesh(
            points=original.points + translation,
            cells=original.cells,
            point_data=original.point_data,
            cell_data=original.cell_data,
            global_data=original.global_data,
        )

        # Compute adjacency for translated mesh
        adj_translated = translated.get_point_to_points_adjacency()
        neighbors_translated = adj_translated.to_list()

        # Connectivity should be identical
        assert neighbors_original == neighbors_translated, (
            "Translation changed point-to-points connectivity (topology violation)"
        )

    def test_rotation_invariance_point_to_points(self, sphere_mesh):
        """Verify point-to-points adjacency is invariant under rotation."""
        original = sphere_mesh

        # Compute adjacency for original mesh
        adj_original = original.get_point_to_points_adjacency()
        neighbors_original = adj_original.to_list()

        # Rotate by 45 degrees around arbitrary axis [1, 1, 1]
        axis = torch.tensor([1.0, 1.0, 1.0], device=original.points.device)
        angle = torch.pi / 4
        rotation_matrix = self._create_rotation_matrix(axis, angle)

        rotated_points = torch.matmul(original.points, rotation_matrix.T)
        rotated = Mesh(
            points=rotated_points,
            cells=original.cells,
            point_data=original.point_data,
            cell_data=original.cell_data,
            global_data=original.global_data,
        )

        # Compute adjacency for rotated mesh
        adj_rotated = rotated.get_point_to_points_adjacency()
        neighbors_rotated = adj_rotated.to_list()

        # Connectivity should be identical
        assert neighbors_original == neighbors_rotated, (
            "Rotation changed point-to-points connectivity (topology violation)"
        )

    def test_reflection_invariance_point_to_points(self, sphere_mesh):
        """Verify point-to-points adjacency is invariant under reflection."""
        original = sphere_mesh

        # Compute adjacency for original mesh
        adj_original = original.get_point_to_points_adjacency()
        neighbors_original = adj_original.to_list()

        # Reflect across plane with normal [1, 0, 0] (yz-plane)
        normal = torch.tensor([1.0, 0.0, 0.0], device=original.points.device)
        reflection_matrix = self._create_reflection_matrix(normal)

        reflected_points = torch.matmul(original.points, reflection_matrix.T)
        reflected = Mesh(
            points=reflected_points,
            cells=original.cells,
            point_data=original.point_data,
            cell_data=original.cell_data,
            global_data=original.global_data,
        )

        # Compute adjacency for reflected mesh
        adj_reflected = reflected.get_point_to_points_adjacency()
        neighbors_reflected = adj_reflected.to_list()

        # Connectivity should be identical
        assert neighbors_original == neighbors_reflected, (
            "Reflection changed point-to-points connectivity (topology violation)"
        )

    def test_translation_invariance_cell_to_cells(self, sphere_mesh):
        """Verify cell-to-cells adjacency is invariant under translation."""
        original = sphere_mesh

        # Compute adjacency for original mesh
        adj_original = original.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors_original = adj_original.to_list()

        # Translate by arbitrary vector
        translation = torch.tensor([10.0, -5.0, 7.5], device=original.points.device)
        translated = Mesh(
            points=original.points + translation,
            cells=original.cells,
            point_data=original.point_data,
            cell_data=original.cell_data,
            global_data=original.global_data,
        )

        # Compute adjacency for translated mesh
        adj_translated = translated.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors_translated = adj_translated.to_list()

        # Connectivity should be identical
        assert neighbors_original == neighbors_translated, (
            "Translation changed cell-to-cells connectivity (topology violation)"
        )

    def test_rotation_invariance_cell_to_cells(self, sphere_mesh):
        """Verify cell-to-cells adjacency is invariant under rotation."""
        original = sphere_mesh

        # Compute adjacency for original mesh
        adj_original = original.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors_original = adj_original.to_list()

        # Rotate by 60 degrees around z-axis
        axis = torch.tensor([0.0, 0.0, 1.0], device=original.points.device)
        angle = torch.pi / 3
        rotation_matrix = self._create_rotation_matrix(axis, angle)

        rotated_points = torch.matmul(original.points, rotation_matrix.T)
        rotated = Mesh(
            points=rotated_points,
            cells=original.cells,
            point_data=original.point_data,
            cell_data=original.cell_data,
            global_data=original.global_data,
        )

        # Compute adjacency for rotated mesh
        adj_rotated = rotated.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors_rotated = adj_rotated.to_list()

        # Connectivity should be identical
        assert neighbors_original == neighbors_rotated, (
            "Rotation changed cell-to-cells connectivity (topology violation)"
        )

    def test_reflection_invariance_cell_to_cells(self, sphere_mesh):
        """Verify cell-to-cells adjacency is invariant under reflection."""
        original = sphere_mesh

        # Compute adjacency for original mesh
        adj_original = original.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors_original = adj_original.to_list()

        # Reflect across xy-plane (normal [0, 0, 1])
        normal = torch.tensor([0.0, 0.0, 1.0], device=original.points.device)
        reflection_matrix = self._create_reflection_matrix(normal)

        reflected_points = torch.matmul(original.points, reflection_matrix.T)
        reflected = Mesh(
            points=reflected_points,
            cells=original.cells,
            point_data=original.point_data,
            cell_data=original.cell_data,
            global_data=original.global_data,
        )

        # Compute adjacency for reflected mesh
        adj_reflected = reflected.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors_reflected = adj_reflected.to_list()

        # Connectivity should be identical
        assert neighbors_original == neighbors_reflected, (
            "Reflection changed cell-to-cells connectivity (topology violation)"
        )

    def test_translation_invariance_point_to_cells(self, sphere_mesh):
        """Verify point-to-cells adjacency is invariant under translation."""
        original = sphere_mesh

        # Compute adjacency for original mesh
        adj_original = original.get_point_to_cells_adjacency()
        stars_original = adj_original.to_list()

        # Translate by arbitrary vector
        translation = torch.tensor([10.0, -5.0, 7.5], device=original.points.device)
        translated = Mesh(
            points=original.points + translation,
            cells=original.cells,
            point_data=original.point_data,
            cell_data=original.cell_data,
            global_data=original.global_data,
        )

        # Compute adjacency for translated mesh
        adj_translated = translated.get_point_to_cells_adjacency()
        stars_translated = adj_translated.to_list()

        # Connectivity should be identical
        assert stars_original == stars_translated, (
            "Translation changed point-to-cells connectivity (topology violation)"
        )

    def test_rotation_invariance_point_to_cells(self, sphere_mesh):
        """Verify point-to-cells adjacency is invariant under rotation."""
        original = sphere_mesh

        # Compute adjacency for original mesh
        adj_original = original.get_point_to_cells_adjacency()
        stars_original = adj_original.to_list()

        # Rotate by 30 degrees around x-axis
        axis = torch.tensor([1.0, 0.0, 0.0], device=original.points.device)
        angle = torch.pi / 6
        rotation_matrix = self._create_rotation_matrix(axis, angle)

        rotated_points = torch.matmul(original.points, rotation_matrix.T)
        rotated = Mesh(
            points=rotated_points,
            cells=original.cells,
            point_data=original.point_data,
            cell_data=original.cell_data,
            global_data=original.global_data,
        )

        # Compute adjacency for rotated mesh
        adj_rotated = rotated.get_point_to_cells_adjacency()
        stars_rotated = adj_rotated.to_list()

        # Connectivity should be identical
        assert stars_original == stars_rotated, (
            "Rotation changed point-to-cells connectivity (topology violation)"
        )

    def test_reflection_invariance_point_to_cells(self, sphere_mesh):
        """Verify point-to-cells adjacency is invariant under reflection."""
        original = sphere_mesh

        # Compute adjacency for original mesh
        adj_original = original.get_point_to_cells_adjacency()
        stars_original = adj_original.to_list()

        # Reflect across xz-plane (normal [0, 1, 0])
        normal = torch.tensor([0.0, 1.0, 0.0], device=original.points.device)
        reflection_matrix = self._create_reflection_matrix(normal)

        reflected_points = torch.matmul(original.points, reflection_matrix.T)
        reflected = Mesh(
            points=reflected_points,
            cells=original.cells,
            point_data=original.point_data,
            cell_data=original.cell_data,
            global_data=original.global_data,
        )

        # Compute adjacency for reflected mesh
        adj_reflected = reflected.get_point_to_cells_adjacency()
        stars_reflected = adj_reflected.to_list()

        # Connectivity should be identical
        assert stars_original == stars_reflected, (
            "Reflection changed point-to-cells connectivity (topology violation)"
        )

    def test_translation_invariance_cells_to_points(self, sphere_mesh):
        """Verify cells-to-points adjacency is invariant under translation."""
        original = sphere_mesh

        # Compute adjacency for original mesh
        adj_original = original.get_cells_to_points_adjacency()
        vertices_original = adj_original.to_list()

        # Translate by arbitrary vector
        translation = torch.tensor([10.0, -5.0, 7.5], device=original.points.device)
        translated = Mesh(
            points=original.points + translation,
            cells=original.cells,
            point_data=original.point_data,
            cell_data=original.cell_data,
            global_data=original.global_data,
        )

        # Compute adjacency for translated mesh
        adj_translated = translated.get_cells_to_points_adjacency()
        vertices_translated = adj_translated.to_list()

        # Connectivity should be identical
        assert vertices_original == vertices_translated, (
            "Translation changed cells-to-points connectivity (topology violation)"
        )

    def test_rotation_invariance_cells_to_points(self, sphere_mesh):
        """Verify cells-to-points adjacency is invariant under rotation."""
        original = sphere_mesh

        # Compute adjacency for original mesh
        adj_original = original.get_cells_to_points_adjacency()
        vertices_original = adj_original.to_list()

        # Rotate by 90 degrees around y-axis
        axis = torch.tensor([0.0, 1.0, 0.0], device=original.points.device)
        angle = torch.pi / 2
        rotation_matrix = self._create_rotation_matrix(axis, angle)

        rotated_points = torch.matmul(original.points, rotation_matrix.T)
        rotated = Mesh(
            points=rotated_points,
            cells=original.cells,
            point_data=original.point_data,
            cell_data=original.cell_data,
            global_data=original.global_data,
        )

        # Compute adjacency for rotated mesh
        adj_rotated = rotated.get_cells_to_points_adjacency()
        vertices_rotated = adj_rotated.to_list()

        # Connectivity should be identical
        assert vertices_original == vertices_rotated, (
            "Rotation changed cells-to-points connectivity (topology violation)"
        )

    def test_reflection_invariance_cells_to_points(self, sphere_mesh):
        """Verify cells-to-points adjacency is invariant under reflection."""
        original = sphere_mesh

        # Compute adjacency for original mesh
        adj_original = original.get_cells_to_points_adjacency()
        vertices_original = adj_original.to_list()

        # Reflect across arbitrary plane with normal [1, 1, 1]
        normal = torch.tensor([1.0, 1.0, 1.0], device=original.points.device)
        reflection_matrix = self._create_reflection_matrix(normal)

        reflected_points = torch.matmul(original.points, reflection_matrix.T)
        reflected = Mesh(
            points=reflected_points,
            cells=original.cells,
            point_data=original.point_data,
            cell_data=original.cell_data,
            global_data=original.global_data,
        )

        # Compute adjacency for reflected mesh
        adj_reflected = reflected.get_cells_to_points_adjacency()
        vertices_reflected = adj_reflected.to_list()

        # Connectivity should be identical
        assert vertices_original == vertices_reflected, (
            "Reflection changed cells-to-points connectivity (topology violation)"
        )
