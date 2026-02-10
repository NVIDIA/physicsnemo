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

"""Tests for neighbor and adjacency computation.

Tests validate physicsnemo.mesh adjacency computations against PyVista's VTK-based
implementations as ground truth, and verify correctness across spatial dimensions,
manifold dimensions, and compute backends.
"""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh

# PyVista is optional; tests that cross-validate against it are skipped if unavailable
pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io.io_pyvista import from_pyvista  # noqa: E402

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


def _compute_adjacency(mesh, adj_type):
    """Compute adjacency of the given type on a mesh.

    Helper used by disjoint-mesh and transformation-invariance tests.
    """
    if adj_type == "point_to_points":
        return mesh.get_point_to_points_adjacency()
    elif adj_type == "cell_to_cells":
        return mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
    elif adj_type == "point_to_cells":
        return mesh.get_point_to_cells_adjacency()
    elif adj_type == "cells_to_points":
        return mesh.get_cell_to_points_adjacency()
    else:
        raise ValueError(f"Unknown adjacency type: {adj_type}")


### Parametrization Constants ###

# Common dimension configs for adjacency tests (excludes 0-manifold point clouds)
_SIMPLEX_DIM_CONFIGS = [
    (2, 1),  # Edges in 2D
    (2, 2),  # Triangles in 2D
    (3, 1),  # Edges in 3D
    (3, 2),  # Surfaces in 3D
    (3, 3),  # Volumes in 3D
]

# All four adjacency types for parametrized tests
_ADJACENCY_TYPES = [
    pytest.param("point_to_points", id="point_to_points"),
    pytest.param("cell_to_cells", id="cell_to_cells"),
    pytest.param("point_to_cells", id="point_to_cells"),
    pytest.param("cells_to_points", id="cells_to_points"),
]

# Geometric transforms for invariance tests
_TRANSFORMS = [
    pytest.param("translation", id="translation"),
    pytest.param("rotation", id="rotation"),
    pytest.param("reflection", id="reflection"),
]

### Test Fixtures ###


@pytest.fixture(
    params=[
        pytest.param("airplane", id="airplane"),
        pytest.param("tetbeam", id="tetbeam"),
    ]
)
def real_mesh_pair(request, device):
    """Real-world mesh pair (physicsnemo + pyvista) for cross-validation.

    Parametrized over airplane (2D surface in 3D) and tetbeam (3D volume) meshes.
    """
    if request.param == "airplane":
        pv_mesh = pv.examples.load_airplane()
    else:
        pv_mesh = pv.examples.load_tetbeam()
    tm_mesh = from_pyvista(pv_mesh)
    tm_mesh = Mesh(
        points=tm_mesh.points.to(device),
        cells=tm_mesh.cells.to(device),
        point_data=tm_mesh.point_data,
        cell_data=tm_mesh.cell_data,
    )
    return tm_mesh, pv_mesh


@pytest.fixture
def simple_triangles(device):
    """Simple triangle mesh for basic testing (shared across test classes)."""
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


class TestPointToPointsAdjacency:
    """Test point-to-points (edge) adjacency computation."""

    ### Cross-validation against PyVista ###

    def test_cross_validation_point_neighbors(self, real_mesh_pair):
        """Validate point-to-points adjacency against PyVista."""
        tm_mesh, pv_mesh = real_mesh_pair
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
            tm_sorted = sorted(tm_nbrs)
            pv_sorted = sorted(pv_nbrs)
            assert tm_sorted == pv_sorted, (
                f"Point {i} neighbors mismatch:\n  physicsnemo.mesh: {tm_sorted}\n  pyvista:   {pv_sorted}"
            )

    ### Symmetry Tests ###

    def test_symmetry_real_mesh(self, real_mesh_pair):
        """Verify point adjacency is symmetric on real-world meshes."""
        tm_mesh, _ = real_mesh_pair

        adj = tm_mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                # If j is a neighbor of i, then i must be a neighbor of j
                assert i in neighbors[j], (
                    f"Asymmetric adjacency: {i} neighbors {j}, but {j} doesn't neighbor {i}"
                )

    ### Parametrized Tests on Synthetic Meshes (Exhaustive Dimensional Coverage) ###

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", _SIMPLEX_DIM_CONFIGS)
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

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", _SIMPLEX_DIM_CONFIGS)
    def test_no_self_loops_parametrized(self, n_spatial_dims, n_manifold_dims, device):
        """Verify no point is its own neighbor across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        adj = mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            assert i not in nbrs, (
                f"Point {i} is listed as its own neighbor ({n_spatial_dims=}, {n_manifold_dims=})"
            )

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", _SIMPLEX_DIM_CONFIGS)
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

    def test_cross_validation_cell_neighbors(self, real_mesh_pair):
        """Validate cell-to-cells adjacency against PyVista."""
        tm_mesh, pv_mesh = real_mesh_pair
        device = tm_mesh.points.device.type

        ### Compute adjacency using physicsnemo.mesh
        # Codimension=1: sharing edges for triangles, sharing faces for tets
        adj = tm_mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        assert_on_device(adj.offsets, device)
        assert_on_device(adj.indices, device)

        tm_neighbors = adj.to_list()

        ### Get ground truth from PyVista
        # Map codimension=1 to PyVista connection type based on manifold dimension
        pv_connection = {2: "edges", 3: "faces"}[tm_mesh.n_manifold_dims]
        pv_neighbors = []
        for i in range(pv_mesh.n_cells):
            neighbors = pv_mesh.cell_neighbors(i, connections=pv_connection)
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

    ### Symmetry Tests ###

    def test_symmetry_real_mesh(self, real_mesh_pair):
        """Verify cell adjacency is symmetric on real-world meshes."""
        tm_mesh, _ = real_mesh_pair

        adj = tm_mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                assert i in neighbors[j], (
                    f"Asymmetric adjacency: cell {i} neighbors cell {j}, "
                    f"but cell {j} doesn't neighbor cell {i}"
                )

    ### Parametrized Tests on Synthetic Meshes (Exhaustive Dimensional Coverage) ###

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", _SIMPLEX_DIM_CONFIGS)
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

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", _SIMPLEX_DIM_CONFIGS)
    def test_no_self_loops_parametrized(self, n_spatial_dims, n_manifold_dims, device):
        """Verify no cell is its own neighbor across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            assert i not in nbrs, (
                f"Cell {i} is listed as its own neighbor ({n_spatial_dims=}, {n_manifold_dims=})"
            )

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", _SIMPLEX_DIM_CONFIGS)
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

    def test_consistency_real_mesh(self, real_mesh_pair):
        """Verify consistency of point-to-cells adjacency on real-world meshes."""
        tm_mesh, _ = real_mesh_pair

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

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", _SIMPLEX_DIM_CONFIGS)
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

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", _SIMPLEX_DIM_CONFIGS)
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

    def test_simple_triangle_vertices(self, simple_triangles):
        """Test cells-to-points on simple triangle mesh."""
        mesh = simple_triangles
        device = mesh.points.device.type

        adj = mesh.get_cell_to_points_adjacency()
        assert_on_device(adj.offsets, device)
        assert_on_device(adj.indices, device)

        vertices = adj.to_list()

        # Cell 0 has vertices [0, 1, 2]
        assert vertices[0] == [0, 1, 2]

        # Cell 1 has vertices [1, 3, 2]
        assert vertices[1] == [1, 3, 2]

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", _SIMPLEX_DIM_CONFIGS)
    def test_matches_cells_array_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Verify cells-to-points matches the cells array across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        adj = mesh.get_cell_to_points_adjacency()
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

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", _SIMPLEX_DIM_CONFIGS)
    def test_all_cells_same_size_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Verify all cells have the correct number of vertices."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        adj = mesh.get_cell_to_points_adjacency()
        vertices = adj.to_list()

        # All cells should have (n_manifold_dims + 1) vertices
        expected_size = n_manifold_dims + 1
        for i, verts in enumerate(vertices):
            assert len(verts) == expected_size, (
                f"Cell {i} has {len(verts)} vertices, expected {expected_size} "
                f"({n_spatial_dims=}, {n_manifold_dims=})"
            )

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", _SIMPLEX_DIM_CONFIGS)
    def test_inverse_of_point_to_cells_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Verify cells-to-points is inverse of point-to-cells."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        # Get both adjacencies
        cells_to_points = mesh.get_cell_to_points_adjacency().to_list()
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
        adj = mesh.get_cell_to_points_adjacency()
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
            mesh.get_cell_to_points_adjacency(),
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
        cells_to_points = mesh.get_cell_to_points_adjacency()
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
        cells_to_points = mesh.get_cell_to_points_adjacency().to_list()
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

    @staticmethod
    def _disjoint_offsets(sphere_a, adj_type):
        """Return (n_sources_a, n_targets_a) for a disjoint merge.

        These determine how indices are offset in the merged mesh:
        - n_sources_a: number of source entities in sphere A (index offset for B's sources)
        - n_targets_a: number of target entities in sphere A (value offset for B's targets)
        """
        if adj_type == "point_to_points":
            return sphere_a.n_points, sphere_a.n_points
        elif adj_type == "cell_to_cells":
            return sphere_a.n_cells, sphere_a.n_cells
        elif adj_type == "point_to_cells":
            return sphere_a.n_points, sphere_a.n_cells
        elif adj_type == "cells_to_points":
            return sphere_a.n_cells, sphere_a.n_points
        raise ValueError(f"Unknown adj_type: {adj_type}")

    @pytest.mark.parametrize("adj_type", _ADJACENCY_TYPES)
    def test_disjoint_adjacency(self, sphere_pair, adj_type):
        """Verify adjacency for disjoint meshes preserves individual connectivity."""
        sphere_a, sphere_b = sphere_pair

        # Compute adjacency for individual meshes
        adj_a = _compute_adjacency(sphere_a, adj_type)
        adj_b = _compute_adjacency(sphere_b, adj_type)

        results_a = adj_a.to_list()
        results_b = adj_b.to_list()

        # Merge the meshes
        merged = Mesh.merge([sphere_a, sphere_b])
        adj_merged = _compute_adjacency(merged, adj_type)
        results_merged = adj_merged.to_list()

        n_sources_a, n_targets_a = self._disjoint_offsets(sphere_a, adj_type)

        # cells_to_points preserves vertex ordering; others use set comparison
        order_sensitive = adj_type == "cells_to_points"
        normalize = (lambda x: x) if order_sensitive else sorted

        # Check sphere A's entries in merged mesh
        for i in range(n_sources_a):
            expected = normalize(results_a[i])
            actual = normalize(results_merged[i])
            assert actual == expected, (
                f"Source {i} (sphere A) {adj_type} mismatch in merged mesh:\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )

        # Check sphere B's entries (targets offset by n_targets_a)
        for i in range(len(results_b)):
            expected = normalize([n + n_targets_a for n in results_b[i]])
            actual = normalize(results_merged[i + n_sources_a])
            assert actual == expected, (
                f"Source {i} (sphere B, index {i + n_sources_a} in merged) "
                f"{adj_type} mismatch:\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )

        # Verify no cross-mesh connections (critical for disjoint property)
        for i in range(n_sources_a):
            for target in results_merged[i]:
                assert target < n_targets_a, (
                    f"Source {i} in sphere A has target {target} from sphere B "
                    f"({adj_type} disjoint violation)"
                )

        for i in range(len(results_b)):
            merged_idx = i + n_sources_a
            for target in results_merged[merged_idx]:
                assert target >= n_targets_a, (
                    f"Source {merged_idx} in sphere B has target {target} "
                    f"from sphere A ({adj_type} disjoint violation)"
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

    def _apply_transform(self, points: torch.Tensor, transform: str) -> torch.Tensor:
        """Apply a geometric transformation to mesh points.

        Args:
            points: Point coordinates, shape (N, 3)
            transform: One of "translation", "rotation", "reflection"

        Returns:
            Transformed points, shape (N, 3)
        """
        if transform == "translation":
            translation = torch.tensor([10.0, -5.0, 7.5], device=points.device)
            return points + translation
        elif transform == "rotation":
            axis = torch.tensor([1.0, 1.0, 1.0], device=points.device)
            angle = torch.pi / 4
            rotation_matrix = self._create_rotation_matrix(axis, angle)
            return torch.matmul(points, rotation_matrix.T)
        elif transform == "reflection":
            normal = torch.tensor([1.0, 0.0, 0.0], device=points.device)
            reflection_matrix = self._create_reflection_matrix(normal)
            return torch.matmul(points, reflection_matrix.T)
        else:
            raise ValueError(f"Unknown transform: {transform}")

    @pytest.mark.parametrize("adj_type", _ADJACENCY_TYPES)
    @pytest.mark.parametrize("transform", _TRANSFORMS)
    def test_adjacency_transformation_invariant(self, sphere_mesh, adj_type, transform):
        """Verify adjacency is invariant under geometric transformations."""
        original = sphere_mesh

        # Compute adjacency for original mesh
        adj_original = _compute_adjacency(original, adj_type)
        results_original = adj_original.to_list()

        # Apply transformation
        transformed_points = self._apply_transform(original.points, transform)
        transformed = Mesh(
            points=transformed_points,
            cells=original.cells,
            point_data=original.point_data,
            cell_data=original.cell_data,
            global_data=original.global_data,
        )

        # Compute adjacency for transformed mesh
        adj_transformed = _compute_adjacency(transformed, adj_type)
        results_transformed = adj_transformed.to_list()

        # Connectivity should be identical
        assert results_original == results_transformed, (
            f"{transform.title()} changed {adj_type} connectivity (topology violation)"
        )
