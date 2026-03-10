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

"""Tests for physicsnemo.mesh.io module - 3D mesh conversion."""

import numpy as np
import pytest
import torch

pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io.io_pyvista import from_pyvista  # noqa: E402


class TestFromPyvista3D:
    """Tests for converting 3D (volume) meshes."""

    def test_tetbeam_mesh_auto_detection(self):
        """Test automatic detection of 3D manifold from tetbeam mesh."""
        pv_mesh = pv.examples.load_tetbeam()

        # Verify it's all tetrahedral cells
        assert list(pv_mesh.cells_dict.keys()) == [pv.CellType.TETRA]

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 3
        assert mesh.n_spatial_dims == 3
        assert mesh.cells.shape[1] == 4  # Tetrahedral cells
        assert mesh.n_points == pv_mesh.n_points
        assert mesh.n_cells == pv_mesh.n_cells

    def test_tetbeam_mesh_explicit_dim(self):
        """Test explicit manifold_dim specification for 3D mesh."""
        pv_mesh = pv.examples.load_tetbeam()

        mesh = from_pyvista(pv_mesh, manifold_dim=3)

        assert mesh.n_manifold_dims == 3
        assert mesh.cells.shape[1] == 4

    def test_hexbeam_mesh_triangulation(self):
        """Test automatic triangulation of hexahedral mesh to tetrahedral.

        The hexbeam mesh contains hexahedral cells which must be converted
        to tetrahedral cells for our simplex-based mesh representation.
        """
        pv_mesh = pv.examples.load_hexbeam()

        # Verify it contains hexahedral cells (not tetrahedral)
        assert pv.CellType.HEXAHEDRON in pv_mesh.celltypes
        assert pv.CellType.TETRA not in pv_mesh.celltypes
        original_n_points = pv_mesh.n_points

        # Convert - should automatically triangulate to tetrahedra
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 3
        assert mesh.n_spatial_dims == 3
        assert mesh.cells.shape[1] == 4  # Tetrahedral cells
        # triangulate() does not add new vertices
        assert mesh.n_points == original_n_points
        # Each hexahedron is decomposed into at least 5 tetrahedra
        assert mesh.n_cells >= 5 * pv_mesh.n_cells

    def test_simple_tetrahedron(self):
        """Test conversion of a single tetrahedron."""
        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        cells = np.array([4, 0, 1, 2, 3])
        celltypes = np.array([pv.CellType.TETRA])

        pv_mesh = pv.UnstructuredGrid(cells, celltypes, points)
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 3
        assert mesh.n_points == 4
        assert mesh.n_cells == 1
        assert mesh.cells.shape == (1, 4)

        # Verify the face connectivity is correct
        expected_cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        assert torch.equal(mesh.cells, expected_cells)


def _make_pentagonal_prism() -> "pv.UnstructuredGrid":
    """Create a pentagonal-prism polyhedron (10 vertices, 7 faces).

    This is a nontrivial VTK_POLYHEDRON cell that exercises the
    variable-length connectivity path that breaks ``cells_dict``.
    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.866, 0.0],
            [0.5, 1.366, 0.0],
            [-0.5, 0.866, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [1.5, 0.866, 2.0],
            [0.5, 1.366, 2.0],
            [-0.5, 0.866, 2.0],
        ],
        dtype=np.float64,
    )
    # fmt: off
    connectivity = [
        7,                        # number of faces
        5, 0, 1, 2, 3, 4,        # bottom pentagon
        4, 0, 1, 6, 5,           # side 1
        4, 1, 2, 7, 6,           # side 2
        4, 2, 3, 8, 7,           # side 3
        4, 3, 4, 9, 8,           # side 4
        4, 4, 0, 5, 9,           # side 5
        5, 5, 6, 7, 8, 9,        # top pentagon
    ]
    # fmt: on
    cell = [len(connectivity), *connectivity]
    return pv.UnstructuredGrid(cell, [pv.CellType.POLYHEDRON], points)


class TestFromPyvistaPolyhedra:
    """Tests for converting meshes with VTK_POLYHEDRON cells."""

    def test_polyhedron_auto_detection(self):
        """Test that a polyhedron-only grid is detected as 3D."""
        pv_mesh = _make_pentagonal_prism()
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 3
        assert mesh.n_spatial_dims == 3
        assert mesh.cells.shape[1] == 4
        assert mesh.n_points == 10
        assert mesh.n_cells > 1  # prism decomposes into multiple tets

    def test_polyhedron_explicit_dim(self):
        """Test explicit manifold_dim=3 with polyhedra."""
        pv_mesh = _make_pentagonal_prism()
        mesh = from_pyvista(pv_mesh, manifold_dim=3)

        assert mesh.n_manifold_dims == 3
        assert mesh.cells.shape[1] == 4
        assert mesh.n_points == 10

    def test_polyhedron_preserves_point_data(self):
        """Test that point data survives polyhedra triangulation."""
        pv_mesh = _make_pentagonal_prism()
        pv_mesh.point_data["temperature"] = np.arange(10, dtype=np.float32)

        mesh = from_pyvista(pv_mesh, manifold_dim=3)

        assert "temperature" in mesh.point_data
        assert torch.allclose(
            mesh.point_data["temperature"],
            torch.arange(10, dtype=torch.float32),
        )

    def test_mixed_hex_and_polyhedra(self):
        """Test a mesh with both hexahedral and polyhedral cells.

        This is the realistic CFD scenario: volume meshes from solvers
        like Cadence often contain both standard and polyhedral cells.
        """
        # Hexahedron (vertices 0-7)
        hex_pts = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float64,
        )
        # Pentagonal prism polyhedron (vertices 8-17)
        poly_pts = np.array(
            [
                [2, 0, 0],
                [3, 0, 0],
                [3.5, 0.5, 0],
                [3, 1, 0],
                [2, 1, 0],
                [2, 0, 1],
                [3, 0, 1],
                [3.5, 0.5, 1],
                [3, 1, 1],
                [2, 1, 1],
            ],
            dtype=np.float64,
        )
        all_pts = np.vstack([hex_pts, poly_pts])

        hex_cell = [8, 0, 1, 2, 3, 4, 5, 6, 7]
        # fmt: off
        poly_conn = [
            7,
            5, 8, 9, 10, 11, 12,
            4, 8, 9, 14, 13,
            4, 9, 10, 15, 14,
            4, 10, 11, 16, 15,
            4, 11, 12, 17, 16,
            4, 12, 8, 13, 17,
            5, 13, 14, 15, 16, 17,
        ]
        # fmt: on
        poly_cell = [len(poly_conn), *poly_conn]

        cells = hex_cell + poly_cell
        celltypes = [pv.CellType.HEXAHEDRON, pv.CellType.POLYHEDRON]
        pv_mesh = pv.UnstructuredGrid(cells, celltypes, all_pts)

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 3
        assert mesh.cells.shape[1] == 4
        assert mesh.n_points == 18
        assert mesh.n_cells > 2  # both cells decompose into multiple tets

    def test_polyhedron_geometric_invariants(self):
        """Verify that volume, bounding box, surface area, and manifold
        property are all preserved through the from_pyvista conversion."""
        from physicsnemo.mesh.io import to_pyvista

        pv_mesh = _make_pentagonal_prism()
        mesh = from_pyvista(pv_mesh, manifold_dim=3)

        ### Bounding box preservation (triangulate adds no vertices)
        original_bounds = np.array(pv_mesh.bounds).reshape(3, 2)
        pts = mesh.points.numpy()
        result_bounds = np.column_stack([pts.min(axis=0), pts.max(axis=0)])
        np.testing.assert_allclose(result_bounds, original_bounds, atol=1e-5)

        ### Volume preservation via signed-tet determinant formula
        cells = mesh.cells
        v0 = mesh.points[cells[:, 0]]
        edges = mesh.points[cells[:, 1:]] - v0.unsqueeze(1)  # (n_tets, 3, 3)
        cross = torch.cross(edges[:, 1], edges[:, 2], dim=1)
        tet_volumes = (edges[:, 0] * cross).sum(dim=1).abs() / 6.0
        total_volume = tet_volumes.sum().double().item()
        expected_volume = pv_mesh.triangulate().volume
        assert abs(total_volume - expected_volume) / expected_volume < 1e-5

        ### Manifold check: the boundary surface must be watertight
        pv_result = to_pyvista(mesh)
        surface = pv_result.extract_surface(algorithm=None)
        assert surface.is_manifold

        ### Surface area preservation
        original_area = pv_mesh.extract_surface(algorithm=None).area
        result_area = surface.area
        assert abs(result_area - original_area) / original_area < 1e-5

    def test_polyhedron_face_areas_decompose(self):
        """Each original polyhedron face area must equal the sum of the
        coplanar boundary-triangle areas in the resulting tet mesh.

        This checks that the triangulation exactly tiles each original
        face, not just that the total surface area is preserved.
        """
        from collections import defaultdict

        from physicsnemo.mesh.io import to_pyvista

        pv_mesh = _make_pentagonal_prism()

        ### Compute original face areas by fan-triangulating each face
        cell_obj = pv_mesh.get_cell(0)
        pts_f64 = pv_mesh.points
        original_face_areas: list[float] = []
        for i in range(cell_obj.n_faces):
            face = cell_obj.get_face(i)
            fp = pts_f64[face.point_ids]
            area = sum(
                0.5 * np.linalg.norm(np.cross(fp[j] - fp[0], fp[j + 1] - fp[0]))
                for j in range(1, len(fp) - 1)
            )
            original_face_areas.append(area)

        ### Convert through from_pyvista and extract boundary surface
        mesh = from_pyvista(pv_mesh, manifold_dim=3)
        pv_result = to_pyvista(mesh)
        surface = pv_result.extract_surface(algorithm=None)

        ### Compute unit normals and signed plane-distances for each
        ### boundary triangle, then group coplanar triangles together.
        tri_faces = surface.regular_faces  # (n_tri, 3)
        tri_pts = surface.points[tri_faces]  # (n_tri, 3, 3)
        v1 = tri_pts[:, 1] - tri_pts[:, 0]
        v2 = tri_pts[:, 2] - tri_pts[:, 0]
        normals = np.cross(v1, v2)
        areas = 0.5 * np.linalg.norm(normals, axis=1)
        unit_normals = normals / (2 * areas[:, None])
        distances = (unit_normals * tri_pts[:, 0]).sum(axis=1)

        # Canonicalize normal direction: flip so that the first nonzero
        # component is positive, ensuring coplanar faces hash together.
        for i in range(len(unit_normals)):
            for comp in unit_normals[i]:
                if abs(comp) > 1e-8:
                    if comp < 0:
                        unit_normals[i] *= -1
                        distances[i] *= -1
                    break

        face_groups: dict[tuple, float] = defaultdict(float)
        for i in range(len(areas)):
            key = tuple(np.round(np.append(unit_normals[i], distances[i]), decimals=4))
            face_groups[key] += areas[i]

        ### The sorted group sums must match the sorted original face areas
        np.testing.assert_allclose(
            sorted(face_groups.values()),
            sorted(original_face_areas),
            rtol=1e-4,
        )
