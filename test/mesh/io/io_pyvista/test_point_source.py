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

"""Tests for from_pyvista point_source and warn_on_lost_data parameters."""

import warnings

import numpy as np
import pytest
import torch

pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io.io_pyvista import from_pyvista  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tet_with_data() -> "pv.UnstructuredGrid":
    """Single tetrahedron with both point_data and cell_data."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    cells = np.array([4, 0, 1, 2, 3])
    celltypes = np.array([pv.CellType.TETRA])
    mesh = pv.UnstructuredGrid(cells, celltypes, points)
    mesh.point_data["temperature"] = np.array([100.0, 200.0, 300.0, 400.0])
    mesh.cell_data["pressure"] = np.array([999.0])
    return mesh


def _make_hex_pair() -> "pv.UnstructuredGrid":
    """Two adjacent hexahedra sharing a face, with point and cell data."""
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [2, 0, 0],
            [2, 1, 0],
            [2, 0, 1],
            [2, 1, 1],
        ],
        dtype=np.float64,
    )
    hex0 = [8, 0, 1, 2, 3, 4, 5, 6, 7]
    hex1 = [8, 1, 8, 9, 2, 5, 10, 11, 6]
    cells = hex0 + hex1
    celltypes = [pv.CellType.HEXAHEDRON, pv.CellType.HEXAHEDRON]
    mesh = pv.UnstructuredGrid(cells, celltypes, points)
    mesh.point_data["vel"] = np.arange(12, dtype=np.float32)
    mesh.cell_data["zone"] = np.array([0, 1])
    return mesh


# ---------------------------------------------------------------------------
# point_source="vertices" (default behavior preservation)
# ---------------------------------------------------------------------------


class TestPointSourceVerticesDefault:
    """Existing behavior is unchanged when point_source is not specified."""

    def test_default_matches_original(self):
        mesh_default = from_pyvista(_make_tet_with_data(), manifold_dim=3)
        mesh_explicit = from_pyvista(
            _make_tet_with_data(), manifold_dim=3, point_source="vertices"
        )
        assert mesh_default.n_points == mesh_explicit.n_points
        assert mesh_default.n_cells == mesh_explicit.n_cells


# ---------------------------------------------------------------------------
# point_source="vertices", manifold_dim=0 (point cloud from 3D mesh)
# ---------------------------------------------------------------------------


class TestVerticesPointCloud:
    """Vertex-based 0D extraction via from_pyvista: strips cell connectivity from a
    3D mesh, retaining original vertices as a point cloud with point_data preserved."""

    def test_0d_from_3d_mesh(self):
        pv_mesh = _make_tet_with_data()
        mesh = from_pyvista(pv_mesh, manifold_dim=0, warn_on_lost_data=False)

        assert mesh.n_manifold_dims == 0
        assert mesh.n_points == 4
        assert mesh.n_cells == 0
        assert "temperature" in mesh.point_data

    def test_0d_preserves_point_data(self):
        pv_mesh = _make_tet_with_data()
        mesh = from_pyvista(pv_mesh, manifold_dim=0, warn_on_lost_data=False)

        np.testing.assert_allclose(
            mesh.point_data["temperature"].numpy(),
            [100.0, 200.0, 300.0, 400.0],
        )

    def test_integer_coordinates_are_promoted_to_float32(self):
        points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
        pv_mesh = pv.PointSet(points, force_float=False)

        mesh = from_pyvista(pv_mesh)

        assert mesh.points.numpy().dtype == np.float32
        np.testing.assert_array_equal(mesh.points.numpy(), points)


# ---------------------------------------------------------------------------
# point_source="vertices", manifold_dim=1 (edge graph from 3D mesh)
# ---------------------------------------------------------------------------


class TestVerticesEdgeGraph:
    """Vertex-based 1D extraction via from_pyvista: extracts unique edges from
    volumetric cells (tets, hexes, polyhedra) with shared-edge deduplication."""

    def test_edge_graph_from_tet(self):
        """A single tet has 6 edges."""
        pv_mesh = _make_tet_with_data()
        mesh = from_pyvista(pv_mesh, manifold_dim=1, warn_on_lost_data=False)

        assert mesh.n_manifold_dims == 1
        assert mesh.n_points == 4
        assert mesh.cells.shape[1] == 2
        assert mesh.n_cells == 6  # tet has 6 edges

    def test_edge_graph_from_hex_pair(self):
        """Two adjacent hexahedra share edges; total unique edges < 2*12."""
        pv_mesh = _make_hex_pair()
        mesh = from_pyvista(pv_mesh, manifold_dim=1, warn_on_lost_data=False)

        assert mesh.n_manifold_dims == 1
        assert mesh.cells.shape[1] == 2
        assert mesh.n_points == 12
        # Two hexes have 12 edges each but share 4 edges on the common face
        assert mesh.n_cells == 20

    def test_edge_graph_preserves_point_data(self):
        pv_mesh = _make_hex_pair()
        mesh = from_pyvista(pv_mesh, manifold_dim=1, warn_on_lost_data=False)

        assert "vel" in mesh.point_data
        assert mesh.point_data["vel"].shape[0] == 12

    def test_edge_graph_from_polyhedron(self):
        """Edge graph works on polyhedra without tetrahedralization."""
        from test.mesh.io.io_pyvista.test_from_pyvista_3d import (
            _make_pentagonal_prism,
        )

        pv_mesh = _make_pentagonal_prism()
        mesh = from_pyvista(pv_mesh, manifold_dim=1, warn_on_lost_data=False)

        assert mesh.n_manifold_dims == 1
        assert mesh.cells.shape[1] == 2
        assert mesh.n_points == 10
        # Pentagonal prism: 5 bottom + 5 top + 5 side = 15 edges
        assert mesh.n_cells == 15


# ---------------------------------------------------------------------------
# point_source="cell_centroids", manifold_dim=0 (centroid point cloud)
# ---------------------------------------------------------------------------


class TestCellCentroidsPointCloud:
    """Centroid-based 0D extraction via from_pyvista: replaces mesh vertices with cell
    centroids, remaps cell_data to point_data, and validates auto-detection of
    manifold dimension."""

    def test_centroid_point_cloud(self):
        pv_mesh = _make_tet_with_data()
        mesh = from_pyvista(
            pv_mesh,
            manifold_dim=0,
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )

        assert mesh.n_manifold_dims == 0
        assert mesh.n_points == 1  # 1 cell -> 1 centroid
        assert mesh.n_cells == 0

    def test_centroid_position(self):
        """Centroid of a unit tet at (0,0,0),(1,0,0),(0,1,0),(0,0,1) is (0.25,0.25,0.25)."""
        pv_mesh = _make_tet_with_data()
        mesh = from_pyvista(
            pv_mesh,
            manifold_dim=0,
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )

        np.testing.assert_allclose(
            mesh.points.numpy(),
            [[0.25, 0.25, 0.25]],
            atol=1e-5,
        )

    def test_centroid_preserves_float64_large_offset(self):
        """Centroid extraction must not collapse small translated geometry."""
        points = np.array(
            [
                [1e8, 1e8, 0.0],
                [1e8 + 1.0, 1e8, 0.0],
                [1e8, 1e8 + 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        pv_mesh = pv.PolyData.from_regular_faces(points, np.array([[0, 1, 2]]))

        mesh = from_pyvista(
            pv_mesh,
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )

        assert mesh.points.numpy().dtype == np.float64
        np.testing.assert_allclose(
            mesh.points.numpy(),
            [[1e8 + 1.0 / 3.0, 1e8 + 1.0 / 3.0, 0.0]],
            rtol=0.0,
            atol=2e-8,
        )

    def test_cell_data_becomes_point_data(self):
        pv_mesh = _make_tet_with_data()
        mesh = from_pyvista(
            pv_mesh,
            manifold_dim=0,
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )

        assert "pressure" in mesh.point_data
        np.testing.assert_allclose(
            mesh.point_data["pressure"].numpy(),
            [999.0],
        )

    def test_centroid_auto_resolves_to_0d(self):
        pv_mesh = _make_tet_with_data()
        mesh = from_pyvista(
            pv_mesh,
            manifold_dim="auto",
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )

        assert mesh.n_manifold_dims == 0

    def test_centroid_invalid_dim_raises(self):
        pv_mesh = _make_tet_with_data()
        with pytest.raises(ValueError, match="only supports manifold_dim"):
            from_pyvista(
                pv_mesh,
                manifold_dim=3,
                point_source="cell_centroids",
            )

    def test_centroid_from_polyhedron(self):
        """Centroid extraction works on polyhedra without tetrahedralization."""
        from test.mesh.io.io_pyvista.test_from_pyvista_3d import (
            _make_pentagonal_prism,
        )

        pv_mesh = _make_pentagonal_prism()
        pv_mesh.cell_data["vol_id"] = np.array([42])

        mesh = from_pyvista(
            pv_mesh,
            manifold_dim=0,
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )

        assert mesh.n_manifold_dims == 0
        assert mesh.n_points == 1
        assert mesh.point_data["vol_id"].item() == 42

    @pytest.mark.parametrize("as_unstructured", [False, True])
    def test_centroid_of_variable_size_cells_is_vertex_mean(self, as_unstructured):
        """Polygon, polyline, strip and poly-vertex centroids are vertex means."""
        angles = np.linspace(0, 2 * np.pi, 6)[:-1]
        pentagon = np.stack([np.cos(angles), np.sin(angles), np.zeros(5)], axis=1)
        others = np.array(
            [[0, 0, 1], [1, 0, 1], [2, 0, 1], [0, 1, 2], [1, 1, 2], [2, 1, 2]],
            dtype=np.float64,
        )
        pv_mesh = pv.PolyData(
            np.concatenate([pentagon, others]),
            verts=[4, 5, 6, 7, 10],  # poly-vertex
            lines=[3, 5, 6, 7],  # straight polyline
            faces=[5, 0, 1, 2, 3, 4],  # regular pentagon
            strips=[6, 5, 6, 7, 8, 9, 10],  # triangle strip
        )
        if as_unstructured:
            pv_mesh = pv_mesh.cast_to_unstructured_grid()

        mesh = from_pyvista(
            pv_mesh,
            manifold_dim=0,
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )

        # PolyData cell order: verts, lines, polys, strips.
        expected = [
            [1.25, 0.25, 1.25],  # mean of (0,0,1), (1,0,1), (2,0,1), (2,1,2)
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 1.5],
        ]
        np.testing.assert_allclose(mesh.points.numpy(), expected, atol=1e-12)

    @pytest.mark.parametrize("as_unstructured", [False, True])
    @pytest.mark.parametrize(
        "dtype,x",
        [
            (np.int64, 2**62),
            (np.float32, 2e38),
            (np.float64, 8e307),
            (np.float64, np.finfo(np.float64).max),
        ],
    )
    def test_variable_size_centroid_near_dtype_limit(self, as_unstructured, dtype, x):
        """A polyline centroid is exact even when its vertex sum would overflow."""
        pv_mesh = pv.PolyData(
            np.array([[x, 0, 0], [x, 1, 0], [x, 2, 0]], dtype=dtype),
            lines=[3, 0, 1, 2],
            force_float=False,
        )
        if as_unstructured:
            pv_mesh = pv_mesh.cast_to_unstructured_grid()

        mesh = from_pyvista(
            pv_mesh,
            manifold_dim=0,
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )

        expected = [[float(dtype(x)), 1.0, 0.0]]
        np.testing.assert_allclose(mesh.points.numpy(), expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# point_source="cell_centroids", manifold_dim=1 (dual graph)
# ---------------------------------------------------------------------------


class TestCellCentroidsDualGraph:
    """Centroid-based dual graph via from_pyvista: builds a 1D graph where cell
    centroids are nodes and facet-adjacent cell pairs are edges (a facet is a
    face for volume cells, an edge for surface cells, a vertex for line cells),
    with cell_data remapped to point_data. Covers volume, surface, and line
    meshes to exercise the dimension-generic facet adjacency."""

    def test_dual_graph_two_hexes(self):
        """Two adjacent hexes sharing a face produce 1 dual-graph edge."""
        pv_mesh = _make_hex_pair()
        mesh = from_pyvista(
            pv_mesh,
            manifold_dim=1,
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )

        assert mesh.n_manifold_dims == 1
        assert mesh.n_points == 2  # 2 cells -> 2 centroids
        assert mesh.n_cells == 1  # 1 shared face -> 1 edge
        assert mesh.cells.shape == (1, 2)

    def test_dual_graph_cell_data_as_point_data(self):
        pv_mesh = _make_hex_pair()
        mesh = from_pyvista(
            pv_mesh,
            manifold_dim=1,
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )

        assert "zone" in mesh.point_data
        assert mesh.point_data["zone"].shape[0] == 2

    def test_dual_graph_isolated_cell_no_edges(self):
        """A single cell has no facet-neighbors, so the dual graph has 0 edges."""
        pv_mesh = _make_tet_with_data()
        mesh = from_pyvista(
            pv_mesh,
            manifold_dim=1,
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )

        assert mesh.n_points == 1
        assert mesh.n_cells == 0

    def test_dual_graph_triangulated_surface(self):
        """Surface cells share edges (1-faces), so a surface mesh has a non-empty
        dual graph. Regression for the face-only bug, which returned 0 edges here
        because triangles report ``GetNumberOfFaces() == 0``."""
        surf = pv.Plane(i_resolution=2, j_resolution=2).triangulate()  # 8 triangles
        mesh = from_pyvista(
            surf,
            manifold_dim=1,
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )

        assert mesh.n_manifold_dims == 1
        assert mesh.n_points == 8  # 8 triangles -> 8 centroids
        assert mesh.cells.shape == (8, 2)
        assert mesh.n_cells == 8  # edge-sharing adjacencies among the triangles

    def test_dual_graph_quad_surface(self):
        """A 2x2 grid of quads yields the expected 4-edge grid adjacency."""
        surf = pv.Plane(i_resolution=2, j_resolution=2)  # 4 quads
        mesh = from_pyvista(
            surf,
            manifold_dim=1,
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )

        assert mesh.n_manifold_dims == 1
        assert mesh.n_points == 4  # 4 quads -> 4 centroids
        assert mesh.n_cells == 4  # 2 horizontal + 2 vertical neighbor pairs

    def test_dual_graph_polyline(self):
        """Full dimension-genericity: a polyline's cells (segments) share endpoint
        vertices, so its dual graph is the segment-adjacency line graph."""
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64
        )
        # Three segments: (0, 1), (1, 2), (2, 3).
        lines = np.array([2, 0, 1, 2, 1, 2, 2, 2, 3])
        poly = pv.PolyData(points, lines=lines)
        mesh = from_pyvista(
            poly,
            manifold_dim=1,
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )

        assert mesh.n_manifold_dims == 1
        assert mesh.n_points == 3  # 3 segments -> 3 centroids
        assert mesh.n_cells == 2  # consecutive segments share a vertex

    @staticmethod
    def _simplex_dual_graph_cases() -> dict[str, "pv.DataSet"]:
        rng = np.random.default_rng(0)
        tets = pv.PolyData(rng.random((60, 3))).delaunay_3d()
        # Shuffle the cells and permute each cell's vertices.
        tet_cells = np.asarray(tets.cell_connectivity).reshape(-1, 4)
        tet_cells = tet_cells[rng.permutation(len(tet_cells))][:, [2, 0, 3, 1]]
        return {
            "tet delaunay": tets,
            "tet shuffled": pv.UnstructuredGrid(
                {pv.CellType.TETRA: tet_cells}, tets.points
            ),
            "tri plane": pv.Plane(i_resolution=4, j_resolution=3).triangulate(),
            # Four triangles share edge (0, 1); the last cell duplicates the first.
            "tri non-manifold": pv.PolyData.from_regular_faces(
                rng.random((7, 3)),
                np.array(
                    [[0, 1, 2], [1, 0, 3], [0, 1, 4], [5, 1, 0], [1, 2, 6], [0, 1, 2]]
                ),
            ),
            "tri grid": pv.UnstructuredGrid(
                {pv.CellType.TRIANGLE: np.array([[0, 1, 2], [2, 1, 3], [3, 1, 0]])},
                rng.random((4, 3)),
            ),
            # Five lines share vertex 0.
            "line star": pv.UnstructuredGrid(
                {pv.CellType.LINE: np.array([[0, 1], [2, 0], [0, 3], [4, 0], [0, 5]])},
                rng.random((6, 3)),
            ),
            "line polydata": pv.PolyData(
                rng.random((5, 3)), lines=np.array([2, 0, 1, 2, 1, 2, 2, 3, 2, 2, 2, 4])
            ),
        }

    @pytest.mark.parametrize(
        "case",
        [
            "tet delaunay",
            "tet shuffled",
            "tri plane",
            "tri non-manifold",
            "tri grid",
            "line star",
            "line polydata",
        ],
    )
    @pytest.mark.parametrize("packed_keys", [True, False])
    def test_simplex_fast_path_matches_vtk_loop(self, case, packed_keys, monkeypatch):
        """The vectorized LINE/TRIANGLE/TETRA path gives the VTK loop's edges, in
        order, both with packed int64 facet keys and with the tuple-lexsort fallback."""
        import physicsnemo.mesh.io.io_pyvista as io_pyvista

        pv_mesh = self._simplex_dual_graph_cases()[case]
        simplex_cells = io_pyvista._homogeneous_simplex_cells(pv_mesh)
        assert simplex_cells is not None
        local_facets = io_pyvista._SIMPLEX_FACETS[simplex_cells.shape[1]]
        assert io_pyvista._packed_facet_keys(simplex_cells, local_facets) is not None
        if not packed_keys:
            monkeypatch.setattr(io_pyvista, "_packed_facet_keys", lambda *_: None)
        assert io_pyvista._simplex_dual_graph_edges(simplex_cells) is not None
        fast = io_pyvista._build_dual_graph_edges(pv_mesh)

        monkeypatch.setattr(io_pyvista, "_homogeneous_simplex_cells", lambda _: None)
        reference = io_pyvista._build_dual_graph_edges(pv_mesh)

        assert len(reference) > 0
        assert fast.dtype == reference.dtype == torch.long
        assert torch.equal(fast, reference)

    def test_packed_facet_keys_fall_back_on_overflow(self):
        """Facet keys that could overflow int64 are not packed."""
        from physicsnemo.mesh.io.io_pyvista import _SIMPLEX_FACETS, _packed_facet_keys

        cells = np.array([[0, 1, 2, 3], [1, 2, 3, 2**21]])
        assert _packed_facet_keys(cells, _SIMPLEX_FACETS[4]) is None
        assert _packed_facet_keys(cells[:, :3], _SIMPLEX_FACETS[3]) is not None

    def test_simplex_fast_path_falls_back_on_repeated_vertex(self):
        """A cell that repeats a vertex is left to the VTK loop."""
        from physicsnemo.mesh.io.io_pyvista import _simplex_dual_graph_edges

        assert _simplex_dual_graph_edges(np.array([[0, 1, 2], [1, 1, 3]])) is None


# ---------------------------------------------------------------------------
# warn_on_lost_data
# ---------------------------------------------------------------------------


class TestWarnOnLostData:
    """Verifies from_pyvista emits UserWarnings naming discarded fields when dimension
    reduction drops point_data or cell_data, and that warnings are suppressed when
    disabled or when no data is lost."""

    def test_warns_on_lost_cell_data(self):
        """Extracting 0D from 3D with cell_data should warn."""
        pv_mesh = _make_tet_with_data()
        with pytest.warns(UserWarning, match="cell_data"):
            from_pyvista(pv_mesh, manifold_dim=0, warn_on_lost_data=True)

    def test_warns_on_lost_point_data(self):
        """Using cell_centroids with point_data should warn."""
        pv_mesh = _make_tet_with_data()
        with pytest.warns(UserWarning, match="point_data"):
            from_pyvista(
                pv_mesh,
                manifold_dim=0,
                point_source="cell_centroids",
                warn_on_lost_data=True,
            )

    def test_no_warn_when_data_empty(self):
        """No warning when the discarded data category is empty."""
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        cells = np.array([4, 0, 1, 2, 3])
        celltypes = np.array([pv.CellType.TETRA])
        pv_mesh = pv.UnstructuredGrid(cells, celltypes, points)
        # No point_data or cell_data added

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            from_pyvista(pv_mesh, manifold_dim=0, warn_on_lost_data=True)
            from_pyvista(
                pv_mesh,
                manifold_dim=0,
                point_source="cell_centroids",
                warn_on_lost_data=True,
            )

    def test_no_warn_when_disabled(self):
        """warn_on_lost_data=False suppresses all warnings."""
        pv_mesh = _make_tet_with_data()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            from_pyvista(pv_mesh, manifold_dim=0, warn_on_lost_data=False)
            from_pyvista(
                pv_mesh,
                manifold_dim=0,
                point_source="cell_centroids",
                warn_on_lost_data=False,
            )

    def test_warn_names_dropped_fields(self):
        """Warning message includes the names of the dropped fields."""
        pv_mesh = _make_tet_with_data()
        with pytest.warns(UserWarning, match="pressure"):
            from_pyvista(pv_mesh, manifold_dim=0, warn_on_lost_data=True)

    def test_no_warn_at_same_dim(self):
        """No cell_data warning when manifold_dim matches detected dim."""
        pv_mesh = _make_tet_with_data()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            from_pyvista(pv_mesh, manifold_dim=3, warn_on_lost_data=True)
