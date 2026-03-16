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


# ---------------------------------------------------------------------------
# point_source="cell_centroids", manifold_dim=1 (dual graph)
# ---------------------------------------------------------------------------


class TestCellCentroidsDualGraph:
    """Centroid-based dual graph via from_pyvista: builds a 1D graph where cell
    centroids are nodes and face-adjacent cell pairs are edges, with cell_data
    remapped to point_data."""

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
        """A single cell has no face-neighbors, so the dual graph has 0 edges."""
        pv_mesh = _make_tet_with_data()
        mesh = from_pyvista(
            pv_mesh,
            manifold_dim=1,
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )

        assert mesh.n_points == 1
        assert mesh.n_cells == 0


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
