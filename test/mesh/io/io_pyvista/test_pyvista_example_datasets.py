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

"""Tests for physicsnemo.mesh.io module - PyVista example datasets."""

import pytest

pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io.io_pyvista import from_pyvista  # noqa: E402


def _download_or_skip(loader):
    r"""Call a pyvista ``download_*`` loader, skipping on upstream failures.

    The ``download_*`` example datasets are fetched from the ``pyvista/data``
    GitHub repository when not already cached locally. Transient upstream
    problems (HTTP 5xx such as the 502 seen in CI, connection resets,
    DNS/proxy errors, timeouts) are outside this project's control and should
    not fail CI, so they are converted into a skip. Only the download is
    wrapped, so genuine ``from_pyvista`` bugs still fail the test.

    We deliberately catch the builtin :class:`OSError` rather than a specific
    HTTP library's exception (e.g. ``requests.exceptions.RequestException``).
    This avoids coupling the test to pyvista's transitive dependencies: every
    network/HTTP error hierarchy pyvista's downloader can surface -- ``requests``
    (whose ``RequestException`` subclasses ``OSError``), ``urllib``, and
    ``socket`` -- derives from :class:`OSError`. So the skip keeps working with
    no extra import even if pyvista swaps out its HTTP backend. Because only the
    download call is wrapped, an ``OSError`` here can only mean the dataset
    could not be fetched (network failure or a local cache I/O error), never a
    conversion bug.

    Parameters
    ----------
    loader : Callable[[], pyvista.DataSet]
        A zero-argument pyvista ``download_*`` function.

    Returns
    -------
    pyvista.DataSet
        The downloaded mesh.
    """
    try:
        return loader()
    except OSError as exc:
        pytest.skip(
            f"Upstream pyvista data server unavailable for {loader.__name__!r}: {exc!r}"
        )


class TestPyVistaExampleDatasets:
    """Tests for various PyVista example datasets covering edge cases."""

    def test_cow_mesh_mixed_cells(self):
        """Test cow mesh which has a mix of triangular and quad cells.

        The cow mesh is a classic test case that contains both triangular
        and quadrilateral cells, requiring automatic triangulation.
        """
        pv_mesh = _download_or_skip(pv.examples.download_cow)

        # Verify it has mixed cell types (not all triangles)
        assert not pv_mesh.is_all_triangles

        # Convert - should automatically triangulate
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.cells.shape[1] == 3  # All triangulated
        # After triangulation, should have more or equal cells
        assert mesh.n_cells >= pv_mesh.n_cells
        assert mesh.n_points == pv_mesh.n_points

    def test_bunny_mesh(self):
        """Test Stanford bunny mesh (classic computer graphics mesh)."""
        pv_mesh = _download_or_skip(pv.examples.download_bunny)

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.cells.shape[1] == 3
        assert mesh.n_points == pv_mesh.n_points

    def test_frog_tissues_3d(self):
        """Test frog tissues dataset (3D medical imaging volume data).

        This loads a 3D ImageData and extracts its outer surface to test conversion.
        """
        # Load the frog dataset - it's ImageData (uniform grid)
        pv_mesh = pv.examples.load_frog_tissues()

        # Extract the outer surface of the volume data
        surface = pv_mesh.extract_surface()

        # Now test the surface conversion
        mesh = from_pyvista(surface, manifold_dim="auto")

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.cells.shape[1] == 3
        # Should have a reasonable number of points
        assert mesh.n_points > 100

    def test_sphere_decimated(self):
        """Test a decimated sphere (irregular triangulation)."""
        pv_mesh = pv.Sphere(radius=1.0, theta_resolution=50, phi_resolution=50)
        # Decimate to create irregular triangulation
        pv_mesh = pv_mesh.decimate(0.5)  # Reduce by 50%

        assert pv_mesh.is_all_triangles

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 2
        assert mesh.cells.shape[1] == 3

    def test_ant_mesh(self):
        """Test ant mesh from primitives."""
        pv_mesh = pv.examples.load_ant()

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.cells.shape[1] == 3

    def test_globe_mesh(self):
        """Test globe mesh (sphere with texture coordinates)."""
        pv_mesh = pv.examples.load_globe()

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 2
        assert mesh.cells.shape[1] == 3

    def test_drill_scan_mesh(self):
        """Test drill scan mesh (high-resolution surface scan).

        The drill dataset is a laser-scanned PolyData mesh from Laser Design,
        representing a detailed 3D surface scan of a power drill.
        """
        pv_mesh = _download_or_skip(pv.examples.download_drill)

        # Verify it's PolyData (surface mesh)
        assert isinstance(pv_mesh, pv.PolyData)

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.cells.shape[1] == 3  # Triangular surface mesh
        assert mesh.n_points == pv_mesh.n_points

        # Drill scan should have a reasonable number of points
        assert mesh.n_points > 1000
        assert mesh.n_cells > 1000
