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

"""Main remeshing entry point.

This module wires together all components of the remeshing pipeline.
"""

from typing import TYPE_CHECKING

from physicsnemo.core.version_check import OptionalImport, require_version_spec

### Optional dependency. ``pyacvd`` is a lazy proxy: construction does not
### import the package; the friendly ``ImportError`` (with the
### ``[mesh-extras]`` install hint) fires only on first attribute access. The
### ``@require_version_spec("pyacvd")`` decorator on ``remesh`` raises that
### same error proactively before any function-body work happens.
if TYPE_CHECKING:
    import pyacvd

    from physicsnemo.mesh.mesh import Mesh
else:
    pyacvd = OptionalImport("pyacvd")


@require_version_spec("pyacvd")
def remesh(
    mesh: "Mesh",
    n_clusters: int,
) -> "Mesh":
    """Uniform remeshing of a 2D triangle surface (in 3D) via clustering.

    Creates a simplified mesh with approximately n_clusters cells uniformly
    distributed across the geometry. Uses the ACVD (Approximate Centroidal
    Voronoi Diagram) clustering algorithm.

    The algorithm:
    1. Weights vertices by their dual volumes (Voronoi areas)
    2. Initializes clusters via area-based region growing
    3. Minimizes energy by iteratively reassigning vertices
    4. Reconstructs a simplified mesh from cluster adjacency

    This is restricted to 2D triangle surfaces embedded in 3D space -- the only
    case the underlying ``pyacvd`` ACVD clustering supports.

    Parameters
    ----------
    mesh : Mesh
        Input mesh to remesh
    n_clusters : int
        Target number of output cells. The actual number may vary
        slightly depending on mesh topology.

    Returns
    -------
    Mesh
        Remeshed mesh with approximately n_clusters cells. The vertices are
        cluster centroids, and cells connect adjacent clusters.

    Raises
    ------
    NotImplementedError
        If the mesh is not a 2D triangle surface embedded in 3D.
    ImportError
        If the optional ``pyacvd`` dependency is not installed.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    >>> from physicsnemo.mesh.remeshing import remesh
    >>> mesh = sphere_icosahedral.load(subdivisions=3)
    >>> # Remesh a triangle mesh to ~100 triangles
    >>> simplified = remesh(mesh, n_clusters=100)
    >>> assert simplified.n_cells > 0

    Notes
    -----
    - Restricted to 2D triangle surfaces embedded in 3D (``pyacvd`` limitation)
    - Preserves mesh topology qualitatively but not quantitatively
    - Point and cell data are not transferred (topology changes fundamentally)
    - Output cell orientation may differ from input
    """
    from physicsnemo.mesh.io.io_pyvista import from_pyvista, to_pyvista
    from physicsnemo.mesh.mesh import Mesh
    from physicsnemo.mesh.repair import repair_mesh

    # pyacvd ACVD clustering is a triangle-surface algorithm: it only handles a
    # PolyData of triangles (a 2D manifold in 3D). Guard explicitly so any other
    # mesh gets a clear error instead of a confusing downstream pyacvd failure.
    if mesh.n_manifold_dims != 2 or mesh.n_spatial_dims != 3:
        raise NotImplementedError(
            "remesh only supports 2D triangle surfaces embedded in 3D "
            "(the pyacvd ACVD clustering is surface-only). Got "
            f"n_manifold_dims={mesh.n_manifold_dims}, "
            f"n_spatial_dims={mesh.n_spatial_dims}."
        )

    clustering = pyacvd.Clustering(to_pyvista(mesh))
    clustering.cluster(n_clusters)
    new_mesh, _stats = repair_mesh(from_pyvista(clustering.create_mesh()))

    # pyacvd/pyvista round-trip through float32 on CPU. Restore the input's device
    # and dtype. (Mesh.to(dtype) can't be used here -- it would also cast the
    # integer cells; remesh discards field data, so only points/cells are kept.)
    return Mesh(
        points=new_mesh.points.to(device=mesh.points.device, dtype=mesh.points.dtype),
        cells=new_mesh.cells.to(device=mesh.points.device),
    )
