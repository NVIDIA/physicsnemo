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

"""Public Mesh API for CPU and GPU surface remeshing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from physicsnemo.core.version_check import OptionalImport, require_version_spec
from physicsnemo.nn.functional.geometry.remeshing import WarpRemeshOptions, remeshing

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh

_PYACVD_INSTALL_HINT = (
    "Install the optional CPU backend with:\n"
    '  uv pip install "pyacvd>=0.3.2" "pyvista>=0.47.0"\n'
    '  pip install "pyacvd>=0.3.2" "pyvista>=0.47.0"'
)

pyacvd = OptionalImport("pyacvd", package_hint=_PYACVD_INSTALL_HINT)


def _validate_remesh_inputs(
    mesh: Mesh,
    n_clusters: int,
    max_iterations: int | None,
    *,
    check_tensor_values: bool,
) -> None:
    """Validate backend-independent Mesh remeshing invariants."""
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
        raise TypeError(
            f"n_clusters must be an integer, got {type(n_clusters).__name__}"
        )
    if mesh.n_manifold_dims != 2 or mesh.n_spatial_dims != 3:
        raise NotImplementedError(
            "remesh only supports 2D triangle surfaces embedded in 3D; got "
            f"n_manifold_dims={mesh.n_manifold_dims} and "
            f"n_spatial_dims={mesh.n_spatial_dims}"
        )
    if n_clusters < 3:
        raise ValueError(f"n_clusters must be at least 3, got {n_clusters}")
    if n_clusters > mesh.n_points:
        raise ValueError(
            "n_clusters cannot exceed the input point count; got "
            f"n_clusters={n_clusters} and mesh.n_points={mesh.n_points}"
        )
    if max_iterations is not None:
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise TypeError(
                "max_iterations must be an integer or None, got "
                f"{type(max_iterations).__name__}"
            )
        if max_iterations < 0:
            raise ValueError(
                f"max_iterations must be non-negative, got {max_iterations}"
            )

    if mesh.n_cells == 0:
        raise ValueError("remesh requires at least one triangle")
    if not torch.is_floating_point(mesh.points):
        raise TypeError(
            f"mesh points must use a floating-point dtype, got {mesh.points.dtype}"
        )
    if not check_tensor_values:
        return

    # Collect the remaining device-side predicates into one synchronization.
    checks = torch.stack(
        [
            torch.isfinite(mesh.points).all(),
            mesh.cells.min() >= 0,
            mesh.cells.max() < mesh.n_points,
        ]
    ).to(device="cpu")
    finite_points, lower_bound_ok, upper_bound_ok = [bool(value) for value in checks]
    if not finite_points:
        raise ValueError("mesh points must contain only finite coordinates")
    if not lower_bound_ok or not upper_bound_ok:
        raise ValueError(f"mesh cell indices must lie in [0, {mesh.n_points})")


@require_version_spec("pyacvd", "0.3.2")
@require_version_spec("pyvista", "0.47.0")
def _remesh_pyacvd(
    mesh: Mesh,
    n_clusters: int,
    *,
    max_iterations: int | None,
) -> Mesh:
    """Run the CPU Approximate Centroidal Voronoi Diagram implementation."""
    from physicsnemo.mesh.io.io_pyvista import from_pyvista, to_pyvista
    from physicsnemo.mesh.mesh import Mesh
    from physicsnemo.mesh.repair import repair_mesh

    geometry = Mesh(points=mesh.points, cells=mesh.cells)
    clustering = pyacvd.Clustering(to_pyvista(geometry))
    clustering.cluster(
        n_clusters,
        maxiter=100 if max_iterations is None else max_iterations,
    )
    new_mesh, _stats = repair_mesh(from_pyvista(clustering.create_mesh()))

    # PyACVD and PyVista execute on CPU. Explicit PyACVD selection for a CUDA
    # mesh performs a host round trip before restoring the input device.
    return Mesh(
        points=new_mesh.points.to(device=mesh.points.device, dtype=mesh.points.dtype),
        cells=new_mesh.cells.to(device=mesh.points.device, dtype=torch.int64),
        global_data=mesh.global_data.clone(),
    )


def remesh(
    mesh: Mesh,
    n_clusters: int,
    *,
    max_iterations: int | None = None,
    warp_options: WarpRemeshOptions | None = None,
    implementation: Literal["warp", "pyacvd"] | None = None,
) -> Mesh:
    """Uniformly remesh a triangle surface using a CPU or GPU backend.

    The Warp backend performs area-weighted centroidal clustering, projects
    cluster centers back to the source surface with a GPU bounding volume
    hierarchy, and reconstructs compact triangle connectivity on CUDA. The
    PyACVD backend runs Approximate Centroidal Voronoi Diagram clustering on
    CPU.

    With ``implementation=None``, CUDA meshes select Warp and CPU meshes select
    PyACVD. Explicit PyACVD selection for a CUDA mesh copies geometry through
    CPU.

    Parameters
    ----------
    mesh : Mesh
        Input triangle surface. Only 2D triangle manifolds embedded in 3D are
        supported.
    n_clusters : int
        Target output vertex count. Cleanup can produce slightly fewer vertices.
        Must be between 3 and the input point count, inclusive.
    max_iterations : int | None, optional
        Maximum centroid-relaxation iterations. ``None`` uses a backend-tuned
        default: 4 for Warp and 100 for PyACVD. Values must be non-negative.
    warp_options : WarpRemeshOptions | None, optional
        Warp-specific performance and initialization controls. Only valid with
        the Warp backend.
    implementation : {"warp", "pyacvd"} | None, optional
        Backend selection. ``None`` dispatches from the input device.

    Returns
    -------
    Mesh
        Geometry-only remeshed surface on the input device. Point and cell data
        are discarded because topology changes; global data is preserved.

    Raises
    ------
    TypeError
        If counts, options, or point coordinates have invalid types.
    ValueError
        If a count is out of range, coordinates or connectivity are invalid,
        or backend-specific options are used with PyACVD.
    NotImplementedError
        If ``mesh`` is not a 2D triangle surface embedded in 3D.
    KeyError
        If ``implementation`` is not ``"warp"`` or ``"pyacvd"``.
    ImportError
        If dependencies for the selected backend are unavailable.
    RuntimeError
        If cleanup cannot reconstruct a nonempty manifold triangle surface.

    Notes
    -----
    Remeshing is intentionally non-differentiable. The Warp backend computes in
    centered and scaled coordinates in float32, then restores the input point
    dtype and coordinate frame. Because Warp clusters by spatial distance
    rather than mesh connectivity, very close disconnected sheets can be
    assigned to a common cluster.
    """
    if implementation not in (None, "warp", "pyacvd"):
        raise KeyError(f"No remeshing implementation named {implementation!r}")
    if warp_options is not None and not isinstance(warp_options, WarpRemeshOptions):
        raise TypeError(
            "warp_options must be a WarpRemeshOptions instance or None, got "
            f"{type(warp_options).__name__}"
        )

    selected = implementation or ("warp" if mesh.points.is_cuda else "pyacvd")
    _validate_remesh_inputs(
        mesh,
        n_clusters,
        max_iterations,
        check_tensor_values=selected == "pyacvd",
    )

    if selected == "pyacvd":
        if warp_options is not None:
            raise ValueError("warp_options can only be used with implementation='warp'")
        return _remesh_pyacvd(
            mesh,
            n_clusters,
            max_iterations=max_iterations,
        )

    if not mesh.points.is_cuda:
        raise ValueError(
            "The Warp remeshing backend requires a CUDA mesh; move the mesh "
            "to CUDA or use implementation='pyacvd'."
        )
    output_points, output_cells = remeshing(
        mesh.points,
        mesh.cells,
        n_clusters,
        max_iterations=max_iterations,
        warp_options=warp_options,
        implementation="warp",
    )

    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=output_points,
        cells=output_cells,
        global_data=mesh.global_data.clone(),
    )


__all__ = ["WarpRemeshOptions", "remesh"]
