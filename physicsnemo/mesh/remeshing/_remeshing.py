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

"""Public dispatch and backend implementations for surface remeshing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.core.version_check import OptionalImport
from physicsnemo.mesh.remeshing._config import WarpRemeshOptions

if TYPE_CHECKING:
    import pyacvd

    from physicsnemo.mesh.mesh import Mesh
else:
    pyacvd = OptionalImport("pyacvd")


def _validate_remesh_inputs(
    mesh: "Mesh",
    n_clusters: int,
    max_iterations: int | None,
    *,
    require_float32_range: bool,
) -> None:
    """Validate backend-independent surface remeshing invariants."""
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
            f"n_clusters cannot exceed the input point count; got "
            f"{n_clusters=} and mesh.n_points={mesh.n_points}"
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

    # Collect device-side checks into one synchronization. Connectivity bounds
    # must be validated before a Warp kernel dereferences the input indices.
    device_checks = [
        torch.isfinite(mesh.points).all(),
        mesh.cells.min() >= 0,
        mesh.cells.max() < mesh.n_points,
    ]
    if require_float32_range:
        device_checks.append(
            (mesh.points.abs() <= torch.finfo(torch.float32).max).all()
        )
    checks = torch.stack(device_checks).to(device="cpu")
    finite_points, lower_bound_ok, upper_bound_ok = [
        bool(value) for value in checks[:3]
    ]
    if not finite_points:
        raise ValueError("mesh points must contain only finite coordinates")
    if not lower_bound_ok or not upper_bound_ok:
        raise ValueError(f"mesh cell indices must lie in [0, {mesh.n_points})")
    if require_float32_range and not bool(checks[3]):
        raise ValueError(
            "Warp remeshing computes in float32; mesh points must remain finite "
            "when converted to float32"
        )


def _remesh_pyacvd(
    mesh: "Mesh",
    n_clusters: int,
    *,
    max_iterations: int | None,
) -> "Mesh":
    """Run the legacy CPU ACVD implementation."""
    from physicsnemo.mesh.io.io_pyvista import from_pyvista, to_pyvista
    from physicsnemo.mesh.mesh import Mesh
    from physicsnemo.mesh.repair import repair_mesh

    clustering = pyacvd.Clustering(to_pyvista(mesh))
    clustering.cluster(
        n_clusters,
        maxiter=100 if max_iterations is None else max_iterations,
    )
    new_mesh, _stats = repair_mesh(from_pyvista(clustering.create_mesh()))

    # PyACVD/PyVista execute on CPU. Restore the input device and point dtype;
    # connectivity remains int64, matching the historical backend contract.
    return Mesh(
        points=new_mesh.points.to(device=mesh.points.device, dtype=mesh.points.dtype),
        cells=new_mesh.cells.to(device=mesh.points.device, dtype=torch.int64),
        global_data=mesh.global_data.clone(),
    )


class Remesh(FunctionSpec):
    """Uniformly remesh a triangle surface using a CPU or GPU backend.

    The Warp backend performs area-weighted centroidal clustering, projects
    cluster centers back to the source surface with a GPU BVH, and reconstructs
    compact triangle connectivity entirely on the input CUDA device. The
    PyACVD backend retains the existing CPU ACVD implementation.

    With ``implementation=None``, CUDA meshes select Warp and CPU meshes select
    PyACVD. Pass an implementation explicitly to compare backends or to retain
    the legacy CUDA-to-CPU PyACVD path.

    Parameters
    ----------
    mesh : Mesh
        Input triangle surface. Only 2D triangle manifolds embedded in 3D are
        supported.
    n_clusters : int
        Target output vertex count. Cleanup can produce slightly fewer vertices
        when a cluster is unused or all of its incident faces collapse.
    max_iterations : int | None, optional
        Maximum centroid-relaxation iterations. ``None`` uses a backend-tuned
        default: 4 for Warp and 100 for PyACVD.
    warp_options : WarpRemeshOptions | None, optional
        Warp-specific performance and initialization controls. This is only
        valid when the selected implementation is ``"warp"``.
    implementation : {"warp", "pyacvd"} | None, optional
        Backend selection. ``None`` dispatches from the input device.

    Returns
    -------
    Mesh
        Geometry-only remeshed surface on the input device. Point and cell data
        are discarded because topology changes; global data is preserved.

    Notes
    -----
    Remeshing is intentionally non-differentiable. The Warp backend computes in
    float32 and restores the input point dtype on return. Its spatial clustering
    is optimized for uniformly sampled surfaces; unlike topology-aware ACVD,
    very close disconnected sheets can be assigned to a common cluster.
    """

    @FunctionSpec.register(
        name="warp",
        required_imports=("warp>=1.14.0",),
        rank=0,
    )
    def warp_forward(
        mesh: "Mesh",
        n_clusters: int,
        *,
        max_iterations: int | None = None,
        warp_options: WarpRemeshOptions | None = None,
    ) -> "Mesh":
        """Run Warp-accelerated CUDA remeshing."""
        from physicsnemo.mesh.remeshing._warp_impl import remesh_warp

        return remesh_warp(
            mesh,
            n_clusters,
            max_iterations=max_iterations,
            options=warp_options or WarpRemeshOptions(),
        )

    @FunctionSpec.register(
        name="pyacvd",
        required_imports=("pyacvd>=0.3.2", "pyvista>=0.47.0"),
        rank=1,
        baseline=True,
    )
    def pyacvd_forward(
        mesh: "Mesh",
        n_clusters: int,
        *,
        max_iterations: int | None = None,
        warp_options: WarpRemeshOptions | None = None,
    ) -> "Mesh":
        """Run CPU ACVD remeshing through PyACVD and PyVista."""
        if warp_options is not None:
            raise ValueError("warp_options can only be used with implementation='warp'")
        return _remesh_pyacvd(
            mesh,
            n_clusters,
            max_iterations=max_iterations,
        )

    @classmethod
    def dispatch(
        cls,
        mesh: "Mesh",
        n_clusters: int,
        *,
        max_iterations: int | None = None,
        warp_options: WarpRemeshOptions | None = None,
        implementation: Literal["warp", "pyacvd"] | None = None,
    ) -> "Mesh":
        """Select Warp for CUDA meshes and PyACVD for CPU meshes."""
        implementations = cls._get_impls()
        cls._check_impl(implementation, implementations)
        selected_name = (
            implementation
            if implementation is not None
            else ("warp" if mesh.points.is_cuda else "pyacvd")
        )
        _validate_remesh_inputs(
            mesh,
            n_clusters,
            max_iterations,
            require_float32_range=selected_name == "warp",
        )
        if warp_options is not None and not isinstance(warp_options, WarpRemeshOptions):
            raise TypeError(
                "warp_options must be a WarpRemeshOptions instance or None, got "
                f"{type(warp_options).__name__}"
            )

        if selected_name == "warp" and not mesh.points.is_cuda:
            raise ValueError(
                "The Warp remeshing backend requires a CUDA mesh; move the mesh "
                "to CUDA or use implementation='pyacvd'."
            )
        if selected_name == "pyacvd" and warp_options is not None:
            raise ValueError("warp_options can only be used with implementation='warp'")

        selected = implementations[selected_name]
        if not selected.available:
            if selected_name == "pyacvd":
                raise ImportError(
                    "The PyACVD remeshing backend requires the 'mesh-extras' "
                    "dependencies (pyacvd and pyvista)."
                )
            raise ImportError("The Warp remeshing backend requires warp>=1.14.0.")
        return selected.func(
            mesh,
            n_clusters,
            max_iterations=max_iterations,
            warp_options=warp_options,
        )


def remesh(
    mesh: Mesh,
    n_clusters: int,
    *,
    max_iterations: int | None = None,
    warp_options: WarpRemeshOptions | None = None,
    implementation: Literal["warp", "pyacvd"] | None = None,
) -> Mesh:
    """Uniformly remesh a triangle surface using a CPU or GPU backend."""
    return Remesh.dispatch(
        mesh,
        n_clusters,
        max_iterations=max_iterations,
        warp_options=warp_options,
        implementation=implementation,
    )


# Keep the detailed user-facing API documentation in one place.
remesh.__doc__ = Remesh.__doc__

__all__ = ["Remesh", "WarpRemeshOptions", "remesh"]
