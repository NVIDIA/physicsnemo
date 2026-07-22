# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shape-to-shape fitting for triangle surface meshes."""

from typing import TYPE_CHECKING, Literal

from physicsnemo.mesh.transformations.deform._utils import (
    _mesh_with_deformed_points,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _validate_triangle_surface(mesh: object, argument_name: str) -> "Mesh":
    """Require a triangle surface mesh embedded in three dimensions."""

    from physicsnemo.mesh.mesh import Mesh

    if not isinstance(mesh, Mesh):
        raise TypeError(f"{argument_name} must be a Mesh, got {type(mesh).__name__}")
    if mesh.n_manifold_dims != 2 or mesh.n_spatial_dims != 3:
        raise ValueError(
            f"{argument_name} must be a triangle surface mesh embedded in 3D, got "
            f"n_manifold_dims={mesh.n_manifold_dims} and "
            f"n_spatial_dims={mesh.n_spatial_dims}"
        )
    return mesh


def fit_template(
    template: "Mesh",
    target: "Mesh",
    *,
    fit_weight: float = 1.0,
    arap_weight: float = 0.1,
    steps: int = 10,
    cg_tolerance: float = 1.0e-6,
    cg_max_iterations: int = 256,
    implementation: Literal["torch", "warp"] | None = None,
) -> "Mesh":
    """Fit a prealigned triangle template to a target surface.

    The template's vertices move toward closest points on the target while an
    as-rigid-as-possible term discourages local distortion. Only template point
    coordinates change; its connectivity and attached fields are retained.

    Parameters
    ----------
    template : Mesh
        Prealigned 3D triangle surface mesh to deform. The source mesh is not
        modified.
    target : Mesh
        Prealigned 3D triangle surface mesh defining the fitting surface. Its
        topology may differ from the template and it is not modified.
    fit_weight : float, optional
        Positive closest-surface fitting weight. Default is ``1.0``.
    arap_weight : float, optional
        Nonnegative as-rigid-as-possible regularization weight. Default is
        ``0.1``.
    steps : int, optional
        Number of fixed local/global fitting iterations. Zero returns a new
        mesh with cloned template points. Default is ``10``.
    cg_tolerance : float, optional
        Positive scaled residual tolerance for the linear solves. Default is
        ``1e-6``.
    cg_max_iterations : int, optional
        Positive maximum iteration count for each linear solve. Default is
        ``256``.
    implementation : {"warp", "torch"} or None, optional
        Correspondence-search backend. ``None`` selects Warp for CUDA float32
        coordinates when available and Torch otherwise.

    Returns
    -------
    Mesh
        New mesh with fitted points and the template's connectivity and
        attached fields.

    Notes
    -----
    Inputs are assumed to be prealigned. Attached fields are treated as
    Lagrangian data and are not pushed forward. Geometry-dependent caches are
    invalidated and template topology caches are retained. The fit does not
    guarantee freedom from inverted, degenerate, or self-intersecting cells;
    call :meth:`~physicsnemo.mesh.mesh.Mesh.validate` explicitly when needed.
    """

    template = _validate_triangle_surface(template, "template")
    target = _validate_triangle_surface(target, "target")

    from physicsnemo.nn.functional import fit_template_points

    points = fit_template_points(
        template.points,
        template.cells,
        target.points,
        target.cells,
        fit_weight=fit_weight,
        arap_weight=arap_weight,
        steps=steps,
        cg_tolerance=cg_tolerance,
        cg_max_iterations=cg_max_iterations,
        implementation=implementation,
    )
    return _mesh_with_deformed_points(template, points)


__all__ = ["fit_template"]
