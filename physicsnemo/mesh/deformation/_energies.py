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

"""Mesh-aware wrappers for differentiable deformation energies."""

from __future__ import annotations

from typing import Literal

import torch
from jaxtyping import Float, Int

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.utilities._topology import (
    _deformation_energy_cells,
    _validate_closed_oriented_triangle_surface,
    extract_triangle_hinges,
)
from physicsnemo.nn.functional.geometry.deform import (
    closed_surface_volume_energy as _closed_surface_volume_energy,
)
from physicsnemo.nn.functional.geometry.deform import (
    simplex_inversion_energy as _simplex_inversion_energy,
)
from physicsnemo.nn.functional.geometry.deform import (
    simplex_measure_energy as _simplex_measure_energy,
)
from physicsnemo.nn.functional.geometry.deform import (
    simplex_strain_energy as _simplex_strain_energy,
)
from physicsnemo.nn.functional.geometry.deform import (
    surface_bending_energy as _surface_bending_energy,
)
from physicsnemo.nn.functional.geometry.deform import (
    total_measure_energy as _total_measure_energy,
)

_Reduction = Literal["none", "sum", "mean"]
_Implementation = Literal["torch", "warp"]


def _validate_points(
    reference_mesh: Mesh,
    points: Float[torch.Tensor, "num_points num_dims"],
) -> None:
    """Validate current coordinates against an unbatched reference mesh."""

    if not isinstance(reference_mesh, Mesh):
        raise TypeError(
            f"reference_mesh must be a Mesh, got {type(reference_mesh).__name__}"
        )
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"points must be a torch.Tensor, got {type(points).__name__}")
    if points.ndim != 2:
        raise ValueError(
            "mesh deformation-energy wrappers require unbatched points with "
            f"shape (N, D), got {tuple(points.shape)}"
        )
    if points.shape != reference_mesh.points.shape:
        raise ValueError(
            "points must have the same shape as reference_mesh.points; got "
            f"{tuple(points.shape)} and {tuple(reference_mesh.points.shape)}"
        )
    if points.dtype != reference_mesh.points.dtype:
        raise TypeError(
            "points and reference_mesh.points must have the same dtype; got "
            f"{points.dtype} and {reference_mesh.points.dtype}"
        )
    if points.device != reference_mesh.points.device:
        raise ValueError(
            "points and reference_mesh.points must be on the same device; got "
            f"{points.device} and {reference_mesh.points.device}"
        )


def _validate_simplex_inputs(
    reference_mesh: Mesh,
    points: Float[torch.Tensor, "num_points num_dims"],
) -> Int[torch.Tensor, "num_cells simplex_vertices"]:
    """Validate geometry and return normalized simplex connectivity."""

    _validate_points(reference_mesh, points)
    return _deformation_energy_cells(reference_mesh)


def simplex_strain_energy(
    reference_mesh: Mesh,
    points: Float[torch.Tensor, "num_points num_dims"],
    *,
    lame_lambda: float = 1.0,
    shear_modulus: float = 1.0,
    reduction: _Reduction = "sum",
    implementation: _Implementation | None = None,
) -> Float[torch.Tensor, "..."]:
    r"""Evaluate reference-relative St. Venant--Kirchhoff strain energy.

    The reference mesh supplies undeformed coordinates and simplex
    connectivity. ``points`` supplies current coordinates with the exact same
    unbatched shape, dtype, and device. The energy supports edge, triangle, and
    tetrahedral meshes, including simplices embedded in a higher-dimensional
    space.

    Parameters
    ----------
    reference_mesh : Mesh
        Undeformed reference mesh.
    points : torch.Tensor
        Current point coordinates with shape ``(N, D)``.
    lame_lambda : float, default=1.0
        First Lamé parameter. It may be negative, but
        ``lame_lambda + 2 * shear_modulus / m`` must be nonnegative for simplex
        dimension ``m``.
    shear_modulus : float, default=1.0
        Nonnegative shear modulus (second Lamé parameter).
    reduction : {"none", "sum", "mean"}, default="sum"
        Reduction applied to per-simplex contributions.
    implementation : {"torch", "warp"}, optional
        Backend override. By default, dispatch follows the point device.

    Returns
    -------
    torch.Tensor
        Per-simplex contributions for ``reduction="none"`` or a scalar energy.
    """

    cells = _validate_simplex_inputs(reference_mesh, points)
    return _simplex_strain_energy(
        points,
        reference_mesh.points,
        cells,
        lame_lambda=lame_lambda,
        shear_modulus=shear_modulus,
        reduction=reduction,
        implementation=implementation,
    )


def simplex_measure_energy(
    reference_mesh: Mesh,
    points: Float[torch.Tensor, "num_points num_dims"],
    *,
    target_ratio: float = 1.0,
    reduction: _Reduction = "sum",
    implementation: _Implementation | None = None,
) -> Float[torch.Tensor, "..."]:
    """Penalize local simplex length, area, or volume changes.

    Each simplex is compared independently with its reference measure. For a
    global conservation constraint that permits local redistribution, use
    :func:`total_measure_energy`.

    Parameters
    ----------
    reference_mesh : Mesh
        Undeformed edge, triangle, or tetrahedral mesh.
    points : torch.Tensor
        Current point coordinates with shape ``(N, D)``.
    target_ratio : float, default=1.0
        Strictly positive desired current-to-reference measure ratio for every
        simplex.
    reduction : {"none", "sum", "mean"}, default="sum"
        Reduction applied to per-simplex contributions.
    implementation : {"torch", "warp"}, optional
        Backend override. By default, dispatch follows the point device.

    Returns
    -------
    torch.Tensor
        Per-simplex contributions for ``reduction="none"`` or a scalar energy.
    """

    cells = _validate_simplex_inputs(reference_mesh, points)
    return _simplex_measure_energy(
        points,
        reference_mesh.points,
        cells,
        target_ratio=target_ratio,
        reduction=reduction,
        implementation=implementation,
    )


def total_measure_energy(
    reference_mesh: Mesh,
    points: Float[torch.Tensor, "num_points num_dims"],
    *,
    target_ratio: float = 1.0,
    reduction: _Reduction = "sum",
    implementation: _Implementation | None = None,
) -> Float[torch.Tensor, "..."]:
    """Penalize change in the mesh's total intrinsic measure.

    The total length, area, or volume is constrained relative to the reference
    mesh while individual simplices may exchange measure. For a
    full-dimensional mesh, current contributions are signed relative to their
    reference cells. An inverted cell can therefore cancel a non-inverted cell
    in this aggregate objective. Combine it with
    :func:`simplex_inversion_energy` when local orientation matters. Embedded
    simplex contributions are unsigned. The reference mesh must contain at
    least one simplex.

    Parameters
    ----------
    reference_mesh : Mesh
        Undeformed edge, triangle, or tetrahedral mesh.
    points : torch.Tensor
        Current point coordinates with shape ``(N, D)``.
    target_ratio : float, default=1.0
        Strictly positive desired current-to-reference total measure ratio.
    reduction : {"none", "sum", "mean"}, default="sum"
        Reduction applied to the global constraint term.
    implementation : {"torch", "warp"}, optional
        Backend override. By default, dispatch follows the point device.

    Returns
    -------
    torch.Tensor
        Global measure constraint energy.

    Raises
    ------
    ValueError
        If the reference mesh has no simplices.
    """

    cells = _validate_simplex_inputs(reference_mesh, points)
    return _total_measure_energy(
        points,
        reference_mesh.points,
        cells,
        target_ratio=target_ratio,
        reduction=reduction,
        implementation=implementation,
    )


def simplex_inversion_energy(
    reference_mesh: Mesh,
    points: Float[torch.Tensor, "num_points num_dims"],
    *,
    minimum_jacobian: float = 0.1,
    reduction: _Reduction = "sum",
    implementation: _Implementation | None = None,
) -> Float[torch.Tensor, "..."]:
    """Penalize full-dimensional simplices below a Jacobian threshold.

    Signed Jacobians require the manifold and spatial dimensions to match.
    Embedded curves and surfaces do not have an intrinsic inversion sign and
    are rejected.

    Parameters
    ----------
    reference_mesh : Mesh
        Undeformed full-dimensional edge, triangle, or tetrahedral mesh.
    points : torch.Tensor
        Current point coordinates with shape ``(N, D)``.
    minimum_jacobian : float, default=0.1
        Nonnegative relative signed-Jacobian threshold below which the penalty
        is active.
    reduction : {"none", "sum", "mean"}, default="sum"
        Reduction applied to per-simplex contributions.
    implementation : {"torch", "warp"}, optional
        Backend override. By default, dispatch follows the point device.

    Returns
    -------
    torch.Tensor
        Per-simplex contributions for ``reduction="none"`` or a scalar energy.
    """

    cells = _validate_simplex_inputs(reference_mesh, points)
    if reference_mesh.n_manifold_dims != reference_mesh.n_spatial_dims:
        raise ValueError(
            "simplex inversion energy requires full-dimensional simplices; got "
            f"manifold dimension {reference_mesh.n_manifold_dims} in spatial "
            f"dimension {reference_mesh.n_spatial_dims}"
        )
    return _simplex_inversion_energy(
        points,
        reference_mesh.points,
        cells,
        minimum_jacobian=minimum_jacobian,
        reduction=reduction,
        implementation=implementation,
    )


def surface_bending_energy(
    reference_mesh: Mesh,
    points: Float[torch.Tensor, "num_points 3"],
    *,
    reduction: _Reduction = "sum",
    implementation: _Implementation | None = None,
) -> Float[torch.Tensor, "..."]:
    """Evaluate a reference-relative hinge energy on a triangle surface.

    Interior edge hinges are extracted from connectivity once and cached on
    the reference mesh. Connectivity must contain unique triangles, and each
    edge may be incident to at most two triangles. Boundary edges are omitted
    and do not contribute. The energy requires a triangle surface embedded in
    three dimensions.

    Parameters
    ----------
    reference_mesh : Mesh
        Undeformed triangle surface in three-dimensional space.
    points : torch.Tensor
        Current point coordinates with shape ``(N, 3)``.
    reduction : {"none", "sum", "mean"}, default="sum"
        Reduction applied to per-hinge contributions.
    implementation : {"torch", "warp"}, optional
        Backend override. By default, dispatch follows the point device.

    Returns
    -------
    torch.Tensor
        Per-hinge contributions for ``reduction="none"`` or a scalar energy.
    """

    _validate_points(reference_mesh, points)
    if reference_mesh.n_manifold_dims != 2 or reference_mesh.n_spatial_dims != 3:
        raise ValueError(
            "surface bending energy requires a triangle surface embedded in "
            "three dimensions"
        )
    hinges = extract_triangle_hinges(reference_mesh)
    return _surface_bending_energy(
        points,
        reference_mesh.points,
        hinges,
        reduction=reduction,
        implementation=implementation,
    )


def closed_surface_volume_energy(
    reference_mesh: Mesh,
    points: Float[torch.Tensor, "num_points 3"],
    *,
    target_ratio: float = 1.0,
    reduction: _Reduction = "sum",
    implementation: _Implementation | None = None,
) -> Float[torch.Tensor, "..."]:
    """Penalize enclosed-volume change of a closed triangle surface.

    The reference surface must be one nonempty, edge-connected, edge-closed
    component with consistently oriented triangles. Either inward or outward
    winding is valid. Evaluate disconnected components separately.
    Self-intersection and vertex-manifoldness are not tested. Connectivity
    validation is cached on the reference mesh.

    Parameters
    ----------
    reference_mesh : Mesh
        Undeformed closed triangle surface in three-dimensional space.
    points : torch.Tensor
        Current point coordinates with shape ``(N, 3)``.
    target_ratio : float, default=1.0
        Strictly positive desired current-to-reference enclosed-volume ratio.
    reduction : {"none", "sum", "mean"}, default="sum"
        Reduction applied to the global constraint term.
    implementation : {"torch", "warp"}, optional
        Backend override. By default, dispatch follows the point device.

    Returns
    -------
    torch.Tensor
        Enclosed-volume constraint energy.
    """

    _validate_points(reference_mesh, points)
    _validate_closed_oriented_triangle_surface(reference_mesh)
    return _closed_surface_volume_energy(
        points,
        reference_mesh.points,
        _deformation_energy_cells(reference_mesh),
        target_ratio=target_ratio,
        reduction=reduction,
        implementation=implementation,
    )
