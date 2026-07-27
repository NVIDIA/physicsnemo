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

"""Uniform-mass P1 Sobolev deformation for simplicial meshes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from jaxtyping import Bool, Float

from physicsnemo.mesh.transformations.deform._utils import (
    _mesh_with_deformed_points,
    _resolve_point_field,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def sobolev_deform(
    mesh: "Mesh",
    displacement: str
    | tuple[str, ...]
    | Float[torch.Tensor, "n_points n_spatial_dims"],
    *,
    length_scale: float,
    fixed_points: str | tuple[str, ...] | Bool[torch.Tensor, " n_points"] | None = None,
    max_iterations: int = 128,
    tolerance: float | None = None,
    implementation: Literal["torch", "warp"] | None = None,
) -> "Mesh":
    r"""Deform a mesh with a uniform-mass P1 Sobolev displacement.

    Filters a dense per-vertex displacement by solving

    .. math::

       (M + \ell^2 K)u = M d

    and returns a new mesh with points ``mesh.points + u``. Here
    :math:`M=\bar m I` is a uniform vertex mass matrix scaled by the mean
    positive lumped P1 mass. :math:`K` is the P1 stiffness matrix, and
    :math:`\ell` is ``length_scale`` in mesh coordinate units. The uniform mass
    makes the filter self-adjoint for PyTorch vertex tensors. Connectivity and
    attached fields are unchanged.

    Call it as ``sobolev_deform(mesh, ...)`` or as
    ``mesh.sobolev_deform(...)``. The bound method supplies ``mesh``
    automatically.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh whose points are deformed. The source is not modified.
    displacement : str, tuple[str, ...], or torch.Tensor
        Raw displacement with shape
        ``(mesh.n_points, mesh.n_spatial_dims)``, or a
        :attr:`~physicsnemo.mesh.mesh.Mesh.point_data` key resolving to one.
        It must match the point dtype and device.
    length_scale : float
        Nonnegative physical smoothing length. Zero applies the raw
        displacement directly at unfixed points.
    fixed_points : str, tuple[str, ...], torch.Tensor, or None, optional
        Optional bool mask with shape ``(mesh.n_points,)``, or a point-data key
        resolving to one. True entries receive zero displacement. Default is
        ``None``.
    max_iterations : int, optional
        Maximum PCG iterations. Default is ``128``.
    tolerance : float or None, optional
        Positive relative residual tolerance. ``None`` selects a
        dtype-dependent default. Default is ``None``.
    implementation : {"torch", "warp"} or None, optional
        Backend override. ``None`` selects Torch on CPU. On CUDA, it selects
        Warp for segments, triangles, and tetrahedra when available. It
        otherwise selects Torch, with a one-time :class:`RuntimeWarning` when
        Warp is unavailable. The Warp backend requires CUDA tensors.

    Returns
    -------
    Mesh
        New mesh with Sobolev-filtered points, unchanged connectivity, and
        unchanged attached fields.

    Notes
    -----
    Unfixed mesh boundaries use the natural homogeneous Neumann condition.
    Constant displacements are retained when no points are fixed. Isolated
    points receive their raw displacement.

    Both backends participate in autograd through the source points and the raw
    displacement. Their reverse-mode derivatives solve the adjoint Helmholtz
    system, which makes the operation suitable for smooth vertex-based
    optimization. The Warp backend evaluates the geometry vector-Jacobian
    product analytically. A forward or adjoint solve that does not reach
    ``tolerance`` within ``max_iterations`` raises a :class:`RuntimeError`.
    Warp supports segment, triangle, and tetrahedron cells. Higher-dimensional
    simplices use Torch by default.
    Warp CUDA results and point gradients may vary at roundoff between runs.
    CUDA Graph capture is not supported because P1 operator assembly and
    solver diagnostics are not capture-safe.

    Geometry caches are invalidated and topology caches are retained. At
    positive length scales, cells must be finite, nondegenerate simplices. The
    operation does not detect inverted or self-intersecting output cells.
    """

    displacement_t = _resolve_point_field(
        mesh,
        displacement,
        argument_name="displacement",
    )
    fixed_points_t = (
        None
        if fixed_points is None
        else _resolve_point_field(
            mesh,
            fixed_points,
            argument_name="fixed_points",
        )
    )

    from physicsnemo.nn.functional.geometry.deform import sobolev_deform_points

    points = sobolev_deform_points(
        mesh.points,
        mesh.cells,
        displacement_t,
        length_scale=length_scale,
        fixed_points=fixed_points_t,
        max_iterations=max_iterations,
        tolerance=tolerance,
        implementation=implementation,
    )
    return _mesh_with_deformed_points(mesh, points)


__all__ = ["sobolev_deform"]
