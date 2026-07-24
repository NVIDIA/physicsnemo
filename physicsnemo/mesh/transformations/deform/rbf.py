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

"""Thin-plate-spline radial-basis mesh deformation."""

from typing import TYPE_CHECKING, Literal

import torch
from jaxtyping import Bool, Float

from physicsnemo.mesh.transformations.deform._utils import (
    _mesh_with_deformed_points,
    _resolve_point_field,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def radial_basis_function_deform(
    mesh: "Mesh",
    control_points: Float[torch.Tensor, "n_controls n_spatial_dims"],
    control_displacements: Float[torch.Tensor, "n_controls n_spatial_dims"],
    *,
    kernel: Literal["thin_plate_spline"] = "thin_plate_spline",
    polynomial: bool = True,
    smoothing: float = 0.0,
    point_weights: str
    | tuple[str, ...]
    | Bool[torch.Tensor, " n_points"]
    | Float[torch.Tensor, " n_points"]
    | None = None,
    implementation: Literal["torch", "warp"] | None = None,
) -> "Mesh":
    """Deform a mesh with a global thin-plate-spline RBF field.

    A thin-plate-spline radial field is fitted to the prescribed sparse
    control displacements and evaluated at every mesh point. With the default
    affine polynomial tail, zero smoothing, and a nonsingular control layout,
    the unweighted field interpolates every control displacement up to solver
    precision.

    Parameters
    ----------
    mesh : Mesh
        Mesh whose points are deformed. The source mesh is not modified.
    control_points : torch.Tensor
        World-coordinate controls with shape
        ``(n_controls, mesh.n_spatial_dims)`` and the same dtype and device as
        ``mesh.points``.
    control_displacements : torch.Tensor
        Displacement vectors, not destination coordinates, with exactly the
        same shape, dtype, and device as ``control_points``.
    kernel : {"thin_plate_spline"}, optional
        Radial kernel used by the interpolant. Default is
        ``"thin_plate_spline"``.
    polynomial : bool, optional
        Add the standard affine polynomial tail and side constraints. This
        reproduces affine displacement fields. The controls must affinely span
        the coordinate space, and the augmented system must be nonsingular.
        Default is ``True``.
    smoothing : float, optional
        Nonnegative diagonal regularization added to the radial system. Zero
        gives exact interpolation for a nonsingular control layout up to solver
        precision. Positive values relax interpolation accuracy. Default is
        ``0.0``.
    point_weights : str, tuple[str, ...], torch.Tensor, or None, optional
        Optional bool or floating mesh-point weights with shape
        ``(mesh.n_points,)``, or a
        :attr:`~physicsnemo.mesh.mesh.Mesh.point_data` key resolving to those
        weights. Values scale or mask the fitted field after interpolation.
        Bool weights must be on the same device as the mesh points. Floating
        weights must have the same dtype and device as the mesh points.
    implementation : {"torch", "warp"} or None, optional
        Evaluation-backend override. Both backends use PyTorch for the dense
        coefficient solve. ``None`` selects Torch on CPU and Warp on CUDA when
        Warp is available, otherwise Torch.

    Returns
    -------
    Mesh
        New mesh with deformed points and unchanged connectivity and attached
        fields.

    Raises
    ------
    TypeError
        If control tensors or Python arguments have unsupported types, or if
        tensor dtypes are unsupported or mismatched.
    ValueError
        If tensor shapes, devices, control layout, point weights, or RBF
        options are invalid.
    KeyError
        If a point-data key is missing or ``implementation`` does not name a
        registered backend.
    ImportError
        If an explicitly requested backend is unavailable.
    RuntimeError
        If runtime validation or coefficient fitting fails, including for a
        singular system or during CUDA Graph capture.

    Notes
    -----
    The field has global support. Unlike compact Shepard morphing, every
    control generally influences every mesh point. Attached fields are treated
    as Lagrangian data and are not pushed forward. Geometry-dependent caches
    are invalidated and topology caches are retained. The operation does not
    detect or repair inverted, degenerate, or self-intersecting cells. Call
    :meth:`~physicsnemo.mesh.mesh.Mesh.validate` explicitly when needed.
    Coefficient fitting is not supported inside CUDA Graph capture because the
    singular-system check requires host interaction.
    """
    if not isinstance(control_points, torch.Tensor):
        raise TypeError(
            "control_points must be a torch.Tensor, got "
            f"{type(control_points).__name__}"
        )
    if not isinstance(control_displacements, torch.Tensor):
        raise TypeError(
            "control_displacements must be a torch.Tensor, got "
            f"{type(control_displacements).__name__}"
        )
    point_weights_t = (
        None
        if point_weights is None
        else _resolve_point_field(mesh, point_weights, argument_name="point_weights")
    )

    from physicsnemo.nn.functional.geometry.deform import (
        radial_basis_function_deform_points,
    )

    points = radial_basis_function_deform_points(
        mesh.points,
        control_points,
        control_displacements,
        kernel=kernel,
        polynomial=polynomial,
        smoothing=smoothing,
        point_weights=point_weights_t,
        implementation=implementation,
    )
    return _mesh_with_deformed_points(mesh, points)


__all__ = ["radial_basis_function_deform"]
