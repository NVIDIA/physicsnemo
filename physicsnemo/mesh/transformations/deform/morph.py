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

"""Sparse control-point morphing for simplicial meshes."""

from typing import TYPE_CHECKING, Literal

import torch

from physicsnemo.mesh.transformations.deform._utils import (
    _mesh_with_deformed_points,
    _resolve_point_field,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def morph(
    mesh: "Mesh",
    control_points: torch.Tensor,
    control_displacements: torch.Tensor,
    *,
    radius: float | torch.Tensor,
    point_weights: str | tuple[str, ...] | torch.Tensor | None = None,
    kernel: Literal["wendland_c2"] = "wendland_c2",
    implementation: Literal["torch", "warp"] | None = None,
) -> "Mesh":
    """Morph a mesh from sparse, compactly supported control displacements.

    Control influence uses a Wendland-C2 compact Shepard field with a stationary
    zero-displacement background. Each control's influence vanishes smoothly at
    its support boundary. Where supports overlap, all active controls and the
    background are blended together; the result is not a simple sum or average.
    The field is zero outside the union of all supports.

    With no point weights, the field at a unique control coordinate is exactly
    that control's displacement. Duplicate controls at one coordinate contribute
    their mean displacement. A control may be anywhere in world coordinates and
    need not coincide with a mesh point.

    Parameters
    ----------
    mesh : Mesh
        Mesh whose points are morphed. The source mesh is not modified.
    control_points : torch.Tensor
        World-coordinate controls with shape
        ``(n_controls, mesh.n_spatial_dims)`` and the same dtype and device as
        ``mesh.points``.
    control_displacements : torch.Tensor
        Displacement vectors, not destination coordinates, with exactly the
        same shape, dtype, and device as ``control_points``.
    radius : float or torch.Tensor
        Support distance in mesh coordinate units. Supply one scalar for every
        control or a tensor with shape ``(n_controls,)`` that matches the control
        dtype and device. Every tensor value must remain positive and finite;
        values are not validated at runtime.
    point_weights : str, tuple[str, ...], torch.Tensor, or None, optional
        Optional bool or floating mesh-point weights with shape
        ``(mesh.n_points,)``, or a
        :attr:`~physicsnemo.mesh.mesh.Mesh.point_data` key resolving to those
        point weights. These are query-point weights, not per-control values.
        Default is ``None``.
    kernel : {"wendland_c2"}, optional
        Compact radial kernel used to blend control displacements. Default is
        ``"wendland_c2"``.
    implementation : {"torch", "warp"} or None, optional
        Backend override. ``None`` selects Torch on CPU and Warp on CUDA when
        Warp is available, otherwise Torch.

    Returns
    -------
    Mesh
        New mesh with morphed points and unchanged connectivity and attached
        fields.

    Notes
    -----
    Attached fields are treated as Lagrangian data and are not pushed forward.
    Geometry-dependent caches are invalidated and topology caches are retained.
    Parameterize learned radii to remain positive, for example as
    ``torch.nn.functional.softplus(raw_radius) + eps``.
    The operation does not detect or repair inverted, degenerate, or
    self-intersecting cells; call
    :meth:`~physicsnemo.mesh.mesh.Mesh.validate` explicitly when needed.
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
    from physicsnemo.nn.functional.geometry.deform import morph_points

    points = morph_points(
        mesh.points,
        control_points,
        control_displacements,
        radius=radius,
        point_weights=point_weights_t,
        kernel=kernel,
        implementation=implementation,
    )
    return _mesh_with_deformed_points(mesh, points)
