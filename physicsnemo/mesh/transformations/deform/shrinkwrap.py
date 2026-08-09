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

"""Nearest-surface shrinkwrap deformation for meshes."""

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


_INTEGER_CELL_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
}


def shrinkwrap(
    mesh: "Mesh",
    target: "Mesh",
    *,
    offset: float | Float[torch.Tensor, ""] = 0.0,
    max_distance: float | None = None,
    point_weights: str
    | tuple[str, ...]
    | Bool[torch.Tensor, " n_points"]
    | Float[torch.Tensor, " n_points"]
    | None = None,
    implementation: Literal["torch", "warp"] | None = None,
) -> "Mesh":
    r"""Project mesh vertices onto the nearest locations of a target surface.

    For each source vertex :math:`x_i`, Shrinkwrap selects the closest point
    :math:`p_i` on ``target`` and returns

    .. math::

       x'_i = x_i + w_i\left(p_i + \delta n_i - x_i\right),

    where :math:`w_i` is an optional point weight, :math:`\delta` is ``offset``,
    and :math:`n_i` is the oriented unit normal of the selected target face.
    Source connectivity and attached data are retained.

    Call it as ``shrinkwrap(mesh, target, ...)`` or as
    ``mesh.shrinkwrap(target, ...)``. Neither source nor target is modified.

    Parameters
    ----------
    mesh : Mesh
        Source mesh whose 3D point coordinates are projected. Its connectivity
        may have any supported topology.
    target : Mesh
        Nonempty triangle surface embedded in 3D. Source and target points must
        share their float32 or float64 dtype and device. Target connectivity may
        use any non-bool integer dtype supported by :class:`Mesh` and is
        normalized to int64 for this operation.
    offset : float or torch.Tensor, optional
        Signed scalar distance from the target along the selected face normal.
        A scalar tensor must match the mesh point dtype and device and may
        require gradients. The value must be finite. Positive values follow
        target face winding. Default is ``0.0``.
    max_distance : float or None, optional
        Positive finite nearest-surface search radius in mesh coordinate units.
        Vertices without a target strictly closer than this distance remain
        unchanged. Values that round to zero in the mesh point dtype are
        rejected. ``None`` performs an unbounded search. Default is ``None``.
    point_weights : str, tuple[str, ...], torch.Tensor, or None, optional
        Optional bool or floating per-source-point weights, or a key/path in
        :attr:`~physicsnemo.mesh.mesh.Mesh.point_data` resolving to one. Zero
        leaves a vertex unchanged and one applies the full projection. Floating
        values are not clamped. Default is ``None``.
    implementation : {"torch", "warp"} or None, optional
        Nearest-face backend. ``None`` selects Torch on CPU and Warp on CUDA
        when available.

    Returns
    -------
    Mesh
        New mesh with projected points, preserved connectivity and data,
        invalidated geometry caches, and retained topology caches.

    Notes
    -----
    Nearest-face selection is discrete. Away from target face, edge, vertex,
    and distance-cutoff transitions, gradients propagate through source points,
    selected target vertices, floating point weights, and tensor-valued
    ``offset``. Degenerate target faces are ignored. Target orientation must be
    consistent when using a nonzero offset.

    Shrinkwrap does not detect or prevent inverted or self-intersecting source
    cells. Partial point weights interpolate toward the projected surface and
    generally do not place the resulting vertex exactly on it. Shrinkwrap is
    not supported inside CUDA Graph capture with either backend. Float64
    targets use the Torch search because Warp searches in float32. Safe float32
    coordinates are searched unchanged. Warp falls back to Torch for unsafe
    coordinate magnitudes or face geometry.
    """

    from physicsnemo.mesh.mesh import Mesh

    if not isinstance(target, Mesh):
        raise TypeError(f"target must be a Mesh, got {type(target).__name__}")
    if target.cells.dtype not in _INTEGER_CELL_DTYPES:
        raise TypeError(
            f"target cells must have a non-bool integer dtype, got {target.cells.dtype}"
        )
    target_faces = target.cells.to(torch.int64)

    resolved_weights = (
        None
        if point_weights is None
        else _resolve_point_field(
            mesh,
            point_weights,
            argument_name="point_weights",
        )
    )

    from physicsnemo.nn.functional.geometry.deform import shrinkwrap_points

    points = shrinkwrap_points(
        mesh.points,
        target.points,
        target_faces,
        offset=offset,
        max_distance=max_distance,
        point_weights=resolved_weights,
        implementation=implementation,
    )
    return _mesh_with_deformed_points(mesh, points)


__all__ = ["shrinkwrap"]
