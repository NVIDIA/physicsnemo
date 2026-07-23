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

"""Dense displacement for simplicial meshes."""

from typing import TYPE_CHECKING, Literal

import torch

from physicsnemo.mesh.transformations.deform._utils import (
    _mesh_with_deformed_points,
    _resolve_point_field,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def displace(
    mesh: "Mesh",
    displacement: str | tuple[str, ...] | torch.Tensor,
    *,
    point_weights: str | tuple[str, ...] | torch.Tensor | None = None,
    implementation: Literal["torch"] | None = None,
) -> "Mesh":
    """Displace every mesh point by a dense vector field.

    Computes ``points + displacement`` without changing connectivity, optionally
    multiplying the displacement by ``point_weights[..., None]``.
    ``displacement`` and ``point_weights`` may be raw tensors or keys (including
    nested tuple keys) in
    :attr:`~physicsnemo.mesh.mesh.Mesh.point_data`.

    Parameters
    ----------
    mesh : Mesh
        Mesh whose points are displaced. The source mesh is not modified.
    displacement : str, tuple[str, ...], or torch.Tensor
        Dense displacement vectors with shape
        ``(mesh.n_points, mesh.n_spatial_dims)``, or a point-data key resolving
        to such a tensor. The tensor and ``mesh.points`` must have the same
        float32 or float64 dtype and device.
    point_weights : str, tuple[str, ...], torch.Tensor, or None, optional
        Optional bool or floating-point weights with shape
        ``(mesh.n_points,)``, or a point-data key resolving to those point weights.
        Floating-point weights may be signed or greater than one. Default is
        ``None``.
    implementation : {"torch"} or None, optional
        Backend override. ``None`` selects Torch for dense displacement.

    Returns
    -------
    Mesh
        New mesh with displaced points and unchanged connectivity and attached
        fields.

    Notes
    -----
    Attached fields are treated as Lagrangian data and are not pushed forward.
    Geometry-dependent caches are invalidated and topology caches are retained.
    The operation does not detect or repair inverted, degenerate, or
    self-intersecting cells; call
    :meth:`~physicsnemo.mesh.mesh.Mesh.validate` explicitly when needed.
    """
    displacement_t = _resolve_point_field(
        mesh, displacement, argument_name="displacement"
    )
    point_weights_t = (
        None
        if point_weights is None
        else _resolve_point_field(mesh, point_weights, argument_name="point_weights")
    )
    from physicsnemo.nn.functional.geometry.deform import displace_points

    points = displace_points(
        mesh.points,
        displacement_t,
        point_weights=point_weights_t,
        implementation=implementation,
    )
    return _mesh_with_deformed_points(mesh, points)
