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

"""Lattice free-form deformation for simplicial meshes."""

import math
from collections.abc import Sequence
from numbers import Real
from typing import TYPE_CHECKING, Literal, TypeAlias

import torch
from jaxtyping import Bool, Float

from physicsnemo.mesh.transformations.deform._utils import (
    _mesh_with_deformed_points,
    _resolve_point_field,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


_FFDBasis: TypeAlias = Literal[
    "bernstein", "bspline", "linear", "cubic_hermite", "quintic_hermite"
]


def _origin_tensor_for_extent(
    origin: Float[torch.Tensor, " n_spatial_dims"] | Sequence[float],
    points: Float[torch.Tensor, "n_points n_spatial_dims"],
) -> Float[torch.Tensor, " n_spatial_dims"]:
    """Validate an explicit origin before using it to derive an extent."""
    num_dims = points.shape[-1]
    if isinstance(origin, torch.Tensor):
        if origin.requires_grad:
            raise ValueError(
                "origin is non-differentiable lattice configuration and must not "
                "require grad. Optimize control_displacements instead"
            )
        if tuple(origin.shape) != (num_dims,):
            raise ValueError(
                f"origin must have shape ({num_dims},), got {tuple(origin.shape)}"
            )
        if origin.device != points.device:
            raise ValueError(
                "points and origin must be on the same device, got "
                f"{points.device} and {origin.device}"
            )
        if origin.dtype != points.dtype:
            raise TypeError(
                "points and origin must have the same dtype, got "
                f"{points.dtype} and {origin.dtype}"
            )
        return origin
    if isinstance(origin, Sequence) and not isinstance(origin, (str, bytes)):
        if len(origin) != num_dims or not all(
            isinstance(entry, Real) and not isinstance(entry, bool) for entry in origin
        ):
            raise TypeError(
                f"origin must contain exactly {num_dims} real values, got {origin!r}"
            )
        values = [float(entry) for entry in origin]
        if not all(math.isfinite(entry) for entry in values):
            raise ValueError(f"origin values must be finite, got {origin!r}")
        if any(abs(entry) > torch.finfo(points.dtype).max for entry in values):
            raise ValueError(
                f"origin values must be finite in the points dtype {points.dtype}, "
                f"got {origin!r}"
            )
        return torch.tensor(values, dtype=points.dtype, device=points.device)
    raise TypeError(
        f"origin must be a torch.Tensor or a sequence of {num_dims} reals, got "
        f"{type(origin).__name__}"
    )


def _default_lattice_box(
    points: Float[torch.Tensor, "n_points n_spatial_dims"],
    origin: Float[torch.Tensor, " n_spatial_dims"] | Sequence[float] | None,
    extent: Float[torch.Tensor, " n_spatial_dims"] | Sequence[float] | None,
) -> tuple[
    Float[torch.Tensor, " n_spatial_dims"] | Sequence[float],
    Float[torch.Tensor, " n_spatial_dims"] | Sequence[float],
]:
    """Derive missing lattice-box values from the axis-aligned point bounds.

    Validating a derived extent synchronizes with the device and is not CUDA
    Graph capture-safe. Pass explicit device-tensor ``origin`` and ``extent``
    to avoid bounds reductions in performance-critical or captured loops.
    """
    if points.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            f"points must have dtype torch.float32 or torch.float64, got {points.dtype}"
        )
    if origin is not None and extent is not None:
        return origin, extent

    if points.shape[0] == 0:
        if origin is None:
            origin = torch.zeros(
                points.shape[-1], dtype=points.dtype, device=points.device
            )
        if extent is None:
            extent = torch.ones(
                points.shape[-1], dtype=points.dtype, device=points.device
            )
        return origin, extent

    with torch.no_grad():
        if origin is None:
            origin = points.amin(dim=0)
        if extent is None:
            maximum = points.amax(dim=0)
            origin_t = _origin_tensor_for_extent(origin, points)
            origin = origin_t
            extent = maximum - origin_t.detach()
            if not bool((torch.isfinite(extent) & (extent > 0)).all()):
                raise ValueError(
                    "the derived lattice extent must be finite and strictly positive "
                    "along every axis. The point bounds are degenerate or non-finite, "
                    "or the supplied origin is not below their maximum. Supply an "
                    "explicit extent."
                )
    return origin, extent


def free_form_deform(
    mesh: "Mesh",
    control_displacements: Float[torch.Tensor, "*lattice_resolution n_spatial_dims"],
    *,
    origin: Float[torch.Tensor, " n_spatial_dims"] | Sequence[float] | None = None,
    extent: Float[torch.Tensor, " n_spatial_dims"] | Sequence[float] | None = None,
    basis: _FFDBasis = "bernstein",
    point_weights: str
    | tuple[str, ...]
    | Bool[torch.Tensor, " n_points"]
    | Float[torch.Tensor, " n_points"]
    | None = None,
    implementation: Literal["torch", "warp"] | None = None,
) -> "Mesh":
    """Deform a mesh with a control-point lattice by free-form deformation.

    An ``n_1 x ... x n_D`` array of control displacements defines a field over
    the axis-aligned box ``[origin, origin + extent]``. Mesh points inside the
    box move with the tensor-product basis interpolation of those values.
    Points outside the box are unchanged. A lattice of zero displacements is
    exactly the identity, and a constant lattice translates every point inside
    the box.

    Parameters
    ----------
    mesh : Mesh
        Mesh whose points are deformed. The source mesh is not modified.
    control_displacements : torch.Tensor
        Displacement vectors, not destination coordinates, for every lattice
        node, with shape ``(n_1, ..., n_D, mesh.n_spatial_dims)`` and the same
        float32 or float64 dtype and device as ``mesh.points``. Each axis needs
        at least two nodes for ``"bernstein"`` and the node-interpolating
        bases, and four for ``"bspline"``.
    origin : torch.Tensor, sequence of float, or None, optional
        Minimum corner of the lattice box with shape
        ``(mesh.n_spatial_dims,)``. ``None`` uses the minimum corner of the
        mesh bounds. For repeated GPU calls with an explicit box, create
        ``origin`` and ``extent`` once as device tensors. Reuse them to avoid
        recreating and transferring sequence values. Default is ``None``.
    extent : torch.Tensor, sequence of float, or None, optional
        Edge lengths of the lattice box with the same accepted shapes as
        ``origin``. Every value must be finite and strictly positive. The
        operation does not validate tensor values at runtime. ``None`` sizes
        the box from ``origin`` to the maximum corner of the mesh bounds.
        Validating a derived extent synchronizes with the device and is not
        CUDA Graph capture-safe. For capture, pass both ``origin`` and
        ``extent`` as device tensors. Every point-coordinate axis must have
        positive range when the extent is derived. Supply an explicit extent
        for lower-dimensional geometry embedded in a higher-dimensional space.
        Default is ``None``.
    basis : {"bernstein", "bspline", "linear", "cubic_hermite", "quintic_hermite"}, optional
        Per-axis basis family:

        - ``"bernstein"`` provides classic global-support FFD. Every lattice
          node influences every point inside the box.
        - ``"bspline"`` uses a uniform cubic B-spline with local
          four-node-per-axis support and C2 continuity between knot spans.
          Coefficient index ``i`` corresponds to local coordinate
          ``(i - 1) / (n - 3)``. The first and last coefficient planes lie
          outside the evaluation box.
        - ``"linear"`` uses upper-node weight :math:`s(t)=t` within each
          lattice cell. It interpolates every node. It is continuous (C0)
          across cell boundaries, but its slope can jump.
        - ``"cubic_hermite"`` uses the cubic Hermite blend
          :math:`s(t)=3t^2-2t^3`. Its first derivative vanishes at both cell
          endpoints. This gives C1 continuity across cell boundaries.
        - ``"quintic_hermite"`` uses the quintic Hermite blend
          :math:`s(t)=6t^5-15t^4+10t^3`. Its first and second derivatives
          vanish at both endpoints. This gives C2 continuity across cell
          boundaries. Perlin introduced this improved interpolant in
          "Improving Noise" [1].

        The node-interpolating bases use only the two neighboring nodes per
        axis. Here, ``t`` is the local cell coordinate in ``[0, 1]``. The
        upper-node weight is :math:`s(t)`, and the lower-node weight is
        :math:`1-s(t)`. Default is ``"bernstein"``.
    point_weights : str, tuple[str, ...], torch.Tensor, or None, optional
        Optional bool or floating mesh-point weights with shape
        ``(mesh.n_points,)``, or a
        :attr:`~physicsnemo.mesh.mesh.Mesh.point_data` key resolving to those
        point weights. All weights must match the point device. Floating
        weights must also match the point dtype. Default is ``None``.
    implementation : {"torch", "warp"} or None, optional
        Backend override. ``None`` selects Torch on CPU. On CUDA, it selects
        Warp when available and otherwise Torch.

    Returns
    -------
    Mesh
        New mesh with deformed points and unchanged connectivity and fields.

    Raises
    ------
    TypeError
        If tensors, lattice values, or point weights have unsupported types or
        dtypes.
    ValueError
        If shapes, devices, lattice parameters, point weights, or ``basis``
        are invalid.
    KeyError
        If a point-data key or ``implementation`` name is not found.
    ImportError
        If an explicitly requested backend is unavailable.

    Notes
    -----
    The operation treats attached fields as Lagrangian data and does not push
    them forward. It invalidates geometry-dependent caches and retains topology
    caches. The deformation is generally not continuous across the lattice box
    boundary. To keep the exterior fixed, zero the outermost coefficient plane
    on every Bernstein or node-interpolating face. For cubic B-splines, zero the
    first and last three coefficient planes on every axis. ``origin`` and
    ``extent`` are non-differentiable lattice parameters. Optimize
    ``control_displacements`` instead. The operation does not detect or repair
    inverted, degenerate, or self-intersecting cells. Call
    :meth:`~physicsnemo.mesh.mesh.Mesh.validate` explicitly when needed.

    References
    ----------
    [1] Perlin, K. (2002). "Improving Noise." ACM Transactions on Graphics,
    21(3), 681-682. https://doi.org/10.1145/566654.566636
    """
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
    origin, extent = _default_lattice_box(mesh.points, origin, extent)
    from physicsnemo.nn.functional.geometry.deform import free_form_deform_points

    points = free_form_deform_points(
        mesh.points,
        control_displacements,
        origin=origin,
        extent=extent,
        basis=basis,
        point_weights=point_weights_t,
        implementation=implementation,
    )
    return _mesh_with_deformed_points(mesh, points)
