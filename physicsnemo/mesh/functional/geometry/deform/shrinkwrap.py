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

"""Backend-dispatched nearest-surface shrinkwrap deformation."""

from __future__ import annotations

import math
from numbers import Real
from typing import Literal

import torch
from jaxtyping import Bool, Float, Int

from physicsnemo.core.function_spec import FunctionSpec

from ._shrinkwrap_torch_impl import (
    _valid_triangles,
    nearest_surface_faces_torch,
    replay_shrinkwrap_projection,
)
from ._utils import (
    _as_batched,
    _normalize_point_weights,
    _validate_layout,
    _validate_points,
    restore_point_rank,
)
from ._warp_impl import nearest_surface_faces_warp


def _check_tensor_predicate(
    predicate: torch.Tensor,
    message: str,
) -> None:
    """Raise for a false scalar tensor while preserving tracing support."""

    if torch.compiler.is_compiling():
        torch._assert_async(predicate, message)
        return

    from torch._subclasses.fake_tensor import is_fake

    if is_fake(predicate):
        torch._assert_async(predicate, message)
    elif not bool(predicate):
        raise ValueError(message)


def _normalize_offset(
    offset: float | torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Normalize a scalar offset without severing tensor autograd."""

    if isinstance(offset, torch.Tensor):
        if offset.ndim != 0:
            raise ValueError(
                f"tensor-valued offset must be scalar, got shape {tuple(offset.shape)}"
            )
        if offset.device != points.device:
            raise ValueError(
                "offset and points must be on the same device, got "
                f"{offset.device} and {points.device}"
            )
        if offset.dtype != points.dtype:
            raise TypeError(
                "offset and points must have the same dtype, got "
                f"{offset.dtype} and {points.dtype}"
            )
        _check_tensor_predicate(
            torch.isfinite(offset),
            "offset must be finite",
        )
        return offset

    if not isinstance(offset, Real) or isinstance(offset, bool):
        raise TypeError(
            "offset must be a real scalar or scalar torch.Tensor, got "
            f"{type(offset).__name__}"
        )
    finfo = torch.finfo(points.dtype)
    if torch.compiler.is_compiling():
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        if statically_known_true(offset != offset) or statically_known_true(
            abs(offset) == math.inf
        ):
            torch._check_value(False, lambda: "offset must be finite")
        if statically_known_true(abs(offset) > finfo.max):
            torch._check_value(
                False,
                lambda: "offset must be finite in the points dtype",
            )
        torch._check(
            offset >= -finfo.max,
            lambda: "offset must be finite",
        )
        torch._check(
            offset <= finfo.max,
            lambda: "offset must be finite",
        )
        return torch.tensor(
            offset,
            dtype=points.dtype,
            device=points.device,
        )

    offset_value = float(offset)
    if not math.isfinite(offset_value):
        raise ValueError("offset must be finite")
    if abs(offset_value) > finfo.max:
        raise ValueError(f"offset must be finite in the points dtype {points.dtype}")
    return torch.tensor(
        offset_value,
        dtype=points.dtype,
        device=points.device,
    )


def _normalize_max_distance(
    max_distance: float | None,
    dtype: torch.dtype,
) -> float:
    """Validate a static nearest-surface search radius."""

    if max_distance is None:
        return float("inf")
    if not isinstance(max_distance, Real) or isinstance(max_distance, bool):
        raise TypeError(
            "max_distance must be a positive finite real scalar or None, got "
            f"{type(max_distance).__name__}"
        )
    finfo = torch.finfo(dtype)
    max_representable = finfo.max
    # IEEE round-to-nearest maps values at or below half the smallest
    # subnormal to zero. For float64 this threshold is itself below the
    # smallest positive Python float and therefore evaluates to zero.
    zero_rounding_threshold = 0.5 * finfo.tiny * finfo.eps
    dtype_positive_message = (
        f"max_distance must remain positive in the points dtype {dtype}"
    )
    if torch.compiler.is_compiling():
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        if statically_known_true(max_distance != max_distance) or (
            statically_known_true(abs(max_distance) == math.inf)
        ):
            torch._check_value(
                False,
                lambda: "max_distance must be positive and finite",
            )
        if statically_known_true(max_distance <= 0):
            torch._check_value(
                False,
                lambda: "max_distance must be positive and finite",
            )
        if statically_known_true(max_distance > max_representable):
            torch._check_value(
                False,
                lambda: "max_distance must be finite in the points dtype",
            )
        if statically_known_true(max_distance <= zero_rounding_threshold):
            torch._check_value(
                False,
                lambda: "max_distance must remain positive in the points dtype",
            )
        torch._check(
            max_distance > 0,
            lambda: "max_distance must be positive and finite",
        )
        torch._check(
            max_distance <= max_representable,
            lambda: "max_distance must be finite in the points dtype",
        )
        torch._check(
            max_distance > zero_rounding_threshold,
            lambda: "max_distance must remain positive in the points dtype",
        )
        return max_distance

    value = float(max_distance)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("max_distance must be positive and finite")
    if value > max_representable:
        raise ValueError(f"max_distance must be finite in the points dtype {dtype}")
    if torch.tensor(value, dtype=dtype).item() == 0.0:
        raise ValueError(dtype_positive_message)
    return value


def _normalize_shrinkwrap_inputs(
    points: torch.Tensor,
    target_points: torch.Tensor,
    target_faces: torch.Tensor,
    point_weights: torch.Tensor | None,
    offset: float | torch.Tensor,
    max_distance: float | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    float,
    bool,
]:
    """Validate public shrinkwrap inputs and normalize the source batch."""

    _validate_points(points, "points")
    _validate_points(target_points, "target_points")
    if points.shape[-1] != 3:
        raise ValueError(
            f"points must have three coordinates, got shape {tuple(points.shape)}"
        )
    if target_points.ndim != 2 or target_points.shape[-1] != 3:
        raise ValueError(
            f"target_points must have shape (M, 3), got {tuple(target_points.shape)}"
        )
    _validate_layout(target_points, points, "points and target_points")

    if not isinstance(target_faces, torch.Tensor):
        raise TypeError(
            f"target_faces must be a torch.Tensor, got {type(target_faces).__name__}"
        )
    if target_faces.ndim != 2 or target_faces.shape[-1] != 3:
        raise ValueError(
            f"target_faces must have shape (F, 3), got {tuple(target_faces.shape)}"
        )
    if target_faces.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "target_faces must have dtype torch.int32 or torch.int64, got "
            f"{target_faces.dtype}"
        )
    if target_faces.device != points.device:
        raise ValueError(
            "target_faces and points must be on the same device, got "
            f"{target_faces.device} and {points.device}"
        )
    if target_faces.shape[0] == 0:
        raise ValueError("target_faces must contain at least one triangle")

    valid_indices = (
        (target_faces >= 0) & (target_faces < target_points.shape[0])
    ).all()
    _check_tensor_predicate(
        valid_indices,
        "target_faces contain indices outside the target point range",
    )
    _check_tensor_predicate(
        torch.isfinite(points).all(),
        "points must contain only finite coordinates",
    )

    with torch.no_grad():
        triangles = target_points.detach()[target_faces.to(torch.long)]
        has_valid_triangle = _valid_triangles(triangles).any()
    _check_tensor_predicate(
        has_valid_triangle,
        "target must contain at least one finite, nondegenerate triangle",
    )

    points_b3, was_unbatched = _as_batched(points)
    weights_b2 = _normalize_point_weights(
        point_weights,
        points_b3,
        was_unbatched,
    )
    return (
        points_b3,
        target_points,
        target_faces,
        weights_b2,
        _normalize_offset(offset, points),
        _normalize_max_distance(max_distance, points.dtype),
        was_unbatched,
    )


def _make_benchmark_surface(
    device: torch.device,
    dtype: torch.dtype,
    resolution: int = 24,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a deterministic wavy triangle patch for benchmarks."""

    axis = torch.linspace(-1.0, 1.0, resolution, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    grid_z = 0.12 * torch.cos(1.7 * grid_x) * torch.cos(1.3 * grid_y)
    target_points = torch.stack((grid_x, grid_y, grid_z), dim=-1).reshape(-1, 3)

    rows = torch.arange(resolution - 1, device=device)
    cols = torch.arange(resolution - 1, device=device)
    cell_i, cell_j = torch.meshgrid(rows, cols, indexing="ij")
    lower_left = cell_i * resolution + cell_j
    lower_right = lower_left + 1
    upper_left = lower_left + resolution
    upper_right = upper_left + 1
    target_faces = torch.stack(
        (
            torch.stack((lower_left, lower_right, upper_right), dim=-1),
            torch.stack((lower_left, upper_right, upper_left), dim=-1),
        ),
        dim=-2,
    ).reshape(-1, 3)
    return target_points, target_faces


def _sample_benchmark_queries(
    target_points: torch.Tensor,
    target_faces: torch.Tensor,
    num_points: int,
    generator: torch.Generator,
    distance: float,
) -> torch.Tensor:
    """Sample queries above triangle interiors, away from feature ties."""

    face_indices = torch.randint(
        target_faces.shape[0],
        (num_points,),
        generator=generator,
        device=target_points.device,
    )
    triangles = target_points[target_faces[face_indices]]
    barycentric = torch.rand(
        (num_points, 3),
        generator=generator,
        device=target_points.device,
        dtype=target_points.dtype,
    )
    barycentric = 0.15 + 0.55 * (barycentric / barycentric.sum(dim=-1, keepdim=True))
    projected = (barycentric.unsqueeze(-1) * triangles).sum(dim=1)
    normals = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals = normals / torch.linalg.vector_norm(
        normals,
        dim=-1,
        keepdim=True,
    )
    return projected + distance * normals


class ShrinkwrapPoints(FunctionSpec):
    r"""Project points onto the nearest locations of a triangle surface.

    For every source point :math:`x_i`, the operation selects the closest
    location :math:`p_i` on the target triangle surface and applies

    .. math::

       x'_i = x_i + w_i\left(p_i + \delta n_i - x_i\right),

    where :math:`w_i` is an optional point weight, :math:`\delta` is ``offset``,
    and :math:`n_i` is the oriented unit normal of the selected target face.
    Positive offsets follow the target winding. Points with no target triangle
    within ``max_distance`` remain unchanged.

    Source points may have shape ``(N, 3)`` or ``(B, N, 3)``. A single target
    surface is shared by every source batch. Float32 and float64 are supported.

    Parameters
    ----------
    points : torch.Tensor
        Source coordinates with shape ``(N, 3)`` or ``(B, N, 3)``.
    target_points : torch.Tensor
        Target-surface vertices with shape ``(M, 3)`` and the same dtype and
        device as ``points``.
    target_faces : torch.Tensor
        Target triangle connectivity with shape ``(F, 3)``, dtype
        ``torch.int32`` or ``torch.int64``, and the same device as ``points``.
        Out-of-range indices raise an error. Triangles with nonfinite
        coordinates or degenerate geometry are ignored, but at least one
        finite nondegenerate triangle is required.
    offset : float or torch.Tensor, optional
        Signed scalar offset along the selected target face normal. A scalar
        tensor must match the source dtype and device and may require gradients.
        Default is ``0.0``.
    max_distance : float or None, optional
        Positive finite search radius measured to the un-offset target.
        A point exactly at the cutoff is left unchanged. ``None`` performs an
        unbounded nearest-surface search. Values that round to zero in the
        source dtype are rejected. Default is ``None``.
    point_weights : torch.Tensor or None, optional
        Optional bool or floating source-point weights with shape ``(N,)`` or
        ``(B, N)``. Zero leaves a point unchanged and one applies the full
        projection. Floating values are not clamped. Default is ``None``.
    implementation : {"torch", "warp"} or None, optional
        Explicit backend for nearest-face search. ``None`` selects Torch on CPU
        and Warp on CUDA when available.

    Returns
    -------
    torch.Tensor
        Shrinkwrapped points with the source shape, dtype, and device.

    Notes
    -----
    Nearest-face selection, closest-feature changes, and ``max_distance`` gating
    are discrete. With those choices fixed, both backends propagate first-order
    gradients through source points, selected target vertices, floating point
    weights, and tensor-valued ``offset``. Ties and transitions between target
    faces, edges, and vertices are nonsmooth.

    The Warp backend accelerates only the discrete nearest-face search. Both
    backends replay the selected point-to-triangle projection with Torch in the
    original dtype, which supplies identical piecewise geometry derivatives.
    Float64 targets use the Torch search because Warp searches in float32. Safe
    float32 coordinates are searched unchanged. Warp falls back to Torch for
    unsafe coordinate magnitudes or face geometry.
    Target topology is non-differentiable. Nonzero offsets require consistently
    oriented target faces. At shared features, the selected face determines the
    normal. The operation does not prevent source-cell inversion or
    self-intersection. Shrinkwrap is not supported inside CUDA Graph capture
    with either backend.
    """

    _COMPARE_ATOL = 2.0e-5
    _COMPARE_RTOL = 2.0e-5
    _COMPARE_BACKWARD_ATOL = 1.0e-4
    _COMPARE_BACKWARD_RTOL = 1.0e-4

    @FunctionSpec.register(name="warp", required_imports=("warp>=1.0.0",), rank=0)
    def warp_forward(
        points: Float[torch.Tensor, "*batch num_points 3"],
        target_points: Float[torch.Tensor, "num_target_points 3"],
        target_faces: Int[torch.Tensor, "num_target_faces 3"],
        *,
        offset: float | Float[torch.Tensor, ""] = 0.0,
        max_distance: float | None = None,
        point_weights: Bool[torch.Tensor, "*batch num_points"]
        | Float[torch.Tensor, "*batch num_points"]
        | None = None,
    ) -> Float[torch.Tensor, "*batch num_points 3"]:
        """Apply shrinkwrap with Warp nearest-face search."""

        normalized = _normalize_shrinkwrap_inputs(
            points,
            target_points,
            target_faces,
            point_weights,
            offset,
            max_distance,
        )
        (
            points_b3,
            target_points_t,
            target_faces_t,
            weights_b2,
            offset_t,
            max_distance_value,
            was_unbatched,
        ) = normalized
        query_points = points_b3.reshape(-1, 3)
        nearest_faces = nearest_surface_faces_warp(
            target_points_t.detach(),
            target_faces_t,
            query_points.detach(),
            max_distance_value,
        ).reshape(points_b3.shape[:-1])
        output = replay_shrinkwrap_projection(
            points_b3,
            target_points_t,
            target_faces_t,
            nearest_faces,
            weights_b2,
            offset_t,
            max_distance_value,
        )
        return restore_point_rank(output, was_unbatched)

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: Float[torch.Tensor, "*batch num_points 3"],
        target_points: Float[torch.Tensor, "num_target_points 3"],
        target_faces: Int[torch.Tensor, "num_target_faces 3"],
        *,
        offset: float | Float[torch.Tensor, ""] = 0.0,
        max_distance: float | None = None,
        point_weights: Bool[torch.Tensor, "*batch num_points"]
        | Float[torch.Tensor, "*batch num_points"]
        | None = None,
    ) -> Float[torch.Tensor, "*batch num_points 3"]:
        """Apply shrinkwrap with the Torch nearest-face reference."""

        normalized = _normalize_shrinkwrap_inputs(
            points,
            target_points,
            target_faces,
            point_weights,
            offset,
            max_distance,
        )
        (
            points_b3,
            target_points_t,
            target_faces_t,
            weights_b2,
            offset_t,
            max_distance_value,
            was_unbatched,
        ) = normalized
        query_points = points_b3.reshape(-1, 3)
        nearest_faces = nearest_surface_faces_torch(
            query_points.detach(),
            target_points_t.detach(),
            target_faces_t,
            max_distance_value,
        ).reshape(points_b3.shape[:-1])
        output = replay_shrinkwrap_projection(
            points_b3,
            target_points_t,
            target_faces_t,
            nearest_faces,
            weights_b2,
            offset_t,
            max_distance_value,
        )
        return restore_point_rank(output, was_unbatched)

    @classmethod
    def dispatch(
        cls,
        points: Float[torch.Tensor, "*batch num_points 3"],
        target_points: Float[torch.Tensor, "num_target_points 3"],
        target_faces: Int[torch.Tensor, "num_target_faces 3"],
        *,
        offset: float | Float[torch.Tensor, ""] = 0.0,
        max_distance: float | None = None,
        point_weights: Bool[torch.Tensor, "*batch num_points"]
        | Float[torch.Tensor, "*batch num_points"]
        | None = None,
        implementation: Literal["torch", "warp"] | None = None,
    ) -> Float[torch.Tensor, "*batch num_points 3"]:
        """Select a search backend and apply shrinkwrap."""

        if implementation is None:
            implementations = cls._get_impls()
            warp_implementation = implementations.get("warp")
            if isinstance(points, torch.Tensor) and points.is_cuda:
                if warp_implementation is not None and warp_implementation.available:
                    implementation = "warp"
                else:
                    cls._warn_fallback(
                        warp_implementation,
                        implementations["torch"],
                    )
                    implementation = "torch"
            else:
                implementation = "torch"
        return super().dispatch(
            points,
            target_points,
            target_faces,
            offset=offset,
            max_distance=max_distance,
            point_weights=point_weights,
            implementation=implementation,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative nearest-surface benchmark cases."""

        device = torch.device(device)
        target_points, target_faces = _make_benchmark_surface(
            device,
            torch.float32,
        )
        for num_points in (1024, 8192):
            generator = torch.Generator(device=device).manual_seed(2701 + num_points)
            points = _sample_benchmark_queries(
                target_points,
                target_faces,
                num_points,
                generator,
                distance=1.0e-3,
            )
            yield (
                f"n{num_points}-target-f{target_faces.shape[0]}",
                (points, target_points, target_faces),
                {},
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield a representative all-gradient benchmark case."""

        device = torch.device(device)
        target_points = torch.tensor(
            [
                [-1.2, -1.0, 0.0],
                [1.4, -0.8, 0.15],
                [-0.1, 1.5, -0.1],
            ],
            device=device,
            dtype=torch.float32,
        )
        target_faces = torch.tensor(
            [[0, 1, 2]],
            device=device,
            dtype=torch.int64,
        )
        generator = torch.Generator(device=device).manual_seed(2801)
        points = _sample_benchmark_queries(
            target_points,
            target_faces,
            2048,
            generator,
            distance=0.25,
        )
        weights = torch.rand((2048,), generator=generator, device=device)
        offset = torch.tensor(0.01, device=device)
        yield (
            "n2048-all-gradients",
            (
                points.requires_grad_(True),
                target_points.requires_grad_(True),
                target_faces,
            ),
            {
                "point_weights": weights.requires_grad_(True),
                "offset": offset.requires_grad_(True),
            },
        )

    @classmethod
    def compare_forward(
        cls,
        output: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        """Compare Warp and Torch projected coordinates."""

        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_ATOL,
            rtol=cls._COMPARE_RTOL,
        )

    @classmethod
    def compare_backward(
        cls,
        output: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        """Compare Warp and Torch first-order gradients."""

        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_BACKWARD_ATOL,
            rtol=cls._COMPARE_BACKWARD_RTOL,
        )


shrinkwrap_points = ShrinkwrapPoints.make_function("shrinkwrap_points")

__all__ = ["ShrinkwrapPoints", "shrinkwrap_points"]
