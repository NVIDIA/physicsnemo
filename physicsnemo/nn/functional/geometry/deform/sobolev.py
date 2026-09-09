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

"""Backend-dispatched Sobolev point deformation."""

from __future__ import annotations

import math
from numbers import Real
from typing import Literal

import torch
from jaxtyping import Bool, Float, Int

from physicsnemo.core.function_spec import FunctionSpec

from ._sobolev_torch_impl import sobolev_deform_points_torch
from ._warp_impl import sobolev_deform_points_warp


def _validate_scalar_options(
    length_scale: float,
    max_iterations: int,
    tolerance: float | None,
    dtype: torch.dtype,
) -> tuple[float, int, float]:
    """Validate scalar Sobolev deformation options."""

    if not isinstance(length_scale, Real) or isinstance(length_scale, bool):
        raise TypeError(
            "length_scale must be a nonnegative finite Python real scalar, got "
            f"{type(length_scale).__name__}"
        )
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise TypeError(
            f"max_iterations must be an int, got {type(max_iterations).__name__}"
        )
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if tolerance is not None and (
        not isinstance(tolerance, Real) or isinstance(tolerance, bool)
    ):
        raise TypeError(
            "tolerance must be a positive finite Python real scalar or None, got "
            f"{type(tolerance).__name__}"
        )

    finfo = torch.finfo(dtype)
    if torch.compiler.is_compiling():
        # Dynamo can generalize call-time Python scalars into SymFloat values.
        # Keep the checks in the graph without forcing scalar specialization.
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        if statically_known_true(length_scale != length_scale) or (
            statically_known_true(abs(length_scale) == math.inf)
        ):
            torch._check_value(False, lambda: "length_scale must be finite")
        if statically_known_true(length_scale < 0):
            torch._check_value(
                False,
                lambda: "length_scale must be nonnegative",
            )
        if statically_known_true(length_scale > math.sqrt(finfo.max)):
            torch._check_value(
                False,
                lambda: "length_scale squared must be finite in the points dtype",
            )
        torch._check(
            length_scale >= 0,
            lambda: "length_scale must be nonnegative",
        )
        torch._check(
            length_scale <= math.sqrt(finfo.max),
            lambda: "length_scale squared must be finite in the points dtype",
        )
        length_scale_value = length_scale
    else:
        length_scale_value = float(length_scale)
        if not math.isfinite(length_scale_value) or length_scale_value < 0:
            raise ValueError("length_scale must be nonnegative and finite")
        if length_scale_value > math.sqrt(finfo.max):
            raise ValueError(
                f"length_scale squared must be finite in the points dtype {dtype}"
            )

    if tolerance is None:
        tolerance_value = 1.0e-6 if dtype == torch.float32 else 1.0e-10
    elif torch.compiler.is_compiling():
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        if statically_known_true(tolerance != tolerance) or statically_known_true(
            abs(tolerance) == math.inf
        ):
            torch._check_value(False, lambda: "tolerance must be finite")
        if statically_known_true(tolerance <= 0):
            torch._check_value(
                False,
                lambda: "tolerance must be positive",
            )
        torch._check(tolerance > 0, lambda: "tolerance must be positive")
        torch._check(tolerance <= finfo.max, lambda: "tolerance must be finite")
        tolerance_value = tolerance
    else:
        tolerance_value = float(tolerance)
        if not math.isfinite(tolerance_value) or tolerance_value <= 0:
            raise ValueError("tolerance must be positive and finite")
    return length_scale_value, max_iterations, tolerance_value


def _normalize_sobolev_inputs(
    points: torch.Tensor,
    cells: torch.Tensor,
    displacement: torch.Tensor,
    fixed_points: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
    """Validate and normalize Sobolev deformation tensors."""

    for name, value in (
        ("points", points),
        ("cells", cells),
        ("displacement", displacement),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor, got {type(value).__name__}"
            )

    if points.ndim not in (2, 3):
        raise ValueError(
            f"points must have shape (N, D) or (B, N, D), got {tuple(points.shape)}"
        )
    if tuple(displacement.shape) != tuple(points.shape):
        raise ValueError(
            "points and displacement must have identical shapes, got "
            f"{tuple(points.shape)} and {tuple(displacement.shape)}"
        )
    if points.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            f"points must have dtype torch.float32 or torch.float64, got {points.dtype}"
        )
    if displacement.dtype != points.dtype:
        raise TypeError(
            "points and displacement must have the same dtype, got "
            f"{points.dtype} and {displacement.dtype}"
        )
    if displacement.device != points.device:
        raise ValueError(
            "points and displacement must be on the same device, got "
            f"{points.device} and {displacement.device}"
        )

    if cells.ndim != 2:
        raise ValueError(f"cells must have shape (C, V), got {tuple(cells.shape)}")
    if cells.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"cells must have dtype torch.int32 or torch.int64, got {cells.dtype}"
        )
    if cells.device != points.device:
        raise ValueError(
            f"points and cells must be on the same device, got "
            f"{points.device} and {cells.device}"
        )

    num_points = points.shape[-2]
    num_spatial_dims = points.shape[-1]
    num_cell_points = cells.shape[-1]
    if torch.compiler.is_compiling():
        torch._check(
            num_cell_points >= 2,
            lambda: "cells must contain at least two points",
        )
        torch._check(
            num_cell_points <= num_spatial_dims + 1,
            lambda: "cell manifold dimension must not exceed the spatial dimension",
        )
    else:
        if num_cell_points < 2:
            raise ValueError("cells must contain at least two points")
        if num_cell_points > num_spatial_dims + 1:
            raise ValueError(
                "cell manifold dimension must not exceed the spatial dimension"
            )

    valid_indices = ((cells >= 0) & (cells < num_points)).all()
    if torch.compiler.is_compiling():
        torch._assert_async(
            valid_indices,
            "cells contain indices outside the point range",
        )
    else:
        from torch._subclasses.fake_tensor import is_fake

        if is_fake(cells):
            torch._assert_async(
                valid_indices,
                "cells contain indices outside the point range",
            )
        elif not bool(valid_indices):
            raise ValueError(f"cells contain indices outside [0, {num_points})")

    expected_fixed_shape = points.shape[:-1]
    if fixed_points is not None:
        if not isinstance(fixed_points, torch.Tensor):
            raise TypeError(
                "fixed_points must be a torch.Tensor or None, got "
                f"{type(fixed_points).__name__}"
            )
        if tuple(fixed_points.shape) != tuple(expected_fixed_shape):
            raise ValueError(
                "fixed_points must match points without its coordinate axis, got "
                f"{tuple(fixed_points.shape)} and expected "
                f"{tuple(expected_fixed_shape)}"
            )
        if fixed_points.dtype != torch.bool:
            raise TypeError(
                f"fixed_points must have dtype torch.bool, got {fixed_points.dtype}"
            )
        if fixed_points.device != points.device:
            raise ValueError(
                "points and fixed_points must be on the same device, got "
                f"{points.device} and {fixed_points.device}"
            )

    was_unbatched = points.ndim == 2
    if was_unbatched:
        points = points.unsqueeze(0)
        displacement = displacement.unsqueeze(0)
        if fixed_points is not None:
            fixed_points = fixed_points.unsqueeze(0)
    return points, cells.to(dtype=torch.long), displacement, fixed_points, was_unbatched


class SobolevDeformPoints(FunctionSpec):
    r"""Deform simplicial-mesh points with a smooth Sobolev displacement.

    The operation filters a prescribed per-vertex displacement :math:`d` by
    solving the uniform-mass discrete Helmholtz system

    .. math::

       (M + \ell^2 K)u = M d,
       \qquad x' = x + u.

    Here :math:`M=\bar m I` is a uniform vertex mass matrix. Its scalar
    :math:`\bar m` is the mean positive lumped P1 vertex mass. :math:`K` is the
    P1 stiffness matrix, and :math:`\ell` is ``length_scale`` in the same
    physical units as ``points``. This uniform mass makes the forward filter
    self-adjoint in standard Euclidean vertex coordinates. The reverse pass
    therefore applies the same smoothing operator to the displacement
    adjoint. The solve uses a matrix-free Jacobi-preconditioned conjugate
    gradient method. Each ambient component is filtered independently.

    ``points`` and ``displacement`` may be unbatched ``(N, D)`` or aligned
    batched ``(B, N, D)`` tensors. ``cells`` has shape ``(C, V)`` and defines
    one topology shared by every batch entry. It must contain simplices with
    ``V - 1 <= D``. All floating tensors must use float32 or float64.

    Parameters
    ----------
    points : torch.Tensor
        Vertex coordinates with shape ``(N, D)`` or ``(B, N, D)``.
    cells : torch.Tensor
        Shared simplex connectivity with shape ``(C, V)`` and int32 or int64
        dtype.
    displacement : torch.Tensor
        Raw per-vertex displacement with the same shape, dtype, and device as
        ``points``.
    length_scale : float
        Nonnegative physical smoothing length. Zero applies the raw
        displacement exactly at unfixed points.
    fixed_points : torch.Tensor or None, optional
        Optional bool mask with shape ``(N,)`` or ``(B, N)``. True entries use a
        zero Dirichlet displacement. Other mesh boundaries use the natural
        homogeneous Neumann condition. Default is ``None``.
    max_iterations : int, optional
        Maximum PCG iterations. Default is ``128``.
    tolerance : float or None, optional
        Positive relative residual tolerance. ``None`` selects ``1e-6`` for
        float32 and ``1e-10`` for float64. Default is ``None``.
    implementation : {"torch", "warp"} or None, optional
        Explicit backend. ``None`` selects Torch on CPU. On CUDA, it selects
        Warp for segments, triangles, and tetrahedra when available. It
        otherwise selects Torch, with a one-time :class:`RuntimeWarning` when
        Warp is unavailable. The Warp backend requires CUDA tensors.

    Returns
    -------
    torch.Tensor
        Deformed points with the same shape, dtype, and device as ``points``.

    Raises
    ------
    TypeError
        If argument types or tensor dtypes are unsupported.
    ValueError
        If shapes, devices, indices, scalar options, or simplex geometry are
        invalid.
    KeyError
        If ``implementation`` does not name a registered backend.
    ImportError
        If an explicitly requested backend is unavailable.
    RuntimeError
        If CUDA Graph capture is active or the forward or adjoint PCG solve
        does not reach ``tolerance`` within ``max_iterations``.

    Notes
    -----
    Constant displacements are retained to solver precision when no points are
    fixed.
    Isolated points receive their raw displacement. The uniform mass scale is
    computed over all nonisolated vertices in the supplied topology.

    Both backends provide first-order gradients with respect to ``points`` and
    ``displacement``. Their implicit reverse-mode derivatives solve the
    adjoint of the forward Helmholtz system. This is not an identity-valued
    surrogate gradient. The Warp backend evaluates the geometry
    vector-Jacobian product analytically because Warp's supplied linear
    solvers do not generate automatic backward kernels. Higher-order gradients
    are not supported for a positive length scale.

    The Warp backend supports segments, triangles, and tetrahedra. The Torch
    backend also supports higher-dimensional simplices. Default dispatch keeps
    higher-dimensional CUDA simplices on Torch.
    Warp CUDA assembly and geometry pullback use atomic accumulation, so
    results and point gradients may vary at roundoff between runs.

    Forward and backward each run at most ``max_iterations`` matrix-free PCG
    steps. A nonconverged solve raises an error instead of returning an
    inconsistent implicit gradient. At positive length scales, cells must be
    finite and nondegenerate. The operation does not check for inverted or
    self-intersecting output cells.
    """

    @FunctionSpec.register(name="warp", required_imports=("warp>=1.14.0",), rank=0)
    def warp_forward(
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells num_cell_points"],
        displacement: Float[torch.Tensor, "*batch num_points num_dims"],
        *,
        length_scale: float,
        fixed_points: Bool[torch.Tensor, "*batch num_points"] | None = None,
        max_iterations: int = 128,
        tolerance: float | None = None,
    ) -> Float[torch.Tensor, "*batch num_points num_dims"]:
        """Apply Sobolev deformation with the Warp CUDA backend."""

        if (
            isinstance(points, torch.Tensor)
            and points.is_cuda
            and not torch.compiler.is_compiling()
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "Sobolev deformation is not supported during CUDA Graph capture "
                "because P1 operator assembly and solver diagnostics are not "
                "capture-safe"
            )

        points_b3, cells, displacement_b3, fixed_b2, was_unbatched = (
            _normalize_sobolev_inputs(
                points,
                cells,
                displacement,
                fixed_points,
            )
        )
        if points_b3.device.type != "cuda":
            raise ValueError("the Warp Sobolev backend requires CUDA tensors")
        length_scale, max_iterations, tolerance = _validate_scalar_options(
            length_scale,
            max_iterations,
            tolerance,
            points_b3.dtype,
        )
        output = sobolev_deform_points_warp(
            points_b3,
            cells,
            displacement_b3,
            fixed_b2,
            length_scale=length_scale,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        return output.squeeze(0) if was_unbatched else output

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells num_cell_points"],
        displacement: Float[torch.Tensor, "*batch num_points num_dims"],
        *,
        length_scale: float,
        fixed_points: Bool[torch.Tensor, "*batch num_points"] | None = None,
        max_iterations: int = 128,
        tolerance: float | None = None,
    ) -> Float[torch.Tensor, "*batch num_points num_dims"]:
        """Apply Sobolev deformation with the pure-Torch backend."""

        if (
            isinstance(points, torch.Tensor)
            and points.is_cuda
            and not torch.compiler.is_compiling()
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "Sobolev deformation is not supported during CUDA Graph capture "
                "because P1 operator assembly and solver diagnostics are not "
                "capture-safe"
            )

        points_b3, cells, displacement_b3, fixed_b2, was_unbatched = (
            _normalize_sobolev_inputs(
                points,
                cells,
                displacement,
                fixed_points,
            )
        )
        length_scale, max_iterations, tolerance = _validate_scalar_options(
            length_scale,
            max_iterations,
            tolerance,
            points_b3.dtype,
        )
        output = sobolev_deform_points_torch(
            points_b3,
            cells,
            displacement_b3,
            fixed_b2,
            length_scale=length_scale,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        return output.squeeze(0) if was_unbatched else output

    @classmethod
    def dispatch(
        cls,
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells num_cell_points"],
        displacement: Float[torch.Tensor, "*batch num_points num_dims"],
        *,
        length_scale: float,
        fixed_points: Bool[torch.Tensor, "*batch num_points"] | None = None,
        max_iterations: int = 128,
        tolerance: float | None = None,
        implementation: Literal["torch", "warp"] | None = None,
    ) -> Float[torch.Tensor, "*batch num_points num_dims"]:
        """Select Warp for CUDA inputs and Torch for CPU inputs by default."""

        if implementation is None:
            impls = cls._get_impls()
            warp_impl = impls.get("warp")
            if isinstance(points, torch.Tensor) and points.is_cuda:
                warp_cells = (
                    isinstance(cells, torch.Tensor)
                    and cells.ndim == 2
                    and cells.shape[-1] in (2, 3, 4)
                )
                if not warp_cells:
                    implementation = "torch"
                elif warp_impl is not None and warp_impl.available:
                    implementation = "warp"
                else:
                    cls._warn_fallback(warp_impl, impls["torch"])
                    implementation = "torch"
            else:
                implementation = "torch"
        return super().dispatch(
            points,
            cells,
            displacement,
            length_scale=length_scale,
            fixed_points=fixed_points,
            max_iterations=max_iterations,
            tolerance=tolerance,
            implementation=implementation,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative Sobolev forward benchmark cases."""

        device = torch.device(device)
        for label, batch_size, num_points, dtype in (
            ("small-n1024-d2", 1, 1024, torch.float32),
            ("medium-b4-n4096-d2", 4, 4096, torch.float32),
            ("float64-n2048-d2", 1, 2048, torch.float64),
        ):
            x = torch.linspace(0, 1, num_points, device=device, dtype=dtype)
            points = torch.stack((x, torch.zeros_like(x)), dim=-1)
            displacement = torch.stack(
                (torch.zeros_like(x), 0.05 * torch.sin(16 * torch.pi * x)),
                dim=-1,
            )
            if batch_size > 1:
                points = points.unsqueeze(0).expand(batch_size, -1, -1).clone()
                displacement = (
                    displacement.unsqueeze(0).expand(batch_size, -1, -1).clone()
                )
            cells = torch.stack(
                (
                    torch.arange(num_points - 1, device=device),
                    torch.arange(1, num_points, device=device),
                ),
                dim=-1,
            )
            yield (
                label,
                (points, cells, displacement),
                {
                    "length_scale": 2.0 / (num_points - 1),
                    "max_iterations": 64,
                },
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative differentiable Sobolev benchmark cases."""

        device = torch.device(device)
        num_points = 2048
        x = torch.linspace(0, 1, num_points, device=device)
        points = torch.stack((x, torch.zeros_like(x)), dim=-1).requires_grad_(True)
        displacement = torch.stack(
            (torch.zeros_like(x), 0.05 * torch.sin(16 * torch.pi * x)),
            dim=-1,
        ).requires_grad_(True)
        cells = torch.stack(
            (
                torch.arange(num_points - 1, device=device),
                torch.arange(1, num_points, device=device),
            ),
            dim=-1,
        )
        yield (
            "medium-n2048-d2",
            (points, cells, displacement),
            {
                "length_scale": 2.0 / (num_points - 1),
                "max_iterations": 64,
            },
        )

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare Sobolev benchmark outputs."""

        torch.testing.assert_close(output, reference, atol=1.0e-5, rtol=1.0e-5)

    @classmethod
    def compare_backward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare Sobolev benchmark gradients."""

        torch.testing.assert_close(output, reference, atol=1.0e-5, rtol=1.0e-5)


sobolev_deform_points = SobolevDeformPoints.make_function("sobolev_deform_points")


__all__ = ["SobolevDeformPoints", "sobolev_deform_points"]
