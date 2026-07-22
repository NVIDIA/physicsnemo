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

"""Differentiable local/global template fitting."""

from __future__ import annotations

import math
from numbers import Real
from typing import Literal

import torch

from physicsnemo.core.function_spec import FunctionSpec

from ._fit_torch_impl import fit_template_points_torch


def _validate_weight(value: float, name: str, *, strictly_positive: bool) -> float:
    """Return a finite Python weight with the requested sign constraint."""

    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a finite Python real scalar")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and normalized <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    if not strictly_positive and normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _validate_integer(value: int, name: str, *, allow_zero: bool) -> int:
    """Validate an iteration count without accepting booleans as integers."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    lower_bound = 0 if allow_zero else 1
    if value < lower_bound:
        qualifier = "nonnegative" if allow_zero else "strictly positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _validate_triangles(
    triangles: torch.Tensor,
    points: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Validate triangle connectivity and normalize it to int64."""

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"{name} must have shape (F, 3), got {tuple(triangles.shape)}")
    if triangles.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must have dtype torch.int32 or torch.int64")
    if triangles.device != points.device:
        raise ValueError(
            f"{name} and its point tensor must be on the same device, got "
            f"{triangles.device} and {points.device}"
        )
    if triangles.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one triangle")

    normalized = triangles.to(torch.int64)
    torch._assert_async(
        torch.all((normalized >= 0) & (normalized < points.shape[0])),
        f"{name} contains a point index outside [0, {points.shape[0]})",
    )
    torch._assert_async(
        torch.all(
            (normalized[:, 0] != normalized[:, 1])
            & (normalized[:, 1] != normalized[:, 2])
            & (normalized[:, 2] != normalized[:, 0])
        ),
        f"{name} must not contain repeated indices within a triangle",
    )
    # The asynchronous bounds assertion may surface after later device work has
    # already been enqueued. Keep invalid values away from gather/scatter ops in
    # the meantime; the returned tensor is unchanged for every valid input.
    return normalized.clamp(0, points.shape[0] - 1)


def _normalize_fit_inputs(
    template_points: torch.Tensor,
    template_triangles: torch.Tensor,
    target_points: torch.Tensor,
    target_triangles: torch.Tensor,
    fit_weight: float,
    arap_weight: float,
    steps: int,
    cg_tolerance: float,
    cg_max_iterations: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
    int,
    float,
    int,
]:
    """Validate the first unbatched 3D triangle-mesh fitting contract."""

    for points, name in (
        (template_points, "template_points"),
        (target_points, "target_points"),
    ):
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"{name} must have shape (N, 3), got {tuple(points.shape)}"
            )
        if points.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"{name} must have dtype torch.float32 or torch.float64, got "
                f"{points.dtype}"
            )
        if points.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one point")
        torch._assert_async(
            torch.isfinite(points).all(),
            f"{name} must contain only finite coordinates",
        )

    if target_points.dtype != template_points.dtype:
        raise TypeError(
            "template_points and target_points must have the same dtype, got "
            f"{template_points.dtype} and {target_points.dtype}"
        )
    if target_points.device != template_points.device:
        raise ValueError(
            "template_points and target_points must be on the same device, got "
            f"{template_points.device} and {target_points.device}"
        )

    template_triangles = _validate_triangles(
        template_triangles, template_points, "template_triangles"
    )
    target_triangles = _validate_triangles(
        target_triangles, target_points, "target_triangles"
    )
    fit_weight = _validate_weight(fit_weight, "fit_weight", strictly_positive=True)
    arap_weight = _validate_weight(arap_weight, "arap_weight", strictly_positive=False)
    steps = _validate_integer(steps, "steps", allow_zero=True)
    cg_tolerance = _validate_weight(
        cg_tolerance, "cg_tolerance", strictly_positive=True
    )
    cg_max_iterations = _validate_integer(
        cg_max_iterations, "cg_max_iterations", allow_zero=False
    )
    return (
        template_points,
        template_triangles,
        target_points,
        target_triangles,
        fit_weight,
        arap_weight,
        steps,
        cg_tolerance,
        cg_max_iterations,
    )


def _benchmark_mesh(
    side: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct a translated planar grid pair for benchmark inputs."""

    coordinate = torch.linspace(-1.0, 1.0, side, device=device)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    zz = torch.zeros_like(xx)
    template = torch.stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)), dim=-1)
    cells = []
    for row in range(side - 1):
        for column in range(side - 1):
            lower_left = row * side + column
            lower_right = lower_left + 1
            upper_left = lower_left + side
            upper_right = upper_left + 1
            cells.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    triangles = torch.tensor(cells, dtype=torch.int64, device=device)
    target = template + template.new_tensor((0.03, -0.02, 0.05))
    return template, triangles, target, triangles.clone()


class FitTemplatePoints(FunctionSpec):
    r"""Fit a prealigned triangle template to a target surface.

    The template's points are updated by fixed-count local/global iterations.
    Each local step finds closest points on the target surface and one proper
    rotation per template point. Each global step jointly solves

    .. math::

       (w_{fit} I + 2 w_{arap} L) X
       = w_{fit} C + 2 w_{arap} B(R),

    where :math:`L` is the uniform template-edge Laplacian, :math:`C` contains
    closest target-surface points, and :math:`B(R)` is the ARAP rotation term.
    Connectivity is preserved because only template point coordinates change.

    Parameters
    ----------
    template_points : torch.Tensor
        Prealigned template coordinates with shape ``(N, 3)`` and dtype
        float32 or float64.
    template_triangles : torch.Tensor
        Template connectivity with shape ``(F, 3)`` and dtype int32 or int64.
    target_points : torch.Tensor
        Target coordinates with shape ``(M, 3)`` and the same dtype and device
        as ``template_points``.
    target_triangles : torch.Tensor
        Target connectivity with shape ``(G, 3)`` and dtype int32 or int64.
    fit_weight : float, optional
        Positive closest-surface fitting weight. Default is ``1.0``.
    arap_weight : float, optional
        Nonnegative as-rigid-as-possible regularization weight. Zero performs
        unconstrained closest-surface projection. Default is ``0.1``.
    steps : int, optional
        Number of local/global iterations. Zero returns a clone of the template
        points. Default is ``10``.
    cg_tolerance : float, optional
        Positive scaled residual tolerance for both forward and adjoint CG
        solves. Default is ``1e-6``.
    cg_max_iterations : int, optional
        Positive maximum CG iteration count. Failure to reach the tolerance
        raises ``RuntimeError``. Default is ``256``.
    implementation : {"warp", "torch"} or None, optional
        Correspondence-search backend. Warp accelerates nearest-face selection;
        local rotations, ARAP assembly, and the implicit CG solve remain in
        Torch. ``None`` selects Warp for CUDA float32 coordinates when available
        and Torch for CPU or float64 coordinates. Explicit Warp supports float32
        only and never downcasts float64 inputs.

    Returns
    -------
    torch.Tensor
        Fitted template coordinates with shape ``(N, 3)``.

    Notes
    -----
    The functional provides first-order gradients through the exact fixed-step
    algorithm up to the requested CG tolerance. Each linear solve uses an
    implicit adjoint solve, so CG iteration histories are not retained.

    Nearest target face, closest triangle feature, mesh topology, and proper
    rotation orientation branches are discrete. Gradients with respect to
    template and target coordinates are defined for the selected branches and
    are valid almost everywhere. Higher-order differentiation is not supported.
    Inputs are assumed to be prealigned; rigid registration is a separate step.
    """

    _FORWARD_BENCHMARK_CASES = (
        ("small-grid8-steps3", 8, 3),
        ("medium-grid16-steps5", 16, 5),
    )

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        template_points: torch.Tensor,
        template_triangles: torch.Tensor,
        target_points: torch.Tensor,
        target_triangles: torch.Tensor,
        *,
        fit_weight: float = 1.0,
        arap_weight: float = 0.1,
        steps: int = 10,
        cg_tolerance: float = 1.0e-6,
        cg_max_iterations: int = 256,
    ) -> torch.Tensor:
        """Use Warp face search inside the otherwise Torch fitting solver."""

        normalized = _normalize_fit_inputs(
            template_points,
            template_triangles,
            target_points,
            target_triangles,
            fit_weight,
            arap_weight,
            steps,
            cg_tolerance,
            cg_max_iterations,
        )
        if template_points.dtype != torch.float32:
            raise TypeError(
                "the Warp template-fitting backend supports only torch.float32 "
                "coordinates"
            )
        return fit_template_points_torch(*normalized, point_implementation="warp")

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        template_points: torch.Tensor,
        template_triangles: torch.Tensor,
        target_points: torch.Tensor,
        target_triangles: torch.Tensor,
        *,
        fit_weight: float = 1.0,
        arap_weight: float = 0.1,
        steps: int = 10,
        cg_tolerance: float = 1.0e-6,
        cg_max_iterations: int = 256,
    ) -> torch.Tensor:
        """Fit template points with the pure-Torch backend."""

        normalized = _normalize_fit_inputs(
            template_points,
            template_triangles,
            target_points,
            target_triangles,
            fit_weight,
            arap_weight,
            steps,
            cg_tolerance,
            cg_max_iterations,
        )
        return fit_template_points_torch(*normalized, point_implementation="torch")

    @classmethod
    def dispatch(
        cls,
        template_points: torch.Tensor,
        template_triangles: torch.Tensor,
        target_points: torch.Tensor,
        target_triangles: torch.Tensor,
        *,
        fit_weight: float = 1.0,
        arap_weight: float = 0.1,
        steps: int = 10,
        cg_tolerance: float = 1.0e-6,
        cg_max_iterations: int = 256,
        implementation: Literal["torch", "warp"] | None = None,
    ) -> torch.Tensor:
        """Select Warp automatically only for CUDA float32 coordinates."""

        if implementation is None:
            implementations = cls._get_impls()
            warp_implementation = implementations.get("warp")
            use_warp = (
                isinstance(template_points, torch.Tensor)
                and isinstance(target_points, torch.Tensor)
                and template_points.is_cuda
                and target_points.is_cuda
                and template_points.dtype == torch.float32
                and target_points.dtype == torch.float32
            )
            if use_warp and warp_implementation is not None:
                if warp_implementation.available:
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
            template_points,
            template_triangles,
            target_points,
            target_triangles,
            fit_weight=fit_weight,
            arap_weight=arap_weight,
            steps=steps,
            cg_tolerance=cg_tolerance,
            cg_max_iterations=cg_max_iterations,
            implementation=implementation,
        )

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare fitted coordinates across correspondence backends."""

        torch.testing.assert_close(output, reference, atol=3.0e-5, rtol=3.0e-5)

    @classmethod
    def compare_backward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare template or target gradients across backends."""

        torch.testing.assert_close(output, reference, atol=5.0e-5, rtol=5.0e-5)

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative local/global fitting benchmark cases."""

        device = torch.device(device)
        for label, side, steps in cls._FORWARD_BENCHMARK_CASES:
            template, template_cells, target, target_cells = _benchmark_mesh(
                side, device
            )
            yield (
                label,
                (template, template_cells, target, target_cells),
                {"steps": steps},
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield a differentiable template-and-target benchmark case."""

        device = torch.device(device)
        template, template_cells, target, target_cells = _benchmark_mesh(8, device)
        yield (
            "small-grid8-steps3-all-gradients",
            (
                template.requires_grad_(True),
                template_cells,
                target.requires_grad_(True),
                target_cells,
            ),
            {"steps": 3},
        )


fit_template_points = FitTemplatePoints.make_function("fit_template_points")


__all__ = ["FitTemplatePoints", "fit_template_points"]
