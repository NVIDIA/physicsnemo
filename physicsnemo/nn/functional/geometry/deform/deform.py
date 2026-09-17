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

"""Backend-dispatched point displacement and compact Shepard morphing."""

from __future__ import annotations

from typing import Literal

import torch

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import displace_points_torch, morph_points_torch
from ._utils import (
    normalize_displace_inputs,
    normalize_morph_inputs,
    restore_point_rank,
)
from ._warp_impl import morph_points_warp


def _validate_kernel(kernel: str) -> None:
    """Validate the public morph-kernel selector."""

    if kernel != "wendland_c2":
        raise ValueError(f"kernel must be 'wendland_c2', got {kernel!r}")


class DisplacePoints(FunctionSpec):
    r"""Apply an aligned dense displacement field to points.

    The operation is

    .. math::

       x'_i = x_i + w_i\,d_i,

    where ``point_weights`` contains optional per-point values :math:`w_i` and
    ``displacement`` contains vectors :math:`d_i`. Signed or greater-than-one
    point weights are permitted.

    Inputs may be unbatched ``(N, D)`` or batched ``(B, N, D)``. Batched
    inputs are aligned rather than broadcast: ``points`` and ``displacement``
    must have identical shapes. Float32 and float64 are supported.

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates with shape ``(N, D)`` or ``(B, N, D)``.
    displacement : torch.Tensor
        Dense displacement vectors with exactly the same shape, dtype, and
        device as ``points``.
    point_weights : torch.Tensor or None, optional
        Optional bool or floating per-point weights. Accepted shapes are ``(N,)``
        for unbatched inputs and ``(B, N)`` for batched inputs. Floating point
        weights must match the point dtype and device; bool values act as hard
        masks. Values are used as supplied without clamping. Default is ``None``.
    implementation : {"torch"} or None, optional
        Explicit backend. ``None`` selects Torch.

    Returns
    -------
    torch.Tensor
        Displaced points with the same shape, dtype, and device as ``points``.

    The operation is implemented with native Torch tensor operations and
    participates in autograd and :func:`torch.compile`.
    """

    _FORWARD_BENCHMARK_CASES = (
        ("small-n4096-d3-no-point-weights", 1, 4096, 3, "none"),
        ("medium-b4-n16384-d3-bool-point-weights", 4, 16384, 3, "bool"),
        ("large-b8-n32768-d3-float-point-weights", 8, 32768, 3, "float"),
    )
    _BACKWARD_BENCHMARK_CASES = (
        ("medium-n16384-d3-displacement-only", 1, 16384, 3, "displacement"),
        ("medium-n16384-d3-all-gradients", 1, 16384, 3, "all"),
    )
    _COMPARE_ATOL = 1.0e-6
    _COMPARE_RTOL = 1.0e-6

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(
        points: torch.Tensor,
        displacement: torch.Tensor,
        *,
        point_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply dense displacement with the pure-Torch backend."""

        points_b3, displacement_b3, point_weights_b2, was_unbatched = (
            normalize_displace_inputs(points, displacement, point_weights)
        )
        output = displace_points_torch(points_b3, displacement_b3, point_weights_b2)
        return restore_point_rank(output, was_unbatched)

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark cases."""

        device = torch.device(device)
        for seed, (label, batch_size, num_points, num_dims, weight_mode) in enumerate(
            cls._FORWARD_BENCHMARK_CASES
        ):
            generator = torch.Generator(device=device).manual_seed(1701 + seed)
            shape = (
                (num_points, num_dims)
                if batch_size == 1
                else (batch_size, num_points, num_dims)
            )
            points = torch.rand(shape, generator=generator, device=device)
            displacement = 0.1 * torch.randn(shape, generator=generator, device=device)
            if weight_mode == "none":
                point_weights = None
            elif weight_mode == "bool":
                point_weights = (
                    torch.rand(shape[:-1], generator=generator, device=device) > 0.25
                )
            else:
                point_weights = torch.rand(
                    shape[:-1], generator=generator, device=device
                )
            yield label, (points, displacement), {"point_weights": point_weights}

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield differentiable dense-displacement parity cases."""

        device = torch.device(device)
        for (
            label,
            batch_size,
            num_points,
            num_dims,
            gradient_mode,
        ) in cls._BACKWARD_BENCHMARK_CASES:
            # Paired gradient-mode cases intentionally use identical values so
            # their timings isolate the requested gradient set.
            generator = torch.Generator(device=device).manual_seed(1801)
            shape = (
                (num_points, num_dims)
                if batch_size == 1
                else (batch_size, num_points, num_dims)
            )
            points = torch.rand(shape, generator=generator, device=device)
            displacement = torch.randn(shape, generator=generator, device=device)
            point_weights = torch.rand(shape[:-1], generator=generator, device=device)
            all_gradients = gradient_mode == "all"
            yield (
                label,
                (
                    points.requires_grad_(all_gradients),
                    displacement.requires_grad_(True),
                ),
                {
                    "point_weights": point_weights.requires_grad_(all_gradients),
                },
            )

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare dense-displacement benchmark outputs."""

        torch.testing.assert_close(
            output, reference, atol=cls._COMPARE_ATOL, rtol=cls._COMPARE_RTOL
        )

    @classmethod
    def compare_backward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare dense-displacement benchmark gradients."""

        torch.testing.assert_close(
            output, reference, atol=cls._COMPARE_ATOL, rtol=cls._COMPARE_RTOL
        )


class MorphPoints(FunctionSpec):
    r"""Morph points from sparse control displacements using compact Shepard blending.

    For query point :math:`x`, control :math:`c_j`, radius :math:`r_j`, and
    normalized distance :math:`q_j=\lVert x-c_j\rVert/r_j`, define

    .. math::

       \phi(q) = (1-q)^4(4q+1), \qquad
       a_j = \frac{\phi(q_j)}{q_j^2}, \quad 0 < q_j < 1.

    A stationary background of weight one blends the active controls toward
    zero displacement:

    .. math::

       u(x) = \frac{\sum_j a_j d_j}{1 + \sum_j a_j}, \qquad
       x' = x + \mathtt{point\_weights}\,u(x).

    Each control's influence vanishes smoothly at its own support boundary. The
    total field is zero wherever no control support is active; it need not be
    zero at one control's boundary if another support overlaps that location.
    At an exact control coincidence, the unscaled field returns the mean
    displacement of all controls at that coordinate. ``point_weights`` are then
    applied to that field. The implementation uses a stably scaled equivalent
    of the quotient near controls.

    Inputs may be unbatched ``(N, D)``/``(C, D)`` or aligned batched
    ``(B, N, D)``/``(B, C, D)``. Point and control batches are not implicitly
    broadcast. All coordinate and displacement tensors must use float32 or
    float64 and have the same dtype and device.

    Parameters
    ----------
    points : torch.Tensor
        Query points with shape ``(N, D)`` or ``(B, N, D)``.
    control_points : torch.Tensor
        World-coordinate control locations with shape ``(C, D)`` or
        ``(B, C, D)``. Controls need not be query points.
    control_displacements : torch.Tensor
        Control displacement vectors, not destination coordinates. The shape,
        dtype, and device must exactly match ``control_points``.
    radius : float or torch.Tensor
        Support radius. Accepts a scalar, per-control ``(C,)`` tensor, or aligned
        batched ``(B, C)`` tensor. Every value must be positive and finite.
        Tensor radii must match the control dtype and device; their numerical
        values are not validated at runtime.
    point_weights : torch.Tensor or None, optional
        Optional bool or floating per-point weights. Shapes follow
        :class:`DisplacePoints`; point weights are not per-control values. Signed
        and amplifying values are permitted and are used without clamping.
    kernel : {"wendland_c2"}, optional
        Compact radial kernel used by Shepard blending. The explicit name
        reserves the algorithm-selection extension point. Default is
        ``"wendland_c2"``.
    implementation : {"warp", "torch"} or None, optional
        Explicit backend. ``None`` selects Torch on CPU and Warp on CUDA when
        Warp is available, otherwise Torch with a one-time
        :class:`RuntimeWarning`.

    Returns
    -------
    torch.Tensor
        Morphed points with the same shape, dtype, and device as ``points``.

    Notes
    -----
    Both backends propagate first-order gradients through points, controls,
    control displacements, tensor-valued radii, and floating-point weights.
    Only first-order gradients are part of the Warp backend's public contract.

    A learned radius should be parameterized to remain positive, for example as
    ``torch.nn.functional.softplus(raw_radius) + eps`` with a positive ``eps``
    appropriate to the coordinate scale. With zero controls, the operation is
    the identity and the numerical value of a scalar ``radius`` is unused.
    """

    _FORWARD_BENCHMARK_CASES = (
        (
            "zero-controls-n8192-d3",
            1,
            8192,
            0,
            3,
            torch.float32,
            "none",
            False,
        ),
        (
            "float64-n1024-c16-d3",
            1,
            1024,
            16,
            3,
            torch.float64,
            "none",
            False,
        ),
        (
            "exact-handles-n2048-c16-d3",
            1,
            2048,
            16,
            3,
            torch.float32,
            "none",
            True,
        ),
        (
            "small-n2048-c16-d3-no-point-weights",
            1,
            2048,
            16,
            3,
            torch.float32,
            "none",
            False,
        ),
        (
            "control-heavy-n128-c1024-d3",
            1,
            128,
            1024,
            3,
            torch.float32,
            "none",
            False,
        ),
        (
            "medium-b2-n8192-c32-d3-bool-point-weights",
            2,
            8192,
            32,
            3,
            torch.float32,
            "bool",
            False,
        ),
        (
            "large-b4-n16384-c64-d3-float-point-weights",
            4,
            16384,
            64,
            3,
            torch.float32,
            "float",
            False,
        ),
    )
    _BACKWARD_BENCHMARK_CASES = (
        (
            "float64-n1024-c16-d3-all-gradients",
            1,
            1024,
            16,
            3,
            torch.float64,
            "all",
        ),
        (
            "medium-n8192-c32-d3-control-displacement-only",
            1,
            8192,
            32,
            3,
            torch.float32,
            "control_displacement",
        ),
        (
            "medium-n8192-c32-d3-all-gradients",
            1,
            8192,
            32,
            3,
            torch.float32,
            "all",
        ),
        (
            "large-n16384-c64-d3-control-displacement-checkpoint",
            1,
            16384,
            64,
            3,
            torch.float32,
            "control_displacement",
        ),
    )
    _COMPARE_ATOL = 2.0e-5
    _COMPARE_RTOL = 2.0e-5
    _COMPARE_BACKWARD_ATOL = 2.0e-4
    _COMPARE_BACKWARD_RTOL = 2.0e-4

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: torch.Tensor,
        control_points: torch.Tensor,
        control_displacements: torch.Tensor,
        *,
        radius: float | torch.Tensor,
        point_weights: torch.Tensor | None = None,
        kernel: Literal["wendland_c2"] = "wendland_c2",
    ) -> torch.Tensor:
        """Apply compact Shepard morphing with the Warp backend."""

        _validate_kernel(kernel)
        normalized = normalize_morph_inputs(
            points,
            control_points,
            control_displacements,
            radius,
            point_weights,
        )
        (
            points_b3,
            controls_b3,
            control_displacements_b3,
            radius_b2,
            point_weights_b2,
            was_unbatched,
        ) = normalized
        output = morph_points_warp(
            points_b3,
            controls_b3,
            control_displacements_b3,
            radius_b2,
            point_weights_b2,
        )
        return restore_point_rank(output, was_unbatched)

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: torch.Tensor,
        control_points: torch.Tensor,
        control_displacements: torch.Tensor,
        *,
        radius: float | torch.Tensor,
        point_weights: torch.Tensor | None = None,
        kernel: Literal["wendland_c2"] = "wendland_c2",
    ) -> torch.Tensor:
        """Apply compact Shepard morphing with the pure-Torch backend."""

        _validate_kernel(kernel)
        normalized = normalize_morph_inputs(
            points,
            control_points,
            control_displacements,
            radius,
            point_weights,
        )
        (
            points_b3,
            controls_b3,
            control_displacements_b3,
            radius_b2,
            point_weights_b2,
            was_unbatched,
        ) = normalized
        output = morph_points_torch(
            points_b3,
            controls_b3,
            control_displacements_b3,
            radius_b2,
            point_weights_b2,
        )
        return restore_point_rank(output, was_unbatched)

    @classmethod
    def dispatch(
        cls,
        points: torch.Tensor,
        control_points: torch.Tensor,
        control_displacements: torch.Tensor,
        *,
        radius: float | torch.Tensor,
        point_weights: torch.Tensor | None = None,
        kernel: Literal["wendland_c2"] = "wendland_c2",
        implementation: Literal["torch", "warp"] | None = None,
    ) -> torch.Tensor:
        """Select Warp for CUDA inputs and Torch for CPU inputs by default.

        Falling back to Torch on CUDA inputs because Warp is unavailable emits
        the standard one-time :class:`RuntimeWarning`.
        """

        if implementation is None:
            impls = cls._get_impls()
            warp_impl = impls.get("warp")
            if isinstance(points, torch.Tensor) and points.is_cuda:
                if warp_impl is not None and warp_impl.available:
                    implementation = "warp"
                else:
                    cls._warn_fallback(warp_impl, impls["torch"])
                    implementation = "torch"
            else:
                implementation = "torch"
        return super().dispatch(
            points,
            control_points,
            control_displacements,
            radius=radius,
            point_weights=point_weights,
            kernel=kernel,
            implementation=implementation,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative compact-Shepard forward benchmark cases."""

        device = torch.device(device)
        for seed, (
            label,
            batch_size,
            num_points,
            num_controls,
            num_dims,
            dtype,
            weight_mode,
            exact_handles,
        ) in enumerate(cls._FORWARD_BENCHMARK_CASES):
            generator = torch.Generator(device=device).manual_seed(2701 + seed)
            point_shape = (
                (num_points, num_dims)
                if batch_size == 1
                else (batch_size, num_points, num_dims)
            )
            control_shape = (
                (num_controls, num_dims)
                if batch_size == 1
                else (batch_size, num_controls, num_dims)
            )
            points = (
                2
                * torch.rand(
                    point_shape, generator=generator, device=device, dtype=dtype
                )
                - 1
            )
            controls = (
                2
                * torch.rand(
                    control_shape, generator=generator, device=device, dtype=dtype
                )
                - 1
            )
            if exact_handles:
                # Exercise exact-coordinate handling without making the entire
                # workload degenerate.
                points[..., :num_controls, :] = controls
            control_displacements = 0.1 * torch.randn(
                control_shape, generator=generator, device=device, dtype=dtype
            )
            if weight_mode == "none":
                point_weights = None
            elif weight_mode == "bool":
                point_weights = (
                    torch.rand(point_shape[:-1], generator=generator, device=device)
                    > 0.25
                )
            else:
                point_weights = torch.rand(
                    point_shape[:-1],
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            yield (
                label,
                (points, controls, control_displacements),
                {"radius": 0.75, "point_weights": point_weights},
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield differentiable compact-Shepard parity cases."""

        device = torch.device(device)
        for (
            label,
            batch_size,
            num_points,
            num_controls,
            num_dims,
            dtype,
            gradient_mode,
        ) in cls._BACKWARD_BENCHMARK_CASES:
            # Paired gradient-mode cases intentionally use identical values so
            # their timings isolate the requested gradient set.
            generator = torch.Generator(device=device).manual_seed(2801)
            point_shape = (
                (num_points, num_dims)
                if batch_size == 1
                else (batch_size, num_points, num_dims)
            )
            control_shape = (
                (num_controls, num_dims)
                if batch_size == 1
                else (batch_size, num_controls, num_dims)
            )
            points = 1.5 * torch.rand(
                point_shape, generator=generator, device=device, dtype=dtype
            )
            controls = 1.5 * torch.rand(
                control_shape, generator=generator, device=device, dtype=dtype
            )
            control_displacements = 0.1 * torch.randn(
                control_shape, generator=generator, device=device, dtype=dtype
            )
            radius = torch.full(control_shape[:-1], 0.9, device=device, dtype=dtype)
            point_weights = torch.rand(
                point_shape[:-1], generator=generator, device=device, dtype=dtype
            )
            all_gradients = gradient_mode == "all"
            yield (
                label,
                (
                    points.requires_grad_(all_gradients),
                    controls.requires_grad_(all_gradients),
                    control_displacements.requires_grad_(True),
                ),
                {
                    "radius": radius.requires_grad_(all_gradients),
                    "point_weights": point_weights.requires_grad_(all_gradients),
                },
            )

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare compact-Shepard outputs across backends."""

        torch.testing.assert_close(
            output, reference, atol=cls._COMPARE_ATOL, rtol=cls._COMPARE_RTOL
        )

    @classmethod
    def compare_backward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare compact-Shepard gradients across backends."""

        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_BACKWARD_ATOL,
            rtol=cls._COMPARE_BACKWARD_RTOL,
        )


displace_points = DisplacePoints.make_function("displace_points")
morph_points = MorphPoints.make_function("morph_points")


__all__ = [
    "DisplacePoints",
    "MorphPoints",
    "displace_points",
    "morph_points",
]
