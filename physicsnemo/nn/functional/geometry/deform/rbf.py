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

"""Backend-dispatched thin-plate-spline radial-basis point deformation."""

from __future__ import annotations

import math
from numbers import Real
from typing import Literal

import torch
from jaxtyping import Bool, Float

from physicsnemo.core.function_spec import FunctionSpec

from ._rbf_torch_impl import fit_rbf_coefficients_torch, rbf_field_torch
from ._torch_impl import displace_points_torch
from ._utils import _zero_dependency, normalize_rbf_inputs, restore_point_rank
from ._warp_impl import rbf_field_warp


def _validate_rbf_options(
    kernel: str,
    polynomial: bool,
    smoothing: float,
    dtype: torch.dtype,
) -> float:
    """Validate non-tensor options shared by both RBF backends."""

    if kernel != "thin_plate_spline":
        raise ValueError(f"kernel must be 'thin_plate_spline', got {kernel!r}")
    if not isinstance(polynomial, bool):
        raise TypeError(f"polynomial must be a bool, got {type(polynomial).__name__}")
    if not isinstance(smoothing, Real) or isinstance(smoothing, bool):
        raise TypeError(
            "smoothing must be a nonnegative finite Python real scalar, got "
            f"{type(smoothing).__name__}"
        )

    finfo = torch.finfo(dtype)
    if torch.compiler.is_compiling():
        # Dynamo can generalize a call-time Python scalar into a SymFloat. Keep
        # validation for statically known literals without specializing a
        # symbolic smoothing value or forcing a graph break.
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        if statically_known_true(smoothing != smoothing) or statically_known_true(
            abs(smoothing) == math.inf
        ):
            torch._check_value(False, lambda: "smoothing must be finite")
        if statically_known_true(smoothing < 0):
            torch._check_value(False, lambda: "smoothing must be nonnegative")
        if statically_known_true(smoothing > finfo.max):
            torch._check_value(
                False,
                lambda: "smoothing must be finite in the control dtype",
            )
        # Preserve one full graph when Dynamo generalizes a call-time float to
        # SymFloat, while rejecting invalid values before they reach the solve.
        torch._check(
            smoothing >= 0,
            lambda: "smoothing must be nonnegative",
        )
        torch._check(
            smoothing <= finfo.max,
            lambda: "smoothing must be finite in the control dtype",
        )
        return smoothing

    if smoothing < 0:
        raise ValueError("smoothing must be nonnegative")
    if smoothing > finfo.max:
        raise ValueError("smoothing must be finite in the control dtype")
    smoothing_value = float(smoothing)
    if not math.isfinite(smoothing_value):
        raise ValueError("smoothing must be finite")
    return smoothing_value


def _normalize_and_fit(
    points: torch.Tensor,
    control_points: torch.Tensor,
    control_displacements: torch.Tensor,
    point_weights: torch.Tensor | None,
    kernel: str,
    polynomial: bool,
    smoothing: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    bool,
]:
    """Normalize public inputs and fit the shared Torch coefficient solve."""

    normalized = normalize_rbf_inputs(
        points,
        control_points,
        control_displacements,
        point_weights,
    )
    (
        points_b3,
        controls_b3,
        control_displacements_b3,
        point_weights_b2,
        was_unbatched,
    ) = normalized
    smoothing = _validate_rbf_options(kernel, polynomial, smoothing, controls_b3.dtype)

    num_controls = controls_b3.shape[1]
    num_dims = controls_b3.shape[2]
    if polynomial and 0 < num_controls < num_dims + 1:
        raise ValueError(
            "polynomial=True requires at least D + 1 controls. Got "
            f"C={num_controls} and D={num_dims}"
        )

    radial_coefficients, polynomial_coefficients = fit_rbf_coefficients_torch(
        controls_b3,
        control_displacements_b3,
        polynomial=polynomial,
        smoothing=smoothing,
    )
    return (
        points_b3,
        controls_b3,
        radial_coefficients,
        polynomial_coefficients,
        point_weights_b2,
        was_unbatched,
    )


class RadialBasisFunctionDeformPoints(FunctionSpec):
    r"""Deform points with a global thin-plate-spline RBF field.

    Given controls :math:`c_j` with prescribed displacements :math:`d_j`, the
    displacement field is

    .. math::

       u(x) = \sum_j \phi(\lVert x-c_j\rVert)w_j + a_0 + A x,
       \qquad \phi(r)=r^2\log(r), \quad \phi(0)=0.

    By default, the radial coefficients :math:`w_j` and affine coefficients
    :math:`(a_0,A)` are fitted from the standard augmented interpolation system.
    The affine side constraints make the fit unique for distinct controls in
    nondegenerate position and reproduce affine displacement fields exactly.
    With ``smoothing=0``, the field interpolates every control displacement up
    to solver precision. Positive smoothing relaxes exact interpolation. This
    formulation follows the thin-plate-spline interpolant described by
    Bookstein [1].

    Inputs may be unbatched ``(N, D)``/``(C, D)`` or aligned batched
    ``(B, N, D)``/``(B, C, D)``. Point and control batches are not implicitly
    broadcast. All coordinate and displacement tensors must use float32 or
    float64 and have the same dtype and device. The formulation is dimensionally
    generic.

    Parameters
    ----------
    points : torch.Tensor
        Query points with shape ``(N, D)`` or ``(B, N, D)``.
    control_points : torch.Tensor
        World-coordinate control locations with shape ``(C, D)`` or
        ``(B, C, D)``.
    control_displacements : torch.Tensor
        Prescribed displacement vectors, not destination coordinates. Shape,
        dtype, and device must exactly match ``control_points``.
    kernel : {"thin_plate_spline"}, optional
        Radial kernel. Default is ``"thin_plate_spline"``.
    polynomial : bool, optional
        Include the affine polynomial tail and its side constraints. This
        requires at least ``D + 1`` controls when controls are present. Disabling
        the tail can make the radial system singular. Default is ``True``.
    smoothing : float, optional
        Nonnegative diagonal regularization added to the control-kernel block.
        Zero gives exact interpolation for a nonsingular control layout up to
        solver precision. Positive values relax interpolation accuracy. Default
        is ``0.0``.
    point_weights : torch.Tensor or None, optional
        Optional bool or floating per-point multipliers with shape ``(N,)`` or
        ``(B, N)``. They scale the fitted field after interpolation. Signed and
        amplifying values are used without clamping. Bool weights must be on
        the same device as ``points``. Floating weights must have the same
        dtype and device as ``points``.
    implementation : {"torch", "warp"} or None, optional
        Field-evaluation backend. Both implementations fit coefficients with
        a checked differentiable :func:`torch.linalg.solve_ex`. ``None`` selects
        Torch on CPU and Warp on CUDA when Warp is available, otherwise Torch
        with a one-time :class:`RuntimeWarning`.

    Returns
    -------
    torch.Tensor
        Deformed points with the same shape, dtype, and device as ``points``.

    Raises
    ------
    TypeError
        If tensor dtypes or Python argument types are unsupported.
    ValueError
        If tensor shapes, devices, control layout, point weights, or RBF
        options are invalid.
    KeyError
        If ``implementation`` does not name a registered backend.
    ImportError
        If an explicitly requested backend is unavailable.
    RuntimeError
        If runtime validation or coefficient fitting fails, including for a
        singular system or during CUDA Graph capture.

    Notes
    -----
    The dense coefficient solve costs :math:`O(C^3)` and stores an
    :math:`O(C^2)` system. Field evaluation costs :math:`O(NC)`. Reuse of a fit
    across multiple independent query sets is not currently exposed by this
    convenience API.

    Duplicate controls or controls that do not span the affine basis can make
    the augmented system singular. In that case the checked linear solve raises
    an error. With zero controls, the operation is the identity.

    Coefficient fitting is not supported inside CUDA Graph capture because the
    singular-system check requires host interaction.

    Both backends propagate first-order gradients through points, control
    locations, control displacements, and floating point weights. Only
    first-order gradients are part of the Warp evaluator's public contract.

    References
    ----------
    [1] F. L. Bookstein, "Principal warps: thin-plate splines and the
        decomposition of deformations," IEEE Transactions on Pattern Analysis
        and Machine Intelligence, vol. 11, no. 6, pp. 567-585, 1989.
        https://doi.org/10.1109/34.24792
    """

    _FORWARD_BENCHMARK_CASES = (
        (
            "float64-n1024-c12-d2-exact",
            1,
            1024,
            12,
            2,
            torch.float64,
            True,
            0.0,
            "none",
        ),
        (
            "small-n4096-c16-d2-exact",
            1,
            4096,
            16,
            2,
            torch.float32,
            True,
            0.0,
            "none",
        ),
        (
            "medium-b2-n8192-c32-d3-smoothed-float-point-weights",
            2,
            8192,
            32,
            3,
            torch.float32,
            True,
            1.0e-4,
            "float",
        ),
        (
            "no-polynomial-n4096-c24-d3-bool-point-weights",
            1,
            4096,
            24,
            3,
            torch.float32,
            False,
            1.0e-3,
            "bool",
        ),
    )
    _BACKWARD_BENCHMARK_CASES = (
        (
            "float64-n1024-c12-d2-all-gradients",
            1,
            1024,
            12,
            2,
            torch.float64,
            True,
            0.0,
            "all",
        ),
        (
            "medium-n8192-c32-d3-control-displacement-only",
            1,
            8192,
            32,
            3,
            torch.float32,
            True,
            1.0e-4,
            "control_displacement",
        ),
        (
            "medium-n8192-c32-d3-all-gradients",
            1,
            8192,
            32,
            3,
            torch.float32,
            True,
            1.0e-4,
            "all",
        ),
    )
    _COMPARE_ATOL = 3.0e-5
    _COMPARE_RTOL = 3.0e-5
    _COMPARE_BACKWARD_ATOL = 5.0e-4
    _COMPARE_BACKWARD_RTOL = 5.0e-4

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        control_points: Float[torch.Tensor, "*batch num_controls num_dims"],
        control_displacements: Float[torch.Tensor, "*batch num_controls num_dims"],
        *,
        kernel: Literal["thin_plate_spline"] = "thin_plate_spline",
        polynomial: bool = True,
        smoothing: float = 0.0,
        point_weights: Bool[torch.Tensor, "*batch num_points"]
        | Float[torch.Tensor, "*batch num_points"]
        | None = None,
    ) -> Float[torch.Tensor, "*batch num_points num_dims"]:
        """Fit an RBF field with Torch and evaluate it with Warp.

        Parameters
        ----------
        points : torch.Tensor
            Unbatched ``(N, D)`` or batched ``(B, N, D)`` query points.
        control_points : torch.Tensor
            Unbatched or batch-aligned control locations with shape ``(C, D)``
            or ``(B, C, D)``.
        control_displacements : torch.Tensor
            Displacement vectors with the same shape as ``control_points``.
        kernel : {"thin_plate_spline"}, optional
            Radial kernel used by the interpolant.
        polynomial : bool, optional
            Whether to include the affine polynomial tail.
        smoothing : float, optional
            Nonnegative diagonal regularization for the control-kernel block.
        point_weights : torch.Tensor or None, optional
            Optional bool or floating per-point weights.

        Returns
        -------
        torch.Tensor
            Deformed points with the same shape, dtype, and device as
            ``points``.

        Raises
        ------
        TypeError
            If tensor dtypes or Python argument types are unsupported.
        ValueError
            If shapes, devices, control layout, weights, or RBF options are
            invalid.
        RuntimeError
            If runtime validation or coefficient fitting fails, including for
            a singular system or during CUDA Graph capture.
        """

        (
            points_b3,
            controls_b3,
            radial_coefficients,
            polynomial_coefficients,
            point_weights_b2,
            was_unbatched,
        ) = _normalize_and_fit(
            points,
            control_points,
            control_displacements,
            point_weights,
            kernel,
            polynomial,
            smoothing,
        )
        if controls_b3.shape[1] == 0:
            zero = _zero_dependency(
                controls_b3,
                radial_coefficients,
                polynomial_coefficients,
                point_weights_b2,
            )
            return restore_point_rank(points_b3 + zero, was_unbatched)
        field = rbf_field_warp(
            points_b3,
            controls_b3,
            radial_coefficients,
            polynomial_coefficients,
        )
        output = displace_points_torch(points_b3, field, point_weights_b2)
        return restore_point_rank(output, was_unbatched)

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        control_points: Float[torch.Tensor, "*batch num_controls num_dims"],
        control_displacements: Float[torch.Tensor, "*batch num_controls num_dims"],
        *,
        kernel: Literal["thin_plate_spline"] = "thin_plate_spline",
        polynomial: bool = True,
        smoothing: float = 0.0,
        point_weights: Bool[torch.Tensor, "*batch num_points"]
        | Float[torch.Tensor, "*batch num_points"]
        | None = None,
    ) -> Float[torch.Tensor, "*batch num_points num_dims"]:
        """Fit and evaluate an RBF field with Torch.

        Parameters
        ----------
        points : torch.Tensor
            Unbatched ``(N, D)`` or batched ``(B, N, D)`` query points.
        control_points : torch.Tensor
            Unbatched or batch-aligned control locations with shape ``(C, D)``
            or ``(B, C, D)``.
        control_displacements : torch.Tensor
            Displacement vectors with the same shape as ``control_points``.
        kernel : {"thin_plate_spline"}, optional
            Radial kernel used by the interpolant.
        polynomial : bool, optional
            Whether to include the affine polynomial tail.
        smoothing : float, optional
            Nonnegative diagonal regularization for the control-kernel block.
        point_weights : torch.Tensor or None, optional
            Optional bool or floating per-point weights.

        Returns
        -------
        torch.Tensor
            Deformed points with the same shape, dtype, and device as
            ``points``.

        Raises
        ------
        TypeError
            If tensor dtypes or Python argument types are unsupported.
        ValueError
            If shapes, devices, control layout, weights, or RBF options are
            invalid.
        RuntimeError
            If runtime validation or coefficient fitting fails, including for
            a singular system or during CUDA Graph capture.
        """

        (
            points_b3,
            controls_b3,
            radial_coefficients,
            polynomial_coefficients,
            point_weights_b2,
            was_unbatched,
        ) = _normalize_and_fit(
            points,
            control_points,
            control_displacements,
            point_weights,
            kernel,
            polynomial,
            smoothing,
        )
        if controls_b3.shape[1] == 0:
            zero = _zero_dependency(
                controls_b3,
                radial_coefficients,
                polynomial_coefficients,
                point_weights_b2,
            )
            return restore_point_rank(points_b3 + zero, was_unbatched)
        field = rbf_field_torch(
            points_b3,
            controls_b3,
            radial_coefficients,
            polynomial_coefficients,
        )
        output = displace_points_torch(points_b3, field, point_weights_b2)
        return restore_point_rank(output, was_unbatched)

    @classmethod
    def dispatch(
        cls,
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        control_points: Float[torch.Tensor, "*batch num_controls num_dims"],
        control_displacements: Float[torch.Tensor, "*batch num_controls num_dims"],
        *,
        kernel: Literal["thin_plate_spline"] = "thin_plate_spline",
        polynomial: bool = True,
        smoothing: float = 0.0,
        point_weights: Bool[torch.Tensor, "*batch num_points"]
        | Float[torch.Tensor, "*batch num_points"]
        | None = None,
        implementation: Literal["torch", "warp"] | None = None,
    ) -> Float[torch.Tensor, "*batch num_points num_dims"]:
        """Select Warp for CUDA inputs and Torch for CPU inputs by default.

        Falling back to Torch on CUDA inputs because Warp is unavailable emits
        the standard one-time :class:`RuntimeWarning`.

        Parameters
        ----------
        points : torch.Tensor
            Unbatched ``(N, D)`` or batched ``(B, N, D)`` query points.
        control_points : torch.Tensor
            Unbatched or batch-aligned control locations with shape ``(C, D)``
            or ``(B, C, D)``.
        control_displacements : torch.Tensor
            Displacement vectors with the same shape as ``control_points``.
        kernel : {"thin_plate_spline"}, optional
            Radial kernel used by the interpolant.
        polynomial : bool, optional
            Whether to include the affine polynomial tail.
        smoothing : float, optional
            Nonnegative diagonal regularization for the control-kernel block.
        point_weights : torch.Tensor or None, optional
            Optional bool or floating per-point weights.
        implementation : {"torch", "warp"} or None, optional
            Explicit backend selection. ``None`` selects according to the
            point device and backend availability.

        Returns
        -------
        torch.Tensor
            Deformed points with the same shape, dtype, and device as
            ``points``.

        Raises
        ------
        TypeError
            If tensor dtypes or Python argument types are unsupported.
        ValueError
            If shapes, devices, control layout, weights, or RBF options are
            invalid.
        KeyError
            If ``implementation`` does not name a registered backend.
        ImportError
            If an explicitly requested backend is unavailable.
        RuntimeError
            If runtime validation or coefficient fitting fails, including for
            a singular system or during CUDA Graph capture.
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
            kernel=kernel,
            polynomial=polynomial,
            smoothing=smoothing,
            point_weights=point_weights,
            implementation=implementation,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative RBF forward benchmark cases.

        Parameters
        ----------
        device : torch.device or str, optional
            Device on which to construct the benchmark tensors.

        Yields
        ------
        tuple
            Benchmark label, positional arguments, and keyword arguments.
        """

        device = torch.device(device)
        for seed, (
            label,
            batch_size,
            num_points,
            num_controls,
            num_dims,
            dtype,
            polynomial,
            smoothing,
            weight_mode,
        ) in enumerate(cls._FORWARD_BENCHMARK_CASES):
            generator = torch.Generator(device=device).manual_seed(3701 + seed)
            points, controls, control_displacements, point_weights = (
                cls._make_benchmark_tensors(
                    device,
                    generator,
                    batch_size,
                    num_points,
                    num_controls,
                    num_dims,
                    dtype,
                    weight_mode,
                )
            )
            yield (
                label,
                (points, controls, control_displacements),
                {
                    "polynomial": polynomial,
                    "smoothing": smoothing,
                    "point_weights": point_weights,
                },
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield differentiable RBF parity cases.

        Parameters
        ----------
        device : torch.device or str, optional
            Device on which to construct the benchmark tensors.

        Yields
        ------
        tuple
            Benchmark label, differentiable positional arguments, and keyword
            arguments.
        """

        device = torch.device(device)
        for seed, (
            label,
            batch_size,
            num_points,
            num_controls,
            num_dims,
            dtype,
            polynomial,
            smoothing,
            gradient_mode,
        ) in enumerate(cls._BACKWARD_BENCHMARK_CASES):
            generator = torch.Generator(device=device).manual_seed(3801 + seed)
            points, controls, control_displacements, point_weights = (
                cls._make_benchmark_tensors(
                    device,
                    generator,
                    batch_size,
                    num_points,
                    num_controls,
                    num_dims,
                    dtype,
                    "float",
                )
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
                    "polynomial": polynomial,
                    "smoothing": smoothing,
                    "point_weights": point_weights.requires_grad_(all_gradients),
                },
            )

    @staticmethod
    def _make_benchmark_tensors(
        device: torch.device,
        generator: torch.Generator,
        batch_size: int,
        num_points: int,
        num_controls: int,
        num_dims: int,
        dtype: torch.dtype,
        weight_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Construct deterministic, nondegenerate benchmark inputs."""

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
                point_shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            - 1
        )
        controls = (
            2
            * torch.rand(
                control_shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            - 1
        )
        control_displacements = 0.1 * torch.randn(
            control_shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        if weight_mode == "none":
            point_weights = None
        elif weight_mode == "bool":
            point_weights = (
                torch.rand(
                    point_shape[:-1],
                    generator=generator,
                    device=device,
                )
                > 0.25
            )
        else:
            point_weights = torch.rand(
                point_shape[:-1],
                generator=generator,
                device=device,
                dtype=dtype,
            )
        return points, controls, control_displacements, point_weights

    @classmethod
    def compare_forward(
        cls,
        output: Float[torch.Tensor, "*batch num_points num_dims"],
        reference: Float[torch.Tensor, "*batch num_points num_dims"],
    ) -> None:
        """Compare RBF outputs across field-evaluation backends.

        Parameters
        ----------
        output, reference : torch.Tensor
            Backend result and baseline result with matching point shapes.

        Raises
        ------
        AssertionError
            If the results differ beyond the configured tolerances.
        """

        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_ATOL,
            rtol=cls._COMPARE_RTOL,
        )

    @classmethod
    def compare_backward(
        cls,
        output: Float[torch.Tensor, "..."],
        reference: Float[torch.Tensor, "..."],
    ) -> None:
        """Compare RBF gradients across field-evaluation backends.

        Parameters
        ----------
        output, reference : torch.Tensor
            Backend gradient and baseline gradient with matching shapes.

        Raises
        ------
        AssertionError
            If the gradients differ beyond the configured tolerances.
        """

        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_BACKWARD_ATOL,
            rtol=cls._COMPARE_BACKWARD_RTOL,
        )


radial_basis_function_deform_points = RadialBasisFunctionDeformPoints.make_function(
    "radial_basis_function_deform_points"
)


__all__ = [
    "RadialBasisFunctionDeformPoints",
    "radial_basis_function_deform_points",
]
