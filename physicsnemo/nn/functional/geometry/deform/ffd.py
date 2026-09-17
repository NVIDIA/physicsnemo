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

"""Backend-dispatched lattice free-form deformation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from jaxtyping import Bool, Float

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import ffd_points_torch
from ._utils import (
    _FFD_MIN_NODES,
    _FFDBasis,
    normalize_ffd_inputs,
    restore_point_rank,
)
from ._warp_impl import ffd_points_warp

_FFD_BASES = tuple(_FFD_MIN_NODES)


def _validate_basis(basis: str) -> None:
    """Validate the public lattice-basis selector."""

    if basis not in _FFD_BASES:
        raise ValueError(f"basis must be one of {_FFD_BASES}, got {basis!r}")


class FreeFormDeformPoints(FunctionSpec):
    r"""Deform points with a control lattice by free-form deformation.

    An :math:`n_1 \times \dots \times n_D` array of control displacements
    defines a field over the axis-aligned box with corner ``origin`` :math:`o`
    and edge lengths ``extent`` :math:`e`. For each point inside the box, define
    the local coordinates as

    .. math::

       u_d = \frac{x_d - o_d}{e_d} \in [0, 1].

    The tensor-product interpolation of the control displacements
    :math:`\Delta P` gives the displacement:

    .. math::

       d(x) = \sum_{i_1=0}^{n_1-1} \dots \sum_{i_D=0}^{n_D-1}
       \left[\prod_{d=1}^{D} b_{i_d}(u_d)\right] \Delta P_{i_1 \dots i_D},
       \qquad x' = x + \mathtt{point\_weights}\,d(x).

    Points outside the box remain unchanged. All supported bases form a
    partition of unity, so a lattice of zero displacements is exactly the
    identity and a constant lattice translates every point inside the box.

    ``basis="bernstein"`` provides the classic free-form deformation basis from
    Sederberg and Parry (1986) [1]. Its per-axis functions are the Bernstein
    polynomials of degree :math:`p_d = n_d - 1`:

    .. math::

       b_i(u) = \binom{p_d}{i} u^i (1-u)^{p_d - i},

    Every lattice node influences every point in the box. ``basis="bspline"``
    uses uniform cubic B-splines. The axis is divided into
    :math:`n_d - 3` knot spans, and only the four
    coefficients around the containing span influence a point, independent of
    the lattice resolution. Coefficient index :math:`i` is associated with the
    Greville coordinate :math:`(i - 1) / (n_d - 3)`. The first and last
    coefficient planes lie one knot spacing outside the evaluation box.

    ``basis="linear"``, ``"cubic_hermite"``, and ``"quintic_hermite"`` are
    local, node-interpolating alternatives. Each axis is divided into
    :math:`n_d - 1` cells, and only the two nodes bracketing a point contribute
    along that axis. If :math:`t \in [0, 1]` is the coordinate within a cell,
    the upper-node weights are :math:`t`, :math:`3t^2 - 2t^3`, and
    :math:`t^3(6t^2 - 15t + 10)`, respectively. The lower-node weight is one
    minus the upper-node weight. The resulting fields are C0, C1, and C2 across
    cell boundaries, respectively. Perlin introduced the quintic polynomial
    in "Improving Noise" [2] to remove the cubic polynomial's second-derivative
    discontinuities.

    Inputs may be unbatched (``points`` of shape ``(N, D)``) or batched
    (``(B, N, D)``). The corresponding control lattice has shape
    ``(n_1, ..., n_D, D)`` or ``(B, n_1, ..., n_D, D)``. Points, control
    lattices, and point weights must be either unbatched or batch-aligned. They
    are not broadcast. For batched inputs, ``(D,)`` box vectors are shared
    across the batch, while ``(B, D)`` box vectors are aligned. Float32 and
    float64 are supported.

    Parameters
    ----------
    points : torch.Tensor
        Query points with shape ``(N, D)`` or ``(B, N, D)``.
    control_displacements : torch.Tensor
        Displacement vectors, not destination coordinates, for every lattice
        node, with shape ``(n_1, ..., n_D, D)`` or ``(B, n_1, ..., n_D, D)``
        and the same dtype and device as ``points``. Each axis needs at least
        two nodes for ``"bernstein"`` and the node-interpolating bases, and
        four for ``"bspline"``. For ``"bernstein"`` and the interpolating
        bases, node ``(i_1, ..., i_D)`` sits at coordinate
        ``origin_d + extent_d * i_d / (n_d - 1)`` along each axis. Uniform
        cubic B-spline coefficient ``i_d`` is associated with
        ``origin_d + extent_d * (i_d - 1) / (n_d - 3)`` and is generally not
        interpolated by the field.
    origin : torch.Tensor or sequence of float
        Minimum corner of the lattice box with shape ``(D,)`` or aligned
        batched shape ``(B, D)``. Tensor values must match the point dtype and
        device. The operation uses tensor values without runtime validation. It
        converts and validates sequences on the host. For repeated GPU calls,
        create device tensors once and reuse them.
    extent : torch.Tensor or sequence of float
        Edge lengths of the lattice box with the same accepted shapes as
        ``origin``. Every value must be strictly positive and finite. The
        operation does not validate tensor values at runtime.
    basis : {"bernstein", "bspline", "linear", "cubic_hermite", "quintic_hermite"}, optional
        Per-axis basis family. ``"bernstein"`` provides classic global-support
        FFD for coarse lattices. ``"bspline"`` uses local four-node-per-axis
        support and scales to fine lattices. ``"linear"``,
        ``"cubic_hermite"``, and ``"quintic_hermite"`` are local
        two-node-per-axis bases that interpolate every lattice node, with
        progressively smoother transitions. Default is ``"bernstein"``.
    point_weights : torch.Tensor or None, optional
        Optional bool or floating per-point weights. Accepted shapes are
        ``(N,)`` for unbatched inputs and ``(B, N)`` for batched inputs.
        All weights must match the point device. Floating weights must also
        match the point dtype. Bool values act as hard masks. The operation
        uses values as supplied without clamping. Default is ``None``.
    implementation : {"torch", "warp"} or None, optional
        Explicit backend. ``None`` selects Torch on CPU. On CUDA, it selects
        Warp when available and otherwise Torch with a one-time
        :class:`RuntimeWarning`.

    Returns
    -------
    torch.Tensor
        Deformed points with the same shape, dtype, and device as ``points``.

    Raises
    ------
    TypeError
        If tensor dtypes or Python argument types are unsupported.
    ValueError
        If tensor shapes, devices, lattice parameters, point weights, or
        ``basis`` are invalid.
    KeyError
        If ``implementation`` does not name a registered backend.
    ImportError
        If an explicitly requested backend is unavailable.

    Notes
    -----
    Both backends propagate first-order gradients through points, control
    displacements, and floating point weights. ``origin`` and ``extent`` are
    non-differentiable lattice parameters. They must not require gradients.
    Only first-order gradients are part of the Warp backend's public contract.

    The deformation is generally not continuous across the box boundary. An
    outside point stays fixed while a neighboring inside point moves. To keep
    the exterior fixed, set the outermost coefficient plane on every Bernstein
    or node-interpolating face to zero. For cubic B-splines, set the first and
    last three coefficient planes on every axis to zero because three planes
    have nonzero weight at each box face.

    Bernstein degree, global support, and evaluation cost grow with the lattice
    resolution. Use ``"bspline"`` for fine lattices or local control. The
    lattice resolution is a static parameter under :func:`torch.compile`. Each
    distinct resolution compiles its own graph. Eager Torch evaluation chunks
    query points to keep estimated live FFD temporaries within 256 MiB. Compiled
    Torch evaluation instead uses one vectorized block because symbolic chunk
    loops cannot be unrolled, so it does not enforce that budget. Very large
    Bernstein workloads may therefore require substantially more peak memory
    when compiled.

    References
    ----------
    [1] Sederberg, T. W., and Parry, S. R. (1986). "Free-Form Deformation of
    Solid Geometric Models." ACM SIGGRAPH Computer Graphics, 20(4), 151-160.
    https://doi.org/10.1145/15886.15903

    [2] Perlin, K. (2002). "Improving Noise." ACM Transactions on Graphics,
    21(3), 681-682. https://doi.org/10.1145/566654.566636
    """

    _FORWARD_BENCHMARK_CASES = (
        (
            "bernstein-n8192-r5x5x5",
            1,
            8192,
            (5, 5, 5),
            torch.float32,
            "bernstein",
            "none",
            0.0,
        ),
        (
            "bernstein-2d-n8192-r6x6",
            1,
            8192,
            (6, 6),
            torch.float32,
            "bernstein",
            "none",
            0.0,
        ),
        (
            "float64-bernstein-n1024-r4x4x4",
            1,
            1024,
            (4, 4, 4),
            torch.float64,
            "bernstein",
            "none",
            0.0,
        ),
        (
            "bspline-n8192-r16x16x16",
            1,
            8192,
            (16, 16, 16),
            torch.float32,
            "bspline",
            "none",
            0.0,
        ),
        (
            "bspline-large-b4-n16384-r32x32x32",
            4,
            16384,
            (32, 32, 32),
            torch.float32,
            "bspline",
            "float",
            0.0,
        ),
        (
            "bspline-half-outside-n8192-r8x8x8",
            1,
            8192,
            (8, 8, 8),
            torch.float32,
            "bspline",
            "none",
            0.5,
        ),
        (
            "quintic-hermite-n8192-r16x16x16",
            1,
            8192,
            (16, 16, 16),
            torch.float32,
            "quintic_hermite",
            "none",
            0.0,
        ),
        (
            "bernstein-b2-n8192-r5x5x5-bool-point-weights",
            2,
            8192,
            (5, 5, 5),
            torch.float32,
            "bernstein",
            "bool",
            0.0,
        ),
    )
    _BACKWARD_BENCHMARK_CASES = (
        (
            "float64-bernstein-n1024-r4x4x4-all-gradients",
            1,
            1024,
            (4, 4, 4),
            torch.float64,
            "bernstein",
            "all",
        ),
        (
            "bernstein-n8192-r5x5x5-control-displacement-only",
            1,
            8192,
            (5, 5, 5),
            torch.float32,
            "bernstein",
            "control_displacement",
        ),
        (
            "bspline-n8192-r16x16x16-control-displacement-only",
            1,
            8192,
            (16, 16, 16),
            torch.float32,
            "bspline",
            "control_displacement",
        ),
        (
            "bspline-n8192-r16x16x16-all-gradients",
            1,
            8192,
            (16, 16, 16),
            torch.float32,
            "bspline",
            "all",
        ),
    )
    _COMPARE_ATOL = 2.0e-5
    _COMPARE_RTOL = 2.0e-5
    _COMPARE_BACKWARD_ATOL = 2.0e-4
    _COMPARE_BACKWARD_RTOL = 2.0e-4

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        control_displacements: Float[torch.Tensor, "*lattice_resolution num_dims"],
        *,
        origin: Float[torch.Tensor, "*box_batch num_dims"] | Sequence[float],
        extent: Float[torch.Tensor, "*box_batch num_dims"] | Sequence[float],
        basis: _FFDBasis = "bernstein",
        point_weights: Bool[torch.Tensor, "*batch num_points"]
        | Float[torch.Tensor, "*batch num_points"]
        | None = None,
    ) -> Float[torch.Tensor, "*batch num_points num_dims"]:
        """Apply lattice free-form deformation with the Warp backend."""

        _validate_basis(basis)
        (
            points_b3,
            lattice_b3,
            origin_b2,
            extent_b2,
            resolution,
            point_weights_b2,
            was_unbatched,
        ) = normalize_ffd_inputs(
            points, control_displacements, origin, extent, basis, point_weights
        )
        output = ffd_points_warp(
            points_b3,
            lattice_b3,
            origin_b2,
            extent_b2,
            resolution,
            basis,
            point_weights_b2,
        )
        return restore_point_rank(output, was_unbatched)

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        control_displacements: Float[torch.Tensor, "*lattice_resolution num_dims"],
        *,
        origin: Float[torch.Tensor, "*box_batch num_dims"] | Sequence[float],
        extent: Float[torch.Tensor, "*box_batch num_dims"] | Sequence[float],
        basis: _FFDBasis = "bernstein",
        point_weights: Bool[torch.Tensor, "*batch num_points"]
        | Float[torch.Tensor, "*batch num_points"]
        | None = None,
    ) -> Float[torch.Tensor, "*batch num_points num_dims"]:
        """Apply lattice free-form deformation with the Torch backend."""

        _validate_basis(basis)
        (
            points_b3,
            lattice_b3,
            origin_b2,
            extent_b2,
            resolution,
            point_weights_b2,
            was_unbatched,
        ) = normalize_ffd_inputs(
            points, control_displacements, origin, extent, basis, point_weights
        )
        output = ffd_points_torch(
            points_b3,
            lattice_b3,
            origin_b2,
            extent_b2,
            resolution,
            basis,
            point_weights_b2,
        )
        return restore_point_rank(output, was_unbatched)

    @classmethod
    def dispatch(
        cls,
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        control_displacements: Float[torch.Tensor, "*lattice_resolution num_dims"],
        *,
        origin: Float[torch.Tensor, "*box_batch num_dims"] | Sequence[float],
        extent: Float[torch.Tensor, "*box_batch num_dims"] | Sequence[float],
        basis: _FFDBasis = "bernstein",
        point_weights: Bool[torch.Tensor, "*batch num_points"]
        | Float[torch.Tensor, "*batch num_points"]
        | None = None,
        implementation: Literal["torch", "warp"] | None = None,
    ) -> Float[torch.Tensor, "*batch num_points num_dims"]:
        """Select a backend and apply lattice free-form deformation."""

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
            control_displacements,
            origin=origin,
            extent=extent,
            basis=basis,
            point_weights=point_weights,
            implementation=implementation,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark cases.

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
            resolution,
            dtype,
            basis,
            weight_mode,
            outside_fraction,
        ) in enumerate(cls._FORWARD_BENCHMARK_CASES):
            generator = torch.Generator(device=device).manual_seed(3701 + seed)
            num_dims = len(resolution)
            point_shape = (
                (num_points, num_dims)
                if batch_size == 1
                else (batch_size, num_points, num_dims)
            )
            lattice_shape = (
                (*resolution, num_dims)
                if batch_size == 1
                else (batch_size, *resolution, num_dims)
            )
            points = torch.rand(
                point_shape, generator=generator, device=device, dtype=dtype
            )
            num_outside = int(num_points * outside_fraction)
            if num_outside:
                points[..., :num_outside, :] += 1.5
            control_displacements = 0.1 * torch.randn(
                lattice_shape, generator=generator, device=device, dtype=dtype
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
                (points, control_displacements),
                {
                    "origin": [0.0] * num_dims,
                    "extent": [1.0] * num_dims,
                    "basis": basis,
                    "point_weights": point_weights,
                },
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative backward benchmark cases.

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
        for (
            label,
            batch_size,
            num_points,
            resolution,
            dtype,
            basis,
            gradient_mode,
        ) in cls._BACKWARD_BENCHMARK_CASES:
            # Paired gradient-mode cases intentionally use identical values so
            # their timings isolate the requested gradient set.
            generator = torch.Generator(device=device).manual_seed(3801)
            num_dims = len(resolution)
            point_shape = (
                (num_points, num_dims)
                if batch_size == 1
                else (batch_size, num_points, num_dims)
            )
            lattice_shape = (
                (*resolution, num_dims)
                if batch_size == 1
                else (batch_size, *resolution, num_dims)
            )
            points = torch.rand(
                point_shape, generator=generator, device=device, dtype=dtype
            )
            control_displacements = 0.1 * torch.randn(
                lattice_shape, generator=generator, device=device, dtype=dtype
            )
            point_weights = torch.rand(
                point_shape[:-1], generator=generator, device=device, dtype=dtype
            )
            all_gradients = gradient_mode == "all"
            yield (
                label,
                (
                    points.requires_grad_(all_gradients),
                    control_displacements.requires_grad_(True),
                ),
                {
                    "origin": [0.0] * num_dims,
                    "extent": [1.0] * num_dims,
                    "basis": basis,
                    "point_weights": point_weights.requires_grad_(all_gradients),
                },
            )

    @classmethod
    def compare_forward(
        cls,
        output: Float[torch.Tensor, "*batch num_points num_dims"],
        reference: Float[torch.Tensor, "*batch num_points num_dims"],
    ) -> None:
        """Compare forward results from two backends.

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
            output, reference, atol=cls._COMPARE_ATOL, rtol=cls._COMPARE_RTOL
        )

    @classmethod
    def compare_backward(
        cls,
        output: Float[torch.Tensor, "..."],
        reference: Float[torch.Tensor, "..."],
    ) -> None:
        """Compare backward results from two backends.

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


free_form_deform_points = FreeFormDeformPoints.make_function("free_form_deform_points")


__all__ = [
    "FreeFormDeformPoints",
    "free_form_deform_points",
]
