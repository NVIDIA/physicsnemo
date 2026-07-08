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

"""Differentiable deformation driven by a regular lattice."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from jaxtyping import Bool, Float

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import lattice_deform_points_torch
from ._warp_impl import lattice_deform_points_warp
from .utils import LatticeInterpolation


class LatticeDeformPoints(FunctionSpec):
    r"""Deform points with displacements stored on a regular control lattice.

    A low-resolution, axis-aligned lattice provides one displacement vector at
    each control node. The displacement of every query point is interpolated
    from nearby controls and added to its original position:

    .. math::

       \mathbf{x}' = \mathbf{x} + w(\mathbf{x})
       \sum_{i \in \mathcal{N}(\mathbf{x})}
       \phi_i(\mathbf{x})\,\mathbf{d}_i.

    This gives geometry optimization a compact set of design variables while
    preserving gradients with respect to the input points, control
    displacements, and optional point weights. It is a regular-lattice
    deformation, not a general cage deformation or a classical Bernstein-basis
    free-form deformation.

    Parameters
    ----------
    points : torch.Tensor
        Unbatched query points with shape ``(num_points, dim)``. ``dim`` must be
        1, 2, or 3 and the dtype must be float32 or float64. Points must be
        finite and lie inside ``lattice_bounds``; coordinate values are not
        validated at runtime.
    control_displacements : torch.Tensor
        Channel-first displacement lattice with shape
        ``(dim, n_1, ..., n_dim)``. The spatial sizes determine the number of
        uniformly spaced controls along each axis. Values must be finite; the
        dtype and device must match ``points``. Values are not validated at
        runtime.
    lattice_bounds : Sequence[tuple[float, float]]
        Inclusive ``(lower, upper)`` bounds for each lattice axis. Bounds are
        static metadata and are not differentiable.
    interpolation_type : {"linear", "smooth_step_1", "smooth_step_2"}, optional
        Local interpolation basis. ``linear`` exactly reproduces affine
        displacement fields. ``smooth_step_1`` and ``smooth_step_2`` provide
        progressively smoother transitions between controls. Default
        ``"smooth_step_2"``.
    point_weights : torch.Tensor or None, optional
        Per-point deformation multiplier with shape ``(num_points,)``. A value
        of zero fixes a point and one applies the full lattice displacement.
        Negative values reverse displacement and values above one amplify it.
        Boolean masks are accepted; floating weights must match the dtype and
        device of ``points`` and contain finite values. Weight values are not
        validated at runtime. Default ``None`` (all points use weight 1).
    implementation : {"warp", "torch"} or None, optional
        Backend selection. ``None`` selects Warp when it is available and falls
        back to Torch with a one-time warning otherwise.

    Returns
    -------
    torch.Tensor
        Deformed points with the same shape, dtype, and device as ``points``.

    Raises
    ------
    TypeError
        If tensor inputs, dtypes, bounds metadata, or interpolation type have
        invalid types.
    ValueError
        If tensor shapes, devices, bounds values, or interpolation choice are
        incompatible with lattice deformation.

    Notes
    -----
    The operation is currently unbatched and the lattice is uniformly spaced
    and axis-aligned. World coordinates are normalized to the unit lattice
    before interpolation, preventing large coordinate offsets from collapsing
    the grid metadata. The delegated implementations still construct lattice
    coordinates in float32, and Warp evaluates the interpolation itself in
    float32, so float64 results do not provide full double precision.

    The Warp implementation is a composite adapter around the existing
    custom-op-backed Warp implementation of ``grid_to_point_interpolation``;
    this functional does not launch raw Warp kernels directly. Both backends
    support eager autograd. ``torch.compile`` supports the Torch forward and
    backward paths and the Warp forward path. Compiled Warp backward is not
    currently supported by the delegated interpolation custom op; select
    ``implementation="torch"`` for compiled training.

    The delegated Torch interpolator currently supports one backward traversal
    per forward graph. Recompute the forward result before requesting another
    vector-Jacobian product from the same inputs.

    """

    _BENCHMARK_CASES = (
        ("2d-g16-n4096", 2, 16, 4096),
        ("3d-g12-n4096", 3, 12, 4096),
        ("3d-g24-n16384", 3, 24, 16384),
    )
    _COMPARE_ATOL = 5e-5
    _COMPARE_RTOL = 1e-4
    _COMPARE_BACKWARD_ATOL = 2e-2
    _COMPARE_BACKWARD_RTOL = 5e-2

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: Float[torch.Tensor, "num_points dim"],
        control_displacements: Float[torch.Tensor, "dim *lattice_shape"],  # noqa: F821
        lattice_bounds: Sequence[tuple[float, float]],
        *,
        interpolation_type: LatticeInterpolation = "smooth_step_2",
        point_weights: Float[torch.Tensor, "num_points"]  # noqa: F821
        | Bool[torch.Tensor, "num_points"]  # noqa: F821
        | None = None,
    ) -> Float[torch.Tensor, "num_points dim"]:
        """Dispatch regular-lattice deformation to the Warp adapter."""
        return lattice_deform_points_warp(
            points=points,
            control_displacements=control_displacements,
            lattice_bounds=lattice_bounds,
            interpolation_type=interpolation_type,
            point_weights=point_weights,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: Float[torch.Tensor, "num_points dim"],
        control_displacements: Float[torch.Tensor, "dim *lattice_shape"],  # noqa: F821
        lattice_bounds: Sequence[tuple[float, float]],
        *,
        interpolation_type: LatticeInterpolation = "smooth_step_2",
        point_weights: Float[torch.Tensor, "num_points"]  # noqa: F821
        | Bool[torch.Tensor, "num_points"]  # noqa: F821
        | None = None,
    ) -> Float[torch.Tensor, "num_points dim"]:
        """Dispatch regular-lattice deformation to eager PyTorch."""
        return lattice_deform_points_torch(
            points=points,
            control_displacements=control_displacements,
            lattice_bounds=lattice_bounds,
            interpolation_type=interpolation_type,
            point_weights=point_weights,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark and parity inputs."""
        device = torch.device(device)
        for case_index, (label, dim, grid_size, num_points) in enumerate(
            cls._BENCHMARK_CASES
        ):
            generator = torch.Generator(device=device).manual_seed(4100 + case_index)
            points = (
                1.8
                * torch.rand(
                    num_points,
                    dim,
                    device=device,
                    generator=generator,
                )
                - 0.9
            )
            controls = 0.05 * torch.randn(
                (dim,) + (grid_size,) * dim,
                device=device,
                generator=generator,
            )
            point_weights = torch.rand(
                num_points,
                device=device,
                generator=generator,
            )
            bounds = [(-1.0, 1.0)] * dim
            yield (
                label,
                (points, controls, bounds),
                {
                    "interpolation_type": "smooth_step_2",
                    "point_weights": point_weights,
                },
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative backward benchmark and parity inputs."""
        device = torch.device(device)
        for case_index, (label, dim, grid_size, num_points) in enumerate(
            cls._BENCHMARK_CASES
        ):
            generator = torch.Generator(device=device).manual_seed(5100 + case_index)
            points = (
                1.8
                * torch.rand(
                    num_points,
                    dim,
                    device=device,
                    generator=generator,
                )
                - 0.9
            ).requires_grad_(True)
            controls = (
                0.05
                * torch.randn(
                    (dim,) + (grid_size,) * dim,
                    device=device,
                    generator=generator,
                )
            ).requires_grad_(True)
            point_weights = torch.rand(
                num_points,
                device=device,
                generator=generator,
                requires_grad=True,
            )
            bounds = [(-1.0, 1.0)] * dim
            yield (
                label,
                (points, controls, bounds),
                {
                    "interpolation_type": "smooth_step_2",
                    "point_weights": point_weights,
                },
            )

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare forward outputs across implementations."""
        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_ATOL,
            rtol=cls._COMPARE_RTOL,
        )

    @classmethod
    def compare_backward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare backward gradients across implementations."""
        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_BACKWARD_ATOL,
            rtol=cls._COMPARE_BACKWARD_RTOL,
        )


lattice_deform_points = LatticeDeformPoints.make_function("lattice_deform_points")

__all__ = ["LatticeDeformPoints", "lattice_deform_points"]
