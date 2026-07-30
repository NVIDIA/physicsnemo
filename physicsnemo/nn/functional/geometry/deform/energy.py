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

"""Backend-dispatched differentiable deformation energies."""

from __future__ import annotations

import math
from numbers import Real
from typing import Callable, Literal

import torch
from jaxtyping import Float, Int

from physicsnemo.core.function_spec import FunctionSpec

from ._energy_torch_impl import (
    closed_surface_volume_contributions_torch,
    hinge_bending_terms_torch,
    simplex_inversion_terms_torch,
    simplex_measure_components_torch,
    simplex_stvk_terms_torch,
)
from ._energy_utils import (
    _Reduction,
    normalize_closed_surface_inputs,
    normalize_hinge_inputs,
    normalize_simplex_inputs,
    reduce_energy_terms,
)
from ._warp_impl import (
    closed_surface_volume_contributions_warp,
    hinge_bending_terms_warp,
    simplex_inversion_terms_warp,
    simplex_measure_components_warp,
    simplex_stvk_terms_warp,
)

_Implementation = Literal["torch", "warp"] | None


def _validate_finite_real(
    value: float,
    name: str,
    dtype: torch.dtype,
) -> float:
    """Validate a finite Python scalar without dropping compiled checks."""

    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(
            f"{name} must be a finite Python real scalar, got {type(value).__name__}"
        )

    maximum = torch.finfo(dtype).max
    if torch.compiler.is_compiling():
        # Dynamo may generalize a call-time float to SymFloat. Statically known
        # invalid literals still fail while symbolic values retain runtime
        # guards in the compiled graph.
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        if statically_known_true(value != value) or statically_known_true(
            abs(value) == math.inf
        ):
            raise ValueError(f"{name} must be finite")
        if statically_known_true(abs(value) > maximum):
            raise ValueError(f"{name} must be finite in the coordinate dtype")
        torch._check(
            abs(value) <= maximum,
            lambda: "deformation-energy scalar must be finite in the coordinate dtype",
        )
        return value

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    if abs(numeric_value) > maximum:
        raise ValueError(f"{name} must be finite in the coordinate dtype")
    return numeric_value


def _validate_nonnegative_real(
    value: float,
    name: str,
    dtype: torch.dtype,
) -> float:
    """Validate a finite nonnegative Python scalar."""

    value = _validate_finite_real(value, name, dtype)
    if torch.compiler.is_compiling():
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        if statically_known_true(value < 0):
            raise ValueError(f"{name} must be nonnegative")
        torch._check(
            value >= 0,
            lambda: "deformation-energy scalar must be nonnegative",
        )
    elif value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _validate_positive_real(
    value: float,
    name: str,
    dtype: torch.dtype,
) -> float:
    """Validate a finite strictly positive Python scalar."""

    value = _validate_finite_real(value, name, dtype)
    minimum = torch.finfo(dtype).tiny * torch.finfo(dtype).eps
    if torch.compiler.is_compiling():
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        if statically_known_true(value <= 0):
            raise ValueError(f"{name} must be strictly positive")
        if statically_known_true(value < minimum):
            raise ValueError(
                f"{name} must remain strictly positive in the coordinate dtype"
            )
        torch._check(
            value >= minimum,
            lambda: (
                "deformation-energy scalar must remain strictly positive in the "
                "coordinate dtype"
            ),
        )
    elif value <= 0:
        raise ValueError(f"{name} must be strictly positive")
    elif value < minimum:
        raise ValueError(
            f"{name} must remain strictly positive in the coordinate dtype"
        )
    return value


def _validate_lame_parameters(
    lame_lambda: float,
    shear_modulus: float,
    simplex_dimension: int,
    dtype: torch.dtype,
) -> tuple[float, float]:
    """Validate Lamé parameters for a nonnegative isotropic StVK density."""

    lame_lambda = _validate_finite_real(lame_lambda, "lame_lambda", dtype)
    shear_modulus = _validate_nonnegative_real(shear_modulus, "shear_modulus", dtype)
    # Check half of the conventional stability expression. This equivalent
    # form keeps every product representable for finite input parameters.
    stability = 0.5 * lame_lambda + shear_modulus / simplex_dimension
    stability = _validate_finite_real(
        stability,
        "0.5 * lame_lambda + shear_modulus / simplex_dimension",
        dtype,
    )
    if torch.compiler.is_compiling():
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        if statically_known_true(stability < 0):
            raise ValueError(
                "lame_lambda + 2 * shear_modulus / simplex_dimension must be "
                "nonnegative"
            )
        torch._check(
            stability >= 0,
            lambda: (
                "lame_lambda + 2 * shear_modulus / simplex_dimension "
                "must be nonnegative"
            ),
        )
    elif stability < 0:
        raise ValueError(
            "lame_lambda + 2 * shear_modulus / simplex_dimension must be nonnegative"
        )
    return lame_lambda, shear_modulus


def _local_measure_terms(
    ratio: Float[torch.Tensor, "..."],
    reference_measure: Float[torch.Tensor, "..."],
    target_ratio: float,
) -> Float[torch.Tensor, "..."]:
    """Compose local measure-preservation terms from backend components."""

    return 0.5 * reference_measure * (ratio - target_ratio).square()


def _total_measure_terms(
    ratio: Float[torch.Tensor, "..."],
    reference_measure: Float[torch.Tensor, "..."],
    target_ratio: float,
) -> Float[torch.Tensor, "..."]:
    """Compose one total-measure term for each aligned batch item."""

    reference_total = reference_measure.sum(dim=-1)
    current_total = (reference_measure * ratio).sum(dim=-1)
    total_ratio = current_total / reference_total
    terms = 0.5 * reference_total * (total_ratio - target_ratio).square()
    valid = (reference_total > 0.0) & torch.isfinite(reference_total)
    return torch.where(valid, terms, torch.full_like(terms, torch.nan))


def _closed_volume_terms(
    current_contributions: Float[torch.Tensor, "..."],
    reference_contributions: Float[torch.Tensor, "..."],
    target_ratio: float,
) -> Float[torch.Tensor, "..."]:
    """Compose one enclosed-volume term for each aligned batch item."""

    current_volume = current_contributions.sum(dim=-1)
    reference_volume = reference_contributions.sum(dim=-1)
    reference_scale = reference_volume.abs()
    ratio = current_volume / reference_volume
    terms = 0.5 * reference_scale * (ratio - target_ratio).square()
    valid = (reference_scale > 0.0) & torch.isfinite(reference_volume)
    return torch.where(valid, terms, torch.full_like(terms, torch.nan))


class _DeformationEnergySpec(FunctionSpec):
    """Shared dispatch, benchmark, and comparison behavior for energies."""

    _COMPARE_ATOL = 3.0e-5
    _COMPARE_RTOL = 3.0e-5
    _COMPARE_BACKWARD_ATOL = 5.0e-4
    _COMPARE_BACKWARD_RTOL = 5.0e-4

    @classmethod
    def _select_implementation(
        cls,
        points: object,
        implementation: _Implementation,
    ) -> Literal["torch", "warp"] | None:
        """Select Warp for supported CUDA coordinates and Torch otherwise."""

        if implementation == "warp" and isinstance(points, torch.Tensor):
            if points.ndim in (2, 3) and points.shape[-1] > 3:
                raise ValueError(
                    "the Warp deformation-energy backend supports coordinate "
                    f"dimensions D <= 3, got D={points.shape[-1]}; use "
                    "implementation='torch'"
                )
        if implementation is not None:
            return implementation

        impls = cls._get_impls()
        warp_impl = impls.get("warp")
        warp_dimension = (
            isinstance(points, torch.Tensor)
            and points.ndim in (2, 3)
            and points.shape[-1] <= 3
        )
        if isinstance(points, torch.Tensor) and points.is_cuda and warp_dimension:
            if warp_impl is not None and warp_impl.available:
                return "warp"
            cls._warn_fallback(warp_impl, impls["torch"])
        return "torch"

    @classmethod
    def compare_forward(
        cls,
        output: Float[torch.Tensor, "..."],
        reference: Float[torch.Tensor, "..."],
    ) -> None:
        """Compare deformation-energy outputs across backends."""

        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_ATOL,
            rtol=cls._COMPARE_RTOL,
            equal_nan=True,
        )

    @classmethod
    def compare_backward(
        cls,
        output: Float[torch.Tensor, "..."],
        reference: Float[torch.Tensor, "..."],
    ) -> None:
        """Compare deformation-energy gradients across backends."""

        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_BACKWARD_ATOL,
            rtol=cls._COMPARE_BACKWARD_RTOL,
            equal_nan=True,
        )


def _make_simplex_benchmark_tensors(
    device: torch.device,
    dtype: torch.dtype,
    num_simplices: int,
    batch_size: int,
    *,
    requires_grad: bool,
) -> tuple[
    Float[torch.Tensor, "*batch num_points 3"],
    Float[torch.Tensor, "*batch num_points 3"],
    Int[torch.Tensor, "num_simplices 4"],
]:
    """Build deterministic, nondegenerate tetrahedra for benchmarks."""

    template = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    offsets = torch.arange(num_simplices, dtype=dtype, device=device)[:, None, None]
    offset_axis = torch.tensor([2.0, 0.0, 0.0], dtype=dtype, device=device)
    reference = template[None] + offsets * offset_axis
    reference = reference.reshape(-1, 3)
    cells = torch.arange(4 * num_simplices, dtype=torch.int64, device=device).reshape(
        num_simplices, 4
    )
    deformation = torch.tensor(
        [[1.04, 0.03, 0.00], [0.01, 0.97, 0.02], [0.00, 0.01, 1.02]],
        dtype=dtype,
        device=device,
    )
    points = reference @ deformation.T
    if batch_size > 1:
        reference = reference.unsqueeze(0).expand(batch_size, -1, -1).clone()
        scales = torch.linspace(0.98, 1.02, batch_size, dtype=dtype, device=device)[
            :, None, None
        ]
        points = points.unsqueeze(0) * scales
    points.requires_grad_(requires_grad)
    reference.requires_grad_(requires_grad)
    return points, reference, cells


def _make_hinge_benchmark_tensors(
    device: torch.device,
    dtype: torch.dtype,
    num_hinges: int,
    batch_size: int,
    *,
    requires_grad: bool,
) -> tuple[
    Float[torch.Tensor, "*batch num_points 3"],
    Float[torch.Tensor, "*batch num_points 3"],
    Int[torch.Tensor, "num_hinges 4"],
]:
    """Build deterministic independent triangle hinges for benchmarks."""

    template = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    offsets = torch.arange(num_hinges, dtype=dtype, device=device)[:, None, None]
    offset_axis = torch.tensor([2.0, 0.0, 0.0], dtype=dtype, device=device)
    reference = (template[None] + offsets * offset_axis).reshape(-1, 3)
    points = reference.clone()
    points.reshape(num_hinges, 4, 3)[:, 3, 2] = 0.15
    hinges = torch.arange(4 * num_hinges, dtype=torch.int64, device=device).reshape(
        num_hinges, 4
    )
    if batch_size > 1:
        reference = reference.unsqueeze(0).expand(batch_size, -1, -1).clone()
        points = points.unsqueeze(0).expand(batch_size, -1, -1).clone()
        points[:, :, 2] *= torch.linspace(
            0.8, 1.2, batch_size, dtype=dtype, device=device
        )[:, None]
    points.requires_grad_(requires_grad)
    reference.requires_grad_(requires_grad)
    return points, reference, hinges


def _make_closed_surface_benchmark_tensors(
    device: torch.device,
    dtype: torch.dtype,
    num_faces: int,
    batch_size: int,
    *,
    requires_grad: bool,
) -> tuple[
    Float[torch.Tensor, "*batch num_points 3"],
    Float[torch.Tensor, "*batch num_points 3"],
    Int[torch.Tensor, "num_faces 3"],
]:
    """Build one deterministic closed bipyramid surface for benchmarks."""

    if num_faces < 6 or num_faces % 2 != 0:
        raise ValueError("a bipyramid benchmark requires an even num_faces >= 6")
    num_ring = num_faces // 2
    angles = (
        2.0 * math.pi * torch.arange(num_ring, dtype=dtype, device=device) / num_ring
    )
    ring = torch.stack(
        (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=-1
    )
    poles = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=dtype, device=device
    )
    reference = torch.cat((ring, poles), dim=0)
    points = reference * torch.tensor([1.03, 0.98, 1.01], dtype=dtype, device=device)
    current = torch.arange(num_ring, dtype=torch.int64, device=device)
    following = (current + 1) % num_ring
    top = torch.full_like(current, num_ring)
    bottom = torch.full_like(current, num_ring + 1)
    triangles = torch.cat(
        (
            torch.stack((top, current, following), dim=-1),
            torch.stack((bottom, following, current), dim=-1),
        ),
        dim=0,
    )
    if batch_size > 1:
        reference = reference.unsqueeze(0).expand(batch_size, -1, -1).clone()
        points = points.unsqueeze(0).expand(batch_size, -1, -1).clone()
        points *= torch.linspace(0.99, 1.01, batch_size, dtype=dtype, device=device)[
            :, None, None
        ]
    points.requires_grad_(requires_grad)
    reference.requires_grad_(requires_grad)
    return points, reference, triangles


class _SimplexEnergySpec(_DeformationEnergySpec):
    """Shared benchmark generators for simplex energies."""

    _BENCHMARK_KWARGS: dict[str, object] = {}

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative simplex-energy forward cases."""

        device = torch.device(device)
        for label, dtype, count, batch_size in (
            ("float64-c64-tetrahedra", torch.float64, 64, 1),
            ("batched-float32-b2-c2048-tetrahedra", torch.float32, 2048, 2),
        ):
            points, reference, cells = _make_simplex_benchmark_tensors(
                device, dtype, count, batch_size, requires_grad=False
            )
            yield label, (points, reference, cells), dict(cls._BENCHMARK_KWARGS)

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative differentiable simplex-energy cases."""

        device = torch.device(device)
        points, reference, cells = _make_simplex_benchmark_tensors(
            device, torch.float32, 1024, 1, requires_grad=True
        )
        yield (
            "float32-c1024-current-and-reference-gradients",
            (points, reference, cells),
            dict(cls._BENCHMARK_KWARGS),
        )


class SimplexStrainEnergy(_SimplexEnergySpec):
    r"""Evaluate reference-integrated St. Venant--Kirchhoff simplex energy.

    For each edge, triangle, or tetrahedron, an orthonormalized reference edge
    frame is used to express the right Cauchy--Green tensor :math:`C`
    intrinsically. With Green--Lagrange strain
    :math:`\varepsilon=(C-I)/2`, the per-simplex energy is

    .. math::

       V_0\left[\mu\lVert\varepsilon\rVert_F^2
       + \frac{\lambda}{2}\operatorname{tr}(\varepsilon)^2\right].

    The formulation supports one-, two-, and three-dimensional simplices
    embedded in any coordinate dimension supported by the selected backend.
    It is invariant to rigid transformations and reflections. Add
    :func:`simplex_inversion_energy` when element orientation must be preserved.

    Parameters
    ----------
    points : torch.Tensor
        Current coordinates with shape ``(N, D)`` or ``(B, N, D)``.
    reference_points : torch.Tensor
        Rest coordinates with exactly the same shape, dtype, and device as
        ``points``. Batches are aligned and are not implicitly broadcast.
    cells : torch.Tensor
        Shared simplex connectivity with shape ``(M, m + 1)``, where
        ``m`` is 1, 2, or 3 and ``D >= m``. Values must be distinct, valid
        in-range point indices. Int32 and int64 are accepted.
    lame_lambda : float, optional
        First Lamé parameter. It may be negative for a stable auxetic material,
        but ``lame_lambda + 2 * shear_modulus / m`` must be nonnegative.
        Default is ``1.0``.
    shear_modulus : float, optional
        Nonnegative shear modulus. Default is ``1.0``.
    reduction : {"none", "sum", "mean"}, optional
        ``"none"`` returns one term per simplex. ``"sum"`` returns the
        physically integrated total. ``"mean"`` averages all batch/simplex
        terms. Empty inputs have zero sum and mean. Default is ``"sum"``.
    implementation : {"torch", "warp"} or None, optional
        Explicit backend. ``None`` selects Torch on CPU and Warp on CUDA when
        Warp is available and ``D <= 3``. Higher coordinate dimensions use
        Torch.

    Returns
    -------
    torch.Tensor
        Per-simplex terms or a scalar according to ``reduction``.

    Notes
    -----
    This function returns an energy term for use in an objective. It does not
    run an optimizer or impose a hard constraint. Degenerate reference cells
    produce NaN so invalid rest geometry is not silently regularized. Both
    coordinate tensors are differentiable. Connectivity is discrete and is
    not differentiable. Torch supports higher-order derivatives. The Warp
    backend's contract is first-order differentiation.
    """

    _BENCHMARK_KWARGS = {"lame_lambda": 0.8, "shear_modulus": 1.2}

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells simplex_vertices"],
        *,
        lame_lambda: float = 1.0,
        shear_modulus: float = 1.0,
        reduction: _Reduction = "sum",
    ) -> Float[torch.Tensor, "..."]:
        """Evaluate StVK terms with Warp."""

        points, reference_points, cells, was_unbatched = normalize_simplex_inputs(
            points, reference_points, cells
        )
        lame_lambda, shear_modulus = _validate_lame_parameters(
            lame_lambda,
            shear_modulus,
            cells.shape[1] - 1,
            points.dtype,
        )
        terms = simplex_stvk_terms_warp(
            points, reference_points, cells, lame_lambda, shear_modulus
        )
        return reduce_energy_terms(terms, reduction, was_unbatched)

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells simplex_vertices"],
        *,
        lame_lambda: float = 1.0,
        shear_modulus: float = 1.0,
        reduction: _Reduction = "sum",
    ) -> Float[torch.Tensor, "..."]:
        """Evaluate StVK terms with Torch."""

        points, reference_points, cells, was_unbatched = normalize_simplex_inputs(
            points, reference_points, cells
        )
        lame_lambda, shear_modulus = _validate_lame_parameters(
            lame_lambda,
            shear_modulus,
            cells.shape[1] - 1,
            points.dtype,
        )
        terms = simplex_stvk_terms_torch(
            points, reference_points, cells, lame_lambda, shear_modulus
        )
        return reduce_energy_terms(terms, reduction, was_unbatched)

    @classmethod
    def dispatch(
        cls,
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells simplex_vertices"],
        *,
        lame_lambda: float = 1.0,
        shear_modulus: float = 1.0,
        reduction: _Reduction = "sum",
        implementation: _Implementation = None,
    ) -> Float[torch.Tensor, "..."]:
        """Dispatch to a device-appropriate deformation-energy backend."""

        implementation = cls._select_implementation(points, implementation)
        return super().dispatch(
            points,
            reference_points,
            cells,
            lame_lambda=lame_lambda,
            shear_modulus=shear_modulus,
            reduction=reduction,
            implementation=implementation,
        )


class SimplexMeasureEnergy(_SimplexEnergySpec):
    r"""Penalize local simplex measure changes relative to rest geometry.

    Each term is :math:`\tfrac12 V_0(q-q_*)^2`, where ``target_ratio`` is
    :math:`q_*`. For a full-dimensional simplex, :math:`q=\det(E)/\det(E_0)`
    is signed, so a reflected cell does not falsely satisfy the target. For an
    embedded simplex, :math:`q=\sqrt{\det(G)/\det(G_0)}` is the unsigned
    intrinsic length, area, or volume ratio.

    Use :func:`total_measure_energy` when local redistribution is allowed and
    only the aggregate measure should remain near its target.

    Parameters
    ----------
    points, reference_points : torch.Tensor
        Current and rest coordinates with identical ``(N, D)`` or
        ``(B, N, D)`` shape, dtype, and device.
    cells : torch.Tensor
        Shared ``(M, m + 1)`` connectivity for edges, triangles, or
        tetrahedra. Values must be distinct, valid in-range indices. Int32 and
        int64 are accepted.
    target_ratio : float, optional
        Strictly positive target measure relative to the reference. Default is
        ``1.0``.
    reduction : {"none", "sum", "mean"}, optional
        ``"none"`` returns ``(M,)`` or ``(B, M)``. ``"sum"`` and ``"mean"``
        reduce every term to a scalar. Empty inputs have zero sum and mean.
        Default is ``"sum"``.
    implementation : {"torch", "warp"} or None, optional
        Explicit backend. ``None`` selects Torch on CPU and, when available,
        Warp on CUDA for coordinate dimensions up to three. Higher-dimensional
        CUDA inputs use Torch. Explicitly selecting Warp for them is rejected.

    Returns
    -------
    torch.Tensor
        Per-simplex terms or a scalar according to ``reduction``.

    Notes
    -----
    This is a penalty energy, not an exact constraint. Reference-degenerate
    cells produce NaN. Both current and reference coordinates are
    differentiable. Connectivity is not. At exact collapse, unsigned embedded
    measure uses a zero current-coordinate subgradient, so this term cannot by
    itself reopen an already collapsed embedded cell. Warp provides first-order
    gradients.
    """

    _BENCHMARK_KWARGS = {"target_ratio": 1.0}

    @staticmethod
    def _forward(
        backend: Callable[
            [
                Float[torch.Tensor, "batch num_points num_dims"],
                Float[torch.Tensor, "batch num_points num_dims"],
                Int[torch.Tensor, "num_cells simplex_vertices"],
            ],
            tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]],
        ],
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells simplex_vertices"],
        target_ratio: float,
        reduction: _Reduction,
    ) -> Float[torch.Tensor, "..."]:
        points, reference_points, cells, was_unbatched = normalize_simplex_inputs(
            points, reference_points, cells
        )
        target_ratio = _validate_positive_real(
            target_ratio, "target_ratio", points.dtype
        )
        ratio, reference_measure = backend(points, reference_points, cells)
        terms = _local_measure_terms(ratio, reference_measure, target_ratio)
        return reduce_energy_terms(terms, reduction, was_unbatched)

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells simplex_vertices"],
        *,
        target_ratio: float = 1.0,
        reduction: _Reduction = "sum",
    ) -> Float[torch.Tensor, "..."]:
        """Evaluate local measure terms with Warp."""

        return SimplexMeasureEnergy._forward(
            simplex_measure_components_warp,
            points,
            reference_points,
            cells,
            target_ratio,
            reduction,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells simplex_vertices"],
        *,
        target_ratio: float = 1.0,
        reduction: _Reduction = "sum",
    ) -> Float[torch.Tensor, "..."]:
        """Evaluate local measure terms with Torch."""

        return SimplexMeasureEnergy._forward(
            simplex_measure_components_torch,
            points,
            reference_points,
            cells,
            target_ratio,
            reduction,
        )

    @classmethod
    def dispatch(
        cls,
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells simplex_vertices"],
        *,
        target_ratio: float = 1.0,
        reduction: _Reduction = "sum",
        implementation: _Implementation = None,
    ) -> Float[torch.Tensor, "..."]:
        """Dispatch to a device-appropriate deformation-energy backend."""

        implementation = cls._select_implementation(points, implementation)
        return super().dispatch(
            points,
            reference_points,
            cells,
            target_ratio=target_ratio,
            reduction=reduction,
            implementation=implementation,
        )


class TotalMeasureEnergy(_SimplexEnergySpec):
    r"""Penalize change in the total simplex measure.

    The energy for each batch item is
    :math:`\tfrac12 V_{0,\mathrm{tot}}(V/V_{0,\mathrm{tot}}-q_*)^2`.
    Unlike :func:`simplex_measure_energy`, local cells may expand and contract
    while their aggregate measure remains at the target. Full-dimensional cell
    contributions retain their orientation relative to the corresponding rest
    cell. Embedded-simplex contributions are unsigned.

    Parameters
    ----------
    points, reference_points : torch.Tensor
        Current and rest coordinates with identical ``(N, D)`` or
        ``(B, N, D)`` shape, dtype, and device.
    cells : torch.Tensor
        Nonempty shared ``(M, m + 1)`` simplex connectivity. Values must be
        distinct, valid in-range indices. Int32 and int64 are accepted.
    target_ratio : float, optional
        Strictly positive target total-measure ratio. Default is ``1.0``.
    reduction : {"none", "sum", "mean"}, optional
        ``"none"`` returns a scalar for unbatched input or ``(B,)`` for batched
        input. ``"sum"`` and ``"mean"`` reduce across the batch. Default is
        ``"sum"``.
    implementation : {"torch", "warp"} or None, optional
        Explicit backend. ``None`` selects Torch on CPU and, when available,
        Warp on CUDA for coordinate dimensions up to three. Higher-dimensional
        CUDA inputs use Torch. Explicitly selecting Warp for them is rejected.

    Returns
    -------
    torch.Tensor
        One term per batch item or a scalar according to ``reduction``.

    Notes
    -----
    This is a soft aggregate constraint. It does not prevent individual cells
    from collapsing or inverting. Combine it with strain and inversion terms
    when local validity matters. Empty connectivity is rejected, and a zero or
    invalid total reference measure produces NaN. At exact collapse, unsigned
    embedded measure uses a zero current-coordinate subgradient.
    """

    _BENCHMARK_KWARGS = {"target_ratio": 1.0}

    @staticmethod
    def _forward(
        backend: Callable[
            [
                Float[torch.Tensor, "batch num_points num_dims"],
                Float[torch.Tensor, "batch num_points num_dims"],
                Int[torch.Tensor, "num_cells simplex_vertices"],
            ],
            tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]],
        ],
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells simplex_vertices"],
        target_ratio: float,
        reduction: _Reduction,
    ) -> Float[torch.Tensor, "..."]:
        points, reference_points, cells, was_unbatched = normalize_simplex_inputs(
            points, reference_points, cells
        )
        if cells.shape[0] == 0:
            raise ValueError("total measure energy requires at least one simplex")
        target_ratio = _validate_positive_real(
            target_ratio, "target_ratio", points.dtype
        )
        ratio, reference_measure = backend(points, reference_points, cells)
        terms = _total_measure_terms(ratio, reference_measure, target_ratio)
        return reduce_energy_terms(terms, reduction, was_unbatched)

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells simplex_vertices"],
        *,
        target_ratio: float = 1.0,
        reduction: _Reduction = "sum",
    ) -> Float[torch.Tensor, "..."]:
        """Evaluate total-measure terms with Warp."""

        return TotalMeasureEnergy._forward(
            simplex_measure_components_warp,
            points,
            reference_points,
            cells,
            target_ratio,
            reduction,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells simplex_vertices"],
        *,
        target_ratio: float = 1.0,
        reduction: _Reduction = "sum",
    ) -> Float[torch.Tensor, "..."]:
        """Evaluate total-measure terms with Torch."""

        return TotalMeasureEnergy._forward(
            simplex_measure_components_torch,
            points,
            reference_points,
            cells,
            target_ratio,
            reduction,
        )

    @classmethod
    def dispatch(
        cls,
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells simplex_vertices"],
        *,
        target_ratio: float = 1.0,
        reduction: _Reduction = "sum",
        implementation: _Implementation = None,
    ) -> Float[torch.Tensor, "..."]:
        """Dispatch to a device-appropriate deformation-energy backend."""

        implementation = cls._select_implementation(points, implementation)
        return super().dispatch(
            points,
            reference_points,
            cells,
            target_ratio=target_ratio,
            reduction=reduction,
            implementation=implementation,
        )


class SimplexInversionEnergy(_SimplexEnergySpec):
    r"""Penalize full-dimensional cells below a signed Jacobian threshold.

    Each term is
    :math:`\tfrac12 V_0\max(0,J_{\min}-J)^2`, where
    :math:`J=\det(E)/\det(E_0)`. It is zero above ``minimum_jacobian`` and
    increases quadratically through collapse and inversion.

    Parameters
    ----------
    points, reference_points : torch.Tensor
        Current and rest coordinates with identical ``(N, D)`` or
        ``(B, N, D)`` shape, dtype, and device.
    cells : torch.Tensor
        Shared full-dimensional simplex connectivity: edges in 1D, triangles
        in 2D, or tetrahedra in 3D. Values must be distinct, valid in-range
        indices. Int32 and int64 are accepted.
    minimum_jacobian : float, optional
        Nonnegative signed-Jacobian threshold. Default is ``0.1``.
    reduction : {"none", "sum", "mean"}, optional
        ``"none"`` returns one term per cell. ``"sum"`` and ``"mean"`` reduce
        all terms to a scalar. Empty inputs have zero sum and mean. Default is
        ``"sum"``.
    implementation : {"torch", "warp"} or None, optional
        Explicit backend. ``None`` selects Torch on CPU and Warp on CUDA.

    Returns
    -------
    torch.Tensor
        Per-cell penalties or a scalar according to ``reduction``.

    Notes
    -----
    The squared hinge is continuously differentiable, but its second derivative
    is discontinuous at the threshold. This energy discourages inversion but is
    not an infinite barrier and cannot guarantee validity under a finite
    penalty. Embedded simplices are outside this determinant-based term's
    domain and are rejected.
    """

    _BENCHMARK_KWARGS = {"minimum_jacobian": 1.2}

    @staticmethod
    def _forward(
        backend: Callable[
            [
                Float[torch.Tensor, "batch num_points num_dims"],
                Float[torch.Tensor, "batch num_points num_dims"],
                Int[torch.Tensor, "num_cells simplex_vertices"],
                float,
            ],
            Float[torch.Tensor, "..."],
        ],
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells simplex_vertices"],
        minimum_jacobian: float,
        reduction: _Reduction,
    ) -> Float[torch.Tensor, "..."]:
        points, reference_points, cells, was_unbatched = normalize_simplex_inputs(
            points, reference_points, cells
        )
        simplex_dimension = cells.shape[1] - 1
        if simplex_dimension != points.shape[-1]:
            raise ValueError(
                "simplex inversion energy requires full-dimensional simplices, "
                f"got simplex dimension {simplex_dimension} in D={points.shape[-1]}"
            )
        minimum_jacobian = _validate_nonnegative_real(
            minimum_jacobian, "minimum_jacobian", points.dtype
        )
        terms = backend(
            points,
            reference_points,
            cells,
            minimum_jacobian,
        )
        return reduce_energy_terms(terms, reduction, was_unbatched)

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells simplex_vertices"],
        *,
        minimum_jacobian: float = 0.1,
        reduction: _Reduction = "sum",
    ) -> Float[torch.Tensor, "..."]:
        """Evaluate inversion terms with Warp."""

        return SimplexInversionEnergy._forward(
            simplex_inversion_terms_warp,
            points,
            reference_points,
            cells,
            minimum_jacobian,
            reduction,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells simplex_vertices"],
        *,
        minimum_jacobian: float = 0.1,
        reduction: _Reduction = "sum",
    ) -> Float[torch.Tensor, "..."]:
        """Evaluate inversion terms with Torch."""

        return SimplexInversionEnergy._forward(
            simplex_inversion_terms_torch,
            points,
            reference_points,
            cells,
            minimum_jacobian,
            reduction,
        )

    @classmethod
    def dispatch(
        cls,
        points: Float[torch.Tensor, "*batch num_points num_dims"],
        reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
        cells: Int[torch.Tensor, "num_cells simplex_vertices"],
        *,
        minimum_jacobian: float = 0.1,
        reduction: _Reduction = "sum",
        implementation: _Implementation = None,
    ) -> Float[torch.Tensor, "..."]:
        """Dispatch to a device-appropriate deformation-energy backend."""

        implementation = cls._select_implementation(points, implementation)
        return super().dispatch(
            points,
            reference_points,
            cells,
            minimum_jacobian=minimum_jacobian,
            reduction=reduction,
            implementation=implementation,
        )


class SurfaceBendingEnergy(_DeformationEnergySpec):
    r"""Evaluate reference-relative discrete bending over triangle hinges.

    A hinge row ``(i, j, k, l)`` denotes the consistently oriented adjacent
    faces ``(i, j, k)`` and ``(j, i, l)``. If :math:`\theta` and
    :math:`\theta_0` are their signed current and reference dihedral angles,
    each term is

    .. math::

       \frac12\frac{\ell_0^2}{A_{0,L}+A_{0,R}}
       \operatorname{wrap}(\theta-\theta_0)^2.

    Parameters
    ----------
    points, reference_points : torch.Tensor
        Current and rest coordinates with identical ``(N, 3)`` or
        ``(B, N, 3)`` shape, dtype, and device.
    hinges : torch.Tensor
        Shared oriented hinge connectivity with shape ``(H, 4)``. Values must
        be distinct, valid in-range indices. Int32 and int64 are accepted.
        Construct hinges from unique triangles with at most two incident
        triangles per edge, and omit boundary edges. Build them once from fixed
        topology and reuse them during optimization. The
        :func:`physicsnemo.mesh.deformation.surface_bending_energy` wrapper
        validates these conditions, constructs the hinges, and caches them.
    reduction : {"none", "sum", "mean"}, optional
        ``"none"`` returns ``(H,)`` or ``(B, H)``. ``"sum"`` and ``"mean"``
        return a scalar. A surface without interior hinges has zero sum and
        mean. Default is ``"sum"``.
    implementation : {"torch", "warp"} or None, optional
        Explicit backend. ``None`` selects Torch on CPU and Warp on CUDA.

    Returns
    -------
    torch.Tensor
        Per-hinge terms or a scalar according to ``reduction``.

    Notes
    -----
    This is a geometric thin-shell regularizer, not a calibrated physical shell
    model: material thickness and constitutive scaling are not included.
    Degenerate current or reference hinge triangles produce NaN. Dihedral
    wrapping is continuous except at the equivalent-angle branch cut. Both
    coordinate tensors are differentiable. The discrete hinge construction is
    not. Warp provides first-order gradients.
    """

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative bending-energy forward cases."""

        device = torch.device(device)
        for label, dtype, count, batch_size in (
            ("float64-h64", torch.float64, 64, 1),
            ("batched-float32-b2-h2048", torch.float32, 2048, 2),
        ):
            inputs = _make_hinge_benchmark_tensors(
                device, dtype, count, batch_size, requires_grad=False
            )
            yield label, inputs, {}

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield a differentiable bending-energy case."""

        device = torch.device(device)
        inputs = _make_hinge_benchmark_tensors(
            device, torch.float32, 1024, 1, requires_grad=True
        )
        yield "float32-h1024-current-and-reference-gradients", inputs, {}

    @staticmethod
    def _forward(
        backend: Callable[
            [
                Float[torch.Tensor, "batch num_points 3"],
                Float[torch.Tensor, "batch num_points 3"],
                Int[torch.Tensor, "num_hinges 4"],
            ],
            Float[torch.Tensor, "..."],
        ],
        points: Float[torch.Tensor, "*batch num_points 3"],
        reference_points: Float[torch.Tensor, "*batch num_points 3"],
        hinges: Int[torch.Tensor, "num_hinges 4"],
        reduction: _Reduction,
    ) -> Float[torch.Tensor, "..."]:
        points, reference_points, hinges, was_unbatched = normalize_hinge_inputs(
            points, reference_points, hinges
        )
        terms = backend(points, reference_points, hinges)
        return reduce_energy_terms(terms, reduction, was_unbatched)

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: Float[torch.Tensor, "*batch num_points 3"],
        reference_points: Float[torch.Tensor, "*batch num_points 3"],
        hinges: Int[torch.Tensor, "num_hinges 4"],
        *,
        reduction: _Reduction = "sum",
    ) -> Float[torch.Tensor, "..."]:
        """Evaluate hinge bending with Warp."""

        return SurfaceBendingEnergy._forward(
            hinge_bending_terms_warp,
            points,
            reference_points,
            hinges,
            reduction,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: Float[torch.Tensor, "*batch num_points 3"],
        reference_points: Float[torch.Tensor, "*batch num_points 3"],
        hinges: Int[torch.Tensor, "num_hinges 4"],
        *,
        reduction: _Reduction = "sum",
    ) -> Float[torch.Tensor, "..."]:
        """Evaluate hinge bending with Torch."""

        return SurfaceBendingEnergy._forward(
            hinge_bending_terms_torch,
            points,
            reference_points,
            hinges,
            reduction,
        )

    @classmethod
    def dispatch(
        cls,
        points: Float[torch.Tensor, "*batch num_points 3"],
        reference_points: Float[torch.Tensor, "*batch num_points 3"],
        hinges: Int[torch.Tensor, "num_hinges 4"],
        *,
        reduction: _Reduction = "sum",
        implementation: _Implementation = None,
    ) -> Float[torch.Tensor, "..."]:
        """Dispatch to a device-appropriate deformation-energy backend."""

        implementation = cls._select_implementation(points, implementation)
        return super().dispatch(
            points,
            reference_points,
            hinges,
            reduction=reduction,
            implementation=implementation,
        )


class ClosedSurfaceVolumeEnergy(_DeformationEnergySpec):
    r"""Penalize enclosed-volume change of an oriented closed triangle surface.

    With :math:`y_i=x_i-o` for one common per-batch origin, the signed volume
    is assembled from face contributions
    :math:`v_f=y_i\cdot(y_j\times y_k)/6`. The closed-surface sum is independent
    of :math:`o`. For each batch item, the energy is

    .. math::

       \frac12 |V_0|\left(V/V_0-q_*\right)^2.

    Parameters
    ----------
    points, reference_points : torch.Tensor
        Current and rest coordinates with identical ``(N, 3)`` or
        ``(B, N, 3)`` shape, dtype, and device.
    triangles : torch.Tensor
        Nonempty shared ``(F, 3)`` triangle connectivity for one edge-connected,
        edge-closed, consistently oriented component. Every edge must have two
        oppositely directed incident triangles. Values must be distinct, valid
        in-range indices. Int32 and int64 are accepted.
    target_ratio : float, optional
        Strictly positive target enclosed-volume ratio. Default is ``1.0``.
    reduction : {"none", "sum", "mean"}, optional
        ``"none"`` returns a scalar for unbatched input or ``(B,)`` for batched
        input. ``"sum"`` and ``"mean"`` reduce across the batch. Default is
        ``"sum"``.
    implementation : {"torch", "warp"} or None, optional
        Explicit backend. ``None`` selects Torch on CPU and Warp on CUDA.

    Returns
    -------
    torch.Tensor
        One volume term per batch item or a scalar according to ``reduction``.

    Notes
    -----
    This function supplies a penalty term, not an exact volume constraint.
    It assumes the stated connectivity, edge-closure, and orientation contract.
    Checking those discrete properties belongs outside an iterative optimization
    loop. Evaluate disconnected components separately. An empty triangle set is
    rejected. Zero or nonfinite reference volume produces NaN.
    Vertex-manifoldness and self-intersection are not checked, and local face
    inversion can cancel in the signed total. Combine this term with local
    validity energies when needed.
    """

    _BENCHMARK_KWARGS = {"target_ratio": 1.0}

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative enclosed-volume forward cases."""

        device = torch.device(device)
        for label, dtype, count, batch_size in (
            ("float64-f256", torch.float64, 256, 1),
            ("batched-float32-b2-f8192", torch.float32, 8192, 2),
        ):
            inputs = _make_closed_surface_benchmark_tensors(
                device, dtype, count, batch_size, requires_grad=False
            )
            yield label, inputs, dict(cls._BENCHMARK_KWARGS)

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield a differentiable enclosed-volume case."""

        device = torch.device(device)
        inputs = _make_closed_surface_benchmark_tensors(
            device, torch.float32, 4096, 1, requires_grad=True
        )
        yield (
            "float32-f4096-current-and-reference-gradients",
            inputs,
            dict(cls._BENCHMARK_KWARGS),
        )

    @staticmethod
    def _forward(
        backend: Callable[
            [
                Float[torch.Tensor, "batch num_points 3"],
                Float[torch.Tensor, "batch num_points 3"],
                Int[torch.Tensor, "num_triangles 3"],
            ],
            tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]],
        ],
        points: Float[torch.Tensor, "*batch num_points 3"],
        reference_points: Float[torch.Tensor, "*batch num_points 3"],
        triangles: Int[torch.Tensor, "num_triangles 3"],
        target_ratio: float,
        reduction: _Reduction,
    ) -> Float[torch.Tensor, "..."]:
        points, reference_points, triangles, was_unbatched = (
            normalize_closed_surface_inputs(points, reference_points, triangles)
        )
        if triangles.shape[0] == 0:
            raise ValueError(
                "closed surface volume energy requires at least one triangle"
            )
        target_ratio = _validate_positive_real(
            target_ratio, "target_ratio", points.dtype
        )
        current, reference = backend(points, reference_points, triangles)
        terms = _closed_volume_terms(current, reference, target_ratio)
        return reduce_energy_terms(terms, reduction, was_unbatched)

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: Float[torch.Tensor, "*batch num_points 3"],
        reference_points: Float[torch.Tensor, "*batch num_points 3"],
        triangles: Int[torch.Tensor, "num_triangles 3"],
        *,
        target_ratio: float = 1.0,
        reduction: _Reduction = "sum",
    ) -> Float[torch.Tensor, "..."]:
        """Evaluate enclosed-volume terms with Warp."""

        return ClosedSurfaceVolumeEnergy._forward(
            closed_surface_volume_contributions_warp,
            points,
            reference_points,
            triangles,
            target_ratio,
            reduction,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: Float[torch.Tensor, "*batch num_points 3"],
        reference_points: Float[torch.Tensor, "*batch num_points 3"],
        triangles: Int[torch.Tensor, "num_triangles 3"],
        *,
        target_ratio: float = 1.0,
        reduction: _Reduction = "sum",
    ) -> Float[torch.Tensor, "..."]:
        """Evaluate enclosed-volume terms with Torch."""

        return ClosedSurfaceVolumeEnergy._forward(
            closed_surface_volume_contributions_torch,
            points,
            reference_points,
            triangles,
            target_ratio,
            reduction,
        )

    @classmethod
    def dispatch(
        cls,
        points: Float[torch.Tensor, "*batch num_points 3"],
        reference_points: Float[torch.Tensor, "*batch num_points 3"],
        triangles: Int[torch.Tensor, "num_triangles 3"],
        *,
        target_ratio: float = 1.0,
        reduction: _Reduction = "sum",
        implementation: _Implementation = None,
    ) -> Float[torch.Tensor, "..."]:
        """Dispatch to a device-appropriate deformation-energy backend."""

        implementation = cls._select_implementation(points, implementation)
        return super().dispatch(
            points,
            reference_points,
            triangles,
            target_ratio=target_ratio,
            reduction=reduction,
            implementation=implementation,
        )


simplex_strain_energy = SimplexStrainEnergy.make_function("simplex_strain_energy")
simplex_measure_energy = SimplexMeasureEnergy.make_function("simplex_measure_energy")
total_measure_energy = TotalMeasureEnergy.make_function("total_measure_energy")
simplex_inversion_energy = SimplexInversionEnergy.make_function(
    "simplex_inversion_energy"
)
surface_bending_energy = SurfaceBendingEnergy.make_function("surface_bending_energy")
closed_surface_volume_energy = ClosedSurfaceVolumeEnergy.make_function(
    "closed_surface_volume_energy"
)


__all__ = [
    "ClosedSurfaceVolumeEnergy",
    "SimplexInversionEnergy",
    "SimplexMeasureEnergy",
    "SimplexStrainEnergy",
    "SurfaceBendingEnergy",
    "TotalMeasureEnergy",
    "closed_surface_volume_energy",
    "simplex_inversion_energy",
    "simplex_measure_energy",
    "simplex_strain_energy",
    "surface_bending_energy",
    "total_measure_energy",
]
