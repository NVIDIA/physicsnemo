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

"""Pure-Torch fitting and evaluation for radial-basis deformation."""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from ._utils import _zero_dependency

# Query evaluation is linear in the number of controls, but forming all
# point/control differences at once can be much larger than the coefficient
# solve. Bound the live forward workspace and checkpoint differentiable chunks
# so their pairwise tensors are recomputed rather than retained for backward.
_RBF_PAIRWISE_TEMPORARY_BYTE_BUDGET = 256 * 1024 * 1024
_RBF_PAIRWISE_LIVE_VALUE_FACTOR = 2


def _thin_plate_spline_from_squared_distance(
    squared_distance: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate :math:`\phi(r)=r^2\log(r)` with :math:`\phi(0)=0`.

    Expressing the kernel as ``0.5 * s * log(s)`` for ``s = r**2`` avoids a
    square root. Replacing the logarithm's argument only on the zero branch
    gives the defined zero derivative at coincident points without evaluating
    ``log(0)`` anywhere in the autograd graph.
    """

    safe_squared_distance = torch.where(
        squared_distance > 0,
        squared_distance,
        torch.ones_like(squared_distance),
    )
    return 0.5 * squared_distance * torch.log(safe_squared_distance)


def _affine_basis(points: torch.Tensor) -> torch.Tensor:
    """Return the affine polynomial basis ``[1, x_1, ..., x_D]``."""

    return torch.cat((torch.ones_like(points[..., :1]), points), dim=-1)


def _checked_solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve a dense system and preserve actionable singularity errors."""

    if (
        matrix.is_cuda
        and not torch.compiler.is_compiling()
        and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(
            "RBF coefficient fitting is not supported during CUDA Graph capture "
            "because singular systems require a host-side error check"
        )
    solution, _ = torch.linalg.solve_ex(matrix, rhs, check_errors=True)
    return solution


def fit_rbf_coefficients_torch(
    control_points: torch.Tensor,
    control_displacements: torch.Tensor,
    *,
    polynomial: bool,
    smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Fit thin-plate-spline coefficients on normalized rank-3 inputs.

    With the affine polynomial tail enabled, the augmented interpolation system
    is

    .. math::

       \begin{bmatrix} K + \lambda I & P \\ P^T & 0 \end{bmatrix}
       \begin{bmatrix} W \\ A \end{bmatrix}
       = \begin{bmatrix} \Delta \\ 0 \end{bmatrix},

    where ``P = [1, control_points]``. ``torch.linalg.solve_ex`` intentionally
    performs the checked dense solve. Singular systems therefore raise instead
    of silently producing invalid coefficients.
    """

    batch_size, num_controls, num_dims = control_points.shape
    num_polynomial_terms = num_dims + 1 if polynomial else 0
    polynomial_coefficients = control_points.new_zeros(
        (batch_size, num_polynomial_terms, num_dims)
    )
    if num_controls == 0:
        return control_displacements, polynomial_coefficients

    difference = control_points.unsqueeze(2) - control_points.unsqueeze(1)
    squared_distance = difference.square().sum(dim=-1)
    kernel_matrix = _thin_plate_spline_from_squared_distance(squared_distance)
    identity = torch.eye(
        num_controls,
        dtype=control_points.dtype,
        device=control_points.device,
    ).expand(batch_size, -1, -1)
    kernel_matrix = kernel_matrix + smoothing * identity

    if not polynomial:
        radial_coefficients = _checked_solve(kernel_matrix, control_displacements)
        return radial_coefficients, polynomial_coefficients

    polynomial_matrix = _affine_basis(control_points)
    polynomial_size = num_dims + 1
    zero_block = control_points.new_zeros(
        (batch_size, polynomial_size, polynomial_size)
    )
    system = torch.cat(
        (
            torch.cat((kernel_matrix, polynomial_matrix), dim=-1),
            torch.cat((polynomial_matrix.transpose(-2, -1), zero_block), dim=-1),
        ),
        dim=-2,
    )
    right_hand_side = torch.cat(
        (
            control_displacements,
            control_displacements.new_zeros((batch_size, polynomial_size, num_dims)),
        ),
        dim=-2,
    )
    coefficients = _checked_solve(system, right_hand_side)
    return (
        coefficients[:, :num_controls],
        coefficients[:, num_controls:],
    )


def _rbf_query_chunk_torch(
    points: torch.Tensor,
    control_points: torch.Tensor,
    radial_coefficients: torch.Tensor,
    polynomial_coefficients: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one query chunk from already-fitted TPS coefficients."""

    difference = points.unsqueeze(2) - control_points.unsqueeze(1)
    squared_distance = difference.square().sum(dim=-1)
    radial_basis = _thin_plate_spline_from_squared_distance(squared_distance)
    radial_field = torch.bmm(radial_basis, radial_coefficients)
    if polynomial_coefficients.shape[1] == 0:
        return radial_field
    polynomial_field = torch.bmm(_affine_basis(points), polynomial_coefficients)
    return radial_field + polynomial_field


def _query_chunk_size(
    batch_size: int,
    num_points: int,
    num_controls: int,
    num_dims: int,
    element_size: int,
) -> int:
    """Choose a query block size within the pairwise workspace budget."""

    if num_points == 0 or num_controls == 0:
        return max(num_points, 1)
    bytes_per_query = (
        max(batch_size, 1)
        * num_controls
        * max(element_size, 1)
        * (num_dims + 2)
        * _RBF_PAIRWISE_LIVE_VALUE_FACTOR
    )
    return min(
        num_points,
        max(1, _RBF_PAIRWISE_TEMPORARY_BYTE_BUDGET // bytes_per_query),
    )


def rbf_field_torch(
    points: torch.Tensor,
    control_points: torch.Tensor,
    radial_coefficients: torch.Tensor,
    polynomial_coefficients: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a fitted TPS field on normalized rank-3 inputs.

    Large differentiable eager workloads checkpoint their query chunks. This
    keeps backward storage proportional to the input/output tensors rather than
    the full ``(B, N, C, D)`` point/control expansion without imposing
    recomputation overhead on calls that already fit within the byte budget.
    Compiled execution uses one checkpointed vectorized block because Dynamo
    cannot unroll loops whose bounds contain symbolic dimensions.
    """

    batch_size, num_points, num_dims = points.shape
    num_controls = control_points.shape[1]
    if batch_size == 0 or num_points == 0 or num_controls == 0:
        zero = _zero_dependency(
            control_points,
            radial_coefficients,
            polynomial_coefficients,
        )
        return points * 0 + zero

    differentiable = torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (
            points,
            control_points,
            radial_coefficients,
            polynomial_coefficients,
        )
    )
    if torch.compiler.is_compiling():
        if differentiable:
            return checkpoint(
                _rbf_query_chunk_torch,
                points,
                control_points,
                radial_coefficients,
                polynomial_coefficients,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return _rbf_query_chunk_torch(
            points,
            control_points,
            radial_coefficients,
            polynomial_coefficients,
        )

    query_chunk = _query_chunk_size(
        batch_size,
        num_points,
        num_controls,
        num_dims,
        points.element_size(),
    )
    checkpoint_chunks = differentiable and query_chunk < num_points
    chunks: list[torch.Tensor] = []
    for query_start in range(0, num_points, query_chunk):
        query = points[:, query_start : query_start + query_chunk]
        if checkpoint_chunks:
            field = checkpoint(
                _rbf_query_chunk_torch,
                query,
                control_points,
                radial_coefficients,
                polynomial_coefficients,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            field = _rbf_query_chunk_torch(
                query,
                control_points,
                radial_coefficients,
                polynomial_coefficients,
            )
        chunks.append(field)

    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)


__all__ = [
    "fit_rbf_coefficients_torch",
    "rbf_field_torch",
]
