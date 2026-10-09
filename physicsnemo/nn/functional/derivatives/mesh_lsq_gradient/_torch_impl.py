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

from __future__ import annotations

import torch
from jaxtyping import Float

from .utils import resolve_safe_epsilon, validate_inputs


def mesh_lsq_gradient_torch(
    points: torch.Tensor,
    values: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    weight_power: float = 2.0,
    min_neighbors: int = 0,
    safe_epsilon: float | None = None,
) -> torch.Tensor:
    """Compute weighted LSQ mesh gradients with PyTorch tensor ops."""
    ### Validate inputs before building LSQ systems.
    validate_inputs(
        points=points,
        values=values,
        neighbor_offsets=neighbor_offsets,
        neighbor_indices=neighbor_indices,
        min_neighbors=min_neighbors,
    )

    ### Normalize dtypes/layout for stable downstream linear algebra.
    points = points.contiguous()
    values = values.contiguous()
    neighbor_offsets = neighbor_offsets.to(
        dtype=torch.int64, device=points.device
    ).contiguous()
    neighbor_indices = neighbor_indices.to(
        dtype=torch.int64, device=points.device
    ).contiguous()

    n_entities = points.shape[0]
    n_dims = points.shape[1]
    value_shape = values.shape[1:]
    counts = neighbor_offsets[1:] - neighbor_offsets[:-1]

    ### Flatten component dimensions so scalar and tensor fields share one solve path.
    values_flat = values.reshape(n_entities, -1)
    n_components = values_flat.shape[1]
    gradients_flat = torch.zeros(
        (n_entities, n_dims, n_components),
        dtype=values.dtype,
        device=values.device,
    )

    points_cast = points.to(dtype=values.dtype)
    dist_eps = resolve_safe_epsilon(safe_epsilon=safe_epsilon, dtype=points_cast.dtype)

    ### Process one dense batch per neighbor-count group (mesh-module strategy).
    unique_counts = torch.unique(counts)
    for count_tensor in unique_counts:
        n_neighbors = int(count_tensor.item())
        if n_neighbors < min_neighbors or n_neighbors == 0:
            continue

        entity_indices = torch.where(counts == count_tensor)[0]
        if entity_indices.numel() == 0:
            continue

        offsets_group = neighbor_offsets[entity_indices]
        col_range = torch.arange(n_neighbors, device=points.device, dtype=torch.int64)
        flat_indices = offsets_group.unsqueeze(1) + col_range.unsqueeze(0)
        neighbors = neighbor_indices[flat_indices].to(torch.long)

        center_points = points_cast[entity_indices]
        relative = points_cast[neighbors] - center_points.unsqueeze(1)

        values_center = values_flat[entity_indices]
        delta_values = values_flat[neighbors] - values_center.unsqueeze(1)

        dist2 = (relative * relative).sum(dim=-1).clamp_min(dist_eps)
        sqrt_w = dist2.pow(-0.25 * weight_power).unsqueeze(-1)

        A_weighted = sqrt_w * relative
        b_weighted = sqrt_w * delta_values

        solution = small_lstsq(A_weighted, b_weighted)
        gradients_flat[entity_indices] = solution

    ### Restore gradient output shape.
    gradients = gradients_flat.reshape(n_entities, n_dims, *value_shape)

    return gradients


def small_lstsq(
    A: Float[torch.Tensor, "... k d"],
    B: Float[torch.Tensor, "... k n_rhs"],
) -> Float[torch.Tensor, "... d n_rhs"]:
    """Minimum-norm least-squares solution of each small system ``A X = B``.

    Batched ``torch.linalg.lstsq`` factorizes every tiny matrix separately:
    slow on GPUs, and a loop of LAPACK calls on CPUs. This factorizes all of
    them at once, by modified Gram-Schmidt with column pivoting over the
    columns of ``[A | B]`` (backward stable for least squares), with only
    elementwise products and sums: no CUDA synchronization, and autograd
    support. Use it for small ``d``: it loops over the columns in Python.

    Like CPU ``lstsq(A, B, rcond=None)`` (driver ``gelsy``), it treats ``A``
    as rank-deficient where the condition number of the leading block of
    ``R`` exceeds ``1 / rcond``, with ``rcond = eps * max(k, d)``, and returns
    the minimum-norm solution, also for ``k < d``. (CUDA ``lstsq`` has only
    the driver ``gels``, which assumes full rank.)

    Parameters
    ----------
    A : Float[torch.Tensor, "... k d"]
        Matrices of the systems.
    B : Float[torch.Tensor, "... k n_rhs"]
        Right-hand sides.

    Returns
    -------
    Float[torch.Tensor, "... d n_rhs"]
        Minimum-norm least-squares solutions.
    """
    k, d = A.shape[-2:]
    n_rhs = B.shape[-1]
    batch_shape = A.shape[:-2]
    columns = torch.arange(d + n_rhs, device=A.device)
    identity_rows = [(columns[:d] == i).to(A.dtype) for i in range(d)]  # each (d,)

    ### Normalize A by its largest entry, so squared norms stay in range
    scale = A.detach().abs().amax(dim=(-2, -1), keepdim=True)  # (..., 1, 1)
    scale = torch.where(scale > 0, scale, 1.0)
    work = torch.cat([A / scale, B], dim=-1)  # (..., k, d + n_rhs)

    rcond = torch.finfo(A.dtype).eps * max(k, d)
    # With pivoting, R_00 is the largest column norm, within sqrt(d) of the
    # largest singular value.
    R_00 = work[..., :d].detach().square().sum(-2).amax(-1).sqrt()  # (...)

    ### QR factorization by modified Gram-Schmidt with column pivoting
    # Step j orthogonalizes the remaining columns of `work` against column j.
    # R_rows[j] is row j of R (length d, in pivoted column order) and
    # QtB_rows[j] row j of Q^T B. A column whose residual is at most
    # rcond * R_00 is dependent: its row of R is the identity row.
    permutation = columns[:d].expand(*batch_shape, d)
    R_rows: list[torch.Tensor] = []
    QtB_rows: list[torch.Tensor] = []
    dependent: list[torch.Tensor] = []
    for j in range(d):
        # Swap the remaining column with the largest residual into position j
        if j < d - 1:
            residual2 = work[..., :, j:d].square().sum(-2)  # (..., d - j)
            pivot = residual2.argmax(dim=-1, keepdim=True) + j  # (..., 1)
            swap = torch.where(
                columns == j, pivot, torch.where(columns == pivot, j, columns)
            )  # (..., d + n_rhs)
            work = work.take_along_dim(swap.unsqueeze(-2), dim=-1)
            permutation = permutation.take_along_dim(swap[..., :d], dim=-1)
            R_rows = [row.take_along_dim(swap[..., :d], dim=-1) for row in R_rows]

        column = work[..., :, j]  # (..., k)
        norm2 = column.square().sum(-1)  # (...)
        # At most k columns are independent.
        if j < k:
            independent = norm2 > (rcond * R_00) ** 2
        else:
            independent = torch.zeros_like(norm2, dtype=torch.bool)
        norm = torch.where(independent, norm2, 1.0).sqrt()
        q = torch.where(
            independent.unsqueeze(-1), column / norm.unsqueeze(-1), 0.0
        )  # (..., k)

        r = (q.unsqueeze(-1) * work[..., :, j + 1 :]).sum(-2)  # (..., d-j-1+n_rhs)
        work = torch.cat(
            [
                work[..., :, : j + 1],
                work[..., :, j + 1 :] - q.unsqueeze(-1) * r.unsqueeze(-2),
            ],
            dim=-1,
        )
        R_rows.append(
            torch.cat(
                [
                    norm.new_zeros((*batch_shape, j)),
                    norm.unsqueeze(-1),
                    r[..., : d - j - 1],
                ],
                dim=-1,
            )
        )
        QtB_rows.append(r[..., d - j - 1 :])
        dependent.append(~independent)

    ### Rank decision, as in gelsy
    # Column j is also dependent when the smallest singular value of the leading
    # (j + 1) x (j + 1) block of R, within sqrt(d) of the reciprocal of the
    # largest column norm of its inverse, is at most rcond * R_00.
    R_inverse = _solve_upper_triangular(
        [row.detach() for row in R_rows],
        [row.expand(*batch_shape, d) for row in identity_rows],
    )  # (..., d, d)
    inverse_norm2 = R_inverse.square().sum(-2).cummax(dim=-1).values  # (..., d)
    ill_conditioned = inverse_norm2 * ((rcond * R_00) ** 2).unsqueeze(-1) >= 1
    dependent = [dependent[j] | ill_conditioned[..., j] for j in range(d)]
    R_rows = [
        torch.where(dependent[j].unsqueeze(-1), identity_rows[j], R_rows[j])
        for j in range(d)
    ]
    QtB_rows = [
        torch.where(dependent[j].unsqueeze(-1), 0.0, QtB_rows[j]) for j in range(d)
    ]

    ### Back substitution for a basic solution and a null-space basis
    # Solving R n = e_j for a dependent column j gives a null vector of A.
    solutions = _solve_upper_triangular(
        R_rows,
        [
            torch.cat(
                [QtB_rows[j], identity_rows[j] * dependent[j].unsqueeze(-1)], dim=-1
            )
            for j in range(d)
        ],
    )  # (..., d, n_rhs + d)
    X = solutions[..., :n_rhs]  # (..., d, n_rhs)

    ### Project out the null space for the minimum-norm solution
    null_basis: list[torch.Tensor] = []
    for j in range(d):
        u = solutions[..., :, n_rhs + j]  # (..., d), zero for independent j
        for v in null_basis:
            u = u - (u * v).sum(-1, keepdim=True) * v
        u_norm = torch.where(dependent[j], u.square().sum(-1), 1.0).sqrt()
        u = torch.where(dependent[j].unsqueeze(-1), u / u_norm.unsqueeze(-1), 0.0)
        null_basis.append(u)
        X = X - u.unsqueeze(-1) * (u.unsqueeze(-1) * X).sum(-2, keepdim=True)

    ### Undo the column pivoting and the normalization
    unpermute = permutation.unsqueeze(-1).expand(*batch_shape, d, n_rhs)
    return torch.zeros_like(X).scatter(-2, unpermute, X) / scale


def _solve_upper_triangular(
    R_rows: list[torch.Tensor],
    rhs_rows: list[torch.Tensor],
) -> Float[torch.Tensor, "... d n_rhs"]:
    """Solve ``R X = rhs`` by back substitution, given the rows of each."""
    solution_rows: list[torch.Tensor] = []  # rows i + 1, ..., d - 1
    for i in reversed(range(len(R_rows))):
        rhs = rhs_rows[i]
        for offset, row in enumerate(solution_rows, start=i + 1):
            rhs = rhs - R_rows[i][..., offset, None] * row
        solution_rows.insert(0, rhs / R_rows[i][..., i, None])
    return torch.stack(solution_rows, dim=-2)
