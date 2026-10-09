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

"""Closed-form determinants, inverses and least squares for batches of small matrices.

Batched ``torch.linalg`` calls factorize every tiny matrix separately, which is
slow on GPUs and a loop of LAPACK calls on CPUs. Meshes have one such matrix
per cell, so these helpers use closed forms for matrices up to 3x3, and
``small_lstsq`` a Gram-Schmidt QR over the few columns of each system
(``batched_lstsq`` picks it or ``torch.linalg.lstsq``, whichever is faster). They use
only elementwise products and sums, so they need no CUDA synchronization and
support autograd and reduced-precision dtypes.
"""

import torch
from jaxtyping import Float


def small_det(
    matrices: Float[torch.Tensor, "... n n"],
) -> Float[torch.Tensor, "..."]:
    """Determinant of each small square matrix, in closed form for n <= 3.

    Batched ``torch.linalg.det`` factorizes every tiny matrix separately: slow
    on GPUs, and a loop of LAPACK calls on CPUs.
    """
    n = matrices.shape[-1]
    m = matrices
    if n == 1:
        return m[..., 0, 0]
    if n == 2:
        return m[..., 0, 0] * m[..., 1, 1] - m[..., 0, 1] * m[..., 1, 0]
    if n == 3:  # scalar triple product of the rows
        return (m[..., 0, :] * torch.linalg.cross(m[..., 1, :], m[..., 2, :])).sum(-1)
    return torch.linalg.det(matrices)


def small_inverse(
    matrices: Float[torch.Tensor, "... n n"],
) -> Float[torch.Tensor, "... n n"]:
    """Inverse of each small invertible matrix, by its adjugate for n <= 3.

    The caller guarantees invertibility, so no error checks (which would
    synchronize CUDA callers) are made.
    """
    n = matrices.shape[-1]
    m = matrices
    if n == 1:
        return 1.0 / m
    if n == 2:
        adjugate = torch.stack(
            [
                torch.stack([m[..., 1, 1], -m[..., 0, 1]], dim=-1),
                torch.stack([-m[..., 1, 0], m[..., 0, 0]], dim=-1),
            ],
            dim=-2,
        )
        det = small_det(matrices)
    elif n == 3:
        # Row i of the cofactor matrix is the cross product of rows i + 1 and
        # i + 2 (mod 3); the adjugate is its transpose. Expanding along row 0
        # gives the determinant from the same cofactors.
        cofactors = torch.linalg.cross(m.roll(-1, dims=-2), m.roll(-2, dims=-2), dim=-1)
        adjugate = cofactors.transpose(-1, -2)
        det = (m[..., 0, :] * cofactors[..., 0, :]).sum(-1)
    else:
        return torch.linalg.inv_ex(matrices, check_errors=False).inverse
    return adjugate / det[..., None, None]


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


def batched_lstsq(
    A: Float[torch.Tensor, "... k d"],
    B: Float[torch.Tensor, "... k n_rhs"],
) -> Float[torch.Tensor, "... d n_rhs"]:
    """Minimum-norm least-squares solution of each small system ``A X = B``.

    Uses the faster method for the device, as measured on meshes of 1M cells
    (32 Grace cores) and 10M cells (GB300). On CPU, :func:`small_lstsq` is
    15-50x faster than batched ``torch.linalg.lstsq``, which loops over LAPACK
    calls. On CUDA, batched ``torch.linalg.lstsq`` is 2-4x faster than
    :func:`small_lstsq`, but its only driver, ``gels``, assumes full rank. It
    raises on exactly rank-deficient systems, such as flat stencils in a
    coordinate plane: those batches fall back to :func:`small_lstsq`. Systems
    that are rank-deficient only up to rounding (collinear or underdetermined
    stencils, say) still get the answer of ``gels`` on CUDA, which is not the
    minimum-norm solution.
    """
    if A.is_cuda:
        try:
            return torch.linalg.lstsq(A, B).solution
        except torch.linalg.LinAlgError:
            pass
    return small_lstsq(A, B)


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
