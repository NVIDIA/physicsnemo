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
(``batched_lstsq`` picks it or closed-form normal equations, whichever is
faster). They use only elementwise products, sums and tiny matrix products, so
they need no CUDA synchronization and support autograd. Tests against tolerances (singularity, rank) need at least
float32 precision, so 16-bit inputs are computed in float32.
"""

import itertools
from collections.abc import Callable

import torch
from jaxtyping import Bool, Float, Int

_HALF_DTYPES = (torch.float16, torch.bfloat16)


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
    synchronize CUDA callers) are made. :func:`normalized_inverse` also tests
    for singularity.
    """
    if matrices.shape[-1] > 3:
        return torch.linalg.inv_ex(matrices, check_errors=False).inverse
    adjugate, det = _adjugate_and_det(matrices)
    return adjugate / det[..., None, None]


def _adjugate_and_det(
    m: Float[torch.Tensor, "... n n"],
) -> tuple[Float[torch.Tensor, "... n n"], Float[torch.Tensor, "..."]]:
    """Adjugate and determinant of each matrix, n <= 3, from the same cofactors."""
    n = m.shape[-1]
    if n == 1:
        return torch.ones_like(m), m[..., 0, 0]
    if n == 2:
        adjugate = torch.stack(
            [
                torch.stack([m[..., 1, 1], -m[..., 0, 1]], dim=-1),
                torch.stack([-m[..., 1, 0], m[..., 0, 0]], dim=-1),
            ],
            dim=-2,
        )
        return adjugate, small_det(m)
    # Row i of the cofactor matrix is the cross product of rows i + 1 and i + 2
    # (mod 3); the adjugate is its transpose. Expanding along row 0 gives the
    # determinant from the same cofactors.
    cofactors = torch.linalg.cross(m.roll(-1, dims=-2), m.roll(-2, dims=-2), dim=-1)
    return cofactors.transpose(-1, -2), (m[..., 0, :] * cofactors[..., 0, :]).sum(-1)


def _unit_rows(
    matrices: Float[torch.Tensor, "... n d"],
) -> tuple[
    Float[torch.Tensor, "... n d"],
    Float[torch.Tensor, "... n"],
    Bool[torch.Tensor, "..."],
]:
    """Rows scaled to unit length, the row lengths, and whether any row is too short.

    A row whose length squared would underflow carries no direction; it keeps
    length 1, and its matrix is flagged.
    """
    row_lengths = torch.linalg.vector_norm(matrices, dim=-1)  # (..., n)
    has_length = row_lengths > torch.finfo(matrices.dtype).tiny ** 0.5
    row_lengths = torch.where(has_length, row_lengths, 1.0)
    return matrices / row_lengths[..., None], row_lengths, ~has_length.all(dim=-1)


def _invert_or_identity(
    matrices: Float[torch.Tensor, "... n n"],
    is_singular: Bool[torch.Tensor, "..."],
    adjugate: Float[torch.Tensor, "... n n"] | None,
    det: Float[torch.Tensor, "..."],
) -> Float[torch.Tensor, "... n n"]:
    """Inverse of each nonsingular matrix and the identity for the others.

    Only finite values are divided, so gradients stay finite at singular matrices.
    """
    n = matrices.shape[-1]
    eye = torch.eye(n, dtype=matrices.dtype, device=matrices.device)
    if adjugate is None:  # n > 3
        safe = torch.where(is_singular[..., None, None], eye, matrices)
        return torch.linalg.inv_ex(safe, check_errors=False).inverse
    safe_adjugate = torch.where(is_singular[..., None, None], eye, adjugate)
    safe_det = torch.where(is_singular, 1.0, det)
    return safe_adjugate / safe_det[..., None, None]


def best_base_first(
    vertices: Float[torch.Tensor, "... n_vertices n_dims"],
) -> tuple[Float[torch.Tensor, "... n_vertices n_dims"], Int[torch.Tensor, "..."]]:
    """Rotate the vertices of each simplex so that its best base vertex is first.

    Closed forms in the edges from vertex 0 lose accuracy when those edges are
    nearly parallel, as from the tip of a needle, even when the cell is well
    shaped. ``|det E|`` is the same from every vertex, but the product of the
    edge lengths is not: the vertex with the smallest product of incident edge
    lengths gives the largest normalized determinant ``|det E| / prod |e_i|``
    (see :func:`normalized_inverse`), and so the most accurate closed forms.

    Vertex ``i`` of the result is vertex ``(i + base) % n_vertices`` of the
    input. Returns the rotated vertices and ``base``.
    """
    n_vertices = vertices.shape[-2]
    if n_vertices <= 2:  # an edge has the same edge from both ends
        base = torch.zeros(
            vertices.shape[:-2], dtype=torch.long, device=vertices.device
        )
        return vertices, base
    # Squared lengths overflow float16 for edges longer than 256
    v = vertices.detach()
    if v.dtype in _HALF_DTYPES:
        v = v.float()

    def squared_length(i: int, j: int) -> torch.Tensor:
        e = v[..., i, :] - v[..., j, :]
        return (e * e).sum(dim=-1)

    if n_vertices == 3:
        ### Triangles: the edges at a vertex have product abc / (the opposite edge)
        opposite = [squared_length(1, 2), squared_length(2, 0), squared_length(0, 1)]
        base = torch.stack(opposite, dim=-1).argmax(dim=-1)
    else:
        ### Sum of the log lengths of the edges at each vertex
        # Logs, not products, which overflow or underflow for huge or tiny cells
        sums: list[torch.Tensor | None] = [None] * n_vertices
        for i, j in itertools.combinations(range(n_vertices), 2):
            log_length = squared_length(i, j).log()
            for k in (i, j):
                sums[k] = log_length if sums[k] is None else sums[k] + log_length
        base = torch.stack(sums, dim=-1).argmin(dim=-1)
    ### Rotate: vertex i of the result is vertex (i + base) % n_vertices
    order = (
        torch.arange(n_vertices, device=vertices.device) + base[..., None]
    ) % n_vertices
    return vertices.gather(-2, order[..., None].expand(vertices.shape)), base


def undo_base_rotation(
    values: Float[torch.Tensor, "... n_vertices"],
    base: Int[torch.Tensor, "..."],
) -> Float[torch.Tensor, "... n_vertices"]:
    """Per-vertex values, such as barycentric coordinates, in the vertex order
    from before :func:`best_base_first`. ``base`` broadcasts against the
    leading dimensions of ``values``."""
    n_vertices = values.shape[-1]
    index = (
        torch.arange(n_vertices, device=values.device) - base[..., None]
    ) % n_vertices
    return values.gather(-1, index.expand(values.shape))


def normalized_inverse(
    matrices: Float[torch.Tensor, "... n n"],
    tolerance: float | None = None,
) -> tuple[
    Float[torch.Tensor, "... n n"],
    Float[torch.Tensor, "... n"],
    Bool[torch.Tensor, "..."],
]:
    r"""Inverse of each small square matrix ``M``, with a scale-free singularity test.

    Each row is first scaled to unit length, ``M = D \hat{M}`` with ``D`` the
    diagonal of row lengths. Then :math:`|\det \hat{M}| \le 1` (Hadamard's
    inequality), with equality for orthogonal rows of any lengths, so it measures
    how close the rows are to linear dependence, not how stretched ``M`` is. A
    matrix counts as singular when a row has no usable length or
    :math:`|\det \hat{M}| \le` ``tolerance`` (by default ``n * eps``, the rounding
    error of an exactly singular matrix). Singular matrices get the identity
    instead of :math:`\hat{M}^{-1}`, keeping values and gradients finite; callers
    choose what to return for them.

    Returns
    -------
    inverse : Float[torch.Tensor, "... n n"]
        :math:`\hat{M}^{-1}`, so that :math:`M^{-1} = \hat{M}^{-1} D^{-1}`, i.e.
        ``inverse / row_lengths[..., None, :]``.
    row_lengths : Float[torch.Tensor, "... n"]
        Length of each row (1 for rows too short to scale).
    is_singular : Bool[torch.Tensor, "..."]
        Matrices that are singular to working precision, or below ``tolerance``.
    """
    n = matrices.shape[-1]
    if tolerance is None:
        tolerance = n * torch.finfo(matrices.dtype).eps
    unit, row_lengths, has_short_row = _unit_rows(matrices)
    if n <= 3:
        adjugate, det = _adjugate_and_det(unit)
    else:
        adjugate, det = None, torch.linalg.det(unit)
    is_singular = has_short_row | ~(det.abs() > tolerance)
    return (
        _invert_or_identity(unit, is_singular, adjugate, det),
        row_lengths,
        is_singular,
    )


def normalized_gram_inverse(
    rows: Float[torch.Tensor, "... n d"],
    tolerance: float | None = None,
) -> tuple[
    Float[torch.Tensor, "... n n"],
    Float[torch.Tensor, "... n"],
    Bool[torch.Tensor, "..."],
]:
    r"""Inverse of the Gram matrix of the unit rows of each ``n x d`` matrix, n <= d.

    With ``E = D \hat{E}`` (rows scaled to unit length), the Gram matrix
    :math:`G = E E^\top = D \hat{G} D` and :math:`\hat{G} = \hat{E} \hat{E}^\top` has a
    unit diagonal, so :math:`0 \le \det \hat{G} \le 1`. The singularity test is
    that of :func:`normalized_inverse`, :math:`\sqrt{\det \hat{G}} \le`
    ``tolerance``, since :math:`\det \hat{G} = \det(\hat{E})^2` for square ``E``.

    Returns
    -------
    inverse : Float[torch.Tensor, "... n n"]
        :math:`\hat{G}^{-1}`, so that :math:`G^{-1} = D^{-1} \hat{G}^{-1} D^{-1}`.
    row_lengths : Float[torch.Tensor, "... n"]
        Length of each row (1 for rows too short to scale).
    is_singular : Bool[torch.Tensor, "..."]
        Matrices whose rows are dependent to working precision, or below
        ``tolerance``; their inverse is the identity.
    """
    n = rows.shape[-2]
    if tolerance is None:
        tolerance = n * torch.finfo(rows.dtype).eps
    unit, row_lengths, has_short_row = _unit_rows(rows)
    # Broadcast-and-sum: a batched matmul of millions of tiny matrices is slow.
    gram = (unit.unsqueeze(-2) * unit.unsqueeze(-3)).sum(dim=-1)  # (..., n, n)
    if n <= 3:
        adjugate, det = _adjugate_and_det(gram)
    else:
        adjugate, det = None, torch.linalg.det(gram)
    is_singular = has_short_row | ~(det > tolerance**2)
    return (
        _invert_or_identity(gram, is_singular, adjugate, det),
        row_lengths,
        is_singular,
    )


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
    if A.dtype in _HALF_DTYPES:  # the rank test needs float32 precision
        return small_lstsq(A.float(), B.float()).to(A.dtype)
    k, d = A.shape[-2:]
    n_rhs = B.shape[-1]
    batch_shape = A.shape[:-2]
    columns = torch.arange(d + n_rhs, device=A.device)
    identity_rows = [(columns[:d] == i).to(A.dtype) for i in range(d)]  # each (d,)

    ### Normalize A by its largest entry, so squared norms stay in range
    # NaN or inf entries keep a NaN or inf scale, so they propagate.
    scale = A.detach().abs().amax(dim=(-2, -1), keepdim=True)  # (..., 1, 1)
    scale = torch.where(scale == 0, 1.0, scale)
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
        # At most k columns are independent. Written so that NaN counts as
        # independent and propagates, rather than zeroing the solution.
        if j < k:
            independent = ~(norm2 <= (rcond * R_00) ** 2)
        else:
            independent = torch.zeros_like(norm2, dtype=torch.bool)
        norm = torch.where(independent, norm2, 1.0).sqrt()
        q = torch.where(
            independent.unsqueeze(-1), column / norm.unsqueeze(-1), 0.0
        )  # (..., k)
        del column  # a view, which would keep this step's `work` alive

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

    del work  # only R and Q^T B are needed from here on

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

    Float32 systems with at most 3 unknowns, such as LSQ gradient stencils,
    use :func:`_normal_equations_lstsq`; all others use :func:`small_lstsq`.
    Both give the minimum-norm solution with the rank decision of CPU
    ``torch.linalg.lstsq``, on every device, and need no CUDA synchronization.
    (CUDA ``torch.linalg.lstsq`` has only the driver ``gels``, which assumes
    full rank: it returns NaN, inf or a large wrong answer for rank-deficient
    systems such as flat stencils.) The normal equations are 3 to 8 times
    faster than :func:`small_lstsq` on GB300 and 1.6 to 8 times faster on
    CPU. On GB300 they are as fast as CUDA ``torch.linalg.lstsq`` for 2
    unknowns, and about 2 times slower for 3.
    """
    if A.dtype in _HALF_DTYPES:  # the rank tests need float32 precision
        return batched_lstsq(A.float(), B.float()).to(A.dtype)
    if A.dtype == torch.float32 and A.shape[-1] <= 3:
        return _normal_equations_lstsq(A, B)
    return small_lstsq(A, B)


def _lstsq_solver(
    A: Float[torch.Tensor, "... k d"],
) -> Callable[[Float[torch.Tensor, "... k n_rhs"]], Float[torch.Tensor, "... d n_rhs"]]:
    """Solver of ``A X = B`` for any number of right-hand sides ``B``.

    ``_lstsq_solver(A)(B)`` is ``batched_lstsq(A, B)``, bit for bit. The
    normal equations factorize ``A`` once, for every ``B``; :func:`small_lstsq`
    solves each ``B`` from scratch, since its Gram-Schmidt sums over the
    columns of ``[A | B]`` together, whose order depends on their number.
    """
    if A.dtype in _HALF_DTYPES:  # the rank tests need float32 precision
        solve = _lstsq_solver(A.float())
        return lambda B: solve(B.float()).to(A.dtype)
    if A.dtype == torch.float32 and A.shape[-1] <= 3:
        return _normal_equations_solver(A)[1]
    return lambda B: small_lstsq(A, B)


def _uses_normal_equations(dtype: torch.dtype, d: int) -> bool:
    """Whether :func:`batched_lstsq` solves systems of ``d`` unknowns of
    ``dtype`` by the normal equations, which :func:`_segment_lstsq_solver` can."""
    return dtype in (torch.float32, *_HALF_DTYPES) and d <= 3


def _segment_lstsq_solver(
    A: Float[torch.Tensor, "n_rows d"],
    offsets: Int[torch.Tensor, " n_systems_plus_1"],
) -> Callable[[Float[torch.Tensor, "n_rows n_rhs"]], Float[torch.Tensor, "..."]]:
    """Solver of the systems whose rows are consecutive segments of ``A``.

    System ``s`` has rows ``offsets[s]:offsets[s + 1]`` of ``A`` and ``B``,
    any number of them (none gives zero). The solver returns the solutions,
    ``(n_systems, d, n_rhs)``, of ``batched_lstsq`` of each system, for the
    ``A`` for which :func:`_uses_normal_equations`: those of
    :func:`_normal_equations_solver`, but for all sizes of systems at once.
    Its moments are summed in float64 over each segment of rows, in their
    order, so they may differ from the batched sums in the last bits of
    float64, rarely enough to change a float32 solution.
    """
    if A.dtype in _HALF_DTYPES:  # the rank tests need float32 precision
        solve = _segment_lstsq_solver(A.float(), offsets)
        return lambda B: solve(B.float()).to(A.dtype)
    d = A.shape[-1]
    # The tolerance of each system for its own number of rows
    eps = torch.finfo(A.dtype).eps
    counts = offsets[1:] - offsets[:-1]
    eigenvalue_tolerance = (eps * counts.clamp_min(d).double()) ** 2

    def segment_sums(x: torch.Tensor) -> torch.Tensor:
        return _SegmentSum.apply(x, offsets)

    # Products of float32 entries are exact in float64: one float64 column of A
    # at a time is cast, and the other promoted in the product
    moments = {}
    for i in range(d):
        a_i = A[:, i].double()
        for j in range(i, d):
            moments[i, j] = segment_sums(a_i * A[:, j])
    del a_i
    solve_moments = _moments_solver(moments, eigenvalue_tolerance)

    # Plain annotations: jaxtyping ones would be built again on every call
    def solve(B: torch.Tensor) -> torch.Tensor:
        """Solutions ``(n_systems, d, n_rhs)`` for ``B`` of shape ``(n_rows, n_rhs)``."""
        # B is promoted to float64 in the products, as A^T B.double()
        rhs = [segment_sums(A[:, i, None].double() * B) for i in range(d)]
        return solve_moments(rhs).to(A.dtype)

    return solve


class _SegmentSum(torch.autograd.Function):
    """Sums of the rows of ``x`` over the segments given by ``offsets``.

    ``torch.segment_reduce``, which sums each segment in order (unlike the
    atomics of ``index_add_``, so the same on every run), with a backward that
    is itself differentiable, for second derivatives.
    """

    @staticmethod
    def forward(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        # Rows of one entry would go to CUB's segmented reduction, one thread
        # block per segment: 15 times slower for segments of 10 to 20 rows
        rows = x.reshape(x.shape[0], x.shape[1:].numel())
        sums = torch.segment_reduce(rows, "sum", offsets=offsets, unsafe=True)
        return sums.view(offsets.shape[0] - 1, *x.shape[1:])

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        x, offsets = inputs
        ctx.save_for_backward(offsets)
        ctx.n_rows = x.shape[0]

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        (offsets,) = ctx.saved_tensors
        counts = offsets[1:] - offsets[:-1]
        return grad.repeat_interleave(counts, dim=0, output_size=ctx.n_rows), None


def _normal_equations_lstsq(
    A: Float[torch.Tensor, "... k d"],
    B: Float[torch.Tensor, "... k n_rhs"],
) -> Float[torch.Tensor, "... d n_rhs"]:
    """Least squares by :func:`_normal_equations_solver`, for ``B`` only."""
    return _normal_equations_solver(A, B)[0]


def _normal_equations_solver(
    A: Float[torch.Tensor, "... k d"],
    B: Float[torch.Tensor, "... k n_rhs"] | None = None,
) -> tuple[
    Float[torch.Tensor, "... d n_rhs"] | None,
    Callable[[Float[torch.Tensor, "... k n_rhs"]], Float[torch.Tensor, "... d n_rhs"]],
]:
    """Minimum-norm least squares of float32 systems with ``d <= 3`` unknowns.

    Returns the solution for ``B`` (``None`` without ``B``) and the solver of
    ``A X = B`` for further right-hand sides: the pseudo-inverse of ``M``
    below is computed once, here. ``A^T B`` uses the float64 columns of ``A``
    that form ``M``; the solver casts them again, one at a time.

    Solves the normal equations ``M X = A^T B``, ``M = A^T A``, formed in
    float64, in which products of float32 entries are exact. Squaring the
    condition number then costs less accuracy than a float32 QR factorization
    loses: on random systems with condition numbers up to 1e5, the largest
    errors are about 1000 times smaller than those of :func:`small_lstsq`.

    ``M`` is divided by its trace, so that its eigenvalues sum to 1. As in
    :func:`small_lstsq`, singular values of ``A`` below ``rcond`` times the
    largest count as zero, with ``rcond = eps * max(k, d)`` of the input
    dtype, so eigenvalues of ``M`` below ``rcond**2`` do. The pseudo-inverse is
    in closed form: the inverse at full rank, and ``M`` itself at rank 1, since
    then ``M = v v^T`` for a unit vector ``v`` (which also gives 0 for
    ``A = 0``). For ``d = 3``, the largest diagonal entry of ``M`` is
    eliminated first, as in pivoted Cholesky, so that the 2x2 Schur complement
    resolves the two smallest eigenvalues even when both are small. The
    elimination gives a generalized inverse ``G``; at rank 2, the
    pseudo-inverse is ``P G P``, with ``P`` the projector onto the range of
    ``M``.

    Every branch is finite for every input, so gradients are finite also for
    rank-deficient systems, and NaN or inf entries propagate. Each entry of
    ``M`` and of its pseudo-inverse is a separate tensor over the batch: in
    float64, batched matrix products of tiny matrices are many times slower
    than elementwise operations, on GPUs and on CPUs.
    """
    k, d = A.shape[-2:]
    eigenvalue_tolerance = (torch.finfo(A.dtype).eps * max(k, d)) ** 2
    a = [A[..., i].double() for i in range(d)]  # columns of A, each (..., k)
    m = {(i, j): (a[i] * a[j]).sum(-1) for i in range(d) for j in range(i, d)}
    rhs = None
    if B is not None:
        B = B.double()
        rhs = [(a[i].unsqueeze(-1) * B).sum(-2) for i in range(d)]  # each (..., n_rhs)
    del a, B  # the float64 copies

    solve_moments = _moments_solver(m, eigenvalue_tolerance)

    # Plain annotations: jaxtyping ones would be built again on every call
    def solve(B: torch.Tensor) -> torch.Tensor:
        """Solution ``(..., d, n_rhs)`` for ``B`` of shape ``(..., k, n_rhs)``."""
        B = B.double()
        # One float64 column of A at a time: at most one copy is alive
        rhs = [(A[..., i].double().unsqueeze(-1) * B).sum(-2) for i in range(d)]
        del B  # the float64 copy
        return solve_moments(rhs).to(A.dtype)

    if rhs is None:
        return None, solve
    return solve_moments(rhs).to(A.dtype), solve


def _moments_solver(
    moments: dict[tuple[int, int], torch.Tensor],
    eigenvalue_tolerance: float | torch.Tensor,
) -> Callable[[list[torch.Tensor]], torch.Tensor]:
    """Closed-form solver of the normal equations ``M X = A^T B`` of
    :func:`_normal_equations_solver`, from their float64 moments.

    ``moments`` holds the upper entries ``(i, j)``, ``i <= j < d``, of ``M``,
    each of shape ``(...)``; ``eigenvalue_tolerance`` is a float or a tensor
    that broadcasts against them. The solver takes the ``d`` rows of ``A^T B``,
    each ``(..., n_rhs)``, and returns ``X``, ``(..., d, n_rhs)``, in float64.
    Both empty the containers they are given, so that their entries are freed
    as soon as they are used.
    """
    d = max(j for _, j in moments) + 1
    m = moments

    ### Divide M by its trace, the sum of its eigenvalues
    trace = sum(m[i, i] for i in range(d))  # (...)
    trace = torch.where(trace == 0, 1.0, trace)  # A = 0; keeps NaN and inf
    m = {key: value / trace for key, value in m.items()}
    moments.clear()  # the caller's dict: frees the unnormalized entries

    def times(
        symmetric: dict[tuple[int, int], torch.Tensor], v: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Product of a symmetric matrix, given by its upper entries, and v.

        Entries of v may have a trailing dimension of right-hand sides.
        """

        def entry(i: int, j: int) -> torch.Tensor:
            x = symmetric[min(i, j), max(i, j)]
            return x.unsqueeze(-1) if v[j].dim() > x.dim() else x

        return [sum(entry(i, j) * v[j] for j in range(d)) for i in range(d)]

    def _pseudo_inverse(
        m: dict[tuple[int, int], torch.Tensor],
    ) -> dict[tuple[int, int], torch.Tensor]:
        """Closed-form pseudo-inverse; its intermediates are freed on return."""
        ### Pseudo-inverse of M in closed form, by its upper entries
        # Written so that NaN counts as full rank and propagates.
        if d == 1:
            pseudo_inverse = m
        elif d == 2:
            det = m[0, 0] * m[1, 1] - m[0, 1].square()
            # det is the product of the eigenvalues, the larger in [1/2, 1]
            full_rank = ~(det <= eigenvalue_tolerance)
            det = torch.where(full_rank, det, 1.0)
            pseudo_inverse = {
                (0, 0): torch.where(full_rank, m[1, 1] / det, m[0, 0]),
                (0, 1): torch.where(full_rank, -m[0, 1] / det, m[0, 1]),
                (1, 1): torch.where(full_rank, m[0, 0] / det, m[1, 1]),
            }
        else:
            # Symmetric elimination of the largest diagonal entry, as in pivoted
            # Cholesky: the 2x2 Schur complement S holds the two smallest
            # eigenvalues at their own scale, so that its determinant resolves
            # them (the 3x3 determinant cannot, when both are small).
            p0 = (m[0, 0] >= m[1, 1]) & (m[0, 0] >= m[2, 2])
            p1 = ~p0 & (m[1, 1] >= m[2, 2])

            def pick(
                x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor
            ) -> torch.Tensor:
                """x_p for the pivot p."""
                return torch.where(p0, x0, torch.where(p1, x1, x2))

            # Rows and columns in the order (p, q, r), with (q, r) cyclic after p
            pivot = pick(m[0, 0], m[1, 1], m[2, 2])  # at least 1/3, or 0 for A = 0
            pivot = torch.where(pivot > 0, pivot, 1.0)
            b_q, b_r = pick(m[0, 1], m[1, 2], m[0, 2]), pick(m[0, 2], m[0, 1], m[1, 2])
            S_qq = pick(m[1, 1], m[2, 2], m[0, 0]) - b_q.square() / pivot
            S_rr = pick(m[2, 2], m[0, 0], m[1, 1]) - b_r.square() / pivot
            S_qr = pick(m[1, 2], m[0, 2], m[0, 1]) - b_q * b_r / pivot
            trace_S = S_qq + S_rr  # within a factor 3 of the two smallest eigenvalues
            det_S = S_qq * S_rr - S_qr.square()
            rank_2 = ~(trace_S <= eigenvalue_tolerance)
            full_rank = rank_2 & ~(det_S <= eigenvalue_tolerance * trace_S)

            # Inverse of S at full rank, else its pseudo-inverse S / tr(S)^2 (rank 1)
            det_S = torch.where(full_rank, det_S, 1.0)
            trace_S2 = torch.where(rank_2, trace_S, 1.0).square()
            Sg_qq = torch.where(full_rank, S_rr / det_S, S_qq / trace_S2)
            Sg_rr = torch.where(full_rank, S_qq / det_S, S_rr / trace_S2)
            Sg_qr = torch.where(full_rank, -S_qr / det_S, S_qr / trace_S2)
            del trace_S, det_S, trace_S2  # each intermediate is freed once used
            # A generalized inverse G of M by block elimination: M G M = M
            h_q = (Sg_qq * b_q + Sg_qr * b_r) / pivot
            h_r = (Sg_qr * b_q + Sg_rr * b_r) / pivot
            G = {
                (0, 0): (1 + b_q * h_q + b_r * h_r) / pivot,
                (0, 1): -h_q,
                (0, 2): -h_r,
                (1, 1): Sg_qq,
                (1, 2): Sg_qr,
                (2, 2): Sg_rr,
            }
            del h_q, h_r

            # At rank 2 the pseudo-inverse is P G P, with P = I - n n^T the
            # projector onto the range of M. Its null vector n solves S (s_q, s_r)
            # = 0; take the column of adj(S) with the larger diagonal entry.
            larger_rr = S_rr >= S_qq
            s_q = torch.where(larger_rr, S_rr, -S_qr)
            s_r = torch.where(larger_rr, -S_qr, S_qq)
            s_p = -(b_q * s_q + b_r * s_r) / pivot
            del pivot, b_q, b_r, S_qq, S_rr, S_qr
            # Guarded before the square root, whose gradient at 0 is infinite
            null_norm2 = s_p.square() + s_q.square() + s_r.square()
            null_norm = torch.where(null_norm2 > 0, null_norm2, 1.0).sqrt()
            # n = 0 at full rank, where P G P = G
            n = [torch.where(full_rank, 0.0, x / null_norm) for x in (s_p, s_q, s_r)]
            del s_p, s_q, s_r, null_norm2, null_norm
            Gn = times(G, n)
            nGn = sum(n_i * g_i for n_i, g_i in zip(n, Gn))
            PGP = {
                (i, j): G[i, j] - n[i] * Gn[j] - Gn[i] * n[j] + nGn * n[i] * n[j]
                for i in range(3)
                for j in range(i, 3)
            }
            del G, n, Gn, nGn

            # Back to the original order (p, q, r = 0, 1, 2 / 1, 2, 0 / 2, 0, 1),
            # and M itself at rank 1
            unpermuted = {
                (0, 0): pick(PGP[0, 0], PGP[2, 2], PGP[1, 1]),
                (0, 1): pick(PGP[0, 1], PGP[0, 2], PGP[1, 2]),
                (0, 2): pick(PGP[0, 2], PGP[1, 2], PGP[0, 1]),
                (1, 1): pick(PGP[1, 1], PGP[0, 0], PGP[2, 2]),
                (1, 2): pick(PGP[1, 2], PGP[0, 1], PGP[0, 2]),
                (2, 2): pick(PGP[2, 2], PGP[1, 1], PGP[0, 0]),
            }
            del PGP
            pseudo_inverse = {
                key: torch.where(rank_2, unpermuted.pop(key), m[key])
                for key in list(unpermuted)
            }
        return pseudo_inverse

    pseudo_inverse = _pseudo_inverse(m)
    del m

    # Plain annotations: jaxtyping ones would be built again on every call
    def solve(rhs: list[torch.Tensor]) -> torch.Tensor:
        """Solution ``(..., d, n_rhs)`` for the rows ``rhs`` of ``A^T B``."""
        X = times(pseudo_inverse, rhs)
        rhs.clear()  # the caller's list: frees its entries
        X = torch.stack(X, dim=-2)  # frees its rows
        return X / trace[..., None, None]

    return solve


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
