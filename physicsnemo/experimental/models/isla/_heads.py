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

"""ISLA's output side: the rank-r equivariant expansion.

:class:`_OutputLayout` does the coefficient bookkeeping of one declared output field,
and :func:`_assemble_output` contracts those coefficients with tensor products of the
token's basis vectors. :func:`_pseudovectors` and :func:`_isotropic_tensors` build the
odd-parity and constant parts of that basis. Private to
:mod:`physicsnemo.experimental.models.isla`."""

import itertools
from dataclasses import dataclass, field

import torch

from physicsnemo.mesh.fields import RankSpec
from physicsnemo.nn.functional.equivariant_ops import vector_project


def _levi_civita(dim: int) -> torch.Tensor:
    """The Levi-Civita symbol of ``dim`` dimensions, shape ``(dim,)*dim``: the sign
    of the permutation at each permuted multi-index, zero elsewhere. It is an
    isotropic pseudotensor of rank ``dim`` in every dimension (``J`` at ``dim = 2``,
    the usual rank-3 symbol at ``dim = 3``)."""
    eps = torch.zeros((dim,) * dim, dtype=torch.float64)
    for perm in itertools.permutations(range(dim)):
        inversions = sum(1 for i in range(dim) for j in range(i + 1, dim) if perm[i] > perm[j])
        eps[perm] = -1.0 if inversions % 2 else 1.0
    return eps


def _isotropic_tensors(rank: int, symmetric: bool, parity: str, dim: int) -> list[torch.Tensor]:
    """The constant (isotropic) tensors of a rank and parity in ``dim`` spatial
    dimensions, each of shape ``(dim,)*rank``.

    Even parity: the identity at rank 2; at rank 4 the three products of
    Kronecker deltas ``d_ij d_kl, d_ik d_jl, d_il d_jk`` (their sum alone for a
    symmetric spec, the only fully symmetric isotropic rank-4 tensor). Odd
    parity: the Levi-Civita symbol, which has rank ``dim`` (rank 2 in two
    dimensions, rank 3 in three; a pseudotensor, antisymmetric, so not for a
    symmetric spec). No isotropic tensor exists at ranks 0, 1, at even rank 3,
    or at odd ranks other than ``dim``.
    """
    eye = torch.eye(dim, dtype=torch.float64)
    if parity == "even":
        if rank == 2:
            return [eye]
        if rank == 4:
            terms = [
                torch.einsum("ij,kl->ijkl", eye, eye),
                torch.einsum("ik,jl->ijkl", eye, eye),
                torch.einsum("il,jk->ijkl", eye, eye),
            ]
            return [sum(terms)] if symmetric else terms
        return []
    if rank == dim and not symmetric:
        return [_levi_civita(dim)]
    return []


@dataclass(frozen=True, eq=False)
class _OutputLayout:
    """Coefficient bookkeeping of one output field.

    With ``m`` true and ``p`` pseudo basis vectors per token, a rank-``r``
    output of even parity is expanded in tensor products of ``r`` true vectors
    (``m**r`` ordered products, or one per unordered multi-index for a
    symmetric spec), one of odd parity in products with exactly one
    pseudovector factor (``r p m**(r-1)`` ordered; ``p`` times the unordered
    multisets of ``r-1`` true vectors for a symmetric spec), plus the isotropic
    tensors of that rank and parity. ``index`` maps every ordered multi-index
    over the relevant vector set (the true vectors alone for even parity, true
    then pseudo for odd) to its free coefficient, or to the trailing zero slot.
    """

    spec: RankSpec
    n_true: int
    n_pseudo: int
    dim: int
    n_vec: int = field(init=False)
    n_free: int = field(init=False)
    index: torch.Tensor | None = field(init=False)
    iso: list[torch.Tensor] = field(init=False)
    n_coeffs: int = field(init=False)

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        spec = self.spec
        r = spec.rank
        set_(self, "dim", int(self.dim))
        n_vec = self.n_true if spec.parity == "even" else self.n_true + self.n_pseudo
        set_(self, "n_vec", n_vec)
        if r == 0:
            set_(self, "n_free", 1)
            set_(self, "index", None)
            set_(self, "iso", [])
            set_(self, "n_coeffs", 1)
            return
        free: dict[tuple[int, ...], int] = {}
        index = []
        for j in itertools.product(range(n_vec), repeat=r):
            n_odd = sum(1 for a in j if a >= self.n_true)
            valid = n_odd == (0 if spec.parity == "even" else 1)
            if not valid:
                index.append(-1)
                continue
            key = tuple(sorted(j)) if spec.symmetric else j
            index.append(free.setdefault(key, len(free)))
        n_free = len(free)
        set_(self, "n_free", n_free)
        ### -1 (no coefficient) points at the appended zero slot n_free.
        set_(
            self,
            "index",
            torch.tensor([i if i >= 0 else n_free for i in index], dtype=torch.long),
        )
        iso = _isotropic_tensors(r, spec.symmetric, spec.parity, self.dim)
        set_(self, "iso", iso)
        set_(self, "n_coeffs", n_free + len(iso))


def _assemble_output(coeffs, layout: _OutputLayout, true_vecs, pseudo_vecs, index):
    """Contract one output's coefficients ``(B, N, n_coeffs)`` with the token basis
    (rank 1 and above; a rank-0 output is its one coefficient).

    ``true_vecs`` is ``(B, N, m, d)``, ``pseudo_vecs`` ``(B, N, p, d)``; ``index``
    is ``layout.index`` on the coefficients' device. Returns ``(B, N, d, ..., d)``
    with ``rank`` trailing axes. The full coefficient tensor over ordered
    multi-indices is gathered from the free coefficients and contracted one index
    at a time, so the basis itself is never materialized (peak ``(B, N,
    n_vec**(r-1), d)``).
    """
    b, n = coeffs.shape[:2]
    r = layout.spec.rank
    u = (
        true_vecs
        if layout.spec.parity == "even"
        else torch.cat([true_vecs, pseudo_vecs], dim=2)
    )
    c_free = coeffs[..., : layout.n_free]
    c_full = torch.cat([c_free, c_free.new_zeros(b, n, 1)], dim=-1)[..., index]
    x = c_full  # (B, N, n_vec**r), indices (j1, ..., jr) row-major
    ### (explicit sizes: a -1 is ambiguous for an empty range of points)
    for _ in range(r):
        x = x.reshape(b, n, layout.n_vec, x.shape[-1] // layout.n_vec)  # split j1 from the rest
        x = torch.einsum("bnjR,bnja->bnRa", x, u)  # (B, N, rest, d)
        x = x.flatten(2)  # (rest, a1) flattened: j2 leads again
    out = x.reshape(b, n, *layout.spec.shape(layout.dim))
    if layout.iso:
        iso = torch.stack(layout.iso).to(device=out.device, dtype=out.dtype)
        c_iso = coeffs[..., layout.n_free :].to(out.dtype)
        out = out + torch.einsum("bnk,kR->bnR", c_iso, iso.flatten(1)).reshape(
            out.shape
        )
    return out


def _polar_complement(r_hat: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The component of ``v`` orthogonal to the unit vector ``r_hat``, with the
    sign convention of the polar basis vector ``e_theta`` of
    :func:`physicsnemo.nn.functional.equivariant_ops.spherical_basis` (``-v``
    projected off ``r_hat``; zero where ``r_hat`` is zero), in any dimension.
    Left unnormalized, so it vanishes smoothly where ``v`` is parallel to
    ``r_hat``. In three dimensions this is exactly that function's ``e_theta``."""
    e = vector_project(-v, r_hat)
    zero = torch.all(r_hat == 0.0, dim=-1, keepdim=True)
    return torch.where(zero, torch.zeros_like(e), e)


def _pseudovectors(primaries: list[torch.Tensor], dim: int) -> list[torch.Tensor]:
    """The pseudovector basis from a list of true vectors ``(..., dim)`` by the
    Levi-Civita construction: in three dimensions the pairwise cross products
    (in list order, ``i < j``), in two the perpendiculars ``J v = (-v_y, v_x)`` of
    each vector. A pseudovector in ``dim`` dimensions is the contraction of the
    Levi-Civita symbol with ``dim - 1`` vectors. Other dimensions return no
    pseudovectors; the constructor refuses odd-parity outputs there."""
    if dim == 3:
        return [
            torch.cross(primaries[i], primaries[j], dim=-1)
            for i in range(len(primaries))
            for j in range(i + 1, len(primaries))
        ]
    if dim == 2:
        return [torch.stack([-v[..., 1], v[..., 0]], dim=-1) for v in primaries]
    return []
