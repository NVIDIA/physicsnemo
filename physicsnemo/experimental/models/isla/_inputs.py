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

"""ISLA's input side: the field declarations and the invariants they contribute.

:func:`_declare` parses and checks one of the constructor's rank-spec declarations.
:class:`_InputSet` holds one token class's per-token fields and turns them into the
seed invariants, the block-level derived vectors and the block-level rank-2 tensors.
:class:`_BoundaryCondition` holds one named boundary condition's input parameters.
Private to :mod:`physicsnemo.experimental.models.isla`."""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from tensordict import PackLayout, TensorDict

from physicsnemo.mesh.fields import FieldSchema, FieldSchemaLike, RankSpec


def _declare(
    kind: str, specs: FieldSchemaLike | None, *, max_rank: int, allow_odd: bool
) -> FieldSchema:
    """Parse and check one of the constructor's field declarations (``None`` is
    the empty schema).

    The grammar is :meth:`~physicsnemo.mesh.fields.FieldSchema.parse` (the ISLA
    module docstring, *Rank specs*). ISLA's own limits on the parsed fields raise
    :class:`NotImplementedError`, as GLOBE does: a rank above ``max_rank``, and a
    pseudotensor unless ``allow_odd``.
    """
    out = FieldSchema({}) if specs is None else FieldSchema.parse(specs, label=kind)
    for name, spec in out.items():
        if spec.rank > max_rank:
            if kind == "output_schema":
                why = "the equivariant basis grows as m**rank"
            elif kind == "global_schema":
                why = (
                    "broadcast the tensor to the boundary cells and declare it as a "
                    "boundary input"
                )
            else:
                why = (
                    "see the ISLA module docstring for how higher ranks would enter"
                )
            raise NotImplementedError(
                f"{kind}[{name!r}]: rank {spec.rank} is not supported "
                f"(maximum {max_rank}; {why})"
            )
        if spec.parity == "odd" and not allow_odd:
            raise NotImplementedError(
                f"{kind}[{name!r}]: pseudotensor (parity='odd') inputs are not supported; "
                "input invariants are formed with true vectors only so that they are "
                "invariant under reflections"
            )
    return out


def _missing_fields(specs, data: TensorDict) -> list[str]:
    """The declared dotted names that are not tensor leaves of ``data``, sorted
    (a name whose prefix is a tensor rather than a group is missing too)."""
    leaves = set(data.keys(include_nested=True, leaves_only=True))
    return sorted(name for name in specs if FieldSchema.key(name) not in leaves)


def _schema_layout(fields) -> PackLayout:
    """A :class:`~tensordict.PackLayout` over ``fields`` (dotted name -> feature
    shape) in the given order, the schema's. :meth:`TensorDict.pack` alone orders
    the columns by key, and a trained weight depends on each column's position. Build
    it once, outside any compiled region: a cached layout is a constant under
    torch.compile. Its batch dimensions are those of the data."""
    template = TensorDict()
    for name, shape in fields.items():
        template.set(FieldSchema.key(name), torch.zeros(shape))
    _, layout = template.pack()  # the nesting; the column order follows below
    return dataclasses.replace(
        layout,
        keys=tuple(FieldSchema.key(name) for name in fields),
        shapes=tuple(fields.values()),
        device=None,
    )


def _pack(data: TensorDict, layout: PackLayout) -> torch.Tensor:
    """The leaves of ``data`` named by ``layout``, each flattened after the batch
    dimensions and concatenated in the layout's order: ``(*batch, layout.numel)``."""
    return data.select(*layout.keys).pack(layout=layout)[0]


### ---------------------------------------------------------------------------
### Input invariants
### ---------------------------------------------------------------------------


def _pair_indices(f: int, symmetric: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Index pairs ``(a, b)`` over ``f`` frame vectors: ``a <= b`` for a symmetric
    tensor (``a^T T b == b^T T a``), all ordered pairs otherwise."""
    if symmetric:
        return torch.triu_indices(f, f)
    grid = torch.cartesian_prod(torch.arange(f), torch.arange(f))
    return grid[:, 0], grid[:, 1]


@dataclass(frozen=True, eq=False)
class _InputSet:
    """The per-cell tensor inputs of one boundary condition and the invariants
    they contribute.

    With ``F = 1 + K`` frame vectors ``{n_hat, g_1..g_K}`` per token, ``P`` rank-1
    inputs and ``Q`` rank-2 inputs (``Q_s`` of them symmetric):

    * seed features (into the token's embedding): the rank-0 fields, then
      per rank-1 input its ``F`` frame cosines and its norm and the
      ``P(P-1)/2`` pairwise dot products of the rank-1 inputs, then per rank-2
      input the ``F(F+1)/2`` (symmetric) or ``F**2`` frame contractions
      ``a^T T b``, its trace and its Frobenius norm;
    * block vectors (contracted with the token-to-anchor direction in every
      slice layer): the ``P`` rank-1 inputs and, per rank-2 input, the ``F``
      derived true vectors ``T a`` (plus ``T^T a`` when not symmetric);
    * block tensors (``rho^T T rho`` in every slice layer): the ``Q`` rank-2 inputs.
    """

    specs: Mapping[str, RankSpec]
    n_frame: int
    dim: int
    n_scalar: int = field(init=False)
    rank1: list[str] = field(init=False)
    rank2: list[tuple[str, bool]] = field(init=False)
    n_features: int = field(init=False)
    n_block_vecs: int = field(init=False)
    n_block_tens: int = field(init=False)
    scalar_layout: PackLayout = field(init=False)
    vector_layout: PackLayout = field(init=False)
    tensor_layout: PackLayout = field(init=False)

    def __post_init__(self) -> None:
        f = int(self.n_frame)
        rank1 = [name for name, s in self.specs.items() if s.rank == 1]
        rank2 = [(name, s.symmetric) for name, s in self.specs.items() if s.rank == 2]
        n_scalar = sum(1 for s in self.specs.values() if s.rank == 0)
        p, q = len(rank1), len(rank2)
        n_pairs = sum(f * (f + 1) // 2 if sym else f * f for _, sym in rank2)
        set_ = object.__setattr__
        set_(self, "n_frame", f)
        set_(self, "dim", int(self.dim))
        set_(self, "n_scalar", n_scalar)
        set_(self, "rank1", rank1)
        set_(self, "rank2", rank2)
        set_(
            self,
            "n_features",
            n_scalar
            + p * f
            + p
            + p * (p - 1) // 2
            + n_pairs
            + 2 * q,
        )
        set_(self, "n_block_vecs", p + sum(f if sym else 2 * f for _, sym in rank2))
        set_(self, "n_block_tens", q)
        ### The fields of each rank as columns, in schema order (cached layouts).
        for rank, kind in enumerate(("scalar", "vector", "tensor")):
            fields = {k: (self.dim,) * rank for k, s in self.specs.items() if s.rank == rank}
            set_(self, f"{kind}_layout", _schema_layout(fields))

    def __bool__(self) -> bool:
        return bool(self.specs)

    def features(self, fields: TensorDict, frame):
        """Invariants of one token class.

        ``fields`` holds the declared fields over the ``(B, N)`` tokens; ``frame``
        is ``(B, N, F, d)``: the token's true frame vectors ``[n_hat, g_1..g_K]``
        in ``d`` spatial dimensions. Returns ``(seed (B, N, n_features), vecs (B,
        N, n_block_vecs, d), tens (B, N, n_block_tens, d, d))``.
        """
        ### The invariants and the derived vectors are geometry, so they stay in the
        ### input precision under autocast. The embedding that consumes the seed is
        ### autocast on its own.
        with torch.autocast(frame.device.type, enabled=False):
            return self._features(fields.to(frame.dtype), frame)

    def _features(self, fields, frame):
        b, n, _, d = frame.shape
        seed, vecs, tens = [], [], []
        if self.n_scalar:
            seed.append(_pack(fields, self.scalar_layout))  # (B, N, S)
        if self.rank1:
            v = _pack(fields, self.vector_layout).unflatten(-1, (-1, d))  # (B, N, P, d)
            seed.append(_contract_vectors(v, frame).flatten(2))  # (B, N, P*F)
            seed.append(v.norm(dim=-1))  # (B, N, P)
            p = v.shape[2]
            if p > 1:
                iu, ju = torch.triu_indices(p, p, offset=1, device=v.device)
                seed.append((v[:, :, iu] * v[:, :, ju]).sum(-1))  # (B, N, P(P-1)/2)
            vecs.append(v)
        if self.rank2:
            tens = _pack(fields, self.tensor_layout).unflatten(-1, (-1, d, d))  # (B, N, Q, d, d)
        for q, (_, symmetric) in enumerate(self.rank2):
            t = tens[:, :, q]
            ### a^T T b over the frame pairs, as (T b) . a with T b a derived vector.
            tb = torch.einsum("bnij,bnfj->bnfi", t, frame)  # (B, N, F, d): T a
            ia, ib = _pair_indices(self.n_frame, symmetric)
            seed.append(
                (frame[:, :, ia.to(t.device)] * tb[:, :, ib.to(t.device)]).sum(-1)
            )
            seed.append(torch.diagonal(t, dim1=-2, dim2=-1).sum(-1, keepdim=True))
            seed.append(t.flatten(2).norm(dim=-1, keepdim=True))
            vecs.append(tb)
            if not symmetric:
                vecs.append(torch.einsum("bnji,bnfj->bnfi", t, frame))  # T^T a
        seed_t = torch.cat(seed, -1) if seed else frame.new_zeros(b, n, 0)
        vecs_t = torch.cat(vecs, 2) if vecs else frame.new_zeros(b, n, 0, d)
        tens_t = tens if self.rank2 else frame.new_zeros(b, n, 0, d, d)
        return seed_t, vecs_t, tens_t


class _BoundaryCondition(nn.Module):
    """One named boundary condition's input parameters.

    ``seed`` is the condition's input layer, ``(hidden, 1 + n_seed)`` without a bias.
    It maps the condition's seed ``[1, n.g_k, field invariants, global scalars]`` to
    the token width. Its first column, on the constant 1, is the condition's learned
    offset, so two conditions' tokens differ even when neither declares a field.

    ``vector_map`` ``(n_vecs, P')`` and ``tensor_map`` ``(n_tens, Q)`` map the
    condition's own block vectors and rank-2 tensors (:class:`_InputSet`) linearly
    to the model's shared channels. A linear combination with scalar weights
    commutes with every orthogonal map, so the channels are true vectors and tensors
    like their inputs. Both maps start as the identity padded with zeros (the
    condition's ``j``-th vector is channel ``j``), and are ``None`` where the
    condition has no such input.
    """

    def __init__(self, inputs: _InputSet, n_seed: int, hidden: int, n_vecs: int, n_tens: int) -> None:
        super().__init__()
        self.inputs = inputs
        self.seed = nn.Linear(1 + n_seed, hidden, bias=False)
        p, q = inputs.n_block_vecs, inputs.n_block_tens
        self.vector_map = nn.Parameter(torch.eye(n_vecs, p)) if p else None
        self.tensor_map = nn.Parameter(torch.eye(n_tens, q)) if q else None


def _contract_vectors(x, y):
    """Dot products of per-token vector stacks ``x (B, N, P, d)`` and ``y (B, N, F, d)``
    -> ``(B, N, P, F)``. Broadcast multiply-and-sum, not einsum: einsum is autocast
    to bf16 and the geometry must stay in the input precision."""
    return (x[:, :, :, None, :] * y[:, :, None, :, :]).sum(-1)
