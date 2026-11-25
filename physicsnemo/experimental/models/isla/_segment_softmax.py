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

r"""The tree read's softmax over each point's pair list (the reference read).

For every prediction point :math:`q` with pairs :math:`p` (a cell or a virtual cell
:math:`j_p` each) the read is

.. math::

    z_{ph} = \kappa_{ph} + \langle q_{qh}, k_{j_p h} \rangle / \sqrt{c} + \log m_p,
    \qquad
    \mathrm{read}_{qh} = \frac{\sum_p e^{z_{ph}} v_{j_p h}}{\sum_p e^{z_{ph}}},

with :math:`\kappa_{ph}` the pair's kernel term (its basis through the block's kernel
weights, a matmul outside this module) and :math:`\log m_p` the source's log measure.
:func:`segment_softmax_read` materializes it pair by pair, with several
``(pairs, heads)`` tensors per block. It is the read on CPU and in float64, and the
oracle of the fused read (:mod:`fused_read`), which serves CUDA and keeps nothing per
pair. The fused read shares the plan (:class:`PairPlan`) and its by-source work items.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch

from physicsnemo.mesh.spatial._ragged import _ragged_arange


def _offsets(sorted_idx: torch.Tensor, n: int) -> torch.Tensor:
    """Each of ``n`` segments' first row in the sorted index, and the row count."""
    bounds = torch.arange(n + 1, device=sorted_idx.device, dtype=sorted_idx.dtype)
    return torch.searchsorted(sorted_idx, bounds, out_int32=sorted_idx.dtype == torch.int32)


@dataclass
class PairPlan:
    """A pair list ``(qi, bj)`` grouped by point, with its regrouping by source.

    ``q_offsets`` ``(n_q + 1,)`` delimit each point's pairs; ``perm`` orders the pairs
    by source (stable) and ``b_offsets`` ``(n_b + 1,)`` delimit each source's pairs in
    that order; ``qi_by_b = qi[perm]``. The by-source half (``perm``, ``b_offsets``,
    ``qi_by_b``) is what the read's backward takes: it is built on first use
    (:attr:`by_source`), so a read without a backward never sorts the pairs.
    """

    qi: torch.Tensor
    bj: torch.Tensor
    n_q: int
    n_b: int
    q_offsets: torch.Tensor
    narrow: bool = False  # 32-bit indices with 64-bit offsets (build)

    @classmethod
    def build(
        cls, qi: torch.Tensor, bj: torch.Tensor, n_q: int, n_b: int, narrow: bool = False
    ) -> "PairPlan":
        """Index the pair list (``qi`` non-decreasing) by point and, on first use, by
        source. ``narrow`` (fewer than 2**31 pairs, points and sources) stores the
        indices ``qi``, ``bj``, ``perm`` and ``qi_by_b`` in int32, which the fused read
        widens on load, and the offsets in int64: half the plan's memory."""
        if narrow:
            qi, bj = qi.int(), bj.int()
        q_offsets = _offsets(qi, n_q)
        return cls(qi, bj, n_q, n_b, q_offsets.long() if narrow else q_offsets, narrow)

    @functools.cached_property
    def by_source(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(perm, b_offsets, qi_by_b)``."""
        ### A stable sort is unique, so 32-bit keys give the same permutation.
        perm = torch.argsort(self.bj.int() if self.n_b <= 2**31 else self.bj, stable=True)
        b_offsets, qi_by_b = _offsets(self.bj[perm], self.n_b), self.qi[perm]
        return (perm.int(), b_offsets.long(), qi_by_b) if self.narrow else (perm, b_offsets, qi_by_b)

    @property
    def perm(self) -> torch.Tensor:
        """The pairs in by-source order: a stable sort of the pairs by their source."""
        return self.by_source[0]

    @property
    def b_offsets(self) -> torch.Tensor:
        """Each source's first position in the by-source order, ``(n_b + 1,)``."""
        return self.by_source[1]

    @property
    def qi_by_b(self) -> torch.Tensor:
        """Each pair's point in the by-source order."""
        return self.by_source[2]


def _source_items(plan: PairPlan, chunk: int):
    """The by-source work items: (source, chunk index) for every chunk of ``chunk``
    pairs of every source's segment."""
    counts = plan.b_offsets[1:] - plan.b_offsets[:-1]
    n_chunks = (counts + (chunk - 1)) // chunk
    chunk_idx, src = _ragged_arange(torch.zeros_like(n_chunks), n_chunks)
    return src, chunk_idx


def _eager_segment_softmax_read(ker, logm, q, k, v, plan: PairPlan, scale: float):
    """The read pair by pair (:func:`segment_softmax_read`)."""
    acc = torch.promote_types(q.dtype, torch.float32)
    qi, bj = plan.qi.long(), plan.bj.long()
    ### Cast before the gathers, so that their backward adds each source's (and each
    ### point's) pair gradients in acc and rounds once, as the fused read does.
    score = (q.to(acc)[qi] * k.to(acc)[bj]).sum(-1)
    logit = ker.to(acc) + score * scale + logm.to(acc)[:, None]
    counts = plan.q_offsets[1:] - plan.q_offsets[:-1]
    seg_max = torch.segment_reduce(logit.detach(), "max", lengths=counts, axis=0, unsafe=True)
    e = torch.exp(logit - seg_max[qi])
    den = e.new_zeros(plan.n_q, e.shape[1]).index_add(0, qi, e)
    num = e.new_zeros(plan.n_q, e.shape[1], v.shape[2]).index_add(0, qi, e[..., None] * v.to(acc)[bj])
    return num / den.clamp_min(torch.finfo(acc).tiny)[..., None]


def segment_softmax_read(
    ker: torch.Tensor,
    logm: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: PairPlan,
    scale: float | None = None,
) -> torch.Tensor:
    """Each point's softmax-weighted read of its pairs' values, ``(n_q, H, C)`` in float32
    (float64 if the inputs are): ``ker`` ``(P, H)`` the pairs' kernel terms, ``logm``
    ``(P,)``, ``q`` ``(n_q, H, KC)``, ``k`` ``(n_b, H, KC)``, ``v`` ``(n_b, H, C)``;
    ``scale`` defaults to ``1 / sqrt(KC)``. Differentiable in ``ker``, ``logm``, ``q``,
    ``k`` and ``v``."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[2])
    return _eager_segment_softmax_read(ker, logm, q, k, v, plan, float(scale))
