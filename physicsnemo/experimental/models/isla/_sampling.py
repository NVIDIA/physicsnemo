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

r"""ISLA's boundary-token sample: systematic sampling proportional to size.

An interior model reads its boundary as a fixed budget of tokens, drawn inside
the forward pass from the full-resolution surfaces (:func:`sample_boundary_cells`):

- **Budget.** Water filling (:func:`water_fill`) splits the total ``budget``
  over the named conditions: equal shares, with a condition that has fewer cells
  than its share kept whole and its unused share given equally to the others,
  until stable. The shares are real numbers that sum to the budget.
- **Inclusion probabilities.** Within a condition of share :math:`b`, the cell
  :math:`i` of measure :math:`a_i` is drawn with probability
  :math:`\pi_i = \min(1, c\,a_i)`, :math:`c` the one constant with
  :math:`\sum_i \pi_i = b`. This is the capped fixed point of the iterative
  adjustment, found exactly from the sorted measures (the library's
  ``_pps_inclusion_probabilities`` in
  :mod:`physicsnemo.mesh.sampling.systematic_sampling`). A condition kept whole
  has every :math:`\pi_i = 1`. A share past the count of positive measures keeps
  every positive cell, and the cells of zero measure fill the rest of the share
  equally, at Horvitz-Thompson weight 0 (the library's ``_cell_lines``, per
  condition bit for bit the ``_cell_line`` that
  :func:`~physicsnemo.mesh.sampling.sample_cells_systematic` uses).
- **Draw.** Systematic sampling
  (:func:`~physicsnemo.mesh.sampling.systematic_sampling._systematic_sample`)
  lays the cells along a line, each an interval of length :math:`\pi_i`, and
  keeps the cells hit by the points :math:`u + m`, :math:`m = 0, \dots, k - 1`.
  Each condition's cells lie in the Morton (Z-order) order of their centroids in
  the boundary's canonical frame
  (:func:`~physicsnemo.mesh.spatial._frame.canonical_frame` over every cell), on a
  grid spanning the condition's cells of positive measure (``_cell_lines``). The
  intervals are half-open, so a hit exactly on a boundary takes the later cell.
  The count is exactly :math:`k = \min(\sum_c N_c, \text{budget})`, a function of
  the cell counts only (static shapes). Over a uniform start
  :math:`u \in [0, 1)`, every cell is kept with probability :math:`\pi_i` (up to
  the line's grid, below), and a cell with :math:`\pi_i = 1` exactly once. The
  kept cells are spread along the space-filling curve, a spatially stratified
  sample.
- **Weights.** A kept cell stands for :math:`a_i / \pi_i` of the boundary (the
  Horvitz-Thompson weight). So every measure-weighted sum over the tokens is
  unbiased for the sum over the full surface, and exact on a condition kept
  whole.
- **Precision.** The draw reads measures and centroids computed in float64 from
  the mesh's coordinates as given (the library's ``_draw_inputs``), whatever the
  model's precision. So the draw is a function of the input mesh only: a float32
  and a float64 run on the same inputs keep the same cells. The tail sums of the
  measures are a fixed-order float64 scan (the library's
  ``physicsnemo::mesh_cumsum``), the same bits on every device. The line is exact
  int64 arithmetic in units of :math:`2^{-Q}`, :math:`Q = 62 - b`, :math:`b` the
  bit length of the budget (a Python int), and each cell is an interval of
  :math:`\operatorname{round}(\pi_i 2^Q)` units. A cell with :math:`\pi_i = 1` is
  exactly :math:`2^Q` units, and the start is clamped by the line's rounding
  residual, so such a cell holds exactly one hit. The drawn inclusion probability
  is within :math:`(\lvert D \rvert + 1/2)\,2^{-Q}` of :math:`\pi_i`, for the line's
  residual :math:`D` in units (2.6e-11 measured for a million cells and 200,000
  samples). So the draw is bitwise the same run to run, and on CPU and CUDA given
  the same float64 inputs. The Morton order reads the canonical frame, which
  devices compute to roundoff, so a centroid on a grid line could still order
  differently.

Everything is tensor arithmetic without a data-dependent shape, so the draw
traces into one graph under ``torch.compile(fullgraph=True, dynamic=True)``.
It synchronizes with the host only in the canonical frame's 3 x 3 eigensolve
(:func:`~physicsnemo.mesh.spatial._frame.canonical_frame`).
"""

import torch
from jaxtyping import Float, Int

from physicsnemo.mesh.sampling.systematic_sampling import _cell_lines, _systematic_sample
from physicsnemo.mesh.spatial import canonical_frame
from physicsnemo.utils._tracing import is_tracing


def water_fill(
    counts: Float[torch.Tensor, " conditions"], budget: int
) -> Float[torch.Tensor, " conditions"]:
    """The budget's share of each condition, ``min(counts, level)``.

    ``level`` is the water level of the budget over the conditions: equal shares,
    a condition with at most its share of cells kept whole and its unused share
    split equally over the rest, iterated to the fixed point (at most one
    iteration per condition). The shares sum to ``budget`` when the conditions
    have more cells than that in total; otherwise every condition is whole.
    """
    n = counts.shape[0]
    level = counts.new_full((), budget / n)
    for _ in range(n):
        whole = counts <= level
        rest = (~whole).sum()
        level = torch.where(
            rest > 0,
            (budget - torch.where(whole, counts, 0).sum()) / rest.clamp_min(1),
            level,
        )
    return torch.minimum(counts, level)


def _water_fill_host(counts: list[int], budget: int) -> list[float]:
    """:func:`water_fill` of Python counts on the host, bit for bit: the counts and
    their sums are integers below 2**53, so each level is the same correctly rounded
    float64 quotient of the same two integers."""
    n = len(counts)
    level = budget / n
    for _ in range(n):
        kept = [c for c in counts if c <= level]
        if len(kept) < n:
            level = (budget - sum(kept)) / (n - len(kept))
    return [min(float(c), level) for c in counts]


def sample_boundary_cells(
    measures: list[Float[torch.Tensor, " cells"]],
    centroids: list[Float[torch.Tensor, "cells dim"]],
    budget: int,
    start: Float[torch.Tensor, ""],
) -> tuple[Int[torch.Tensor, " tokens"], Float[torch.Tensor, " tokens"]]:
    r"""The boundary token sample of one case: kept cells and their weights.

    ``measures`` and ``centroids`` hold each named condition's cells in float64
    (the library's ``_draw_inputs``), in the order their tokens are concatenated.
    ``budget`` is the total token budget, a Python int (it fixes the line's grid),
    and ``start`` the systematic draw's start :math:`u \in [0, 1)` (a float64 scalar).

    Returns the indices of the kept cells into the concatenation of the
    conditions' cells, in increasing order, ``(k,)`` with :math:`k = \min(\sum_c N_c,` ``budget``
    :math:`)`, and their Horvitz-Thompson measure weights :math:`a_i / \pi_i`
    ``(k,)``, in the dtype of the measures, with the float64 :math:`\pi_i`; they carry
    the measures' gradient, with :math:`\pi_i` held fixed. The draw is bitwise the
    same run to run and on every device for the same order (module docstring,
    **Precision**).
    """
    with torch.no_grad():
        if is_tracing():
            ### (tensor arithmetic: the counts are symbolic, and a host comparison
            ### would fix the graph to the conditions it keeps whole)
            counts = torch.stack([m.new_full((), m.shape[0], dtype=torch.float64) for m in measures])
            shares = water_fill(counts, budget)
        else:
            shares = _water_fill_host([m.shape[0] for m in measures], budget)
            shares = torch.stack([measures[0].new_full((), share, dtype=torch.float64) for share in shares])
        ### The Morton grid lies in the boundary's own frame over all its cells, so a
        ### rigid motion of the case leaves the line, and the draw, unchanged (up to
        ### roundoff and the frame's ties).
        a, x = torch.cat(measures).detach(), torch.cat(centroids).detach()
        frame = canonical_frame(x, a)
        ### The library's per-group rules, for the conditions with cells in one pass:
        ### the probabilities with the zero-measure fill, and the Morton order on the
        ### positive-measure grid in the frame.
        present = [i for i, m in enumerate(measures) if m.shape[0]]
        pi, order = _cell_lines(
            a.double(),
            x,
            torch.stack([shares[i] for i in present]),
            frame,
            [measures[i].shape[0] for i in present],
        )
        total = sum(m.shape[0] for m in measures)
        ### k = min(total, budget) in integer division: 1 // (total // budget + 1)
        ### is 1 when every cell is kept, else 0. Inductor's FX-graph cache replays
        ### a symbolic Min's guards with the builtin min, which fixes it to one side
        ### of the budget: the other side recompiles.
        k = budget + (total - budget) * (1 // (total // budget + 1))
        ### In the cells' own order, so a condition kept whole is its mesh as given.
        kept = _systematic_sample(pi, k, start, order=order, max_n_samples=budget)
        pi_kept = pi[kept]
    measure = torch.cat(measures)
    return kept, measure[kept] / pi_kept.to(measure.dtype)
