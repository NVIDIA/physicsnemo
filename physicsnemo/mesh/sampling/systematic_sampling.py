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

r"""Systematic sampling of mesh cells with probability proportional to size.

:func:`sample_cells_systematic` draws a fixed number of distinct cells of a mesh
and gives each a weight, so that weighted sums over the sample estimate sums over
the whole mesh:

- **Inclusion probabilities.** For an expected sample size :math:`b`, the cell
  :math:`i` of measure :math:`a_i` is drawn with probability
  :math:`\pi_i = \min(1, c\,a_i)`, with :math:`c` the one constant such that
  :math:`\sum_i \pi_i = b`. It is the capped fixed point of the iterative
  adjustment, found exactly from the sorted measures
  (:func:`_pps_inclusion_probabilities`). A size :math:`b` of at least the count
  of positive measures gives each of them :math:`\pi_i = 1`; a zero measure gets
  :math:`\pi_i = 0`.
- **Draw.** Systematic sampling (:func:`_systematic_sample`): the cells are laid
  along a line, each cell an interval of length :math:`\pi_i`, and the cells hit
  by the points :math:`u + m`, :math:`m = 0, \dots, k - 1`, are kept. The
  intervals are half-open, so a hit exactly on a boundary takes the later cell.
  :func:`sample_cells_systematic` lays the cells in the Morton (Z-order) order of
  their centroids in the mesh's canonical frame (the principal axes of the
  measure-weighted centroids, signed by their skewness), so the draw is the same
  in every pose of the mesh. The count is exactly
  :math:`k = \min(N, b)` for :math:`N` cells, a function of the cell count only
  (a static shape). Every cell is kept with probability :math:`\pi_i` over a
  uniform start :math:`u \in [0, 1)` (up to the line's grid, below), and a cell
  with :math:`\pi_i = 1` exactly once. The kept cells are spread along the
  space-filling curve, a spatially stratified sample.
- **Weights.** A kept cell stands for :math:`a_i / \pi_i` of the mesh (the
  Horvitz-Thompson weight), so every measure-weighted sum over the sample is
  unbiased for the sum over the whole mesh, and exact when every cell is kept.
- **Precision.** The measures and centroids are computed in float64 from the
  mesh's points as given, so a float32 and a float64 copy of the same mesh keep
  the same cells. The tail sums of the measures are a fixed-order float64 scan,
  the same bits on every device. The line is exact int64 arithmetic in units of
  :math:`2^{-Q}`, :math:`Q = 62 - B`, with :math:`B` the bit length of the
  largest sample size (a Python int). Each cell is an interval of
  :math:`\operatorname{round}(\pi_i 2^Q)` units, so a cell with
  :math:`\pi_i = 1` is exactly :math:`2^Q` units. The start is clamped by the
  line's rounding residual, so such a cell holds exactly one hit. The drawn
  inclusion probability is within :math:`(\lvert D \rvert + 1/2)\,2^{-Q}` of
  :math:`\pi_i`, for the line's residual :math:`D` in units: at most half a unit
  per cell plus the rounding of :math:`\sum_i \pi_i` (2.6e-11 measured for a
  million cells and 200,000 samples). So the draw is bitwise the same run to
  run, and on CPU and CUDA given the same float64 inputs. (The Morton order reads
  the canonical frame, which devices compute to roundoff, so a centroid on a grid
  line could still order differently.)

Everything is tensor arithmetic without a data-dependent shape, so the draw
traces into one graph under ``torch.compile(fullgraph=True, dynamic=True)``.
The graph serves every cell count, except that Inductor's stable sort recompiles
once for large meshes, and it is specialized to the sample size. The draw
synchronizes with the host only in the canonical frame's small eigensolve.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float, Int

from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY, cell_measures
from physicsnemo.mesh.geometry._cell_areas import compute_cell_areas
from physicsnemo.mesh.spatial._frame import canonical_frame, frame_coordinates
from physicsnemo.mesh.spatial.bvh import _compute_morton_codes
from physicsnemo.mesh.utilities._gather import gather_rows

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


@torch.library.custom_op("physicsnemo::mesh_cumsum", mutates_args=())
def _cumsum(x: torch.Tensor) -> torch.Tensor:
    r"""``x.cumsum(0)``, run under torch.compile as in eager: ATen's for integers, a
    doubling (Kogge-Stone) scan of elementwise adds for floats.

    The draw's sums run over every cell, an axis long enough for Inductor to emit
    a split scan; under ``mode="max-autotune-no-cudagraphs"`` that kernel failed on
    GB300 (an illegal memory access, or "operation not supported on global/shared
    address space"; torch 2.12 and 2.14.1). As an opaque op the scan runs eagerly,
    so the compiled draw is the eager one, bitwise. Integer sums are exact in any
    order. A float sum depends on its order, and ATen's CUDA float scan is not
    run-to-run deterministic; in the doubling scan every partial sum is a fixed
    binary tree of correctly rounded adds, so its bits are the same run to run and
    on every device, within about :math:`\log_2 n` roundings of the exact sum for
    nonnegative ``x``.
    """
    if not x.is_floating_point():
        return x.cumsum(0)
    n = x.shape[0]
    y, z = x.clone(), torch.empty_like(x)
    s = 1
    while s < n:
        z[:s] = y[:s]
        torch.add(y[s:], y[: n - s], out=z[s:])
        y, z = z, y
        s *= 2
    return y


@_cumsum.register_fake
def _(x):
    return torch.empty_like(x)


def _pps_inclusion_probabilities(
    measures: Float[torch.Tensor, " n_cells"],
    n_samples: float | Float[torch.Tensor, ""],
) -> Float[torch.Tensor, " n_cells"]:
    r"""Inclusion probabilities proportional to size, capped at one, summing to ``n_samples``.

    :math:`\pi_i = 1` for the :math:`m` largest measures and :math:`\pi_i = c\,a_i`
    for the rest, with :math:`c = (b - m) / \sum_{\text{rest}} a_i`. Here :math:`m`
    is the smallest count with :math:`c\,a_{(m+1)} < 1` (measures sorted in
    decreasing order), which is the fixed point of the iterative adjustment (cap,
    rescale the rest, repeat). The sort is stable, so ties resolve the same on
    every device, and the tail sums of the sorted measures are a fixed-order scan,
    so the probabilities are bitwise the same on every device. Zero measures get
    :math:`\pi_i = 0`, so the probabilities sum to the smaller of ``n_samples`` and
    the count of positive measures.

    Parameters
    ----------
    measures : torch.Tensor
        Nonnegative cell measures, shape ``(n_cells,)``, ``n_cells >= 1``. Float64
        is recommended. Their sums must stay in the dtype's normal range (in
        float64, about 1e-308 to 1e308).
    n_samples : float | torch.Tensor
        The expected sample size :math:`b`, the sum of the probabilities. It may be
        fractional. At least the count of positive measures gives every cell of
        positive measure :math:`\pi_i = 1` exactly.

    Returns
    -------
    torch.Tensor
        The inclusion probabilities, shape ``(n_cells,)``, in the dtype of
        ``measures``.

    Examples
    --------
    >>> import torch
    >>> measures = torch.tensor([4.0, 1.0, 1.0, 2.0], dtype=torch.float64)
    >>> _pps_inclusion_probabilities(measures, 2.0)
    tensor([1.0000, 0.2500, 0.2500, 0.5000], dtype=torch.float64)
    """
    if measures.ndim != 1:
        raise ValueError(f"measures must be 1-D, got shape {tuple(measures.shape)}")
    share = torch.as_tensor(n_samples, dtype=torch.float64, device=measures.device)
    n = measures.shape[0]
    a, order = measures.sort(descending=True, stable=True)
    tail = _cumsum(a.flip(0)).flip(0)  # sum of the measures from rank m on
    rank = torch.arange(n, device=a.device, dtype=a.dtype)
    uncapped = (share - rank) * a < tail
    ### The first uncapped rank as a min, not an argmax: Inductor splits a min over
    ### every cell across thread blocks, but runs an argmax in one.
    m = torch.where(uncapped, torch.arange(n, device=a.device), n).amin()
    ### (a gather, not tail[m]: indexing by a 0-d tensor would read it on the host)
    tail_m = tail.gather(0, m.clamp_max(n - 1).reshape(1))[0]
    c = (share - m) / tail_m.clamp_min(torch.finfo(a.dtype).tiny)
    pi = torch.where(rank < m, 1.0, (c * a).clamp(max=1.0))
    ### share >= n_pos keeps every positive cell, and no zero one: float tails can
    ### leave one a rounding below 1, and the search caps the zeros at 1.
    n_pos = (a > 0).sum()
    pi = torch.where(share >= n_pos, (a > 0).to(pi.dtype), pi)
    return torch.empty_like(pi).scatter_(0, order, pi)


def _systematic_sample(
    inclusion_probabilities: Float[torch.Tensor, " n_cells"],
    n_samples: int,
    start: float | Float[torch.Tensor, ""],
    *,
    order: Int[torch.Tensor, " n_cells"] | None = None,
    max_n_samples: int | None = None,
) -> Int[torch.Tensor, " n_samples"]:
    r"""Systematic sampling of the cells along a line, in exact int64 fixed point.

    The cells are laid end to end in ``order`` (default: as given), each an
    interval of length :math:`\pi_i`, and the cells hit by the points
    :math:`u + m`, :math:`m < k` (``n_samples``), are kept. The intervals are
    half-open: a hit on a boundary takes the later cell.

    The line is in units of :math:`2^{-Q}`, :math:`Q = 62 -`
    ``max_n_samples.bit_length()``: each interval is
    :math:`\operatorname{round}(\pi_i 2^Q)` units, and hit :math:`m` is the integer
    :math:`U + m 2^Q`, :math:`U = \lfloor u 2^Q \rfloor`. The line's end :math:`T`
    differs from :math:`k 2^Q` by the rounding residual :math:`D`; :math:`U` is
    clamped to :math:`[\max(0, D), \min(2^Q, 2^Q + D) - 1]`, so every hit lies on
    the line and every window of :math:`2^Q` units on it (an interval with
    :math:`\pi_i = 1`) holds exactly one hit, whenever :math:`|D| < 2^Q`. So a cell
    with :math:`\pi_i = 1` is kept exactly once.

    Parameters
    ----------
    inclusion_probabilities : torch.Tensor
        The probabilities :math:`\pi_i \in [0, 1]`, shape ``(n_cells,)``, in any
        floating dtype (scaled to the line in float64), summing to ``n_samples`` up
        to the line's grid (:func:`_pps_inclusion_probabilities`). Otherwise the hits
        past the end of the line repeat the last cell.
    n_samples : int
        The number of hits :math:`k`. It may be a symbolic size under
        ``torch.compile`` when ``max_n_samples`` is given.
    start : float | torch.Tensor
        The start :math:`u \in [0, 1)`, a float64 scalar. A start outside it is
        clamped into it, and NaN acts as 0.
    order : torch.Tensor, optional
        A permutation of the cells, shape ``(n_cells,)``, the order in which they
        are laid along the line, such as a space-filling curve's.
    max_n_samples : int, optional
        A Python int at least ``n_samples`` and the sum of the probabilities,
        which fixes the line's grid, so the line fits in int64. Defaults to
        ``n_samples``.

    Returns
    -------
    torch.Tensor
        The indices of the kept cells, shape ``(n_samples,)``, int64, distinct
        and increasing (when every probability is positive).

    Raises
    ------
    TypeError
        If the line's grid size (``max_n_samples``, or ``n_samples`` without it)
        is not a Python int.

    Examples
    --------
    >>> import torch
    >>> pi = torch.tensor([0.5, 0.5, 1.0, 0.25, 0.75], dtype=torch.float64)
    >>> _systematic_sample(pi, 3, 0.5)
    tensor([1, 2, 4])
    """
    budget = n_samples if max_n_samples is None else max_n_samples
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError(
            "The line's grid needs a Python int: pass max_n_samples as a Python "
            f"int when n_samples is symbolic (got {type(budget).__name__})."
        )
    if isinstance(n_samples, int) and n_samples > budget:
        raise ValueError(f"n_samples ({n_samples}) exceeds max_n_samples ({budget})")
    pi = inclusion_probabilities
    if order is not None:
        pi = pi[order]
    start = torch.as_tensor(start, dtype=torch.float64, device=pi.device)
    k = n_samples
    Q = 62 - budget.bit_length()
    line = _cumsum((pi.double() * 2.0**Q).round().to(torch.int64))
    ### D = T - k 2**Q in shifts by Q and small integers only: Inductor folds an
    ### int64 constant times an index or size into an int32 Triton expression, which
    ### Triton rejects.
    T = line[-1]
    D = (((T >> Q) - k) << Q) + (T - ((T >> Q) << Q))
    U = torch.floor(start.nan_to_num(0.0).clamp(0.0, 1.0) * 2.0**Q).to(torch.int64)
    U = torch.maximum(torch.minimum(U, D.clamp_max(0) + ((1 << Q) - 1)), D.clamp_min(0))
    ### line_j > U + m 2**Q  <=>  (line_j - U - 1) >> Q >= m, so the first interval
    ### past hit m is a search for m among these hit counts (nondecreasing), without
    ### the product m 2**Q.
    count = (line - U - 1) >> Q
    hit = torch.arange(k, device=pi.device, dtype=torch.int64)
    position = torch.searchsorted(count, hit).clamp_max(pi.shape[0] - 1)
    if order is None:
        return position
    return order[position].sort().values


def sample_cells_systematic(
    mesh: "Mesh",
    n_samples: int,
    start: float | Float[torch.Tensor, ""] | None = None,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Int[torch.Tensor, " n_kept"], Float[torch.Tensor, " n_kept"]]:
    r"""Draw cells of a mesh with probability proportional to measure, systematically.

    Keeps :math:`\min(N, b)` distinct cells of the :math:`N` cells, for
    ``n_samples`` :math:`b`. The cell :math:`i` is kept with probability
    :math:`\pi_i = \min(1, c\,a_i)`, with :math:`c` chosen so the probabilities
    sum to the sample size, and :math:`a_i` its effective measure
    (``cell_data["_effective_measure"]``, which must be nonnegative) when stored,
    else its geometric measure. The cells are laid along the Morton order of their
    centroids in the mesh's canonical frame, so the draw is the same in every pose
    and numbering of the mesh, and drawn by systematic sampling along that line.
    Cells of zero measure, such as those :meth:`~physicsnemo.mesh.mesh.Mesh.pad`
    adds, move neither the frame nor the Morton grid, so a padded mesh keeps the
    same cells while ``n_samples`` is at most its count of positive measures.
    Each kept cell carries its Horvitz-Thompson weight :math:`a_i / \pi_i`, so a
    weighted sum over the kept cells is unbiased for the sum over all cells. See
    the module docstring.

    The measures and centroids are computed in float64 from the points as given,
    so the kept cells and their weights are bitwise the same run to run and for
    float32 and float64 copies of the mesh. Across devices, and between compiled
    and eager code, they are bitwise the same given stored effective measures (up
    to the frame's roundoff, see the module docstring). Measures computed from the
    points round differently there, which moves the weights in their last bits. No
    gradients flow through the draw.

    Parameters
    ----------
    mesh : Mesh
        The mesh whose cells are drawn.
    n_samples : int
        The sample size :math:`b`, a Python int ``>= 1``. It fixes the line's grid,
        so ``torch.compile`` compiles one graph per value.
    start : float | torch.Tensor, optional
        The systematic start :math:`u \in [0, 1)`. ``None`` draws it uniformly from
        ``generator``. Under ``torch.compile``, pass a tensor: a Python float is a
        constant there, and each new value recompiles. A tensor start is not
        checked, which would synchronize with the host: outside :math:`[0, 1)` it
        is clamped into it, and NaN acts as 0.
    generator : torch.Generator, optional
        The random generator for the start, on the mesh's device. Defaults to the
        global generator.

    Returns
    -------
    cell_indices : torch.Tensor
        The kept cells, shape ``(min(n_cells, n_samples),)``, int64, increasing.
    weights : torch.Tensor
        Their Horvitz-Thompson weights :math:`a_i / \pi_i`, shape
        ``(min(n_cells, n_samples),)``, in the dtype of ``mesh.points``. Cells of
        zero measure are drawn only to fill the count (when ``n_samples`` exceeds
        the cells of positive measure), at weight 0, so weighted sums are
        unchanged.

    Raises
    ------
    TypeError
        If ``n_samples`` is not a Python int.
    ValueError
        If ``n_samples < 1``, or ``start`` is a Python number outside
        :math:`[0, 1)`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    >>> mesh = sphere_icosahedral.load(subdivisions=2)  # 320 triangles
    >>> cells, weights = sample_cells_systematic(mesh, 100, start=0.5)
    >>> cells.shape, weights.shape
    (torch.Size([100]), torch.Size([100]))
    >>> # At least one sample per cell keeps every cell once, at its own area
    >>> cells, weights = sample_cells_systematic(mesh, 1000, start=0.5)
    >>> torch.equal(cells, torch.arange(mesh.n_cells))
    True
    >>> torch.allclose(weights, mesh.cell_areas)
    True
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise TypeError(
            f"n_samples must be a Python int, got {type(n_samples).__name__}"
        )
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples=}")
    device = mesh.points.device
    if start is None:
        start = torch.rand((), dtype=torch.float64, device=device, generator=generator)
    elif not isinstance(start, torch.Tensor):
        if not 0 <= start < 1:  # also NaN
            raise ValueError(f"start must be in [0, 1), got {start=}")
        ### torch.full fills on the device; as_tensor would copy from the host and sync.
        start = torch.full((), start, dtype=torch.float64, device=device)
    start = torch.as_tensor(start, dtype=torch.float64, device=device)
    with torch.no_grad():
        ### Measures and centroids in float64 from the points as given, never the
        ### mesh's cached geometry, which is in the mesh's precision.
        verts = gather_rows(mesh.points.detach().double(), mesh.cells)
        if (
            EFFECTIVE_MEASURE_KEY in mesh.cell_data
            or "_measure_weights" in mesh.cell_data  # legacy: cell_measures raises
        ):
            measure = cell_measures(mesh)
        else:
            measure = compute_cell_areas(verts[:, 1:] - verts[:, :1])
        measure = measure.detach().double()
        centroids = verts.mean(dim=1)
        n_cells = measure.shape[0]
        if n_cells == 0:
            return (
                torch.empty(0, dtype=torch.int64, device=device),
                torch.empty(0, dtype=mesh.points.dtype, device=device),
            )
        count = measure.new_full((), n_cells, dtype=torch.float64)
        share = torch.minimum(count, count.new_full((), n_samples))
        ### The Morton grid lies in the mesh's own frame, so a rigid motion of the
        ### mesh leaves the line, and the draw, unchanged (up to roundoff and the
        ### frame's ties).
        frame = canonical_frame(centroids, measure)
        pi = _pps_inclusion_probabilities(measure, share)
        ### More samples than positive-measure cells keeps every positive cell, and
        ### the zero-measure cells fill the rest of the count equally, at
        ### Horvitz-Thompson weight 0.
        zero = measure == 0
        n_pos = n_cells - zero.sum()
        pi = torch.where(
            zero, ((share - n_pos) / (n_cells - n_pos).clamp_min(1)).clamp_min(0), pi
        )
        ### Morton order of the centroids on cubic cells, ties broken by index. The
        ### grid spans the cells of positive measure (all if none), and the others
        ### are clamped onto it, so padding leaves the order of the rest alone.
        xyz = frame_coordinates(centroids, frame)
        inside = (~zero | zero.all())[:, None]
        lo = torch.where(inside, xyz, torch.inf).amin(0)
        hi = torch.where(inside, xyz, -torch.inf).amax(0)
        code = _compute_morton_codes(xyz.clamp(lo, hi), isotropic=True)
        order = code.argsort(stable=True)
        ### k = min(n_cells, n_samples) in integer division: 1 // (n_cells //
        ### n_samples + 1) is 1 when every cell is kept, else 0. A symbolic Min
        ### would be fixed to one side of n_samples when Inductor's FX-graph cache
        ### replays the guards that hold it (evaluated with the builtin min), a
        ### recompile on the other.
        k = n_samples + (n_cells - n_samples) * (1 // (n_cells // n_samples + 1))
        kept = _systematic_sample(pi, k, start, order=order, max_n_samples=n_samples)
        weights = measure[kept] / pi[kept]
    return kept, weights.to(mesh.points.dtype)
