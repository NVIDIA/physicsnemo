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

"""Tests for systematic sampling proportional to size (``sampling.systematic_sampling``).

Cells are drawn with probability proportional to their measure (capped at one) by
systematic sampling along a line, and the kept cells carry Horvitz-Thompson
weights. These tests pin the exact counts, the inclusion probabilities, the
unbiasedness of the measure sums, the cells kept whole, the spatial spread, the
exact int64 line, the bitwise reproducibility, the invariance of the Mesh-level
draw under pose, numbering, precision and padding, its ties broken by index, and
one compiled graph for every cell count.

Several groups of cells (``_draw``) are composed as one line: per-group
probabilities, one canonical frame, each group in its own Morton order after the
previous ones, and one :func:`_systematic_sample` over the whole line.
"""

import pytest
import torch
from torch._inductor.utils import fresh_cache

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY
from physicsnemo.mesh.sampling import sample_cells_systematic
from physicsnemo.mesh.sampling.systematic_sampling import (
    _cumsum,
    _pps_inclusion_probabilities,
    _systematic_sample,
)
from physicsnemo.mesh.spatial._frame import canonical_frame, frame_coordinates
from physicsnemo.mesh.spatial.bvh import _compute_morton_codes

D = torch.float64
#: Cell counts whose shares under 200,000 samples are 31250, 58315.5, 52119,
#: 58315.5: at the start 0.5 every cell boundary of the third group (kept whole,
#: after a half-integer offset) is on a hit.
HALF_SIZES = [31_250, 113_946, 52_119, 151_699]
HALF_SHARES = [31_250, 58_315.5, 52_119, 58_315.5]


### Helpers ###


def _groups(sizes, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    measures = [torch.rand(n, generator=g, dtype=D) ** 3 + 1e-3 for n in sizes]
    centroids = [torch.randn(n, 3, generator=g, dtype=D) for n in sizes]
    return [a.to(device) for a in measures], [x.to(device) for x in centroids]


def _line(measures, centroids, shares):
    """The line of several groups of cells: each group's probabilities for its share,
    and every group in its Morton order in one canonical frame over all cells, one
    after the other. Returns the probabilities, the order, the measures and the
    cell count."""
    frame = canonical_frame(torch.cat(centroids), torch.cat(measures))
    pis, orders, offset = [], [], 0
    for a, x, share in zip(measures, centroids, shares):
        if a.shape[0] == 0:
            continue
        pis.append(_pps_inclusion_probabilities(a, share))
        code = _compute_morton_codes(frame_coordinates(x, frame), isotropic=True)
        orders.append(code.argsort(stable=True) + offset)
        offset += a.shape[0]
    return torch.cat(pis), torch.cat(orders), torch.cat(measures), offset


def _draw(measures, centroids, shares, budget, start):
    """Several groups of cells drawn as one line (:func:`_line`): one systematic
    sample of min(cells, budget), and the kept cells' weights."""
    pi, order, measure, total = _line(measures, centroids, shares)
    k = min(total, budget)
    kept = _systematic_sample(pi, k, start, order=order, max_n_samples=budget)
    return kept, measure[kept] / pi[kept]


def _per_group(kept, weights, sizes):
    bounds = torch.tensor(sizes).cumsum(0)[:-1]
    gid = torch.bucketize(kept.cpu(), bounds, right=True)
    counts = torch.zeros(len(sizes), dtype=torch.long).index_add(
        0, gid, torch.ones_like(gid)
    )
    sums = torch.zeros(len(sizes), dtype=weights.dtype).index_add(0, gid, weights.cpu())
    return counts, sums


def _whole_kept_once(kept, measures, shares):
    """Asserts the draw keeps distinct cells and every cell with pi = 1 exactly once."""
    pi = torch.cat(
        [_pps_inclusion_probabilities(a, s) for a, s in zip(measures, shares)]
    )
    assert torch.equal(kept, kept.unique())
    assert bool(torch.isin((pi.cpu() == 1).nonzero()[:, 0], kept.cpu()).all())


def _soup(n, seed=0, *, effective=True, shift=0.0, size=0.05):
    """A triangle soup of ``n`` small triangles centred on an anisotropic random
    cloud, with stored effective measures (``effective``) or without."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n, 3, generator=g, dtype=D) * torch.tensor([3.0, 2.0, 1.0])
    offsets = torch.randn(n, 3, 3, generator=g, dtype=D) * size
    points = (centers[:, None] + shift + offsets).reshape(-1, 3)
    cell_data = {}
    if effective:
        cell_data[EFFECTIVE_MEASURE_KEY] = torch.rand(n, generator=g, dtype=D) + 0.5
    return Mesh(
        points=points, cells=torch.arange(3 * n).reshape(n, 3), cell_data=cell_data
    )


def _orthogonal(seed, improper):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=D))
    q = q * torch.sign(torch.diagonal(r))
    if (torch.linalg.det(q) < 0) != improper:
        q[:, 0] = -q[:, 0]
    return q


### Inclusion probabilities ###


def test_inclusion_probabilities_are_capped_proportional_to_size(device):
    """pi = min(1, c a) with sum pi equal to n_samples: the largest cells are capped
    at one, the rest proportional to their measure; n_samples of at least the count
    keeps every cell."""
    g = torch.Generator().manual_seed(0)
    a = (torch.rand(1000, generator=g, dtype=D) ** 4).to(device)
    a[:5] = 50.0
    pi = _pps_inclusion_probabilities(a, torch.tensor(100.0, dtype=D, device=device))
    assert float(pi.sum()) == pytest.approx(100.0, rel=1e-12)
    capped = pi == 1
    assert bool(capped[:5].all()) and int(capped.sum()) == 5
    ratio = pi[~capped] / a[~capped]
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), rtol=1e-12)
    assert bool((ratio[0] * a[capped] >= 1).all())
    assert torch.equal(_pps_inclusion_probabilities(a, 1000.0), torch.ones_like(a))
    assert torch.equal(_pps_inclusion_probabilities(a, 100.0), pi)


def test_a_share_of_the_count_caps_every_cell(device):
    """n_samples of at least the cell count gives every pi = 1 on full-precision
    float64 measures."""
    for seed in range(4):
        g = torch.Generator().manual_seed(seed)
        for n in (2, 40, 3000, 52_119):
            a = (torch.rand(n, generator=g, dtype=D) ** 3 + 1e-3).to(device)
            for share in (n, n + 0.5):
                pi = _pps_inclusion_probabilities(a, float(share))
                assert torch.equal(pi, torch.ones_like(a)), (seed, n, share)


def test_equal_measures_with_a_share_of_the_count_are_kept_whole(device):
    """Equal measures (a uniform mesh) with n_samples of at least the count give every
    pi exactly 1, though the float tail sums of equal values can exceed their count
    times the value, which leaves a rank uncapped with a level below 1 / a (a = 0.1,
    n = 100: rank 9, and 91 cells a rounding below 1 from the tails alone)."""
    for value in (0.1, 0.3, 0.7, 3.3):
        for n in (7, 100, 3000, 300_001):
            a = torch.full((n,), value, dtype=D, device=device)
            for share in (n, n + 0.5):
                pi = _pps_inclusion_probabilities(a, float(share))
                assert torch.equal(pi, torch.ones_like(a)), (value, n, share)


def test_zero_measures_get_probability_zero(device):
    """Zero measures get pi = 0, and the probabilities sum to the smaller of the share
    and the positive count: [1, 0, 0] at 2 gives [1, 0, 0], and on random measures
    with exact zeros a share above the positive count keeps every positive cell."""
    one = torch.tensor([1.0, 0.0, 0.0], dtype=D, device=device)
    assert torch.equal(_pps_inclusion_probabilities(one, 2.0), one)
    g = torch.Generator().manual_seed(0)
    a = (torch.rand(1000, generator=g, dtype=D) ** 3).to(device)
    a[::4] = 0.0
    zero = a == 0
    pi = _pps_inclusion_probabilities(a, 300.0)
    assert bool((pi[zero] == 0).all())
    assert float(pi.sum()) == pytest.approx(300.0, rel=1e-12)
    pi = _pps_inclusion_probabilities(a, 800.0)
    assert bool((pi[zero] == 0).all()) and bool((pi[~zero] == 1).all())
    assert float(pi.sum()) == 750.0


def test_tiny_measures_keep_their_share(device):
    """100 cells of ordinary measure and 4,900 of 1e-17 in one group of share 260:
    the large cells are capped and the tiny ones share the rest, so the probabilities
    are nonnegative and sum to the share, the 300 kept cells are distinct, the second
    group is whole, and the first's Horvitz-Thompson sum is its measure."""
    g = torch.Generator().manual_seed(0)
    a = torch.cat(
        [
            torch.rand(100, generator=g, dtype=D) + 0.5,
            torch.full((4900,), 1e-17, dtype=D),
        ]
    )
    measures = [x.to(device) for x in (a, torch.rand(40, generator=g, dtype=D))]
    centroids = [torch.randn(n, 3, generator=g, dtype=D).to(device) for n in (5000, 40)]
    a = measures[0]
    pi = _pps_inclusion_probabilities(a, 260.0)
    assert bool((pi >= 0).all())
    assert abs(float(pi.sum()) - 260) < 1e-9
    kept, w = _draw(measures, centroids, [260, 40], 300, torch.tensor(0.5, dtype=D))
    counts, sums = _per_group(kept, w, [5000, 40])
    assert kept.unique().shape[0] == 300
    assert counts.tolist() == [260, 40]
    assert abs(float(sums[0] / a.sum()) - 1) < 1e-12


### The systematic line ###


@pytest.mark.parametrize(
    "budget, shares",
    [
        (1, [1 / 3] * 3),
        (77, [77 / 3] * 3),
        (300, [130, 40, 130]),
        (3539, [500, 40, 2999]),
        (3540, [500, 40, 3000]),
        (10**6, [500, 40, 3000]),
    ],
)
def test_the_count_is_exact(budget, shares, device):
    """Exactly min(total cells, budget) distinct cells, in increasing order."""
    sizes = [500, 40, 3000]
    measures, centroids = _groups(sizes, device=device)
    for u in (0.0, 0.37, 0.999):
        kept, w = _draw(measures, centroids, shares, budget, torch.tensor(u, dtype=D))
        assert kept.shape == w.shape == (min(sum(sizes), budget),)
        assert torch.equal(kept, kept.unique())


def test_horvitz_thompson_measure_sums_are_unbiased():
    """Over random starts each group's sum of weights is unbiased for its total
    measure (within four standard errors over 2,000 draws), the group kept whole
    (40 cells under a share of 100) exactly; every cell's inclusion frequency
    matches its probability."""
    sizes = [500, 40, 3000]
    shares = [130, 40, 130]
    measures, centroids = _groups(sizes)
    budget, draws = 300, 2_000
    pi, order, measure, _ = _line(measures, centroids, shares)
    total = torch.stack([a.sum() for a in measures])
    g = torch.Generator().manual_seed(0)
    sums, hits = [], torch.zeros(sum(sizes), dtype=D)
    for _ in range(draws):
        start = torch.rand((), dtype=D, generator=g)
        kept = _systematic_sample(pi, budget, start, order=order)
        w = measure[kept] / pi[kept]
        counts, s = _per_group(kept, w, sizes)
        assert int(counts[1]) == 40
        assert float(s[1]) == pytest.approx(float(total[1]), rel=1e-12)
        sums.append(s)
        hits[kept] += 1
    sums = torch.stack(sums)
    se = sums.std(0) / draws**0.5
    assert bool(((sums.mean(0) - total).abs() <= 4 * se + 1e-9 * total).all())
    ### Inclusion frequencies: a binomial count per cell, within five of its standard
    ### deviations plus two counts (the rare cells' counts are small integers).
    sd = (draws * pi * (1 - pi)).sqrt()
    assert bool(((hits - draws * pi).abs() <= 5 * sd + 2).all())


def test_the_sample_is_spread_over_the_surface(device):
    """Along the Morton order the kept cells are evenly spaced in cumulative
    probability, so a uniform measure on a grid gives samples in every octant of the
    box in proportion to its cells (to one sample), unlike an unstratified draw that
    could leave one empty."""
    g = torch.arange(16, dtype=D, device=device)
    xyz = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    a = torch.ones(xyz.shape[0], dtype=D, device=device)
    bits = torch.tensor([4, 2, 1], device=device)
    for u in (0.1, 0.5, 0.9):
        kept, _ = _draw([a], [xyz], [64], 64, torch.tensor(u, dtype=D))
        octant = ((xyz[kept] >= 8).long() * bits).sum(-1)
        counts = torch.bincount(octant, minlength=8)
        assert torch.equal(counts, torch.full((8,), 8, device=device))


def test_an_empty_group_takes_no_samples(device):
    """A group without cells is skipped."""
    measures, centroids = _groups([0, 50, 80], device=device)
    kept, w = _draw(measures, centroids, [0, 30, 30], 60, torch.tensor(0.5, dtype=D))
    assert kept.shape == (60,) and bool((kept < 130).all())
    counts, _ = _per_group(kept, w, [0, 50, 80])
    assert counts.tolist() == [0, 30, 30]


def test_a_hit_on_a_cell_boundary_takes_the_later_cell(device):
    """With a half-integer share before a group kept whole, every hit at the start
    0.5 lands exactly on a boundary of the whole group's cells, and on the boundary
    between the groups: each hit takes the later cell (half-open intervals), so the
    first group gets 300 of its share 300.5, the whole one every cell once, and the
    last 301."""
    g = torch.Generator().manual_seed(0)
    sizes = [1024, 50, 1024]
    shares = [300.5, 50, 300.5]
    ### equal measures: pi = 300.5 / 1024, on the line's grid, so the line is exact
    measures = [torch.ones(1024, dtype=D), torch.rand(50, generator=g, dtype=D) ** 3]
    measures.append(torch.rand(1024, generator=g, dtype=D) ** 3 + 1e-3)
    centroids = [torch.randn(n, 3, generator=g, dtype=D).to(device) for n in sizes]
    measures = [a.to(device) for a in measures]
    kept, w = _draw(measures, centroids, shares, 651, torch.tensor(0.5, dtype=D))
    counts, _ = _per_group(kept, w, sizes)
    assert counts.tolist() == [300, 50, 301]
    _whole_kept_once(kept, measures, shares)


@pytest.mark.parametrize(
    "u",
    [
        0.0,
        1e-12,
        0.5,
        float(torch.nextafter(torch.tensor(1.0, dtype=D), torch.tensor(0.0, dtype=D))),
    ],
)
def test_a_group_kept_whole_last_is_kept_at_every_start(u, device):
    """The line's end differs from k units by the rounding of the intervals; the start
    is clamped by that residual, so a group kept whole at the end of the line keeps
    every cell once at any start, including the ends of [0, 1). On full-precision
    float64 measures and on float32 stored ones, whose line (seed 1) is shorter than k
    units, so the start's upper clamp is needed."""
    sizes, shares = [5000, 40], [260, 40]
    full, centroids = _groups(sizes, seed=3, device=device)
    g = torch.Generator().manual_seed(1)
    stored = [
        (torch.rand(n, generator=g, dtype=D) + 1e-3).float().double().to(device)
        for n in sizes
    ]
    for measures in (full, stored):
        kept, w = _draw(measures, centroids, shares, 300, torch.tensor(u, dtype=D))
        counts, _ = _per_group(kept, w, sizes)
        assert counts.tolist() == [260, 40]
        _whole_kept_once(kept, measures, shares)


def test_the_integer_search_is_the_search_for_the_hits(device):
    """The search for the hit counts, (line - U - 1) >> Q, finds for every hit
    U + m 2**Q the first interval past it, as a direct search would; U the start on the
    grid, clamped by the line's residual. On random probabilities with some 0 and 1
    and on the half-integer construction, at starts 0, 0.5 and the largest below 1.
    With an order, the hits are those of the reordered line, in the cells' indices."""
    g = torch.Generator().manual_seed(1)
    pi = torch.rand(2000, generator=g, dtype=D) * 0.3
    pi[::7], pi[3::11] = 1.0, 0.0
    exact = torch.cat(
        [torch.full((1024,), 300.5 / 1024, dtype=D), torch.ones(50, dtype=D), pi[:1000]]
    )
    for p in (pi.to(device), exact.to(device)):
        budget = int(p.sum())
        k, Q = budget, 62 - budget.bit_length()
        line = (p * 2.0**Q).round().long().cumsum(0)
        residual = int(line[-1]) - k * 2**Q
        for u in (0.0, 0.5, 1.0 - 2.0**-53):
            U = min(max(int(u * 2**Q), residual, 0), 2**Q - 1, 2**Q - 1 + residual)
            hits = U + torch.arange(k, dtype=torch.int64, device=device) * 2**Q
            direct = torch.searchsorted(line, hits, right=True)
            assert int(direct.max()) < p.shape[0]
            position = _systematic_sample(p, k, torch.tensor(u, dtype=D))
            assert torch.equal(position, direct)
            assert torch.equal(
                _systematic_sample(p, k, u, max_n_samples=budget), direct
            )
        order = torch.randperm(p.shape[0], generator=g).to(device)
        kept = _systematic_sample(p, k, 0.5, order=order)
        assert torch.equal(kept, order[_systematic_sample(p[order], k, 0.5)].sort()[0])


def test_half_and_single_probabilities_draw_as_double(device):
    """float16 and float32 probabilities give bitwise the float64 draw: the line is
    scaled in float64, where pi 2**Q does not overflow."""
    g = torch.Generator().manual_seed(0)
    random = (torch.rand(1000, generator=g, dtype=D) * 0.3).half().double().to(device)
    halves = torch.full((4,), 0.5, dtype=D, device=device)
    for p, k in ((halves, 2), (random, int(random.sum()))):
        ref = _systematic_sample(p, k, 0.5)
        for dtype in (torch.float16, torch.float32):
            assert torch.equal(_systematic_sample(p.to(dtype), k, 0.5), ref), dtype


def test_the_grid_needs_a_python_int():
    """The line's grid is fixed by a Python int: another type for n_samples without
    max_n_samples, or n_samples above max_n_samples, is an error."""
    pi = torch.full((10,), 0.5, dtype=D)
    with pytest.raises(TypeError, match="max_n_samples"):
        _systematic_sample(pi, torch.tensor(5), 0.5)
    with pytest.raises(ValueError, match="exceeds"):
        _systematic_sample(pi, 5, 0.5, max_n_samples=4)


def test_the_float_scan_is_the_doubling_scan_bitwise(device):
    """The draw's float scan is the doubling (Kogge-Stone) scan, bitwise, which the
    sequential ``cumsum`` is not on 100 copies of 0.1; its integer scan is ``cumsum``."""

    def doubling(x):
        y, n, s = x.clone(), x.shape[0], 1
        while s < n:
            y = torch.cat([y[:s], y[s:] + y[: n - s]])
            s *= 2
        return y

    tenths = torch.full((100,), 0.1, dtype=D)
    assert not torch.equal(tenths.cumsum(0), doubling(tenths))  # on the CPU
    g = torch.Generator().manual_seed(0)
    for x in (tenths, torch.rand(1000, generator=g, dtype=D) ** 3):
        x = x.to(device)
        assert torch.equal(_cumsum(x), doubling(x))
    k = torch.randint(0, 2**40, (1000,), generator=g).to(device)
    assert torch.equal(_cumsum(k), k.cumsum(0))


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
def test_the_cumsum_op_passes_opcheck(device, dtype):
    """The draw's opaque scan (``physicsnemo::mesh_cumsum``) is a well-formed custom
    op, for the float64 and int64 scans."""
    x = torch.arange(1000, device=device).to(dtype) * 3 + 1
    torch.library.opcheck(_cumsum, (x,))


### Reproducibility ###


def _half_integer_case(device):
    return _groups(HALF_SIZES, device=device)


@pytest.mark.cuda
def test_the_draw_is_bitwise_reproducible_on_cuda():
    """Six draws on CUDA of 349,014 cells with full-precision float64 measures at the
    start 0.5, with half-integer shares before a group kept whole (hits exactly on
    its cell boundaries): bitwise the same kept cells and weights, no duplicate, every
    pi = 1 cell kept once."""
    measures, centroids = _half_integer_case("cuda")
    start = torch.tensor(0.5, dtype=D, device="cuda")
    runs = [_draw(measures, centroids, HALF_SHARES, 200_000, start) for _ in range(6)]
    for kept, w in runs:
        assert torch.equal(kept, runs[0][0])
        assert torch.equal(w.view(torch.int64), runs[0][1].view(torch.int64))
        _whole_kept_once(kept, measures, HALF_SHARES)


@pytest.mark.cuda
def test_the_draw_is_bitwise_the_same_on_cpu_and_cuda():
    """The same float64 inputs on CPU and on CUDA keep the same cells with bitwise the
    same weights."""
    start = torch.tensor(0.5, dtype=D)
    k0, w0 = _draw(*_half_integer_case("cpu"), HALF_SHARES, 200_000, start)
    k1, w1 = _draw(*_half_integer_case("cuda"), HALF_SHARES, 200_000, start.cuda())
    assert torch.equal(k1.cpu(), k0)
    assert torch.equal(w1.cpu().view(torch.int64), w0.view(torch.int64))


### The Mesh-level draw ###


def test_the_mesh_draw_is_the_composed_draw(device):
    """``sample_cells_systematic`` is the draw of one group of the mesh's cells, with
    the float64 measures and vertex-mean centroids, and weights in the points' dtype;
    the geometric measures when no effective measures are stored."""
    for effective in (True, False):
        mesh = _soup(3000, effective=effective).to(device)
        start = torch.tensor(0.3, dtype=D, device=device)
        kept, w = sample_cells_systematic(mesh, 700, start)
        if effective:
            measure = mesh.cell_data[EFFECTIVE_MEASURE_KEY]
        else:
            measure = mesh.cell_areas
        centroids = mesh.points[mesh.cells].mean(1)
        ref_kept, ref_w = _draw([measure], [centroids], [700], 700, start)
        assert torch.equal(kept, ref_kept)
        assert torch.equal(w.view(torch.int64), ref_w.view(torch.int64))


def test_the_mesh_draw_keeps_min_cells_and_samples(device):
    """min(n_cells, n_samples) cells; a mesh with at most n_samples cells is kept
    whole with its measures as weights; an empty mesh gives empty tensors; the start
    defaults to a draw from the generator."""
    mesh = _soup(500).to(device)
    for n_samples in (1, 77, 499, 500, 501, 10**6):
        kept, w = sample_cells_systematic(mesh, n_samples, 0.5)
        assert kept.shape == w.shape == (min(500, n_samples),)
        assert torch.equal(kept, kept.unique())
    kept, w = sample_cells_systematic(mesh, 500, 0.5)
    assert torch.equal(kept, torch.arange(500, device=device))
    assert torch.equal(w, mesh.cell_data[EFFECTIVE_MEASURE_KEY])
    draws = [
        sample_cells_systematic(
            mesh, 50, generator=torch.Generator(device=device).manual_seed(s)
        )
        for s in (0, 0, 1)
    ]
    assert torch.equal(draws[0][0], draws[1][0])
    assert not torch.equal(draws[0][0], draws[2][0])
    empty = Mesh(points=torch.randn(4, 3), cells=torch.zeros(0, 3, dtype=torch.long))
    kept, w = sample_cells_systematic(empty.to(device), 10, 0.5)
    assert kept.shape == w.shape == (0,) and w.dtype == torch.float32


def test_zero_area_cells_fill_the_count_at_weight_zero(device):
    """A mesh with degenerate (zero-area) cells and more samples than positive cells
    keeps min(n_cells, n_samples) distinct cells, every positive cell among them; the
    zero cells fill the count at weight 0, so the weights sum to the area."""
    mesh = _soup(500, effective=False)
    points = mesh.points.reshape(500, 3, 3).clone()
    points[::5] = points[::5, :1]  # every fifth triangle collapsed to a point
    mesh = Mesh(points=points.reshape(-1, 3), cells=mesh.cells).to(device)
    area = mesh.cell_areas
    zero = area == 0
    assert int(zero.sum()) == 100
    for u in (0.0, 0.37, 0.999):
        kept, w = sample_cells_systematic(mesh, 450, u)
        assert kept.shape == (450,)
        assert torch.equal(kept, kept.unique())
        assert bool(torch.isin((~zero).nonzero()[:, 0], kept).all())
        assert bool((w[zero[kept]] == 0).all())
        assert float(w.sum()) == pytest.approx(float(area.sum()), rel=1e-12)


def test_the_mesh_draw_validates_n_samples():
    mesh = _soup(10)
    with pytest.raises(ValueError, match="at least 1"):
        sample_cells_systematic(mesh, 0, 0.5)
    with pytest.raises(TypeError, match="Python int"):
        sample_cells_systematic(mesh, 5.0, 0.5)


def test_a_start_outside_the_unit_interval_is_an_error_or_clamped(device):
    """A Python start outside [0, 1), or NaN, is an error. A tensor start is not
    checked, which would sync with the host, but clamped: below 0 it draws as 0, at
    or above 1 as 1 does (the last start on the line's grid), and NaN as 0."""
    mesh = _soup(500).to(device)
    for u in (-0.1, 1.0, 1.5, 100.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match=r"start must be in \[0, 1\)"):
            sample_cells_systematic(mesh, 50, u)

    def kept(u):
        return sample_cells_systematic(
            mesh, 50, torch.tensor(u, dtype=D, device=device)
        )[0]

    assert not torch.equal(kept(0.0), kept(1.0))
    for u in (-0.1, -1e300, float("-inf"), float("nan")):
        assert torch.equal(kept(u), kept(0.0)), u
    for u in (1.5, 100.0, 1e300, float("inf")):
        assert torch.equal(kept(u), kept(1.0)), u


def test_the_mesh_draw_reads_stored_measures_through_cell_measures():
    """A stored measure is validated as ``cell_measures`` does, and the legacy
    ``_measure_weights`` field is an error rather than silently ignored."""
    mesh = _soup(10, effective=False)
    mesh.cell_data[EFFECTIVE_MEASURE_KEY] = torch.ones(10, dtype=torch.int64)
    with pytest.raises(TypeError, match="floating-point"):
        sample_cells_systematic(mesh, 5, 0.5)
    legacy = _soup(10, effective=False)
    legacy.cell_data["_measure_weights"] = torch.full((10,), 10.0, dtype=D)
    with pytest.raises(ValueError, match="Legacy _measure_weights"):
        sample_cells_systematic(legacy, 5, 0.5)


def test_the_draw_is_the_same_at_every_precision(device):
    """A float32 mesh and its float64 copy keep bitwise the same cells, with the same
    weights up to the final cast: the draw reads the measures and centroids in
    float64 from the points as given. The cells sit far from the origin, so float32
    centroids and areas would be rounded by many Morton grid units."""
    mesh32 = _soup(4000, effective=False, shift=300.0, size=1e-3).to(torch.float32)
    mesh32 = mesh32.to(device)
    mesh64 = Mesh(points=mesh32.points.double(), cells=mesh32.cells)
    for u in (0.5, 0.123):
        k32, w32 = sample_cells_systematic(mesh32, 500, u)
        k64, w64 = sample_cells_systematic(mesh64, 500, u)
        assert k32.shape == (500,)
        assert torch.equal(k32, k64)
        assert w32.dtype == torch.float32 and w64.dtype == D
        assert torch.equal(w32, w64.float())


@pytest.mark.parametrize("effective", [True, False])
def test_the_draw_is_independent_of_the_numbering(effective, device):
    """Renumbering the cells and the vertices of a float32 mesh keeps the same
    physical cells with bitwise the same weights."""
    mesh = _soup(2500, effective=effective, shift=40.0).to(torch.float32)
    g = torch.Generator().manual_seed(7)
    p = torch.randperm(mesh.n_points, generator=g)  # new vertex j is old vertex p[j]
    c = torch.randperm(mesh.n_cells, generator=g)  # new cell j is old cell c[j]
    cells = torch.empty_like(p).scatter_(0, p, torch.arange(mesh.n_points))
    renumbered = Mesh(
        points=mesh.points[p], cells=cells[mesh.cells[c]], cell_data=mesh.cell_data[c]
    )
    mesh, renumbered, c = mesh.to(device), renumbered.to(device), c.to(device)
    for u in (0.5, 0.123):
        k0, w0 = sample_cells_systematic(mesh, 700, u)
        k1, w1 = sample_cells_systematic(renumbered, 700, u)
        order = c[k1].argsort()
        assert torch.equal(c[k1][order], k0)
        assert torch.equal(w1[order], w0)


def test_the_draw_is_the_same_in_every_pose(device):
    """A rotation, reflection and translation of the mesh keep the same cells with
    the same weights: the Morton grid lies in the mesh's canonical frame, not in the
    lab axes, so the line along which the cells are drawn moves with the mesh."""
    mesh = _soup(3000).to(device)
    k0, w0 = sample_cells_systematic(mesh, 700, 0.5)
    for seed, improper in ((1, False), (2, True)):
        R = _orthogonal(seed, improper).to(device)
        shift = torch.tensor([3.0, -1.0, 2.0], dtype=D, device=device)
        moved = Mesh(
            points=mesh.points @ R.T + shift, cells=mesh.cells, cell_data=mesh.cell_data
        )
        k1, w1 = sample_cells_systematic(moved, 700, 0.5)
        assert torch.equal(k1, k0)
        torch.testing.assert_close(w1, w0, atol=0, rtol=1e-12)


def test_cells_on_one_centroid_are_laid_in_index_order(device):
    """Cells whose centroids share a Morton code lie along the line in index order
    (a stable sort): with one centroid for every cell, the draw is the systematic
    sample of the cells in index order, which an unstable sort would reorder."""
    s = torch.randperm(100, generator=torch.Generator().manual_seed(0)).to(D) + 1
    ### triangles (2s, 0, 0), (-s, s, 0), (-s, -s, 0): area 3 s**2, centroid exactly 0
    corners = torch.tensor([[2.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]])
    points = (s[:, None, None] * corners.to(D)).reshape(-1, 3)
    mesh = Mesh(points=points, cells=torch.arange(300).reshape(100, 3)).to(device)
    pi = _pps_inclusion_probabilities(mesh.cell_areas, 30.0)
    for u in (0.1, 0.5, 0.9):
        kept, _ = sample_cells_systematic(mesh, 30, u)
        assert torch.equal(kept, _systematic_sample(pi, 30, u))


@pytest.mark.parametrize("effective", [True, False])
def test_a_padded_mesh_keeps_the_same_cells(effective, device):
    """Mesh.pad's cells sit on the last point, here far outside the other cells, at
    zero measure: they move neither the canonical frame nor the Morton grid, so the
    padded mesh keeps the same cells as the mesh itself, while n_samples is at most
    the count of positive measures. The weights are bitwise the same: the doubling
    scan of the tails gives the same bits after leading zeros."""
    soup = _soup(3000, effective=effective)
    far = torch.tensor([[40.0, -30.0, 20.0]], dtype=D)  # used by no cell
    mesh = Mesh(
        points=torch.cat([soup.points, far]), cells=soup.cells, cell_data=soup.cell_data
    ).to(device)
    padded = mesh.pad(target_n_points=mesh.n_points + 5, target_n_cells=3500)
    for n_samples, u in ((700, 0.05), (700, 0.5), (3000, 0.37)):
        k0, w0 = sample_cells_systematic(mesh, n_samples, u)
        k1, w1 = sample_cells_systematic(padded, n_samples, u)
        assert torch.equal(k1, k0)
        assert torch.equal(w1, w0)


### torch.compile ###


def test_the_compiled_draw_is_the_eager_one_bitwise(device):
    """torch.compile(fullgraph=True, dynamic=True) of the draw gives bitwise the eager
    kept cells and weights at two cell counts, the second without a recompile (both
    below the stable sort's 32,767 size guard)."""

    def draw(mesh, u):
        return sample_cells_systematic(mesh, 2_001, u)

    torch._dynamo.reset()
    compiled = torch.compile(draw, fullgraph=True, dynamic=True)
    for i, n in enumerate([9_000, 12_000]):
        mesh = _soup(n, seed=i).to(device)
        start = torch.tensor(0.5, dtype=D, device=device)
        with torch._dynamo.config.patch(error_on_recompile=i > 0):
            kept, w = compiled(mesh, start)
        ref_kept, ref_w = draw(mesh, start)
        assert torch.equal(kept, ref_kept)
        assert torch.equal(w.view(torch.int64), ref_w.view(torch.int64))


#: Cell counts on both sides of n_samples = 300.
BUDGET_SIZES = [500, 700, 100, 2000, 150]


def _budget_draw(mesh, h, u):
    kept, w = sample_cells_systematic(mesh, 300, u)
    return (h[kept] * w[:, None]).sum(0), kept


def test_one_compiled_graph_for_every_cell_count():
    """Under torch.compile(fullgraph=True, dynamic=True) the draw is one graph whether
    the cells exceed n_samples or not, and matches eager."""
    torch._dynamo.reset()
    compiled = torch.compile(_budget_draw, fullgraph=True, dynamic=True)
    for i, n in enumerate(BUDGET_SIZES):
        mesh = _soup(n, seed=i, effective=False).to(torch.float32)
        h = torch.randn(n, 4, generator=torch.Generator().manual_seed(i))
        start = torch.tensor(0.5, dtype=D)
        with torch._dynamo.config.patch(error_on_recompile=i > 0):
            out, kept = compiled(mesh, h, start)
        ref, ref_kept = _budget_draw(mesh, h, start)
        assert kept.shape == (min(n, 300),)
        assert torch.equal(kept, ref_kept)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_a_warm_compile_cache_reuses_the_graph_across_the_budget():
    """A second run on the first run's Inductor cache (a cache hit on its first
    case) compiles no second graph for the cases on the other side of n_samples.

    Inductor's FX-graph cache replays a cached graph's shape guards on the new
    run's symbolic sizes, evaluating them as Python code in which a ``Min`` of
    the sizes is the builtin ``min``: its comparison would fix the sample count's
    ``Min(n_cells, n_samples)`` to one side of n_samples (a recompile on the other)."""
    with fresh_cache():
        for sizes in (BUDGET_SIZES[:1], BUDGET_SIZES):  # cold, then warm
            torch._dynamo.reset()
            compiled = torch.compile(_budget_draw, fullgraph=True, dynamic=True)
            for i, n in enumerate(sizes):
                mesh = _soup(n, seed=i, effective=False).to(torch.float32)
                h = torch.randn(n, 4, generator=torch.Generator().manual_seed(i))
                with torch._dynamo.config.patch(error_on_recompile=i > 0):
                    compiled(mesh, h, torch.tensor(0.5, dtype=D))


@pytest.mark.cuda
def test_the_draw_compiles_under_max_autotune_at_full_resolution():
    """``mode="max-autotune-no-cudagraphs"`` at 1.4 million cells: the draw's sums
    over every cell are long enough for Inductor's split scan, whose kernel failed
    on GB300 in this compile; the draw runs them eagerly (``_cumsum``). The compiled
    draw gives the eager one."""
    torch._dynamo.reset()

    def draw(mesh, u):
        return sample_cells_systematic(mesh, 200_000, u)

    compiled = torch.compile(
        draw, fullgraph=True, dynamic=True, mode="max-autotune-no-cudagraphs"
    )
    mesh = _soup(1_400_000).to("cuda")
    start = torch.tensor(0.5, dtype=D, device="cuda")
    kept, w = compiled(mesh, start)
    torch.cuda.synchronize()
    ref_kept, ref_w = draw(mesh, start)
    assert kept.shape == (200_000,)
    assert torch.equal(kept, ref_kept)
    torch.testing.assert_close(w, ref_w, atol=0, rtol=1e-12)
