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

"""The boundary-token sample of the interior task (``isla._sampling``).

The budget is water-filled over the named conditions, each condition's cells are
drawn proportional to their measure (capped at one) by systematic sampling along
the Morton order of their centroids, and the kept cells carry Horvitz-Thompson
weights. These tests pin the exact counts, the unbiasedness of the measure sums,
the conditions kept whole, the zero-measure fill, the spatial spread, and one
compiled graph for every cell count. The inclusion probabilities, the line and its
scan are the library's, tested in test/mesh/sampling/test_systematic_sampling.py.
"""

import pytest
import torch

from physicsnemo.experimental.models.isla._sampling import (
    _water_fill_host,
    sample_boundary_cells,
    water_fill,
)
from physicsnemo.mesh.sampling.systematic_sampling import _pps_inclusion_probabilities
from test.experimental.models.isla._forced import in_another_thread_while_compiling, ran
from test.experimental.models.isla._marks import DEVICES, requires_cuda

D = torch.float64
#: Cell counts of four conditions, whose water fill under a budget of 200,000 gives
#: shares 31250, 58315.5, 52119, 58315.5: at the start 0.5 every cell boundary of
#: the third condition (kept whole, after a half-integer offset) is on a hit.
HALF_SIZES = [31_250, 113_946, 52_119, 151_699]


def _conditions(sizes, seed=0):
    g = torch.Generator().manual_seed(seed)
    measures = [torch.rand(n, generator=g, dtype=D) ** 3 + 1e-3 for n in sizes]
    centroids = [torch.randn(n, 3, generator=g, dtype=D) for n in sizes]
    return measures, centroids


def _per_condition(kept, weights, sizes):
    bounds = torch.tensor(sizes).cumsum(0)[:-1]
    cid = torch.bucketize(kept, bounds, right=True)
    counts = torch.zeros(len(sizes), dtype=torch.long).index_add(
        0, cid, torch.ones_like(kept)
    )
    sums = torch.zeros(len(sizes), dtype=weights.dtype).index_add(0, cid, weights)
    return counts, sums


def test_water_fill_shares_the_budget():
    """Equal shares; a condition below its share is whole and its unused share goes
    to the others; the shares sum to the budget, or every condition is whole."""
    c = lambda *v: torch.tensor(v, dtype=D)  # noqa: E731
    assert torch.equal(water_fill(c(10, 1000, 30, 5000), 400), c(10, 180, 30, 180))
    assert torch.equal(water_fill(c(10**6), 200_000), c(200_000))
    assert torch.equal(
        water_fill(c(10**6, 10**6, 10**6, 10**6), 200_000), c(*[50_000] * 4)
    )
    assert torch.equal(
        water_fill(c(23_380, 54_853, 291_030, 54_621), 200_000),
        c(23_380, 54_853, 67_146, 54_621),
    )
    assert torch.equal(water_fill(c(10, 20), 400), c(10, 20))
    assert torch.equal(water_fill(c(0, 500), 100), c(0, 100))
    shares = water_fill(c(3, 7, 400, 401, 9_000), 1_000)
    assert float(shares.sum()) == pytest.approx(1_000) and torch.equal(
        shares[:2], c(3, 7)
    )


def _random_counts(seed):
    """Up to eight conditions of 0 to 10**6 cells and a budget up to their total."""
    g = torch.Generator().manual_seed(seed)
    n = int(torch.randint(1, 9, (), generator=g))
    counts = torch.randint(0, 10**6, (n,), generator=g)
    counts[torch.rand(n, generator=g) < 0.3] //= 1000  # some small conditions
    budget = int(torch.randint(1, int(counts.sum()) + 2, (), generator=g))
    return counts.tolist(), budget


@pytest.mark.parametrize(
    "counts, budget",
    [
        ([10, 1000, 30, 5000], 400),
        ([10**6], 200_000),
        ([0, 500], 100),
        ([0, 0, 7], 3),
        ([10, 20], 400),
        ([1, 1, 1], 2),
        ([3, 7, 400, 401, 9_000], 1_000),
        (HALF_SIZES, 200_000),
        ([999_999, 3, 3, 3, 3, 3, 3, 3], 200_001),
    ]
    + [_random_counts(seed) for seed in range(10)],
)
def test_the_host_water_fill_is_the_tensor_one_bitwise(counts, budget):
    """The draw's host water fill (eager) and its tensor one (which the draw runs under
    torch.compile) give the same float64 shares, bit for bit."""
    host = torch.tensor(_water_fill_host(counts, budget), dtype=D)
    tensor = water_fill(torch.tensor(counts, dtype=D), budget)
    assert torch.equal(host.view(torch.int64), tensor.view(torch.int64))


@pytest.mark.parametrize("budget", [1, 77, 300, 3539, 3540, 10**6])
def test_the_count_is_exact(budget):
    """Exactly min(total cells, budget) distinct cells, in increasing order."""
    sizes = [500, 40, 3000]
    measures, centroids = _conditions(sizes)
    for u in (0.0, 0.37, 0.999):
        kept, w = sample_boundary_cells(
            measures, centroids, budget, torch.tensor(u, dtype=D)
        )
        assert kept.shape == w.shape == (min(sum(sizes), budget),)
        assert torch.equal(kept, kept.unique())


### 2,000 draws: 5 s on CPU; 300 draws keep the property in the fast tier (the
### bounds are in standard errors and deviations, so they hold at any count)
@pytest.mark.parametrize("draws", [300, pytest.param(2_000, marks=pytest.mark.slow)])
def test_horvitz_thompson_measure_sums_are_unbiased(draws):
    """Over random starts each condition's sum of token weights is unbiased for its
    total measure (within four standard errors over the draws), the condition kept
    whole (40 cells under a share of 100) exactly; every cell's inclusion frequency
    matches its probability."""
    sizes = [500, 40, 3000]
    measures, centroids = _conditions(sizes)
    budget = 300
    shares = water_fill(torch.tensor(sizes, dtype=D), budget)
    pi = torch.cat(
        [_pps_inclusion_probabilities(a, s) for a, s in zip(measures, shares)]
    )
    total = torch.stack([a.sum() for a in measures])
    torch.manual_seed(0)
    sums, hits = [], torch.zeros(sum(sizes), dtype=D)
    for _ in range(draws):
        kept, w = sample_boundary_cells(
            measures, centroids, budget, torch.rand((), dtype=D)
        )
        counts, s = _per_condition(kept, w, sizes)
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


def test_the_sample_is_spread_over_the_surface():
    """Along the Morton order the kept cells are evenly spaced in cumulative
    probability, so a uniform measure on a grid gives tokens in every octant of the
    box in proportion to its cells (to one token), unlike an unstratified draw that
    could leave one empty."""
    g = torch.arange(16, dtype=D)
    xyz = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    a = torch.ones(xyz.shape[0], dtype=D)
    for u in (0.1, 0.5, 0.9):
        kept, _ = sample_boundary_cells([a], [xyz], 64, torch.tensor(u, dtype=D))
        octant = ((xyz[kept] >= 8).long() * torch.tensor([4, 2, 1])).sum(-1)
        assert torch.equal(torch.bincount(octant, minlength=8), torch.full((8,), 8))


def test_an_empty_condition_takes_no_tokens():
    """A condition without cells (a case without slip walls) is skipped."""
    measures, centroids = _conditions([0, 50, 80])
    kept, w = sample_boundary_cells(measures, centroids, 60, torch.tensor(0.5, dtype=D))
    assert kept.shape == (60,) and bool((kept < 130).all())
    counts, _ = _per_condition(kept, w, [0, 50, 80])
    assert counts.tolist() == [0, 30, 30]


def _whole_kept_once(kept, measures, budget):
    """Asserts the draw keeps distinct cells and every cell with pi = 1 exactly once."""
    shares = water_fill(torch.tensor([a.shape[0] for a in measures], dtype=D), budget)
    pi = torch.cat(
        [_pps_inclusion_probabilities(a, s) for a, s in zip(measures, shares)]
    )
    assert torch.equal(kept, kept.unique())
    assert bool(torch.isin((pi.cpu() == 1).nonzero()[:, 0], kept.cpu()).all())


def test_a_hit_on_a_cell_boundary_takes_the_later_cell():
    """With a half-integer share before a condition kept whole, every eval hit (start
    0.5) lands exactly on a boundary of the whole condition's cells, and on the
    boundary between the conditions: each hit takes the later cell (half-open
    intervals), so the first condition gets 300 of its share 300.5, the whole one
    every cell once, and the last 301."""
    g = torch.Generator().manual_seed(0)
    sizes = [1024, 50, 1024]
    ### equal measures: pi = 300.5 / 1024, on the line's grid, so the line is exact
    measures = [torch.ones(1024, dtype=D), torch.rand(50, generator=g, dtype=D) ** 3]
    measures.append(torch.rand(1024, generator=g, dtype=D) ** 3 + 1e-3)
    centroids = [torch.randn(n, 3, generator=g, dtype=D) for n in sizes]
    assert water_fill(torch.tensor(sizes, dtype=D), 651).tolist() == [300.5, 50, 300.5]
    kept, w = sample_boundary_cells(
        measures, centroids, 651, torch.tensor(0.5, dtype=D)
    )
    counts, _ = _per_condition(kept, w, sizes)
    assert counts.tolist() == [300, 50, 301]
    _whole_kept_once(kept, measures, 651)


@pytest.mark.parametrize(
    "u",
    [
        0.0,
        1e-12,
        0.5,
        float(torch.nextafter(torch.tensor(1.0, dtype=D), torch.tensor(0.0, dtype=D))),
    ],
)
def test_a_condition_kept_whole_last_is_kept_at_every_start(u):
    """The line's end differs from k units by the rounding of the intervals; the start
    is clamped by that residual, so a condition kept whole at the end of the line keeps
    every cell once at any start, including the ends of [0, 1). On full-precision
    float64 measures and on float32 stored ones, whose line (seed 1) is shorter than k
    units, so the start's upper clamp is needed."""
    sizes = [5000, 40]
    full, centroids = _conditions(sizes, seed=3)
    g = torch.Generator().manual_seed(1)
    stored = [
        (torch.rand(n, generator=g, dtype=D) + 1e-3).float().double() for n in sizes
    ]
    for measures in (full, stored):
        kept, w = sample_boundary_cells(
            measures, centroids, 300, torch.tensor(u, dtype=D)
        )
        counts, _ = _per_condition(kept, w, sizes)
        assert counts.tolist() == [260, 40]
        _whole_kept_once(kept, measures, 300)


def test_tiny_measures_keep_their_share():
    """100 cells of ordinary measure and 4,900 of 1e-17 in one condition of share 260:
    the large cells are capped and the tiny ones share the rest, so the probabilities
    are nonnegative and sum to the share, the 300 kept cells are distinct, the second
    condition is whole, and the first's Horvitz-Thompson sum is its measure."""
    g = torch.Generator().manual_seed(0)
    a = torch.cat(
        [
            torch.rand(100, generator=g, dtype=D) + 0.5,
            torch.full((4900,), 1e-17, dtype=D),
        ]
    )
    measures = [a, torch.rand(40, generator=g, dtype=D)]
    centroids = [torch.randn(n, 3, generator=g, dtype=D) for n in (5000, 40)]
    pi = _pps_inclusion_probabilities(a, torch.tensor(260.0, dtype=D))
    assert bool((pi >= 0).all())
    assert abs(float(pi.sum()) - 260) < 1e-9
    kept, w = sample_boundary_cells(
        measures, centroids, 300, torch.tensor(0.5, dtype=D)
    )
    counts, sums = _per_condition(kept, w, [5000, 40])
    assert kept.unique().shape[0] == 300
    assert counts.tolist() == [260, 40]
    assert abs(float(sums[0] / a.sum()) - 1) < 1e-12


def test_zero_measures_fill_a_condition_past_its_positive_cells():
    """A condition of 200 cells, 150 of them of measure exactly 0, and one of 1,000
    equal measures under a budget of 300 (shares 150, 150): the first condition's 50
    positive cells are all kept and its zero-measure cells fill the rest of its share
    at weight 0, so each condition keeps its share, no cell twice, every weight is
    finite, and each condition's Horvitz-Thompson sum is its measure."""
    g = torch.Generator().manual_seed(0)
    sizes = [200, 1000]
    a = torch.zeros(200, dtype=D)
    positive = torch.randperm(200, generator=g)[:50]
    a[positive] = torch.rand(50, generator=g, dtype=D) + 0.5
    measures = [a, torch.full((1000,), 0.3, dtype=D)]
    centroids = [torch.randn(n, 3, generator=g, dtype=D) for n in sizes]
    assert water_fill(torch.tensor(sizes, dtype=D), 300).tolist() == [150, 150]
    measure = torch.cat(measures)
    for u in (0.0, 0.5, 0.999):
        kept, w = sample_boundary_cells(
            measures, centroids, 300, torch.tensor(u, dtype=D)
        )
        counts, sums = _per_condition(kept, w, sizes)
        assert counts.tolist() == [150, 150], u
        assert torch.equal(kept, kept.unique())
        assert bool(torch.isin(positive, kept).all())
        assert bool(torch.isfinite(w).all()) and bool(
            (w[measure[kept] == 0] == 0).all()
        )
        for s, m in zip(sums, measures):
            assert float(s) == pytest.approx(float(m.sum()), rel=1e-12)


def _half_integer_case(device):
    g = torch.Generator().manual_seed(0)
    measures = [torch.rand(n, generator=g, dtype=D) ** 3 + 1e-3 for n in HALF_SIZES]
    centroids = [torch.randn(n, 3, generator=g, dtype=D) for n in HALF_SIZES]
    return [a.to(device) for a in measures], [x.to(device) for x in centroids]


@pytest.mark.cuda
@requires_cuda
def test_the_draw_is_bitwise_reproducible_on_cuda_and_the_cpu_one():
    """Three draws on CUDA of 349,014 cells with full-precision float64 measures at the
    eval start, with half-integer shares before a condition kept whole (hits exactly on
    its cell boundaries), each keep the CPU draw's cells with bitwise its weights, no
    duplicate, every pi = 1 cell kept once. The float64 scans of the line once varied
    run to run here."""
    assert water_fill(torch.tensor(HALF_SIZES, dtype=D), 200_000).tolist() == [
        31_250,
        58_315.5,
        52_119,
        58_315.5,
    ]
    start = torch.tensor(0.5, dtype=D)
    k0, w0 = sample_boundary_cells(*_half_integer_case("cpu"), 200_000, start)
    measures, centroids = _half_integer_case("cuda")
    for _ in range(3):
        kept, w = sample_boundary_cells(measures, centroids, 200_000, start.cuda())
        assert torch.equal(kept.cpu(), k0)
        assert torch.equal(w.cpu().view(torch.int64), w0.view(torch.int64))
        _whole_kept_once(kept, measures, 200_000)


@pytest.mark.slow  # 54 s on CPU: a compile over every cell count
@pytest.mark.parametrize("device", DEVICES)
def test_the_compiled_draw_is_the_eager_one_bitwise(device):
    """torch.compile(fullgraph=True, dynamic=True) of the draw gives bitwise the eager
    kept cells and weights at three cell counts, two above the budget and one below,
    with one graph for all three (every condition below the stable sort's 32,767 size
    guard)."""

    def draw(a1, a2, a3, x1, x2, x3, u):
        return sample_boundary_cells([a1, a2, a3], [x1, x2, x3], 2_001, u)

    torch._dynamo.reset()
    compiled = torch.compile(draw, fullgraph=True, dynamic=True)
    for i, sizes in enumerate(
        [(9_000, 400, 7_000), (12_000, 300, 6_500), (900, 300, 600)]
    ):
        measures, centroids = _conditions(sizes, seed=i)
        args = [t.to(device) for t in (*measures, *centroids)]
        args.append(torch.tensor(0.5, dtype=D, device=device))
        with torch._dynamo.config.patch(error_on_recompile=i > 0):
            kept, w = compiled(*args)
        ref_kept, ref_w = draw(*args)
        assert torch.equal(kept, ref_kept)
        assert torch.equal(w.view(torch.int64), ref_w.view(torch.int64))


def test_the_draw_stays_eager_in_another_thread_while_one_compiles():
    """Eager code that runs while another thread compiles (a data-loading thread, say)
    shares the budget on the host, as eager code does: only a trace takes the tensor
    water fill. Both give the same draw (bitwise, tested above)."""
    measures, centroids = _conditions((900, 300, 600))
    start = torch.tensor(0.5, dtype=D)
    with ran(_water_fill_host, water_fill) as r:
        kept, w = in_another_thread_while_compiling(
            lambda: sample_boundary_cells(measures, centroids, 1_000, start)
        )
    assert r.count(_water_fill_host) == 1 and r.count(water_fill) == 0
    ref_kept, ref_w = sample_boundary_cells(measures, centroids, 1_000, start)
    assert torch.equal(kept, ref_kept) and torch.equal(w, ref_w)


def test_the_compiled_draw_runs_its_sums_as_the_opaque_scan():
    """The draw traced with one graph (the graph Dynamo hands the backend, before
    Inductor) runs its sums as the custom op ``physicsnemo::mesh_cumsum`` (the
    library's systematic sampling, which the draw calls) and has no
    cumsum of its own: the op (a fixed-order scan for floats, ATen's for the int64
    line) keeps Inductor's split scan, whose kernel can fail at full resolution under
    a max-autotune compile, out of the compiled draw. This guards that on any device;
    test_isla_compile's max-autotune test runs the full resolution on CUDA."""
    graphs = []

    def capture(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(
        sample_boundary_cells, fullgraph=True, dynamic=True, backend=capture
    )
    measures, centroids = _conditions([40, 71])
    kept, _ = compiled(measures, centroids, 60, torch.tensor(0.3, dtype=D))
    assert kept.shape == (60,) and len(graphs) == 1
    nodes = graphs[0].graph.nodes
    scans = [n.target for n in nodes if "cumsum" in str(n.target)]
    assert scans and set(scans) == {torch.ops.physicsnemo.mesh_cumsum.default}, scans


def test_inference_is_deterministic_and_training_draws_from_the_global_rng():
    """An interior model that keeps 60 of 300 cells: in evaluation the draw is
    fixed, so two forwards after different global seeds give bitwise equal outputs;
    in training its start comes from the global torch RNG, so they differ."""
    from physicsnemo.experimental.models.isla import ISLA
    from test.experimental.models.isla._meshes import cloud, live, outputs, soup

    torch.manual_seed(0)
    m = live(
        ISLA(
            output_schema={"p": {"rank": 0}},
            boundary_schemas={"a": {}, "b": {}},
            interior=True,
            n_boundary_tokens=60,
            hidden=16,
            n_layers=1,
            n_slices=4,
        ).double()
    )
    meshes = {k: soup(*cloud(150, seed=i)[:3]) for i, k in enumerate("ab")}
    q = torch.randn(10, 3, dtype=D, generator=torch.Generator().manual_seed(5))

    def run(seed):
        torch.manual_seed(seed)
        with torch.no_grad():
            return outputs(m(boundary_meshes=meshes, prediction_points=q))["p"]

    m.eval()
    assert torch.equal(run(0), run(1))
    m.train()
    assert not torch.equal(run(0), run(1))
    assert torch.equal(run(3), run(3))


def _model_draw(model, meshes, q):
    """The kept cells and the tokens' final measures of one evaluation forward."""
    import physicsnemo.experimental.models.isla.model as isla_model

    seen, case_inputs, draw = (
        {},
        model._case_inputs,
        isla_model.sample_boundary_cells,
    )

    def spy_draw(*a, **k):
        seen["kept"], seen["raw"] = out = draw(*a, **k)
        return out

    def spy_inputs(*a, **k):
        c = case_inputs(*a, **k)
        seen["measure"] = c.boundary["measure"][0].clone()
        return c

    model._case_inputs = spy_inputs
    isla_model.sample_boundary_cells = spy_draw
    try:
        with torch.no_grad():
            model.eval()(boundary_meshes=meshes, prediction_points=q)
    finally:
        isla_model.sample_boundary_cells = draw
        del model._case_inputs
    return seen


def test_the_draw_is_the_same_at_every_model_precision():
    """On the same float32 meshes a float32 and a float64 model keep bitwise the same
    cells, and the tokens' measures agree up to the final cast: the draw reads the
    cells' measures and centroids in float64 from the input coordinates. The cells sit
    far from the origin, so float32 centroids and areas are rounded by many Morton
    grid units and a draw on them is a different sample."""
    import copy

    from physicsnemo.experimental.models.isla import ISLA
    from physicsnemo.mesh import Mesh
    from test.experimental.models.isla._meshes import cloud, soup

    torch.manual_seed(0)
    m32 = ISLA(
        output_schema={"p": {"rank": 0}},
        boundary_schemas={"a": {}, "b": {}},
        interior=True,
        n_boundary_tokens=500,
        hidden=16,
        n_layers=1,
        n_slices=4,
    ).float()
    m64 = copy.deepcopy(m32).double()
    shift = torch.tensor([300.0, -200.0, 100.0], dtype=D)
    meshes = {}
    for i, k in enumerate("ab"):
        pts, nrm, _, _ = cloud(1500, seed=i)
        meshes[k] = soup(pts * 1e-2 + shift, nrm, size=1e-3).to(torch.float32)
    q = (
        torch.randn(20, 3, generator=torch.Generator().manual_seed(5)) * 1e-2 + shift
    ).float()

    def fresh(dtype):
        return {
            k: Mesh(points=m.points.to(dtype), cells=m.cells) for k, m in meshes.items()
        }

    d32 = _model_draw(m32, fresh(torch.float32), q)
    d64 = _model_draw(m64, fresh(torch.float64), q.double())
    assert d32["kept"].shape == (500,)
    assert torch.equal(d32["kept"], d64["kept"])
    assert torch.equal(d32["raw"], d64["raw"]) and d32["raw"].dtype == torch.float64
    assert (
        d32["measure"].dtype == torch.float32 and d64["measure"].dtype == torch.float64
    )
    assert torch.equal(d32["measure"], d64["measure"].float())


def test_the_draw_is_independent_of_the_numbering():
    """Renumbering the cells and the vertices of float32 meshes keeps the same physical
    cells with bitwise the same weights."""
    from physicsnemo.mesh import Mesh
    from physicsnemo.mesh.sampling.systematic_sampling import _draw_inputs
    from test.experimental.models.isla._meshes import cloud, soup

    meshes = []
    for i, n in enumerate((900, 2500)):
        pts, nrm, w, _ = cloud(n, seed=i)
        meshes.append(soup(pts + 40.0, nrm, w if i else None).to(torch.float32))
    g = torch.Generator().manual_seed(7)
    renumbered, cell_perms = [], []
    for m in meshes:
        p = torch.randperm(m.n_points, generator=g)  # new vertex j is old vertex p[j]
        c = torch.randperm(m.n_cells, generator=g)  # new cell j is old cell c[j]
        cells = torch.empty_like(p).scatter_(0, p, torch.arange(m.n_points))[m.cells[c]]
        renumbered.append(
            Mesh(points=m.points[p], cells=cells, cell_data=m.cell_data[c])
        )
        cell_perms.append(c)
    old_index = torch.cat(
        [c + sum(m.n_cells for m in meshes[:i]) for i, c in enumerate(cell_perms)]
    )
    for u in (0.5, 0.123):
        start = torch.tensor(u, dtype=D)
        draws = []
        for ms in (meshes, renumbered):
            inputs = [_draw_inputs(m) for m in ms]
            draws.append(
                sample_boundary_cells(
                    [a for a, _ in inputs], [x for _, x in inputs], 700, start
                )
            )
        (k0, w0), (k1, w1) = draws
        order = old_index[k1].argsort()
        assert torch.equal(old_index[k1][order], k0)
        assert torch.equal(w1[order], w0)


def test_the_draw_is_the_same_in_every_pose():
    """A rotation, reflection and translation of the case keep the same cells with the
    same weights: the Morton grid lies in the boundary's canonical frame, not in the
    dataset's axes, so the line along which the cells are drawn moves with the case."""
    from physicsnemo.mesh.sampling.systematic_sampling import _draw_inputs
    from test.experimental.models.isla._meshes import cloud, orthogonal, soup, transform

    meshes = [soup(*cloud(n, seed=i)[:3]) for i, n in enumerate((900, 2500))]
    start = torch.tensor(0.5, dtype=D)

    def draw(ms):
        inputs = [_draw_inputs(m) for m in ms]
        return sample_boundary_cells(
            [a for a, _ in inputs], [x for _, x in inputs], 700, start
        )

    k0, w0 = draw(meshes)
    for seed, improper in ((1, False), (2, True)):
        R = orthogonal(3, seed, improper)
        shift = torch.tensor([3.0, -1.0, 2.0], dtype=D)
        k1, w1 = draw([transform(m, R, shift) for m in meshes])
        assert torch.equal(k1, k0)
        torch.testing.assert_close(w1, w0, atol=0, rtol=1e-12)
