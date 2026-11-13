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

"""The fast search of the prediction points' boundary geometry against the reference.

``exact_boundary_geometry(kernel="fast")`` runs the Triton weld, canonical-cell and
weight kernels where the reference runs their torch code; every output of the search,
the winding number, the distance, the direction and their gradients must equal the
reference's bitwise, on closed, open, folded and repeated surfaces, at ties, on the
wall and in float32 and float64.
"""

import pytest
import torch

from physicsnemo.experimental.models.isla import _geometry as geometry
from physicsnemo.experimental.models.isla import _geometry_fast as geometry_fast
from physicsnemo.experimental.models.isla._geometry import exact_boundary_geometry
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.geometry import compute_cell_normals
from physicsnemo.mesh.primitives.surfaces import cube_surface
from physicsnemo.mesh.spatial._lbvh import build_lbvh_topology
from test.experimental.models.isla._forced import ran
from test.experimental.models.isla._geometry_cases import (
    FAST_DEVICES,
    KERNEL_DEVICES,
    _bitwise,
    _box_ties,
    _box_wall,
    _capped_cube,
    _circle,
    _face_queries,
    _fan,
    _finned_cube,
    _interior,
    _naca,
    _near_wall,
    _polygon,
    _rotation,
    _soup,
    _sphere,
    _torus,
)
from test.experimental.models.isla._marks import requires_cuda, requires_cuda_triton
from test.experimental.models.isla._meshes import orthogonal, outputs, transform

D = torch.float64

_DEVICES = ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_cuda])]


def _cases():
    """(name, surface, queries), float64: closed, open, folded, repeated surfaces and
    a soup, with on-wall, tied and far queries; in 3-D, then in 2-D (a NACA section
    with its sharp trailing edge, a square's ties and walls in a generic pose, a
    circle's vertices and edge midpoints on the wall, a square with a folded fin)."""
    g = torch.Generator().manual_seed(0)
    sphere = _sphere(3)
    edge_mid = sphere.points[sphere.cells[:, :2]].mean(1)
    yield (
        "sphere",
        sphere,
        torch.cat(
            [
                _near_wall(3, 1500, 1, band=0.5),
                sphere.points,
                edge_mid,
                sphere.cell_centroids,
            ]
        ),
    )
    tor = _torus(24, 12)
    t = torch.linspace(0, 6.28, 64, dtype=D)
    core = torch.stack([torch.cos(t), torch.sin(t), 0 * t], -1)
    yield "torus", tor, torch.cat([torch.randn(1500, 3, dtype=D, generator=g), core])
    yield (
        "torus repeated",
        Mesh(points=tor.points, cells=torch.cat([tor.cells, tor.cells[:40]])),
        core,
    )
    R, t = orthogonal(3, 7), torch.tensor([1.0, 2.0, 3.0], dtype=D) / 3
    for name, (mesh, q) in (
        ("box ties", _box_ties(3)[:2]),
        ("box wall", _box_wall(3)[:2]),
    ):
        yield name, transform(mesh, R, t), q @ R.T + t
    yield (
        "finned cube",
        _finned_cube(),
        torch.rand(1500, 3, dtype=D, generator=g) * 2.4 - 0.7,
    )
    cube = cube_surface.load().subdivide(levels=2, filter="linear")
    open_cube = Mesh(
        points=cube.points.double(), cells=cube.cells[cube.cell_centroids[:, 2] < 0.49]
    )
    yield "open cube", open_cube, torch.randn(1500, 3, dtype=D, generator=g) * 2
    tri = torch.rand(60, 3, 3, dtype=D, generator=g)
    soup = Mesh(points=tri.reshape(-1, 3), cells=torch.arange(180).reshape(60, 3))
    yield "soup", soup, torch.rand(1500, 3, dtype=D, generator=g) * 1.4 - 0.2
    flat = Mesh(
        points=cube.points.double(), cells=cube.cells[cube.cell_centroids[:, 2] < -0.49]
    )
    bumped = flat.points.clone()
    bumped[:, 2] += 1e-9 * torch.randn(bumped.shape[0], dtype=D, generator=g)
    q = torch.cat([flat.points, flat.cell_centroids]) + torch.tensor(
        [0.0, 0.0, 1e-3], dtype=D
    )
    yield "near-duplicate normals", Mesh(points=bumped, cells=flat.cells), q
    naca = _polygon(_naca(65))
    yield (
        "2d naca",
        naca,
        torch.cat(
            [
                torch.rand(1500, 2, dtype=D, generator=g)
                * torch.tensor([1.6, 0.6], dtype=D)
                - torch.tensor([0.3, 0.3], dtype=D),
                naca.points,
                naca.cell_centroids,
            ]
        ),
    )
    R2, t2 = orthogonal(2, 7), torch.tensor([1.0, 2.0], dtype=D) / 3
    for name, (mesh, q) in (
        ("2d box ties", _box_ties(2)[:2]),
        ("2d box wall", _box_wall(2)[:2]),
    ):
        yield name, transform(mesh, R2, t2), q @ R2.T + t2
    circle = _circle(48)
    yield (
        "2d circle on the wall",
        circle,
        torch.cat(
            [circle.points, circle.cell_centroids, _near_wall(2, 500, 5, band=0.3)]
        ),
    )
    square = _polygon(
        torch.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=D)
    )
    fin = torch.tensor([[0.5, 0.0], [1.5, 0.0]], dtype=D)
    sheet = torch.tensor([[4, 5]])
    folded = Mesh(
        points=torch.cat([square.points, fin]),
        cells=torch.cat([square.cells, sheet, sheet.flip(-1)]),
    )
    off = torch.tensor([1e-4, 1e-2, 0.1], dtype=D)[
        torch.randint(3, (600,), generator=g)
    ]
    side = torch.where(torch.rand(600, dtype=D, generator=g) < 0.5, 1.0, -1.0)
    along = 0.6 + 0.8 * torch.rand(600, dtype=D, generator=g)
    yield (
        "2d folded fin",
        folded,
        torch.cat(
            [
                torch.stack([along, side * off], -1),
                torch.rand(600, 2, dtype=D, generator=g) * 2.4 - 0.9,
            ]
        ),
    )


def _run(kernel, q, points, cells, monkeypatch):
    """d, u, their gradients in the queries and the vertices, every output of the
    search and the winding number, under ``kernel``."""
    seen = {}
    for name in ("_closest_candidates", "_closest_candidates_fast"):
        op = getattr(geometry, name)

        def spy(*args, _op=op):
            seen["search"] = _op(*args)
            return seen["search"]

        monkeypatch.setattr(geometry, name, spy)
    sign = geometry._winding_sign

    def spy_sign(sigma, w):
        seen["w"] = w
        return sign(sigma, w)

    monkeypatch.setattr(geometry, "_winding_sign", spy_sign)
    q, points = q.clone().requires_grad_(), points.clone().requires_grad_()
    d, u = exact_boundary_geometry(q, points, cells, kernel=kernel)
    (
        d.sum()
        + (
            u
            * torch.linspace(0, 1, u.numel(), dtype=u.dtype, device=u.device).reshape(
                u.shape
            )
        ).sum()
    ).backward()
    monkeypatch.undo()
    return [d, u, q.grad, points.grad, *seen["search"], seen["w"]]


@pytest.mark.parametrize("device", FAST_DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", [c[0] for c in _cases()])
def test_fast_search_equals_the_reference_bitwise(case, dtype, device, monkeypatch):
    """On every case, every output of the fast search (d, u, their gradients, the
    search's outputs and the winding number) is the reference's bitwise."""
    _, mesh, q = next(c for c in _cases() if c[0] == case)
    args = (q.to(device, dtype), mesh.points.to(device, dtype), mesh.cells.to(device))
    ### (one thread: the vertex gradient sums over the cells in a thread-dependent order)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        ref = _run("reference", *args, monkeypatch)
        fast = _run("fast", *args, monkeypatch)
    finally:
        torch.set_num_threads(threads)
    names = [
        "d",
        "u",
        "dq",
        "dp",
        "cell",
        "tied",
        "tie direction",
        "sign",
        "on wall",
        "weights",
        "w",
    ]
    for name, a, b in zip(names, ref, fast):
        assert _bitwise(a, b), name


@pytest.mark.parametrize("f", [1, 2, 3, 5, 17, 100, 1000])
def test_the_preorder_tree_is_the_library_lbvh_renumbered(f):
    """The search's tree (one kernel, nodes in depth-first preorder) is
    ``build_lbvh_topology``'s midpoint-split tree with one cell per leaf, its nodes
    renumbered in preorder: the same children, the node after each subtree, the same
    sorted cell at each leaf and the same inner nodes at each level."""
    g = torch.Generator().manual_seed(f)
    simplex = torch.rand(f, 3, 3, dtype=D, generator=g)
    (left, right, escape, levels, leaf_cell, _, _), _, _ = geometry._preorder_cell_tree(
        simplex
    )
    topo = build_lbvh_topology(f, 1, torch.device("cpu"))
    lib_left, lib_right = topo.left_child.tolist(), topo.right_child.tolist()
    pre, size, stack = {}, {}, [0]
    while stack:
        n = stack.pop()
        pre[n] = len(pre)
        if lib_left[n] >= 0:
            stack += [lib_right[n], lib_left[n]]
    for n in sorted(pre, key=pre.get, reverse=True):
        size[n] = 1 + (
            size[lib_left[n]] + size[lib_right[n]] if lib_left[n] >= 0 else 0
        )
    total = 2 * f - 1
    assert len(pre) == total == left.shape[0]
    for n, p in pre.items():
        inner = lib_left[n] >= 0
        assert int(left[p]) == (pre[lib_left[n]] if inner else -1)
        assert int(right[p]) == (pre[lib_right[n]] if inner else -1)
        assert int(escape[p]) == (p + size[n] if p + size[n] < total else -1)
        assert int(leaf_cell[p]) == (-1 if inner else int(topo.leaf_start[n]))
    assert [sorted(level.tolist()) for level in levels] == [
        sorted(pre[n] for n in level.tolist())
        for level in topo.internal_nodes_per_level
    ]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("built", [True, False], ids=["slabs", "no slabs"])
@pytest.mark.parametrize("f", [5120, 1], ids=["sphere", "one cell"])
def test_the_seed_is_a_nearest_cell_to_float32_roundoff(f, built, device):
    """The seed search reads float32 copies of the coordinates and skips nodes by
    their boxes and, without a pad, by their slabs: the float64 distance to the cell
    it returns is within ``8 eps32 (|x| + reach)`` of the least over all cells (a
    brute force). The copies move a query by at most ``|x| eps32 / 2`` and a vertex
    by ``reach eps32 / 2``, so the seed's distance and the least one by their sum
    each, 1 ``eps32 (|x| + reach)`` together; the float32 distance's arithmetic and
    the slabs' rounding take a few more. On a sphere of 5,120 cells and on one cell
    (the root is its leaf), for queries near the centre, near the wall and outside,
    with the tree and the slabs built as the search builds them, or with the slabs
    it takes above ``_SLAB_CELLS_PER_QUERY`` cells per query, which hold every point."""
    mesh = _sphere(4)
    tree, _, s = geometry._preorder_cell_tree(mesh.points[mesh.cells][:f].to(device))
    if built:
        slabs = geometry._node_slabs(s, torch.ones_like(s[:, 0, 0]), tree)
    else:
        slabs = geometry._no_slabs(tree, s)
    q = torch.cat(
        [
            _near_wall(3, 100, 1, band=0.2, radius=0.2),
            _near_wall(3, 100, 2, band=1e-3),
            _near_wall(3, 100, 3, band=0.3, radius=1.6),
        ]
    ).to(device)
    seed = geometry._nearest_cells(
        q.float(), s, tree, geometry._child_records(tree, slabs)
    )
    n = q.shape[0]
    a = q[:, None].expand(n, f, 3).reshape(-1, 3)
    b = s[None].expand(n, f, 3, 3).reshape(-1, 3, 3)
    w = geometry._closest_weights(a, b)
    dist = (a - (w[..., None] * b).sum(1)).norm(dim=-1).reshape(n, f)
    slack = (
        8 * torch.finfo(torch.float32).eps * (q.norm(dim=-1) + s.norm(dim=-1).amax())
    )
    assert bool((dist.gather(1, seed[:, None])[:, 0] <= dist.amin(1) + slack).all())


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", ["first leaf", "random"])
def test_the_search_does_not_depend_on_its_seed_cell(seed, dtype, device, monkeypatch):
    """The seed cell only sets the radius of the ball search, so with any cell in its
    place (the first leaf's for every query, or a random one for each) every output
    is bitwise the default's. The cube with its sides split and a zero-area cell
    beside it (a segment); queries at ties inside the cube, on its wall (vertices,
    edge midpoints, centroids), the same moved outward, where the closest point is a
    vertex or on an edge that cells share, and on and beside the segment."""
    mesh, on_wall, _ = _box_wall(3)
    seg = torch.tensor([[1.0, 1, 1], [1, 1, 1], [1.5, 1, 1]], dtype=D)
    points = torch.cat([mesh.points, seg])
    cells = torch.cat([mesh.cells, mesh.n_points + torch.arange(3)[None]])
    q = torch.cat(
        [
            _box_ties(3)[1],
            on_wall,
            1.5 * on_wall,
            seg,
            seg[1:].mean(0)[None],
            seg + 0.25,
        ]
    )
    args = (q.to(device, dtype), points.to(device, dtype), cells.to(device))
    default = _run("reference", *args, monkeypatch)
    calls = []

    def fake(query, simplex, tree, children):
        """As ``_nearest_cells``: a row of ``simplex`` (the leaves' order) per query."""
        calls.append(seed)
        if seed == "first leaf":
            return query.new_zeros(query.shape[0], dtype=torch.int64)
        g = torch.Generator().manual_seed(0)
        return torch.randint(simplex.shape[0], query.shape[:1], generator=g).to(
            query.device
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(geometry, "_nearest_cells", fake)
        patched = _run("reference", *args, monkeypatch)
    assert calls
    for a, b in zip(default, patched):
        assert _bitwise(a, b)


@pytest.mark.parametrize("device", _DEVICES)
def test_slabs_are_built_only_up_to_their_cells_per_query(device, monkeypatch):
    """The search builds a slab per node up to ``_SLAB_CELLS_PER_QUERY`` cells per
    query and none above it (a sphere of 5,120 cells: 160 queries build them, 159 do
    not), and every output is bitwise the same with them as without: on 159 queries
    near the wall and outside, with the limit as it is and with none."""
    mesh = _sphere(4)
    points, cells = mesh.points.to(device, D), mesh.cells.to(device)
    limit = geometry._SLAB_CELLS_PER_QUERY
    assert cells.shape[0] == 5120 == 160 * limit
    q = torch.cat(
        [_near_wall(3, 80, 2, band=1e-3), _near_wall(3, 80, 3, band=0.3, radius=1.6)]
    ).to(device, D)
    built, node_slabs = [], geometry._node_slabs

    def spy(*args):
        built.append(args[0].shape[0])
        return node_slabs(*args)

    ### (a context of its own: ``_run`` undoes the fixture's patches when it returns)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(geometry, "_node_slabs", spy)
        exact_boundary_geometry(q, points, cells, kernel="reference")
        assert built == [5120]
        without = _run("reference", q[:159], points, cells, monkeypatch)
        assert built == [5120]
        mp.setattr(geometry, "_SLAB_CELLS_PER_QUERY", 10**9)
        with_slabs = _run("reference", q[:159], points, cells, monkeypatch)
        assert built == [5120, 5120]
    for a, b in zip(without, with_slabs):
        assert _bitwise(a, b)


@pytest.mark.parametrize("device", _DEVICES)
def test_the_seed_search_opens_nan_bounds_with_slabs_as_without(device):
    """A cell with a NaN vertex gives its leaf a NaN box and every node above it a NaN
    slab direction. Warp's ``max`` and ``min`` skip a NaN, so the distance to a NaN
    bound counts as 0 and is never NaN: the seed traversal opens such a node, with the
    slabs as without them, and a NaN cell (a NaN distance) is never the nearest. So
    every finite query's seed is a finite cell at the least distance of the finite
    cells (the brute force's, within the float32 search's rounding), also with the
    slabs replaced by ones that bound nothing; a NaN query keeps cell 0."""
    mesh = _sphere(3)
    tree, _, s = geometry._preorder_cell_tree(mesh.points[mesh.cells].to(device))
    bad = torch.zeros(s.shape[0], dtype=torch.bool, device=device)
    bad[::97] = True
    s[bad, 1, 2] = torch.nan
    tree = (*tree[:5], *geometry._tree_boxes(s, tree[0], tree[1], tree[3], tree[4]))
    assert bool(tree[5].isnan().any()) and not bool(tree[5][0].isnan().any())
    q = _near_wall(3, 500, 3, band=0.6).to(device)
    q[::50, 0] = torch.nan
    finite = torch.isfinite(q).all(-1)
    slabs = geometry._node_slabs(s, torch.ones_like(s[:, 0, 0]), tree)
    assert bool(slabs[0].isnan().any()) and bool(slabs[0].isfinite().all(-1).any())

    def pairs(a, b):
        w = geometry._closest_weights(a, b)
        return (a - (w[..., None] * b).sum(1)).norm(dim=-1)

    n, f = q.shape[0], s.shape[0]
    brute = pairs(
        q[:, None].expand(n, f, 3).reshape(-1, 3),
        s[None].expand(n, f, 3, 3).reshape(-1, 3, 3),
    )
    brute = brute.reshape(n, f).nan_to_num(nan=torch.inf).amin(1)
    for bounds in (slabs, tuple(torch.zeros_like(t) for t in slabs)):
        seed = geometry._nearest_cells(
            q.float(), s, tree, geometry._child_records(tree, bounds)
        )
        assert not bool(bad[seed[finite]].any())
        assert bool((seed[~finite] == 0).all())
        torch.testing.assert_close(
            pairs(q, s[seed])[finite], brute[finite], rtol=1e-5, atol=2e-6
        )


@pytest.mark.parametrize("device", _DEVICES)
def test_the_ball_search_returns_every_cell_within_its_radius(device):
    """The ball search skips the inner nodes whose slabs lie beyond the padded radius,
    besides the nodes whose boxes do: its pairs still hold every cell whose float64
    distance (a brute force) is within the query's radius. A sphere, where a query far
    inside sees every patch at nearly its nearest distance and the patches' slabs
    decide, with thin cells (no certified plane) and zero-area cells (no float64
    distance in the kernel) beside it; queries near the centre, near the wall and
    outside, each with a radius just above its nearest distance."""
    mesh = _sphere(3)
    g = torch.Generator().manual_seed(0)
    a = 1.2 * torch.nn.functional.normalize(
        torch.randn(40, 3, dtype=D, generator=g), dim=-1
    )
    u, v = 0.1 * torch.randn(2, 40, 3, dtype=D, generator=g)
    thin = torch.stack([a, a + u, a + 0.5 * u + 1e-10 * v], 1)
    segments = torch.stack([a + v, a + v, a + v + u], 1)
    s = torch.cat([mesh.points[mesh.cells], thin, segments]).to(device)
    tree, order, s = geometry._preorder_cell_tree(s)
    flat = (order >= order.shape[0] - 40).to(device)
    assert (
        int((geometry._thin_bounds(s)[0] > torch.finfo(torch.float32).eps / 2).sum())
        >= 40
    )
    q = torch.cat(
        [
            _near_wall(3, 150, 1, band=0.2, radius=0.2),
            _near_wall(3, 150, 2, band=0.3),
            _near_wall(3, 100, 3, band=0.3, radius=1.6),
        ]
    ).to(device)
    n, f = q.shape[0], s.shape[0]
    w = geometry._closest_weights(
        q[:, None].expand(n, f, 3).reshape(-1, 3),
        s[None].expand(n, f, 3, 3).reshape(-1, 3, 3),
        flat[None].expand(n, f).reshape(-1),
    )
    dist = (q[:, None] - (w.reshape(n, f, 3, 1) * s[None]).sum(2)).norm(dim=-1)
    radius = dist.amin(1) * (1 + 1e-3) + 1e-9
    children = geometry._child_records(
        tree, geometry._node_slabs(s, torch.ones_like(s[:, 0, 0]), tree)
    )
    li, pk = geometry._ball_candidates(
        q, radius, s, tree, children, reach=s.norm(dim=-1).amax(), flat=flat
    )
    found = torch.zeros(n, f, dtype=torch.bool, device=device)
    found[li, pk] = True
    within = dist <= radius[:, None] * (1 - 1e-9)
    assert bool(within.any(1).all()) and bool((found | ~within).all())
    ### (a search: a small part of the pairs)
    assert int(found.sum()) < n * f // 20


def test_ball_search_in_blocks_of_rows_equals_one_launch(monkeypatch):
    """The ball search launched over blocks of a few query rows, as it is from 2**25 =
    33,554,432 queries on (Warp refuses 2**31 candidate slots), gives every output of
    one launch bitwise, its second pass included: a fan of 96 cells, 50 queries above
    its hub (96 candidates each, more than the 64 slots; 1,600 extra candidates fill
    the overflow list of 450) and 400 elsewhere."""
    q, points, cells = _fan()
    one = _run("reference", q, points, cells, monkeypatch)
    caps, launch = [], geometry.wp.launch

    def spy(k, dim, *args, **kwargs):
        if k is geometry._ball_candidates_kernel:
            caps.append((kwargs["inputs"][13], dim))  # (slots per row, rows)
        return launch(k, dim, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        ### Blocks of 64 rows in the first pass, of 42 in the second.
        mp.setattr(geometry, "WARP_MAX_ELEMENTS", 2**12 + 1)
        mp.setattr(geometry.wp, "launch", spy)
        blocks = _run("reference", q, points, cells, monkeypatch)
    assert caps == [(64, 64)] * 7 + [(64, 2), (96, 42), (96, 8)]
    for a, b in zip(one, blocks):
        assert torch.equal(a, b)


@pytest.mark.parametrize("kernel, device", KERNEL_DEVICES)
@pytest.mark.parametrize("case", ["list", "fallback"])
def test_ball_search_overflow_list_and_its_fallback_equal_the_default_slots(
    case, kernel, device, monkeypatch
):
    """With two candidate slots per query, the candidates past them go to the
    overflow list (``list``: a sphere's queries, whose few extra candidates fit in
    it, so one launch) or, when it is full, to a second pass (``fallback``: a fan's
    hub queries, 94 extra candidates each, against a list of 450 entries); every
    output is bitwise that of the default 64 slots."""
    if case == "list":
        q, sphere = _near_wall(3, 400, 15, band=0.5), _sphere(2)
        points, cells = sphere.points, sphere.cells
    else:
        q, points, cells = _fan()
    q, points, cells = q.to(device), points.to(device), cells.to(device)
    default = _run(kernel, q, points, cells, monkeypatch)
    caps, pairs = [], []
    launch, ball = geometry.wp.launch, geometry._ball_candidates

    def spy(k, dim, *args, **kwargs):
        if k is geometry._ball_candidates_kernel:
            caps.append(kwargs["inputs"][13])
        return launch(k, dim, *args, **kwargs)

    def spy_ball(*args, **kwargs):
        pairs.append(ball(*args, **kwargs))
        return pairs[-1]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(geometry, "_BALL_SLOTS", 2)
        mp.setattr(geometry.wp, "launch", spy)
        mp.setattr(geometry, "_ball_candidates", spy_ball)
        two = _run(kernel, q, points, cells, monkeypatch)
    per_query = torch.bincount(pairs[0][0], minlength=q.shape[0])
    extra = int((per_query - 2).clamp_min(0).sum())
    assert extra > 0
    if case == "list":
        assert extra <= q.shape[0] and caps == [2]
    else:
        assert extra > q.shape[0] and caps == [2, int(per_query.max())]
    for a, b in zip(default, two):
        assert _bitwise(a, b)


@pytest.mark.cuda
@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize("kernel", ["reference", "fast"])
def test_ball_search_past_2_31_slots(kernel):
    """2**25 + 64 queries, whose 64 candidate slots each pass Warp's 2**31-entry
    arrays: the last 64 queries' distances and directions equal those of the same
    queries alone, bitwise (each query's search, closest point and winding number are
    its own). Needs about 60 GiB of device memory (1.6-1.8 kB per query, measured on
    CPU at 2**20 queries): the 17.2 GB of slots, their masks and the candidates'
    closest points."""
    tail = 64
    ### A previous variant's cached blocks would count as used memory.
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 96 * 2**30:
        pytest.skip("needs 96 GiB of free device memory (it uses about 60 GiB)")
    sphere = _sphere(2)
    points, cells = sphere.points.cuda(), sphere.cells.cuda()
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.rand(2**25 + tail, 3, generator=g, device="cuda", dtype=D) * 3 - 1.5
    whole = [t[-tail:] for t in exact_boundary_geometry(q, points, cells, kernel)]
    alone = exact_boundary_geometry(q[-tail:].clone(), points, cells, kernel)
    for a, b in zip(whole, alone):
        assert torch.equal(a, b)


def test_geometry_kernel_auto_resolves_to_the_reference_off_cuda_and_honors_explicit_choices():
    """_geometry_kernel "auto" picks the fast search on CUDA only; on CPU it resolves
    to the reference, caches it, and a model forced to the fast search gives the same
    outputs bitwise."""
    sphere = _sphere(2)
    q = _near_wall(3, 200, 3, band=0.5)
    gd = {"g": torch.tensor([1.0, 0.0, 0.0], dtype=D)}
    auto = _interior(3)
    assert (
        auto._geometry_kernel == "auto"
        and auto._resolve_geometry_kernel(q) == "reference"
    )
    fast = _interior(3)
    fast._geometry_kernel = "fast"
    assert fast._resolve_geometry_kernel(q) == "fast"
    with torch.no_grad():
        ya = outputs(
            auto(boundary_meshes={"wall": sphere}, prediction_points=q, global_data=gd)
        )
        yf = outputs(
            fast(boundary_meshes={"wall": sphere}, prediction_points=q, global_data=gd)
        )
    assert all(torch.equal(ya[k], yf[k]) for k in ya)
    assert auto._geometry_kernel_cache[1] == "reference"
    if torch.cuda.is_available():
        assert auto._resolve_geometry_kernel(q.cuda()) == (
            "fast" if geometry_fast.HAS_TRITON else "reference"
        )


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("n", [1, 5, 1023, 1024, 1025, 70000])
def test_run_start_kernel_equals_the_torch_scan(n):
    """The run-start kernel gives each entry the index of the first entry of its run
    (a running maximum of the run starts), as the torch scan does, on one block and
    across block edges."""
    g = torch.Generator().manual_seed(n)
    new = torch.rand(n, generator=g) < torch.rand(1, generator=g)
    new[0] = True
    ref = torch.cummax(torch.where(new, torch.arange(n), 0), 0).values
    assert torch.equal(geometry_fast.run_start(new.cuda()).cpu(), ref)


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("d, k", [(2, 2), (3, 3), (3, 2), (2, 3)])
def test_canonical_cells_kernel_equals_the_torch_sort(d, k, dtype):
    """Ties, signed zeros, NaN and repeated vertices: the same order and parity."""
    g = torch.Generator().manual_seed(d * 10 + k)
    v = 700
    pts = torch.randint(0, 4, (v, d), generator=g).to(dtype) * 0.5
    pts[torch.rand(v, d, generator=g) < 0.03] = float("nan")
    pts[torch.rand(v, d, generator=g) < 0.05] = -0.0
    cells = torch.randint(0, v, (5000, k), generator=g)
    cells[:50, 1] = cells[:50, 0]
    ref = geometry._canonical_cells(pts, cells)
    out = geometry_fast.canonical_cells(
        pts.cuda(), cells.cuda(), geometry._canonical_cells
    )
    assert all(torch.equal(a.cpu(), b) and a.dtype == b.dtype for a, b in zip(out, ref))


def _guard_meshes():
    """(name, queries, points, cells) for the thin-cell guard, float64: caps
    and needles in a rotated, shifted cube (float32-derived and float64) and sliver
    soups at h = 1e-16 and 1e-13, random ordinary triangles with queries on their
    faces, edges and vertices, near and far, and cells of zero area (coincident and
    collinear vertices) beside others."""
    for kind in ("cap", "needle"):
        for rho in (1e-3, 1e-6):
            p, c, _ = _capped_cube(rho, kind)
            g = torch.Generator().manual_seed(int(rho * 1e7) + len(kind))
            q = torch.cat(
                [
                    _face_queries(g, p[c[0]], rho, 40),
                    torch.rand(80, 3, dtype=D, generator=g) * 1.4 - 0.2,
                ]
            )
            rot, shift = _rotation(7), torch.tensor([0.3, -0.2, 0.7], dtype=D)
            q, p = q @ rot.T + shift, p @ rot.T + shift
            yield f"{kind} {rho} float32", q.float().double(), p.float().double(), c
            yield f"{kind} {rho} float64", q, p, c
    for seed in (0, 1):
        for h in (1e-16, 1e-13):
            p, c, q = _soup(seed, h)
            yield f"soup {seed} {h}", q, p, c
    g = torch.Generator().manual_seed(5)
    tri = torch.randn(64, 3, 3, dtype=D, generator=g)
    b = torch.rand(64, 3, dtype=D, generator=g)
    t = torch.rand(64, 1, dtype=D, generator=g)
    q = torch.cat(
        [
            ((b / b.sum(1, keepdim=True))[:, :, None] * tri).sum(1),
            tri[:, 0] + t * (tri[:, 1] - tri[:, 0]),
            tri.reshape(-1, 3),
            torch.randn(64, 3, dtype=D, generator=g),
            1e3 * torch.randn(64, 3, dtype=D, generator=g),
        ]
    )
    cells = torch.arange(192).reshape(-1, 3)
    yield (
        "ordinary float32",
        q.float().double(),
        tri.reshape(-1, 3).float().double(),
        cells,
    )
    yield "ordinary float64", q, tri.reshape(-1, 3), cells
    p = torch.randn(12, 3, dtype=D, generator=g)
    p[3] = p[2]
    p[5] = 0.25 * p[4] + 0.75 * p[6]
    p[8] = 0.5 * (p[7] + p[9])
    c = torch.tensor(
        [[0, 1, 2], [2, 3, 1], [4, 5, 6], [7, 8, 9], [9, 10, 11], [1, 1, 1], [0, 4, 8]]
    )
    yield "zero area", torch.cat([p, torch.randn(60, 3, dtype=D, generator=g)]), p, c


def _guard_pairs(q, p, c, device):
    """Every query against every canonical cell: queries, cells, zero-area flags."""
    cells, _ = geometry._canonical_cells(p, geometry._weld(p)[c])
    s = p[cells]
    flat, any_flat = geometry._zero_area(s, cells)
    qi = torch.arange(q.shape[0]).repeat_interleave(cells.shape[0])
    ki = torch.arange(cells.shape[0]).repeat(q.shape[0])
    return q[qi].to(device), s[ki].to(device), flat[ki].to(device) if any_flat else None


def _pairs_of(case, device="cuda"):
    """:func:`_guard_pairs` of the set ``case`` of :func:`_guard_meshes`."""
    return _guard_pairs(*next(m for m in _guard_meshes() if m[0] == case)[1:], device)


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("switch", [1, float("inf")])
@pytest.mark.parametrize("case", [c[0] for c in _guard_meshes()] + ["empty"])
def test_guard_kernel_equals_the_torch_weights_bitwise(case, switch, monkeypatch):
    """The fused weights (Ericson's, the guard and the accurate point, the zero-area
    cells' segments) are the torch code's bitwise, with the switch as set and with it
    never taken; on the caps and needles the switch fires. A column-major ``q`` takes
    the torch code."""
    if case == "empty":
        q, s, flat = torch.zeros(0, 3, dtype=D), torch.zeros(0, 3, 3, dtype=D), None
        q, s = q.cuda(), s.cuda()
    else:
        q, s, flat = _pairs_of(case)
    monkeypatch.setattr(geometry, "_SWITCH_UNITS", switch)
    ref = geometry._closest_weights(q, s, flat)
    fast = geometry._closest_weights(q, s, flat, fast=True)
    assert torch.equal(ref.view(torch.int64), fast.view(torch.int64))
    q = q.T.contiguous().T
    ref = geometry._closest_weights(q, s, flat)
    fast = geometry._closest_weights(q, s, flat, fast=True)
    assert torch.equal(ref.view(torch.int64), fast.view(torch.int64))
    if switch == 1 and case.startswith(("cap", "needle")):
        monkeypatch.setattr(geometry, "_SWITCH_UNITS", float("inf"))
        assert bool((geometry._closest_weights(q, s, flat) != ref).any())


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("n", [1, 7, 255])
def test_guard_kernel_on_a_partial_block(n, monkeypatch):
    """A single partial block of pairs, the switched ones first, gives the torch
    weights bitwise."""
    q, s, flat = _pairs_of("cap 0.001 float64")
    ref = geometry._closest_weights(q, s, flat)
    monkeypatch.setattr(geometry, "_SWITCH_UNITS", float("inf"))
    switched = (geometry._closest_weights(q, s, flat) != ref).any(-1)
    monkeypatch.undo()
    rows = torch.cat([switched.nonzero()[:, 0], (~switched).nonzero()[:, 0]])[:n]
    q, s, flat = q[rows], s[rows], None if flat is None else flat[rows]
    ref = geometry._closest_weights(q, s, flat)
    fast = geometry._closest_weights(q, s, flat, fast=True)
    assert bool(switched[rows[0]])
    assert torch.equal(ref.view(torch.int64), fast.view(torch.int64))


@pytest.mark.cuda
@requires_cuda_triton
def test_guard_kernel_without_guarded_pairs(monkeypatch):
    """Pairs of which none is guarded (far queries of a well-shaped triangle) give the
    torch weights bitwise."""
    g = torch.Generator().manual_seed(3)
    q = (1e2 * torch.randn(300, 3, dtype=D, generator=g)).cuda()
    s = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=D).expand(300, 3, 3)
    s = s.contiguous().cuda()
    ### (the premise, read where the torch code runs: no pair takes the accurate point)
    with ran(geometry._accurate_weights) as r:
        ref = geometry._closest_weights(q, s)
    assert r.count(geometry._accurate_weights) == 0

    def unguarded(*args):
        raise AssertionError("a guarded pair")

    monkeypatch.setattr(geometry, "_accurate_weights", unguarded)
    fast = geometry._closest_weights(q, s, fast=True)
    assert torch.equal(ref.view(torch.int64), fast.view(torch.int64))


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("case", ["ordinary float64", "zero area"])
def test_guard_kernel_with_nan_coordinates(case):
    """NaN coordinates in queries and cells give the torch weights bitwise, NaN
    payloads included."""
    q, s, flat = _pairs_of(case)
    q[::97, 1] = float("nan")
    s[::89, 2, 0] = float("nan")
    ref = geometry._closest_weights(q, s, flat)
    fast = geometry._closest_weights(q, s, flat, fast=True)
    assert bool(ref.isnan().any())
    assert torch.equal(ref.view(torch.int64), fast.view(torch.int64))


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("columns", [False, True], ids=["rows", "columns"])
@pytest.mark.parametrize("case", [c[0] for c in _guard_meshes()])
def test_fast_search_equals_the_reference_on_thin_cells(case, columns):
    """The fast op's outputs are the reference op's, bitwise, on the guard's sets,
    with the queries row-major and column-major."""
    _, q, p, c = next(m for m in _guard_meshes() if m[0] == case)
    q, p = q.cuda(), p.cuda()
    if columns:
        q = q.T.contiguous().T
    cells, parity = geometry._canonical_cells(p, geometry._weld(p)[c.cuda()])
    s = p[cells]
    n64 = parity[:, None].double() * compute_cell_normals(s[:, 1:] - s[:, :1])
    ref = geometry._closest_candidates(q, p, cells, parity, n64)
    fast = geometry._closest_candidates_fast(q, p, cells, parity, n64)
    for a, b in zip(ref, fast):
        assert _bitwise(a, b)


def test_guard_kernel_falls_back_to_the_torch_code_off_cuda(monkeypatch):
    """Off CUDA the fast weights are the torch code's, from the torch code itself."""
    q, s, flat = _pairs_of("zero area", "cpu")
    calls = []
    closest = geometry._closest_weights

    def spy(*args):
        calls.append(args)
        return closest(*args)

    monkeypatch.setattr(geometry, "_closest_weights", spy)
    fast = closest(q, s, flat, fast=True)
    assert len(calls) == 1
    assert torch.equal(fast.view(torch.int64), closest(q, s, flat).view(torch.int64))
