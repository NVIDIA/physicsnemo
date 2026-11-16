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

"""The interior task's boundary forest builds one set of face trees per declared condition,
each in coordinates about its condition's own measure-weighted centroid, the frame
and its rotation bound staying the whole boundary's, so a node summarises one
condition.

On one condition the forest is bitwise the ungrouped one; with two, no tree or node
mixes conditions, every point's list still covers every cell once, each condition's
trees live about its own centroid, the fast build and walk are the reference's, a
small body far from the whole boundary's centroid keeps its tree membership under
a rigid pose (in the model's forest too, whose per-condition normal tolerance
decides the faces of normals that tie), and a condition of nearly equal variances
keeps the whole boundary's rotation bound.
"""

import math

import pytest
import torch

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.experimental.models.isla import _tree_read as tree_read
from physicsnemo.experimental.models.isla._tree_read import (
    BoundaryForest,
    canonical_frame,
    frame_coordinates,
)
from test.experimental.models.isla._meshes import cloud, points_mesh, soup, transform

F = torch.float32
D = torch.float64
_KERNELS = pytest.mark.parametrize("kernel", ["reference", "fast"])


def _equal(a, b) -> bool:
    """Bitwise equality of two forest fields (tensors, trees, lists, scalars)."""
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if hasattr(a, "to_dict"):
        da, db = a.to_dict(), b.to_dict()
        return da.keys() == db.keys() and all(_equal(da[k], db[k]) for k in da)
    return a == b


def _differing(a: BoundaryForest, b: BoundaryForest) -> list[str]:
    ### ``per_tree`` makes ``trees`` and ``tree_cells``, compared in its place.
    names = [k for k in a.__dataclass_fields__ if k != "per_tree"]
    return [
        k
        for k in [*names, "trees", "tree_cells"]
        if not _equal(getattr(a, k), getattr(b, k))
    ]


def _two_conditions(seed=0, n_body=120, n_far=200):
    """A 2D boundary of two conditions, as an airfoil case: a small closed body (an
    ellipse of chord 1 at the origin, condition 0) inside a large far field (a
    half-disc of radius 200 joined to a box to x = 200, sheared so that the whole
    boundary has no mirror symmetry, condition 1), whose centroid lies tens of chords
    from the body. Cells as (centroid, unit outward normal of the fluid side,
    length), float64."""
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 2 * math.pi, n_body + 1, dtype=D)[:-1] + 0.01 * torch.rand(
        n_body, generator=g, dtype=D
    )
    body = torch.stack(
        [0.5 * torch.cos(t), 0.06 * torch.sin(t) + 0.02 * torch.sin(2 * t)], -1
    )
    s = torch.linspace(math.pi / 2, 3 * math.pi / 2, n_far // 2 + 1, dtype=D)
    q = n_far // 4
    far = torch.cat(
        [
            torch.stack([200 * torch.cos(s), 200 * torch.sin(s)], -1),  # the inlet arc, x <= 0
            torch.stack([torch.linspace(0, 200, q + 1, dtype=D)[1:], torch.full((q,), -200.0, dtype=D)], -1),
            torch.stack([torch.full((q,), 200.0, dtype=D), torch.linspace(-200, 200, q + 1, dtype=D)[1:]], -1),
            torch.stack([torch.linspace(200, 0, q + 1, dtype=D)[1:-1], torch.full((q - 1,), 200.0, dtype=D)], -1),
        ]
    )  # fmt: skip
    far = far + torch.stack([torch.zeros_like(far[:, 0]), 0.15 * far[:, 0]], -1)

    def cells(v, outward):
        e = torch.roll(v, -1, 0) - v
        length = e.norm(dim=-1)
        n = torch.stack([e[:, 1], -e[:, 0]], -1) / length[:, None]
        mid = v + 0.5 * e
        flip = ((n * mid).sum(-1) < 0) != outward
        return mid, torch.where(flip[:, None], -n, n), length

    pb, nb, wb = cells(body, True)  # the fluid lies outside the body
    pf, nf, wf = cells(far, False)  # and inside the far field
    groups = torch.cat(
        [torch.zeros(len(pb), dtype=torch.long), torch.ones(len(pf), dtype=torch.long)]
    )
    return torch.cat([pb, pf]), torch.cat([nb, nf]), torch.cat([wb, wf]), groups


def _three_conditions_3d(seed=0):
    """A 3D boundary of three conditions: a small body (condition 0)
    above a floor (condition 2) inside a large box (condition 1, the condition
    numbers out of the cells' order)."""
    body, nb, wb, _ = cloud(300, d=3, seed=seed, scale=(0.4, 0.2, 0.1))
    box, nx, wx, _ = cloud(200, d=3, seed=seed + 1, scale=(40.0, 25.0, 15.0))
    g = torch.Generator().manual_seed(seed + 2)
    floor = torch.cat(
        [
            (torch.rand(150, 2, generator=g, dtype=D) - 0.5) * 30,
            torch.full((150, 1), -1.5, dtype=D),
        ],
        -1,
    )
    nf = torch.tensor([0.0, 0.0, 1.0], dtype=D).expand(150, 3)
    wf = torch.rand(150, generator=g, dtype=D) + 0.5
    groups = torch.cat(
        [
            torch.zeros(300, dtype=torch.long),
            torch.ones(200, dtype=torch.long),
            torch.full((150,), 2, dtype=torch.long),
        ]
    )
    return (
        torch.cat([body, box + 3.0, floor]),
        torch.cat([nb, -nx, nf]),
        torch.cat([0.01 * wb, 50 * wx, wf]),
        groups,
    )


def _f32(pts, nrm, w):
    return pts.to(F), nrm.to(F), (w / w.sum()).to(F)


def _rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s], [s, c]], dtype=D)


@_KERNELS
@pytest.mark.parametrize("dim", [2, 3])
def test_one_group_is_the_ungrouped_forest_bitwise(dim, kernel):
    """groups of one condition and groups=None build the same forest bit for bit
    (every field, every tree's tensors) and the same pair lists and counts."""
    pts, nrm, w = _f32(*cloud(400, d=dim, seed=3)[:3])
    q = (
        torch.randn(150, dim, generator=torch.Generator().manual_seed(4), dtype=D) * 3
    ).to(F)
    tol = torch.rand(400, generator=torch.Generator().manual_seed(5), dtype=D) * 1e-3
    for theta in (0.5, 1.0):
        kw = dict(theta=theta, normal_tol=tol, build=kernel)
        a = BoundaryForest.build(pts, w, nrm, None, **kw)
        b = BoundaryForest.build(pts, w, nrm, None, groups=torch.full((400,), 3), **kw)
        assert _differing(a, b) == []
        assert a.tree_group.tolist() == [0] * len(a.trees)
        assert torch.equal(a.group_origin, a.origin[None])
        assert a.frame_points(q).shape == (150, dim)
        for x, y in zip(a.point_pairs(q, walk=kernel), b.point_pairs(q, walk=kernel)):
            assert _equal(x, y)


@_KERNELS
@pytest.mark.parametrize("case", ["two_conditions_2d", "three_conditions_3d"])
def test_each_tree_holds_one_condition_about_its_centroid_and_lists_cover_every_cell_once(
    case, kernel
):
    """The trees come in the conditions' order and each holds cells of one condition,
    every point's list covers every cell once, and each condition's trees live about
    its own centroid in the whole boundary's frame."""
    pts, nrm, w, groups = (
        _two_conditions() if case == "two_conditions_2d" else _three_conditions_3d()
    )
    n, d = pts.shape
    pts32, nrm32, w32 = _f32(pts, nrm, w)
    f = BoundaryForest.build(
        pts32, w32, nrm32, None, theta=1.0, groups=groups, build=kernel
    )
    ### Trees group-major (in the conditions' order), then face; each tree one condition.
    present = torch.unique(groups).tolist()
    tg = [int(groups[c[0]]) for c in f.tree_cells]
    assert tg == sorted(tg) and set(tg) == set(present)
    assert tg == [present[g] for g in f.tree_group.tolist()]
    for c in f.tree_cells:
        assert groups[c].unique().numel() == 1
    for m in range(f.n_nodes):
        cells = f.cells[f.range_start[m] : f.range_start[m] + f.range_count[m]]
        assert groups[cells].unique().numel() == 1
    ### Every point's list covers every cell exactly once, as itself or in one node.
    g = torch.Generator().manual_seed(1)
    span = pts.abs().amax(0)
    q = torch.cat(
        [
            torch.randn(60, d, generator=g, dtype=D) * 0.6,  # near the body
            (torch.rand(60, d, generator=g, dtype=D) - 0.5)
            * 1.9
            * span,  # in the far field
        ]
    ).to(F)
    assert f.frame_points(q).shape == (len(present), 120, d)
    qi, si, _ = f.point_pairs(q, walk=kernel)
    for p in range(q.shape[0]):
        s = si[qi == p]
        node_cells, _ = f.node_cells(s[s >= n] - n)
        count = torch.bincount(torch.cat([s[s < n], node_cells]), minlength=n)
        assert torch.equal(count, torch.ones(n, dtype=count.dtype))
    ### Each condition's trees live about its own measure-weighted centroid, in the
    ### whole boundary's frame.
    assert torch.equal(f.frame, canonical_frame(pts32.double(), w32))
    for i, k in enumerate(present):
        sel = (groups == k).nonzero().squeeze(-1)
        wk = w32[sel].double() / w32[sel].double().sum()
        assert torch.equal(
            f.group_origin[i], (wk[:, None] * pts32[sel].double()).sum(0)
        )
    for t, (tree, c) in enumerate(zip(f.trees, f.tree_cells)):
        local = frame_coordinates(
            pts32[c].double() - f.group_origin[f.tree_group[t]], f.frame
        ).to(F)
        assert torch.equal(tree.node_aabb_min[0], local.amin(0))
        assert torch.equal(tree.node_aabb_max[0], local.amax(0))
    ### The body's trees sit about the body, not about the whole boundary's centroid
    ### (several body sizes away).
    assert float((f.origin - f.group_origin[0]).norm()) > 4 * float(
        pts[groups == 0].abs().amax()
    )


@pytest.mark.parametrize("case", ["two_conditions_2d", "three_conditions_3d"])
def test_the_fast_build_and_walk_are_the_reference_s_with_groups(case):
    """With conditions, the fast forest build and the fast walk give bitwise the
    reference's forest, lists and pair counts, at two opening angles."""
    pts, nrm, w, groups = (
        _two_conditions() if case == "two_conditions_2d" else _three_conditions_3d()
    )
    pts32, nrm32, w32 = _f32(pts, nrm, w)
    d = pts.shape[1]
    q = (
        torch.randn(200, d, generator=torch.Generator().manual_seed(2), dtype=D) * 5
    ).to(F)
    for theta in (0.5, 1.0):
        ref = BoundaryForest.build(
            pts32, w32, nrm32, None, theta=theta, groups=groups, build="reference"
        )
        fast = BoundaryForest.build(
            pts32, w32, nrm32, None, theta=theta, groups=groups, build="fast"
        )
        assert _differing(ref, fast) == []
        lists = [ref.point_pairs(q, walk=walk) for walk in ("reference", "fast")]
        assert _equal(lists[0], lists[1])
        counts = ref.frame_pairs(ref.frame_points(q), walk="fast", count_only=True)
        assert torch.equal(counts, torch.bincount(lists[0][0], minlength=200))


def test_a_range_of_points_walks_as_among_all_of_them_with_groups():
    """walk_points_range and walk_pair_counts slice each group's coordinates of the
    points alike (query_chunks' walk)."""
    pts, nrm, w, groups = _two_conditions()
    pts32, nrm32, w32 = _f32(pts, nrm, w)
    q = (
        torch.randn(90, 2, generator=torch.Generator().manual_seed(2), dtype=D) * 3
    ).to(F)
    kw = dict(
        query_r=q[None],
        query_n=torch.nn.functional.normalize(q, dim=-1)[None],
        query_wall=q.norm(dim=-1)[None],
        boundary_r=pts32[None],
        boundary_n=nrm32[None],
        boundary_vecs=None,
        log_frac=torch.log(w32)[None],
        g_unit=torch.tensor([[[1.0, 0.0]]]),
        theta=1.0,
        eps=1e-12,
        conditions=groups[None],
    )
    whole = tree_read.tree_read_walk(**kw)
    lazy = tree_read.tree_read_walk(walk_points=False, **kw)
    qi, si, _ = whole.lists[0]
    assert torch.equal(
        tree_read.walk_pair_counts(lazy), torch.bincount(qi, minlength=90)
    )
    part = tree_read.walk_points_range(lazy, 30, 70)
    pq, ps, _ = part.lists[0]
    sel = (qi >= 30) & (qi < 70)
    assert torch.equal(pq + 30, qi[sel]) and torch.equal(ps, si[sel])


def test_a_small_body_far_from_the_centroid_keeps_its_tree_membership_under_pose():
    """A rigid pose (rotation and translation in float64, cast once to float32)
    leaves each tree's set of cells unchanged."""
    pts, nrm, w, groups = _two_conditions(seed=2)
    w32 = (w / w.sum()).to(F)
    native = BoundaryForest.build(
        pts.to(F), w32, nrm.to(F), None, theta=1.0, groups=groups
    )
    for angle, shift in [
        (0.7, (3.7, -1.2)),
        (-2.1, (-40.0, 15.5)),
        (math.pi / 2, (0.0, 0.0)),
    ]:
        rot = _rotation(angle)
        posed = BoundaryForest.build(
            (pts @ rot.t() + torch.tensor(shift, dtype=D)).to(F),
            w32,
            (nrm @ rot.t()).to(F),
            None,
            theta=1.0,
            groups=groups,
        )
        assert torch.equal(posed.tree_group, native.tree_group)
        assert [c.sort().values.tolist() for c in posed.tree_cells] == [
            c.sort().values.tolist() for c in native.tree_cells
        ]


def _airfoil_model():
    torch.manual_seed(0)
    return ISLA(
        output_schema={"p": {"rank": 0}},
        boundary_schemas={"airfoil": {}, "freestream": {}},
        global_schema={"g": {"rank": 1}},
        interior=True,
        n_spatial_dims=2,
        hidden=32,
        n_layers=2,
        n_slices=8,
        n_boundary_tokens=150,
    ).eval()


def _airfoil_meshes(seed=0):
    pts, nrm, w, groups = _two_conditions(seed=seed)
    return {
        "airfoil": soup(pts[groups == 0], nrm[groups == 0], w[groups == 0]).to(F),
        "freestream": soup(pts[groups == 1], nrm[groups == 1], w[groups == 1]).to(F),
    }


def _queries():
    return points_mesh(
        (
            torch.randn(40, 2, generator=torch.Generator().manual_seed(3), dtype=D) * 2
        ).to(F)
    )


def test_the_model_reads_one_set_of_trees_per_condition_with_its_own_tol(monkeypatch):
    """forward passes each kept token's declared condition to the forest, and each
    condition's normal tolerance is measured from its own tokens' centroid."""
    m = _airfoil_model()
    seen = {}
    build = BoundaryForest.build

    def spy(*args, **kw):
        seen.update(kw)
        seen["forest"] = build(*args, **kw)
        return seen["forest"]

    monkeypatch.setattr(BoundaryForest, "build", spy)
    with torch.no_grad():
        m(
            boundary_meshes=_airfoil_meshes(),
            prediction_points=_queries(),
            global_data={"g": torch.tensor([1.0, 0.0])},
        )
    groups, forest = seen["groups"], seen["forest"]
    assert set(groups.tolist()) == {0, 1}
    assert forest.group_origin.shape[0] == 2
    for c, t in zip(forest.tree_cells, forest.tree_group.tolist()):
        assert groups[c].unique().tolist() == [t]
    ### The body's tol is a body-sized bound (vertices within a chord of its own
    ### centroid), not one measured from the whole boundary's centroid tens of
    ### chords away.
    body = seen["normal_tol"][groups == 0]
    assert torch.isfinite(body).all() and float(body.max()) < 1e-3


def test_one_given_condition_of_two_builds_the_ungrouped_forest(monkeypatch):
    """A model of two conditions given one builds the forest of that boundary alone,
    bitwise the ungrouped one (an absent condition keeps its number)."""
    m = _airfoil_model()
    seen = []
    build = BoundaryForest.build

    def spy(*args, **kw):
        seen.append((args, kw, build(*args, **kw)))
        return seen[-1][2]

    monkeypatch.setattr(BoundaryForest, "build", spy)
    with torch.no_grad():
        m(
            boundary_meshes={"freestream": _airfoil_meshes()["freestream"]},
            prediction_points=_queries(),
            global_data={"g": torch.tensor([1.0, 0.0])},
        )
    args, kw, forest = seen[0]
    assert set(kw["groups"].tolist()) == {1}
    kw = {k: v for k, v in kw.items() if k != "groups"}
    assert _differing(forest, build(*args, **kw)) == []


def test_forward_is_finite_and_differentiable_with_two_conditions():
    """A float32 training step of the two-condition airfoil model is finite and gives
    every parameter a gradient."""
    m = _airfoil_model().train()
    out = m(
        boundary_meshes=_airfoil_meshes(),
        prediction_points=_queries(),
        global_data={"g": torch.tensor([1.0, 0.0])},
    )
    p = out.point_data["p"]
    assert torch.isfinite(p).all()
    p.square().mean().backward()
    assert not [n for n, x in m.named_parameters() if x.grad is None]


def _loop(corners, n_side):
    """Cells of a closed polygon through ``corners`` ``(K, 2)``, ``n_side`` cells per
    side: (centroid, unit normal pointing away from the polygon's centroid, length)."""
    v = torch.cat(
        [
            a + (b - a) * torch.linspace(0, 1, n_side + 1, dtype=D)[:-1, None]
            for a, b in zip(corners, torch.roll(corners, -1, 0))
        ]
    )
    e = torch.roll(v, -1, 0) - v
    length = e.norm(dim=-1)
    n = torch.stack([e[:, 1], -e[:, 0]], -1) / length[:, None]
    mid = v + 0.5 * e
    n = torch.where(((n * (mid - mid.mean(0))).sum(-1) < 0)[:, None], -n, n)
    return mid, n, length


@_KERNELS
def test_a_condition_with_nearly_equal_variances_uses_the_frames_rotation_bound(kernel):
    """The trees live in the whole boundary's frame, so the bound on that frame's
    rotation is the whole boundary's. A condition whose own covariance has two nearly
    equal variances (a square inlet) must not inflate its trees' separation
    diameters: its own eigenvalue gap would bound the rotation of a frame the forest
    never uses, and would multiply the pairs."""
    ### Condition 0: a long thin rectangle (the whole boundary's frame is well defined);
    ### condition 1: a square of aspect 1 + 2e-6, whose variance gap is about 8 times
    ### the float32 rounding of its covariance.
    rect = torch.tensor(
        [[-20.0, -1.0], [20.0, -1.0], [20.0, 1.0], [-20.0, 1.0]], dtype=D
    )
    s = 1 + 2e-6
    square = torch.tensor([[-1.0, -s], [1.0, -s], [1.0, s], [-1.0, s]], dtype=D)
    square = square + torch.tensor([30.0, 0.0], dtype=D)
    p0, n0, w0 = _loop(rect, 100)
    p1, n1, w1 = _loop(square, 100)
    pts, nrm, w = _f32(torch.cat([p0, p1]), torch.cat([n0, n1]), torch.cat([w0, w1]))
    groups = torch.cat(
        [torch.zeros(len(p0), dtype=torch.long), torch.ones(len(p1), dtype=torch.long)]
    )
    g = torch.Generator().manual_seed(7)
    q = (torch.rand(400, 2, generator=g, dtype=D) * torch.tensor([70.0, 20.0], dtype=D)
         - torch.tensor([25.0, 10.0], dtype=D)).to(F)  # fmt: skip
    grouped = BoundaryForest.build(
        pts, w, nrm, None, theta=1.0, groups=groups, build=kernel
    )
    whole = BoundaryForest.build(pts, w, nrm, None, theta=1.0, build=kernel)
    for t, tree in enumerate(grouped.trees):
        diag = (tree.node_aabb_max[0].double() - tree.node_aabb_min[0].double()).norm()
        assert tree.node_diameter_sq[0].double().sqrt() / diag < 1.01, (
            t,
            int(grouped.tree_group[t]),
        )
    n_grouped = grouped.point_pairs(q, walk=kernel)[0].numel()
    n_whole = whole.point_pairs(q, walk=kernel)[0].numel()
    assert n_grouped < 2 * n_whole, (n_grouped, n_whole)


def test_the_model_s_forest_keeps_its_tree_membership_under_pose(monkeypatch):
    """The forest the model builds, with each condition's normal tolerance as forward
    measures it from the posed meshes' float32 vertices, keeps each tree's set of
    tokens under a rigid pose (the meshes moved in float64, cast once to float32) that
    keeps the body within its own size of the origin, though the body's normals tie
    between two faces of the forest's frame, so its rounding would decide them without
    the tolerance. (A pose that moves the body many sizes away rounds its vertices
    beyond any bound that is a function of the boundary alone: normal_error_bound.)"""
    pts, nrm, w, groups = _two_conditions()
    pf, nf, wf = pts[groups == 1], nrm[groups == 1], w[groups == 1]

    ### The body: a diamond of diagonal 1 at the origin with its corners on the whole
    ### boundary's frame axes, so every normal lies at 45 degrees to them. Its cells
    ### have 4-fold symmetry, so their covariance is isotropic and the frame is the
    ### same in every orientation of the diamond.
    def diamond(frame):
        return _loop(0.5 * torch.cat([frame.t(), -frame.t()]), 100)

    pb, nb, wb = diamond(torch.eye(2, dtype=D))
    pb, nb, wb = diamond(canonical_frame(torch.cat([pb, pf]), torch.cat([wb, wf])))
    meshes = {"airfoil": soup(pb, nb, wb), "freestream": soup(pf, nf, wf)}
    torch.manual_seed(0)
    m = ISLA(
        output_schema={"p": {"rank": 0}},
        boundary_schemas={"airfoil": {}, "freestream": {}},
        global_schema={"g": {"rank": 1}},
        interior=True,
        n_spatial_dims=2,
        hidden=32,
        n_layers=2,
        n_slices=8,
        n_boundary_tokens=len(pb) + len(pf),  # every cell a token, so the draw is fixed
    ).eval()
    seen = []
    build = BoundaryForest.build

    def spy(*args, **kw):
        seen.append((args, kw, build(*args, **kw)))
        return seen[-1][2]

    monkeypatch.setattr(BoundaryForest, "build", spy)
    q, g = _queries().points.double(), torch.tensor([1.0, 0.0], dtype=D)
    poses = [(0.0, (0.0, 0.0)), (0.7, (0.0, 0.0)), (-2.1, (0.0, 0.0)), (math.pi / 2, (0.0, 0.0)),
             (0.7, (0.3, -0.2)), (0.7, (1.0, 0.5))]  # fmt: skip
    for angle, shift in poses:
        rot, t = _rotation(angle), torch.tensor(shift, dtype=D)
        with torch.no_grad():
            m(
                boundary_meshes={
                    k: transform(v, rot, t).to(F) for k, v in meshes.items()
                },
                prediction_points=points_mesh((q @ rot.t() + t).to(F)),
                global_data={"g": (g @ rot.t()).to(F)},
            )

    def membership(forest):
        return [c.sort().values.tolist() for c in forest.tree_cells]

    args, kw, native = seen[0]
    assert set(kw["groups"].tolist()) == {0, 1}
    ### The tolerance decides the body's faces: without it, rounding does.
    untol = build(*args, **{k: v for k, v in kw.items() if k != "normal_tol"})
    assert membership(untol) != membership(native)
    for (_, _, posed), pose in zip(seen[1:], poses[1:]):
        assert torch.equal(posed.tree_group, native.tree_group), pose
        assert membership(posed) == membership(native), pose
