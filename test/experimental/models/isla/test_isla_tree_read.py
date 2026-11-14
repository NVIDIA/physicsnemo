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

"""Contract tests of the interior task's boundary read (``interior=True``).

Prediction points read every boundary cell after every block: a softmax over the
cells with the log measure fraction as bias, a learned kernel of the pair invariants
and a query-key match. The read walks a forest of Barnes-Hut cluster trees over the
boundary cells (GLOBE's ``ClusterTree``, one tree per normal cube face) once per
prediction point, at the opening angle ``TREE_READ_THETA``.

At ``theta = 0`` (the model's ``tree_read_theta``, which these tests set) the read
takes every cell pair, so it must equal the brute-force all-pairs read kept in this
file, be invariant under similarity, refinement and a rescale of the measures, and
give the same outputs on CUDA as on CPU. Query independence holds exactly at every ``theta``: a point's
walk depends on the forest and the point alone. At a finite ``theta`` the read is an
approximation: the tests measure its error, and check that each point's pairs cover
the boundary measure exactly once and admit only well-separated, orientation-coherent
clusters. The forest is a function of the boundary alone: the same in float32 poses
of a mesh aligned with its own frame (where rounding would decide sign octants and
the box test's ties), bitwise the same under bf16 autocast, and the same under a
renumbering of the cells. Covariance under rigid motions is tested with the other
model contracts, in test_isla_dtt_contracts.py.

The read's output projection starts at zero, so most tests give every read parameter
a random value first.
"""

import copy
import itertools
import logging
import math
import types

import pytest
import torch

import physicsnemo.experimental.models.isla._tree_build as tree_build
import physicsnemo.experimental.models.isla._tree_read as tree_read
import physicsnemo.experimental.models.isla.model as model_mod
import physicsnemo.mesh.spatial.cluster_tree as cluster_tree
from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.experimental.models.isla._tree_read import (
    KERNEL_READ_HEADS,
    KERNEL_READ_KEY,
    KERNEL_READ_WIDTH,
    TREE_READ_LEAF_SIZE,
    TREE_READ_THETA,
    BoundaryForest,
    _KernelBasis,
    _TreeKernelRead,
    frame_coordinates,
    kernel_geo_width,
    normal_error_bound,
    normal_faces,
    tree_pair_invariants,
    tree_read_pairs,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY
from physicsnemo.mesh.spatial._point_walk_warp import point_walk
from physicsnemo.mesh.spatial.cluster_tree import ClusterTree
from test.experimental.models.isla._forced import in_another_thread_while_compiling, ran
from test.experimental.models.isla._kernel_cases import (
    _basis_case,
    _basis_step,
    _read_case,
    _SinkRead,
    _to,
    _two_x,
)
from test.experimental.models.isla._marks import (
    DEVICES,
    requires_cuda,
    requires_cuda_triton,
)
from test.experimental.models.isla._meshes import (
    cloud,
    orthogonal,
    outputs,
    points_mesh,
    refine,
    same,
    soup,
    transform,
)

PV = {"p": {"rank": 0}, "v": {"rank": 1}}
G = {"g": {"rank": 1}}
D = torch.float64


def _model(dim=3, theta=TREE_READ_THETA, randomize=True, seed=0, **kw):
    torch.manual_seed(seed)
    kw = {
        "hidden": 32,
        "n_layers": 2,
        "n_slices": 8,
        "boundary_schemas": {"wall": {}},
        **kw,
    }
    m = ISLA(
        output_schema=PV,
        global_schema=G,
        interior=True,
        n_spatial_dims=dim,
        tree_read_theta=theta,
        **kw,
    )
    if randomize:
        with torch.no_grad():
            for p in [*m.kernel_reads.parameters(), *m.kernel_basis.parameters()]:
                p.normal_(0.0, 0.3)
    return m


def _tree(dim=3, theta=0.0, **kw):
    """A float64 model in eval mode, its read at ``theta``."""
    return _model(dim, theta, **kw).double().eval()


def _exact_pairs(
    bnd, qry, q_wall, g_unit, normal_tol=None, conditions=None, shifts=None
):
    """What the brute-force read needs of the tokens (one case), with the cells' log
    share shift of the per-condition quadrature (``None``: none)."""
    return {
        "bnd": bnd,
        "qry": qry,
        "q_wall": q_wall,
        "g_unit": g_unit,
        "log_shift": None if shifts is None else shifts[0],
    }


def _exact_read(model, read, h_b, h_q, tok):
    r"""The boundary read of one block over every (point, cell) pair, written out
    densely: the invariants :math:`[r, \log r, \log(r / |d_q|), \hat\rho\cdot n_j,
    \hat\rho\cdot n_q, n_q\cdot n_j, \hat\rho\cdot g_k, n_j\cdot g_k, \hat\rho\cdot u,
    n_q\cdot u]`, the kernel basis, and a softmax over the
    cells of the kernel term, the query-key match and the log measure fraction."""
    bnd, qry, wall, g = tok["bnd"], tok["qry"], tok["q_wall"][0], tok["g_unit"][0]
    nb = model._n_boundary_vecs
    y, n_j = bnd["r"][0], bnd["n_hat"][0]
    x, n_q = qry["r"][0], qry["n_hat"][0]
    rel = x[:, None] - y[None]
    dist = rel.norm(dim=-1)
    rho = rel / dist.clamp_min(1e-12)[..., None]
    nq_, nb_ = x.shape[0], y.shape[0]
    log_r = dist.clamp_min(1e-12).log()
    cols = [
        torch.stack(
            [
                dist,
                log_r,
                log_r - wall.clamp_min(1e-12).log()[:, None],
                (rho * n_j[None]).sum(-1),
                (rho * n_q[:, None]).sum(-1),
                (n_q[:, None] * n_j[None]).sum(-1),
            ],
            -1,
        ),
        torch.einsum("qnd,kd->qnk", rho, g),
        (n_j @ g.t())[None].expand(nq_, nb_, -1),
    ]
    if "vecs" in bnd.keys() and nb:
        u = bnd["vecs"][0][:, :nb]
        cols += [
            torch.einsum("qnd,npd->qnp", rho, u),
            torch.einsum("qd,npd->qnp", n_q, u),
        ]
    phi = model.kernel_basis(torch.cat(cols, -1))  # (Q, N, BASIS)
    nh, c, kc = (
        KERNEL_READ_HEADS,
        KERNEL_READ_WIDTH // KERNEL_READ_HEADS,
        KERNEL_READ_KEY,
    )
    hb = read.norm_b(h_b)
    v = read.value(hb).view(nb_, nh, c)
    k = read.key(hb).view(nb_, nh, kc)
    q = read.query(read.norm_q(h_q)).view(nq_, nh, kc)
    log_w = bnd["log_w"][0, :, 0]
    log_frac = log_w - torch.logsumexp(log_w, 0)
    if tok.get("log_shift") is not None:
        log_frac = log_frac + tok["log_shift"][0].to(log_frac.dtype)
    logit = phi @ read.kernel.weight.t() + torch.einsum(
        "qhc,nhc->qnh", q, k
    ) / math.sqrt(kc)
    att = torch.softmax(logit + log_frac[None, :, None], dim=1)
    return read.out(torch.einsum("qnh,nhc->qhc", att, v).reshape(nq_, nh * c))


def _with_exact_read(model):
    """A copy of ``model`` whose boundary read is :func:`_exact_read` (the reference),
    for a forward pass whose points run in one chunk: each block's sources are the
    boundary states, read with the tokens its forest is built from."""
    ref = copy.deepcopy(model)
    walk, query_tokens, tok = ref._tree_read_walk, ref._query_tokens, {}

    def spy(bnd, q_r, g_unit, normal_tol=None, conditions=None):
        tok.update(bnd=bnd, g_unit=g_unit)
        return walk(bnd, q_r, g_unit, normal_tol, conditions=conditions)

    def spy_tokens(*args):
        ### (the chunk's points with their geometry, as the read's walk takes them)
        q_tokens, q_wall = query_tokens(*args)
        tok.update(_exact_pairs(tok["bnd"], q_tokens, q_wall, tok["g_unit"]))
        return q_tokens, q_wall

    ref._tree_read_walk = spy
    ref._query_tokens = spy_tokens
    for read in ref.kernel_reads:
        read.sources = types.MethodType(
            lambda self, h_b, walk, compiled=False: h_b, read
        )
        read.read = types.MethodType(
            lambda self, h_b, h_q, pairs, compiled=False: _exact_read(
                ref, self, h_b, h_q, tok
            ),
            read,
        )
    return ref


def _case(dim=3, n=300, nq=120, seed=0):
    pts, nrm, w, drv = cloud(n, d=dim, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    near = pts[: nq - 10] + 0.15 * torch.randn(nq - 10, dim, generator=g, dtype=D)
    far = 40.0 + torch.randn(10, dim, generator=g, dtype=D)
    q = torch.cat([near, far])
    return soup(pts, nrm, w), drv, q


def _sphere_case(n=4000, nq=200, seed=0):
    """A smooth boundary: a sphere of radius 2 with outward normals and areas, and
    points at radii 2.05 to 6 plus a few far away."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.nn.functional.normalize(
        torch.randn(n, 3, dtype=D, generator=g), dim=-1
    )
    pts, nrm = 2.0 * dirs, dirs
    w = torch.full((n,), 4 * math.pi * 4.0 / n, dtype=D)
    drv = torch.tensor([0.3, 1.0, 0.2], dtype=D)
    q_dirs = torch.nn.functional.normalize(
        torch.randn(nq, 3, dtype=D, generator=g), dim=-1
    )
    radii = torch.cat(
        [
            2.05 + 3.95 * torch.rand(nq - 10, dtype=D, generator=g),
            torch.full((10,), 40.0, dtype=D),
        ]
    )
    q = q_dirs * radii[:, None]
    return soup(pts, nrm, w), drv, q


def _run(m, mesh, drv, q):
    with torch.no_grad():
        return m(
            boundary_meshes={"wall": mesh},
            prediction_points=points_mesh(q),
            global_data={"g": drv},
        )


def _rows(out, idx):
    """Rows ``idx`` of an outputs mapping."""
    return {k: v[idx] for k, v in out.items()}


def _rel(a, b):
    return max(float((a[k] - b[k]).norm() / b[k].norm().clamp_min(1e-30)) for k in b)


@pytest.mark.parametrize("vectors", [False, True], ids=["plain", "boundary_vectors"])
@pytest.mark.parametrize("dim", [3, 2])
def test_tree_read_at_theta_zero_is_the_exact_read(dim, vectors):
    """At theta 0 the forest read is the all-pairs read (the brute-force reference
    above), with and without rank-1 boundary inputs; at the model's angle it is not."""
    extra = {"boundary_schemas": {"wall": {"tr": {"rank": 1}}}} if vectors else {}
    tree = _tree(dim, 0.0, **extra)
    assert (
        isinstance(tree.kernel_reads[0], _TreeKernelRead)
        and len(tree.kernel_reads) == 2
    )
    mesh, drv, q = _case(dim)
    if vectors:
        tr = torch.randn(
            mesh.n_cells, dim, generator=torch.Generator().manual_seed(5), dtype=D
        )
        mesh = mesh.with_data(
            cell_data={
                EFFECTIVE_MEASURE_KEY: mesh.cell_data[EFFECTIVE_MEASURE_KEY],
                "tr": tr,
            }
        )
    ref = outputs(_run(_with_exact_read(tree), mesh, drv, q))
    assert _rel(outputs(_run(tree, mesh, drv, q)), ref) < 1e-10
    tree.tree_read_theta = TREE_READ_THETA
    assert _rel(outputs(_run(tree, mesh, drv, q)), ref) > 1e-6


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
def test_tree_read_pairs_cover_the_boundary_measure_once(theta):
    """Every point's pair list (cells and nodes) represents the whole boundary measure
    exactly once, whatever theta; at theta = 0 there are no node pairs."""
    torch.manual_seed(0)
    pts, nrm, w, drv = cloud(500, d=3, seed=3)
    q, _, _, _ = cloud(150, d=3, seed=4, scale=(4.0, 4.0, 4.0))
    log_frac = torch.log(w / w.sum())[None]
    basis = _KernelBasis(kernel_geo_width(1, 0)).double()
    pairs = tree_read_pairs(
        basis,
        query_r=q[None],
        query_n=torch.nn.functional.normalize(q, dim=-1)[None],
        query_wall=torch.rand(1, 150, dtype=D) + 0.1,
        boundary_r=pts[None],
        boundary_n=nrm[None],
        boundary_vecs=None,
        log_frac=log_frac,
        g_unit=torch.nn.functional.normalize(drv, dim=-1)[None, None],
        theta=theta,
        eps=1e-12,
    )
    plan = pairs.plan
    mass = torch.zeros(plan.n_q, dtype=D).index_add(0, plan.qi, torch.exp(pairs.log_m))
    assert torch.allclose(mass, torch.ones(plan.n_q, dtype=D), atol=1e-12)
    assert pairs.basis.shape == (pairs.n_pairs, tree_read.KERNEL_READ_BASIS)
    assert pairs.basis.isfinite().all()
    if theta == 0.0:
        assert pairs.n_pairs_node == 0 and pairs.n_pairs == 150 * 500
    else:
        assert pairs.n_pairs_node > 0 and pairs.n_pairs < 150 * 500


def _node_cells(f, i):
    cells, _ = f.node_cells(torch.tensor([i]))
    return cells


@pytest.mark.parametrize("dim", [3, 2])
def test_boundary_forest_splits_by_normal_face_and_its_nodes_are_tight(dim):
    """One cluster tree per normal cube face: the trees' cells partition the cells and
    each tree's cells share a face; every node's box holds its cells; its virtual
    cell is their measure-weighted centroid, renormalized mean normal, mean rank-1
    input and total measure; its coherence is the RMS-chord test; the children split
    their parent's range; the leaves tile the cells with at most the leaf size each."""
    pts, nrm, w, _ = cloud(1000, d=dim, seed=8)
    vecs = torch.nn.functional.normalize(torch.randn(1000, 2, dim, dtype=D), dim=-1)
    theta = 0.5
    f = BoundaryForest.build(pts, w, nrm, vecs, theta=theta, leaf_size=8, eps=1e-12)
    n, m = f.n_cells, f.n_nodes
    assert (
        n == 1000
        and len(f.trees) == len(f.tree_cells) == int(f.node_offset.shape[0]) - 1
    )
    assert len(f.trees) == 2 * dim
    assert sum(t.n_nodes for t in f.trees) == m
    assert torch.equal(torch.sort(torch.cat(f.tree_cells)).values, torch.arange(n))
    assert torch.equal(torch.sort(f.cells).values, torch.arange(n))
    ### Faces and boxes live in the boundary's own frame about its centroid.
    assert torch.allclose(f.frame.t() @ f.frame, torch.eye(dim, dtype=D), atol=1e-12)
    assert torch.allclose(f.origin, (w[:, None] * pts).sum(0) / w.sum())
    pts_f = frame_coordinates(pts - f.origin, f.frame)
    face = normal_faces(frame_coordinates(nrm, f.frame), torch.zeros(1000, dtype=D))
    for sel in f.tree_cells:
        assert len(set(face[sel].tolist())) == 1
    for i in range(m):
        cells = _node_cells(f, i)
        assert bool((pts_f[cells] >= f.aabb_min[i] - 1e-9).all())
        assert bool((pts_f[cells] <= f.aabb_max[i] + 1e-9).all())
        assert f.diameter_sq[i] >= (f.aabb_max[i] - f.aabb_min[i]).square().sum()
        assert torch.allclose(f.measure[i], w[cells].sum())
        assert torch.allclose(
            f.center[i], (pts[cells] * w[cells, None]).sum(0) / w[cells].sum()
        )
        mean_n = (nrm[cells] * w[cells, None]).sum(0) / w[cells].sum()
        assert torch.allclose(f.normal[i], mean_n / mean_n.norm(), atol=1e-9)
        mean_u = (vecs[cells] * w[cells, None, None]).sum(0) / w[cells].sum()
        assert torch.allclose(f.vecs[i], mean_u, atol=1e-9)
        rms_chord = math.sqrt(2 * (1 - float(mean_n.norm())))
        assert bool(f.coherent[i]) == (rms_chord < theta)
        left, right = int(f.left[i]), int(f.right[i])
        if left >= 0:
            a, b = sorted(
                [(int(f.range_start[c]), int(f.range_count[c])) for c in (left, right)]
            )
            assert a[0] == int(f.range_start[i]) and a[0] + a[1] == b[0]
            assert b[0] + b[1] == int(f.range_start[i] + f.range_count[i])
        else:
            assert 1 <= int(f.range_count[i]) <= 8
    leaves = f.leaf_ids
    ends = torch.cumsum(f.range_count[leaves], 0)
    assert torch.equal(f.range_start[leaves][1:], ends[:-1]) and int(ends[-1]) == n
    assert torch.allclose(f.normal.norm(dim=-1), torch.ones(m, dtype=D))
    assert f.vecs.shape == (m, 2, dim)


def test_boundary_forest_never_mixes_opposite_normals():
    """A thin plate in a random orientation (two faces a hair apart, opposite normals)
    lands in two trees, one per face, so no node holds cells of both faces whatever
    its position, and every node is coherent with its face's normal."""
    g = torch.Generator().manual_seed(11)
    xy = torch.rand(400, 2, generator=g, dtype=D)
    up = torch.cat([xy, torch.full((400, 1), 1e-3, dtype=D)], -1)
    down = torch.cat([xy, torch.full((400, 1), -1e-3, dtype=D)], -1)
    R = orthogonal(3, 5, False)
    pts = torch.cat([up, down]) @ R.T
    nrm = (
        torch.cat(
            [
                torch.tensor([[0.0, 0.0, 1.0]] * 400, dtype=D),
                torch.tensor([[0.0, 0.0, -1.0]] * 400, dtype=D),
            ]
        )
        @ R.T
    )
    w = torch.ones(800, dtype=D)
    f = BoundaryForest.build(pts, w, nrm, None, theta=0.5, leaf_size=8, eps=1e-12)
    assert len(f.trees) == 2
    for sel in f.tree_cells:
        assert len(set((sel >= 400).tolist())) == 1
    assert bool(f.coherent.all())
    assert torch.allclose(f.normal, nrm[f.cells[f.range_start]], atol=1e-9)


@pytest.mark.parametrize("dim", [2, 3])
def test_normal_faces_are_antipodal_and_ties_take_the_lowest_axis(dim):
    """Every normal with components in {-1, 0, 1}: -n lands on the face opposite n's;
    a normal whose largest components are equal (or within twice their error bound)
    takes the lowest of those axes, and one outside the bound its largest axis."""
    grid = torch.tensor(list(itertools.product((-1.0, 0.0, 1.0), repeat=dim)), dtype=D)
    nrm = torch.nn.functional.normalize(grid[grid.abs().sum(-1) > 0], dim=-1)
    for tol in (0.0, 1e-3):
        t = torch.full((nrm.shape[0],), tol, dtype=D)
        faces = normal_faces(nrm, t)
        assert torch.equal(normal_faces(-nrm, t), faces ^ 1)
        lowest = (nrm.abs() >= nrm.abs().amax(1, keepdim=True)).to(torch.int8).argmax(1)
        assert torch.equal(faces // 2, lowest)
    if dim == 3:
        ### Components 1 and 1 + 4e-4: tied within twice a bound of 1e-3 (the lower axis,
        ### 1), not within twice 1e-4 (the larger, 2).
        n = torch.tensor([[0.0, 1.0, 1.0 + 4e-4]], dtype=D)
        assert int(normal_faces(n, torch.full((1,), 1e-3, dtype=D))) == 3
        assert int(normal_faces(n, torch.full((1,), 1e-4, dtype=D))) == 5


def test_boundary_forest_partition_is_the_same_under_a_winding_flip():
    """A 2 x 1 rectangle, symmetric about both axes, so its own frame is exact and
    its normals there have zero components: reversing every normal maps each tree's
    cells to the opposite face, so the forest partitions the cells the same way."""
    s = (torch.arange(8, dtype=D) + 0.5) / 8
    bottom = torch.stack([2 * s - 1, torch.full_like(s, -0.5)], -1)
    left = torch.stack([torch.full_like(s, -1.0), s - 0.5], -1)
    pts = torch.cat([bottom, -bottom, left, -left])
    axes = torch.tensor([[0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]], dtype=D)
    nrm = axes.repeat_interleave(8, 0)
    w = torch.cat([torch.full((16,), 0.25), torch.full((16,), 0.125)]).to(D)
    forests = [
        BoundaryForest.build(pts, w, n, None, theta=0.5, leaf_size=4, eps=1e-12)
        for n in (nrm, -nrm)
    ]
    assert bool((frame_coordinates(nrm, forests[0].frame) == 0).any())
    blocks = [{frozenset(c.tolist()) for c in f.tree_cells} for f in forests]
    assert blocks[0] == blocks[1]


def test_node_means_are_the_measure_weighted_means_of_every_node():
    """The scatter-and-sum-up node means equal the explicit weighted mean of each node's
    cells, and carry a gradient to the cells' rows."""
    pts, nrm, w, _ = cloud(300, d=3, seed=9)
    f = BoundaryForest.build(pts, w, nrm, None, theta=0.5, leaf_size=8, eps=1e-12)
    x = torch.randn(300, 4, 5, dtype=D, requires_grad=True)
    means = tree_read.node_means(x, f, w)
    assert means.shape == (f.n_nodes, 4, 5)
    for i in range(f.n_nodes):
        cells = _node_cells(f, i)
        ref = (x[cells] * w[cells, None, None]).sum(0) / w[cells].sum()
        assert torch.allclose(means[i], ref, atol=1e-5), i
    means.sum().backward()
    ### d(sum of all node means)/dx_j = w_j * sum over the nodes holding j of 1 / measure.
    ref = torch.zeros(300, dtype=D)
    for i in range(f.n_nodes):
        cells = _node_cells(f, i)
        ref[cells] += w[cells] / w[cells].sum()
    assert torch.allclose(x.grad, ref[:, None, None].expand_as(x), atol=1e-5)


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
def test_point_pairs_admit_only_separated_coherent_nodes_and_cover_every_cell(theta):
    """Each point's walk admits a node only when the node's box diagonal is below theta
    times the distance from the point to the box and the node is coherent (RMS normal
    chord below theta), and its cells and nodes together hold every cell exactly once;
    a point's list is the same whether it walks alone or with the others."""
    pts, nrm, w, _ = cloud(700, d=3, seed=6)
    q, _, _, _ = cloud(90, d=3, seed=7, scale=(3.0, 3.0, 3.0))
    f = BoundaryForest.build(pts, w, nrm, None, theta=theta)
    qi, si, n_node = f.point_pairs(q)
    n = f.n_cells
    assert torch.equal(qi, qi.sort().values)
    is_node = si >= n
    assert int(is_node.sum()) == n_node
    node = si[is_node] - n
    p = frame_coordinates(q - f.origin, f.frame)[qi[is_node]]  # the boxes' frame
    gap = torch.clamp(torch.maximum(p - f.aabb_max[node], f.aabb_min[node] - p), min=0)
    assert bool((gap.square().sum(-1) * theta**2 > f.diameter_sq[node]).all())
    assert bool(f.coherent[node].all())
    ### Coverage: a cell is counted through itself or through exactly one node.
    counts = torch.zeros(q.shape[0], n, dtype=torch.long)
    counts.index_put_(
        (qi[~is_node], si[~is_node]), torch.ones_like(qi[~is_node]), accumulate=True
    )
    cells, which = f.node_cells(node)
    counts.index_put_(
        (qi[is_node][which], cells), torch.ones_like(cells), accumulate=True
    )
    assert torch.equal(counts, torch.ones_like(counts))
    if theta == 0.0:
        assert n_node == 0
    ### Query independence of the list itself.
    qi_one, si_one, _ = f.point_pairs(q[17:18])
    assert torch.equal(torch.sort(si_one).values, torch.sort(si[qi == 17]).values)
    assert bool((qi_one == 0).all())


@pytest.mark.parametrize("theta", [0.5, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, "bfloat16 autocast"])
@pytest.mark.parametrize("dim", [2, 3])
def test_fast_walk_gives_the_reference_lists_in_the_reference_order(dim, dtype, theta):
    """The fast walk (Warp; forced here, on CPU) gives the reference walk's lists
    entry for entry and in its order, with the same node count, for points near and
    far from the boundary, a NaN point (opened down to every cell) and an infinite
    one; under bfloat16 autocast (as in training) too, in whatever precision the
    reference forest is built."""
    autocast = dtype == "bfloat16 autocast"
    dtype = torch.float32 if autocast else dtype
    pts, nrm, w, _ = cloud(900, d=dim, seed=3, dtype=dtype)
    q, _, _, _ = cloud(150, d=dim, seed=4, dtype=dtype, scale=(3.0,) * dim)
    q[5] = float("nan")
    q[6, 0] = float("inf")
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        f = BoundaryForest.build(pts, w, nrm, None, theta=theta)
        qi, si, n_node = f.point_pairs(q, walk="reference")
        qi_f, si_f, n_node_f = f.point_pairs(q, walk="fast")
    assert torch.equal(qi_f, qi) and torch.equal(si_f, si)
    assert n_node_f == n_node > 0
    assert torch.equal(torch.sort(si[qi == 5]).values, torch.arange(f.n_cells))


def test_tree_walk_resolution_and_the_models_walks_agree(monkeypatch):
    """``auto`` walks by the reference on CPU and at theta 0, a forced walk is kept and
    an unknown one refused; the model's two walks (the resolvers patched to force
    each, and Warp for the fast one) give the same pairs, so the same outputs
    bitwise."""
    pts, nrm, w, _ = cloud(300, d=3, seed=1)
    f = BoundaryForest.build(pts, w, nrm, None, theta=1.0)
    assert tree_read.resolve_walk("auto", pts, 1.0, f) == "reference"
    assert tree_read.resolve_walk("fast", pts, 1.0, f) == "fast"
    assert tree_read.resolve_walk("fast", pts, 0.0, f) == "reference"
    with pytest.raises(ValueError, match="walk must be"):
        tree_read.resolve_walk("warp", pts, 1.0, f)
    tree = _tree(3, 0.5)
    mesh, drv, q = _sphere_case(n=1500, nq=120)
    runs = {}
    for walk in ("reference", "fast"):
        monkeypatch.setattr(tree_read, "resolve_walk", lambda *a, w=walk: w)
        if walk == "fast":
            monkeypatch.setattr(
                cluster_tree, "_resolve_walk", lambda i, *a: i or "warp"
            )
        with ran(point_walk) as r:
            runs[walk] = outputs(_run(tree, mesh, drv, q))
        assert (r.count(point_walk) > 0) == (walk == "fast")
        assert None not in r.returns(point_walk)  # no silent fallback
    assert all(
        torch.equal(runs["fast"][k], runs["reference"][k]) for k in runs["reference"]
    )


_KERNELS = pytest.mark.parametrize("kernel", ["reference", "fast"])


@_KERNELS
@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
def test_forest_and_every_list_are_the_same_in_every_pose_at_finite_theta(
    improper, kernel
):
    """The trees live in the boundary's own frame (principal axes signed by the third
    moments), so a rotated, reflected, translated and rescaled copy of a generic case
    gives the same frame (moved with the case), the same cells in the same order, the
    same nodes and, for every point, the same far-field list at theta 1."""
    pts, nrm, w, _ = cloud(700, d=3, seed=6)
    q, _, _, _ = cloud(90, d=3, seed=7, scale=(3.0, 3.0, 3.0))
    R = orthogonal(3, 4, improper)
    t = torch.arange(1.0, 4.0, dtype=D)
    k = 2.5
    f0 = BoundaryForest.build(pts, w, nrm, None, theta=1.0, build=kernel)
    f1 = BoundaryForest.build(
        k * pts @ R.T + t, 3.0 * w, nrm @ R.T, None, theta=1.0, build=kernel
    )
    assert torch.allclose(f1.frame, R @ f0.frame, atol=1e-9)
    assert torch.equal(f1.cells, f0.cells)
    assert torch.equal(f1.range_start, f0.range_start) and torch.equal(
        f1.range_count, f0.range_count
    )
    assert torch.equal(f1.coherent, f0.coherent)
    assert torch.allclose(f1.center, k * f0.center @ R.T + t, atol=1e-9)
    assert torch.allclose(f1.normal, f0.normal @ R.T, atol=1e-9)
    qi0, si0, n0 = f0.point_pairs(q, walk=kernel)
    qi1, si1, n1 = f1.point_pairs(k * q @ R.T + t, walk=kernel)
    assert n0 == n1 and torch.equal(qi0, qi1) and torch.equal(si0, si1)


def _lattice_cells():
    """Cells on a product lattice with unit measures, ``(points, normals)`` in float64:
    the coordinates take uneven values per axis, so the cells' covariance is diagonal
    (to float64 roundoff, in any precision of the coordinates) with distinct variances
    and every third moment nonzero: the canonical frame is the coordinate axes, with no
    tie of its own. The normals are the six axis directions and the twelve directions
    at exactly 45 degrees between two axes, so in their own frame they have zero
    components and equal largest components: the orientation split has exact ties. The
    values are generic (no dyadic fractions of the extent, which would sit on lines of
    the trees' Morton grid; the box test's ties have their own test)."""
    axes = [
        torch.tensor(v, dtype=D)
        for v in (
            [0.0, 0.29, 0.61, 1.07, 1.73, 2.03],
            [0.0, 0.37, 0.83, 1.01, 1.39],
            [0.0, 0.53, 0.79, 1.61],
        )
    ]
    pts = torch.cartesian_prod(*axes)
    eye = torch.eye(3, dtype=D)
    dirs = [s * eye[i] for i in range(3) for s in (1.0, -1.0)]
    dirs += [
        (s * eye[i] + t * eye[j]) / math.sqrt(2.0)
        for i, j in ((0, 1), (0, 2), (1, 2))
        for s in (1.0, -1.0)
        for t in (1.0, -1.0)
    ]
    nrm = torch.stack(dirs)[torch.arange(pts.shape[0]) % len(dirs)]
    return pts, nrm


def _lattice_queries():
    g = [
        torch.arange(-4, 13, dtype=D),
        torch.arange(-4, 10, dtype=D),
        torch.arange(-4, 11, dtype=D),
    ]
    return torch.cartesian_prod(*g) * 0.2713 + 0.0917


def _lattice_forest(R=None, t=None, kernel="reference"):
    """The lattice case posed by ``R``, ``t`` in float64 and rounded to float32, its
    forest at the model's angle and every query's list, built and walked by
    ``kernel``."""
    pts, nrm = _lattice_cells()
    q = _lattice_queries()
    if R is not None:
        pts, nrm, q = pts @ R.T + t, nrm @ R.T, q @ R.T + t
    f = BoundaryForest.build(
        pts.float(),
        torch.ones(pts.shape[0]),
        nrm.float(),
        None,
        theta=TREE_READ_THETA,
        build=kernel,
    )
    return f, f.point_pairs(q.float(), walk=kernel)


def _float32_cells(simplex):
    """A float32 mesh's cells as ISLA reads them: centroids, unit normals and areas
    computed in float32, and the bound on each normal's rounding."""
    e1, e2 = simplex[:, 1] - simplex[:, 0], simplex[:, 2] - simplex[:, 0]
    cr = torch.linalg.cross(e1, e2, dim=-1)
    area = 0.5 * cr.norm(dim=-1)
    centroid = simplex.mean(1)
    centre = (area.double()[:, None] * centroid.double()).sum(0) / area.double().sum()
    return (
        centroid,
        cr / cr.norm(dim=-1, keepdim=True),
        area,
        normal_error_bound(simplex, centre),
    )


def test_forest_is_the_same_in_float32_poses_of_a_frame_aligned_case():
    """Posing in float32 a case whose normals lie along its own frame or at exactly 45
    degrees to it (``_lattice_cells``) rounds every input differently; the faces take
    the normals' error bounds, so the posed forest holds the same cells in the same
    trees and nodes and gives every query the same list. Sign octants, decided by the
    sign of the rounding of the zero components, would give different lists."""
    f0, (qi0, si0, n0) = _lattice_forest()
    assert len(f0.trees) == 6 and n0 > 0
    assert torch.allclose(
        f0.frame.abs().amax(0), torch.ones(3, dtype=D)
    )  # the axes, in ascending variance
    for seed in range(4):
        R = orthogonal(3, 20 + seed, seed % 2 == 1)
        t = torch.tensor([0.37, -1.21, 2.03], dtype=D) * (seed + 1)
        f1, (qi1, si1, n1) = _lattice_forest(R, t)
        assert torch.equal(f1.cells, f0.cells), seed
        assert torch.equal(f1.range_start, f0.range_start) and torch.equal(
            f1.range_count, f0.range_count
        )
        assert n1 == n0 and torch.equal(qi1, qi0) and torch.equal(si1, si0), seed


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_cuda])]
)
def test_forest_is_bitwise_the_same_under_bf16_autocast(device):
    """The forest is built and walked outside autocast in the inputs' precision: under
    bf16 autocast its lists, boxes, diameters, measures and virtual cells are bitwise
    those of the float32 build (a training forward reads the forest a float32 score
    reads)."""
    pts, nrm = _lattice_cells()
    R, t = orthogonal(3, 7, False), torch.tensor([1.5, -0.5, 3.0], dtype=D)
    pts, nrm = (pts @ R.T + t).float().to(device), (nrm @ R.T).float().to(device)
    q32, w = (
        (_lattice_queries() @ R.T + t).float().to(device),
        torch.ones(pts.shape[0], device=device),
    )

    def forest():
        f = BoundaryForest.build(
            pts, w, nrm, None, theta=TREE_READ_THETA, build="reference"
        )
        return f, f.point_pairs(q32, walk="reference")

    f0, l0 = forest()
    with torch.autocast(device, dtype=torch.bfloat16):
        f1, l1 = forest()
    assert l0[2] == l1[2] and torch.equal(l0[0], l1[0]) and torch.equal(l0[1], l1[1])
    for name in (
        "cells",
        "aabb_min",
        "aabb_max",
        "diameter_sq",
        "measure",
        "center",
        "normal",
        "coherent",
    ):
        a, b = getattr(f0, name), getattr(f1, name)
        assert a.dtype == b.dtype and torch.equal(a, b), name
    assert f1.aabb_min.dtype == torch.float32 and f1.measure.dtype == torch.float32


def test_forest_is_the_same_under_renumbering_of_the_cells():
    """The forest is a function of the cells' positions, normals and measures, not of
    their order: a permutation of the cells permutes its cell indices and nothing else."""
    pts, nrm = _lattice_cells()
    R, t = orthogonal(3, 9, False), torch.full((3,), 0.5, dtype=D)
    pts, nrm, q = (
        (pts @ R.T + t).float(),
        (nrm @ R.T).float(),
        (_lattice_queries() @ R.T + t).float(),
    )
    w = torch.ones(pts.shape[0])
    f0 = BoundaryForest.build(
        pts, w, nrm, None, theta=TREE_READ_THETA, build="reference"
    )
    qi0, si0, _ = f0.point_pairs(q, walk="reference")
    perm = torch.randperm(pts.shape[0], generator=torch.Generator().manual_seed(4))
    f1 = BoundaryForest.build(
        pts[perm], w, nrm[perm], None, theta=TREE_READ_THETA, build="reference"
    )
    qi1, si1, _ = f1.point_pairs(q, walk="reference")
    assert torch.equal(perm[f1.cells], f0.cells)
    n = pts.shape[0]
    mapped = torch.where(si1 < n, perm[si1.clamp_max(n - 1)], si1)
    order0, order1 = (
        torch.argsort(qi0 * (2 * n + f0.n_nodes) + si0),
        torch.argsort(qi1 * (2 * n + f1.n_nodes) + mapped),
    )
    assert torch.equal(qi0[order0], qi1[order1]) and torch.equal(
        si0[order0], mapped[order1]
    )


def test_normal_error_bound_covers_the_rounding_of_rotated_float32_normals():
    """Each component of a float32 cell normal, computed from its rotated and rounded
    vertices, is within the bound of the exact normal of the exact rotated cell, for
    cells from equilateral to thin."""
    g = torch.Generator().manual_seed(5)
    base = torch.rand(4000, 3, 3, generator=g, dtype=D) * 2 - 1
    thin = base.clone()
    thin[:2000, 2] = thin[:2000, 0] + 1e-3 * (thin[:2000, 2] - thin[:2000, 0])
    for seed in range(3):
        R = orthogonal(3, 30 + seed, False)
        exact = thin @ R.T
        n64 = torch.linalg.cross(
            exact[:, 1] - exact[:, 0], exact[:, 2] - exact[:, 0], dim=-1
        )
        n64 = n64 / n64.norm(dim=-1, keepdim=True)
        _, n32, _, tol = _float32_cells(exact.float())
        assert bool(((n32.double() - n64).abs().amax(-1) <= tol).all())
        assert (
            float(((n32.double() - n64).abs().amax(-1) / tol).max()) > 1e-3
        )  # not vacuous


def _tie_queries(f, nodes, theta):
    """For each node, the float32 lab point at exactly ``1 / theta`` times its box
    diagonal from the box, beyond the middle of its upper face along the frame's first
    axis (computed in float64, then rounded)."""
    lo, hi = f.aabb_min[nodes].double(), f.aabb_max[nodes].double()
    diam = (hi - lo).norm(dim=-1)
    q_f = 0.5 * (lo + hi)
    q_f[:, 0] = hi[:, 0] + diam / theta
    return (f.origin + q_f @ f.frame.t()).float()


def test_box_test_ties_are_opened():
    """A point at exactly theta times a node's box diagonal from its box is a tie of the
    admission criterion, which rounding of the point would decide either way: the walk
    never admits the node for it (it reads the node's children or cells instead)."""
    pts, nrm, w, _ = cloud(2000, d=3, seed=12)
    f = BoundaryForest.build(
        pts.float(),
        w.float(),
        nrm.float(),
        None,
        theta=TREE_READ_THETA,
        build="reference",
    )
    inner = (f.left >= 0).nonzero().squeeze(-1)
    nodes = inner[
        torch.randperm(inner.numel(), generator=torch.Generator().manual_seed(1))[:400]
    ]
    q = _tie_queries(f, nodes, TREE_READ_THETA)
    qi, si, _ = f.point_pairs(q, walk="reference")
    admitted = (si == f.n_cells + nodes[qi]).sum()
    assert int(admitted) == 0
    ### A point a little beyond the tie (1 % farther) reads the node whole.
    far = _tie_queries(f, nodes, TREE_READ_THETA / 1.01)
    qi, si, _ = f.point_pairs(far, walk="reference")
    assert float((si == f.n_cells + nodes[qi]).sum()) >= 0.9 * nodes.numel()


def _nearly_tied_ellipsoid(n, gap_over_bound, seed):
    r"""A float32 boundary like a near-spherical far field: ``n`` cells of equal measure
    on an ellipsoid of semi-axes 1, 1 and 1.5, normals inward, and 400 points inside it
    near the wall (at 0.8 to 0.99 of its radius), in a random orientation. The two lower
    variances of the cells' covariance are set apart by ``gap_over_bound`` times the
    rounding bound :math:`E = 4 u \operatorname{tr} C` of
    :func:`~physicsnemo.experimental.models.isla._tree_read.frame_rotation_bound`, so the
    canonical frame's two lower axes tie to within that many times their rounding, as
    those of spherical and hemispherical far fields do."""
    g = torch.Generator().manual_seed(seed)
    axes = torch.tensor([1.0, 1.0, 1.5], dtype=D)
    x = torch.nn.functional.normalize(torch.randn(n, 3, generator=g, dtype=D), dim=-1)
    nrm = -x
    x = x * axes
    c = x.mean(0)
    lam, V = torch.linalg.eigh((x - c).t() @ (x - c) / n)
    y = (x - c) @ V
    y[:, 1] *= math.sqrt((lam[0] + gap_over_bound * 4 * 2.0**-24 * lam.sum()) / lam[1])
    R = orthogonal(3, seed + 1)
    d = torch.nn.functional.normalize(torch.randn(400, 3, generator=g, dtype=D), dim=-1)
    q = d * axes * (0.8 + 0.19 * torch.rand(400, 1, generator=g, dtype=D))
    w = torch.full((n,), 1.0 / n)
    return (y @ R.T).float(), (nrm @ V @ R.T).float(), w, ((q - c) @ V @ R.T).float()


def test_a_nearly_tied_frame_widens_the_separation_test_by_the_nodes_own_size(
    monkeypatch,
):
    """When the canonical frame's variances nearly tie, its rotation bound eta is large
    (about 0.05 here). The rotation moves the points and the cells together, so it
    widens each node's separation diameter in proportion to the node's own size, and
    the pairs stay within 1.4 times those of the same forest with eta at zero. An error
    of eta times each coordinate's distance from the centroid (kappa = 3u + eta) would
    open the small nodes near the wall for the points beside them: 3.0 times the
    pairs."""
    pts, nrm, w, q = _nearly_tied_ellipsoid(6000, 21.0, seed=3)
    seen = {}
    bound = tree_read.frame_rotation_bound

    def spy(x, wt, u):
        seen["eta"] = float(bound(x, wt, u))
        return bound(x, wt, u)

    monkeypatch.setattr(tree_read, "frame_rotation_bound", spy)
    tied = BoundaryForest.build(pts, w, nrm, None, theta=TREE_READ_THETA)
    assert 0.03 < seen["eta"] < 0.08
    monkeypatch.setattr(
        tree_read, "frame_rotation_bound", lambda x, wt, u: torch.zeros((), dtype=D)
    )
    exact = BoundaryForest.build(pts, w, nrm, None, theta=TREE_READ_THETA)
    n_tied, n_exact = tied.point_pairs(q)[0].numel(), exact.point_pairs(q)[0].numel()
    assert n_tied <= 1.6 * n_exact, (n_tied, n_exact)


def _admitted_boxes_in(f, q, qi, si, frame_map):
    """For every admitted (point, node) pair, the squared gap from the point to the
    node's box and the box's squared diagonal, both in the coordinates ``frame_map``
    gives the trees' coordinates (float64)."""
    n = f.n_cells
    is_node = si >= n
    node = si[is_node] - n
    cells, which = f.node_cells(node)
    pts_f = frame_map(frame_coordinates(f._points - f.origin, f.frame))
    p = frame_map(frame_coordinates(q.double() - f.origin, f.frame))[qi[is_node]]
    lo = torch.full((node.numel(), 3), math.inf, dtype=D).scatter_reduce(
        0, which[:, None].expand(-1, 3), pts_f[cells], "amin"
    )
    hi = torch.full((node.numel(), 3), -math.inf, dtype=D).scatter_reduce(
        0, which[:, None].expand(-1, 3), pts_f[cells], "amax"
    )
    gap = torch.clamp(torch.maximum(p - hi, lo - p), min=0).square().sum(-1)
    return gap, (hi - lo).square().sum(-1)


@pytest.mark.parametrize("eta", [0.0, 0.01, 0.2, 2.0])
def test_admitted_nodes_are_separated_in_every_frame_within_the_rotation_bound(
    eta, monkeypatch
):
    """The separation diameter certifies the criterion in the exact frame: for every
    node a point admits, the point's distance from the node's box exceeds the box's
    diagonal over theta in every frame rotated from the trees' by at most eta per
    coordinate (any frame once eta is 2: the ball about the box's centre certifies it
    there)."""
    monkeypatch.setattr(
        tree_read, "frame_rotation_bound", lambda x, wt, u: torch.tensor(eta, dtype=D)
    )
    pts, nrm, w, _ = cloud(800, d=3, seed=6)
    q, _, _, _ = cloud(150, d=3, seed=7, scale=(3.0, 3.0, 3.0))
    f = BoundaryForest.build(pts, w, nrm, None, theta=TREE_READ_THETA)
    f._points = pts
    qi, si, n_node = f.point_pairs(q)
    assert n_node > 0
    g = torch.Generator().manual_seed(2)
    for k in range(6):
        if eta >= 2.0:
            Q = orthogonal(3, 40 + k)
        else:
            ### a rotation by the angle whose chord is eta, about a random axis
            axis = torch.nn.functional.normalize(
                torch.randn(3, generator=g, dtype=D), dim=0
            )
            phi = 2 * math.asin(eta / 2)
            K = torch.tensor(
                [
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0],
                ],
                dtype=D,
            )
            Q = torch.eye(3, dtype=D) + math.sin(phi) * K + (1 - math.cos(phi)) * K @ K
        gap, diag = _admitted_boxes_in(f, q, qi, si, lambda x: x @ Q.T)
        assert bool((gap * TREE_READ_THETA**2 > diag).all()), k


def test_separation_diameter_is_bounded_as_the_frame_ties():
    """The separation diameter grows with the frame's rotation bound, never past the
    frame-free ball bound, and in proportion to the node's own size: a node far from the
    centroid widens as much as one near it."""
    pts, nrm, w, _ = cloud(600, d=3, seed=9)
    f = BoundaryForest.build(pts.float(), w.float(), nrm.float(), None, theta=1.0)
    tree = f.trees[0]
    near = (tree.node_aabb_min, tree.node_aabb_max)
    far = (tree.node_aabb_min + 1e3, tree.node_aabb_max + 1e3)
    u = 2.0**-24
    sep = tree_read.separation_diameter_sq
    etas = [0.0, 1e-4, 0.01, 0.05, 0.3, 0.57, 0.58, 1.0, 10.0]
    near_d = [sep(*near, 1.0, torch.tensor(e, dtype=D), 3, u).double() for e in etas]
    far_d = [sep(*far, 1.0, torch.tensor(e, dtype=D), 3, u).double() for e in etas]
    for a, b in zip(near_d, near_d[1:]):
        assert bool((b >= a).all())
    ### Past sqrt(3) eta = 1 no rotation bound certifies anything, and the ball does.
    assert torch.equal(near_d[-1], near_d[-2]) and torch.equal(near_d[-2], near_d[-3])
    diam_sq = (tree.node_aabb_max - tree.node_aabb_min).double().square().sum(-1)
    big = diam_sq > 1e-4
    ball = (3 * math.sqrt(3) / 2) ** 2 * diam_sq[big]
    assert torch.allclose(near_d[-1][big], ball, rtol=1e-4)
    ### At eta 0.05 the widening is about f = (1 + 2 sqrt(3) eta) / (1 - sqrt(3) eta),
    ### the same near the centroid and 1000 units away (up to rounding's own share).
    f05 = (1 + 2 * math.sqrt(3) * 0.05) / (1 - math.sqrt(3) * 0.05)
    for d_ in (near_d, far_d):
        ratio = (d_[3][big] / d_[0][big]).sqrt()
        assert bool((ratio <= f05 * (1 + 1e-3)).all()), float(ratio.max())


def test_model_forward_is_the_same_in_float32_poses_of_a_frame_aligned_case(
    monkeypatch,
):
    """The interior model on a float32 boundary whose normals lie along its own frame
    or at exactly 45 degrees to it (``_lattice_cells`` as small triangles): the forest
    of a posed copy, built from the posed mesh's normals with the bounds on their
    rounding, gives every point the same pairs, so the outputs equal the native ones
    to float32 roundoff."""
    pts, nrm = _lattice_cells()
    mesh = soup(pts, nrm, torch.full((pts.shape[0],), 1e-3, dtype=D))
    q = _lattice_queries()[::7] + 0.013
    drv = torch.tensor([0.3, 1.0, 0.2], dtype=D)
    m = _model().eval()
    plans = []
    inner = model_mod.walk_range_pairs

    def recording(*a, **kw):
        out = inner(*a, **kw)
        plans.append((out.plan.qi.clone(), out.plan.bj.clone()))
        return out

    monkeypatch.setattr(model_mod, "walk_range_pairs", recording)
    f = lambda t: t.float()  # noqa: E731

    def f32(mm):
        data = {k: f(v) for k, v in mm.cell_data.items()}
        return Mesh(points=f(mm.points), cells=mm.cells, cell_data=data)

    base = outputs(_run(m, f32(mesh), f(drv), f(q)))
    for seed in range(2):
        R = orthogonal(3, 40 + seed, False)
        t = torch.tensor([0.37, -1.21, 2.03], dtype=D) * (seed + 1)
        posed = f32(transform(mesh, R, t))
        out = outputs(_run(m, posed, f(drv @ R.T), f(q @ R.T + t)))
        assert torch.equal(plans[-1][0], plans[0][0]) and torch.equal(
            plans[-1][1], plans[0][1]
        ), seed
        assert float((out["p"] - base["p"]).norm() / base["p"].norm()) < 1e-5
        rot = base["v"] @ R.T.float()
        assert float((out["v"] - rot).norm() / rot.norm()) < 1e-5


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
def test_tree_read_is_exactly_query_independent_at_every_theta(theta):
    """A prediction depends on its own point and the boundary only, whatever theta (a
    point's walk of the tree does not see the other points): alone, a prefix, permuted,
    the same rows to 1e-11."""
    tree = _tree(3, theta)
    mesh, drv, q = _case()
    full = outputs(_run(tree, mesh, drv, q))
    idx = torch.tensor([3, 50, 117])
    alone = outputs(_run(tree, mesh, drv, q[idx]))
    assert same(alone, _rows(full, idx), atol=1e-11)
    perm = torch.randperm(q.shape[0], generator=torch.Generator().manual_seed(5))
    permuted = outputs(_run(tree, mesh, drv, q[perm]))
    assert same(permuted, _rows(full, perm), atol=1e-11)
    half = q.shape[0] // 2
    prefix = outputs(_run(tree, mesh, drv, q[:half]))
    assert same(prefix, _rows(full, torch.arange(half)), atol=1e-11)


def test_tree_read_similarity_refinement_and_measure_scale_at_theta_zero():
    """At theta 0 the read is the exact read, so the outputs stay the same (to 1e-10)
    when the case is scaled by 2.5, refined (each cell as two half-measure copies) or
    given three times every cell's measure."""
    tree = _tree(3, 0.0)
    mesh, drv, q = _case()
    base = outputs(_run(tree, mesh, drv, q))
    k = 2.5
    scaled = outputs(_run(tree, transform(mesh, scale=k), drv, k * q))
    assert same(scaled, base, atol=1e-10)
    refined = outputs(_run(tree, refine(mesh), drv, q))
    assert same(refined, base, atol=1e-10)
    data = dict(mesh.cell_data.items())
    data[EFFECTIVE_MEASURE_KEY] = data[EFFECTIVE_MEASURE_KEY] * 3.0
    rescaled = outputs(_run(tree, mesh.with_data(cell_data=data), drv, q))
    assert same(rescaled, base, atol=1e-10)


#: The default read's deviation from the exact read (theta = 0), measured on CPU in
#: float64: the cloud case 3.0e-2 and refined 3.5e-2, the sphere 1.4e-2
#: both. The bounds are about 1.4 times the larger of each pair.
@pytest.mark.parametrize("case, seed, bound", [("cloud", 0, 0.05), ("sphere", 1, 0.02)])
def test_tree_read_at_the_default_theta_under_refinement_and_similarity(
    case, seed, bound
):
    """At the default opening angle the read is within a measured bound of the exact
    read, for the case and for its refinement (each cell as two half-measure copies,
    which the forest sees and the exact read does not). On the sphere, whose smooth
    normals make the nodes good far-field surrogates, the error at theta 0.25 is
    below the default's. Similar copies (lengths times 1e-6 and 1e6) give the same
    approximation to roundoff: the forest and the opening test use frame units only."""
    mesh, drv, q = (_case if case == "cloud" else _sphere_case)()
    exact = outputs(_run(_tree(3, 0.0, seed=seed), mesh, drv, q))
    tree = _tree(3, TREE_READ_THETA, seed=seed)
    base = outputs(_run(tree, mesh, drv, q))
    refined = outputs(_run(tree, refine(mesh), drv, q))
    assert _rel(base, exact) < bound, _rel(base, exact)
    assert _rel(refined, exact) < bound, _rel(refined, exact)
    if case == "sphere":
        tree.tree_read_theta = 0.25
        finer = outputs(_run(tree, mesh, drv, q))
        tree.tree_read_theta = TREE_READ_THETA
        assert _rel(finer, exact) < _rel(base, exact)
    for k in (1e-6, 1e6):
        scaled = outputs(_run(tree, transform(mesh, scale=k), drv, k * q))
        assert _rel(scaled, base) < 1e-12, (k, _rel(scaled, base))


@pytest.mark.parametrize("randomize", [False, True], ids=["initial", "random"])
def test_tree_read_every_parameter_receives_a_gradient(randomize):
    """Every parameter gets a finite gradient (as DDP needs), from the read's
    zero-initialized output projection too; with random read weights every read and
    basis parameter's gradient is nonzero."""
    m = _model(3, 0.5, randomize=randomize).double()
    mesh, drv, q = _case()
    out = m(
        boundary_meshes={"wall": mesh},
        prediction_points=points_mesh(q),
        global_data={"g": drv},
    )
    loss = sum(v.square().sum() for v in outputs(out).values())
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    if randomize:
        assert all(
            p.grad.abs().sum() > 0
            for n_, p in m.named_parameters()
            if n_.startswith("kernel_reads") or n_.startswith("kernel_basis")
        )


def test_tree_read_starts_as_no_update_and_is_live_everywhere():
    """With the zero-initialized output projection the read adds nothing (the model
    equals itself without the read); with trained-like weights it changes every
    point, the far ones included (there is no radius)."""
    mesh, drv, q = _case()
    for randomize in (False, True):
        m = _model(randomize=randomize).double().eval()
        no_read = copy.deepcopy(m)
        for read in no_read.kernel_reads:
            read.read = types.MethodType(
                lambda self, kv, h_q, pairs, compiled=False: torch.zeros_like(h_q), read
            )
        a, b = _run(m, mesh, drv, q), _run(no_read, mesh, drv, q)
        if not randomize:
            assert same(a, outputs(b), atol=1e-12, rtol=0)
        else:
            far = slice(q.shape[0] - 10, None)
            diff = (a.point_data["p"] - b.point_data["p"]).abs()
            assert bool((diff[far] > 1e-8).all()), diff[far]
            assert bool((diff > 1e-8).all())


def test_tree_pair_invariants_are_scale_free_and_the_near_wall_invariant():
    """The invariants: r / L, log(r / L) and log(r / |d_q|), which is zero at the
    point's nearest cell and positive elsewhere; a point on a cell centroid
    (d_q = 0) stays finite (the numerical floor)."""
    g = torch.Generator().manual_seed(0)
    y = torch.randn(50, 3, generator=g, dtype=D)
    n = torch.nn.functional.normalize(torch.randn(50, 3, generator=g, dtype=D), dim=-1)
    x = torch.cat([y[:1], torch.randn(20, 3, generator=g, dtype=D)])
    d = torch.cdist(x, y).min(-1).values
    nq = torch.nn.functional.normalize(x - y[torch.cdist(x, y).argmin(-1)], dim=-1)
    gu = torch.nn.functional.normalize(torch.randn(1, 3, generator=g, dtype=D), dim=-1)
    qi, si = (
        t.reshape(-1)
        for t in torch.meshgrid(torch.arange(21), torch.arange(50), indexing="ij")
    )

    def inv_at(k):
        geo = dict(
            query_r=k * x,
            query_n=nq,
            query_wall=k * d,
            src_r=k * y,
            src_n=n,
            src_vecs=None,
            g_unit=gu,
            eps=1e-12,
        )
        return tree_pair_invariants(qi, si, **geo).reshape(21, 50, -1)

    inv = inv_at(1.0)
    assert inv.shape == (21, 50, kernel_geo_width(1, 0)) and torch.isfinite(inv).all()
    r = torch.cdist(x, y)
    assert torch.allclose(inv[..., 0], r) and torch.allclose(inv[1:, :, 1], r[1:].log())
    near = inv[1:, :, 2]
    assert near.min() > -1e-12 and torch.allclose(
        near.min(-1).values, torch.zeros(20, dtype=D), atol=1e-12
    )
    assert torch.allclose(inv_at(3.1)[1:, :, 2:], inv[1:, :, 2:], atol=1e-12)


def test_tree_read_bf16_autocast_is_finite():
    """Under bf16 autocast the outputs and every gradient are finite, at points on a
    cell centroid (d_q = 0) too."""
    m = _model()
    mesh, drv, q = _case()
    q = torch.cat([q, mesh.cell_centroids[:2]])  # points on a cell centroid: d_q = 0
    f = lambda t: t.float()  # noqa: E731
    bm = {
        "wall": soup(
            f(mesh.cell_centroids),
            f(mesh.cell_normals),
            f(mesh.cell_data[EFFECTIVE_MEASURE_KEY]),
        )
    }
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(
            boundary_meshes=bm,
            prediction_points=points_mesh(f(q)),
            global_data={"g": f(drv)},
        )
    sum(o.float().sum() for o in outputs(out).values()).backward()
    assert all(torch.isfinite(o.float()).all() for o in outputs(out).values())
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )


@pytest.mark.cuda
@requires_cuda_triton
def test_tree_read_cuda_fused_and_compiled_training_path():
    """The protocol's training path on CUDA: fused geometry kernel, compiled blocks,
    bf16 autocast. One training step is finite with every parameter's gradient
    present, the compiled forward matches the eager one to bf16 roundoff, and the
    prediction stays query-independent (float32 roundoff)."""
    dev = "cuda"
    m_e = _model().to(dev)
    m_c = _model(compile_blocks=True).to(dev)
    m_c.load_state_dict(m_e.state_dict())
    mesh, drv, q = _case(n=600, nq=400)
    f = lambda t: t.float().to(dev)  # noqa: E731
    bm = {
        "wall": soup(
            mesh.cell_centroids.float(),
            mesh.cell_normals.float(),
            mesh.cell_data[EFFECTIVE_MEASURE_KEY].float(),
        ).to(dev)
    }
    pp = points_mesh(f(q))
    outs = []
    for m in (m_e, m_c):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = outputs(
                m(boundary_meshes=bm, prediction_points=pp, global_data={"g": f(drv)})
            )
        sum(o.float().square().mean() for o in out.values()).backward()
        assert all(torch.isfinite(o.float()).all() for o in out.values())
        missing = [n for n, p in m.named_parameters() if p.grad is None]
        assert not missing, missing
        outs.append(out)
    assert m_e._resolve_geo_kernel(bm["wall"].points[None]) == "fused"
    for k in outs[0]:
        a, b = outs[0][k].float(), outs[1][k].float()
        assert (a - b).norm() <= 3e-2 * a.norm() + 1e-6, k
    with torch.no_grad():
        full = outputs(
            m_e(boundary_meshes=bm, prediction_points=pp, global_data={"g": f(drv)})
        )
        part = outputs(
            m_e(
                boundary_meshes=bm,
                prediction_points=points_mesh(f(q[:50])),
                global_data={"g": f(drv)},
            )
        )
    assert all(
        torch.allclose(part[k], full[k][:50], atol=1e-4, rtol=1e-4) for k in part
    )


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_cuda])],
)
def test_pair_plan_is_the_histogram_plan_and_builds_without_a_host_sync(device):
    """The plan's offsets and stable permutation equal a histogram of each index and a
    stable sort of the source indices, field by field and in dtype, for 64- and 32-bit
    indices, with points and sources that have no pairs (the last ones among them); on
    CUDA it builds without a host sync."""
    from physicsnemo.experimental.models.isla._segment_softmax import PairPlan

    g = torch.Generator().manual_seed(4)
    n_q, n_b = 300, 5000
    counts = torch.randint(0, 900, (n_q,), generator=g)
    counts[::7] = 0
    counts[-3:] = 0
    qi = torch.repeat_interleave(torch.arange(n_q), counts).to(device)
    bj = torch.randint(0, n_b - 40, (qi.shape[0],), generator=g).to(device)
    bj[: qi.shape[0] // 3] = 17  # one source read by a third of the pairs

    def offsets(idx, n):
        o = idx.new_zeros(n + 1)
        o[1:] = torch.cumsum(torch.bincount(idx, minlength=n), 0)
        return o

    for qi, bj in ((qi, bj), (qi.int(), bj.int())):
        perm = torch.argsort(bj, stable=True)
        ref = (offsets(qi, n_q), perm, offsets(bj, n_b), qi[perm])
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                plan = PairPlan.build(qi, bj, n_q, n_b)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        else:
            plan = PairPlan.build(qi, bj, n_q, n_b)
        got = (plan.q_offsets, plan.perm, plan.b_offsets, plan.qi_by_b)
        for name, a, b in zip(("q_offsets", "perm", "b_offsets", "qi_by_b"), got, ref):
            assert a.dtype == b.dtype and torch.equal(a, b), (name, qi.dtype)


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dim", [2, 3])
def test_fast_walk_on_cuda_gives_the_reference_lists_in_the_reference_order(dim, dtype):
    """On CUDA, ``auto`` picks the fast walk, which gives the reference walk's lists
    on the same device entry for entry and in its order."""
    dev = torch.device("cuda")
    pts, nrm, w, _ = cloud(3000, d=dim, seed=8, dtype=dtype)
    q, _, _, _ = cloud(2000, d=dim, seed=9, dtype=dtype, scale=(3.0,) * dim)
    pts, nrm, w, q = (t.to(dev) for t in (pts, nrm, w, q))
    f = BoundaryForest.build(pts, w, nrm, None, theta=TREE_READ_THETA)
    assert tree_read.resolve_walk("auto", q, TREE_READ_THETA, f) == "fast"
    qi, si, n_node = f.point_pairs(q, walk="reference")
    qi_f, si_f, n_node_f = f.point_pairs(q)
    assert torch.equal(qi_f, qi) and torch.equal(si_f, si)
    assert n_node_f == n_node > 0


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("theta", [0.0, 0.5])
def test_tree_read_cuda_matches_cpu_float64(theta, monkeypatch):
    """The float64 model on CUDA against CPU: both walk the forest to the same pair
    lists (every opening decision agrees), and the outputs agree to 1e-9, at
    theta = 0 (cells only) and at theta > 0, whose virtual cells' means are float64
    in a float64 model (node_means)."""
    import physicsnemo.experimental.models.isla.model as isla_model

    plans = []

    def capture(*args, **kwargs):
        pairs = tree_read.walk_range_pairs(*args, **kwargs)
        plans.append(set(zip(pairs.plan.qi.tolist(), pairs.plan.bj.tolist())))
        return pairs

    monkeypatch.setattr(isla_model, "walk_range_pairs", capture)
    tree = _tree(3, theta)
    mesh, drv, q = _sphere_case(n=1500, nq=120)
    cpu = outputs(_run(tree, mesh, drv, q))
    dev = torch.device("cuda")
    tree_c = tree.to(dev)
    gpu = outputs(_run(tree_c, mesh.to(dev), drv.to(dev), q.to(dev)))
    assert len(plans) == 2 and plans[0] == plans[1]  # the same opening decisions
    gpu = {k: v.cpu() for k, v in gpu.items()}
    assert same(gpu, cpu, atol=1e-9)


_TREE_FIELDS = (
    "node_aabb_min",
    "node_aabb_max",
    "node_diameter_sq",
    "node_left_child",
    "node_right_child",
    "leaf_start",
    "leaf_count",
    "node_range_start",
    "node_range_count",
    "node_total_area",
    "sorted_source_order",
    "source_points",
    "max_depth",
    "root_ids",
)
_FOREST_FIELDS = (
    "frame",
    "origin",
    "node_offset",
    "cells",
    "range_start",
    "range_count",
    "left",
    "right",
    "leaf_ids",
    "position_leaf",
    "aabb_min",
    "aabb_max",
    "diameter_sq",
    "center",
    "normal",
    "vecs",
    "measure",
    "coherent",
)


def _bitwise(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.is_floating_point():
        ints = {8: torch.int64, 4: torch.int32, 2: torch.int16}[a.element_size()]
        a, b = a.contiguous().view(ints), b.contiguous().view(ints)
    return torch.equal(a, b)


def _forest_differences(ref, fast) -> list[str]:
    """The fields in which two forests differ in any bit, their trees' included."""
    bad = [k for k in _FOREST_FIELDS if not _bitwise(getattr(ref, k), getattr(fast, k))]
    if len(ref.levels) != len(fast.levels) or not all(
        _bitwise(x, y) for a, b in zip(ref.levels, fast.levels) for x, y in zip(a, b)
    ):
        bad.append("levels")
    if len(ref.trees) != len(fast.trees):
        return [*bad, "trees"]
    for t, (a, b) in enumerate(zip(ref.trees, fast.trees)):
        bad += [
            f"trees[{t}].{k}"
            for k in _TREE_FIELDS
            if not _bitwise(getattr(a, k), getattr(b, k))
        ]
        if not _bitwise(ref.tree_cells[t], fast.tree_cells[t]):
            bad.append(f"tree_cells[{t}]")
    return bad


def _build_cases(dim, dtype):
    """Boundaries for the fast build: a random cloud with rank-1 inputs, half of it on
    one point (equal Morton codes, midpoint splits), and tiny ones (a tree of one cell,
    trees no larger than a leaf), at three leaf sizes."""
    pts, nrm, w, _ = cloud(1500, d=dim, seed=21, dtype=dtype)
    vecs = torch.randn(1500, 2, dim, dtype=dtype)
    dup = pts.clone()
    dup[:700] = dup[0]
    yield pts, w, nrm, vecs, 4
    yield dup, w, nrm, None, 4
    yield pts, w, nrm, vecs[:, :1], 1
    yield pts, w, nrm, None, 8
    for n in (1, 3, 9, 40):
        yield pts[:n], w[:n], nrm[:n], vecs[:n], 4


def _check_fast_build(dim, dtype, device, autocast):
    for pts, w, nrm, vecs, leaf in _build_cases(dim, dtype):
        pts, w, nrm = pts.to(device), w.to(device), nrm.to(device)
        vecs = None if vecs is None else vecs.to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
            forests = [
                BoundaryForest.build(
                    pts, w, nrm, vecs, theta=1.0, leaf_size=leaf, build=b
                )
                for b in ("reference", "fast")
            ]
            q = pts[:50] * 1.5 + 0.25
            (qa, sa, na), (qb, sb, nb) = (f.point_pairs(q) for f in forests)
        ### The fast build follows the precision the reference builds in (under
        ### autocast too): _forest_differences compares dtypes as well as bits.
        assert not _forest_differences(*forests), (pts.shape[0], leaf)
        assert torch.equal(qa, qb) and torch.equal(sa, sb) and na == nb


_PRECISIONS = pytest.mark.parametrize(
    "dtype, autocast",
    [(torch.float32, False), (D, False), (torch.float32, True)],
    ids=["float32", "float64", "bf16-autocast"],
)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_cuda])]
)
@_PRECISIONS
@pytest.mark.parametrize("dim", [3, 2])
def test_fast_forest_build_is_the_reference_forest(dim, dtype, autocast, device):
    """The fast build (one pass over all faces; on CPU, Warp's CPU backend) gives the
    reference's forest in every bit, under bfloat16 autocast too: node table, levels,
    boxes, diameters, virtual cells, measures, coherence and every tree's own fields,
    so the reference walk makes the same pair lists on it. On CUDA "auto" picks it."""
    if device == "cuda":
        assert (
            tree_build.resolve_tree_build("auto", torch.zeros(1, dim, device="cuda"))
            == "fast"
        )
    _check_fast_build(dim, dtype, device, autocast)


def test_fast_forest_build_makes_its_trees_on_first_use():
    """The fast build makes no per-face ``ClusterTree``: the fast walk reads the tables
    alone, and the first reference walk makes the trees, once, and walks the same
    lists."""
    pts, nrm, w, _ = cloud(1500, d=3, seed=21, dtype=torch.float32)
    q = pts[:50] * 1.5 + 0.25
    f = BoundaryForest.build(
        pts, w, nrm, None, theta=1.0, groups=torch.arange(1500) % 3, build="fast"
    )
    fast = f.point_pairs(q, walk="fast")
    assert "_per_tree" not in vars(f)
    ref = f.point_pairs(q, walk="reference")
    assert "_per_tree" in vars(f) and f.trees is f.trees
    assert len(f.trees) == len(f.tree_cells) == int(f.node_offset.shape[0]) - 1
    assert ref[2] == fast[2] > 0
    assert torch.equal(ref[0], fast[0]) and torch.equal(ref[1], fast[1])


def test_tree_build_auto_resolves_to_the_reference_off_cuda(monkeypatch):
    """The build's "auto" is the reference build off CUDA, a forced "fast" (the
    resolver patched) gives the model the same outputs, and an unknown name is
    refused."""
    from physicsnemo.experimental.models.isla._tree_build import resolve_tree_build

    pts = torch.zeros(4, 3)
    assert resolve_tree_build("auto", pts) == "reference"
    assert resolve_tree_build("fast", pts) == "fast"
    with pytest.raises(ValueError, match="build must be"):
        resolve_tree_build("gpu", pts)
    m = _tree(3, 1.0)
    mesh, drv, q = _case(n=400, nq=60)
    ref = outputs(_run(m, mesh, drv, q))
    monkeypatch.setattr(tree_read, "resolve_tree_build", lambda *a: "fast")
    with ran(tree_build.forest_tables, ClusterTree.from_points) as r:
        fast = outputs(_run(m, mesh, drv, q))
    tables = r.returns(tree_build.forest_tables)
    assert tables and None not in tables and r.count(ClusterTree.from_points) == 0
    assert all(torch.equal(v, fast[k]) for k, v in ref.items())


def test_tree_read_kernel_is_the_tree_reads_one_execution_attribute(monkeypatch):
    """``_tree_read_kernel`` is an execution choice, not in the state dict: "auto" by
    default hands "auto" to the forest build and the walk, "reference" hands them
    "reference" and keeps the reference read, and any other name is refused."""
    seen = []
    build, walk = tree_read.resolve_tree_build, tree_read.resolve_walk
    monkeypatch.setattr(
        tree_read, "resolve_tree_build", lambda b, *a: seen.append(b) or build(b, *a)
    )
    monkeypatch.setattr(
        tree_read, "resolve_walk", lambda w, *a: seen.append(w) or walk(w, *a)
    )
    m = _tree(3, 1.0)
    keys = list(m.state_dict())
    mesh, drv, q = _case(n=400, nq=60)
    assert m._tree_read_kernel == "auto"
    for mode in ("auto", "reference"):
        m._tree_read_kernel = mode
        seen.clear()
        _run(m, mesh, drv, q)
        ### the forest build, the walk counting the pairs and the walk of the points
        assert len(seen) == 3 and set(seen) == {mode}
        assert m._resolve_tree_read_kernel(q) == "reference"
    assert list(m.state_dict()) == keys
    for bad in ("fused", "fast"):
        m._tree_read_kernel = bad
        with pytest.raises(ValueError, match="tree_read_kernel must be"):
            _run(m, mesh, drv, q)


def test_tree_build_auto_logs_its_choice_in_another_thread_while_one_compiles(
    caplog, monkeypatch
):
    """The build's "auto" logs its choice in eager code that runs while another
    thread compiles (a data-loading thread, say): only a trace skips the log."""
    monkeypatch.setattr(tree_build, "_LOGGED", set())
    caplog.set_level(logging.INFO, logger=tree_build.__name__)
    chosen = in_another_thread_while_compiling(
        lambda: tree_build.resolve_tree_build("auto", torch.zeros(4, 3))
    )
    assert chosen == "reference"
    logged = [r.getMessage() for r in caplog.records]
    assert sum("tree_build=auto resolved to 'reference'" in m for m in logged) == 1


def test_fast_forest_build_refuses_what_it_cannot_build():
    """A forced "fast" build refuses a dimension it has no Morton code for ("auto"
    picks the reference there)."""
    import physicsnemo.experimental.models.isla._tree_build as tree_build

    with pytest.raises(ValueError, match="does not take 4-D points"):
        tree_build.resolve_tree_build("fast", torch.zeros(4, 4))
    assert tree_build.resolve_tree_build("auto", torch.zeros(4, 4)) == "reference"


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_cuda])]
)
def test_fast_forest_and_walk_are_the_reference_on_face_and_box_ties(device):
    """On the cases where rounding would decide the forest (``_lattice_cells``: normals
    along the frame or at exactly 45 degrees to it, native, in the poses of the forest
    tests above, posed in float32 and renumbered, under bf16 autocast too; and points
    at a box-test tie), the fast build and walk give the reference's forest in every
    bit and its lists entry for entry. So the forest tests above, which build and walk
    by the reference, hold for the fast build and walk too."""
    pts0, nrm0 = _lattice_cells()
    q0 = _lattice_queries()
    perm = torch.randperm(pts0.shape[0], generator=torch.Generator().manual_seed(4))
    cases = [(pts0, nrm0, q0)]
    ### the poses of the reference-only forest tests above
    shift = torch.tensor([0.37, -1.21, 2.03], dtype=D)
    poses = [(orthogonal(3, 20 + i, i % 2 == 1), shift * (i + 1)) for i in range(4)]
    poses += [
        (orthogonal(3, 7, False), torch.tensor([1.5, -0.5, 3.0], dtype=D)),
        (orthogonal(3, 9, False), torch.full((3,), 0.5, dtype=D)),
    ]
    for R, t in poses:
        cases.append((pts0 @ R.T + t, nrm0 @ R.T, q0 @ R.T + t))
    cases.append((cases[1][0][perm], cases[1][1][perm], cases[1][2]))
    for i, (pts, nrm, q) in enumerate(cases):
        pts, nrm, q = (x.float().to(device) for x in (pts, nrm, q))
        w = torch.ones(pts.shape[0], device=device)
        for autocast in (False, True):
            with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
                ref, fast = (
                    BoundaryForest.build(
                        pts, w, nrm, None, theta=TREE_READ_THETA, build=k
                    )
                    for k in ("reference", "fast")
                )
                lists = [
                    ref.point_pairs(q, walk="reference"),
                    fast.point_pairs(q, walk="fast"),
                ]
            assert not _forest_differences(ref, fast), (i, autocast)
            (qa, sa, na), (qb, sb, nb) = lists
            assert na == nb > 0 and torch.equal(qa, qb) and torch.equal(sa, sb), (
                i,
                autocast,
            )
    p, n, w, _ = cloud(2000, d=3, seed=12)
    p, n, w = (x.float().to(device) for x in (p, n, w))
    ref, fast = (
        BoundaryForest.build(p, w, n, None, theta=TREE_READ_THETA, build=k)
        for k in ("reference", "fast")
    )
    assert not _forest_differences(ref, fast)
    inner = (ref.left >= 0).nonzero().squeeze(-1)[:400]
    q = _tie_queries(ref, inner.cpu(), TREE_READ_THETA).to(device)
    (qa, sa, na), (qb, sb, nb) = (
        ref.point_pairs(q, walk="reference"),
        fast.point_pairs(q, walk="fast"),
    )
    assert na == nb and torch.equal(qa, qb) and torch.equal(sa, sb)


def _step_grads(m, mesh, drv, q, autocast=False):
    """One training step's outputs and every parameter's gradient, in float64."""
    m.zero_grad()
    dev = q.device.type
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=autocast):
        out = outputs(
            m(
                boundary_meshes={"wall": mesh},
                prediction_points=points_mesh(q),
                global_data={"g": drv},
            )
        )
    sum(o.double().square().mean() for o in out.values()).backward()
    grads = {
        n: p.grad.double().cpu() for n, p in m.named_parameters() if p.grad is not None
    }
    return {k: v.detach().double().cpu() for k, v in out.items()}, grads


def test_tree_read_kernel_resolves_to_the_reference_on_cpu(monkeypatch):
    """On CPU ``_tree_read_kernel="auto"`` resolves to the reference read. Forcing
    ``"fused"`` there (the resolver patched) keeps the reference read (the fused
    kernels need CUDA): outputs and every gradient bitwise equal to the reference's."""
    m = _model(3, 0.5).double()
    mesh, drv, q = _case()
    assert (
        m._tree_read_kernel == "auto" and m._resolve_tree_read_kernel(q) == "reference"
    )
    threads = torch.get_num_threads()
    ### Some CPU reductions are not run-to-run bitwise across threads.
    torch.set_num_threads(1)
    try:
        ref = _step_grads(m, mesh, drv, q)
        monkeypatch.setattr(ISLA, "_resolve_tree_read_kernel", lambda self, p: "fused")
        with ran(tree_read.tree_read_walk, tree_read._chunked_basis_from_plan) as r:
            fused = _step_grads(m, mesh, drv, q)
    finally:
        torch.set_num_threads(threads)
    ### The forced name reached the read (else the reference is compared with itself),
    ### and the pairs' basis still came from the plan.
    assert [c["kernel"] for c in r.calls(tree_read.tree_read_walk)] == ["fused"]
    assert r.count(tree_read._chunked_basis_from_plan) > 0
    for a, b in zip(ref, fused):
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16],
    ids=["fp32", "bf16", "fp16"],
)
def test_fused_kernel_read_matches_the_reference_read(dtype):
    """The fused read (kernel term inside the kernels) against the reference read fed
    ``basis @ W.T``, values and every gradient, on a ragged random plan with a point
    without pairs and one source read by half the pairs: in float32, bfloat16 and
    float16 (``_narrow``, as under float16 autocast) each is as close to the float64
    eager read as the reference is (the same rounding points, another summation
    order)."""
    from physicsnemo.experimental.models.isla._fused_read import (
        _narrow,
        fused_kernel_read,
        supported,
    )
    from physicsnemo.experimental.models.isla._segment_softmax import (
        PairPlan,
        _eager_segment_softmax_read,
        segment_softmax_read,
    )

    dev = torch.device("cuda")
    case = _read_case(4)

    def run(kind, device, dt):
        plan = PairPlan.build(
            case.qi.to(device), case.bj.to(device), case.n_q, case.n_b
        )
        b, w, q, k, v = ins = [
            t.detach().clone().to(device, dt).requires_grad_(True) for t in case.leaves
        ]
        lm = case.logm.to(device, torch.float32 if dt != D else D)
        if kind == "fused":
            assert supported(b, q, k, v) and _narrow(b) == (dt != torch.float32)
            out = fused_kernel_read(b, w, lm, q, k, v, plan, 0.5)
        elif kind == "reference":
            out = segment_softmax_read(
                b @ w.t(), lm[plan.bj.long()], q, k, v, plan, 0.5
            )
        else:
            out = _eager_segment_softmax_read(
                b @ w.t(), lm[plan.bj.long()], q, k, v, plan, 0.5
            )
        (out.double() * case.out_grad.to(device)).sum().backward()
        return [out.detach().double().cpu()] + [t.grad.double().cpu() for t in ins]

    exact = run("eager", "cpu", D)
    fused, ref = run("fused", dev, dtype), run("reference", dev, dtype)
    rows = case.counts > 0
    floor = 1e-6 if dtype == torch.float32 else 1e-3
    for name, a, b, e in zip(("out", "basis", "w", "q", "k", "v"), fused, ref, exact):
        if name == "out":
            a, b, e = a[rows], b[rows], e[rows]
        scale = e.abs().max()
        d_fused, d_ref = (a - e).abs().max() / scale, (b - e).abs().max() / scale
        assert d_fused <= 2 * d_ref + floor, name


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16_autocast"])
def test_tree_read_fused_matches_the_reference_on_cuda(autocast):
    """The model on CUDA reads through the fused read by default; one training step's
    outputs and every parameter's gradient are as close to the CPU float64 model's as
    the reference read's are (float32, and bf16 autocast)."""
    m = _model(3, 0.5)
    mesh, drv, q = _sphere_case(n=1500, nq=200)
    exact = _step_grads(copy.deepcopy(m).double(), mesh, drv, q)
    dev = torch.device("cuda")
    m = m.to(dev).float()
    mesh_c = mesh.to(dev).apply(lambda t: t.float() if t.is_floating_point() else t)
    args = (mesh_c, drv.to(dev).float(), q.to(dev).float())
    assert m._resolve_tree_read_kernel(args[2]) == "fused"
    fused = _step_grads(m, *args, autocast=autocast)
    m._tree_read_kernel = "reference"
    ref = _step_grads(m, *args, autocast=autocast)
    floor = 1e-6 if not autocast else 1e-3
    for a, b, e in zip(fused, ref, exact):
        groups = {k.split(".")[0] for k in e}
        for grp in groups:
            keys = [k for k in e if k.split(".")[0] == grp]
            scale = max(float(e[k].abs().max()) for k in keys)
            d_fused = max(float((a[k] - e[k]).abs().max()) for k in keys) / scale
            d_ref = max(float((b[k] - e[k]).abs().max()) for k in keys) / scale
            assert d_fused <= 2 * d_ref + floor, grp
            if a is fused[0] and not autocast:
                assert d_fused < 1e-3, grp  # the outputs, against float64


@pytest.mark.cuda
@requires_cuda_triton
def test_fused_reads_sum_their_basis_gradients_as_autograd_does():
    """Fused reads of one basis that share its sink sum their basis gradients in place
    bitwise as autograd sums them when each read passes its own (bfloat16, the
    running sum rounded after every read), and the step leaves no memory behind."""
    from physicsnemo.experimental.models.isla._fused_read import (
        _plan_source_items,
        basis_sink,
        fused_kernel_read,
    )
    from physicsnemo.experimental.models.isla._segment_softmax import PairPlan

    dev, bf = torch.device("cuda"), torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(5)
    n_q, n_b, h, kc, c, f = 300, 400, 8, 4, 8, 16
    counts = torch.randint(1, 600, (n_q,), generator=g, device=dev)
    qi = torch.repeat_interleave(torch.arange(n_q, device=dev), counts)
    bj = torch.randint(0, n_b, (qi.shape[0],), generator=g, device=dev)
    plan = PairPlan.build(qi, bj, n_q, n_b)
    logm = -8 * torch.rand(n_b, generator=g, device=dev)
    reads = [
        [
            torch.randn(*s, generator=g, device=dev).to(bf)
            for s in ((h, f), (n_q, h, kc), (n_b, h, kc), (n_b, h, c))
        ]
        for _ in range(3)
    ]
    basis0 = torch.randn(qi.shape[0], f, generator=g, device=dev).to(bf)
    _plan_source_items(plan)  # the plan keeps its by-source work items for every block
    base = torch.cuda.memory_allocated()
    grads = []
    for shared in (True, False):
        basis = basis0.clone().requires_grad_(True)
        sink = basis_sink(basis) if shared else None
        out = sum(
            fused_kernel_read(basis, *r[:1], logm, *r[1:], plan, 0.5, sink=sink)
            for r in reads
        )
        out.square().sum().backward()
        grads.append(basis.grad)
        del basis, sink, out
    assert torch.equal(grads[0], grads[1])
    del grads
    assert torch.cuda.memory_allocated() == base


def _int32_plan_past(limit, tail, per=1024, n_b=4096):
    """A plan of int32 indices (so int32 offsets, as ``PairPlan.build`` gives) over
    ``n_b`` sources, ``per`` pairs per point, whose last ``tail`` points' pairs start at
    ``limit``: their indices, the point count and the plan's maker."""
    from physicsnemo.experimental.models.isla._segment_softmax import PairPlan

    n_q = limit // per + tail
    g = torch.Generator(device="cuda").manual_seed(0)
    qi = torch.arange(n_q, device="cuda", dtype=torch.int32).repeat_interleave(per)
    bj = torch.randint(0, n_b, qi.shape, generator=g, device="cuda", dtype=torch.int32)

    def plan(first):
        """The plan of the points from ``first`` on, indexed from zero."""
        p = PairPlan.build(
            qi[first * per :] - first, bj[first * per :], n_q - first, n_b
        )
        assert p.q_offsets.dtype == torch.int32
        return p

    return n_q, g, plan


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.slow
def test_fused_kernel_read_with_int32_offsets_past_2_27_pairs():
    """An int32-offset plan of 2**27 + 65,536 pairs (F = 16): the last 64 points, whose
    pairs' basis offsets p * F pass 2**31, get the outputs, query gradients and pair
    basis gradients of the same points read alone (each point's program reads only its
    own pairs), bitwise. Needs about 21 GiB of device memory: the bfloat16 basis and its
    gradient (4.3 GB each), the backward's (P, 2H) float32 softmax weights and logit
    gradients (8.6 GB) and the plan."""
    from physicsnemo.experimental.models.isla._fused_read import fused_kernel_read

    per, tail, n_b, h, kc, c, f = 1024, 64, 4096, 8, 4, 8, 16
    if torch.cuda.mem_get_info()[0] < 32 * 2**30:
        pytest.skip("needs about 21 GiB of free device memory")
    n_q, g, plan = _int32_plan_past(2**27, tail, per, n_b)
    bf = torch.bfloat16
    basis = torch.randn(n_q * per, f, generator=g, device="cuda", dtype=bf)
    w, q, k, v = (
        torch.randn(*s, generator=g, device="cuda", dtype=bf)
        for s in ((h, f), (n_q, h, kc), (n_b, h, kc), (n_b, h, c))
    )
    logm = -8 * torch.rand(n_b, generator=g, device="cuda")
    seed = torch.randn(tail, h, c, generator=g, device="cuda")

    def run(first):
        b, qq = basis[first * per :].requires_grad_(), q[first:].requires_grad_()
        out = fused_kernel_read(b, w, logm, qq, k, v, plan(first), 0.5)
        (out[-tail:] * seed).sum().backward()
        return [
            t[-tail * n :].detach().clone()
            for t, n in ((out, 1), (qq.grad, 1), (b.grad, per))
        ]

    whole = run(0)
    alone = run(n_q - tail)
    for a, b in zip(whole, alone):
        assert torch.equal(a, b)


@pytest.mark.parametrize("dim", [3, 2])
def test_tree_read_pairs_basis_from_the_plan_is_exact_for_a_batch(
    dim, caplog, monkeypatch
):
    """Two cases in one batch: the basis built from the plan (the fused path's) has the
    reference's plan, values and basis-MLP gradients, bitwise, so each case's point,
    source and pair offsets are right. The fallback to the reference basis (here, on
    CPU) is logged once."""
    monkeypatch.setattr(tree_read, "_BASIS_FALLBACKS_LOGGED", set())
    caplog.set_level(logging.INFO, logger=tree_read.__name__)
    torch.manual_seed(0)
    cases = [cloud(400 + 60 * i, d=dim, seed=3 + i) for i in range(2)]
    n = min(c[0].shape[0] for c in cases)
    pts = torch.stack([c[0][:n] for c in cases])
    nrm = torch.stack([c[1][:n] for c in cases])
    w = torch.stack([c[2][:n] for c in cases])
    q = torch.stack(
        [cloud(90, d=dim, seed=7 + i, scale=(4.0,) * dim)[0] for i in range(2)]
    )
    g = torch.stack([torch.nn.functional.normalize(c[3], dim=-1)[None] for c in cases])
    basis = _KernelBasis(kernel_geo_width(1, 0)).double()
    kw = dict(
        query_r=q,
        query_n=torch.nn.functional.normalize(q, dim=-1),
        query_wall=torch.rand(
            2, 90, dtype=D, generator=torch.Generator().manual_seed(1)
        )
        + 0.1,
        boundary_r=pts,
        boundary_n=nrm,
        boundary_vecs=None,
        log_frac=torch.log(w / w.sum(1, keepdim=True)),
        g_unit=g,
        theta=0.5,
        eps=1e-12,
    )
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        res = []
        for kernel in ("reference", "fused"):
            basis.zero_grad()
            pairs = tree_read_pairs(basis, kernel=kernel, **kw)
            seed = torch.randn(
                pairs.basis.shape, dtype=D, generator=torch.Generator().manual_seed(2)
            )
            (pairs.basis * seed).sum().backward()
            res.append((pairs, [p.grad.clone() for p in basis.parameters()]))
    finally:
        torch.set_num_threads(threads)
    (ref, g_ref), (fus, g_fus) = res
    assert (
        ref.plan.n_q == 180
        and torch.equal(ref.plan.qi, fus.plan.qi)
        and torch.equal(ref.plan.bj, fus.plan.bj)
    )
    assert (
        len({int(x) for x in ref.plan.qi[ref.plan.qi >= 90]}) > 0
    )  # the second case has pairs
    assert torch.equal(ref.basis, fus.basis)
    assert all(torch.equal(a, b) for a, b in zip(g_ref, g_fus))
    tree_read_pairs(basis, kernel="fused", **kw)
    logged = [r.getMessage() for r in caplog.records]
    assert sum("reference kernel basis" in m for m in logged) == 1, logged


def _partial_backward_sequences(read, theta, ws, x, loss_of):
    """theta.grad after each partial-backward sequence, ``read(basis, w)`` the read."""
    out = []
    for seq in ("grad_then_backward", "inputs_w_then_theta"):
        theta.grad = None
        loss = loss_of(read, theta, ws, x)
        if seq == "grad_then_backward":
            torch.autograd.grad(loss, [x], retain_graph=True)
            loss.backward(inputs=[theta])
        else:
            loss.backward(inputs=[ws[0]], retain_graph=True)
            loss.backward(inputs=[theta])
        out.append(theta.grad.clone())
    return out


def test_fused_read_sink_ignores_backward_passes_that_skip_the_basis():
    """A backward pass that runs the reads but not the basis (autograd.grad of another
    input, backward(inputs=[a read weight])) leaves nothing in the sink's sum: the
    next pass's basis gradient equals plain autograd's, with the fused read's sink and
    its backward's choice of target (a stand-in read, CPU)."""
    from physicsnemo.experimental.models.isla._fused_read import basis_sink

    g = torch.Generator().manual_seed(8)
    inv = torch.randn(50, 16, generator=g, dtype=D)
    assert basis_sink(inv) is None  # a basis that needs no gradient has no sink
    theta = torch.randn(16, generator=g, dtype=D).requires_grad_(True)
    ws = [
        torch.randn(8, 16, generator=g, dtype=D).requires_grad_(True) for _ in range(3)
    ]
    x = torch.randn(50, 8, generator=g, dtype=D).requires_grad_(True)
    sinks = []

    def loss_of(read, theta, ws, x):
        basis = inv * theta
        sinks.append(basis_sink(basis) if read is not None else None)
        outs = [
            basis @ w.t()
            if read is None
            else _SinkRead.apply(basis.detach(), *sinks[-1], w)
            for w in ws
        ]
        return (sum(outs) * x).square().sum()

    ref = _partial_backward_sequences(None, theta, ws, x, loss_of)
    fused = _partial_backward_sequences(_SinkRead, theta, ws, x, loss_of)
    for a, b in zip(fused, ref):
        assert torch.allclose(a, b, rtol=1e-12, atol=0)
    assert all(s[1].acc is None for s in sinks[2:])


@pytest.mark.cuda
@requires_cuda_triton
def test_fused_read_partial_backward_matches_the_reference_read():
    """The same partial-backward sequences through the fused reads (float32) against the
    reference read fed ``basis @ W.T``."""
    from physicsnemo.experimental.models.isla._fused_read import (
        basis_sink,
        fused_kernel_read,
    )
    from physicsnemo.experimental.models.isla._segment_softmax import (
        PairPlan,
        segment_softmax_read,
    )

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(9)
    n_q, n_b, h, kc, c, f = 200, 300, 8, 4, 8, 16
    counts = torch.randint(1, 400, (n_q,), generator=g, device=dev)
    qi = torch.repeat_interleave(torch.arange(n_q, device=dev), counts)
    bj = torch.randint(0, n_b, (qi.shape[0],), generator=g, device=dev)
    plan = PairPlan.build(qi, bj, n_q, n_b)
    logm = -8 * torch.rand(n_b, generator=g, device=dev)
    inv = torch.randn(qi.shape[0], f, generator=g, device=dev)
    theta = torch.randn(f, generator=g, device=dev).requires_grad_(True)
    ws = [
        (0.3 * torch.randn(h, f, generator=g, device=dev)).requires_grad_(True)
        for _ in range(3)
    ]
    qkv = [
        torch.randn(*s, generator=g, device=dev)
        for s in ((n_q, h, kc), (n_b, h, kc), (n_b, h, c))
    ]
    x = torch.randn(n_q, h, c, generator=g, device=dev).requires_grad_(True)

    def loss_of(fused, theta, ws, x):
        basis = inv * theta
        sink = basis_sink(basis) if fused else None
        if fused:
            outs = [
                fused_kernel_read(basis, w, logm, *qkv, plan, 0.5, sink=sink)
                for w in ws
            ]
        else:
            outs = [
                segment_softmax_read(basis @ w.t(), logm[plan.bj], *qkv, plan, 0.5)
                for w in ws
            ]
        return (sum(outs) * x).square().sum()

    ref = _partial_backward_sequences(False, theta, ws, x, loss_of)
    fused = _partial_backward_sequences(True, theta, ws, x, loss_of)
    for a, b in zip(fused, ref):
        assert (a - b).abs().max() <= 1e-4 * b.abs().max(), (
            (a - b).abs().max(),
            b.abs().max(),
        )


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_fused_node_means_match_the_reference(dtype):
    """The fused node means against ``node_means`` on a forest of a sphere: the
    backward bitwise (the same float32 additions, one per node), the forward to
    float32 roundoff before the cast (the reference sums each leaf's cells by atomics,
    in no fixed order) and within the 2x-error rule of the float64 means."""
    from physicsnemo.experimental.models.isla._fused_read import fused_node_means

    mesh, _, _ = _sphere_case(n=3000, nq=10)
    dev = torch.device("cuda")
    pts = mesh.cell_centroids.float().to(dev)
    nrm = mesh.cell_normals.float().to(dev)
    w = mesh.cell_data[EFFECTIVE_MEASURE_KEY].float().to(dev)
    omega = w / w.sum()
    forest = BoundaryForest.build(pts, omega, nrm, None, theta=1.0)
    g = torch.Generator(device=dev).manual_seed(6)
    x0 = torch.randn(pts.shape[0], 96, generator=g, device=dev).to(dtype)
    gout = torch.randn(forest.n_nodes, 96, generator=g, device=dev).to(dtype)
    res = []
    for fn in (tree_read.node_means, fused_node_means):
        x = x0.clone().requires_grad_(True)
        y = fn(x, forest, omega)
        y.backward(gout)
        res.append((y.detach().float(), x.grad))
    tol = 1e-6 if dtype == torch.float32 else 8e-3
    assert torch.allclose(
        res[1][0], res[0][0], rtol=tol, atol=tol * res[0][0].abs().max()
    )
    assert torch.equal(res[1][1], res[0][1])
    exact = tree_read.node_means(x0.double(), forest, omega.double())
    _two_x(
        "means", res[1][0], res[0][0], exact, 1e-6 if dtype == torch.float32 else 1e-3
    )


@pytest.mark.cuda
@requires_cuda_triton
def test_fused_node_means_refuse_a_second_derivative():
    """The fused node means' backward is one Triton pass, not differentiable: a
    second backward pass through it raises instead of leaving out its term."""
    from physicsnemo.experimental.models.isla._fused_read import fused_node_means

    mesh, _, _ = _sphere_case(n=300, nq=10)
    dev = torch.device("cuda")
    pts = mesh.cell_centroids.float().to(dev)
    w = mesh.cell_data[EFFECTIVE_MEASURE_KEY].float().to(dev)
    omega = w / w.sum()
    forest = BoundaryForest.build(
        pts, omega, mesh.cell_normals.float().to(dev), None, theta=1.0
    )
    x = torch.randn(pts.shape[0], 8, device=dev, requires_grad=True)
    (gx,) = torch.autograd.grad(
        fused_node_means(x, forest, omega).square().sum(), x, create_graph=True
    )
    with pytest.raises(RuntimeError, match="once_differentiable"):
        (gx.square().sum() + x.sum()).backward()


def _chunked_step(m, kw, loss_of, size):
    """One step through :meth:`ISLA.query_chunks` in chunks of ``size`` points: the
    chunks' outputs, concatenated."""
    chunks = m.query_chunks(**kw)
    parts = []
    for a in range(0, chunks.n_points, size):
        rows = slice(a, min(a + size, chunks.n_points))
        out = outputs(chunks(rows.start, rows.stop))
        loss_of(out, rows).backward()
        parts.append({k: v.detach() for k, v in out.items()})
    chunks.backward()
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def _far_sphere_case():
    """A closed icosphere of radius 1 about 1e3 of its size from the origin, with
    prediction points at its centre (a tie of several faces), on 12 of its cells'
    centroids (on the wall) and near its cells, so that the exact geometry's tie and
    on-wall bands and its centred frame all decide something."""
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    t = torch.tensor([1000.0, -700.0, 400.0], dtype=D)
    mesh = sphere_icosahedral.load(radius=1.0, subdivisions=1)
    mesh = Mesh(points=mesh.points.double() + t, cells=mesh.cells)
    cent = mesh.points[mesh.cells].mean(1)
    g = torch.Generator().manual_seed(0)
    near = cent[:30] + 0.2 * torch.randn(30, 3, generator=g, dtype=D)
    return (
        mesh,
        torch.tensor([0.3, 1.0, 0.2], dtype=D),
        torch.cat([t[None], cent[:12], near]),
    )


@pytest.mark.parametrize("case", ["cloud", "far_sphere"])
@pytest.mark.parametrize("vectors", [False, True], ids=["plain", "boundary_vectors"])
@pytest.mark.parametrize("theta", [0.0, TREE_READ_THETA])
def test_query_chunks_equal_the_forward_pass_and_its_gradients(theta, vectors, case):
    """ISLA.query_chunks (the boundary side once, the points chunk by chunk, then the
    boundary side's backward with the leaves' gradients): in training mode, from the
    same token draw, the outputs of forward and every parameter's gradient of a loss
    summed over the points, to float64 roundoff. The forward pads its token axis, the
    chunks' boundary side pads its own (both unaligned here) and the points run
    unpadded. ``far_sphere`` has points whose geometry is a tie or on the wall."""
    import physicsnemo.experimental.models.isla.model as isla_model
    from physicsnemo.experimental.models.isla._geometry import _closest_features

    extra = {"boundary_schemas": {"wall": {"tr": {"rank": 1}}}} if vectors else {}
    m = _model(3, theta, **extra).double()
    with torch.no_grad():
        for p in m.blocks.parameters():
            p.add_(0.05 * torch.randn_like(p))
    mesh, drv, q = _case() if case == "cloud" else _far_sphere_case()
    if case == "far_sphere":
        _, _, _, tied, _, _, _, on_wall, _, _ = _closest_features(
            q, mesh.points, mesh.cells
        )
        assert bool(tied[0]) and bool(on_wall[1:13].all())
    assert (mesh.n_cells + q.shape[0]) % isla_model._TOKEN_ALIGNMENT
    assert mesh.n_cells % isla_model._TOKEN_ALIGNMENT
    if vectors:
        g = torch.Generator().manual_seed(4)
        data = dict(mesh.cell_data.items())
        data["tr"] = torch.randn(mesh.n_cells, 3, generator=g, dtype=D)
        mesh = mesh.with_data(cell_data=data)
    kw = {
        "boundary_meshes": {"wall": mesh},
        "prediction_points": points_mesh(q),
        "global_data": {"g": drv},
    }
    target = torch.randn(
        q.shape[0], 4, generator=torch.Generator().manual_seed(9), dtype=D
    )

    def loss_of(out, rows):
        return (out["p"] - target[rows, 0]).square().sum() + (
            out["v"] - target[rows, 1:]
        ).square().sum()

    torch.manual_seed(3)
    full = outputs(m(**kw))
    loss_of(full, slice(None)).backward()
    ref = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    chunked = _chunked_step(m, kw, loss_of, 37)
    ### absolute (rtol 0): the chunks sum in another order, and far_sphere's tie and
    ### on-wall points amplify that roundoff (measured 2e-12 to 1.03e-11 at 1-4 threads
    ### on x86, with the BLAS reduction order; 5e-14 on Arm; cloud 1.6e-15)
    assert same(chunked, {k: v.detach() for k, v in full.items()}, atol=1e-10)
    ### gradients that vanish analytically (a bias every slice logit shares) are roundoff on both sides
    scale = max(float(g.abs().max()) for g in ref.values())
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.allclose(p.grad, ref[name], rtol=1e-9, atol=1e-12 * scale), name


def _recorded_geometry(monkeypatch, peaks=False):
    """Record every call of the model's :func:`exact_boundary_geometry`: its points and
    outputs, or with ``peaks`` its point count and the CUDA memory it allocated above
    what was allocated before it."""
    calls = []
    inner = model_mod.exact_boundary_geometry

    def recording(q, *args, **kwargs):
        if peaks:
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        d, u = inner(q, *args, **kwargs)
        if peaks:
            calls.append((q.shape[0], torch.cuda.max_memory_allocated() - base))
        else:
            calls.append((q, d, u))
        return d, u

    monkeypatch.setattr(model_mod, "exact_boundary_geometry", recording)
    return calls


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_cuda])]
)
def test_query_chunks_measure_each_chunk_s_geometry_when_it_runs(device, monkeypatch):
    """ISLA.query_chunks measures the prediction points' geometry chunk by chunk, as
    each chunk runs: each call of exact_boundary_geometry takes one chunk's points
    alone, their distances and directions are bitwise those the forward pass measures
    for every point at once (in one chunk here), and the outputs are the forward
    pass's to the roundoff of test_query_chunks_equal_the_forward_pass_and_its_gradients
    (the chunks' products sum in another order). ``far_sphere`` has points whose
    geometry is a tie or on the wall."""
    calls = _recorded_geometry(monkeypatch)
    m = _tree(3, TREE_READ_THETA).to(device)
    mesh, drv, q = _far_sphere_case()
    mesh, drv, q = mesh.to(device), drv.to(device), q.to(device)
    kw = {
        "boundary_meshes": {"wall": mesh},
        "prediction_points": points_mesh(q),
        "global_data": {"g": drv},
    }
    size = 7
    with torch.no_grad():
        full = outputs(m(**kw))
        chunks = m.query_chunks(**kw)
        assert len(calls) == 1
        parts = [
            outputs(chunks(a, min(a + size, chunks.n_points)))
            for a in range(0, chunks.n_points, size)
        ]
    (q_f, d_f, u_f), *per_chunk = calls
    assert torch.equal(q_f, q)
    assert [c[0].shape[0] for c in per_chunk] == [
        min(size, q.shape[0] - a) for a in range(0, q.shape[0], size)
    ]
    assert torch.equal(torch.cat([c[0] for c in per_chunk]), q)
    assert torch.equal(torch.cat([c[1] for c in per_chunk]), d_f)
    assert torch.equal(torch.cat([c[2] for c in per_chunk]), u_f)
    chunked = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    assert same(chunked, full, atol=1e-10)


@pytest.mark.cuda
@requires_cuda
def test_query_chunks_bound_the_geometry_s_memory_by_the_chunk(monkeypatch):
    """The geometry stage's peak memory follows the chunk, not the point count: one
    million prediction points against a 5,120-face sphere in sixteen chunks, through
    ISLA.query_chunks and through the forward pass in chunks of walk pairs
    (``query_chunk_pairs``), each allocate under a quarter of what
    exact_boundary_geometry allocates for all of them at once."""
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    calls = _recorded_geometry(monkeypatch, peaks=True)
    sphere = sphere_icosahedral.load(radius=1.0, subdivisions=4)
    mesh = Mesh(points=sphere.points.float().cuda(), cells=sphere.cells.cuda())
    g = torch.Generator(device="cuda").manual_seed(0)
    n, n_chunks = 1_000_000, 16
    q = 6 * torch.rand(n, 3, generator=g, device="cuda") - 3
    kw = {
        "boundary_meshes": {"wall": mesh},
        "prediction_points": points_mesh(q),
        "global_data": {"g": torch.tensor([0.3, 1.0, 0.2], device="cuda")},
    }
    m = _model(3, TREE_READ_THETA).cuda().eval()
    with torch.no_grad():
        model_mod.exact_boundary_geometry(q, mesh.points, mesh.cells)
        ((_, whole),) = calls
        chunks = m.query_chunks(**kw)
        size = -(-n // n_chunks)
        for a in range(0, n, size):
            chunks._fields(a, min(a + size, n))
        del chunks
        m.query_chunk_pairs = int(m.query_chunks(**kw).pair_counts.sum()) // n_chunks
        m(**kw)
    by_chunk, by_pairs = calls[1 : n_chunks + 1], calls[n_chunks + 1 :]
    assert [c[0] for c in by_chunk] == [min(size, n - a) for a in range(0, n, size)]
    assert len(by_pairs) >= n_chunks and sum(c[0] for c in by_pairs) == n
    for _, peak in by_chunk + by_pairs:
        assert peak < whole / 4, (peak, whole)


def test_query_chunks_pad_the_boundary_side_to_the_token_alignment():
    """The boundary side of :meth:`ISLA.query_chunks` runs the slice blocks on its
    tokens completed to a multiple of _TOKEN_ALIGNMENT, as forward runs its own."""
    import physicsnemo.experimental.models.isla.model as isla_model

    m = _model(3, TREE_READ_THETA).double()
    mesh, drv, q = _case(n=300, nq=40)
    seen = []
    hook = m.blocks[0].register_forward_pre_hook(
        lambda mod, args, kwargs: seen.append(args[0].shape[1]), with_kwargs=True
    )
    try:
        m.query_chunks(
            boundary_meshes={"wall": mesh},
            prediction_points=points_mesh(q),
            global_data={"g": drv},
        )
    finally:
        hook.remove()
    align = isla_model._TOKEN_ALIGNMENT
    assert mesh.n_cells % align
    assert seen == [mesh.n_cells + (-mesh.n_cells) % align]


def test_query_chunks_read_the_forest_with_the_normals_error_bounds(monkeypatch):
    """The forward and :meth:`ISLA.query_chunks` build the forest from the same drawn
    tokens with the same bounds on their normals (``normal_tol``, from the meshes'
    vertices), so the chunked step walks the forward's forest."""
    import physicsnemo.experimental.models.isla.model as isla_model

    seen = []
    inner = isla_model.tree_read_walk

    def recording(**kw):
        out = inner(**kw)
        seen.append((kw["normal_tol"], out))
        return out

    monkeypatch.setattr(isla_model, "tree_read_walk", recording)
    m = _tree(3, TREE_READ_THETA)
    mesh, drv, q = _case()
    kw = {
        "boundary_meshes": {"wall": mesh},
        "prediction_points": points_mesh(q),
        "global_data": {"g": drv},
    }
    with torch.no_grad():
        m(**kw)
        m.query_chunks(**kw)
    (tol_f, walk_f), (tol_c, walk_c) = seen
    assert tol_f is not None and tol_f.dtype == D and bool((tol_f > 0).all())
    assert torch.equal(tol_c, tol_f)
    assert not _forest_differences(walk_f.forests[0], walk_c.forests[0])
    lists_f, lists_c = (
        w.forests[0].frame_pairs(w.points_f[0], walk=w.walk) for w in (walk_f, walk_c)
    )
    assert all(torch.equal(a, b) for a, b in zip(lists_f[:2], lists_c[:2]))


def test_query_chunks_of_an_empty_range_are_empty_fields():
    """An empty range of prediction points (a caller's own chunking can make one) has
    empty output fields of every rank, as a scalar output does."""
    m = _tree(3, TREE_READ_THETA)
    mesh, drv, q = _case(n=50, nq=20)
    with torch.no_grad():
        chunks = m.query_chunks(
            boundary_meshes={"wall": mesh},
            prediction_points=points_mesh(q),
            global_data={"g": drv},
        )
        empty = chunks(5, 5)
    assert empty.n_points == 0
    assert empty.point_data["p"].shape == (0,)
    assert empty.point_data["v"].shape == (0, 3)


def test_query_chunks_need_an_interior_model():
    """A surface model has no prediction points to chunk."""
    m = ISLA(
        output_schema=PV,
        global_schema=G,
        boundary_schemas={"wall": {}},
        hidden=16,
        n_layers=1,
    )
    mesh, drv, q = _case(n=50, nq=20)
    with pytest.raises(ValueError, match="interior=True"):
        m.query_chunks(
            boundary_meshes={"wall": mesh}, prediction_points=q, global_data={"g": drv}
        )


def test_compile_blocks_compiles_the_reads_projections_as_eager_computes_them():
    """Under ``compile_blocks`` the boundary read's LayerNorms and projections run
    compiled, in the forward and in :meth:`ISLA.query_chunks`: outputs and every
    parameter's gradient equal the eager model's to float64 roundoff."""
    m_e = _model(3, TREE_READ_THETA).double()
    m_c = _model(3, TREE_READ_THETA, compile_blocks=True).double()
    m_c.load_state_dict(m_e.state_dict())
    mesh, drv, q = _case()
    kw = {
        "boundary_meshes": {"wall": mesh},
        "prediction_points": points_mesh(q),
        "global_data": {"g": drv},
    }
    w = torch.linspace(-1, 1, q.shape[0], dtype=D)

    def loss_of(out, rows):
        return (out["p"] * w[rows]).sum() + out["v"].square().sum()

    def grads(m):
        g = {n: p.grad.clone() for n, p in m.named_parameters()}
        m.zero_grad(set_to_none=True)
        return g

    tree_read._COMPILED.clear()
    runs = []
    for m in (m_e, m_c):
        torch.manual_seed(3)
        out = outputs(m(**kw))
        loss_of(out, slice(None)).backward()
        runs.append(({k: v.detach() for k, v in out.items()}, grads(m)))
    torch.manual_seed(3)
    runs.append((_chunked_step(m_c, kw, loss_of, 37), grads(m_c)))
    assert set(tree_read._COMPILED) == {
        tree_read._source_states,
        tree_read._query_states,
    }
    (out_e, grad_e), *compiled = runs
    scale = max(float(g.abs().max()) for g in grad_e.values())
    for out, grad in compiled:
        assert same(out, out_e, atol=1e-12)
        for name, g in grad_e.items():
            assert torch.allclose(grad[name], g, rtol=1e-9, atol=1e-12 * scale), name


class _Walked(Exception):
    """Raised by a spy on the walk to stop a forward pass once the walk is recorded."""


def test_tree_read_theta_and_leaf_size_reach_every_path(monkeypatch):
    """The constructor's ``tree_read_theta`` and ``tree_read_leaf_size`` build the
    forest of every path that reads the boundary: the forward pass in evaluation and in
    training, the forward pass in many chunks of points (``query_chunk_pairs``),
    ``query_chunks``, and a model with ``compile_blocks`` (stopped at the walk, before
    anything compiles: the walk runs eager ahead of the blocks)."""
    seen = []
    inner = model_mod.tree_read_walk

    def recording(**kw):
        out = inner(**kw)
        seen.append((kw["theta"], kw["leaf_size"], out.forests[0]))
        return out

    def stop(**kw):
        recording(**kw)
        raise _Walked

    knobs = dict(tree_read_leaf_size=8, query_chunk_pairs=1)
    mesh, drv, q = _case(nq=12)
    kw = {
        "boundary_meshes": {"wall": mesh},
        "prediction_points": points_mesh(q),
        "global_data": {"g": drv},
    }
    m = _tree(3, 0.5, **knobs)
    monkeypatch.setattr(model_mod, "tree_read_walk", recording)
    with torch.no_grad():
        m(**kw)
        m.query_chunks(**kw)(0, q.shape[0])
        m.train()(**kw)
    monkeypatch.setattr(model_mod, "tree_read_walk", stop)
    compiled = _tree(3, 0.5, compile_blocks=True, **knobs)
    with torch.no_grad(), pytest.raises(_Walked):
        compiled(**kw)
    assert len(seen) == 4
    for theta, leaf_size, forest in seen:
        assert theta == forest.theta == 0.5 and leaf_size == 8
        leaves = forest.range_count[forest.leaf_ids]
        assert int(leaves.max()) <= 8 and int(leaves.max()) > TREE_READ_LEAF_SIZE


def test_tree_read_knobs_change_the_outputs_only_where_they_should():
    """With the same parameters, an opening angle other than the default changes the
    predictions, and the leaf size changes them at a finite angle only: at theta 0
    (the exact read, every cell pair) it changes nothing."""
    mesh, drv, q = _case()
    run = lambda theta, **kw: outputs(_run(_tree(3, theta, **kw), mesh, drv, q))  # noqa: E731
    default = run(TREE_READ_THETA)
    assert _rel(run(0.5), default) > 1e-6
    assert _rel(run(TREE_READ_THETA, tree_read_leaf_size=8), default) > 1e-6
    exact = run(0.0)
    assert _rel(exact, default) > 1e-6
    assert _rel(run(0.0, tree_read_leaf_size=8), exact) < 1e-12


@pytest.mark.parametrize(
    "device,walk,autocast",
    [
        ("cpu", "reference", False),
        ("cpu", "fast", False),
        pytest.param("cuda", "auto", False, marks=[pytest.mark.cuda, requires_cuda]),
        pytest.param("cuda", "auto", True, marks=[pytest.mark.cuda, requires_cuda]),
    ],
    ids=["cpu-reference", "cpu-fast", "cuda", "cuda-bf16_autocast"],
)
def test_walk_of_a_range_of_points_is_those_rows_of_the_walk_of_every_point(
    device, walk, autocast
):
    """A walk made without its lists (``walk_points=False``, as ISLA.query_chunks makes
    it) walks a range of points on demand (``walk_points_range``, here outside the
    autocast it was made in): its lists are those points' rows of the walk of every
    point, raw and in order (numbered from the range's start), with their node counts
    and geometry rows; ``walk_pair_counts`` counts each point's pairs without the
    lists."""
    dt = torch.float32 if device == "cuda" else D
    pts, nrm, w, g = (t.to(device, dt) for t in cloud(500, seed=3))
    q = cloud(150, seed=7, scale=(4.0,) * 3)[0].to(device, dt)
    kw = dict(
        query_r=q[None],
        query_n=torch.nn.functional.normalize(q, dim=-1)[None],
        query_wall=0.1
        + torch.rand(1, 150, generator=torch.Generator().manual_seed(1)).to(device, dt),
        boundary_r=pts[None],
        boundary_n=nrm[None],
        boundary_vecs=None,
        log_frac=torch.log(w / w.sum())[None],
        g_unit=torch.nn.functional.normalize(g, dim=-1)[None, None],
        theta=0.5,
        eps=1e-12,
        walk=walk,
    )
    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        full = tree_read.tree_read_walk(**kw)
        lazy = tree_read.tree_read_walk(walk_points=False, **kw)
    assert lazy.lists == [None]
    qi, si, n_node = full.lists[0]
    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        ref = full.forests[0].point_pairs(q, walk=walk)  # the forest's own walk
    assert torch.equal(ref[0], qi) and torch.equal(ref[1], si) and ref[2] == n_node
    assert 0 < n_node < qi.numel()
    counts = tree_read.walk_pair_counts(lazy)
    assert torch.equal(counts, tree_read.walk_pair_counts(full))
    assert torch.equal(counts, torch.bincount(qi, minlength=150))
    for start, stop in ((0, 150), (40, 110), (149, 150)):
        part = tree_read.walk_points_range(lazy, start, stop)
        a, b = torch.searchsorted(qi, qi.new_tensor([start, stop])).tolist()
        pq, ps, pn = part.lists[0]
        assert torch.equal(pq, qi[a:b] - start) and torch.equal(ps, si[a:b])
        assert pn == int((si[a:b] >= full.n_cells).sum())
        assert part.n_queries == stop - start
        for k in ("query_r", "query_n", "query_wall"):
            assert torch.equal(part.geometry[0][k], full.geometry[0][k][start:stop])


class _MlpProbe(torch.autograd.Function):
    """The basis MLP of one chunk's invariants with given weights, its backward the
    reference's ops (a CPU stand-in for the fused basis kernels)."""

    @staticmethod
    def mlp(inv, w1, b1, w2, b2, w3, b3):
        lin, gelu = torch.nn.functional.linear, torch.nn.functional.gelu
        h = gelu(lin(gelu(lin(inv.to(w1.dtype), w1, b1)), w2, b2))
        return lin(h, w3, b3)

    @staticmethod
    def forward(ctx, inv, *ws):
        ctx.save_for_backward(inv, *ws)
        return _MlpProbe.mlp(inv, *ws)

    @staticmethod
    def backward(ctx, dout):
        inv, *ws = ctx.saved_tensors
        with torch.enable_grad():
            ws = [w.detach().requires_grad_(True) for w in ws]
            grads = torch.autograd.grad(_MlpProbe.mlp(inv, *ws), ws, dout)
        return (None, *grads)


def test_fused_basis_chunks_sum_their_weight_gradients_as_the_reference(monkeypatch):
    """Under bf16 autocast the reference rounds each chunk's weight and bias gradients
    to bfloat16 and sums the chunks' in bfloat16, through autocast's one cached cast
    of each weight. One autograd node per chunk fed one cast per forward pass
    (``mlp_weights``), as the fused basis is, gives the reference's basis and
    gradients bitwise (a stand-in node, CPU)."""

    calls = []

    def stand_in(weights, qi, si, q_off, s_off, geometry):
        calls.append(1)
        with torch.no_grad():
            inv = tree_pair_invariants(qi.long() - q_off, si.long() - s_off, **geometry)
        return _MlpProbe.apply(inv, *weights)

    monkeypatch.setattr(tree_read, "TREE_READ_PAIR_CHUNK", 3000)
    monkeypatch.setattr(tree_read.fused_basis, "fused_kernel_basis", stand_in)
    geometry, plan, basis = _basis_case(3, 1, n_pairs=10000)
    geometry, basis = _to(geometry, "cpu", torch.float32), basis.float()
    seed = 1e-3 * torch.randn(10000, 16, generator=torch.Generator().manual_seed(1))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        ref = _basis_step(basis, plan, geometry, seed, autocast=True)
        fused = _basis_step(basis, plan, geometry, seed, autocast=True, weights=True)
    finally:
        torch.set_num_threads(threads)
    ### 10000 pairs in chunks of 3000 (else both steps run the reference basis).
    assert len(calls) == math.ceil(10000 / tree_read.TREE_READ_PAIR_CHUNK)
    assert ref[0].dtype == torch.bfloat16
    assert all(torch.equal(a, b) for a, b in zip(fused, ref))


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize(
    "gelu_tables", [False, True], ids=["gelu_computed", "gelu_tables"]
)
@pytest.mark.parametrize("dim,n_vecs", [(3, 1), (2, 0)])
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16_autocast"])
def test_fused_kernel_basis_matches_the_reference_basis(
    autocast, dim, n_vecs, gelu_tables, monkeypatch
):
    """The fused basis (invariants and MLP in one kernel per chunk) against the
    reference's chunked basis on CUDA, the basis and every MLP gradient for a given
    basis gradient: each as close to the CPU float64 reference as the reference is
    (float32, and bf16 autocast, its backward's GELU computed or from the tables);
    the fused invariants equal ``tree_pair_invariants``'s to float32 roundoff."""
    from physicsnemo.experimental.models.isla import _fused_basis as fused_basis

    if gelu_tables:
        monkeypatch.setattr(
            fused_basis, "_computed_gelu_is_pytorchs", lambda device, dtype: False
        )
    monkeypatch.setattr(tree_read, "TREE_READ_PAIR_CHUNK", 4096)
    geometry, plan, basis = _basis_case(dim, n_vecs)
    seed = 1e-3 * torch.randn(20000, 16, generator=torch.Generator().manual_seed(1))
    seed = seed.bfloat16().double()
    exact = _basis_step(basis, plan, geometry, seed)
    dev = torch.device("cuda")
    geo_c, plan_c = _to(geometry, dev, torch.float32), _to(plan, dev, torch.float32)
    basis_c = copy.deepcopy(basis).float().to(dev)
    assert fused_basis.unsupported_reason(basis_c, geo_c) is None
    inv = fused_basis.pair_invariants(plan_c.qi, plan_c.bj, 0, 0, geo_c)
    inv_ref = tree_pair_invariants(plan_c.qi.long(), plan_c.bj.long(), **geo_c)
    assert torch.allclose(inv, inv_ref, rtol=1e-6, atol=1e-6)
    seed_c = seed.float().to(dev)
    ref = _basis_step(basis_c, plan_c, geo_c, seed_c, autocast=autocast)
    fused = _basis_step(basis_c, plan_c, geo_c, seed_c, autocast=autocast, weights=True)
    floor = 1e-6 if not autocast else 1e-3
    names = ["basis"] + [n for n, _ in basis.named_parameters()]
    for name, a, b, e in zip(names, fused, ref, exact):
        assert a.dtype == b.dtype, name
        scale = e.abs().max()
        d_fused = (a.double().cpu() - e).abs().max() / scale
        d_ref = (b.double().cpu() - e).abs().max() / scale
        assert d_fused <= 2 * d_ref + floor, name
    if autocast:
        ### The reference's rounding points, which a lost rounding would pass above: the
        ### basis bitwise but where summation order flips a last bit, and the gradients
        ### within a few bf16 ulps (at each tensor's largest value) of the reference's.
        same = (fused[0] == ref[0]).double().mean()
        ulps = [
            float((a - b).abs().max() / 2.0 ** (b.abs().max().log2().floor() - 7))
            for a, b in zip(fused[1:], ref[1:])
        ]
        assert same >= 0.99
        assert max(ulps) <= 4, dict(zip(names[1:], ulps))


@pytest.mark.cuda
@requires_cuda_triton
def test_fused_basis_leaves_the_geometrys_graph_alone():
    """The fused basis gives its geometry no gradient and keeps no edge into the
    geometry's graph: rank-1 inputs from a learned map (as ISLA's are) survive one
    backward per range of points, as :meth:`ISLA.query_chunks` takes, and get no
    gradient."""
    from physicsnemo.experimental.models.isla import _fused_basis as fused_basis

    dev = torch.device("cuda")
    geometry, plan, basis = _basis_case(3, 1)
    geo, plan = _to(geometry, dev, torch.float32), _to(plan, dev, torch.float32)
    basis = basis.float().to(dev)
    raw = geo["src_vecs"].clone().requires_grad_(True)
    geo["src_vecs"] = raw.exp()
    for a, b in ((0, 7000), (7000, 20000)):
        weights = fused_basis.mlp_weights(basis)
        fused_basis.fused_kernel_basis(
            weights, plan.qi[a:b], plan.bj[a:b], 0, 0, geo
        ).sum().backward()
    assert raw.grad is None
    assert all(p.grad is not None for p in basis.parameters())


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_fused_basis_gelu_tables_are_pytorchs_gelu_on_every_input(dtype):
    """The fused basis's GELU tables give PyTorch's 16-bit GELU and its backward
    bitwise at every input value (the backward for random gradients)."""
    from physicsnemo.experimental.models.isla import _fused_basis as fused_basis

    dev = torch.device("cuda")
    f, fg = fused_basis._gelu_tables(dev, dtype)
    f_bwd, g = fg.view(torch.float32).view(-1, 2).unbind(1)
    bits = torch.arange(1 << 16, dtype=torch.int32, device=dev)
    x = bits.to(torch.int16).view(dtype)
    ok = torch.isfinite(x)
    gen = torch.Generator(device=dev).manual_seed(0)
    assert torch.equal(f[ok], torch.nn.functional.gelu(x)[ok])
    assert torch.equal(f_bwd[ok], f.float()[ok])
    for _ in range(8):
        dy = torch.randn(x.shape, generator=gen, device=dev).mul(4).to(dtype)
        ref = torch.ops.aten.gelu_backward(dy, x)
        assert torch.equal((dy.float() * g).to(dtype)[ok], ref[ok])


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_fused_basis_backward_computes_pytorchs_gelu_on_every_input(dtype):
    """The fused basis backward's computed GELU (CUDA 13's erff, ported) is PyTorch's
    16-bit GELU, and its factor PyTorch's float32 ``gelu_backward(1, x)``, bitwise at
    every finite input on PyTorch's CUDA 13 builds, where the check that chooses it
    over the tables passes; it passes only where they hold."""
    from physicsnemo.experimental.models.isla import _fused_basis as fused_basis

    dev = torch.device("cuda")
    x = torch.arange(1 << 16, dtype=torch.int32, device=dev).to(torch.int16).view(dtype)
    out = torch.empty(1 << 16, 2, device=dev)
    fused_basis._gelu_check_kernel[(64,)](
        x, out, 1 << 16, BLOCK=1024, enable_fp_fusion=False
    )
    ok, xf = torch.isfinite(x), x.float()
    gelu = torch.nn.functional.gelu(xf).to(dtype).float()
    factor = torch.ops.aten.gelu_backward(torch.ones_like(xf), xf)
    same = torch.equal(out[ok, 0], gelu[ok]) and torch.equal(out[ok, 1], factor[ok])
    fused_basis._GELU_COMPUTED.clear()
    computed = fused_basis._computed_gelu_is_pytorchs(dev, dtype)
    assert same or not computed
    if int(torch.version.cuda.split(".")[0]) >= 13:
        assert same and computed


@pytest.mark.cuda
@requires_cuda_triton
def test_tree_read_pairs_fused_takes_the_reference_basis_when_the_geometry_requires_grad(
    monkeypatch,
):
    """On CUDA ``tree_read_pairs(kernel="fused")`` with prediction points that require
    a gradient builds the basis on the reference path, whose pair invariants carry
    it: the reference's basis and its gradient in the points, bitwise and nonzero,
    and the fused basis kernel is not called."""
    from physicsnemo.experimental.models.isla import _fused_basis as fused_basis

    calls = []
    real = fused_basis.fused_kernel_basis

    def spy(*args):
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(fused_basis, "fused_kernel_basis", spy)
    torch.manual_seed(0)
    dev = torch.device("cuda")
    pts, nrm, w, g = (t.float().to(dev) for t in cloud(500, seed=3))
    q = cloud(150, seed=7, scale=(4.0,) * 3)[0].float().to(dev)
    kw = dict(
        query_n=torch.nn.functional.normalize(q, dim=-1)[None],
        query_wall=0.1 + torch.rand(1, 150, device=dev),
        boundary_r=pts[None],
        boundary_n=nrm[None],
        boundary_vecs=None,
        log_frac=torch.log(w / w.sum())[None],
        g_unit=torch.nn.functional.normalize(g, dim=-1)[None, None],
        theta=0.5,
        eps=1e-12,
    )
    basis = _KernelBasis(kernel_geo_width(1, 0)).to(dev)
    out = []
    for kernel in ("reference", "fused"):
        q_r = q[None].clone().requires_grad_(True)
        pairs = tree_read_pairs(basis, kernel=kernel, query_r=q_r, **kw)
        seed = torch.randn(
            pairs.basis.shape, device=dev, generator=torch.Generator(dev).manual_seed(2)
        )
        out.append(
            (pairs.basis, torch.autograd.grad((pairs.basis * seed).sum(), q_r)[0])
        )
    assert not calls
    (ref, g_ref), (fus, g_fus) = out
    assert (
        torch.equal(ref, fus)
        and torch.equal(g_ref, g_fus)
        and bool(g_ref.abs().max() > 0)
    )


@pytest.mark.cuda
@requires_cuda_triton
def test_tree_read_pairs_fused_on_cuda_builds_the_basis_in_the_fused_kernel(
    monkeypatch,
):
    """On CUDA ``tree_read_pairs(kernel="fused")`` builds the basis with the fused
    basis kernel: the reference's plan, and its basis to float32 roundoff; so does
    one range of points (``walk_range_pairs``, as :meth:`ISLA.query_chunks` reads)."""
    from physicsnemo.experimental.models.isla import _fused_basis as fused_basis

    calls = []
    real = fused_basis.fused_kernel_basis

    def spy(*args):
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(fused_basis, "fused_kernel_basis", spy)
    torch.manual_seed(0)
    dev = torch.device("cuda")
    pts, nrm, w, g = (t.float().to(dev) for t in cloud(500, seed=3))
    q = cloud(150, seed=7, scale=(4.0,) * 3)[0].float().to(dev)
    kw = dict(
        query_r=q[None],
        query_n=torch.nn.functional.normalize(q, dim=-1)[None],
        query_wall=0.1 + torch.rand(1, 150, device=dev),
        boundary_r=pts[None],
        boundary_n=nrm[None],
        boundary_vecs=None,
        log_frac=torch.log(w / w.sum())[None],
        g_unit=torch.nn.functional.normalize(g, dim=-1)[None, None],
        theta=0.5,
        eps=1e-12,
    )
    basis = _KernelBasis(kernel_geo_width(1, 0)).to(dev)
    ref = tree_read_pairs(basis, kernel="reference", **kw)
    assert not calls
    fus = tree_read_pairs(basis, kernel="fused", **kw)
    assert len(calls) == 1
    assert torch.equal(ref.plan.qi.long(), fus.plan.qi.long())
    assert torch.equal(ref.plan.bj.long(), fus.plan.bj.long())
    scale = float(ref.basis.abs().max())
    assert torch.allclose(fus.basis, ref.basis, rtol=1e-5, atol=1e-5 * scale)
    ref, fus = (
        tree_read.walk_range_pairs(
            basis,
            tree_read.walk_points_range(
                tree_read.tree_read_walk(kernel=k, walk_points=False, **kw), 40, 110
            ),
        )
        for k in ("reference", "fused")
    )
    assert len(calls) == 2
    assert torch.equal(ref.plan.qi.long(), fus.plan.qi.long())
    assert torch.equal(ref.plan.bj.long(), fus.plan.bj.long())
    assert torch.allclose(fus.basis, ref.basis, rtol=1e-5, atol=1e-5 * scale)


@pytest.mark.parametrize("device", DEVICES)
def test_a_forced_reference_runs_the_cluster_trees_torch_code(device, monkeypatch):
    """With ``build="reference"`` and ``walk="reference"`` (as ``reference_kernels``
    sets them) the forest's cluster trees are built and walked by the library's torch
    implementation on every device, Warp's included; under ``"auto"`` the library
    picks."""
    from physicsnemo.mesh.spatial import cluster_tree

    seen = []
    for name in ("_resolve_build", "_resolve_walk"):
        real = getattr(cluster_tree, name)

        def spy(implementation, *args, _real=real, _name=name):
            seen.append((_name, implementation, _real(implementation, *args)))
            return seen[-1][-1]

        monkeypatch.setattr(cluster_tree, name, spy)
    pts, nrm, w, _ = (t.to(device) for t in cloud(300, seed=3))
    q = cloud(40, seed=7, scale=(4.0,) * 3)[0].to(device)
    for kind in ("reference", "auto"):
        seen.clear()
        forest = BoundaryForest.build(
            pts, w / w.sum(), nrm, None, theta=0.5, build=kind
        )
        forest.point_pairs(q, walk=kind)
        if kind == "reference":
            assert {s[0] for s in seen} == {"_resolve_build", "_resolve_walk"}
            assert all(s[1:] == ("torch", "torch") for s in seen), seen
        else:
            assert all(s[1] is None for s in seen), seen
