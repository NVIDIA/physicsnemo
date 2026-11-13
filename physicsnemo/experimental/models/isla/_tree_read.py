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

r"""The interior task's boundary read: passive prediction points read every boundary cell.

In the interior task, every passive prediction point :math:`q` gets one more residual
update after every slice block: a read of every boundary cell :math:`j`. A softmax over
the cells of

.. math::

    z_{qjh} = K_h(\iota_{qj}) + \langle q_{qh}, k_{jh} \rangle / \sqrt{c} + \log \omega_j

weights the cells' values :math:`v_{jh}`. With :math:`\omega_j` the cell's share of the
boundary measure, the read is a measure-weighted mean over the boundary (a
quadrature). The kernel inputs :math:`\iota_{qj}` are invariants of the pair's relative
geometry (:func:`tree_pair_invariants`). Once per forward pass, one MLP maps them to
``KERNEL_READ_BASIS`` basis functions per pair (:class:`_KernelBasis`), and each
block's kernel :math:`K_h` is a learned linear combination of that basis per head. The
values, keys and queries are linear maps of invariant token states (the cell's and the
point's own), so the read is invariant and the equivariant heads assemble every
output. Read pair by pair, it costs one term per (point, cell) pair, :math:`Q N` per
block: 1e8 on an airfoil of a thousand cells, 4e10 on a car of 200,000.

The read runs through GLOBE's cluster tree
(:class:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree`), which each prediction
point walks once
(:meth:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree.find_point_interaction_pairs`),
as an N-body code walks its tree once per body. A point reads the cells of nearby
leaves one by one, and a distant cluster :math:`C` through one *virtual cell*: the
cluster's measure-weighted centroid, its measure-weighted mean normal (renormalized),
its total measure :math:`\Omega_C = \sum_{j \in C} \omega_j`, and the measure-weighted
means of its cells' keys and values at the block's depth. Because the read is a
quadrature, the virtual cell is the cluster's monopole: if the pair logit were constant
over the cluster, :math:`\sum_{j \in C} \omega_j e^{z_{qj}} v_j` would equal
:math:`\Omega_C e^{z_{qC}} v_C` exactly.

The kernel reads a cell's normal only through cosines, so a cluster is a usable virtual
cell only when its normals agree. The mean normal of a cluster spanning both faces of a
thin body is a random direction. The boundary is therefore held as a *forest*
(:class:`BoundaryForest`). The orientation split groups the cells by the cube face of
their normal: the axis of its largest component and that component's sign, so opposite
normals never share a tree. One cluster tree is built per face. When the cells declare
boundary conditions, each condition gets its own face trees, so a node summarises one
condition and its quadrature is that condition's own.

A point walks every tree. A cluster is admitted when its box diagonal is below
:math:`\theta` times its distance to the point (the tree's criterion) and its normals
are coherent: the measure-weighted RMS chord of the normals about their mean,
:math:`\sqrt{2 (1 - |\bar n|)}`, is below :math:`\theta` (the tree's
``source_admissible`` mask). The orientation split already gives
:math:`|\bar n| \ge 1/\sqrt{d}`, so the coherence test rejects no cluster at
:math:`\theta > \sqrt{2 (1 - 1/\sqrt{d})}` (0.92 in three dimensions, 0.77 in two),
the default included; it acts only at smaller angles. A leaf that is not admitted
contributes its cells; any other node hands the point to its two children. At
:math:`\theta = 0` no cluster is admitted and the read is exact, pair for pair. The
model reads at its ``tree_read_theta`` (default ``TREE_READ_THETA``).

Symmetries. A point's list depends on the forest and the point alone, so the read is
exactly query independent at every :math:`\theta`: a point's prediction depends on the
boundary and the point, never on the other prediction points. At :math:`\theta = 0`
every contract of the exact read holds exactly: Euclidean equivariance, the similarity
gauge, boundary-refinement and measure-scale invariance. At finite :math:`\theta` the
forest is built in a frame that moves with the boundary
(:func:`~physicsnemo.mesh.spatial._frame.canonical_frame`: the principal axes of the
cells' measure-weighted covariance, signed by their third moments). A rotation,
reflection, translation or rescale of the case then gives the same clusters and the
same far-field list for every point, up to ties of that frame (equal variances, a
symmetric axis), so the read is the same approximation in every pose. At
:math:`\theta > 0` the read is piecewise smooth: the output jumps where a cluster's
admission changes, as the point or the boundary's vertices move, and a virtual cell's
position and normal take no gradient. At :math:`\theta = 0` the read is smooth, and the
read's own derivatives in the prediction points and in the boundary's cells are complete:
the pair invariants take a gradient whenever the geometry requires one
(:func:`_chunked_basis_from_plan`). The whole model's derivative in the boundary's
vertices also passes through the token draw's weights and the slice blocks (see
``tree_read_theta`` in :class:`~physicsnemo.experimental.models.isla.ISLA`).

Rounding. Rounding makes no discrete choice of the forest that an error bound can
settle, and the caller's precision makes none. The forest is built and
walked outside autocast, in the inputs' precision. Its coordinates, in the boundary's
frame about its centroid, are computed in float64 and rounded once, so a bf16-autocast
forward builds bitwise the float32 forward's forest. A normal's face is tied when its
two largest components lie within the bound on their rounding
(:func:`normal_error_bound`) plus the frame's rotation (:func:`frame_rotation_bound`);
a tied normal takes the lowest tied axis. The box test compares the distance with a
diameter widened by the coordinates' error bound and by the frame's rotation
(:func:`separation_diameter_sq`). The rotation moves the points and the cells
together, so it widens each node in proportion to its own size, never past a
frame-free bound. An exact tie of the criterion is thus opened in every pose, and a
nearly tied frame costs a bounded factor of pairs. The bounds are measured in the
boundary's own frame: they are functions of the boundary alone, and their edges do not
move with the pose. Rounding still decides the generic cases: cells within rounding of
a face boundary, a Morton grid line or the criterion, whose posed inputs differ by the
rounding of the pose itself.

Constants. At a fixed :math:`\theta`, three choices change the cost and the grouping
of the exact pairs, and at :math:`\theta = 0` they change nothing: the leaf size (cells
per leaf: the model's ``tree_read_leaf_size``, default ``TREE_READ_LEAF_SIZE``), the
orientation split into the :math:`2 d` cube faces, and ``TREE_READ_PAIR_CHUNK`` (pairs
per chunk of the basis evaluation).
"""

import dataclasses
import functools
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.utils.checkpoint import checkpoint

from physicsnemo.experimental.models.isla import _fused_basis as fused_basis
from physicsnemo.experimental.models.isla import _fused_read as fused_read
from physicsnemo.experimental.models.isla._segment_softmax import (
    PairPlan,
    segment_softmax_read,
)
from physicsnemo.experimental.models.isla._tree_build import forest_tables, resolve_tree_build
from physicsnemo.mesh.spatial import canonical_frame, frame_coordinates
from physicsnemo.mesh.spatial._ragged import _ragged_arange
from physicsnemo.mesh.spatial.cluster_tree import ClusterTree

logger = logging.getLogger(__name__)
#: Why the fused read used the reference kernel basis, each reason logged once.
_BASIS_FALLBACKS_LOGGED: set[str] = set()
#: The walks' resolutions (:func:`resolve_walk`), each logged once.
_WALKS_LOGGED: set[tuple[str, str]] = set()

#: Heads of the read; each head owns ``KERNEL_READ_WIDTH // KERNEL_READ_HEADS`` value
#: channels.
KERNEL_READ_HEADS = 8
#: Value channels of the read (all heads together).
KERNEL_READ_WIDTH = 64
#: Query/key channels per head.
KERNEL_READ_KEY = 4
#: Hidden width of the shared kernel MLP on the pair invariants.
KERNEL_READ_HIDDEN = 64
#: Kernel basis functions of the pair invariants, shared by every block; each block's
#: kernel is a learned linear combination of them per head.
KERNEL_READ_BASIS = 16
#: The default Barnes-Hut opening angle of the model's read (ISLA's
#: ``tree_read_theta``). Read a model at the angle it was trained at: the kernel learns
#: to read the clusters' aggregates.
TREE_READ_THETA = 1.0

#: The default boundary cells per tree leaf (ISLA's ``tree_read_leaf_size``): at a fixed
#: ``theta`` it sets how many cells an opened leaf contributes; at ``theta = 0`` it
#: changes nothing.
TREE_READ_LEAF_SIZE = 4
#: Pairs per chunk of the kernel-basis evaluation (memory only; each chunk's
#: invariants are built once, its MLP activations are rebuilt in backward).
TREE_READ_PAIR_CHUNK = 1 << 22

#: Inductor options of the read's compiled projections (``compile_blocks``): mix-order
#: reduction off, as for the slice blocks (``model._BLOCK_COMPILE_OPTIONS``).
_READ_COMPILE_OPTIONS = {"triton.mix_order_reduction": False}
_COMPILED: dict = {}


def _compiled(fn):
    """``fn`` through :func:`torch.compile` with dynamic shapes, built once per function.
    Parameters enter as arguments, so one graph serves every block and every point count
    (sizes 0 and 1 aside, which torch.compile specializes)."""
    if fn not in _COMPILED:
        _COMPILED[fn] = torch.compile(fn, dynamic=True, options=_READ_COMPILE_OPTIONS)
    return _COMPILED[fn]


def normal_faces(normals: torch.Tensor, tol: torch.Tensor) -> torch.Tensor:
    """The cube face of each normal, ``0`` to ``2 d - 1``: ``2 k + 1`` when component
    ``k`` is the largest in absolute value and positive, ``2 k`` when it is negative.
    ``tol`` ``(N,)`` bounds the error of each of a normal's components; a component
    within ``2 tol`` of the largest may be the largest, and the face goes to the lowest
    such axis, so rounding never decides it. Antipodal: ``-n`` lands on the face
    opposite ``n``'s (the same axis, the other sign), so opposite normals never share
    a face and a winding flip maps every face's cells to the opposite face. Inside a
    face every normal has ``|n_k| >= 1 / sqrt(d)`` (up to ``tol``)."""
    a = normals.abs()
    tied = a >= a.amax(dim=1, keepdim=True) - 2 * tol[:, None]
    k = tied.to(torch.int8).argmax(dim=1)
    return 2 * k + (normals.gather(1, k[:, None])[:, 0] > 0).long()


def normal_error_bound(simplex: torch.Tensor, centre: torch.Tensor) -> torch.Tensor:
    r"""A bound ``(C,)`` on the error of every component of each cell's unit normal,
    from its vertices ``simplex`` ``(C, d, d)`` as given and the centroid ``centre``:
    the boundary's ``(d,)``, or each cell's own condition's ``(C, d)``.

    Let :math:`u` be the unit roundoff of the vertices' precision, :math:`V` the
    largest distance of a vertex from the centroid and :math:`e_j` the edges from the
    first vertex. The normal :math:`c / |c|` of the edges' generalized cross product
    :math:`c` then moves by at most :math:`u (6 V \sum_k \prod_{j \ne k} |e_j| + 2 (d -
    1) \prod_j |e_j|) / |c| + (d + 2) u`. The terms are each vertex's rounding
    (:math:`u V`), and the edges and the product computed and normalized in that
    precision. :math:`V` is measured from the centroid, so the bound is a function of
    the boundary alone, the same in every pose; a pose far from the origin adds its own
    rounding, which no function of the boundary can follow. Computed in float64; a
    zero-measure cell gets an infinite bound."""
    v = simplex.double()
    d = v.shape[-1]
    u = torch.finfo(simplex.dtype).eps / 2
    c = centre.double()
    big = (v - (c if c.dim() == 1 else c[:, None])).norm(dim=-1).amax(dim=1)
    e = (v[:, 1:] - v[:, :1]).norm(dim=-1)  # (C, d - 1)
    if d == 2:
        c = e[:, 0]
        rest = torch.ones_like(c)
    else:
        edges = v[:, 1:] - v[:, :1]
        c = torch.linalg.cross(edges[:, 0], edges[:, 1], dim=-1).norm(dim=-1) if d == 3 else (
            torch.linalg.det(edges @ edges.transpose(1, 2)).clamp_min(0).sqrt()
        )
        ### Slices: an index list would be copied from the host.
        rest = torch.stack([torch.cat([e[:, :k], e[:, k + 1 :]], 1).prod(dim=1) for k in range(d - 1)], 1).sum(1)
    full = e.prod(dim=1)
    bound = u * (6 * big * rest + 2 * (d - 1) * full) / c + (d + 2) * u
    return torch.where(c > 0, bound, torch.full_like(bound, math.inf))


def frame_rotation_bound(x: torch.Tensor, w: torch.Tensor, u: float) -> torch.Tensor:
    r"""A bound on the rotation of the canonical frame by the rounding of its inputs:
    positions ``x`` ``(N, d)`` about their centroid with normalized weights ``w``, each
    coordinate in error by at most :math:`2 u |x|` (the input's rounding and the division
    by the length unit), move the covariance by at most :math:`E = 4 u \operatorname{tr} C`,
    and its axes by at most :math:`E / (g - E)` with :math:`g` the smallest eigenvalue gap
    (Davis-Kahan). Zero when :math:`E \ge g`: the variances are then equal to rounding, a
    tie of the frame, which no bound on the forest can cover. In float64."""
    x, w = x.double(), w.double()
    cov = torch.stack([(w[:, None] * x * x[:, j, None]).sum(0) for j in range(x.shape[1])], 1)
    lam = torch.linalg.eigvalsh(cov)
    err = 4 * u * lam.sum()
    gap = (lam[1:] - lam[:-1]).amin() if lam.numel() > 1 else lam.new_full((), math.inf)
    return torch.where(gap > err, err / (gap - err), torch.zeros_like(err))


def separation_diameter_sq(
    aabb_min: torch.Tensor, aabb_max: torch.Tensor, theta: float, eta: torch.Tensor, d: int, u: float
) -> torch.Tensor:
    r"""The squared diameter the walk compares a point's distance with for the nodes
    with boxes ``aabb_min``, ``aabb_max`` ``(M, d)`` (elementwise per node), so that a
    node is admitted only when it is certainly well separated in the boundary's exact
    canonical frame: the true distance :math:`r^*` of the point from the node's box and
    the box's true diagonal :math:`D^*` satisfy :math:`\theta r^* > D^*` whenever the
    computed :math:`\theta r > D'`, with :math:`D` the computed diagonal and :math:`R`
    the largest distance of the box from the centroid. :math:`D'` is the smaller of two
    sufficient diameters:

    - :math:`D_\eta = f D_u(\theta / f)`. Rounding moves each coordinate of a point
      :math:`p` by at most :math:`3 u |p|` (in the centred frame), so with the walk's
      own float32 arithmetic :math:`\theta r > D_u(\theta) = (D (1 + 3 \theta u) + 6 (1 +
      \theta) u R) (1 + 2 u) / (1 - (d + 6) u)` certifies the criterion in the computed
      frame. That frame is rotated from the exact one by at most ``eta`` per coordinate
      (:func:`frame_rotation_bound`), and the rotation moves the point and the node's
      cells together: the exact box's diagonal is at most :math:`D (1 + \sqrt d \eta)`
      and its distance at least :math:`r (1 - \sqrt d \eta) - \sqrt d \eta D`, so
      :math:`\theta r > f D` with :math:`f = (1 + \sqrt d \eta (1 + \theta)) / (1 -
      \sqrt d \eta)` certifies it. The rotation thus widens each node in proportion to
      its own size, not to its distance from the centroid. Defined for :math:`\sqrt d
      \eta < 1`.
    - :math:`D_\circ`, which holds whatever the frame's rotation. The node's cells lie
      in the ball of radius :math:`D / 2` about its box's centre :math:`c`, so in any
      frame its box lies in the cube of half side :math:`D / 2` about :math:`c`, and
      :math:`\theta |q - c| > (\theta + 2) \sqrt d D / 2` certifies the criterion; the
      walk's distance to the box is at most :math:`|q - c|`. With the rounding
      (:math:`a = 3 \sqrt d u` on a vector), :math:`D_\circ = ((\theta + 2) \sqrt d (D / 2
      + a R) + \theta a R) (1 + 2 u) / ((1 - a) (1 - (d + 3) u))`.

    So the widening grows with ``eta`` and stays bounded as the frame's variances tie.
    A tie of the criterion (the structured meshes' distance equal to the diameter) is
    opened whichever way rounding falls."""
    lo, hi = aabb_min.double(), aabb_max.double()
    diam = (hi - lo).square().sum(-1).sqrt()
    reach = torch.maximum(lo.abs(), hi.abs()).norm(dim=-1)
    k = 3 * u

    def rounded(t):
        return (diam * (1 + t * k) + 2 * (1 + t) * k * reach) * (1 + 2 * u) / (1 - k - (d + 3) * u)

    a = math.sqrt(d) * k
    ball = ((theta + 2) * math.sqrt(d) * (diam / 2 + a * reach) + theta * a * reach) * (1 + 2 * u)
    ball = ball / ((1 - a) * (1 - (d + 3) * u))
    se = math.sqrt(d) * eta.double()
    sure = se < 1
    se = torch.where(sure, se, torch.zeros_like(se))
    f = (1 + se * (1 + theta)) / (1 - se)
    wide = torch.where(sure, torch.minimum(f * rounded(theta / f), ball), ball)
    return wide.square().to(aabb_min.dtype)


def _reference_tables(
    points, measure, normals, vecs, points_f, face, leaf_size: int, n_trees: int, implementation: str | None = None
) -> dict:
    """:func:`~._tree_build.forest_tables`' fields from the reference build: one GLOBE
    :class:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree` per tree of ``face``
    (each cell's tree, of ``n_trees``), in tree order, its nodes and cells appended to
    the forest's tables. ``implementation`` is the trees' build (``"torch"`` for a
    forced reference; ``None``: the library's choice)."""
    n, d = points.shape
    device = points.device
    trees, tree_cells, tree_index, cells, offsets = [], [], [], [], [0]
    starts, counts, lefts, rights, boxes = [], [], [], [], []
    centers, nrm, vec, meas = [], [], [], []
    c_off = m_off = 0
    for k in range(n_trees):
        sel = (face == k).nonzero().squeeze(-1)
        if sel.numel() == 0:
            continue
        tree_index.append(k)
        tree = ClusterTree.from_points(
            points_f[sel], leaf_size=leaf_size, areas=measure[sel], implementation=implementation
        )
        data = {"n": normals[sel]}
        if vecs is not None:
            data["u"] = vecs[sel]
        agg = tree.compute_source_aggregates(
            points[sel], measure[sel], TensorDict(data, batch_size=[int(sel.numel())])
        )
        m = tree.n_nodes
        trees.append(tree)
        tree_cells.append(sel)
        cells.append(sel[tree.sorted_source_order])
        starts.append(tree.node_range_start + c_off)
        counts.append(tree.node_range_count)
        lefts.append(torch.where(tree.node_left_child >= 0, tree.node_left_child + m_off, -1))
        rights.append(torch.where(tree.node_right_child >= 0, tree.node_right_child + m_off, -1))
        boxes.append((tree.node_aabb_min, tree.node_aabb_max))
        centers.append(agg.node_centroid)
        nrm.append(agg.node_source_data["n"])
        if vecs is not None:
            vec.append(agg.node_source_data["u"])
        meas.append(tree.node_total_area)
        c_off += int(sel.numel())
        m_off += m
        offsets.append(m_off)
    left = torch.cat(lefts)
    right = torch.cat(rights)
    range_start = torch.cat(starts)
    range_count = torch.cat(counts)
    ### The leaves tile the cells: sorted by their first position, their counts
    ### concatenate to the forest order.
    leaf_ids = (left < 0).nonzero().squeeze(-1)
    leaf_ids = leaf_ids[torch.argsort(range_start[leaf_ids])]
    position_leaf = torch.repeat_interleave(
        torch.arange(leaf_ids.shape[0], device=device), range_count[leaf_ids], output_size=n
    )
    ### Internal nodes by depth from the roots, each depth in ascending id; a node's
    ### children are one level deeper.
    node_offset = torch.tensor(offsets, device=device)
    frontier = node_offset[:-1]
    by_depth = []
    while frontier.numel():
        internal = frontier[left[frontier] >= 0]
        if internal.numel() == 0:
            break
        by_depth.append(internal)
        frontier = torch.cat([left[internal], right[internal]])
    levels = [(nodes, left[nodes], right[nodes]) for nodes in (x.sort().values for x in reversed(by_depth))]

    def per_tree(diameter_sq: torch.Tensor) -> tuple[list[ClusterTree], list[torch.Tensor]]:
        """The face trees with their slices of the walk's node diameters ``(M,)``, and
        the cells each holds."""
        m = 0
        for tree in trees:
            tree.node_diameter_sq = diameter_sq[m : m + tree.n_nodes]
            m += tree.n_nodes
        return trees, tree_cells

    return {
        "per_tree": per_tree,
        "node_offset": node_offset,
        "cells": torch.cat(cells),
        "range_start": range_start,
        "range_count": range_count,
        "left": left,
        "right": right,
        "leaf_ids": leaf_ids,
        "position_leaf": position_leaf,
        "levels": levels,
        "aabb_min": torch.cat([b[0] for b in boxes]),
        "aabb_max": torch.cat([b[1] for b in boxes]),
        "center": torch.cat(centers),
        "mean_n": torch.cat(nrm),
        "vecs": torch.cat(vec) if vecs is not None else None,
        "measure": torch.cat(meas),
        "tree_index": torch.tensor(tree_index, dtype=torch.long, device=device),
    }


@dataclass
class BoundaryForest:
    """One case's boundary as a forest of cluster trees, one per normal face (per
    group of cells and face, when the cells are grouped by their declared condition), with
    the trees' nodes in one table (tree by tree, each tree's nodes in its own order)
    and the cells in one order (tree by tree, each tree's cells in its sorted order).

    Node ``m`` holds the cells ``cells[range_start[m] : range_start[m] +
    range_count[m]]``; its virtual cell is ``center[m]``, ``normal[m]``,
    ``vecs[m]`` and ``measure[m]``; ``coherent[m]`` says whether a point may read it
    as one (RMS chord of its normals below ``theta``). ``levels`` groups the internal
    nodes by depth, deepest first, with their children, for the bottom-up sums of
    :func:`node_means`.
    """

    n_cells: int
    theta: float
    frame: torch.Tensor  # (d, d) float64, the boundary's own frame (canonical_frame)
    origin: torch.Tensor  # (d,) float64, the cells' measure-weighted centroid
    group_origin: torch.Tensor  # (G, d) float64, each group's measure-weighted centroid; its trees live in (x - origin_g) @ frame
    tree_group: torch.Tensor  # (T,) int64, each tree's group (row of group_origin)
    per_tree: Callable[[torch.Tensor], tuple[list[ClusterTree], list[torch.Tensor]]]  # ``trees`` and ``tree_cells`` from ``diameter_sq``
    node_offset: torch.Tensor  # (T + 1,) each tree's first node in the table
    cells: torch.Tensor  # (N,) every cell once, in forest order
    range_start: torch.Tensor  # (M,) each node's first position in ``cells``
    range_count: torch.Tensor  # (M,)
    left: torch.Tensor  # (M,) child in the table, -1 at a leaf
    right: torch.Tensor  # (M,)
    leaf_ids: torch.Tensor  # (L,) the leaves, in position order
    position_leaf: torch.Tensor  # (N,) the leaf (counted in ``leaf_ids``) of each position
    levels: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    aabb_min: torch.Tensor  # (M, d)
    aabb_max: torch.Tensor  # (M, d)
    diameter_sq: torch.Tensor  # (M,)
    center: torch.Tensor  # (M, d) measure-weighted centroid
    normal: torch.Tensor  # (M, d) renormalized measure-weighted mean normal
    vecs: torch.Tensor | None  # (M, P, d) measure-weighted mean rank-1 inputs
    measure: torch.Tensor  # (M,) total measure
    coherent: torch.Tensor  # (M,) bool

    @property
    def n_nodes(self) -> int:
        return int(self.range_start.shape[0])

    @functools.cached_property
    def _per_tree(self) -> tuple[list[ClusterTree], list[torch.Tensor]]:
        return self.per_tree(self.diameter_sq)

    @property
    def trees(self) -> list[ClusterTree]:
        """One :class:`ClusterTree` per tree over its slice of the tables, with the walk's
        node diameters: what the reference walk traverses. Made on first use (the fast
        walk reads the tables alone)."""
        return self._per_tree[0]

    @property
    def tree_cells(self) -> list[torch.Tensor]:
        """Per tree, the cells it holds (its sources, in its input order); made with
        :attr:`trees`."""
        return self._per_tree[1]

    @functools.cached_property
    def walk_tree(self) -> ClusterTree:
        """The whole forest as one :class:`ClusterTree` over the tables, one root per
        tree, with the cells in forest order as its sources (it holds no source points):
        what the fast walk traverses, so its near sources are cells and its far sources
        are nodes of the table."""
        leaf = self.left < 0
        return ClusterTree(
            node_aabb_min=self.aabb_min, node_aabb_max=self.aabb_max, node_diameter_sq=self.diameter_sq,
            node_left_child=self.left, node_right_child=self.right,
            leaf_start=torch.where(leaf, self.range_start, -1), leaf_count=torch.where(leaf, self.range_count, 0),
            node_range_start=self.range_start, node_range_count=self.range_count, node_total_area=self.measure,
            sorted_source_order=self.cells, source_points=self.aabb_min[:0],
            max_depth=torch.tensor(len(self.levels), device=self.cells.device), root_ids=self.node_offset[:-1],
            batch_size=torch.Size([]),
        )  # fmt: skip

    @classmethod
    def build(
        cls,
        points: torch.Tensor,
        measure: torch.Tensor,
        normals: torch.Tensor,
        vecs: torch.Tensor | None,
        *,
        theta: float,
        leaf_size: int = TREE_READ_LEAF_SIZE,
        eps: float = 1e-12,
        normal_tol: torch.Tensor | None = None,
        groups: torch.Tensor | None = None,
        build: str = "auto",
    ) -> "BoundaryForest":
        """The forest over cells at ``points`` ``(N, d)`` with ``measure`` ``(N,)``,
        unit ``normals`` ``(N, d)`` and rank-1 inputs ``vecs`` ``(N, P, d)`` or
        ``None``.

        ``normal_tol`` ``(N,)`` bounds the error of each normal's components
        (:func:`normal_error_bound`); ``None`` takes the normals as exact, up to their
        own rounding. ``groups`` ``(N,)`` int64 is each cell's declared condition
        (``None``: one group). Each group gets its own face trees, in coordinates about
        its own measure-weighted centroid; the frame and its rotation bound stay the
        whole boundary's. Trees go group by group in ascending order, then face by face,
        and one group gives the ungrouped forest. No gradient flows into the forest (a
        numerical device), and a forward under autocast builds the same forest (the
        module docstring's *Rounding*). ``build`` picks how the forest is built, never what it is
        (:func:`~._tree_build.resolve_tree_build`): the per-face cluster trees below, or
        the same forest from one pass over all faces
        (:func:`~._tree_build.forest_tables`)."""
        with torch.autocast(points.device.type, enabled=False):
            return cls._build(points, measure, normals, vecs, theta, leaf_size, eps, normal_tol, groups, build)

    @classmethod
    def _build(
        cls, points, measure, normals, vecs, theta, leaf_size, eps, normal_tol, groups, build
    ) -> "BoundaryForest":
        points, measure, normals = points.detach(), measure.detach(), normals.detach()
        vecs = None if vecs is None else vecs.detach()
        n, d = points.shape
        device = points.device
        coherent_length = 1.0 - 0.5 * float(theta) ** 2
        ### The trees and the faces live in the boundary's own frame about its centroid
        ### (the module docstring's Symmetries and Rounding); the virtual cells
        ### (centroids, normals, rank-1 means) stay in the frame the read works in.
        u = torch.finfo(points.dtype).eps / 2
        tiny = torch.finfo(torch.float64).tiny
        w_raw = measure.double().clamp_min(0)
        w64 = w_raw / w_raw.sum().clamp_min(tiny)
        p64 = points.double()
        frame = canonical_frame(p64, measure)
        origin = (w64[:, None] * p64).sum(0)
        x = p64 - origin
        normals_f = frame_coordinates(normals.double(), frame)
        ### The bound on the frame's rotation by its inputs' rounding: the frame is the
        ### whole boundary's, so the bound is too, for every group (a group's own
        ### covariance is not the frame's: a square group's two nearly equal variances
        ### would bound a rotation the forest never makes).
        eta = frame_rotation_bound(x, w64, u)
        if normal_tol is None:
            tol = torch.full((n,), (d + 2) * torch.finfo(normals.dtype).eps / 2, dtype=torch.float64, device=device)
        else:
            tol = normal_tol.double()
        ### One set of face trees per group (declared condition), each about its group's
        ### own centroid; one group is the whole boundary. ``face`` is each cell's tree:
        ### 2 d g plus its normal's face. Finding the groups and their sizes costs one sort
        ### and one readback. Each group's total measure and centroid sum its own cells in
        ### their order (a slice of the cells sorted by group), so each equals the sum on
        ### the group alone; the rest is elementwise over all the cells.
        sizes = None
        if groups is not None:
            by_group = torch.argsort(groups, stable=True)
            present, sizes = torch.unique_consecutive(groups[by_group], return_counts=True)
            sizes = sizes.tolist()
        if sizes is None or len(sizes) == 1:
            points_f = frame_coordinates(x, frame).to(points.dtype)
            face = normal_faces(normals_f, tol + eta)
            group_origin = origin[None]
        else:
            rank = torch.searchsorted(present, groups)
            totals = torch.stack([w.sum() for w in w_raw[by_group].split(sizes)]).clamp_min(tiny)
            weighted = (w_raw / totals[rank])[:, None] * p64
            group_origin = torch.stack([s.sum(0) for s in weighted[by_group].split(sizes)])
            points_f = frame_coordinates(p64 - group_origin[rank], frame).to(points.dtype)
            face = 2 * d * rank + normal_faces(normals_f, tol + eta)
        n_trees = 2 * d * group_origin.shape[0]
        tables = None
        if resolve_tree_build(build, points) == "fast":
            tables = forest_tables(points, measure, normals, vecs, points_f, face, leaf_size=leaf_size, n_trees=n_trees)
        if tables is None:
            ### A forced reference builds its trees by the library's torch code too.
            implementation = "torch" if build == "reference" else None
            tables = _reference_tables(points, measure, normals, vecs, points_f, face, leaf_size, n_trees, implementation)
        tree_group = tables.pop("tree_index") // (2 * d)
        ### The walk's node diameters (elementwise per node), on the whole table; the
        ### trees take their slices of it when they are made.
        diameter_sq = separation_diameter_sq(tables["aabb_min"], tables["aabb_max"], float(theta), eta, d, u)
        mean_n = tables.pop("mean_n")
        length = mean_n.norm(dim=-1)
        return cls(
            n_cells=n,
            theta=float(theta),
            frame=frame,
            origin=origin,
            group_origin=group_origin,
            tree_group=tree_group,
            **tables,
            diameter_sq=diameter_sq,
            normal=mean_n / length[:, None].clamp_min(eps),
            coherent=length > coherent_length,
        )

    def node_cells(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The cells of the listed nodes: ``(cell, which)`` with ``which`` the index into
        ``nodes`` each cell belongs to."""
        pos, which = _ragged_arange(self.range_start[nodes], self.range_count[nodes])
        return self.cells[pos], which

    def point_pairs(self, points: torch.Tensor, walk: str = "auto") -> tuple[torch.Tensor, torch.Tensor, int]:
        """Every point's list ``(qi, si, n_node)``, sorted by point. ``si`` below
        ``n_cells`` names a cell and ``n_cells + m`` names node ``m``; ``n_node`` counts
        the node entries. Each tree is walked once per point at the forest's ``theta``,
        for which its diameters and coherence were made. A node is admitted only when
        its box is well separated from the point and its normals are coherent, so each
        point's list holds every cell exactly once, as itself or inside one node.
        ``walk`` picks how, never what (:func:`resolve_walk`): the fast walk gives the
        same lists in the same order. In each point's list the cells come first, tree by
        tree, then the nodes, tree by tree; each tree's entries are in the order of its
        walk (:meth:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree.find_point_interaction_pairs`)."""
        return self.frame_pairs(self.frame_points(points), walk)

    def frame_points(self, points: torch.Tensor) -> torch.Tensor:
        """The points in the trees' coordinates, as the cells are (:meth:`build`):
        about each group's origin in float64, in the forest's frame, rounded once to the
        points' dtype, outside autocast; ``(Q, d)`` for a forest of one group, ``(G, Q,
        d)`` for ``G`` groups, whose trees each read their own group's row."""
        with torch.autocast(points.device.type, enabled=False):
            x = points.detach().double()
            if self.group_origin.shape[0] == 1:
                return frame_coordinates(x - self.origin, self.frame).to(points.dtype)
            return torch.stack([frame_coordinates(x - o, self.frame) for o in self.group_origin]).to(points.dtype)

    def frame_pairs(
        self, points_f: torch.Tensor, walk: str = "auto", count_only: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, int] | torch.Tensor:
        """:meth:`point_pairs` of points already in the trees' coordinates
        (:meth:`frame_points`); with ``count_only`` each point's number of pairs
        ``(Q,)`` alone, which the walk counts without making the lists."""
        with torch.autocast(points_f.device.type, enabled=False):
            return self._frame_pairs(points_f, self.theta, walk, count_only)

    def _frame_pairs(self, points_f, theta, walk, count_only):
        if resolve_walk(walk, points_f, theta, self) == "fast":
            ### One walk of every tree, each reading its group's row of the points; the
            ### library falls back to its torch walk, the same plan, where Warp cannot walk.
            plan = self.walk_tree.find_point_interaction_pairs(
                points_f,
                theta,
                validate=False,
                source_admissible=self.coherent,
                implementation="warp" if walk == "fast" else None,
                count_only=count_only,
                root_rows=None if points_f.dim() == 2 else self.tree_group,
            )
            if count_only:
                return plan[0] + plan[1]
            ### Each point's cells, then its nodes: both streams are grouped by point, so
            ### an entry's place is its place in its stream plus the other stream's
            ### entries of the points before it (and of its own point, for a node).
            near_t, far_t = plan.near_target_ids, plan.far_target_ids
            ends = torch.arange(points_f.shape[-2] + 1, device=near_t.device)
            near_at = torch.arange(plan.n_near, device=near_t.device) + torch.searchsorted(far_t, ends)[near_t]
            far_at = torch.arange(plan.n_far, device=near_t.device) + torch.searchsorted(near_t, ends)[far_t + 1]
            qi = near_t.new_empty(plan.n_near + plan.n_far)
            si = torch.empty_like(qi)
            qi[near_at], si[near_at] = near_t, plan.near_source_ids
            qi[far_at], si[far_at] = far_t, self.n_cells + plan.far_source_node_ids
            return qi, si, plan.n_far
        qs, ss, n_node = [], [], 0
        for t, tree in enumerate(self.trees):
            a, b = int(self.node_offset[t]), int(self.node_offset[t + 1])
            pts = points_f if points_f.dim() == 2 else points_f[int(self.tree_group[t])]
            plan = tree.find_point_interaction_pairs(
                pts,
                theta=theta,
                validate=False,
                source_admissible=self.coherent[a:b],
                implementation="torch" if walk == "reference" else None,
            )
            qs += [plan.near_target_ids, plan.far_target_ids]
            ss += [self.tree_cells[t][plan.near_source_ids], self.n_cells + a + plan.far_source_node_ids]
            n_node += plan.n_far
        qi, si = torch.cat(qs), torch.cat(ss)
        if count_only:
            return torch.bincount(qi, minlength=points_f.shape[-2])
        ### By point, then cells before nodes, keeping the trees' order (the fast walk's).
        order = torch.argsort(2 * qi + (si >= self.n_cells), stable=True)
        return qi[order], si[order], n_node


def resolve_walk(walk: str, points: torch.Tensor, theta: float, forest: BoundaryForest) -> str:
    """The walk :meth:`BoundaryForest.point_pairs` runs: ``"reference"`` (GLOBE's
    cluster-tree traversal, tree by tree) or ``"fast"`` (one library walk of the whole
    forest, :attr:`BoundaryForest.walk_tree`, in Warp where it can walk: the same pairs
    in the same order). ``"auto"`` picks the fast walk for CUDA points, else the
    reference, and logs each choice once per device type. A forced name is kept
    (``"fast"`` runs Warp on CPU too, and raises where Warp cannot walk; a forced
    ``"reference"`` runs GLOBE's torch traversal, never its Warp one, and
    :meth:`BoundaryForest.build` likewise builds its trees). At ``theta = 0`` there is
    no walk (the plan is every pair), and the reference's construction runs either way.
    Both walks run outside autocast, in the forest's precision, and the Warp walk
    rounds as the torch walk's kernels do."""
    if walk not in ("auto", "reference", "fast"):
        raise ValueError(f"walk must be 'auto', 'reference' or 'fast', got {walk!r}")
    if theta == 0:
        return "reference"
    if walk != "auto":
        return walk
    chosen = "fast" if points.is_cuda else "reference"
    key = (points.device.type, chosen)
    if key not in _WALKS_LOGGED:
        _WALKS_LOGGED.add(key)
        logger.info(
            "ISLA tree walk=auto resolved to %r on %s (forest depth %d, %s boxes)",
            chosen,
            points.device.type,
            len(forest.levels),
            forest.aabb_min.dtype,
        )
    return chosen


@dataclass
class TreeReadPairs:
    """The tree read's pairs of one forward pass, every case flattened: ``plan`` over
    the ``B * Q`` points and the virtual sources (each case's ``N`` cells followed by
    its forest's nodes), the kernel basis of every pair, every source's log measure,
    the cells' measure fractions, and each case's forest."""

    plan: PairPlan
    basis: torch.Tensor  # (P, KERNEL_READ_BASIS)
    src_log_m: torch.Tensor  # (n_b,) every source's log measure
    omega: torch.Tensor  # (B * N,) the cells' measure fractions
    forests: list[BoundaryForest]
    n_cells: int
    n_queries: int
    n_pairs_node: int  # (point, node) pairs over every case
    kernel: str = "reference"  # the blocks' read: "reference" or "fused" (_fused_read.py)
    basis_sink: tuple | None = None  # the fused reads' shared basis gradient (fused_read.basis_sink)

    @property
    def n_pairs(self) -> int:
        return int(self.plan.qi.shape[0])

    @property
    def log_m(self) -> torch.Tensor:
        """Every pair's log measure ``(P,)``, its source's (the reference read's input;
        the fused read gathers ``src_log_m`` itself)."""
        return self.src_log_m[self.plan.bj.long()]


def node_means(x: torch.Tensor, forest: BoundaryForest, omega: torch.Tensor) -> torch.Tensor:
    """The measure-weighted mean ``(M, ...)`` of the cells' rows ``x`` ``(N, ...)`` over
    every node of the forest, differentiable in ``x``. One scatter over the forest
    order gives the leaves' sums; each internal node is then the sum of its two
    children, depth by depth up to the roots (a tree of pairwise sums, no
    cancellation). The sums run in float64 for float64 rows and in float32 for
    float32, bfloat16 and float16 rows."""
    work = torch.float64 if x.dtype == torch.float64 else torch.float32
    cells = forest.cells
    flat = x.reshape(x.shape[0], -1).to(work)[cells]
    weighted = flat * omega[cells, None].to(work)
    n_leaves = forest.leaf_ids.shape[0]
    leaf_sums = weighted.new_zeros(n_leaves, weighted.shape[1]).index_add(
        0, forest.position_leaf, weighted
    )
    sums = weighted.new_zeros(forest.n_nodes, weighted.shape[1]).index_put(
        (forest.leaf_ids,), leaf_sums
    )
    for nodes, left, right in forest.levels:
        sums = sums.index_put((nodes,), sums[left] + sums[right])
    measure = forest.measure.to(work).clamp_min(torch.finfo(work).tiny)
    return (sums / measure[:, None]).to(x.dtype).reshape(forest.n_nodes, *x.shape[1:])


def tree_pair_invariants(
    qi: torch.Tensor,
    si: torch.Tensor,
    *,
    query_r: torch.Tensor,
    query_n: torch.Tensor,
    query_wall: torch.Tensor,
    src_r: torch.Tensor,
    src_n: torch.Tensor,
    src_vecs: torch.Tensor | None,
    g_unit: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    r"""The kernel inputs of the listed (point, source) pairs of one case, ``(P, G)``,
    the sources either cells or virtual cells.

    ``query_r``, ``query_n`` ``(Q, d)``, ``query_wall`` ``(Q,)`` the point's built-in
    wall distance :math:`|d_q|` (frame units); ``src_r``, ``src_n`` ``(S, d)``, ``src_vecs`` ``(S, P_b, d)`` or ``None``;
    ``g_unit`` ``(K, d)``. With :math:`\rho = (x_q - y_j) / L`, :math:`r = |\rho|` and
    :math:`\hat\rho = \rho / r` the invariants are :math:`[r, \log r, \log(r / |d_q|),
    \hat\rho\cdot n_j, \hat\rho\cdot n_q, n_q\cdot n_j, \hat\rho\cdot g_k, n_j\cdot g_k,
    \hat\rho\cdot u, n_q\cdot u]` (``u`` the source's rank-1 inputs): lengths in units of :math:`L` or of the point's
    own wall distance, and cosines, so no length scale is chosen. The only constant is
    ``eps``, a floor on the lengths inside the logarithms and the normalization (a
    numerical guard: :math:`r \ge |d_q| > 0` for every point off the boundary and every
    cell centroid source, since :math:`|d_q|` is the distance to the boundary surface,
    on which the cell centroids lie)."""
    with torch.autocast(query_r.device.type, enabled=False):
        rel = query_r[qi] - src_r[si]
        dist = rel.norm(dim=-1)
        dist_g = dist.clamp_min(eps)
        rho = rel / dist_g[:, None]
        n_q, n_j = query_n[qi], src_n[si]
        log_r = torch.log(dist_g)
        feats = [
            dist,
            log_r,
            log_r - torch.log(query_wall[qi].clamp_min(eps)),
            (rho * n_j).sum(-1),
            (rho * n_q).sum(-1),
            (n_q * n_j).sum(-1),
        ]
        cols = [torch.stack(feats, -1), rho @ g_unit.t(), n_j @ g_unit.t()]
        if src_vecs is not None and src_vecs.shape[1]:
            u = src_vecs[si]
            cols += [(rho[:, None] * u).sum(-1), (n_q[:, None] * u).sum(-1)]
        return torch.cat(cols, -1)


def kernel_geo_width(n_global_vectors: int, n_boundary_vecs: int) -> int:
    """Width of :func:`tree_pair_invariants`: r, log r, log(r / |d_q|), rho.n_j, rho.n_q,
    n_q.n_j, then rho.g_k and n_j.g_k (2K), rho.u and n_q.u per boundary vector (2P_b)."""
    return 6 + 2 * int(n_global_vectors) + 2 * int(n_boundary_vecs)


class _KernelBasis(nn.Module):
    """The kernel basis ``phi(iota)`` ``(P, KERNEL_READ_BASIS)``: one MLP of the pair
    invariants, evaluated once per forward pass and shared by every block's read (the
    per-pair work of a block is then a small linear map, not an MLP)."""

    def __init__(self, n_geo: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_geo, KERNEL_READ_HIDDEN),
            nn.GELU(),
            nn.Linear(KERNEL_READ_HIDDEN, KERNEL_READ_HIDDEN),
            nn.GELU(),
            ### No bias: the basis enters only the reads' logits, through each block's
            ### linear kernel, where a constant basis offset is a per-head constant over
            ### a point's pairs, which cancels in the softmax.
            nn.Linear(KERNEL_READ_HIDDEN, KERNEL_READ_BASIS, bias=False),
        )

    def forward(self, inv: torch.Tensor) -> torch.Tensor:
        return self.net(inv)


def geometry_requires_grad(geometry: dict) -> bool:
    """Whether a gradient is to reach the pair invariants of one case's ``geometry``:
    under gradient, when its points, directions, wall distances, sources or global
    directions require one (its rank-1 inputs, from learned maps, do not decide)."""
    keys = ("query_r", "query_n", "query_wall", "src_r", "src_n", "g_unit")
    return torch.is_grad_enabled() and any(geometry[k] is not None and geometry[k].requires_grad for k in keys)


def _basis_of_pairs(
    basis: _KernelBasis, qi, si, q_off: int, s_off: int, geometry: dict, geo_grad: bool = False
) -> torch.Tensor:
    with torch.enable_grad() if geo_grad else torch.no_grad():
        inv = tree_pair_invariants(qi.long() - q_off, si.long() - s_off, **geometry)
    return basis(inv)


def _chunked_basis_from_plan(
    basis: _KernelBasis,
    plan: PairPlan,
    a: int,
    b: int,
    q_off: int,
    s_off: int,
    geometry: dict,
    weights: tuple | None = None,
) -> torch.Tensor:
    """The kernel basis ``(P, KERNEL_READ_BASIS)`` of one case's pairs, the plan's rows
    ``a:b``, in chunks of ``TREE_READ_PAIR_CHUNK`` pairs. The pair invariants are
    geometry of the inputs: a gradient reaches them only when the geometry requires one
    (:func:`geometry_requires_grad`). Under gradient each chunk's
    invariants and basis MLP are checkpointed: backward rebuilds them from the plan's
    indices (``q_off`` and ``s_off`` the case's first point and source), so the basis is
    all that is kept per pair. With ``weights`` (:func:`fused_basis.mlp_weights`) each
    chunk is one fused kernel of the invariants and the MLP (:mod:`fused_basis`), whose
    geometry takes no gradient."""
    geo_grad = geometry_requires_grad(geometry)
    grad = geo_grad or torch.is_grad_enabled() and any(p.requires_grad for p in basis.parameters())
    ### Without gradient the fused kernel takes every pair in one launch (a pair's basis is its own tile row's,
    ### whatever the chunk): no chunks to concatenate. Under gradient the chunks stay: each one's weight
    ### gradients are summed on their own and rounded.
    chunk = max(b - a, 1) if weights is not None and not grad else TREE_READ_PAIR_CHUNK
    parts = []
    for c in range(a, b, chunk):
        e = min(c + chunk, b)
        if weights is not None:
            parts.append(fused_basis.fused_kernel_basis(weights, plan.qi[c:e], plan.bj[c:e], q_off, s_off, geometry))
            continue
        args = (basis, plan.qi[c:e], plan.bj[c:e], q_off, s_off, geometry, geo_grad)
        ### Kept in the MLP's own dtype (bfloat16 under autocast): the fused read consumes it in every block and its gradient
        ### accumulates over the blocks in that dtype; 2e8 to 7e8 rows at DrivAerML scale.
        parts.append(checkpoint(_basis_of_pairs, *args, use_reentrant=False) if grad else _basis_of_pairs(*args))
    if not parts:
        return geometry["query_r"].new_zeros(0, KERNEL_READ_BASIS)
    return _joined(parts)


@dataclass
class TreeReadWalk:
    """Every case's forest and point lists (:func:`tree_read_walk`), before the kernel
    basis: per case the pairs ``(qi, si, n_node)`` sorted by point, the inputs of their
    invariants (:func:`tree_pair_invariants`) and the sources' log measures.
    :func:`walk_range_pairs` assembles the pairs from it; ``kernel`` is the read's
    implementation (:func:`tree_read_walk`). A walk made with ``walk_points=False`` has
    no lists (``None`` per case). It keeps what :func:`walk_points_range` needs to walk
    a range of points: per case the points in the forest's frame, and the walk's
    implementation. With ``conditions`` given to :func:`tree_read_walk`,
    ``node_condition`` holds per case each node's condition (its tree's, shared by every
    cell of the node), for the per-condition shift of the nodes' log measures in
    :func:`walk_range_pairs`."""

    forests: list[BoundaryForest]
    lists: list[tuple[torch.Tensor, torch.Tensor, int] | None]
    geometry: list[dict]
    src_logm: list[torch.Tensor]
    omega: torch.Tensor  # (B * N,) the cells' measure fractions
    n_cells: int
    n_queries: int
    kernel: str
    points_f: list[torch.Tensor] | None = None
    walk: str = "auto"
    node_condition: list[torch.Tensor] | None = None  # per case (M,), each node's condition


def tree_read_pairs(basis: _KernelBasis, **inputs) -> TreeReadPairs:
    """The tree read's pairs of every point of a batch of cases, with their kernel
    basis: :func:`walk_range_pairs` of :func:`tree_read_walk` (which takes the
    keyword arguments)."""
    return walk_range_pairs(basis, tree_read_walk(**inputs))


def tree_read_walk(
    *,
    query_r: torch.Tensor,
    query_n: torch.Tensor | None = None,
    query_wall: torch.Tensor | None = None,
    boundary_r: torch.Tensor,
    boundary_n: torch.Tensor,
    boundary_vecs: torch.Tensor | None,
    log_frac: torch.Tensor,
    g_unit: torch.Tensor,
    theta: float,
    eps: float,
    normal_tol: torch.Tensor | None = None,
    leaf_size: int = TREE_READ_LEAF_SIZE,
    walk: str = "auto",
    build: str = "auto",
    kernel: str = "reference",
    walk_points: bool = True,
    conditions: torch.Tensor | None = None,
) -> TreeReadWalk:
    r"""Walk the tree read's forests for a batch of cases.

    ``query_r``, ``query_n`` ``(B, Q, d)`` (frame units; the point's built-in
    direction), ``query_wall`` ``(B, Q)`` the point's built-in wall distance (both
    ``None`` for a walk made without its lists: :func:`walk_points_range` takes a
    range's); ``boundary_r``, ``boundary_n`` ``(B, N, d)``, ``boundary_vecs``
    ``(B, N, P_b, d)`` or ``None``; ``log_frac`` ``(B, N)`` the cells' log measure
    fractions; ``g_unit`` ``(B, K, d)``; ``normal_tol`` ``(B, N)`` or ``None`` the
    bound on each cell's normal (:meth:`BoundaryForest.build`); ``leaf_size`` the
    forest's cells per leaf.

    Per case, the forest over the cells (:class:`BoundaryForest`, with the measure
    fractions as the measure) holds the nodes' virtual cells, and each point walks it
    at ``theta`` (:meth:`BoundaryForest.point_pairs`). The forest and the walk run
    without gradient (a numerical device). The pair invariants take a gradient when the
    geometry requires one (:func:`geometry_requires_grad`), and the basis
    (:func:`walk_range_pairs`) carries gradients to the kernel MLP's weights.

    ``walk`` names the walk's implementation (:func:`resolve_walk`) and ``build`` the
    forest's (:meth:`BoundaryForest.build`). ``kernel`` names the read's, recorded for
    :func:`walk_range_pairs`: that builds the basis after the plan, from its indices,
    and rebuilds the invariants in backward (:func:`_chunked_basis_from_plan`). With
    ``"fused"`` the blocks read through :mod:`fused_read`, and each chunk's basis is one
    fused kernel unless :func:`fused_basis.unsupported_reason` gives a reason against
    it. With ``walk_points=False`` no point is walked here: :func:`walk_points_range`
    walks any range of them later. ``conditions`` ``(B, N)`` (or ``None``: one) is each
    cell's declared condition. The forest then builds one set of face trees per
    condition (the ``groups`` of :meth:`BoundaryForest.build`), so each node holds
    cells of one condition, and the walk keeps each node's.
    """
    b, nq, d = query_r.shape
    n = boundary_r.shape[1]
    tiny = torch.finfo(torch.float32).tiny
    has_vecs = boundary_vecs is not None and bool(boundary_vecs.shape[2])
    forests, lists, geometries, src_logms, points_fs, node_conditions = [], [], [], [], [], []
    for i in range(b):
        with torch.no_grad():
            omega = torch.exp(log_frac[i])
            forest = BoundaryForest.build(
                boundary_r[i],
                omega,
                boundary_n[i],
                boundary_vecs[i] if has_vecs else None,
                theta=theta,
                leaf_size=leaf_size,
                eps=eps,
                normal_tol=None if normal_tol is None else normal_tol[i],
                groups=None if conditions is None else conditions[i],
                build=build,
            )
            points_f = forest.frame_points(query_r[i])
            points_fs.append(points_f)
            lists.append(forest.frame_pairs(points_f, walk=walk) if walk_points else None)
            node_measure = forest.measure.to(log_frac.dtype)
            if conditions is not None:
                node_conditions.append(conditions[i][forest.cells[forest.range_start]])
        src_r = torch.cat([boundary_r[i], forest.center.to(boundary_r.dtype)])
        src_n = torch.cat([boundary_n[i], forest.normal.to(boundary_n.dtype)])
        src_vecs = (
            torch.cat([boundary_vecs[i], forest.vecs.to(boundary_vecs.dtype)])
            if has_vecs
            else None
        )
        src_logm = torch.cat([log_frac[i], torch.log(node_measure.clamp_min(tiny))])
        geometry = {
            "query_r": query_r[i],
            "query_n": None if query_n is None else query_n[i],
            "query_wall": None if query_wall is None else query_wall[i],
            "src_r": src_r,
            "src_n": src_n,
            "src_vecs": src_vecs,
            "g_unit": g_unit[i],
            "eps": eps,
        }
        forests.append(forest)
        geometries.append(geometry)
        src_logms.append(src_logm)
    return TreeReadWalk(
        forests=forests,
        lists=lists,
        geometry=geometries,
        src_logm=src_logms,
        omega=torch.exp(log_frac).reshape(-1),
        n_cells=n,
        n_queries=nq,
        kernel=kernel,
        points_f=None if walk_points else points_fs,
        walk=walk,
        node_condition=node_conditions if conditions is not None else None,
    )


def walk_points_range(
    walk: TreeReadWalk,
    start: int,
    stop: int,
    query_n: torch.Tensor | None = None,
    query_wall: torch.Tensor | None = None,
    query_r: torch.Tensor | None = None,
) -> TreeReadWalk:
    """The walk of the points ``start`` to ``stop``, numbered from ``start``, of a walk
    made without its lists (``tree_read_walk(walk_points=False)``), with those points'
    geometry. Each forest is walked for those points alone, from the same coordinates
    in its frame. Each point's list is thus the one the walk of every point gives it,
    in the same order, and :func:`walk_range_pairs` of the result gives the pairs of
    that range. The points' built-in direction, wall distance and positions are the
    walk's rows, or ``query_n`` ``(B, stop - start, d)``, ``query_wall``
    ``(B, stop - start)`` and ``query_r`` ``(B, stop - start, d)`` when given (for a
    walk made without them, or positions in the caller's graph)."""
    with torch.no_grad():
        lists = [f.frame_pairs(p[..., start:stop, :], walk=walk.walk) for f, p in zip(walk.forests, walk.points_f)]
    given = {"query_r": query_r, "query_n": query_n, "query_wall": query_wall}
    geometry = [
        {**g, **{k: g[k][start:stop] if v is None else v[i] for k, v in given.items()}}
        for i, g in enumerate(walk.geometry)
    ]
    return dataclasses.replace(walk, lists=lists, geometry=geometry, n_queries=stop - start, points_f=None)


def walk_pair_counts(walk: TreeReadWalk) -> torch.Tensor:
    """The number of pairs of each point of ``walk`` ``(Q,)``, summed over the cases;
    for a walk made without its lists, counted by each forest's walk without making
    them."""
    if walk.points_f is None:
        return sum(torch.bincount(qi, minlength=walk.n_queries) for qi, _, _ in walk.lists)
    with torch.no_grad():
        return sum(f.frame_pairs(p, walk.walk, count_only=True) for f, p in zip(walk.forests, walk.points_f))


def _joined(parts: list[torch.Tensor]) -> torch.Tensor:
    """``torch.cat(parts)``, without the copy for one part (one case)."""
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def walk_range_pairs(
    basis: _KernelBasis,
    walk: TreeReadWalk,
    log_shift: torch.Tensor | None = None,
    condition_shift: torch.Tensor | None = None,
) -> TreeReadPairs:
    r"""The pairs of every point of each case of ``walk``, with the kernel basis of
    every pair in chunks (a range of points: :func:`walk_points_range`).
    ``log_shift`` ``(B, N)`` and ``condition_shift`` ``(B, C)`` (both or neither; the
    walk made with ``conditions``) are the per-condition quadrature's shifts
    :math:`s_k = -\log A_k + \beta_k`, per cell and per condition, and may carry a
    gradient. A cell's log measure takes its own shift. A node's cells share one
    condition (:func:`tree_read_walk`), and the node takes that condition's shift as
    the sum of a one-hot mask of its condition times the shifts: exactly :math:`s_k`,
    with a product and a sum as its backward, no scatter. The forest's measures never
    take the shifts."""
    n, nq = walk.n_cells, walk.n_queries
    qis, sis, bases = [], [], []
    deferred = []
    src_off = 0
    pair_off = 0
    n_pairs_node = 0
    src_logms = walk.src_logm
    if log_shift is not None:
        src_logms = []
        for i, (cond, logm) in enumerate(zip(walk.node_condition, walk.src_logm)):
            shift = condition_shift[i]
            one_hot = cond[:, None] == torch.arange(shift.shape[0], device=cond.device)
            node_shift = (one_hot.to(shift.dtype) * shift).sum(-1)
            src_logms.append(logm + torch.cat([log_shift[i], node_shift]).to(logm.dtype))
    for i, forest in enumerate(walk.forests):
        qi, si, n_node = walk.lists[i]
        deferred.append((pair_off, pair_off + int(qi.shape[0]), i * nq, src_off, walk.geometry[i]))
        pair_off += int(qi.shape[0])
        ### The first case's lists are the plan's as they are (no shifted copy).
        qis.append(qi + i * nq if i else qi)
        sis.append(si + src_off if i else si)
        src_off += n + forest.n_nodes
        n_pairs_node += n_node
    ### 32-bit pair indices where they fit (the fused read widens them on load): half the plan's memory.
    n_points = len(walk.forests) * nq
    plan_all = PairPlan.build(
        _joined(qis), _joined(sis), n_points, src_off, narrow=max(pair_off, n_points, src_off) < 2**31
    )
    if torch.is_grad_enabled():
        ### Build the by-source half here, not inside the first backward read (the step's memory peak).
        ### Without gradient no read needs it, and it is never built.
        plan_all.by_source
    ### The fused read's basis: one cast of the MLP's weights for every case and chunk, as autocast's weight cache
    ### gives the reference.
    weights = None
    if walk.kernel == "fused":
        reasons = (
            fused_basis.unsupported_reason(basis, item[-1])
            or ("the geometry requires a gradient" if geometry_requires_grad(item[-1]) else None)
            for item in deferred
        )
        reason = next(filter(None, reasons), None)
        weights = fused_basis.mlp_weights(basis) if deferred and reason is None else None
        if reason is not None and reason not in _BASIS_FALLBACKS_LOGGED:
            _BASIS_FALLBACKS_LOGGED.add(reason)
            logger.info("ISLA fused tree read: the reference kernel basis is used (%s)", reason)
    for a, e, q_off, s_off, geometry in deferred:
        bases.append(_chunked_basis_from_plan(basis, plan_all, a, e, q_off, s_off, geometry, weights))
    basis_all = _joined(bases)
    return TreeReadPairs(
        plan=plan_all,
        basis=basis_all,
        src_log_m=_joined(src_logms),
        omega=walk.omega,
        forests=walk.forests,
        n_cells=n,
        n_queries=nq,
        n_pairs_node=n_pairs_node,
        kernel=walk.kernel,
        basis_sink=fused_read.basis_sink(basis_all) if walk.kernel == "fused" else None,
    )


class _TreeKernelRead(nn.Module):
    """One block's read for the passive prediction points, through the forest.

    A softmax over each point's cells and nodes (:class:`TreeReadPairs`) with the
    logits ``K_h(iota_qj) + <q_h, k_jh> / sqrt(c) + log omega_j`` of the module
    docstring's quadrature, where ``K_h`` is a linear combination of the shared kernel
    basis. Logits linear in ``log r`` or ``log(r / |d_q|)`` are power laws, so the
    read can learn any locality; there is no radius, falloff or neighbour count. The
    cells' keys and values are linear maps of their states; a node's are the
    measure-weighted means of its cells' at this depth. The read runs through
    :func:`~physicsnemo.experimental.models.isla._fused_read.fused_kernel_read`, which
    keeps nothing per pair between forward and backward, or the reference
    :func:`~physicsnemo.experimental.models.isla._segment_softmax.segment_softmax_read`.

    ``forward(h_b, h_q, pairs)``, with the boundary and point states flattened over
    the batch, returns the points' residual update. The output projection starts at
    zero, so the read starts as no update. ``compiled`` (``compile_blocks``) runs the
    LayerNorms with the key, value and query maps through :func:`torch.compile`.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        h, w = KERNEL_READ_HEADS, KERNEL_READ_WIDTH
        self.norm_b = nn.LayerNorm(hidden)
        self.norm_q = nn.LayerNorm(hidden)
        self.value = nn.Linear(hidden, w)
        self.key = nn.Linear(hidden, h * KERNEL_READ_KEY, bias=False)
        self.query = nn.Linear(hidden, h * KERNEL_READ_KEY, bias=False)
        ### No bias: a per-head constant cancels in the softmax.
        self.kernel = nn.Linear(KERNEL_READ_BASIS, h, bias=False)
        self.out = nn.Linear(w, hidden, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, h_b, h_q, pairs: TreeReadPairs, compiled: bool = False):
        return self.read(self.sources(h_b, pairs, compiled), h_q, pairs, compiled)

    def sources(self, h_b, pairs: TreeReadPairs | TreeReadWalk, compiled: bool = False):
        """The virtual sources' keys and values at this depth, ``(sources, H KC + W)``,
        from the boundary states ``h_b``."""
        if compiled:
            nb = self.norm_b
            kv = _compiled(_source_states)(
                h_b, nb.weight, nb.bias, nb.eps, self.key.weight, self.value.weight, self.value.bias
            )
        else:
            hb = self.norm_b(h_b)
            v_c, k_c = self.value(hb), self.key(hb)
        ### The virtual sources, case by case: the cells, then the forest's node means
        ### of the keys and values together (one scatter and one sweep per case).
        n = pairs.n_cells
        kv_parts = []
        for i, forest in enumerate(pairs.forests):
            sl = slice(i * n, (i + 1) * n)
            kv_c = kv[sl] if compiled else torch.cat([k_c[sl], v_c[sl]], dim=-1)
            means = (
                fused_read.fused_node_means
                if pairs.kernel == "fused" and fused_read.node_means_supported(kv_c)
                else node_means
            )
            kv_parts += [kv_c, means(kv_c, forest, pairs.omega[sl])]
        return torch.cat(kv_parts)

    def read(self, kv_src, h_q, pairs: TreeReadPairs, compiled: bool = False):
        """The points' residual update from the sources' keys and values (:meth:`sources`)."""
        nh, c, kc = KERNEL_READ_HEADS, KERNEL_READ_WIDTH // KERNEL_READ_HEADS, KERNEL_READ_KEY
        if compiled:
            nq = self.norm_q
            q = _compiled(_query_states)(h_q, nq.weight, nq.bias, nq.eps, self.query.weight).view(-1, nh, kc)
        else:
            q = self.query(self.norm_q(h_q)).view(-1, nh, kc)
        k_src = kv_src[:, : nh * kc].reshape(-1, nh, kc)
        v_src = kv_src[:, nh * kc :].reshape(-1, nh, c)
        ### The pairs' kernel terms at this block: the shared basis through the block's
        ### kernel weights (a matmul; its gradient to the basis accumulates over blocks).
        if pairs.kernel == "fused" and fused_read.supported(pairs.basis, q, k_src, v_src):
            ### The same read with the kernel term computed inside it (_fused_read.py), its
            ### output rounded where the reference's is: to h_q's dtype if that is narrow,
            ### else at the output projection's input under autocast.
            low = (torch.bfloat16, torch.float16)
            dev = h_q.device.type
            if h_q.dtype in low:
                out_dtype = h_q.dtype
            elif torch.is_autocast_enabled(dev) and torch.get_autocast_dtype(dev) in low:
                out_dtype = torch.get_autocast_dtype(dev)
            else:
                out_dtype = torch.float32
            read = fused_read.fused_kernel_read(
                pairs.basis,
                self.kernel.weight.to(pairs.basis.dtype),
                pairs.src_log_m,
                q,
                k_src,
                v_src,
                pairs.plan,
                scale=1.0 / math.sqrt(kc),
                sink=pairs.basis_sink,
                out_dtype=out_dtype,
            )
        else:
            ker = pairs.basis @ self.kernel.weight.t().to(pairs.basis.dtype)
            read = segment_softmax_read(
                ker, pairs.log_m, q, k_src, v_src, pairs.plan, scale=1.0 / math.sqrt(kc)
            )
        return self.out(read.reshape(-1, nh * c).to(h_q.dtype))


def _source_states(h_b, ln_w, ln_b, ln_eps: float, w_key, w_value, b_value):
    """The cells' keys and values ``[k | v]`` from their states (:class:`_TreeKernelRead`)."""
    hb = F.layer_norm(h_b, (h_b.shape[-1],), ln_w, ln_b, ln_eps)
    return torch.cat([F.linear(hb, w_key), F.linear(hb, w_value, b_value)], dim=-1)


def _query_states(h_q, ln_w, ln_b, ln_eps: float, w_query):
    return F.linear(F.layer_norm(h_q, (h_q.shape[-1],), ln_w, ln_b, ln_eps), w_query)
