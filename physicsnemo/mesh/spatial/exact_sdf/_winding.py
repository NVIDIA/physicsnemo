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

"""The generalized winding number that signs the distance.

In three dimensions Barill et al.'s (2018) far-field expansion on the nodes of the
cell tree, packed one record per node; in two, a brute force over the segments.
"""

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.mesh.spatial.bvh import _compute_morton_codes
from physicsnemo.mesh.spatial.exact_sdf._cell_tree import _bottom_up

#: The far-field criterion of the winding number that signs the distance
#: (:func:`_packed_winding_sum`; Barill et al. 2018's ``beta``): a BVH node's
#: expansion stands in for its cells when the query lies farther than this many node
#: radii from its centre. A performance setting: the error falls to zero as it grows. Measured at 2 against
#: the exact sum (float64, 3,000 queries each): at most 3.1e-2 (median 1e-3) on an
#: icosphere of 5,120 triangles, 1.9e-2 on a torus, 1.2e-2 on a cube with a face
#: removed. Off a closed or double-sided surface the exact ``w`` is an integer, half
#: away from the threshold, so the sign is the surface's; it can depend on the tree
#: only where the exact ``|w - sigma / 2|`` is below that error (behind an isolated
#: sheet, across a hole).
WINDING_NUMBER_BETA = 2.0


@wp.func
def _solid_angle(x: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d) -> wp.float64:
    """The signed solid angle of the triangle ``abc`` seen from ``x`` (Van Oosterom
    and Strackee 1983), positive when ``x`` lies on the side opposite to ``(b - a) x
    (c - a)``."""
    p = a - x
    q = b - x
    r = c - x
    lp = wp.length(p)
    lq = wp.length(q)
    lr = wp.length(r)
    den = lp * lq * lr + wp.dot(p, q) * lr + wp.dot(p, r) * lq + wp.dot(q, r) * lp
    return wp.float64(2.0) * wp.atan2(wp.dot(p, wp.cross(q, r)), den)


@wp.kernel
def _node_moments_kernel(
    inner: wp.array(dtype=wp.int64),
    left: wp.array(dtype=wp.int64),
    right: wp.array(dtype=wp.int64),
    weight: wp.array(dtype=wp.float64),
    centre: wp.array(dtype=wp.vec3d),
    dipole: wp.array(dtype=wp.vec3d),
    spread: wp.array(dtype=wp.mat33d),
    radius: wp.array(dtype=wp.float64),
):
    """The far-field moments of each node ``inner[tid]`` from its two children's
    (:func:`_packed_winding_sum`)."""
    i = inner[wp.tid()]
    a = left[i]
    b = right[i]
    s = weight[a] + weight[b]
    p = wp.float64(0.5) * (centre[a] + centre[b])
    if s > wp.float64(0.0):
        p = (weight[a] * centre[a] + weight[b] * centre[b]) / s
    weight[i] = s
    centre[i] = p
    dipole[i] = dipole[a] + dipole[b]
    ea = centre[a] - p
    eb = centre[b] - p
    spread[i] = (
        spread[a] + wp.outer(dipole[a], ea) + spread[b] + wp.outer(dipole[b], eb)
    )
    radius[i] = wp.max(wp.length(ea) + radius[a], wp.length(eb) + radius[b])


def _winding_moments(
    simplex: torch.Tensor, parity: torch.Tensor, area: torch.Tensor, tree
):
    """The far-field data of :func:`_packed_winding_sum` per node of ``tree``, as
    the Warp arrays it reads: the centre, the squared far-field radius,
    the dipole and its spread, then the cells' corners and parities."""
    left, right, escape, levels, leaf_cell, _, _ = tree
    n, leaf = left.shape[0], leaf_cell >= 0
    cell = leaf_cell[leaf]
    centroid = simplex.mean(1)
    weight, centre, dipole = (
        simplex.new_zeros(n),
        simplex.new_zeros(n, 3),
        simplex.new_zeros(n, 3),
    )
    spread, radius = simplex.new_zeros(n, 3, 3), simplex.new_zeros(n)
    weight[leaf], centre[leaf], dipole[leaf] = (
        area[cell].norm(dim=-1),
        centroid[cell],
        area[cell],
    )
    ### (one vertex at a time: the same norms and maximum, without an (F, 3, 3)
    ### temporary)
    radius[leaf] = torch.stack(
        [
            (simplex[cell, k] - centroid[cell]).norm(dim=-1)
            for k in range(simplex.shape[1])
        ],
        -1,
    ).amax(-1)
    _bottom_up(
        _node_moments_kernel,
        levels,
        [
            *(wp.from_torch(c.contiguous(), dtype=wp.int64) for c in (left, right)),
            wp.from_torch(weight, dtype=wp.float64),
            wp.from_torch(centre, dtype=wp.vec3d),
            wp.from_torch(dipole, dtype=wp.vec3d),
            wp.from_torch(spread, dtype=wp.mat33d),
            wp.from_torch(radius, dtype=wp.float64),
        ],
        simplex,
    )
    ### (each float64 moment freed as it is rounded)
    del weight, centroid
    far2 = (WINDING_NUMBER_BETA * radius).square().float().contiguous()
    del radius
    dipole = dipole.float().contiguous()
    spread = spread.float().contiguous()
    return [
        wp.from_torch(centre.contiguous(), dtype=wp.vec3d),
        wp.from_torch(far2, dtype=wp.float32),
        wp.from_torch(dipole, dtype=wp.vec3f),
        wp.from_torch(spread, dtype=wp.mat33f),
        wp.from_torch(simplex.reshape(-1, 3).contiguous(), dtype=wp.vec3d),
        wp.from_torch(parity.contiguous(), dtype=wp.float64),
    ]


@wp.struct
class _WindingNode:
    """What :func:`_packed_winding_kernel` reads of a node, in one record: the centre, the
    squared far-field radius, the dipole and its spread, the node after its subtree
    and its cell (-1 inside)."""

    centre: wp.vec3d
    far2: wp.float32
    dipole: wp.vec3f
    spread: wp.mat33f
    escape: wp.int32
    cell: wp.int32


@wp.kernel
def _pack_winding_kernel(
    centre: wp.array(dtype=wp.vec3d),
    far2: wp.array(dtype=wp.float32),
    dipole: wp.array(dtype=wp.vec3f),
    spread: wp.array(dtype=wp.mat33f),
    escape: wp.array(dtype=wp.int64),
    leaf_cell: wp.array(dtype=wp.int64),
    nodes: wp.array(dtype=_WindingNode),
):
    i = wp.tid()
    node = _WindingNode()
    node.centre = centre[i]
    node.far2 = far2[i]
    node.dipole = dipole[i]
    node.spread = spread[i]
    node.escape = wp.int32(escape[i])
    node.cell = wp.int32(leaf_cell[i])
    nodes[i] = node


@wp.kernel
def _packed_winding_kernel(
    nodes: wp.array(dtype=_WindingNode),
    corners: wp.array(dtype=wp.vec3d),
    parity: wp.array(dtype=wp.float64),
    query: wp.array(dtype=wp.vec3d),
    rows: wp.array(dtype=wp.int64),
    out: wp.array(dtype=wp.float64),
):
    """The sum of the signed solid angles of the cells seen from each query
    ``rows[tid]``: a stackless depth-first traversal of the tree (in depth-first
    preorder, :func:`_preorder_cell_tree`, so an inner node's left child is the next
    node; its nodes packed one record each) that takes a node's expansion (in float32,
    from the float64 offset of its centre) where the query is farther than ``far2``
    (squared) from it, and the exact solid angle (float64, the canonical vertices
    ``corners`` times the cell's ``parity``) at the near leaves."""
    q = rows[wp.tid()]
    x = query[q]
    total = wp.float64(0.0)
    node = wp.int32(0)
    while node >= 0:
        nd = nodes[node]
        e = nd.centre - x
        r = wp.vec3f(wp.float32(e[0]), wp.float32(e[1]), wp.float32(e[2]))
        r2 = wp.dot(r, r)
        if r2 > nd.far2:
            inv2 = wp.float32(1.0) / r2
            s = nd.spread
            near = wp.dot(nd.dipole, r) + s[0, 0] + s[1, 1] + s[2, 2]
            total += wp.float64(
                (near - wp.float32(3.0) * wp.dot(r, wp.mul(s, r)) * inv2)
                * inv2
                * wp.sqrt(inv2)
            )
            node = nd.escape
        else:
            c = nd.cell
            if c >= wp.int32(0):
                i = wp.int64(c) * wp.int64(3)
                total += parity[c] * _solid_angle(
                    x, corners[i], corners[i + wp.int64(1)], corners[i + wp.int64(2)]
                )
                node = nd.escape
            else:
                node = node + wp.int32(1)
    out[q] = total


def _packed_winding_sum(
    query: torch.Tensor,
    simplex: torch.Tensor,
    parity: torch.Tensor,
    area: torch.Tensor,
    tree,
):
    """The generalized winding number times ``4 pi`` of the oriented triangles
    ``simplex`` ``(F, 3, 3)`` (canonical, orientation ``parity`` ``(F,)``, area vectors
    ``area`` ``(F, 3)``) at each query ``(Q, 3)``, all float64, by Barill, Dickson,
    Schmidt, Levin and Jacobson (2018) on a tree from :func:`_preorder_cell_tree`
    whose leaves hold the rows of ``simplex``: every node carries the area-weighted
    centre ``p`` of its cells, a radius about it, their dipole moment ``m = sum a_f
    n_f`` and its spread ``N = sum a_f n_f (c_f - p)^T`` (one bottom-up pass over the
    levels); a node farther than ``WINDING_NUMBER_BETA`` radii takes the expansion of
    the solid angle about ``p`` to first order, ``m . r / |r|^3 + tr(N) / |r|^3 - 3 r .
    N r / |r|^5`` with ``r = p - x``, and the near leaves the exact solid angle. The
    moments, then one record per node and one traversal, the queries in Morton
    order."""
    left, _, escape, _, leaf_cell, _, _ = tree
    centre, far2, dipole, spread, corners, par = _winding_moments(
        simplex, parity, area, tree
    )
    out = torch.empty(query.shape[0], dtype=torch.float64, device=query.device)
    device, stream = FunctionSpec.warp_launch_context(query)
    with FunctionSpec.warp_stream_scope(stream):
        wp.init()
        size = wp.types.type_size_in_bytes(_WindingNode)
        buffer = torch.empty(
            left.shape[0] * size, dtype=torch.uint8, device=query.device
        )
        nodes = wp.array(
            ptr=buffer.data_ptr(),
            dtype=_WindingNode,
            shape=(left.shape[0],),
            device=device,
        )
        wp.launch(
            _pack_winding_kernel,
            dim=left.shape[0],
            inputs=[
                centre,
                far2,
                dipole,
                spread,
                wp.from_torch(escape, dtype=wp.int64),
                wp.from_torch(leaf_cell, dtype=wp.int64),
                nodes,
            ],
            device=device,
            stream=stream,
        )
        wp.launch(
            _packed_winding_kernel,
            dim=query.shape[0],
            inputs=[
                nodes,
                corners,
                par,
                wp.from_torch(query.contiguous(), dtype=wp.vec3d),
                wp.from_torch(_compute_morton_codes(query).argsort(), dtype=wp.int64),
                wp.from_torch(out, dtype=wp.float64),
            ],
            device=device,
            stream=stream,
        )
    return out


@wp.kernel
def _segment_winding_kernel(
    ends: wp.array(dtype=wp.vec2d),
    parity: wp.array(dtype=wp.float64),
    query: wp.array(dtype=wp.vec2d),
    out: wp.array(dtype=wp.float64),
):
    """The sum of the signed angles the segments (``ends`` two by two, times their
    ``parity``) subtend at each query, in the cells' order."""
    tid = wp.tid()
    x = query[tid]
    total = wp.float64(0.0)
    for c in range(parity.shape[0]):
        p = ends[2 * c] - x
        q = ends[2 * c + 1] - x
        total -= parity[c] * wp.atan2(
            p[0] * q[1] - p[1] * q[0], p[0] * q[0] + p[1] * q[1]
        )
    out[tid] = total


def _segment_winding_numbers(
    query: torch.Tensor, simplex: torch.Tensor, parity: torch.Tensor
):
    """The winding number of the oriented segments ``simplex`` ``(F, 2, 2)``
    (canonical, orientation ``parity`` ``(F,)``) at each query ``(Q, 2)``, float64:
    the sum of the signed angles they subtend, over ``2 pi``, positive for a query on
    the side opposite to the normals; a brute force over the segments, one Warp
    thread per query."""
    out = torch.empty(query.shape[0], dtype=torch.float64, device=query.device)
    device, stream = FunctionSpec.warp_launch_context(query)
    with FunctionSpec.warp_stream_scope(stream):
        wp.init()
        wp.launch(
            _segment_winding_kernel,
            dim=query.shape[0],
            inputs=[
                wp.from_torch(simplex.reshape(-1, 2).contiguous(), dtype=wp.vec2d),
                wp.from_torch(parity.contiguous(), dtype=wp.float64),
                wp.from_torch(query.contiguous(), dtype=wp.vec2d),
                wp.from_torch(out, dtype=wp.float64),
            ],
            device=device,
            stream=stream,
        )
    return out / (2 * torch.pi)


def _winding_sign(
    query: torch.Tensor,
    simplex: torch.Tensor,
    parity: torch.Tensor,
    sigma: torch.Tensor,
    w=None,
):
    """The sign of the distance at each query ``(Q, d)``, float64: ``sign(sigma / 2 -
    w)``, with ``w`` the generalized winding number of the oriented cells
    ``simplex`` (canonical, orientation ``parity``; Jacobson, Kavan and
    Sorkine-Hornung 2013) and ``sigma`` the sign of the volume they enclose,
    ``V = sum_f a_f n_f . (c_f - g) / d`` about their area-weighted centroid ``g``
    (+1 at ``V = 0`` and at ``w = sigma / 2``). For a closed surface ``w`` is
    ``(sigma - 1) / 2`` on the side the normals point to and ``(sigma + 1) / 2`` on
    the other (0 and 1 for outward normals, -1 and 0 for normals into the region they
    bound), so ``sigma / 2 - w`` is +1/2 or -1/2: positive on the side the normals
    point to; coincident cells of opposite orientation (a
    zero-thickness fold) cancel in ``w``, so both of its sides take the sign of the
    region around it. ``sigma`` from :func:`_oriented_areas`; ``w`` times ``4 pi`` in
    three dimensions (:func:`_packed_winding_sum`), and in two ``None``: the
    segments' (:func:`_segment_winding_numbers`)."""
    if w is not None:
        w = w / (4 * torch.pi)
    else:
        w = _segment_winding_numbers(query, simplex, parity)
    return torch.where(0.5 * sigma - w >= 0, 1.0, -1.0).to(w)
