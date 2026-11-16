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

r"""The fast build of the interior task's boundary forest (:class:`~._tree_read.BoundaryForest`).

The reference build makes one GLOBE
:class:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree` per normal face. Each
tree's Morton topology splits its ranges level by level, with a host sync per level:
about 180 syncs per case. This build makes the same trees for every face at once, by
:meth:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree.from_point_groups` (one Warp
pass and one host sync on CUDA), which gives each face's tree as its own
``from_points`` would in every bit: topology, node order, boxes and measures. One
more host read sizes the levels and the trees' prefix sums.

The node means come from the reference's own expressions: the prefix sums of the
measure and of the weighted points, normals and rank-1 inputs from the same
``cumsum`` calls on the same per-tree inputs, and the node means from the same
elementwise operations on the joined table.

The forest's ``trees`` are the library's trees over slices of the forest's tables, so
the reference walk (:meth:`~._tree_read.BoundaryForest.point_pairs`) runs on them
unchanged. They are made when first asked for (``per_tree``), since the fast walk
reads the table alone.

"""

import logging

import torch
import torch.nn.functional as F

from physicsnemo.mesh.spatial.bvh import _MORTON_SPREAD_STEPS
from physicsnemo.mesh.spatial.cluster_tree import ClusterTree
from physicsnemo.utils._tracing import is_tracing

logger = logging.getLogger(__name__)

#: Rounds of the depths' pointer jumping (:func:`forest_tables`): depths below
#: ``2 ** _DEPTH_ROUNDS``. Each child holds at least a quarter of its parent's cells
#: (the Morton split's balance), so a forest of fewer than 2**31 cells is at most
#: about 75 deep.
_DEPTH_ROUNDS = 7

_LOGGED: set = set()


def resolve_tree_build(build: str, points: torch.Tensor) -> str:
    """The forest build to run: ``"reference"`` or ``"fast"`` when ``build`` forces
    it; under ``"auto"`` the fast build for CUDA points in two or three dimensions,
    else the reference. Logged once per (device type, choice)."""
    if build not in ("auto", "reference", "fast"):
        raise ValueError(f"build must be 'auto', 'reference' or 'fast', got {build!r}")
    if build == "fast" and points.shape[-1] not in _MORTON_SPREAD_STEPS:
        raise ValueError(f"the fast build does not take {points.shape[-1]}-D points")
    if build != "auto":
        return build
    ok = points.is_cuda and points.shape[-1] in (2, 3)
    chosen = "fast" if ok else "reference"
    key = (points.device.type, chosen)
    if key not in _LOGGED and not is_tracing():
        _LOGGED.add(key)
        logger.info(
            "ISLA tree_build=auto resolved to %r on %s (cuda=%s, dim=%d)",
            chosen,
            points.device.type,
            points.is_cuda,
            points.shape[-1],
        )
    return chosen


def forest_tables(
    points: torch.Tensor,
    measure: torch.Tensor,
    normals: torch.Tensor,
    vecs: torch.Tensor | None,
    points_f: torch.Tensor,
    face: torch.Tensor,
    *,
    leaf_size: int,
    n_trees: int | None = None,
) -> dict | None:
    """The fields of :class:`~._tree_read.BoundaryForest` that its trees make, for the
    cells at lab ``points`` ``(N, d)``. ``points_f`` are the cells in their trees'
    coordinates, and ``face`` is each cell's tree: its normal's face
    (:func:`~._tree_read.normal_faces`), or ``2 d g + face`` with groups, of ``n_trees``
    in all (default ``2 d``).

    All fields but the walk's node diameters (``diameter_sq``, which ``per_tree`` takes
    for the trees' ``node_diameter_sq``); the nodes' raw mean normal comes as ``mean_n``
    and the non-empty trees' numbers as ``tree_index`` ``(T,)``. ``None`` where the
    reference build must make them: no cells or a ``leaf_size`` below 1. On CUDA,
    where Warp cannot build the forest, ``from_point_groups`` builds it by torch."""
    n, d = points_f.shape
    if n == 0 or leaf_size < 1:
        return None
    device = points.device
    n_face = 2 * d if n_trees is None else int(n_trees)
    ### Warp's build: on CUDA where it can build the forest (else torch's), and on CPU,
    ### where only a forced "fast" build comes, always.
    impl = None if points_f.is_cuda else "warp"
    forest, trees = ClusterTree.from_point_groups(
        points_f, face, n_groups=n_face, leaf_size=leaf_size, areas=measure, implementation=impl
    )
    cells, left, right = forest.sorted_source_order, forest.node_left_child, forest.node_right_child
    range_start, range_count = forest.node_range_start, forest.node_range_count
    m_total = forest.n_nodes
    is_leaf = left < 0

    ### Each node's depth by pointer jumping: a node counts its steps to the ancestor
    ### ``up``, then adds that ancestor's own count and jumps to its ancestor.
    ids = torch.arange(m_total, device=device)
    up = torch.full((m_total + 1,), -1, dtype=torch.long, device=device)
    for child in (left, right):
        up.scatter_(0, torch.where(is_leaf, m_total, child), ids)
    up = up[:m_total]
    depth = (up >= 0).long()
    for _ in range(_DEPTH_ROUNDS):
        at = up.clamp_min(0)
        depth = depth + torch.where(up >= 0, depth[at], 0)
        up = torch.where(up >= 0, up[at], -1)
    ### One readback: the cells per tree and the internal nodes per depth.
    cells_per = torch.zeros(n_face, dtype=torch.long, device=device).index_add_(0, face, torch.ones_like(face))
    bins = torch.zeros(2**_DEPTH_ROUNDS, dtype=torch.long, device=device).index_add_(0, depth, (~is_leaf).long())
    stats = torch.cat([cells_per, bins]).tolist()
    cells_per_h, internal_per_depth = stats[:n_face], stats[n_face:]
    n_internal = sum(internal_per_depth)

    tree = face[cells[range_start]]
    leaf_ids = torch.nonzero_static(is_leaf, size=m_total - n_internal)[:, 0]
    leaf_ids = leaf_ids[torch.argsort(range_start[leaf_ids])]
    position_leaf = torch.repeat_interleave(
        torch.arange(leaf_ids.shape[0], device=device), range_count[leaf_ids], output_size=n
    )
    ### Internal nodes by depth, deepest first, each depth in ascending id.
    internal = torch.nonzero_static(~is_leaf, size=n_internal)[:, 0]
    internal = internal[torch.argsort(depth[internal], stable=True, descending=True)]
    sizes = [c for c in reversed(internal_per_depth) if c]
    levels = list(zip(*(torch.split(x, sizes) for x in (internal, left[internal], right[internal]))))

    ### Range sums: per tree, the reference's prefix sums of the sorted measure and of
    ### the weighted points, normals and rank-1 inputs (same inputs, same scans); the
    ### node sums and means on the joined table.
    sorted_w = measure[cells]
    parts = [points[cells], normals[cells]]
    if vecs is not None:
        parts.append(vecs[cells].flatten(1))
    weighted = (torch.cat(parts, dim=1) * sorted_w.unsqueeze(-1)).double()
    trees_h = [k for k in range(n_face) if cells_per_h[k]]
    c_bounds, a = [0], 0
    for k in trees_h:
        a += cells_per_h[k]
        c_bounds.append(a)
    ### Each tree's scans written in place after its leading zero.
    sorted_w64 = sorted_w.double()
    area_prefix = sorted_w64.new_zeros(n + len(trees_h))
    prefix = weighted.new_zeros(weighted.shape[1], n + len(trees_h))
    for t in range(len(trees_h)):
        a, b = c_bounds[t], c_bounds[t + 1]
        torch.cumsum(sorted_w64[a:b], dim=0, out=area_prefix[a + t + 1 : b + t + 1])
        torch.cumsum(weighted[a:b].t(), dim=1, out=prefix[:, a + t + 1 : b + t + 1])
    ordinal = torch.cumsum(cells_per > 0, 0) - 1
    starts = range_start + ordinal[tree]
    ends = starts + range_count
    total_64 = area_prefix[ends] - area_prefix[starts]
    nonzero_total = total_64 != 0
    safe_64 = total_64.where(nonzero_total, 1)
    means = ((prefix[:, ends] - prefix[:, starts]) / safe_64).where(nonzero_total, 0).t()
    data_dtype = normals.dtype
    center = means[:, :d].to(points.dtype).contiguous()
    mean_n = means[:, d : 2 * d].to(data_dtype).contiguous()
    node_vecs = None
    if vecs is not None:
        node_vecs = means[:, 2 * d :].to(data_dtype).reshape(m_total, *vecs.shape[1:]).contiguous()

    def per_tree(diameter_sq: torch.Tensor) -> tuple[list[ClusterTree], list[torch.Tensor]]:
        """For the reference walk: the non-empty trees, each with its slice of the walk's
        node diameters ``(M,)``, and the cells each holds."""
        by_face = torch.argsort(face, stable=True)
        out, m = [], 0
        for k in trees_h:
            t = trees[k]
            t.node_diameter_sq = diameter_sq[m : m + t.n_nodes]
            m += t.n_nodes
            out.append(t)
        return out, [by_face[a:b] for a, b in zip(c_bounds, c_bounds[1:])]

    return {
        "per_tree": per_tree,
        "node_offset": F.pad(forest.root_ids, (0, 1), value=m_total),
        "cells": cells,
        "range_start": range_start,
        "range_count": range_count,
        "left": left,
        "right": right,
        "leaf_ids": leaf_ids,
        "position_leaf": position_leaf,
        "levels": levels,
        "aabb_min": forest.node_aabb_min,
        "aabb_max": forest.node_aabb_max,
        "center": center,
        "mean_n": mean_n,
        "vecs": node_vecs,
        "measure": forest.node_total_area,
        "tree_index": torch.nonzero_static(cells_per, size=len(trees_h))[:, 0],
    }
