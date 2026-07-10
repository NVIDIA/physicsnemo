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

"""Quality-greedy bistellar flips: dimension-generic, pure torch.

A *cluster* is ``d+2`` vertices currently triangulated by 2 mesh cells
(sharing a facet) or 3 mesh cells (sharing a ridge, ``d >= 3``). By Radon's
theorem the cluster's points admit exactly two triangulations, indexed by
the sign classes ``(A, B)`` of their affine dependence; the current
configuration is one of them, and a flip switches to the other iff that
strictly raises the cluster's minimum volume-length quality.

Why this formulation:

- *No Delaunay predicate, no exact arithmetic.* Greed on the quality metric
  replaces the incircle test; mesh minimum quality is monotone
  non-decreasing, so termination is a counting argument (3D Lawson flipping,
  by contrast, can cycle or stall).
- *Region-preserving by construction.* Both triangulations cover the
  cluster's convex hull; a volume-conservation check additionally rejects
  reflex configurations, so flips are safe at boundaries.
- *Parallel.* Winners form a vertex-disjoint independent set (random
  priorities + one ``scatter_reduce_``); vertex-disjointness is strictly
  stronger than cell-disjointness and prevents two clusters minting the
  same new cell.
- *Local.* Candidates are only harvested from the sub-mesh adjacent to
  cells below ``q_focus`` -- flips exist to fix bad neighborhoods, and the
  census over that sub-mesh (not the whole mesh) is what makes a flip pass
  cheap at scale. Cluster pairs straddling the sub-mesh frontier are
  skipped conservatively; later passes see them.
"""

import torch

from physicsnemo.mesh.generate._simplex_ops import (
    _sub_simplices,
    _unique_rows,
    signed_volumes,
    volume_length_quality,
)

__all__ = ["flip_pass", "flip_until_done"]


def _clusters_from_shared(cells, cell_ids, k_share, sub_size):
    """Groups (K, k_share) of GLOBAL cell ids sharing one sub-simplex."""
    subs, owner_local = _sub_simplices(cells[cell_ids], sub_size)
    _, inverse, counts = _unique_rows(subs)
    sel = counts[inverse] == k_share
    if not sel.any():
        return torch.empty(0, k_share, dtype=torch.int64, device=cells.device)
    order = torch.argsort(inverse[sel], stable=True)
    return cell_ids[owner_local[sel][order].reshape(-1, k_share)]


def _radon_signs(points, cluster_verts):
    """Affine-dependence sign pattern per cluster; ok = general position."""
    p = points[cluster_verts]
    k, dp2, d = p.shape
    a = torch.cat(
        [torch.ones(k, 1, dp2, dtype=p.dtype, device=p.device), p.transpose(1, 2)],
        dim=1,
    )
    _, s, vh = torch.linalg.svd(a)
    lam = vh[:, -1, :]
    lam = lam / lam.abs().max(dim=1, keepdim=True).values.clamp_min(1e-30)
    signs = torch.sign(torch.where(lam.abs() < 1e-9, torch.zeros_like(lam), lam))
    ok = (signs != 0).all(dim=1) & (s[:, -2] > 1e-12)
    return signs, ok


def _candidates(points, cells, q_all, cell_ids, k_share, sub_size, h):
    """Improving flips of one kind within the focused sub-mesh, or ``None``."""
    d = points.shape[1]
    groups = _clusters_from_shared(cells, cell_ids, k_share, sub_size)
    if groups.numel() == 0:
        return None
    gverts, _ = torch.sort(cells[groups].reshape(groups.shape[0], -1), dim=1)
    first = torch.ones_like(gverts, dtype=torch.bool)
    first[:, 1:] = gverts[:, 1:] != gverts[:, :-1]
    sel = first.sum(dim=1) == (d + 2)
    if not sel.any():
        return None
    groups, gverts, first = groups[sel], gverts[sel], first[sel]
    cluster_verts = gverts[first].reshape(-1, d + 2)

    signs, ok = _radon_signs(points, cluster_verts)
    # The target class is the sub-simplex shared by ALL current cells:
    # removing each of its vertices from the cluster generates the new
    # cells. Identify it by membership; the Radon sign pattern is only a
    # validity filter (the shared vertices must form exactly one sign
    # class). Identifying the target by sign-class SIZE is ambiguous
    # whenever the two classes tie at k_share (2D 2-2 flips, 4D 3-3
    # flips): the SVD null vector's sign is arbitrary, and resolving the
    # tie to "negative" proposed the identity retriangulation -- silently
    # rejected as zero-gain -- for about half of all improving 2D flips.
    k_new = d + 2 - k_share
    tgt = (
        (cluster_verts[:, :, None, None] == cells[groups][:, None, :, :])
        .any(dim=3)
        .all(dim=2)
    )
    pos = signs > 0
    n_shared = tgt.sum(dim=1)
    n_pos_shared = (pos & tgt).sum(dim=1)
    n_pos = pos.sum(dim=1)
    one_class = ((n_pos_shared == n_shared) & (n_pos == n_shared)) | (
        (n_pos_shared == 0) & (n_pos == (d + 2) - n_shared)
    )
    ok &= one_class & (n_shared == k_new)
    if not ok.any():
        return None
    groups, cluster_verts, tgt = groups[ok], cluster_verts[ok], tgt[ok]

    n_clusters = cluster_verts.shape[0]
    tcol = torch.nonzero(tgt)[:, 1].reshape(n_clusters, k_new)
    arange = torch.arange(d + 2, device=points.device)
    keep = arange[None, None, :] != tcol[:, :, None]
    newc = (
        cluster_verts[:, None, :]
        .expand(-1, k_new, -1)[keep]
        .reshape(n_clusters, k_new, d + 1)
    )

    flat_new = newc.reshape(-1, d + 1)
    vol_new = signed_volumes(points, flat_new)
    swap = vol_new < 0
    if swap.any():
        flat_new = flat_new.clone()
        flat_new[swap] = flat_new[swap][:, [1, 0] + list(range(2, d + 1))]
    q_new = volume_length_quality(points, flat_new).reshape(n_clusters, k_new)
    newc = flat_new.reshape(n_clusters, k_new, d + 1)

    cur_vol = signed_volumes(points, cells[groups].reshape(-1, d + 1)).reshape(
        n_clusters, k_share
    )
    new_vol_sum = vol_new.abs().reshape(n_clusters, k_new).sum(dim=1)
    # Relative term is dtype-aware: a fixed 1e-9 sits below float32's
    # machine epsilon and silently rejected ~11% of legitimate flips.
    rel_eps = max(1e-9, 64.0 * torch.finfo(points.dtype).eps)
    conserved = (new_vol_sum - cur_vol.sum(dim=1)).abs() <= (
        rel_eps * new_vol_sum.clamp_min(1e-30) + 1e-14 * h**d
    )
    gain = q_new.min(dim=1).values - q_all[groups].min(dim=1).values
    vol_ok = (vol_new.abs().reshape(n_clusters, k_new) > 1e-12 * h**d).all(dim=1)
    good = (gain > 1e-6) & vol_ok & conserved
    if not good.any():
        return None
    return groups[good], cluster_verts[good], newc[good], gain[good]


def flip_pass(points, cells, h, generator=None, q_focus: float = 0.5):
    """One parallel flip pass. Returns ``(cells, n_flipped)``."""
    d = points.shape[1]
    device = cells.device
    q_all = volume_length_quality(points, cells)

    bad = q_all < q_focus
    if not bad.any():
        return cells, 0
    # O(n_cells) boolean-mask gather (torch.isin sorts per call and was
    # measured 2x slower end-to-end at 7.5e5 cells).
    seed_mask = torch.zeros(points.shape[0], dtype=torch.bool, device=device)
    seed_mask[cells[bad].reshape(-1)] = True
    cell_ids = torch.nonzero(seed_mask[cells].any(dim=1)).reshape(-1)

    specs = [(2, d)] + ([(3, d - 1)] if d >= 3 else [])
    cands = []
    for k_share, sub_size in specs:
        c = _candidates(points, cells, q_all, cell_ids, k_share, sub_size, h)
        if c is not None:
            groups, cverts, newc, gain = c
            cands.append(
                (
                    torch.nn.functional.pad(groups, (0, 3 - k_share), value=-1),
                    cverts,
                    torch.nn.functional.pad(
                        newc, (0, 0, 0, (d + 1) - newc.shape[1]), value=-1
                    ),
                    gain,
                )
            )
    if not cands:
        return cells, 0

    groups = torch.cat([c[0] for c in cands])
    cverts = torch.cat([c[1] for c in cands])
    newc = torch.cat([c[2] for c in cands])
    gain = torch.cat([c[3] for c in cands])

    n_cand = groups.shape[0]
    # float64 priorities with a deterministic index tie-break: winner
    # selection is exact float equality per vertex, and quantized ties
    # would let two vertex-sharing candidates both win.
    prio = (
        torch.rand(n_cand, generator=generator, dtype=torch.float64).to(device)
        + gain.double().clamp(0.0, 1.0)
        + torch.arange(n_cand, device=device, dtype=torch.float64) * 1e-18
    )
    best = torch.zeros(points.shape[0], dtype=prio.dtype, device=device)
    best.scatter_reduce_(
        0,
        cverts.reshape(-1),
        prio[:, None].expand_as(cverts).reshape(-1),
        reduce="amax",
        include_self=True,
    )
    wins = (best[cverts] == prio[:, None]).all(dim=1)
    if not wins.any():
        return cells, 0

    wg = groups[wins]
    drop = torch.zeros(cells.shape[0], dtype=torch.bool, device=device)
    drop[wg[wg >= 0]] = True
    wn = newc[wins]
    wn = wn[wn[:, :, 0] >= 0]
    return torch.cat([cells[~drop], wn], dim=0), int(wins.sum())


def flip_until_done(
    points,
    cells,
    h,
    max_passes: int = 30,
    generator=None,
    q_focus: float = 0.5,
    min_flips_frac: float = 1e-3,
):
    """Run flip passes until quiescent or ``max_passes``. Returns (cells, n).

    A pass yielding fewer than ``max(1, min_flips_frac * n_cells)`` flips
    ends the invocation: each pass costs a full quality scan plus a
    neighborhood census, which sub-0.1%-of-cells improvements don't repay
    (they are picked up by the next scheduled invocation instead).
    """
    total = 0
    for _ in range(max_passes):
        cells, n = flip_pass(points, cells, h, generator=generator, q_focus=q_focus)
        total += n
        if n < max(1, int(min_flips_frac * cells.shape[0])):
            break
    return cells, total
