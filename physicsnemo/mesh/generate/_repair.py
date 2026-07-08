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

"""Topological repairs for the implicit-domain mesher.

Two defect classes arise that no vertex smoother or region-preserving flip
can fix, so they get dedicated (dimension-generic) repairs:

- **Pinches** (``split_pinched_vertices``): two mesh regions touching at a
  single vertex, produced when a thin concave feature squeezes the eroded
  lattice. The vertex is duplicated per connected component of its
  cell-star; manifoldness is restored with zero geometric change.
- **Boundary pancakes** (``peel_boundary_slivers``): near-flat cells whose
  vertices ALL lie on the boundary -- pinned to the zero set, they cannot
  be inflated. Deleting them costs no covered volume and re-exposes the
  facets beneath; each peel round reverts if it would break the
  closed-manifold boundary invariant.
"""

import torch

from physicsnemo.mesh.generate._simplex_ops import (
    _sub_simplices,
    _unique_rows,
    boundary_is_closed_manifold,
    boundary_vertex_mask,
    compact_mesh,
    facet_census,
    orient_positive,
    signed_volumes,
    volume_length_quality,
)

__all__ = [
    "peel_boundary_slivers",
    "pin_feature_points",
    "split_pinched_vertices",
]


def split_pinched_vertices(points, cells, max_rounds: int = 32):
    """Duplicate vertices whose cell-star is facet-disconnected.

    Star components are found by min-label propagation over (vertex, cell)
    incidence nodes, with edges given by interior facets containing the
    vertex. Returns ``(points, cells, n_split)``.
    """
    m, dp1 = cells.shape
    device = cells.device

    facets, owner = _sub_simplices(cells, dp1 - 1)
    _, inverse, counts = _unique_rows(facets)
    sel = counts[inverse] == 2
    order = torch.argsort(inverse[sel], stable=True)
    pairs = owner[sel][order].reshape(-1, 2)
    fverts = facets[sel][order].reshape(-1, 2, dp1 - 1)[:, 0, :]

    node_v = cells.reshape(-1)
    node_c = torch.arange(m, device=device).repeat_interleave(dp1)
    key = node_v * m + node_c
    key_sorted, key_order = torch.sort(key)
    labels = torch.arange(key.shape[0], device=device)

    def node_id(v, c):
        return torch.searchsorted(key_sorted, v * m + c)

    ev = fverts.reshape(-1)
    na = node_id(ev, pairs[:, 0].repeat_interleave(dp1 - 1))
    nb = node_id(ev, pairs[:, 1].repeat_interleave(dp1 - 1))
    for _ in range(max_rounds):
        la, lb = labels[na], labels[nb]
        new = torch.minimum(la, lb)
        changed = bool((la != new).any() or (lb != new).any())
        labels.scatter_reduce_(0, na, new, reduce="amin")
        labels.scatter_reduce_(0, nb, new, reduce="amin")
        if not changed:
            break

    v_sorted = node_v[key_order]
    vc = v_sorted * (key.shape[0] + 1) + labels
    uv = torch.unique(vc) // (key.shape[0] + 1)
    n_comp = torch.bincount(uv, minlength=points.shape[0])
    pinched = torch.nonzero(n_comp > 1).reshape(-1)
    if pinched.numel() == 0:
        return points, cells, 0

    new_points = [points]
    cells = cells.clone()
    next_id = points.shape[0]
    n_split = 0
    for v in pinched.tolist():
        mask = v_sorted == v
        comps = labels[mask]
        cells_of_v = node_c[key_order][mask]
        for extra in torch.unique(comps)[1:]:
            sel_cells = cells_of_v[comps == extra]
            sub = cells[sel_cells]
            sub[sub == v] = next_id
            cells[sel_cells] = sub
            new_points.append(points[v : v + 1])
            next_id += 1
            n_split += 1
    return torch.cat(new_points, dim=0), cells, n_split


def peel_boundary_slivers(
    points,
    cells,
    phi,
    h,
    q_thresh: float = 0.05,
    rounds: int = 3,
    protect_vertices=None,
):
    """Delete near-flat all-boundary-vertex cells lying on the surface.

    Cells containing a protected vertex (e.g. a pinned feature point) are
    never peeled. Returns ``(points, cells, n_peeled)``.
    """
    n_peeled = 0
    for _ in range(rounds):
        q = volume_length_quality(points, cells)
        bnd = boundary_vertex_mask(points, cells)
        cen_phi = phi(points[cells].mean(dim=1)).abs()
        pancake = (q.abs() < q_thresh) & bnd[cells].all(dim=1) & (cen_phi < 0.3 * h)
        if protect_vertices is not None and protect_vertices.numel() > 0:
            pancake &= ~torch.isin(cells, protect_vertices).any(dim=1)
        if not pancake.any():
            break
        keep_cells = cells[~pancake]
        used = torch.unique(keep_cells.reshape(-1))
        new_points, new_cells = compact_mesh(points, keep_cells)
        if not boundary_is_closed_manifold(new_cells):
            break
        points, cells = new_points, new_cells
        if protect_vertices is not None and protect_vertices.numel() > 0:
            # Compaction renumbered the vertices; remap the protected ids
            # (their cells are protected, so they are always in `used`).
            protect_vertices = torch.searchsorted(used, protect_vertices)
        n_peeled += int(pancake.sum())
    return points, cells, n_peeled


def pin_feature_points(points, cells, targets, h):
    """Insert one vertex exactly at each feature point; return its index.

    Walking an existing vertex to a corner deadlocks against the validity
    gate (the vertex cannot move until its neighbors reshape, and they feel
    no pressure while it is blocked), so features are pinned by TOPOLOGICAL
    insertion instead:

    - a feature strictly inside some cell splits that cell 1 -> d+1 around
      a new vertex at the feature (always valid, by barycentric coords);
    - a feature outside the mesh (the common case: convex corners lie
      beyond the eroded staircase) gets a "tent" -- a new cell over the
      nearest boundary facet whose outward side faces the feature;
    - a feature that admits neither (e.g. farther than ~2h from the mesh)
      raises: it is not resolvable at this resolution.

    Returns ``(points, cells, fixed_idx (n_features,))``.
    """
    d = points.shape[1]
    fixed_idx = []
    for k in range(targets.shape[0]):
        x = targets[k]
        # Containing cell via barycentric coordinates (batched solve).
        p0 = points[cells[:, 0]]
        rel = points[cells[:, 1:]] - p0[:, None, :]
        vol_ok = signed_volumes(points, cells).abs() > 1e-12 * h**d
        bary = torch.zeros(cells.shape[0], d, dtype=points.dtype, device=points.device)
        bary[vol_ok] = torch.linalg.solve(rel[vol_ok], (x - p0[vol_ok])[:, :, None])[
            :, :, 0
        ]
        lam0 = 1.0 - bary.sum(dim=1)
        eps = 1e-9
        inside = vol_ok & (bary > eps).all(dim=1) & (lam0 > eps)
        new_vid = points.shape[0]
        if bool(inside.any()):
            c = int(torch.nonzero(inside)[0])
            host = cells[c]
            # 1 -> d+1 split: replace host with the star of the new vertex.
            new_cells = []
            for drop in range(d + 1):
                keepv = [host[i] for i in range(d + 1) if i != drop]
                new_cells.append(
                    torch.tensor(
                        [int(v) for v in keepv] + [new_vid],
                        dtype=torch.int64,
                        device=cells.device,
                    )
                )
            keep_mask = torch.ones(
                cells.shape[0], dtype=torch.bool, device=cells.device
            )
            keep_mask[c] = False
            points = torch.cat([points, x[None, :]], dim=0)
            cells = torch.cat([cells[keep_mask], torch.stack(new_cells)], dim=0)
        else:
            # Tent over the nearest outward-facing boundary facet.
            uniq, counts, _, _ = facet_census(cells)
            bfacets = uniq[counts == 1]
            cen = points[bfacets].mean(dim=1)
            order = (cen - x).norm(dim=1).argsort()
            placed = False
            for f in order[:8].tolist():
                fv = bfacets[f]
                tent = torch.cat([fv, torch.tensor([new_vid], device=cells.device)])
                trial_points = torch.cat([points, x[None, :]], dim=0)
                vol = signed_volumes(trial_points, tent[None, :])
                if float(vol.abs()) < 1e-9 * h**d:
                    continue  # feature coplanar with this facet; try next
                if float((cen[f] - x).norm()) > 2.5 * h:
                    break
                points = trial_points
                if vol < 0:
                    tent = tent[[1, 0] + list(range(2, d + 1))]
                cells = torch.cat([cells, tent[None, :]], dim=0)
                placed = True
                break
            if not placed:
                raise ValueError(
                    f"feature point {x.tolist()} is not resolvable at "
                    f"h={h}: it lies neither inside the mesh nor within "
                    f"~2.5h of a boundary facet. Decrease h or move the "
                    f"feature point."
                )
        fixed_idx.append(new_vid)
    cells = orient_positive(points, cells)
    return (
        points,
        cells,
        torch.tensor(fixed_idx, dtype=torch.int64, device=cells.device),
    )
