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

"""Conservative node--triangle discovery and live closest-point geometry.

This is an independent geometry implementation, not an OpenRadioss solver port.
Discovery is discrete; all returned closest-point quantities are live Torch
tensors. The explicit Torch broad phase is a chunked validation reference, not
the full-car backend. The Warp backend queries a BVH of expanded facet bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TriangleProjection:
    """Live closest point, barycentric weights, distance, and triangle normal."""

    closest: torch.Tensor
    barycentric: torch.Tensor
    distance: torch.Tensor
    normal: torch.Tensor


def closest_point_triangle(
    points: torch.Tensor, triangles: torch.Tensor, *, degenerate: str = "raise"
) -> TriangleProjection:
    """Project E points onto E closed, nondegenerate triangles, including edges.

    Shapes are [E, 3] and [E, 3, 3]. Region selection is piecewise discrete;
    selected barycentric coordinates and projections retain autograd. Degenerate
    triangles fail closed by default. ``degenerate="edges"`` explicitly treats
    a collapsed material triangle as its closed edges (a point if all coincide).
    Its normal is zero, not an invented shell normal. Near-degeneracy uses the
    same relative floating-point criterion as the strict path; branch selection
    and ties are discrete, with live geometry on the selected branch.
    """
    if degenerate not in ("raise", "edges"):
        raise ValueError("degenerate must be 'raise' or 'edges'")
    if (
        points.ndim != 2
        or points.shape[-1] != 3
        or triangles.shape != (len(points), 3, 3)
    ):
        raise ValueError("points [E,3] and triangles [E,3,3] are required")
    if points.dtype != triangles.dtype or points.device != triangles.device:
        raise ValueError("points and triangles must share dtype and device")
    if points.dtype not in (torch.float32, torch.float64):
        raise ValueError("physical geometry requires float32 or float64")
    if not torch.isfinite(points).all() or not torch.isfinite(triangles).all():
        raise ValueError("geometry must be finite")
    origin = triangles[:, 0]
    local = triangles - origin[:, None]
    # Work in a local, dimensionless frame: squaring a physical cross product
    # otherwise over/underflows at perfectly valid finite length scales. The
    # detached scale is only a change of coordinates, not a learned quantity.
    scale = local.detach().abs().amax(dim=(1, 2))
    if degenerate == "raise" and torch.any(scale == 0):
        raise ValueError("degenerate surface triangle")
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    local = local / scale[:, None, None]
    query = (points - origin) / scale[:, None]
    if not torch.isfinite(local).all() or not torch.isfinite(query).all():
        raise ValueError("surface geometry exceeds supported relative numeric range")
    a, b, c = local.unbind(1)
    ab, ac = b - a, c - a
    cross = torch.linalg.cross(ab, ac)
    area_sq = cross.square().sum(-1)
    scale_sq = torch.maximum(ab.square().sum(-1), ac.square().sum(-1))
    # Relative degeneracy criterion in the normalized local frame.
    eps = torch.finfo(points.dtype).eps
    valid = (area_sq > (eps * scale_sq).square()) & (scale_sq > 0)
    if degenerate == "raise" and not valid.all():
        raise ValueError("degenerate surface triangle")
    safe_area_sq = torch.where(valid, area_sq, torch.ones_like(area_sq))
    normal = torch.where(valid[:, None], cross / safe_area_sq.sqrt()[:, None], 0.0)
    pa = query - a
    # Cross-product formula avoids cancellation in the Gram determinant.
    vb = (torch.linalg.cross(pa, ac) * cross).sum(-1) / safe_area_sq
    vc = (torch.linalg.cross(ab, pa) * cross).sum(-1) / safe_area_sq
    plane_bary = torch.stack((1 - vb - vc, vb, vc), -1)
    choices = [plane_bary]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        start, edge = local[:, i], local[:, j] - local[:, i]
        length_sq = edge.square().sum(-1)
        denominator = torch.where(length_sq > 0, length_sq, torch.ones_like(length_sq))
        t = ((query - start) * edge).sum(-1) / denominator
        t = t.clamp(0, 1)
        basis_i = torch.nn.functional.one_hot(
            torch.tensor(i, device=points.device), 3
        ).to(points.dtype)
        basis_j = torch.nn.functional.one_hot(
            torch.tensor(j, device=points.device), 3
        ).to(points.dtype)
        choices.append((1 - t[:, None]) * basis_i + t[:, None] * basis_j)
    bary = torch.stack(choices, 1)
    projections = (bary[..., None] * local[:, None]).sum(2)
    distances = torch.linalg.vector_norm(projections - query[:, None], dim=-1)
    if not torch.isfinite(distances).all():
        raise ValueError("surface geometry exceeds supported relative numeric range")
    inside = valid & (plane_bary >= 0).all(-1)
    distances = torch.cat(
        (torch.where(inside, distances[:, 0], float("inf"))[:, None], distances[:, 1:]),
        1,
    )
    selected = distances.detach().argmin(-1)
    rows = torch.arange(len(points), device=points.device)
    selected_bary = bary[rows, selected]
    closest = origin + projections[rows, selected] * scale[:, None]
    return TriangleProjection(
        closest,
        selected_bary,
        distances[rows, selected] * scale,
        normal,
    )


@torch.no_grad()
def supported_quad_mask(quads: torch.Tensor) -> torch.Tensor:
    """Identify nonoverlapping center-fans in their area-normal projection.

    The arithmetic center must lie strictly inside the projected polygon's
    visibility kernel: all four fan triangles must have positive signed area.
    This admits convex and mildly concave centroid-star-shaped quads, including
    warped surfaces, but rejects overlapping/bow-tie, collapsed, and numerically
    ambiguous fans. It is a center-fan policy, not a general finite-element
    quality or inversion test. Returned membership is discrete.
    """
    if quads.ndim != 3 or quads.shape[1:] != (4, 3):
        raise ValueError("quads must have shape [E,4,3]")
    if quads.dtype not in (torch.float32, torch.float64):
        raise ValueError("physical geometry requires float32 or float64")
    q = quads - quads[:, :1]
    scale = q.abs().amax(dim=(1, 2))
    q = q / torch.where(scale > 0, scale, torch.ones_like(scale))[:, None, None]
    radial = q - q.mean(1, keepdim=True)
    fan_area = torch.linalg.cross(radial, radial.roll(-1, 1))
    area = fan_area.sum(1)
    area_norm = torch.linalg.vector_norm(area, dim=-1)
    normal = area / area_norm.clamp_min(torch.finfo(q.dtype).tiny)[:, None]
    signed_fan_area = (fan_area * normal[:, None]).sum(-1)
    tolerance = 32 * torch.finfo(q.dtype).eps
    return (
        torch.isfinite(q).all(dim=(1, 2))
        & (scale > 0)
        & (area_norm > tolerance)
        & (signed_fan_area > tolerance).all(1)
    )


def closest_point_facet(
    points: torch.Tensor,
    facets: torch.Tensor,
    *,
    triangle_mask: torch.Tensor | None = None,
    validate_quads: bool = True,
    material_fan: bool = False,
) -> TriangleProjection:
    """Triangle or four-node facet projection with live interpolation weights.

    Four-node facets use four triangles around the arithmetic center, matching
    the geometric decomposition of the pinned TYPE7 reference (not a diagonal
    split). A triangle in a mixed array repeats its third vertex in slot four.
    The returned barycentric weights refer to the ORIGINAL three/four vertices.
    Unsupported quad shapes raise. ``validate_quads=False`` is reserved for
    callers that already checked the same geometry with ``supported_quad_mask``.

    ``material_fan=True`` is a different, explicit surface contract: transport
    the four center-fan triangles of a valid reference quad with the vertices.
    The surface is their union even after folding/overlap, not an inferred simple
    polygon in an area-normal projection. Collapsed triangles retain their
    edges/vertices; no facet is dropped. The caller must validate the reference
    mesh before opting in. This does not repair an inverted FE solution, enforce
    nonpenetration, or claim bilinear-quad/solver equivalence.
    """
    if facets.ndim != 3 or facets.shape[0] != len(points) or facets.shape[2] != 3:
        raise ValueError("facets must have shape [E,3|4,3]")
    if facets.shape[1] == 3:
        return closest_point_triangle(
            points, facets, degenerate="edges" if material_fan else "raise"
        )
    if facets.shape[1] != 4:
        raise ValueError("only triangles and quadrilaterals are supported")
    triangle = (
        (facets[:, 2] == facets[:, 3]).all(-1)
        if triangle_mask is None
        else triangle_mask
    )
    if (
        triangle.shape != (len(points),)
        or triangle.dtype != torch.bool
        or triangle.device != points.device
    ):
        raise ValueError("triangle_mask must be device-matched bool [E]")
    bary = points.new_zeros((len(points), 4))
    normal = torch.zeros_like(points)
    tri_ids = triangle.nonzero().flatten()
    quad_ids = (~triangle).nonzero().flatten()
    if len(tri_ids):
        proj = closest_point_triangle(
            points[tri_ids],
            facets[tri_ids, :3],
            degenerate="edges" if material_fan else "raise",
        )
        bary = bary.index_copy(
            0, tri_ids, torch.nn.functional.pad(proj.barycentric, (0, 1))
        )
        normal = normal.index_copy(0, tri_ids, proj.normal)
    if len(quad_ids):
        q = facets[quad_ids]
        if not material_fan and validate_quads and not supported_quad_mask(q).all():
            raise ValueError(
                "unsupported surface quad: overlapping, folded, or degenerate center fan"
            )
        center = q.mean(1)
        triangles = torch.stack(
            [torch.stack((center, q[:, i], q[:, (i + 1) % 4]), 1) for i in range(4)], 1
        )
        proj = closest_point_triangle(
            points[quad_ids, None].expand(-1, 4, -1).reshape(-1, 3),
            triangles.reshape(-1, 3, 3),
            degenerate="edges" if material_fan else "raise",
        )
        weights = proj.barycentric.reshape(-1, 4, 3)
        # Map the virtual center weight back to all four real vertices.
        basis = torch.eye(4, device=q.device, dtype=q.dtype)
        mapped = (
            weights[:, :, 0, None] / 4
            + weights[:, :, 1, None] * basis
            + weights[:, :, 2, None] * basis.roll(-1, 0)
        )
        selected = proj.distance.reshape(-1, 4).detach().argmin(-1)
        rows = torch.arange(len(q), device=q.device)
        bary = bary.index_copy(0, quad_ids, mapped[rows, selected])
        normal = normal.index_copy(
            0, quad_ids, proj.normal.reshape(-1, 4, 3)[rows, selected]
        )
    closest = (bary[..., None] * facets).sum(1)
    return TriangleProjection(
        closest, bary, torch.linalg.vector_norm(points - closest, dim=-1), normal
    )


@torch.no_grad()
def node_triangle_candidates(
    positions: torch.Tensor,
    faces: torch.Tensor,
    node_padding: torch.Tensor,
    face_padding: torch.Tensor,
    *,
    batch: torch.Tensor | None = None,
    previous_positions: torch.Tensor | None = None,
    excluded_pairs: torch.Tensor | None = None,
    implementation: str = "warp",
    max_pairs: int = 2_000_000,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Collect sorted unique [node, face] pairs, failing on total budget overflow.

    This compatibility API retains the raw-candidate cap before exclusions.
    Use ``node_triangle_candidate_chunks`` for memory-bounded exhaustive
    discovery when broad-phase false positives exceed the final graph budget.
    """
    if not isinstance(max_pairs, int) or isinstance(max_pairs, bool) or max_pairs <= 0:
        raise ValueError("max_pairs must be a positive integer")
    chunks = list(
        node_triangle_candidate_chunks(
            positions,
            faces,
            node_padding,
            face_padding,
            batch=batch,
            previous_positions=previous_positions,
            excluded_pairs=excluded_pairs,
            implementation=implementation,
            max_pairs=max_pairs,
            chunk_size=chunk_size,
        )
    )
    pairs = torch.cat(chunks, 1) if chunks else faces.new_empty((2, 0))
    order = torch.argsort(pairs[0] * len(faces) + pairs[1], stable=True)
    return pairs[:, order]


@torch.no_grad()
def node_triangle_candidate_chunks(
    positions: torch.Tensor,
    faces: torch.Tensor,
    node_padding: torch.Tensor,
    face_padding: torch.Tensor,
    *,
    batch: torch.Tensor | None = None,
    previous_positions: torch.Tensor | None = None,
    excluded_pairs: torch.Tensor | None = None,
    implementation: str = "warp",
    max_pairs: int | None = None,
    chunk_size: int = 256,
    pair_chunk_size: int = 65_536,
):
    """Return an iterator of exhaustive [2,E] pair chunks, E <= pair_chunk_size.

    BVH boxes cover the WHOLE facet, not only its vertices/centroid. Padding is
    nonnegative physical length. With previous_positions, boxes cover the full
    linear sweep of each node/facet; this alone is NOT continuous detection.
    Supports three-node triangles and four-node facets. Incidence and cross-batch
    pairs are excluded; optional [2,E] exclusions refer
    to node and face IDs, not graph hops. Pairs are unique across all chunks,
    but chunk ordering is backend-dependent. Sort after narrow-phase selection
    for canonical model reduction order. ``max_pairs=None`` imposes no total
    raw-pair cap; an explicit cap raises before dropping any pairs. Each page
    remains bounded even when one query overlaps more faces than the page size.

    Validation and bounds construction execute eagerly under no_grad. The
    returned generator only operates on detached bounds and discrete indices.
    """
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or positions.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("positions must be float32/float64 [N,3]")
    if (
        faces.ndim != 2
        or faces.shape[1] not in (3, 4)
        or faces.dtype != torch.long
        or faces.device != positions.device
    ):
        raise ValueError("faces must be device-matched int64 [F,3|4]")
    n, f = len(positions), len(faces)
    if torch.any(faces < 0) or torch.any(faces >= n):
        raise ValueError("face node index out of bounds")
    if max_pairs is not None and (
        not isinstance(max_pairs, int) or isinstance(max_pairs, bool) or max_pairs <= 0
    ):
        raise ValueError("max_pairs must be a positive integer")
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be positive")
    if (
        not isinstance(pair_chunk_size, int)
        or isinstance(pair_chunk_size, bool)
        or pair_chunk_size <= 0
    ):
        raise ValueError("pair_chunk_size must be a positive integer")
    for padding, size in ((node_padding, n), (face_padding, f)):
        if (
            padding.shape != (size,)
            or padding.device != positions.device
            or padding.dtype != positions.dtype
        ):
            raise ValueError("padding must be a dtype/device-matched vector")
        if not torch.isfinite(padding).all() or torch.any(padding < 0):
            raise ValueError("padding must be finite and nonnegative")
    if not torch.isfinite(positions).all():
        raise ValueError("positions must be finite")
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=positions.device)
    if (
        batch.shape != (n,)
        or batch.dtype != torch.long
        or batch.device != positions.device
        or torch.any(batch < 0)
    ):
        raise ValueError("batch must be device-matched nonnegative int64 [N]")
    if torch.any(batch[faces] != batch[faces[:, :1]]):
        raise ValueError("a surface face cannot span batch graphs")
    if previous_positions is None:
        previous_positions = positions
    if (
        previous_positions.shape != positions.shape
        or previous_positions.device != positions.device
        or previous_positions.dtype != positions.dtype
        or not torch.isfinite(previous_positions).all()
    ):
        raise ValueError("previous_positions must match positions and be finite")
    if excluded_pairs is not None:
        if (
            excluded_pairs.ndim != 2
            or excluded_pairs.shape[0] != 2
            or excluded_pairs.dtype != torch.long
            or excluded_pairs.device != positions.device
        ):
            raise ValueError("excluded_pairs must be device-matched int64 [2,E]")
        if (
            torch.any(excluded_pairs < 0)
            or torch.any(excluded_pairs[0] >= n)
            or torch.any(excluded_pairs[1] >= f)
        ):
            raise ValueError("excluded pair out of bounds")
    if implementation not in ("torch", "warp"):
        raise ValueError("implementation must be 'torch' or 'warp'")
    if not n or not f:
        return iter(())
    lower = (
        torch.minimum(positions[faces].amin(1), previous_positions[faces].amin(1))
        - face_padding[:, None]
    )
    upper = (
        torch.maximum(positions[faces].amax(1), previous_positions[faces].amax(1))
        + face_padding[:, None]
    )
    query_lower = torch.minimum(positions, previous_positions) - node_padding[:, None]
    query_upper = torch.maximum(positions, previous_positions) + node_padding[:, None]
    # Outward rounding ensures touching boxes survive floating point arithmetic.
    lower, query_lower = [
        torch.nextafter(x, torch.full_like(x, -float("inf")))
        for x in (lower, query_lower)
    ]
    upper, query_upper = [
        torch.nextafter(x, torch.full_like(x, float("inf")))
        for x in (upper, query_upper)
    ]
    if not all(
        torch.isfinite(x).all() for x in (lower, upper, query_lower, query_upper)
    ):
        raise ValueError("expanded surface bounds overflowed")
    if implementation == "warp":
        if positions.dtype != torch.float32:
            raise ValueError(
                "Warp discovery requires float32; use torch for float64 reference"
            )
        from .surface_contact_warp import bvh_pair_chunks

        chunks = bvh_pair_chunks(
            lower,
            upper,
            query_lower,
            query_upper,
            faces,
            batch,
            pair_chunk_size,
            max_pairs,
        )
    else:

        def torch_chunks():
            total = 0
            # Tile BOTH axes: a single node can overlap arbitrarily many faces.
            query_chunk = min(chunk_size, pair_chunk_size)
            face_chunk = max(1, pair_chunk_size // query_chunk)
            for start in range(0, n, query_chunk):
                stop = min(start + query_chunk, n)
                nodes = torch.arange(start, stop, device=positions.device)
                for fs in range(0, f, face_chunk):
                    fe = min(fs + face_chunk, f)
                    overlap = (
                        (query_lower[start:stop, None] <= upper[fs:fe])
                        & (query_upper[start:stop, None] >= lower[fs:fe])
                    ).all(-1)
                    overlap &= (nodes[:, None, None] != faces[None, fs:fe]).all(-1)
                    overlap &= batch[start:stop, None] == batch[faces[fs:fe, 0]][None]
                    q, face = overlap.nonzero(as_tuple=True)
                    total += len(q)
                    if max_pairs is not None and total > max_pairs:
                        raise RuntimeError(
                            f"surface candidate budget exceeded ({total}>{max_pairs}); no pairs truncated"
                        )
                    if len(q):
                        yield torch.stack((q + start, face + fs))

        chunks = torch_chunks()

    codes = (
        torch.sort(excluded_pairs[0] * f + excluded_pairs[1]).values
        if excluded_pairs is not None and excluded_pairs.numel()
        else None
    )

    def filtered_chunks():
        for pairs in chunks:
            if codes is not None:
                # Sort the static exclusions once per discovery, not the full
                # multi-million-entry list once for every candidate page.
                query_codes = pairs[0] * f + pairs[1]
                indices = torch.searchsorted(codes, query_codes)
                excluded = (indices < len(codes)) & (
                    codes[indices.clamp_max(len(codes) - 1)] == query_codes
                )
                pairs = pairs[:, ~excluded]
            if pairs.shape[1]:
                yield pairs

    return filtered_chunks()


@dataclass(frozen=True)
class SweptContactResult:
    """Linear-motion interval certificate; unresolved is NOT a negative result."""

    hit: torch.Tensor
    unresolved: torch.Tensor
    time: torch.Tensor


@torch.no_grad()
def swept_node_triangle_check(
    start_points: torch.Tensor,
    end_points: torch.Tensor,
    start_triangles: torch.Tensor,
    end_triangles: torch.Tensor,
    clearance: torch.Tensor,
    *,
    max_depth: int = 12,
    max_intervals: int = 1_000_000,
) -> SweptContactResult:
    """Conservative interval test for linearly moving points and triangles.

    A Lipschitz distance bound proves separated intervals safe. Midpoint/end
    samples certify hits; unresolved intervals at the depth/budget limit remain
    explicitly marked. The returned time is a witnessed hit, NOT exact first TOI.
    This diagnostic does not change a model's integration or claim solver CCD.
    """
    if (
        not isinstance(max_depth, int)
        or isinstance(max_depth, bool)
        or max_depth < 0
        or not isinstance(max_intervals, int)
        or isinstance(max_intervals, bool)
        or max_intervals < 1
    ):
        raise ValueError("invalid swept-test budget")
    if (
        end_points.shape != start_points.shape
        or end_triangles.shape != start_triangles.shape
    ):
        raise ValueError("sweep endpoint shapes must match")
    if (
        clearance.shape != (len(start_points),)
        or torch.any(clearance < 0)
        or not torch.isfinite(clearance).all()
    ):
        raise ValueError("clearance must be finite, nonnegative [E]")
    if any(
        x.dtype != start_points.dtype or x.device != start_points.device
        for x in (end_points, start_triangles, end_triangles, clearance)
    ):
        raise ValueError("sweep tensors must share dtype and device")
    e = len(start_points)
    hit = torch.zeros(e, dtype=torch.bool, device=start_points.device)
    unresolved = torch.zeros_like(hit)
    times = torch.full_like(clearance, float("nan"))
    velocity = end_points - start_points
    face_velocity = end_triangles - start_triangles
    speed = torch.linalg.vector_norm(velocity[:, None] - face_velocity, dim=-1).amax(-1)
    coordinate_scale = torch.stack(
        (
            start_points.abs().amax(-1),
            end_points.abs().amax(-1),
            start_triangles.abs().amax((-1, -2)),
            end_triangles.abs().amax((-1, -2)),
        )
    ).amax(0)
    for t in (0.0, 1.0):
        distance = closest_point_triangle(
            start_points + t * velocity, start_triangles + t * face_velocity
        ).distance
        active = distance <= clearance
        times = torch.where(active & ~hit, times.new_full((), t), times)
        hit |= active
    ids = (~hit).nonzero().flatten()
    lo, hi = torch.zeros_like(clearance[ids]), torch.ones_like(clearance[ids])
    for depth in range(max_depth + 1):
        if not len(ids):
            break
        if len(ids) > max_intervals:
            unresolved[ids.unique()] = True
            break
        mid = (lo + hi) * 0.5
        distance = closest_point_triangle(
            start_points[ids] + mid[:, None] * velocity[ids],
            start_triangles[ids] + mid[:, None, None] * face_velocity[ids],
        ).distance
        active = distance <= clearance[ids]
        # Sorting/min reduce avoids duplicate-index assignment races on CUDA.
        witnessed = ids[active]
        if len(witnessed):
            first = torch.full_like(clearance, float("inf"))
            first.scatter_reduce_(0, witnessed, mid[active], reduce="amin")
            new = torch.isfinite(first) & ~hit
            times[new] = first[new]
            hit[witnessed.unique()] = True
        bound = distance - speed[ids] * (hi - lo) * 0.5
        # Small roundoff allowance makes certificates conservative numerically.
        tolerance = (
            64
            * torch.finfo(distance.dtype).eps
            * (1 + coordinate_scale[ids] + distance + speed[ids])
        )
        pending = (bound <= clearance[ids] + tolerance) & ~hit[ids]
        ids, lo, hi, mid = ids[pending], lo[pending], hi[pending], mid[pending]
        if depth == max_depth or 2 * len(ids) > max_intervals:
            unresolved[ids.unique()] = True
            break
        ids, lo, hi = ids.repeat(2), torch.cat((lo, mid)), torch.cat((mid, hi))
    return SweptContactResult(hit, unresolved & ~hit, times)
