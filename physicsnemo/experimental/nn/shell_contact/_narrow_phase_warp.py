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
"""Fused, discrete contact membership; live geometry still uses Torch autograd.

This kernel follows the same closed-triangle/arithmetic-center-fan projection,
variable shell allowance, and two-anchor forecast as the reference. Numerically
ambiguous pairs are marked for reference recomputation, NOT dropped. Collapsed
fans and nearly tied projections with different material coordinates also use
the reference. No approximate nearest-neighbor search or contact truncation.
"""

import torch
import warp as wp

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.struct
class _Projection:
    bary: wp.vec4
    closest: wp.vec3
    distance: float
    uncertain: int


@wp.func
def _maxabs(v: wp.vec3):
    return wp.max(wp.abs(v[0]), wp.max(wp.abs(v[1]), wp.abs(v[2])))


@wp.func
def _triangle(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    result = _Projection()
    ab_world = b - a
    ac_world = c - a
    scale = wp.max(_maxabs(ab_world), _maxabs(ac_world))
    if scale == 0.0:
        scale = 1.0
    ab = ab_world / scale
    ac = ac_world / scale
    query = (p - a) / scale
    cross = wp.cross(ab, ac)
    area_sq = wp.dot(cross, cross)
    scale_sq = wp.max(wp.dot(ab, ab), wp.dot(ac, ac))
    threshold = 1.1920928955078125e-7 * scale_sq
    valid = area_sq > threshold * threshold and scale_sq > 0.0
    # Ill-conditioned but technically valid fans also use the reference:
    # barycentric roundoff can be amplified before the activation-gap test.
    if not valid or area_sq <= 64.0 * 1.1920928955078125e-7 * scale_sq * scale_sq:
        result.uncertain = 1
    denominator = 1.0
    if valid:
        denominator = area_sq
    vb = wp.dot(wp.cross(query, ac), cross) / denominator
    vc = wp.dot(wp.cross(ab, query), cross) / denominator
    best_bary = wp.vec3(1.0 - vb - vc, vb, vc)
    best_distance = float(wp.inf)
    if valid and best_bary[0] >= 0.0 and vb >= 0.0 and vc >= 0.0:
        best_distance = wp.length(vb * ab + vc * ac - query)
    # Reference argmin ordering: plane, edge01, edge12, edge20; first wins ties.
    for edge_id in range(3):
        start = wp.vec3(0.0)
        end = ab
        if edge_id == 1:
            start = ab
            end = ac
        elif edge_id == 2:
            start = ac
            end = wp.vec3(0.0)
        edge = end - start
        length_sq = wp.dot(edge, edge)
        length_denominator = 1.0
        if length_sq > 0.0:
            length_denominator = length_sq
        t = wp.clamp(wp.dot(query - start, edge) / length_denominator, 0.0, 1.0)
        bary = wp.vec3(0.0)
        bary[edge_id] = 1.0 - t
        bary[(edge_id + 1) % 3] = t
        # The reference computes projections from barycentrics, not start+t*edge.
        distance = wp.length(bary[1] * ab + bary[2] * ac - query)
        if distance < best_distance:
            best_distance = distance
            best_bary = bary
    result.bary = wp.vec4(best_bary[0], best_bary[1], best_bary[2], 0.0)
    result.closest = a + (best_bary[1] * ab + best_bary[2] * ac) * scale
    result.distance = best_distance * scale
    if not wp.isfinite(result.distance) or not wp.isfinite(_maxabs(query)):
        result.uncertain = 1
    return result


@wp.func
def _facet(
    p: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    d: wp.vec3,
    width: int,
    triangle: bool,
):
    result = _Projection()
    if triangle:
        result = _triangle(p, a, b, c)
    else:
        center = (a + b + c + d) * 0.25
        best_distance = float(wp.inf)
        best_bary = wp.vec4(0.0)
        for fan in range(4):
            first = a
            second = b
            if fan == 1:
                first = b
                second = c
            elif fan == 2:
                first = c
                second = d
            elif fan == 3:
                first = d
                second = a
            part = _triangle(p, center, first, second)
            result.uncertain = wp.max(result.uncertain, part.uncertain)
            bary = wp.vec4(part.bary[0] * 0.25)
            bary[fan] = bary[fan] + part.bary[1]
            bary[(fan + 1) % 4] = bary[(fan + 1) % 4] + part.bary[2]
            # A folded material fan can have equidistant but materially distinct
            # closest points. Do not change its reference tie/region selection.
            tie_tol = 64.0 * 1.1920928955078125e-7 * (1.0 + part.distance)
            if (
                wp.isfinite(best_distance)
                and wp.abs(part.distance - best_distance) <= tie_tol
            ):
                difference = bary - best_bary
                if (
                    wp.abs(difference[0])
                    + wp.abs(difference[1])
                    + wp.abs(difference[2])
                    + wp.abs(difference[3])
                    > 1.0e-5
                ):
                    result.uncertain = 1
            if part.distance < best_distance:
                best_distance = part.distance
                best_bary = bary
        result.bary = best_bary
    if width == 4:
        result.closest = (
            result.bary[0] * a
            + result.bary[1] * b
            + result.bary[2] * c
            + result.bary[3] * d
        )
        result.distance = wp.length(p - result.closest)
    return result


@wp.func
def _interpolate(weights: wp.vec4, a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    return weights[0] * a + weights[1] * b + weights[2] * c + weights[3] * d


@wp.func
def _sweep(relative: wp.vec3, velocity: wp.vec3, horizon: float):
    speed_sq = wp.dot(velocity, velocity)
    denominator = 1.0
    if speed_sq > 0.0:
        denominator = speed_sq
    time = wp.clamp(-wp.dot(relative, velocity) / denominator, 0.0, horizon)
    return wp.length(relative + time * velocity)


@wp.kernel
def _activation_gaps(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    thickness: wp.array(dtype=float),
    faces: wp.array2d(dtype=wp.int64),
    pairs: wp.array2d(dtype=wp.int64),
    horizon: float,
    band: float,
    sweep_band: float,
    gaps: wp.array(dtype=float),
    uncertain: wp.array(dtype=wp.bool),
):
    tid = wp.tid()
    query = pairs[0, tid]
    face = pairs[1, tid]
    i0, i1, i2 = faces[face, 0], faces[face, 1], faces[face, 2]
    i3 = i2
    if faces.shape[1] == 4:
        i3 = faces[face, 3]
    p = positions[query]
    a, b, c, d = positions[i0], positions[i1], positions[i2], positions[i3]
    triangle = faces.shape[1] == 3 or i2 == i3
    projection = _facet(p, a, b, c, d, faces.shape[1], triangle)
    surface_thickness = (
        projection.bary[0] * thickness[i0]
        + projection.bary[1] * thickness[i1]
        + projection.bary[2] * thickness[i2]
        + projection.bary[3] * thickness[i3]
    )
    half_thickness = (thickness[query] + surface_thickness) * 0.5
    gap = projection.distance - half_thickness
    ambiguous = projection.uncertain != 0
    coordinate_scale = wp.max(
        _maxabs(p),
        wp.max(_maxabs(a), wp.max(_maxabs(b), wp.max(_maxabs(c), _maxabs(d)))),
    )
    if horizon > 0.0:
        vp = velocities[query]
        va, vb, vc, vd = velocities[i0], velocities[i1], velocities[i2], velocities[i3]
        relative = projection.closest - p
        relative_velocity = _interpolate(projection.bary, va, vb, vc, vd) - vp
        gap = (
            wp.min(_sweep(relative, relative_velocity, horizon), projection.distance)
            - half_thickness
        )
        future = _facet(
            p + horizon * vp,
            a + horizon * va,
            b + horizon * vb,
            c + horizon * vc,
            d + horizon * vd,
            faces.shape[1],
            triangle,
        )
        ambiguous = ambiguous or future.uncertain != 0
        future_relative = _interpolate(future.bary, a, b, c, d) - p
        future_velocity = _interpolate(future.bary, va, vb, vc, vd) - vp
        future_thickness = (
            future.bary[0] * thickness[i0]
            + future.bary[1] * thickness[i1]
            + future.bary[2] * thickness[i2]
            + future.bary[3] * thickness[i3]
        )
        future_gap = (
            _sweep(future_relative, future_velocity, horizon)
            - (thickness[query] + future_thickness) * 0.5
        )
        gap = wp.min(gap, future_gap)
        coordinate_scale = coordinate_scale + horizon * wp.max(
            _maxabs(vp),
            wp.max(_maxabs(va), wp.max(_maxabs(vb), wp.max(_maxabs(vc), _maxabs(vd)))),
        )
    # Conservative roundoff neighborhood: reference decides every ambiguous
    # boundary. Include absolute translation and thickness, not just tiny gaps.
    thickness_scale = wp.max(
        thickness[query],
        wp.max(
            thickness[i0], wp.max(thickness[i1], wp.max(thickness[i2], thickness[i3]))
        ),
    )
    tolerance = (
        256.0
        * 1.1920928955078125e-7
        * (1.0 + coordinate_scale + thickness_scale + wp.max(band, sweep_band))
    )
    gaps[tid] = gap
    # The crossing sweep (sweep_band > band) has a second membership boundary;
    # with sweep_band == band this is the original single-boundary condition.
    uncertain[tid] = (
        ambiguous
        or not wp.isfinite(gap)
        or wp.abs(gap - band) <= tolerance
        or wp.abs(gap - sweep_band) <= tolerance
    )


@torch.no_grad()
def fused_activation_gaps(
    positions, velocities, thickness, faces, pairs, horizon, band, sweep_band=None
):
    """Return detached gaps and a reference-recompute mask on the Torch stream.

    Caller validates physical FP32 inputs/connectivity. This is ONLY discrete
    membership acceleration: do not use these outputs as learned edge features.
    ``sweep_band`` (default ``band``) is the crossing sweep's outer membership
    boundary; pairs numerically near either boundary are marked for the
    reference recompute.
    """
    if sweep_band is None:
        sweep_band = band
    if positions.dtype != torch.float32 or not positions.is_cuda:
        raise ValueError("fused membership requires CUDA float32 geometry")
    if pairs.shape[1] == 0:
        return thickness.new_empty(0), torch.empty(
            0, device=positions.device, dtype=torch.bool
        )
    wp.init()
    stream = wp.stream_from_torch(torch.cuda.current_stream(positions.device))
    device = wp.device_from_torch(positions.device)
    if velocities is None:
        if horizon > 0:
            raise ValueError("positive forecast horizon requires velocities")
        # No velocity access occurs at horizon zero: alias instead of allocating
        # an O(N) zero tensor for every candidate page.
        velocities = positions
    gaps = thickness.new_empty(pairs.shape[1])
    uncertain = torch.empty(pairs.shape[1], device=positions.device, dtype=torch.bool)
    with wp.ScopedDevice(device), wp.ScopedStream(stream):
        wp.launch(
            _activation_gaps,
            dim=pairs.shape[1],
            inputs=[
                wp.from_torch(
                    positions.detach().contiguous(), dtype=wp.vec3, requires_grad=False
                ),
                wp.from_torch(
                    velocities.detach().contiguous(), dtype=wp.vec3, requires_grad=False
                ),
                wp.from_torch(thickness.detach().contiguous(), requires_grad=False),
                wp.from_torch(faces.contiguous()),
                wp.from_torch(pairs.contiguous()),
                float(horizon),
                float(band),
                float(sweep_band),
                wp.from_torch(gaps),
                wp.from_torch(uncertain),
            ],
            device=device,
            stream=stream,
        )
    return gaps, uncertain
