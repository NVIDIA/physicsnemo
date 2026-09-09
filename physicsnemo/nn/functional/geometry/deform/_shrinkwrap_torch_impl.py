# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Torch nearest-surface search and differentiable projection replay."""

from __future__ import annotations

import torch

_PAIR_BUDGET = 1 << 18
_FACE_CHUNK = 1024


def _saturating_product(
    nonnegative_scale: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Multiply by a nonnegative scale without returning an infinity."""

    maximum = torch.finfo(value.dtype).max
    safe_scale = torch.where(
        nonnegative_scale > 0,
        nonnegative_scale,
        torch.ones_like(nonnegative_scale),
    )
    representable_value = maximum / safe_scale
    overflow = value.abs() >= representable_value
    value_sign = torch.where(
        value < 0,
        -torch.ones_like(value),
        torch.ones_like(value),
    )
    safe_value = torch.where(overflow, value_sign, value)
    product = safe_scale * safe_value
    finite_product = torch.where(overflow, value_sign * maximum, product)
    return torch.where(
        nonnegative_scale > 0,
        finite_product,
        torch.zeros_like(finite_product),
    )


def _stable_vector_distance_with_key(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute saturated distances and non-saturating ordering keys."""

    coordinate_scale = torch.maximum(first.abs(), second.abs())
    safe_coordinate_scale = torch.where(
        coordinate_scale > 0,
        coordinate_scale,
        torch.ones_like(coordinate_scale),
    ).detach()
    normalized_difference = (
        first / safe_coordinate_scale - second / safe_coordinate_scale
    )

    scale_mantissa, scale_exponent = torch.frexp(coordinate_scale)
    difference_mantissa, difference_exponent = torch.frexp(normalized_difference.abs())
    component_mantissa, component_carry = torch.frexp(
        scale_mantissa * difference_mantissa
    )
    component_exponent = scale_exponent + difference_exponent + component_carry
    zero_component = (coordinate_scale == 0) | (normalized_difference == 0)
    minimum_exponent = torch.iinfo(component_exponent.dtype).min
    component_exponent = torch.where(
        zero_component,
        torch.full_like(component_exponent, minimum_exponent),
        component_exponent,
    )

    largest_component_exponent = component_exponent.amax(dim=-1)
    aligned_component_exponent = torch.where(
        zero_component,
        largest_component_exponent.unsqueeze(-1),
        component_exponent,
    )
    scaled_component = torch.where(
        zero_component,
        torch.zeros_like(component_mantissa),
        torch.ldexp(
            component_mantissa,
            aligned_component_exponent - largest_component_exponent.unsqueeze(-1),
        ),
    )
    scaled_norm = torch.sqrt((scaled_component * scaled_component).sum(dim=-1))
    distance_mantissa, norm_exponent = torch.frexp(scaled_norm)
    distance_exponent = largest_component_exponent + norm_exponent
    zero_distance = zero_component.all(dim=-1)
    distance_exponent = torch.where(
        zero_distance,
        torch.full_like(distance_exponent, minimum_exponent),
        distance_exponent,
    )
    distance_mantissa = torch.where(
        zero_distance,
        torch.zeros_like(distance_mantissa),
        distance_mantissa,
    )

    maximum = torch.full_like(distance_mantissa, torch.finfo(first.dtype).max)
    maximum_mantissa, maximum_exponent = torch.frexp(maximum)
    overflow = (distance_exponent > maximum_exponent) | (
        (distance_exponent == maximum_exponent) & (distance_mantissa > maximum_mantissa)
    )
    safe_distance_exponent = torch.where(
        zero_distance | overflow,
        torch.zeros_like(distance_exponent),
        distance_exponent,
    )
    representable_distance = torch.ldexp(
        distance_mantissa,
        safe_distance_exponent,
    )
    distance = torch.where(
        zero_distance,
        torch.zeros_like(representable_distance),
        torch.where(overflow, maximum, representable_distance),
    )
    return distance, distance_mantissa, distance_exponent


def _stable_vector_distance(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Compute finite point distances when direct subtraction may overflow."""

    distance, _, _ = _stable_vector_distance_with_key(first, second)
    return distance


def _saturating_ratio(
    numerator: torch.Tensor,
    positive_denominator: torch.Tensor,
) -> torch.Tensor:
    """Divide finite values without overflowing a representable quotient."""

    maximum = torch.finfo(numerator.dtype).max
    overflow = positive_denominator < numerator.abs() / maximum
    numerator_sign = torch.where(
        numerator < 0,
        -torch.ones_like(numerator),
        torch.ones_like(numerator),
    )
    safe_numerator = torch.where(
        overflow,
        numerator_sign * positive_denominator,
        numerator,
    )
    quotient = safe_numerator / positive_denominator
    return torch.where(overflow, numerator_sign * maximum, quotient)


def _stable_scaled_difference_dot(
    first: torch.Tensor,
    second: torch.Tensor,
    direction: torch.Tensor,
    positive_scale: torch.Tensor,
) -> torch.Tensor:
    """Return ``dot(first - second, direction) / positive_scale`` safely."""

    coordinate_scale = torch.stack((first.abs(), second.abs()), dim=-1).amax(dim=-1)
    safe_coordinate_scale = torch.where(
        coordinate_scale > 0,
        coordinate_scale,
        torch.ones_like(coordinate_scale),
    ).detach()
    normalized_difference = (
        first / safe_coordinate_scale - second / safe_coordinate_scale
    )

    weighted_direction = safe_coordinate_scale * direction
    term_scale = weighted_direction.abs().amax(dim=-1).detach()
    safe_term_scale = torch.where(
        term_scale > 0,
        term_scale,
        torch.ones_like(term_scale),
    )
    normalized_dot = (
        normalized_difference * (weighted_direction / safe_term_scale.unsqueeze(-1))
    ).sum(dim=-1)
    quotient_scale = _saturating_ratio(term_scale, positive_scale)
    return _saturating_product(quotient_scale, normalized_dot)


def _safe_edge_subtraction_scale(
    coordinate_magnitude: torch.Tensor,
) -> torch.Tensor:
    """Scale extreme coordinates enough to keep pairwise differences finite."""

    threshold = 0.25 * torch.finfo(coordinate_magnitude.dtype).max
    return torch.where(
        coordinate_magnitude > threshold,
        torch.full_like(coordinate_magnitude, 0.25),
        torch.ones_like(coordinate_magnitude),
    ).detach()


def _scale_triangles_for_safe_edges(triangles: torch.Tensor) -> torch.Tensor:
    """Scale each triangle before edge subtraction can overflow."""

    coordinate_magnitude = triangles.abs().amax(dim=(-2, -1))
    scale = _safe_edge_subtraction_scale(coordinate_magnitude)
    return triangles * scale[:, None, None]


def closest_points_on_triangles(
    query_points: torch.Tensor,
    triangles: torch.Tensor,
) -> torch.Tensor:
    """Return paired closest points using scale-normalized triangle math."""

    coordinate_magnitude = torch.maximum(
        query_points.abs().amax(dim=-1),
        triangles.abs().amax(dim=(-2, -1)),
    )
    coordinate_scale = _safe_edge_subtraction_scale(coordinate_magnitude)
    scaled_queries = query_points * coordinate_scale[:, None]
    scaled_triangles = triangles * coordinate_scale[:, None, None]
    # Uniform coordinate scaling cancels between the input and output of a
    # closest-point projection. Preserve that identity in the backward graph
    # so extreme forward scales cannot create overflowing intermediate
    # cotangents.
    query_points = query_points + (scaled_queries - query_points).detach()
    triangles = triangles + (scaled_triangles - triangles).detach()

    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    bc = c - b

    edge_scale = torch.stack(
        (
            ab.abs().amax(dim=-1),
            ac.abs().amax(dim=-1),
            bc.abs().amax(dim=-1),
        ),
        dim=-1,
    ).amax(dim=-1)
    edge_scale = torch.where(
        edge_scale > 0,
        edge_scale,
        torch.ones_like(edge_scale),
    ).detach()
    scale_column = edge_scale.unsqueeze(-1)
    ab_scaled = ab / scale_column
    ac_scaled = ac / scale_column
    bc_scaled = bc / scale_column

    tiny = torch.finfo(query_points.dtype).tiny
    gram_ab = (ab_scaled * ab_scaled).sum(dim=-1)
    gram_ac = (ac_scaled * ac_scaled).sum(dim=-1)
    rhs_ab = _stable_scaled_difference_dot(
        query_points,
        a,
        ab_scaled,
        edge_scale,
    )
    rhs_ac = _stable_scaled_difference_dot(
        query_points,
        a,
        ac_scaled,
        edge_scale,
    )

    primary_is_ab = gram_ab >= gram_ac
    primary = torch.where(primary_is_ab.unsqueeze(-1), ab_scaled, ac_scaled)
    secondary = torch.where(primary_is_ab.unsqueeze(-1), ac_scaled, ab_scaled)
    primary_norm = torch.sqrt((primary * primary).sum(dim=-1))
    safe_primary_norm = primary_norm.clamp_min(tiny)
    axis_u = primary / safe_primary_norm.unsqueeze(-1)
    normal = torch.linalg.cross(primary, secondary)
    normal_norm = torch.sqrt((normal * normal).sum(dim=-1))
    safe_normal_norm = normal_norm.clamp_min(tiny)
    axis_normal = normal / safe_normal_norm.unsqueeze(-1)
    axis_v = torch.linalg.cross(axis_normal, axis_u)
    secondary_u = (secondary * axis_u).sum(dim=-1)
    secondary_v = (secondary * axis_v).sum(dim=-1)

    query_u = _stable_scaled_difference_dot(
        query_points,
        a,
        axis_u,
        edge_scale,
    )
    query_v = _stable_scaled_difference_dot(
        query_points,
        a,
        axis_v,
        edge_scale,
    )
    query_scale = torch.stack(
        (
            query_u.abs(),
            query_v.abs(),
            torch.ones_like(query_u),
        ),
        dim=-1,
    ).amax(dim=-1)
    inverse_query_scale = query_scale.reciprocal()
    scaled_query_u = query_u * inverse_query_scale
    scaled_query_v = query_v * inverse_query_scale

    determinant = primary_norm * secondary_v
    scaled_determinant = determinant * inverse_query_scale
    primary_numerator = secondary_v * scaled_query_u - secondary_u * scaled_query_v
    secondary_numerator = primary_norm * scaled_query_v
    face_inside = (
        (determinant > tiny)
        & (scaled_determinant > 0)
        & (primary_numerator >= 0)
        & (secondary_numerator >= 0)
        & ((primary_numerator + secondary_numerator) <= scaled_determinant)
    )
    face_denominator = torch.where(
        face_inside,
        scaled_determinant,
        torch.ones_like(scaled_determinant),
    )
    primary_weight = primary_numerator / face_denominator
    secondary_weight = secondary_numerator / face_denominator
    face_v = torch.where(primary_is_ab, primary_weight, secondary_weight)
    face_w = torch.where(primary_is_ab, secondary_weight, primary_weight)

    weight_ab = (rhs_ab / gram_ab.clamp_min(tiny)).clamp(0.0, 1.0)
    weight_ac = (rhs_ac / gram_ac.clamp_min(tiny)).clamp(0.0, 1.0)
    gram_bc = (bc_scaled * bc_scaled).sum(dim=-1)
    rhs_bc = _stable_scaled_difference_dot(
        query_points,
        b,
        bc_scaled,
        edge_scale,
    )
    weight_bc = (rhs_bc / gram_bc.clamp_min(tiny)).clamp(0.0, 1.0)

    edge_weights = torch.stack(
        (
            torch.stack((weight_ab, torch.zeros_like(weight_ab)), dim=-1),
            torch.stack((torch.zeros_like(weight_ac), weight_ac), dim=-1),
            torch.stack((1.0 - weight_bc, weight_bc), dim=-1),
        ),
        dim=-2,
    )
    edge_v = edge_weights[..., 0]
    edge_w = edge_weights[..., 1]

    edge_points = edge_v.unsqueeze(-1) * ab_scaled.unsqueeze(-2) + edge_w.unsqueeze(
        -1
    ) * ac_scaled.unsqueeze(-2)
    edge_u = (edge_points * axis_u.unsqueeze(-2)).sum(dim=-1)
    edge_v_coordinate = (edge_points * axis_v.unsqueeze(-2)).sum(dim=-1)
    edge_quadratic = edge_u.square() + edge_v_coordinate.square()
    edge_scores = edge_quadratic * inverse_query_scale.unsqueeze(-1) - 2.0 * (
        edge_u * scaled_query_u.unsqueeze(-1)
        + edge_v_coordinate * scaled_query_v.unsqueeze(-1)
    )
    closest_edge_index = edge_scores.argmin(dim=-1, keepdim=True)
    closest_edge_weights = torch.gather(
        edge_weights,
        dim=-2,
        index=closest_edge_index.unsqueeze(-1).expand(-1, 1, 2),
    ).squeeze(-2)
    closest_weights = torch.where(
        face_inside.unsqueeze(-1),
        torch.stack((face_v, face_w), dim=-1),
        closest_edge_weights,
    )
    closest_v = closest_weights[:, :1]
    closest_w = closest_weights[:, 1:]
    closest_a = 1.0 - closest_v - closest_w
    closest_edge = closest_a * a + closest_v * b + closest_w * c
    closest_face = a + scale_column * (
        query_u.unsqueeze(-1) * axis_u + query_v.unsqueeze(-1) * axis_v
    )
    closest = torch.where(face_inside.unsqueeze(-1), closest_face, closest_edge)
    unscaled_closest = closest / coordinate_scale[:, None]
    return closest + (unscaled_closest - closest).detach()


def _valid_triangles(triangles: torch.Tensor) -> torch.Tensor:
    """Identify finite triangles with non-negligible relative area."""

    scaled_triangles = _scale_triangles_for_safe_edges(triangles)
    a = scaled_triangles[:, 0]
    b = scaled_triangles[:, 1]
    c = scaled_triangles[:, 2]
    ab = b - a
    ac = c - a
    bc = c - b
    edge_scale = torch.stack(
        (
            ab.abs().amax(dim=-1),
            ac.abs().amax(dim=-1),
            bc.abs().amax(dim=-1),
        ),
        dim=-1,
    ).amax(dim=-1)
    safe_scale = torch.where(
        edge_scale > 0,
        edge_scale,
        torch.ones_like(edge_scale),
    )
    scale_column = safe_scale.unsqueeze(-1)
    ab_scaled = ab / scale_column
    ac_scaled = ac / scale_column
    bc_scaled = bc / scale_column
    scale_squared = torch.stack(
        (
            (ab_scaled * ab_scaled).sum(dim=-1),
            (ac_scaled * ac_scaled).sum(dim=-1),
            (bc_scaled * bc_scaled).sum(dim=-1),
        ),
        dim=-1,
    ).amax(dim=-1)
    scaled_cross = torch.linalg.cross(ab_scaled, ac_scaled)
    doubled_area_squared = (scaled_cross * scaled_cross).sum(dim=-1)
    relative_threshold = (torch.finfo(triangles.dtype).eps * scale_squared).square()
    return (
        torch.isfinite(triangles).all(dim=(-2, -1))
        & (edge_scale > 0)
        & (doubled_area_squared > relative_threshold)
    )


@torch.no_grad()
def nearest_surface_faces_torch(
    query_points: torch.Tensor,
    target_points: torch.Tensor,
    target_faces: torch.Tensor,
    max_distance: float,
) -> torch.Tensor:
    """Find nearest target faces with a chunked Torch reference search."""

    num_queries = query_points.shape[0]
    num_faces = target_faces.shape[0]
    if num_queries == 0:
        return torch.empty(0, dtype=torch.long, device=query_points.device)

    triangles = target_points[target_faces.to(torch.long)]
    valid_faces = _valid_triangles(triangles)
    maximum_order_exponent = torch.iinfo(torch.int32).max
    best_exponent = torch.full(
        (num_queries,),
        maximum_order_exponent,
        dtype=torch.int32,
        device=query_points.device,
    )
    best_mantissa = torch.full(
        (num_queries,),
        torch.inf,
        dtype=query_points.dtype,
        device=query_points.device,
    )
    best_faces = torch.full(
        (num_queries,),
        -1,
        dtype=torch.long,
        device=query_points.device,
    )

    face_chunk_size = min(_FACE_CHUNK, num_faces)
    query_chunk_size = max(1, _PAIR_BUDGET // max(face_chunk_size, 1))
    for query_start in range(0, num_queries, query_chunk_size):
        query_end = min(query_start + query_chunk_size, num_queries)
        query_chunk = query_points[query_start:query_end]
        chunk_best_exponent = best_exponent[query_start:query_end]
        chunk_best_mantissa = best_mantissa[query_start:query_end]
        chunk_best_face = best_faces[query_start:query_end]

        for face_start in range(0, num_faces, face_chunk_size):
            face_end = min(face_start + face_chunk_size, num_faces)
            triangle_chunk = triangles[face_start:face_end]
            pair_shape = (query_chunk.shape[0], triangle_chunk.shape[0])
            paired_queries = (
                query_chunk[:, None, :].expand(*pair_shape, 3).reshape(-1, 3)
            )
            paired_triangles = (
                triangle_chunk[None, :, :, :]
                .expand(*pair_shape, 3, 3)
                .reshape(-1, 3, 3)
            )
            closest = closest_points_on_triangles(
                paired_queries,
                paired_triangles,
            ).reshape(*pair_shape, 3)
            (
                distance,
                distance_mantissa,
                distance_exponent,
            ) = _stable_vector_distance_with_key(
                query_chunk[:, None, :],
                closest,
            )
            eligible = valid_faces[face_start:face_end].unsqueeze(0) & (
                distance < max_distance
            )
            ordered_exponent = torch.where(
                eligible,
                distance_exponent,
                torch.full_like(distance_exponent, maximum_order_exponent),
            )
            local_exponent = ordered_exponent.min(dim=1).values
            has_local_exponent = eligible & (
                distance_exponent == local_exponent.unsqueeze(-1)
            )
            ordered_mantissa = torch.where(
                has_local_exponent,
                distance_mantissa,
                torch.full_like(distance_mantissa, torch.inf),
            )
            local_mantissa, local_face = ordered_mantissa.min(dim=1)
            update = (local_exponent < chunk_best_exponent) | (
                (local_exponent == chunk_best_exponent)
                & (local_mantissa < chunk_best_mantissa)
            )
            chunk_best_exponent = torch.where(
                update,
                local_exponent,
                chunk_best_exponent,
            )
            chunk_best_mantissa = torch.where(
                update,
                local_mantissa,
                chunk_best_mantissa,
            )
            chunk_best_face = torch.where(
                update,
                local_face + face_start,
                chunk_best_face,
            )

        best_exponent[query_start:query_end] = chunk_best_exponent
        best_mantissa[query_start:query_end] = chunk_best_mantissa
        best_faces[query_start:query_end] = chunk_best_face

    return best_faces


def replay_shrinkwrap_projection(
    points: torch.Tensor,
    target_points: torch.Tensor,
    target_faces: torch.Tensor,
    nearest_faces: torch.Tensor,
    point_weights: torch.Tensor | None,
    offset: torch.Tensor,
    max_distance: float,
) -> torch.Tensor:
    """Replay selected projections with Torch so autograd sees the geometry."""

    flat_points = points.reshape(-1, 3)
    flat_faces = nearest_faces.reshape(-1)
    hit = flat_faces >= 0
    all_triangles = target_points[target_faces.to(torch.long)]
    fallback_face = _valid_triangles(all_triangles).to(torch.int64).argmax()
    safe_faces = torch.where(hit, flat_faces, fallback_face)
    face_vertices = target_faces.to(torch.long)[safe_faces]
    triangles = target_points[face_vertices]
    closest = closest_points_on_triangles(flat_points, triangles)
    cutoff = torch.tensor(
        max_distance,
        dtype=points.dtype,
        device=points.device,
    )
    hit = hit & (_stable_vector_distance(flat_points, closest) < cutoff)

    scaled_triangles = _scale_triangles_for_safe_edges(triangles)
    edge_ab = scaled_triangles[:, 1] - scaled_triangles[:, 0]
    edge_ac = scaled_triangles[:, 2] - scaled_triangles[:, 0]
    normal_scale = torch.stack(
        (
            edge_ab.abs().amax(dim=-1),
            edge_ac.abs().amax(dim=-1),
        ),
        dim=-1,
    ).amax(dim=-1)
    safe_normal_scale = torch.where(
        normal_scale > 0,
        normal_scale,
        torch.ones_like(normal_scale),
    ).detach()
    normal = torch.linalg.cross(
        edge_ab / safe_normal_scale.unsqueeze(-1),
        edge_ac / safe_normal_scale.unsqueeze(-1),
    )
    normal = normal / torch.linalg.vector_norm(
        normal,
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(points.dtype).tiny)
    projected = closest + offset * normal

    if point_weights is None:
        wrapped = projected
    elif point_weights.dtype == torch.bool:
        wrapped = torch.where(
            point_weights.reshape(-1, 1),
            projected,
            flat_points,
        )
    else:
        weights = point_weights.reshape(-1).to(points.dtype)
        weight_column = weights.unsqueeze(-1)
        wrapped = (1.0 - weight_column) * flat_points + weight_column * projected
    wrapped = torch.where(hit.unsqueeze(-1), wrapped, flat_points)
    return wrapped.reshape_as(points)


__all__ = [
    "closest_points_on_triangles",
    "nearest_surface_faces_torch",
    "replay_shrinkwrap_projection",
]
