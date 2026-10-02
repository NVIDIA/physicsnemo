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

"""Surface-contact adapter; no penalty forces or original-deck equivalence."""

import math

import torch
from torch import nn

from physicsnemo.nn.functional.neighbors.surface_contact import (
    closest_point_facet,
    closest_point_triangle,
    node_triangle_candidate_chunks,
    supported_quad_mask,
)

from .contact import ContactGraph, contact_edge_features, smooth_contact_cutoff


def linear_closest_approach(relative, relative_velocity, horizon):
    """Distance to a transported material point over a causal linear forecast.

    ``relative`` is current source minus query, and ``relative_velocity`` uses
    the same sign. The source's current barycentric weights remain frozen in
    time (but live in autograd). This is NOT the closest point on the evolving
    facet, a time of impact, or a continuous collision detection guarantee.
    Zero relative speed selects time zero without dividing by zero.
    """
    speed_sq = relative_velocity.square().sum(-1)
    denominator = torch.where(speed_sq > 0, speed_sq, torch.ones_like(speed_sq))
    time = (-(relative * relative_velocity).sum(-1) / denominator).clamp(0, horizon)
    return torch.linalg.vector_norm(
        relative + time[:, None] * relative_velocity, dim=-1
    )


class SurfaceContactGraphBuilder(nn.Module):
    """All eligible node--facet pairs with live surface-interpolated messages.

    Broad phase uses a facet BVH and a conservative maximum nodal thickness.
    Narrow phase uses half query thickness + half barycentric facet thickness.
    The latter is a dataset adaptation, NOT a recovered solver contact setting.
    ``activation_distance`` is the positive shell-clearance message band, not
    center distance. The band taper is C1 at zero/band; discrete IDs carry no
    gradients. A direction regularization length bounds its Jacobian near zero.

    All surface nodes are eligible; caller-supplied node/facet exclusions and
    incidence are honored, but no implicit graph-hop/component exclusion occurs.
    Directed latent corrections go to query nodes; this is not a force-balanced
    pair law. Quad sources use TYPE7's arithmetic-center fan representation.

    Broad-phase and membership geometry are streamed in ``candidate_chunk_size``
    pages. ``max_pairs`` limits the final ACTIVE graph, not AABB false positives.
    Exceeding that safety budget raises; no contact is truncated. Live geometry
    is recomputed in canonical node/facet order after membership selection.
    """

    def __init__(
        self,
        *,
        activation_distance=5.0,
        feature_scale=10.0,
        velocity_scale=1000.0,
        normal_epsilon=0.1,
        include_velocity=True,
        implementation="warp",
        max_pairs=2_000_000,
        prediction_horizon=0.0,
        material_fan=False,
        candidate_chunk_size=65_536,
    ):
        super().__init__()
        for name, value in (
            ("activation_distance", activation_distance),
            ("feature_scale", feature_scale),
            ("velocity_scale", velocity_scale),
            ("normal_epsilon", normal_epsilon),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if implementation not in ("torch", "warp"):
            raise ValueError("surface implementation must be 'torch' or 'warp'")
        if (
            not isinstance(max_pairs, int)
            or isinstance(max_pairs, bool)
            or max_pairs <= 0
        ):
            raise ValueError("max_pairs must be a positive integer")
        self.activation_distance = float(activation_distance)
        self.feature_scale = float(feature_scale)
        self.velocity_scale = float(velocity_scale)
        self.normal_epsilon = float(normal_epsilon)
        self.include_velocity = bool(include_velocity)
        self.implementation = implementation
        self.max_pairs = max_pairs
        if (
            not isinstance(candidate_chunk_size, int)
            or isinstance(candidate_chunk_size, bool)
            or candidate_chunk_size <= 0
        ):
            raise ValueError("candidate_chunk_size must be a positive integer")
        self.candidate_chunk_size = candidate_chunk_size
        self.last_discovery_stats = None
        if not math.isfinite(prediction_horizon) or prediction_horizon < 0:
            raise ValueError("prediction_horizon must be finite and nonnegative")
        self.prediction_horizon = float(prediction_horizon)
        self.material_fan = bool(material_fan)

    def _activation_gap(
        self, positions, velocities, thickness, query, source, projection
    ):
        surface_thickness = (thickness[source] * projection.barycentric).sum(-1)
        half_thickness = (thickness[query] + surface_thickness) * 0.5
        if self.prediction_horizon == 0:
            return projection.distance - half_thickness
        relative = projection.closest - positions[query]
        relative_velocity = (
            velocities[source] * projection.barycentric[..., None]
        ).sum(1) - velocities[query]
        distance = linear_closest_approach(
            relative, relative_velocity, self.prediction_horizon
        )
        gap = torch.minimum(distance, projection.distance) - half_thickness
        # The nearest material point can slide along a face/edge. Add the point
        # closest at the forecast endpoint as a second anchor; analytically sweep
        # BOTH material points over the entire interval. This catches crossings
        # between samples, but two anchors are still not full moving-facet CCD.
        future = closest_point_facet(
            positions[query] + self.prediction_horizon * velocities[query],
            positions[source] + self.prediction_horizon * velocities[source],
            triangle_mask=(source[:, 2] == source[:, 3])
            if source.shape[1] == 4
            else None,
            validate_quads=False,  # full current/forecast facet sets checked below
            material_fan=self.material_fan,
        )
        future_relative = (positions[source] * future.barycentric[..., None]).sum(
            1
        ) - positions[query]
        future_velocity = (velocities[source] * future.barycentric[..., None]).sum(
            1
        ) - velocities[query]
        future_distance = linear_closest_approach(
            future_relative, future_velocity, self.prediction_horizon
        )
        future_thickness = (thickness[source] * future.barycentric).sum(-1)
        future_gap = future_distance - (thickness[query] + future_thickness) * 0.5
        return torch.minimum(gap, future_gap)

    def forward(
        self,
        positions,
        *,
        faces,
        shell_thickness,
        velocities=None,
        batch=None,
        excluded_pairs=None,
        structural_edge_index=None,
        reference_positions=None,
    ):
        # structural_edge_index is accepted for the shared recipe call, but is
        # deliberately NOT used as a blanket geodesic exclusion.
        n = len(positions)
        if shell_thickness is None or shell_thickness.shape != (n,):
            raise ValueError(
                "surface contact requires supplied physical nodal thickness [N]"
            )
        thickness = shell_thickness.to(positions)
        if not torch.isfinite(thickness).all() or torch.any(thickness < 0):
            raise ValueError("shell thickness must be finite and nonnegative")
        if faces is None:
            raise ValueError(
                "surface contact requires graph.contact_faces element connectivity"
            )
        if self.include_velocity or self.prediction_horizon > 0:
            if (
                velocities is None
                or velocities.shape != positions.shape
                or velocities.dtype != positions.dtype
                or velocities.device != positions.device
                or not torch.isfinite(velocities).all()
            ):
                raise ValueError("live physical velocities must match positions")
        # Validate connectivity before using it to index thickness.
        if (
            faces.ndim != 2
            or faces.shape[1] not in (3, 4)
            or faces.dtype != torch.long
            or faces.device != positions.device
            or torch.any(faces < 0)
            or torch.any(faces >= n)
        ):
            raise ValueError("invalid surface connectivity")
        # Strict polygon mode validates both live configurations. Material mode
        # instead transports a fixed reference fan, whose union remains defined
        # after folding/collapse. Its reference MUST still be valid, and may not
        # be substituted by future ground truth or an unchecked live state.
        if self.material_fan:
            if (
                reference_positions is None
                or reference_positions.shape != positions.shape
                or reference_positions.dtype != positions.dtype
                or reference_positions.device != positions.device
                or not torch.isfinite(reference_positions).all()
            ):
                raise ValueError(
                    "material fan requires finite, matched reference_positions"
                )
            geometries = [("reference", reference_positions)]
            with torch.no_grad():
                tri_ids = (
                    (faces[:, 2] == faces[:, 3]).nonzero().flatten()
                    if faces.shape[1] == 4
                    else torch.arange(len(faces), device=faces.device)
                )
                for ids in tri_ids.split(16384):
                    tri = reference_positions[faces[ids, :3]]
                    closest_point_triangle(tri[:, 0], tri)
        else:
            geometries = [("current", positions)]
            if self.prediction_horizon > 0:
                geometries.append(
                    ("forecast", positions + self.prediction_horizon * velocities)
                )
        # Validate every reference/strict quad, including distant facets.
        if faces.shape[1] == 4:
            with torch.no_grad():
                quad_ids = (faces[:, 2] != faces[:, 3]).nonzero().flatten()
                for label, geometry in geometries:
                    for ids in quad_ids.split(16384):
                        valid = supported_quad_mask(geometry[faces[ids]])
                        if not valid.all():
                            bad = ids[~valid][:8].cpu().tolist()
                            raise ValueError(
                                "unsupported surface quad: overlapping, folded, or degenerate center fan; "
                                f"{label} facet IDs (first 8): {bad}"
                            )
        node_pad = thickness * 0.5 + self.activation_distance
        face_pad = thickness[faces].amax(1) * 0.5
        search_end = None
        if self.prediction_horizon > 0:
            with torch.no_grad():
                # A common Galilean translation preserves relative geometry at
                # every forecast time, avoiding huge boxes from rigid car motion.
                # One global reference also works for batched graphs; cross-batch
                # exclusion remains the candidate functional's responsibility.
                reference = velocities.mean(0) if n else velocities.new_zeros(3)
                search_end = positions + self.prediction_horizon * (
                    velocities - reference
                )
        chunks = node_triangle_candidate_chunks(
            positions,
            faces,
            node_pad,
            face_pad,
            batch=batch,
            previous_positions=search_end,
            excluded_pairs=excluded_pairs,
            implementation=self.implementation,
            pair_chunk_size=self.candidate_chunk_size,
        )
        # The first projection selects membership only. Do not retain a large
        # backward graph for broad-phase false positives. Recompute LIVE
        # geometry for the selected pairs, preserving all geometric gradients.
        # Keep only selected IDs between pages; never collect all raw pairs.
        self.last_discovery_stats = None
        with torch.no_grad():
            selected = []
            candidates = active = pages = largest_page = 0
            for pairs in chunks:
                pages += 1
                candidates += pairs.shape[1]
                largest_page = max(largest_page, pairs.shape[1])
                query, face_id = pairs
                source_nodes = faces[face_id]
                discovery = closest_point_facet(
                    positions[query],
                    positions[source_nodes],
                    triangle_mask=(source_nodes[:, 2] == source_nodes[:, 3])
                    if faces.shape[1] == 4
                    else None,
                    validate_quads=False,
                    material_fan=self.material_fan,
                )
                candidate_gap = self._activation_gap(
                    positions, velocities, thickness, query, source_nodes, discovery
                )
                kept = pairs[:, candidate_gap < self.activation_distance]
                active += kept.shape[1]
                if active > self.max_pairs:
                    raise RuntimeError(
                        f"surface active contact budget exceeded ({active}>{self.max_pairs}); "
                        "no pairs truncated"
                    )
                if kept.shape[1]:
                    selected.append(kept)
                del discovery, candidate_gap, source_nodes, kept, query, face_id, pairs
            pairs = torch.cat(selected, 1) if selected else faces.new_empty((2, 0))
            del selected, chunks
            order = torch.argsort(pairs[0] * len(faces) + pairs[1], stable=True)
            pairs = pairs[:, order]
            query, face_id = pairs
            source_nodes = faces[face_id]
            self.last_discovery_stats = {
                "candidates": candidates,
                "active_pairs": active,
                "pages": pages,
                "largest_page": largest_page,
            }
        projection = closest_point_facet(
            positions[query],
            positions[source_nodes],
            triangle_mask=(source_nodes[:, 2] == source_nodes[:, 3])
            if faces.shape[1] == 4
            else None,
            validate_quads=False,
            material_fan=self.material_fan,
        )
        source_thickness = (thickness[source_nodes] * projection.barycentric).sum(-1)
        gap = projection.distance - (thickness[query] + source_thickness) * 0.5
        activation_gap = self._activation_gap(
            positions, velocities, thickness, query, source_nodes, projection
        )
        # Features describe CURRENT geometry. Only discovery and the cutoff
        # use predicted closest approach, not a fabricated physical penetration.
        weights = projection.barycentric
        relative = projection.closest - positions[query]
        relative_velocity = None
        if self.include_velocity:
            relative_velocity = (velocities[source_nodes] * weights[..., None]).sum(
                1
            ) - velocities[query]
        features = contact_edge_features(
            relative,
            gap,
            self.feature_scale,
            relative_velocity=relative_velocity,
            velocity_scale=self.velocity_scale,
            normal_epsilon=self.normal_epsilon,
        )
        return ContactGraph(
            edge_index=torch.stack((source_nodes[:, 0], query)),
            edge_features=features,
            obstacle_mask=torch.zeros(
                len(query), dtype=torch.bool, device=positions.device
            ),
            edge_weights=smooth_contact_cutoff(
                activation_gap, self.activation_distance
            ),
            source_nodes=source_nodes,
            source_weights=weights,
        )
