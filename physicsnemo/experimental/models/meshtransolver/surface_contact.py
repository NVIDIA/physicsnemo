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

    ``narrow_phase="auto"`` fuses detached membership on CUDA FP32 with Warp
    discovery; other supported geometry uses the Torch reference. Ambiguous
    numerical cases are decided by Torch in bounded fallback pages. Set
    ``narrow_phase="torch"`` to retain reference membership with the same BVH.
    The fused kernel never supplies learned features or their gradients.

    ``area_weighting=True`` multiplies each active pair's taper weight by the
    face area divided by ``reference_area`` (default: the activation band disk,
    ``pi * activation_distance**2``). This first-order surface quadrature makes
    the aggregate and the activation total approximately independent of mesh
    resolution: a finely meshed panel no longer counts more than a coarse one.
    Areas are the center-fan areas of the reference geometry when supplied and
    of the detached current geometry otherwise; the factor carries no gradient.
    Default ``False`` keeps the validated recipe unchanged.

    ``side_from_reference=True`` compares, for every active pair, the side of
    the facet's oriented plane the node is on now with the side it was on in
    the reference configuration (``reference_positions``). A node that has
    crossed to the other side gets a negative signed distance, so the signed
    gap reports the crossing depth below minus the half thickness and the
    direction channels flip; the pair is fully active. Without it, a node that
    has passed through a shell is indistinguishable from one approaching the
    far side, because the node-to-facet distance is unsigned. A crossing is
    only declared when the node now projects onto the facet's interior (its
    offset from the closest point is normal to the facet, within 2%): a node
    beside a facet, whose closest point lies on an edge or a vertex, may change
    plane side by sliding past the edge without passing through anything. The
    side test is discrete (no gradient) and inconclusive (unsigned) within a
    thousandth of the distance of the facet plane or on a degenerate facet.
    Crossings deeper than the activation band are not discovered, like any far
    pair, unless ``crossing_radius`` is set.

    ``side_symmetric=True`` asks the reference configuration for the same
    interior projection as the current one before a crossing is declared: a
    node that was beside the facet (closest point on an edge or a vertex) in
    the reference and lies under its interior now has slid under the edge,
    not passed through the shell, and stays an ordinary pair with its unsigned
    gap. The default keeps the one-sided test above.

    ``crossing_radius`` (physical length, larger than the activation band)
    adds a crossing sweep: the broad phase is padded to it, and a candidate
    whose unsigned activation gap lies between the band and the radius is
    kept only when the side test certifies a crossing. Such a pair is fully
    active like every crossing; approaching pairs beyond the band are still
    excluded, so the sweep lifts the depth ceiling ``band + half thickness``
    without widening the approach neighbourhood. Membership certification is
    detached; live geometry is recomputed as for every pair.

    ``clearance_approach_only=True`` applies ``reference_clearance`` to
    approaching pairs only: a certified crossing of two panels that were
    assembled within the clearance is kept, since a shell passing through a
    neighbouring shell is new contact whatever their assembled distance. It
    has no effect without a clearance. All three corrections need
    ``side_from_reference``.

    ``reference_clearance`` (physical length) drops active pairs whose
    activation gap in the reference configuration is below it: in-plane
    neighbours beyond the one ring, bends and flanges that are assembled
    close together and already lie inside the structural neighbourhood of
    the mesh operators. What remains is new contact: pairs that were apart in
    the reference and are close now. ``last_discovery_stats`` reports the
    excluded and crossed pair counts.
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
        narrow_phase="auto",
        area_weighting=False,
        reference_area=None,
        side_from_reference=False,
        reference_clearance=None,
        side_symmetric=False,
        crossing_radius=None,
        clearance_approach_only=False,
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
        if narrow_phase not in ("auto", "torch", "warp"):
            raise ValueError("narrow_phase must be auto, torch or warp")
        self.narrow_phase = narrow_phase
        self.area_weighting = bool(area_weighting)
        if reference_area is not None and (
            not math.isfinite(reference_area) or reference_area <= 0
        ):
            raise ValueError("reference_area must be positive and finite")
        self.reference_area = None if reference_area is None else float(reference_area)
        self.side_from_reference = bool(side_from_reference)
        if reference_clearance is not None and (
            not math.isfinite(reference_clearance) or reference_clearance < 0
        ):
            raise ValueError("reference_clearance must be finite and nonnegative")
        self.reference_clearance = (
            None if reference_clearance is None else float(reference_clearance)
        )
        self.side_symmetric = bool(side_symmetric)
        if self.side_symmetric and not self.side_from_reference:
            raise ValueError("side_symmetric needs side_from_reference=True")
        if crossing_radius is not None:
            if not math.isfinite(crossing_radius) or not (
                crossing_radius > self.activation_distance
            ):
                raise ValueError(
                    "crossing_radius must be finite and larger than activation_distance"
                )
            if not self.side_from_reference:
                raise ValueError("crossing_radius needs side_from_reference=True")
            if self.prediction_horizon > 0:
                raise ValueError("crossing_radius needs prediction_horizon=0")
        self.crossing_radius = None if crossing_radius is None else float(crossing_radius)
        self.clearance_approach_only = bool(clearance_approach_only)
        if self.clearance_approach_only and not self.side_from_reference:
            raise ValueError("clearance_approach_only needs side_from_reference=True")

    def _face_area_quadrature(self, geometry, faces, face_id):
        """Detached center-fan face area over the reference area, per pair."""
        with torch.no_grad():
            vertices = geometry.detach()[faces]
            if faces.shape[1] == 3:
                area = 0.5 * torch.linalg.vector_norm(
                    torch.linalg.cross(
                        vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
                    ),
                    dim=-1,
                )
            else:
                # Same center fan as closest_point_facet; a padded triangle's
                # fan sums to the triangle area because its center lies inside.
                radial = vertices - vertices.mean(1, keepdim=True)
                fan = torch.linalg.cross(radial, radial.roll(-1, 1))
                area = 0.5 * torch.linalg.vector_norm(fan, dim=-1).sum(1)
            reference = (
                self.reference_area
                if self.reference_area is not None
                else math.pi * self.activation_distance**2
            )
            return (area / reference).index_select(0, face_id).to(geometry.dtype)

    def _reference_projection(self, reference_positions, thickness, query, source):
        """Detached reference closest point and activation gap of given pairs."""
        with torch.no_grad():
            projection = closest_point_facet(
                reference_positions[query],
                reference_positions[source],
                triangle_mask=(source[:, 2] == source[:, 3])
                if source.shape[1] == 4
                else None,
                validate_quads=False,
                material_fan=self.material_fan,
            )
            surface_thickness = (thickness[source] * projection.barycentric).sum(-1)
            gap = projection.distance - (thickness[query] + surface_thickness) * 0.5
        return gap, projection

    @staticmethod
    def _facet_side(points, closest, normal, distance, interior_only=False):
        """-1, 0 or +1: side of the facet's oriented plane, 0 when inconclusive.

        With ``interior_only`` the side is also 0 unless the closest point is
        an interior projection (offset parallel to the normal within 2%), i.e.
        the node is over the facet rather than beside one of its edges.
        """
        signed = ((points - closest) * normal).sum(-1)
        inconclusive = signed.abs() <= 1e-3 * distance
        if interior_only:
            inconclusive = inconclusive | (signed.abs() < 0.98 * distance)
        return torch.where(inconclusive, torch.zeros_like(signed), torch.sign(signed))

    def _use_fused_membership(self, positions):
        return self.narrow_phase == "warp" or (
            self.narrow_phase == "auto"
            and self.implementation == "warp"
            and positions.is_cuda
            and positions.dtype == torch.float32
        )

    def _fused_candidate_gaps(self, positions, velocities, thickness, faces, pairs):
        from physicsnemo.nn.functional.neighbors.surface_contact_narrow_warp import (
            fused_activation_gaps,
        )

        return fused_activation_gaps(
            positions,
            velocities,
            thickness,
            faces,
            pairs,
            self.prediction_horizon,
            self.activation_distance,
            self.crossing_radius,
        )

    def _crossed(self, points, projection, reference_points, reference_projection):
        """Discrete crossing flag of pairs from their current and reference projections."""
        with torch.no_grad():
            before = self._facet_side(
                reference_points,
                reference_projection.closest,
                reference_projection.normal,
                reference_projection.distance,
                interior_only=self.side_symmetric,
            )
            now = self._facet_side(
                points.detach(),
                projection.closest.detach(),
                projection.normal.detach(),
                projection.distance.detach(),
                interior_only=True,
            )
            return before * now < 0

    def _certify_crossed(self, positions, reference_positions, thickness, faces, pairs):
        """Detached crossing certification of candidate pairs for membership.

        The same side test as the live stage, on the same detached inputs, so
        a pair certified here is flagged crossed there too.
        """
        query, face_id = pairs
        source = faces[face_id]
        with torch.no_grad():
            projection = closest_point_facet(
                positions[query],
                positions[source],
                triangle_mask=(source[:, 2] == source[:, 3])
                if faces.shape[1] == 4
                else None,
                validate_quads=False,
                material_fan=self.material_fan,
            )
            _, reference_projection = self._reference_projection(
                reference_positions, thickness, query, source
            )
            return self._crossed(
                positions[query], projection, reference_positions[query], reference_projection
            )

    def _reference_candidate_gaps(self, positions, velocities, thickness, faces, pairs):
        query, face_id = pairs
        source = faces[face_id]
        projection = closest_point_facet(
            positions[query],
            positions[source],
            triangle_mask=(source[:, 2] == source[:, 3])
            if faces.shape[1] == 4
            else None,
            validate_quads=False,
            material_fan=self.material_fan,
        )
        return self._activation_gap(
            positions, velocities, thickness, query, source, projection
        )

    def _candidate_gaps(self, positions, velocities, thickness, faces, pairs):
        """Discrete membership only; live outputs are always recomputed in Torch."""
        if not self._use_fused_membership(positions):
            return self._reference_candidate_gaps(
                positions, velocities, thickness, faces, pairs
            ), 0
        gaps, ambiguous = self._fused_candidate_gaps(
            positions, velocities, thickness, faces, pairs
        )
        reference_ids = ambiguous.nonzero().flatten()
        if not len(reference_ids):
            return gaps, 0
        reference_gap = self._reference_candidate_gaps(
            positions, velocities, thickness, faces, pairs[:, reference_ids]
        )
        gaps.index_copy_(0, reference_ids, reference_gap)
        return gaps, len(reference_ids)

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
        uses_reference = (
            self.side_from_reference or self.reference_clearance is not None
        )
        if (self.material_fan or uses_reference) and (
            reference_positions is None
            or reference_positions.shape != positions.shape
            or reference_positions.dtype != positions.dtype
            or reference_positions.device != positions.device
            or not torch.isfinite(reference_positions).all()
        ):
            raise ValueError(
                "material fan, side_from_reference and reference_clearance require "
                "finite, matched reference_positions"
            )
        if self.material_fan:
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
        search_band = (
            self.activation_distance
            if self.crossing_radius is None
            else max(self.activation_distance, self.crossing_radius)
        )
        node_pad = thickness * 0.5 + search_band
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
            candidates = active = pages = largest_page = fallback_pairs = 0
            reference_pages = largest_reference_page = 0
            reference_excluded = sweep_pairs = clearance_crossings_kept = 0
            pending = []
            pending_count = 0
            use_fused = self._use_fused_membership(positions)

            def select_active(pair_ids, gaps):
                nonlocal active, reference_excluded, sweep_pairs
                nonlocal clearance_crossings_kept
                near = gaps < self.activation_distance
                from_sweep = None
                if self.crossing_radius is not None:
                    # Crossing sweep: candidates beyond the band but within the
                    # radius are kept only when certified crossed below.
                    from_sweep = ~near & (gaps < self.crossing_radius)
                    keep = near | from_sweep
                    kept = pair_ids[:, keep]
                    from_sweep = from_sweep[keep]
                else:
                    kept = pair_ids[:, near]
                passes = None
                if self.reference_clearance is not None and kept.shape[1]:
                    # Same discrete narrow phase on the reference configuration
                    # (fused kernel where available, Torch reference otherwise).
                    reference_gap, _ = self._candidate_gaps(
                        reference_positions, None, thickness, faces, kept
                    )
                    passes = reference_gap >= self.reference_clearance
                needs_certificate = None
                if from_sweep is not None:
                    needs_certificate = from_sweep
                if passes is not None and self.clearance_approach_only:
                    needs_certificate = (
                        ~passes
                        if needs_certificate is None
                        else needs_certificate | ~passes
                    )
                if needs_certificate is not None and bool(needs_certificate.any()):
                    crossed = torch.zeros_like(needs_certificate)
                    ids = needs_certificate.nonzero().flatten()
                    crossed[ids] = self._certify_crossed(
                        positions, reference_positions, thickness, faces, kept[:, ids]
                    )
                    accept = torch.ones_like(crossed)
                    if from_sweep is not None:
                        accept &= ~from_sweep | crossed
                        sweep_pairs += int((from_sweep & crossed).sum())
                    if passes is not None:
                        exempt = ~passes & crossed if self.clearance_approach_only else None
                        if exempt is not None:
                            clearance_crossings_kept += int(exempt.sum())
                            accept &= passes | exempt
                            reference_excluded += int((~passes & ~crossed).sum())
                        else:
                            accept &= passes
                            reference_excluded += kept.shape[1] - int(passes.sum())
                    kept = kept[:, accept]
                elif passes is not None:
                    reference_excluded += kept.shape[1] - int(passes.sum())
                    kept = kept[:, passes]
                active += kept.shape[1]
                if active > self.max_pairs:
                    raise RuntimeError(
                        f"surface active contact budget exceeded ({active}>{self.max_pairs}); "
                        "no pairs truncated"
                    )
                if kept.shape[1]:
                    selected.append(kept)

            def select_reference(pair_ids):
                nonlocal reference_pages, largest_reference_page
                reference_pages += 1
                largest_reference_page = max(largest_reference_page, pair_ids.shape[1])
                gaps = self._reference_candidate_gaps(
                    positions, velocities, thickness, faces, pair_ids
                )
                select_active(pair_ids, gaps)

            for pairs in chunks:
                pages += 1
                candidates += pairs.shape[1]
                largest_page = max(largest_page, pairs.shape[1])
                if use_fused:
                    candidate_gap, ambiguous = self._fused_candidate_gaps(
                        positions, velocities, thickness, faces, pairs
                    )
                    reference_pairs = pairs[:, ambiguous]
                    count = reference_pairs.shape[1]
                    fallback_pairs += count
                    # Do not launch the full reference geometry for a handful
                    # of ambiguous pairs on EVERY broad-phase page. Collect
                    # discrete IDs into bounded pages; no pair is dropped and
                    # reference geometry still has <= candidate_chunk_size rows.
                    if count:
                        pending.append(reference_pairs)
                        pending_count += count
                    candidate_gap.masked_fill_(ambiguous, float("inf"))
                    select_active(pairs, candidate_gap)
                    if pending_count >= self.candidate_chunk_size:
                        combined = torch.cat(pending, 1)
                        full = pending_count // self.candidate_chunk_size
                        end = full * self.candidate_chunk_size
                        for page in combined[:, :end].split(
                            self.candidate_chunk_size, 1
                        ):
                            select_reference(page)
                        remainder = combined[:, end:]
                        pending_count = remainder.shape[1]
                        pending = [remainder] if pending_count else []
                        del combined, remainder
                    del ambiguous, reference_pairs
                else:
                    candidate_gap = self._reference_candidate_gaps(
                        positions, velocities, thickness, faces, pairs
                    )
                    select_active(pairs, candidate_gap)
                del candidate_gap, pairs
            if pending_count:
                select_reference(torch.cat(pending, 1))
            del pending
            pairs = torch.cat(selected, 1) if selected else faces.new_empty((2, 0))
            selected.clear()
            del chunks
            order = torch.argsort(pairs[0] * len(faces) + pairs[1], stable=True)
            pairs = pairs[:, order]
            query, face_id = pairs
            source_nodes = faces[face_id]
            self.last_discovery_stats = {
                "candidates": candidates,
                "active_pairs": active,
                "pages": pages,
                "largest_page": largest_page,
                "narrow_phase_fallback_pairs": fallback_pairs,
                "narrow_phase_reference_pages": reference_pages,
                "largest_reference_page": largest_reference_page,
                "reference_excluded_pairs": reference_excluded,
                "sweep_pairs": sweep_pairs,
                "clearance_crossings_kept": clearance_crossings_kept,
                "crossed_pairs": 0,
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
        half_thickness = (thickness[query] + source_thickness) * 0.5
        gap = projection.distance - half_thickness
        activation_gap = self._activation_gap(
            positions, velocities, thickness, query, source_nodes, projection
        )
        # Features describe CURRENT geometry. Only discovery and the cutoff
        # use predicted closest approach, not a fabricated physical penetration.
        weights = projection.barycentric
        relative = projection.closest - positions[query]
        contact_normal = None
        if self.side_from_reference:
            # A node that changed side of the facet plane since the reference
            # configuration has crossed the shell: its distance counts as
            # negative, the direction flips, and the pair is fully active.
            _, reference_projection = self._reference_projection(
                reference_positions, thickness, query, source_nodes
            )
            with torch.no_grad():
                crossed = self._crossed(
                    positions[query], projection, reference_positions[query], reference_projection
                )
                side = torch.where(crossed, -1.0, 1.0).to(positions.dtype)
            gap = side * projection.distance - half_thickness
            activation_gap = torch.where(crossed, gap, activation_gap)
            contact_normal = (
                relative
                / (
                    relative.square().sum(-1, keepdim=True) + self.normal_epsilon**2
                ).sqrt()
                * side[:, None]
            )
            self.last_discovery_stats["crossed_pairs"] = int(crossed.sum())
        relative_velocity = None
        if self.include_velocity:
            relative_velocity = (velocities[source_nodes] * weights[..., None]).sum(
                1
            ) - velocities[query]
        features = contact_edge_features(
            relative,
            gap,
            self.feature_scale,
            contact_normal=contact_normal,
            relative_velocity=relative_velocity,
            velocity_scale=self.velocity_scale,
            normal_epsilon=self.normal_epsilon,
        )
        edge_weights = smooth_contact_cutoff(activation_gap, self.activation_distance)
        if self.area_weighting:
            edge_weights = edge_weights * self._face_area_quadrature(
                positions if reference_positions is None else reference_positions,
                faces,
                face_id,
            )
        return ContactGraph(
            edge_index=torch.stack((source_nodes[:, 0], query)),
            edge_features=features,
            obstacle_mask=torch.zeros(
                len(query), dtype=torch.bool, device=positions.device
            ),
            edge_weights=edge_weights,
            source_nodes=source_nodes,
            source_weights=weights,
        )
