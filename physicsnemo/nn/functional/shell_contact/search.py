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
"""Node-to-facet contact discovery on shell meshes.

Discovery finds every node/facet pair whose shell clearance (distance to the
facet minus the half thicknesses of node and facet) lies inside an activation
band. The broad phase queries a facet BVH with conservative paddings in
bounded pages; the narrow phase decides membership from detached geometry
(a fused Warp kernel on CUDA float32, the Torch reference otherwise, with
numerically ambiguous pairs recomputed by the reference); the live geometry
of the selected pairs is then recomputed once in canonical order so that
every returned quantity carries autograd through the node positions.

Optional corrections make the search crossing-aware with respect to a
reference configuration: a node that has passed through a shell since the
reference gets a negative signed gap, pairs that were already assembled
within a clearance in the reference are dropped as structure rather than
contact, and a crossing sweep discovers crossings deeper than the band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from physicsnemo.nn.functional.neighbors.surface_contact import (
    TriangleProjection,
    closest_point_triangle,
    node_triangle_candidate_chunks,
    supported_quad_mask,
)

from .geometry import (
    crossing_flags,
    project_onto_facets,
    reference_facet_projection,
)


@dataclass(frozen=True)
class ShellContactSearch:
    r"""Settings of :func:`shell_contact_pairs`.

    Parameters
    ----------
    activation_distance : float
        Positive shell-clearance band: a pair is active while its activation
        gap lies below it (and the taper of
        :func:`~physicsnemo.nn.functional.shell_contact.smooth_contact_cutoff`
        reaches zero there). Physical length.
    implementation : {"warp", "torch"}
        Broad-phase backend: a Warp BVH (float32 positions only; runs on CPU
        and CUDA) or the chunked Torch reference.
    narrow_phase : {"auto", "warp", "torch"}
        Membership geometry: ``"auto"`` uses the fused Warp kernel on CUDA
        float32 positions when ``implementation`` is Warp and the Torch
        reference otherwise; the explicit values force one of them.
    max_pairs : int
        Safety budget for the number of ACTIVE pairs. Exceeding it raises; no
        pair is ever truncated.
    candidate_chunk_size : int
        Page size of the streamed broad-phase candidates.
    material_fan : bool
        Transport the reference centre fan of four-node facets with the
        vertices (the material surface contract) instead of validating the
        live quad shapes. Requires ``reference_positions``.
    prediction_horizon : float
        Nonnegative causal forecast time: with a positive horizon the
        activation gap is the closest approach of a linear forecast over the
        horizon (two anchored material points), which needs ``velocities``.
    side_from_reference : bool
        Sign the gap of pairs whose node has crossed the facet plane since the
        reference configuration (negative, with the normal flipped), so that a
        node that has passed through a shell is distinguishable from one
        approaching its far side. Needs ``reference_positions``.
    reference_clearance : float, optional
        Drop pairs whose activation gap in the reference configuration is
        below this length: flanges and in-plane neighbours that are assembled
        close together are structure, what remains is new contact. The
        reference gap is always the static one (zero horizon), so the option
        combines freely with ``prediction_horizon``.
    side_symmetric : bool
        Declare a crossing only when the reference configuration shows an
        interior projection as well (a node that was beside the facet and
        lies under its interior now has slid under an edge).
    crossing_radius : float, optional
        Crossing sweep: candidates with an unsigned gap between the band and
        this radius are kept when the side test certifies a crossing, so the
        depth ceiling ``band + half thickness`` is lifted without widening the
        approach neighbourhood. Needs ``side_from_reference`` and a zero
        horizon.
    clearance_approach_only : bool
        Apply ``reference_clearance`` to approaching pairs only: a certified
        crossing of two assembled panels is kept.
    """

    activation_distance: float = 5.0
    implementation: str = "warp"
    narrow_phase: str = "auto"
    max_pairs: int = 2_000_000
    candidate_chunk_size: int = 65_536
    material_fan: bool = False
    prediction_horizon: float = 0.0
    side_from_reference: bool = False
    reference_clearance: float | None = None
    side_symmetric: bool = False
    crossing_radius: float | None = None
    clearance_approach_only: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.activation_distance) or self.activation_distance <= 0:
            raise ValueError("activation_distance must be positive and finite")
        if self.implementation not in ("torch", "warp"):
            raise ValueError("surface implementation must be 'torch' or 'warp'")
        if self.narrow_phase not in ("auto", "torch", "warp"):
            raise ValueError("narrow_phase must be auto, torch or warp")
        if (
            not isinstance(self.max_pairs, int)
            or isinstance(self.max_pairs, bool)
            or self.max_pairs <= 0
        ):
            raise ValueError("max_pairs must be a positive integer")
        if (
            not isinstance(self.candidate_chunk_size, int)
            or isinstance(self.candidate_chunk_size, bool)
            or self.candidate_chunk_size <= 0
        ):
            raise ValueError("candidate_chunk_size must be a positive integer")
        if not math.isfinite(self.prediction_horizon) or self.prediction_horizon < 0:
            raise ValueError("prediction_horizon must be finite and nonnegative")
        if self.reference_clearance is not None and (
            not math.isfinite(self.reference_clearance) or self.reference_clearance < 0
        ):
            raise ValueError("reference_clearance must be finite and nonnegative")
        if self.side_symmetric and not self.side_from_reference:
            raise ValueError("side_symmetric needs side_from_reference=True")
        if self.crossing_radius is not None:
            if not math.isfinite(self.crossing_radius) or not (
                self.crossing_radius > self.activation_distance
            ):
                raise ValueError(
                    "crossing_radius must be finite and larger than activation_distance"
                )
            if not self.side_from_reference:
                raise ValueError("crossing_radius needs side_from_reference=True")
            if self.prediction_horizon > 0:
                raise ValueError("crossing_radius needs prediction_horizon=0")
        if self.clearance_approach_only and not self.side_from_reference:
            raise ValueError("clearance_approach_only needs side_from_reference=True")
        object.__setattr__(self, "activation_distance", float(self.activation_distance))
        object.__setattr__(self, "prediction_horizon", float(self.prediction_horizon))
        object.__setattr__(self, "material_fan", bool(self.material_fan))
        object.__setattr__(self, "side_from_reference", bool(self.side_from_reference))
        object.__setattr__(self, "side_symmetric", bool(self.side_symmetric))
        object.__setattr__(
            self, "clearance_approach_only", bool(self.clearance_approach_only)
        )
        if self.reference_clearance is not None:
            object.__setattr__(
                self, "reference_clearance", float(self.reference_clearance)
            )
        if self.crossing_radius is not None:
            object.__setattr__(self, "crossing_radius", float(self.crossing_radius))

    @property
    def search_band(self) -> float:
        """Broad-phase padding beyond the half thickness: the band or the sweep radius."""
        if self.crossing_radius is None:
            return self.activation_distance
        return max(self.activation_distance, self.crossing_radius)

    @property
    def uses_reference(self) -> bool:
        """Whether the search needs the reference configuration."""
        return self.side_from_reference or self.reference_clearance is not None


@dataclass(frozen=True)
class ShellContactPairs:
    r"""Active node/facet pairs with their live geometry and discovery statistics.

    Pairs are sorted by ``(query, face)``. Every tensor has one row per pair.

    Attributes
    ----------
    query : torch.Tensor
        ``[E]`` node receiving the contact.
    face : torch.Tensor
        ``[E]`` facet index into the ``faces`` array given to the search.
    source_nodes : torch.Tensor
        ``[E, 3|4]`` vertex indices of the facet (padded triangles repeat the
        third vertex).
    barycentric : torch.Tensor
        ``[E, 3|4]`` live interpolation weights of the closest point over the
        facet vertices (a padded slot carries zero weight).
    closest : torch.Tensor
        ``[E, 3]`` live closest point on the facet.
    distance : torch.Tensor
        ``[E]`` live unsigned distance from the node to the facet.
    half_thickness : torch.Tensor
        ``[E]`` half of the node thickness plus the interpolated facet thickness.
    gap : torch.Tensor
        ``[E]`` signed shell clearance ``s * distance - half_thickness`` with
        ``s = -1`` for a crossed pair and ``+1`` otherwise.
    activation_gap : torch.Tensor
        ``[E]`` the gap that drives the activation taper: the forecast-aware
        clearance of approaching pairs, the signed gap of crossed pairs.
    crossed : torch.Tensor
        ``[E]`` boolean crossing flags (all false without the reference side test).
    signed : bool
        Whether the reference side test ran, so that ``gap`` and the contact
        normal are signed by it.
    stats : dict
        Discovery counters: ``candidates``, ``active_pairs``, ``pages``,
        ``largest_page``, ``narrow_phase_fallback_pairs``,
        ``narrow_phase_reference_pages``, ``largest_reference_page``,
        ``reference_excluded_pairs``, ``sweep_pairs``,
        ``clearance_crossings_kept`` and ``crossed_pairs``.
    """

    query: torch.Tensor
    face: torch.Tensor
    source_nodes: torch.Tensor
    barycentric: torch.Tensor
    closest: torch.Tensor
    distance: torch.Tensor
    half_thickness: torch.Tensor
    gap: torch.Tensor
    activation_gap: torch.Tensor
    crossed: torch.Tensor
    signed: bool
    stats: dict

    @property
    def num_pairs(self) -> int:
        """Number of active pairs."""
        return int(self.query.shape[0])


def linear_closest_approach(
    relative: torch.Tensor, relative_velocity: torch.Tensor, horizon: float
) -> torch.Tensor:
    r"""Distance to a transported material point over a causal linear forecast.

    ``relative`` is source minus query and ``relative_velocity`` uses the same
    sign; the source's barycentric weights are frozen in time but live in
    autograd. This is not the closest point on the evolving facet, a time of
    impact, or a continuous collision detection guarantee. Zero relative speed
    selects time zero without dividing by zero.

    Parameters
    ----------
    relative : torch.Tensor
        ``[E, 3]`` current relative position.
    relative_velocity : torch.Tensor
        ``[E, 3]`` relative velocity.
    horizon : float
        Nonnegative forecast time.

    Returns
    -------
    torch.Tensor
        ``[E]`` closest approach distance over ``[0, horizon]``.
    """
    speed_sq = relative_velocity.square().sum(-1)
    denominator = torch.where(speed_sq > 0, speed_sq, torch.ones_like(speed_sq))
    time = (-(relative * relative_velocity).sum(-1) / denominator).clamp(0, horizon)
    return torch.linalg.vector_norm(
        relative + time[:, None] * relative_velocity, dim=-1
    )


def uses_fused_narrow_phase(
    search: ShellContactSearch, positions: torch.Tensor
) -> bool:
    r"""Whether membership runs in the fused Warp kernel for these positions.

    Parameters
    ----------
    search : ShellContactSearch
        The search settings.
    positions : torch.Tensor
        The node positions the search runs on.

    Returns
    -------
    bool
        True for an explicit Warp narrow phase, or for ``"auto"`` with the
        Warp broad phase on CUDA float32 positions.
    """
    return search.narrow_phase == "warp" or (
        search.narrow_phase == "auto"
        and search.implementation == "warp"
        and positions.is_cuda
        and positions.dtype == torch.float32
    )


def _activation_gap(
    positions,
    velocities,
    thickness,
    query,
    source,
    projection: TriangleProjection,
    search: ShellContactSearch,
):
    """Forecast-aware activation gap of pairs from their live projection."""
    surface_thickness = (thickness[source] * projection.barycentric).sum(-1)
    half_thickness = (thickness[query] + surface_thickness) * 0.5
    if search.prediction_horizon == 0:
        return projection.distance - half_thickness
    relative = projection.closest - positions[query]
    relative_velocity = (velocities[source] * projection.barycentric[..., None]).sum(
        1
    ) - velocities[query]
    distance = linear_closest_approach(
        relative, relative_velocity, search.prediction_horizon
    )
    gap = torch.minimum(distance, projection.distance) - half_thickness
    # The nearest material point can slide along a face/edge. Add the point
    # closest at the forecast endpoint as a second anchor; analytically sweep
    # BOTH material points over the entire interval. This catches crossings
    # between samples, but two anchors are still not full moving-facet CCD.
    future = project_onto_facets(
        positions,
        query,
        source,
        material_fan=search.material_fan,
        velocities=velocities,
        horizon=search.prediction_horizon,
    )
    future_relative = (positions[source] * future.barycentric[..., None]).sum(
        1
    ) - positions[query]
    future_velocity = (velocities[source] * future.barycentric[..., None]).sum(
        1
    ) - velocities[query]
    future_distance = linear_closest_approach(
        future_relative, future_velocity, search.prediction_horizon
    )
    future_thickness = (thickness[source] * future.barycentric).sum(-1)
    future_gap = future_distance - (thickness[query] + future_thickness) * 0.5
    return torch.minimum(gap, future_gap)


def _reference_candidate_gaps(positions, velocities, thickness, faces, pairs, search):
    """Torch reference of the membership gaps of candidate pairs."""
    query, face_id = pairs
    source = faces[face_id]
    projection = project_onto_facets(
        positions, query, source, material_fan=search.material_fan
    )
    return _activation_gap(
        positions, velocities, thickness, query, source, projection, search
    )


def _fused_candidate_gaps(positions, velocities, thickness, faces, pairs, search):
    """Fused Warp membership gaps and the ambiguity mask of candidate pairs."""
    from physicsnemo.nn.functional.neighbors.surface_contact_narrow_warp import (
        fused_activation_gaps,
    )

    return fused_activation_gaps(
        positions,
        velocities,
        thickness,
        faces,
        pairs,
        search.prediction_horizon,
        search.activation_distance,
        search.crossing_radius,
    )


def candidate_activation_gaps(
    positions: torch.Tensor,
    velocities: torch.Tensor | None,
    thickness: torch.Tensor,
    faces: torch.Tensor,
    pairs: torch.Tensor,
    search: ShellContactSearch,
) -> tuple[torch.Tensor, int]:
    r"""Membership gaps of candidate pairs, fused where available.

    Discrete membership only: the fused kernel marks numerically ambiguous
    pairs (near a membership boundary, ill-conditioned fans, ties between
    material regions), which are recomputed by the Torch reference. Live
    outputs of a search are always recomputed in Torch afterwards.

    Parameters
    ----------
    positions : torch.Tensor
        ``[N, 3]`` node positions.
    velocities : torch.Tensor, optional
        ``[N, 3]`` velocities, required for a positive forecast horizon.
    thickness : torch.Tensor
        ``[N]`` nodal shell thickness.
    faces : torch.Tensor
        ``[F, 3|4]`` facet connectivity.
    pairs : torch.Tensor
        ``[2, E]`` candidate ``(node, face)`` pairs.
    search : ShellContactSearch
        The search settings.

    Returns
    -------
    tuple[torch.Tensor, int]
        ``[E]`` activation gaps and the number of pairs the reference decided.
    """
    if not uses_fused_narrow_phase(search, positions):
        return _reference_candidate_gaps(
            positions, velocities, thickness, faces, pairs, search
        ), 0
    gaps, ambiguous = _fused_candidate_gaps(
        positions, velocities, thickness, faces, pairs, search
    )
    reference_ids = ambiguous.nonzero().flatten()
    if not len(reference_ids):
        return gaps, 0
    reference_gap = _reference_candidate_gaps(
        positions, velocities, thickness, faces, pairs[:, reference_ids], search
    )
    gaps.index_copy_(0, reference_ids, reference_gap)
    return gaps, len(reference_ids)


@torch.no_grad()
def _certify_crossed(positions, reference_positions, thickness, faces, pairs, search):
    """Detached crossing certification of candidate pairs, for membership.

    The same side test as the live stage on the same detached inputs, so a
    pair certified here is flagged crossed there too.
    """
    query, face_id = pairs
    source = faces[face_id]
    projection = project_onto_facets(
        positions, query, source, material_fan=search.material_fan
    )
    _, reference_projection = reference_facet_projection(
        reference_positions, thickness, query, source, material_fan=search.material_fan
    )
    return crossing_flags(
        positions[query],
        projection,
        reference_positions[query],
        reference_projection,
        symmetric=search.side_symmetric,
    )


def _validate_inputs(
    positions, faces, thickness, reference_positions, velocities, search
):
    n = len(positions)
    if thickness is None or thickness.shape != (n,):
        raise ValueError(
            "surface contact requires supplied physical nodal thickness [N]"
        )
    thickness = thickness.to(positions)
    if not torch.isfinite(thickness).all() or torch.any(thickness < 0):
        raise ValueError("shell thickness must be finite and nonnegative")
    if faces is None:
        raise ValueError(
            "shell contact requires faces, the facet element connectivity [F, 3|4]"
        )
    if search.prediction_horizon > 0 and (
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
    # after folding/collapse. Its reference MUST still be valid:
    # reference_positions are the observed reference configuration the fan is
    # built on, never a predicted or later frame.
    if (search.material_fan or search.uses_reference) and (
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
    if search.material_fan:
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
        if search.prediction_horizon > 0:
            geometries.append(
                ("forecast", positions + search.prediction_horizon * velocities)
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
                            "unsupported surface quad: overlapping, folded, or "
                            f"degenerate center fan; {label} facet IDs (first 8): {bad}"
                        )
    return thickness


def shell_contact_pairs(
    positions: torch.Tensor,
    faces: torch.Tensor,
    thickness: torch.Tensor,
    *,
    reference_positions: torch.Tensor | None = None,
    velocities: torch.Tensor | None = None,
    excluded_pairs: torch.Tensor | None = None,
    batch: torch.Tensor | None = None,
    search: ShellContactSearch | None = None,
) -> ShellContactPairs:
    r"""Discover the active node/facet contact pairs of a shell mesh.

    All surface nodes are eligible; incidence (a node and its own facets),
    cross-batch pairs and the caller's ``excluded_pairs`` are never paired.
    Membership is decided on detached geometry and the live geometry of the
    selected pairs is recomputed afterwards in canonical ``(node, face)``
    order, so the returned closest points, weights, distances and gaps are
    differentiable in ``positions`` (and ``velocities``). The active-pair
    budget of the search raises instead of truncating.

    Parameters
    ----------
    positions : torch.Tensor
        ``[N, 3]`` physical node positions (float32 for the Warp backends).
    faces : torch.Tensor
        ``[F, 3|4]`` int64 facet connectivity; a triangle in a four-wide array
        repeats its third vertex.
    thickness : torch.Tensor
        ``[N]`` finite, nonnegative nodal shell thickness.
    reference_positions : torch.Tensor, optional
        ``[N, 3]`` reference configuration; required by the material fan, the
        reference side test and the reference clearance.
    velocities : torch.Tensor, optional
        ``[N, 3]`` nodal velocities; required by a positive forecast horizon.
    excluded_pairs : torch.Tensor, optional
        ``[2, X]`` int64 ``(node, face)`` pairs that are never active (for
        example the one-ring exclusions of the mesh).
    batch : torch.Tensor, optional
        ``[N]`` int64 graph index of every node for batched meshes.
    search : ShellContactSearch, optional
        The search settings (defaults: 5-unit band, Warp, no corrections).

    Returns
    -------
    ShellContactPairs
        The active pairs with their live geometry and the discovery counters.
    """
    if search is None:
        search = ShellContactSearch()
    n = len(positions)
    thickness = _validate_inputs(
        positions, faces, thickness, reference_positions, velocities, search
    )
    node_pad = thickness * 0.5 + search.search_band
    face_pad = thickness[faces].amax(1) * 0.5
    search_end = None
    if search.prediction_horizon > 0:
        with torch.no_grad():
            # A common Galilean translation preserves relative geometry at
            # every forecast time, avoiding huge boxes from rigid car motion.
            # One global reference also works for batched graphs; cross-batch
            # exclusion remains the candidate functional's responsibility.
            reference = velocities.mean(0) if n else velocities.new_zeros(3)
            search_end = positions + search.prediction_horizon * (
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
        implementation=search.implementation,
        pair_chunk_size=search.candidate_chunk_size,
    )
    # The first projection selects membership only. Do not retain a large
    # backward graph for broad-phase false positives. Recompute LIVE
    # geometry for the selected pairs, preserving all geometric gradients.
    # Keep only selected IDs between pages; never collect all raw pairs.
    with torch.no_grad():
        selected = []
        candidates = active = pages = largest_page = fallback_pairs = 0
        reference_pages = largest_reference_page = 0
        reference_excluded = sweep_pairs = clearance_crossings_kept = 0
        pending = []
        pending_count = 0
        use_fused = uses_fused_narrow_phase(search, positions)

        def select_active(pair_ids, gaps):
            nonlocal active, reference_excluded, sweep_pairs
            nonlocal clearance_crossings_kept
            near = gaps < search.activation_distance
            from_sweep = None
            if search.crossing_radius is not None:
                # Crossing sweep: candidates beyond the band but within the
                # radius are kept only when certified crossed below.
                from_sweep = ~near & (gaps < search.crossing_radius)
                keep = near | from_sweep
                kept = pair_ids[:, keep]
                from_sweep = from_sweep[keep]
            else:
                kept = pair_ids[:, near]
            passes = None
            if search.reference_clearance is not None and kept.shape[1]:
                # Same discrete narrow phase on the reference configuration
                # (fused kernel where available, Torch reference otherwise).
                # The reference carries no forecast, so a positive horizon of
                # the live search must not be applied to it.
                reference_gap, _ = candidate_activation_gaps(
                    reference_positions,
                    None,
                    thickness,
                    faces,
                    kept,
                    replace(search, prediction_horizon=0.0),
                )
                passes = reference_gap >= search.reference_clearance
            needs_certificate = None
            if from_sweep is not None:
                needs_certificate = from_sweep
            if passes is not None and search.clearance_approach_only:
                needs_certificate = (
                    ~passes
                    if needs_certificate is None
                    else needs_certificate | ~passes
                )
            if needs_certificate is not None and bool(needs_certificate.any()):
                crossed = torch.zeros_like(needs_certificate)
                ids = needs_certificate.nonzero().flatten()
                crossed[ids] = _certify_crossed(
                    positions,
                    reference_positions,
                    thickness,
                    faces,
                    kept[:, ids],
                    search,
                )
                accept = torch.ones_like(crossed)
                if from_sweep is not None:
                    accept &= ~from_sweep | crossed
                    sweep_pairs += int((from_sweep & crossed).sum())
                if passes is not None:
                    exempt = (
                        ~passes & crossed if search.clearance_approach_only else None
                    )
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
            if active > search.max_pairs:
                raise RuntimeError(
                    f"surface active contact budget exceeded ({active}>{search.max_pairs}); "
                    "no pairs truncated"
                )
            if kept.shape[1]:
                selected.append(kept)

        def select_reference(pair_ids):
            nonlocal reference_pages, largest_reference_page
            reference_pages += 1
            largest_reference_page = max(largest_reference_page, pair_ids.shape[1])
            gaps = _reference_candidate_gaps(
                positions, velocities, thickness, faces, pair_ids, search
            )
            select_active(pair_ids, gaps)

        for pairs in chunks:
            pages += 1
            candidates += pairs.shape[1]
            largest_page = max(largest_page, pairs.shape[1])
            if use_fused:
                candidate_gap, ambiguous = _fused_candidate_gaps(
                    positions, velocities, thickness, faces, pairs, search
                )
                reference_pairs = pairs[:, ambiguous]
                count = reference_pairs.shape[1]
                fallback_pairs += count
                # Do not launch the full reference geometry for a handful of
                # ambiguous pairs on EVERY broad-phase page. Collect discrete
                # IDs into bounded pages; no pair is dropped and reference
                # geometry still has <= candidate_chunk_size rows.
                if count:
                    pending.append(reference_pairs)
                    pending_count += count
                candidate_gap.masked_fill_(ambiguous, float("inf"))
                select_active(pairs, candidate_gap)
                if pending_count >= search.candidate_chunk_size:
                    combined = torch.cat(pending, 1)
                    full = pending_count // search.candidate_chunk_size
                    end = full * search.candidate_chunk_size
                    for page in combined[:, :end].split(search.candidate_chunk_size, 1):
                        select_reference(page)
                    remainder = combined[:, end:]
                    pending_count = remainder.shape[1]
                    pending = [remainder] if pending_count else []
                    del combined, remainder
                del ambiguous, reference_pairs
            else:
                candidate_gap = _reference_candidate_gaps(
                    positions, velocities, thickness, faces, pairs, search
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
        stats = {
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
    projection = project_onto_facets(
        positions, query, source_nodes, material_fan=search.material_fan
    )
    source_thickness = (thickness[source_nodes] * projection.barycentric).sum(-1)
    half_thickness = (thickness[query] + source_thickness) * 0.5
    gap = projection.distance - half_thickness
    activation_gap = _activation_gap(
        positions, velocities, thickness, query, source_nodes, projection, search
    )
    crossed = torch.zeros(len(query), dtype=torch.bool, device=positions.device)
    if search.side_from_reference:
        # A node that changed side of the facet plane since the reference
        # configuration has crossed the shell: its distance counts as
        # negative and the pair is fully active.
        _, reference_projection = reference_facet_projection(
            reference_positions,
            thickness,
            query,
            source_nodes,
            material_fan=search.material_fan,
        )
        crossed = crossing_flags(
            positions[query],
            projection,
            reference_positions[query],
            reference_projection,
            symmetric=search.side_symmetric,
        )
        side = torch.where(crossed, -1.0, 1.0).to(positions.dtype)
        gap = side * projection.distance - half_thickness
        activation_gap = torch.where(crossed, gap, activation_gap)
        stats["crossed_pairs"] = int(crossed.sum())
    return ShellContactPairs(
        query=query,
        face=face_id,
        source_nodes=source_nodes,
        barycentric=projection.barycentric,
        closest=projection.closest,
        distance=projection.distance,
        half_thickness=half_thickness,
        gap=gap,
        activation_gap=activation_gap,
        crossed=crossed,
        signed=search.side_from_reference,
        stats=stats,
    )
