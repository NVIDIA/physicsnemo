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

"""Surface-contact adapter over :mod:`physicsnemo.experimental.nn.shell_contact`.

The module keeps the ``SurfaceContactGraphBuilder`` interface the mesh-attention
models and recipes were written against; discovery, features and tapers are the
functional module's. No penalty forces or original-deck equivalence.
"""

import math

import torch
from torch import nn

from physicsnemo.experimental.nn.shell_contact import (
    ShellContactSearch,
    candidate_activation_gaps,
    facet_area_quadrature,
    shell_contact_edge_features,
    shell_contact_pairs,
    smooth_contact_cutoff,
    uses_fused_narrow_phase,
)

from .contact import ContactGraph

__all__ = ["SurfaceContactGraphBuilder"]


class SurfaceContactGraphBuilder(nn.Module):
    """All eligible node--facet pairs with live surface-interpolated messages.

    A stateless wrapper over :func:`~physicsnemo.experimental.nn.shell_contact.shell_contact_pairs`
    and :func:`~physicsnemo.experimental.nn.shell_contact.shell_contact_edge_features`
    that returns a :class:`~physicsnemo.experimental.models.meshtransolver.contact.ContactGraph`.
    Broad phase uses a facet BVH and a conservative maximum nodal thickness;
    narrow phase uses half query thickness + half barycentric facet thickness.
    The latter is a dataset adaptation, NOT a recovered solver contact setting.
    ``activation_distance`` is the positive shell-clearance message band, not
    center distance. The band taper is C1 at zero/band; discrete IDs carry no
    gradients. A direction regularization length bounds its Jacobian near zero.

    All surface nodes are eligible; caller-supplied node/facet exclusions and
    incidence are honored, but no implicit graph-hop/component exclusion occurs.
    Directed latent corrections go to query nodes; this is not a force-balanced
    pair law. Quad sources use the arithmetic-center fan representation.

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
    ``pi * activation_distance**2``), a first-order surface quadrature that
    makes the aggregate approximately independent of mesh resolution. Areas are
    the center-fan areas of the reference geometry when supplied and of the
    detached current geometry otherwise; the factor carries no gradient.

    ``side_from_reference``, ``reference_clearance``, ``side_symmetric``,
    ``crossing_radius`` and ``clearance_approach_only`` are the crossing-aware
    discovery corrections documented on
    :class:`~physicsnemo.experimental.nn.shell_contact.ShellContactSearch`: a node
    that has crossed a shell since the reference configuration gets a negative
    signed gap and a flipped direction; pairs assembled within the clearance in
    the reference are dropped; the sweep discovers crossings deeper than the
    band. ``last_discovery_stats`` reports the counters of the last search.

    Parameters
    ----------
    activation_distance : float
        Positive shell-clearance band (physical length).
    feature_scale : float
        Length normalizing the geometric feature channels.
    velocity_scale : float
        Speed normalizing the kinematic feature channels.
    normal_epsilon : float
        Regularization length of the contact normal.
    include_velocity : bool
        Append the four relative-velocity channels (needs ``velocities``).
    implementation : {"warp", "torch"}
        Broad-phase backend.
    max_pairs : int
        Active-pair safety budget.
    prediction_horizon : float
        Causal forecast time of the activation gap (needs ``velocities``).
    material_fan : bool
        Transport the reference centre fan of quads with the vertices.
    candidate_chunk_size : int
        Broad-phase page size.
    narrow_phase : {"auto", "torch", "warp"}
        Membership geometry backend.
    area_weighting : bool
        Face-area quadrature on the taper weights.
    reference_area : float, optional
        Normalizing area of the quadrature.
    side_from_reference, reference_clearance, side_symmetric, crossing_radius,
    clearance_approach_only
        Crossing-aware discovery corrections.
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
        if reference_area is not None and (
            not math.isfinite(reference_area) or reference_area <= 0
        ):
            raise ValueError("reference_area must be positive and finite")
        self.search = ShellContactSearch(
            activation_distance=activation_distance,
            implementation=implementation,
            narrow_phase=narrow_phase,
            max_pairs=max_pairs,
            candidate_chunk_size=candidate_chunk_size,
            material_fan=material_fan,
            prediction_horizon=prediction_horizon,
            side_from_reference=side_from_reference,
            reference_clearance=reference_clearance,
            side_symmetric=side_symmetric,
            crossing_radius=crossing_radius,
            clearance_approach_only=clearance_approach_only,
        )
        self.feature_scale = float(feature_scale)
        self.velocity_scale = float(velocity_scale)
        self.normal_epsilon = float(normal_epsilon)
        self.include_velocity = bool(include_velocity)
        self.area_weighting = bool(area_weighting)
        self.reference_area = None if reference_area is None else float(reference_area)
        self.last_discovery_stats = None

    # Settings are read through the search so that recipes and tests can keep
    # addressing them on the builder.
    @property
    def activation_distance(self) -> float:
        return self.search.activation_distance

    @property
    def implementation(self) -> str:
        return self.search.implementation

    @property
    def narrow_phase(self) -> str:
        return self.search.narrow_phase

    @property
    def max_pairs(self) -> int:
        return self.search.max_pairs

    @property
    def candidate_chunk_size(self) -> int:
        return self.search.candidate_chunk_size

    @property
    def prediction_horizon(self) -> float:
        return self.search.prediction_horizon

    @property
    def material_fan(self) -> bool:
        return self.search.material_fan

    @property
    def side_from_reference(self) -> bool:
        return self.search.side_from_reference

    @property
    def reference_clearance(self):
        return self.search.reference_clearance

    @property
    def side_symmetric(self) -> bool:
        return self.search.side_symmetric

    @property
    def crossing_radius(self):
        return self.search.crossing_radius

    @property
    def clearance_approach_only(self) -> bool:
        return self.search.clearance_approach_only

    def _use_fused_membership(self, positions):
        return uses_fused_narrow_phase(self.search, positions)

    def _candidate_gaps(self, positions, velocities, thickness, faces, pairs):
        """Discrete membership gaps of candidate pairs and the reference fallback count."""
        return candidate_activation_gaps(
            positions, velocities, thickness, faces, pairs, self.search
        )

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
        if self.include_velocity and (
            velocities is None
            or velocities.shape != positions.shape
            or velocities.dtype != positions.dtype
            or velocities.device != positions.device
            or not torch.isfinite(velocities).all()
        ):
            raise ValueError("live physical velocities must match positions")
        pairs = shell_contact_pairs(
            positions,
            faces,
            shell_thickness,
            reference_positions=reference_positions,
            velocities=velocities,
            excluded_pairs=excluded_pairs,
            batch=batch,
            search=self.search,
        )
        self.last_discovery_stats = pairs.stats
        features = shell_contact_edge_features(
            pairs,
            positions,
            feature_scale=self.feature_scale,
            normal_epsilon=self.normal_epsilon,
            velocities=velocities if self.include_velocity else None,
            velocity_scale=self.velocity_scale,
        )
        edge_weights = smooth_contact_cutoff(
            pairs.activation_gap, self.search.activation_distance
        )
        if self.area_weighting:
            edge_weights = edge_weights * facet_area_quadrature(
                positions if reference_positions is None else reference_positions,
                faces,
                pairs.face,
                reference_area=(
                    self.reference_area
                    if self.reference_area is not None
                    else math.pi * self.search.activation_distance**2
                ),
            )
        return ContactGraph(
            edge_index=torch.stack((pairs.source_nodes[:, 0], pairs.query)),
            edge_features=features,
            obstacle_mask=torch.zeros(
                pairs.num_pairs, dtype=torch.bool, device=positions.device
            ),
            edge_weights=edge_weights,
            source_nodes=pairs.source_nodes,
            source_weights=pairs.barycentric,
        )
