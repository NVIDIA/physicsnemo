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

"""Nearest-k contact graphs with discrete discovery and live geometric features."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from physicsnemo.nn.functional import contact_search

from .contact import (
    CONTACT_FEATURE_DIM,
    KINEMATIC_CONTACT_FEATURE_DIM,
    ContactGraph,
    contact_edge_features,
    merge_contact_graphs,
    smooth_contact_cutoff,
)


@dataclass(frozen=True)
class ContactSearchTopology:
    """Static per-graph node maps and sorted exclusion CSR, reusable in a rollout.

    Each group is ``(global_node_ids, excluded_local_ids, exclusion_offsets)``.
    Construct with ``prepare_topology`` and reuse only for the same node ordering,
    batch assignment, topology, and device. Positions are deliberately not cached.
    """

    num_nodes: int
    groups: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]


class FunctionalContactGraphBuilder(nn.Module):
    """Discover nearest-k nonlocal point pairs using the contact functional.

    ``search_radius`` is a physical center-to-center distance, not a shell gap.
    Self and structural neighbors are excluded *before* nearest-k selection.
    Additional static exclusions (for example geodesic neighborhoods) can be
    provided as edges to ``prepare_topology``. Whole-component exclusion is not
    applied: distant portions of the same connected structure can self-contact.

    Thickness contributes a pairwise clearance proxy ``d - (t_i+t_j)/2`` to the
    features; it is not a vertex-face signed gap or an extra post-top-k filter.
    No radius expansion or truncated candidate pool is hidden. ``None`` selects
    Warp explicitly (and fails if unavailable); the dense Torch reference must
    be requested explicitly and is intended only for small validation graphs.

    Only discovery uses detached coordinates. Returned point/velocity features
    and optional smooth weights are recomputed from live tensors for BPTT.
    """

    def __init__(
        self,
        search_radius: float,
        max_neighbors: int = 16,
        implementation: str | None = None,
        include_velocity: bool = False,
        velocity_scale: float = 1000.0,
        smooth_cutoff: bool = False,
        selection_taper: bool = False,
        normal_epsilon: float | None = None,
        activation_distance: float | None = None,
        prune_zero_weight: bool = False,
    ) -> None:
        super().__init__()
        if not math.isfinite(search_radius) or search_radius <= 0:
            raise ValueError("search_radius must be positive and finite")
        if (
            isinstance(max_neighbors, bool)
            or not isinstance(max_neighbors, int)
            or max_neighbors < 1
        ):
            raise ValueError("max_neighbors must be a positive integer")
        if implementation not in (None, "torch", "warp"):
            raise ValueError("implementation must be None, 'torch', or 'warp'")
        if not math.isfinite(velocity_scale) or velocity_scale <= 0:
            raise ValueError("velocity_scale must be positive and finite")
        for name, value in (
            ("normal_epsilon", normal_epsilon),
            ("activation_distance", activation_distance),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")
        if selection_taper and not smooth_cutoff:
            raise ValueError("selection_taper requires smooth_cutoff=True")
        if selection_taper and normal_epsilon is None:
            raise ValueError("selection_taper requires a physical normal_epsilon")
        if activation_distance is not None and not smooth_cutoff:
            raise ValueError("activation_distance requires smooth_cutoff=True")
        if prune_zero_weight and not smooth_cutoff:
            raise ValueError("prune_zero_weight requires smooth_cutoff=True")
        self.search_radius = float(search_radius)
        self.max_neighbors = max_neighbors
        self.implementation = implementation or "warp"
        self.include_velocity = include_velocity
        self.velocity_scale = float(velocity_scale)
        self.smooth_cutoff = smooth_cutoff
        self.selection_taper = bool(selection_taper)
        self.normal_epsilon = normal_epsilon
        self.activation_distance = activation_distance
        self.prune_zero_weight = bool(prune_zero_weight)
        self.feature_dim = (
            KINEMATIC_CONTACT_FEATURE_DIM if include_velocity else CONTACT_FEATURE_DIM
        )

    @staticmethod
    @torch.no_grad()
    def prepare_topology(
        num_nodes: int,
        structural_edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        *,
        device: torch.device | str,
        extra_exclusion_edges: torch.Tensor | None = None,
    ) -> ContactSearchTopology:
        """Precompute symmetric one-hop/self exclusions without dense adjacency."""
        device = torch.device(device)
        if num_nodes < 0 or num_nodes >= torch.iinfo(torch.int32).max:
            raise ValueError("num_nodes must fit in nonnegative int32 indexing")
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        if (
            batch.shape != (num_nodes,)
            or batch.dtype != torch.long
            or batch.device != device
        ):
            raise ValueError("batch must be a device-matched long tensor of shape [N]")
        if torch.any(batch < 0):
            raise ValueError("batch IDs cannot be negative")
        edge_parts = []
        for edges in (structural_edge_index, extra_exclusion_edges):
            if edges is None:
                continue
            if (
                edges.ndim != 2
                or edges.shape[0] != 2
                or edges.dtype != torch.long
                or edges.device != device
            ):
                raise ValueError(
                    "exclusion edges must be device-matched long tensors of shape [2, E]"
                )
            if torch.any(edges < 0) or torch.any(edges >= num_nodes):
                raise ValueError("exclusion edge index out of bounds")
            if torch.any(batch[edges[0]] != batch[edges[1]]):
                raise ValueError(
                    "exclusion edges cannot connect different batch graphs"
                )
            edge_parts.append(edges)
        edges = (
            torch.cat(edge_parts, dim=1)
            if edge_parts
            else torch.empty((2, 0), dtype=torch.long, device=device)
        )
        groups = []
        labels = torch.unique(batch, sorted=True)
        for label in labels:
            node_ids = torch.nonzero(batch == label, as_tuple=False).flatten()
            n = node_ids.numel()
            inverse = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
            inverse[node_ids] = torch.arange(n, device=device)
            local_edges = inverse[edges[:, batch[edges[0]] == label]]
            self_ids = torch.arange(n, device=device)
            codes = torch.cat(
                (
                    local_edges[0] * n + local_edges[1],
                    local_edges[1] * n + local_edges[0],
                    self_ids * n + self_ids,
                )
            ).unique(sorted=True)
            if codes.numel() >= torch.iinfo(torch.int32).max:
                raise ValueError("too many CSR exclusions for int32 indexing")
            rows = codes // n
            counts = torch.bincount(rows, minlength=n)
            offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(torch.int32)
            groups.append((node_ids, (codes % n).to(torch.int32), offsets))
        return ContactSearchTopology(num_nodes, tuple(groups))

    def forward(
        self,
        positions: torch.Tensor,
        structural_edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        shell_thickness: torch.Tensor | None = None,
        velocities: torch.Tensor | None = None,
        topology: ContactSearchTopology | None = None,
    ) -> ContactGraph:
        if (
            positions.ndim != 2
            or positions.shape[-1] != 3
            or not positions.is_floating_point()
        ):
            raise ValueError("positions must be a floating tensor of shape [N, 3]")
        # Keep physical geometry in fp32 under mixed precision, without detaching
        # the feature path. Preserve double precision for the Torch reference.
        physical = positions if positions.dtype == torch.float64 else positions.float()
        if not torch.isfinite(physical).all():
            raise ValueError("positions must be finite before contact discovery")
        n = physical.shape[0]
        if shell_thickness is None:
            thickness = physical.new_zeros(n)
        else:
            thickness = shell_thickness.to(
                device=physical.device, dtype=physical.dtype
            ).reshape(-1)
            if (
                thickness.shape != (n,)
                or not torch.isfinite(thickness).all()
                or torch.any(thickness < 0)
            ):
                raise ValueError(
                    "shell_thickness must contain N finite nonnegative values"
                )
        if self.include_velocity:
            if (
                velocities is None
                or velocities.shape != physical.shape
                or velocities.device != physical.device
            ):
                raise ValueError("live velocities of shape [N, 3] are required")
            velocities = velocities.to(physical.dtype)
            if not torch.isfinite(velocities).all():
                raise ValueError("velocities must be finite")
        if topology is None:
            topology = self.prepare_topology(
                n, structural_edge_index, batch, device=physical.device
            )
        if topology.num_nodes != n:
            raise ValueError("prepared contact topology does not match the node count")
        graphs = []
        for node_ids, excluded, offsets in topology.groups:
            if node_ids.device != physical.device:
                raise ValueError("prepared topology and positions must share a device")
            live_points = physical.index_select(0, node_ids)
            with torch.no_grad():
                indices = contact_search(
                    live_points.detach(),
                    live_points.detach(),
                    self.search_radius,
                    excluded,
                    offsets,
                    max_points=self.max_neighbors + int(self.selection_taper),
                    implementation=self.implementation,
                )
                # K+1 supplies a live support distance; only K messages are kept.
                # A disappearing Kth neighbor therefore has zero contribution.
                retained = indices[:, : self.max_neighbors]
                valid = retained >= 0
                query_ids, _ = valid.nonzero(as_tuple=True)
                source = node_ids[retained[valid].long()]
                destination = node_ids[query_ids]
            relative = physical[source] - physical[destination]
            distance = torch.linalg.vector_norm(relative, dim=-1)
            clearance = distance - 0.5 * (thickness[source] + thickness[destination])
            relative_velocity = (
                velocities[source] - velocities[destination]
                if self.include_velocity
                else None
            )
            weights = None
            if self.smooth_cutoff:
                weights = smooth_contact_cutoff(distance, self.search_radius)
                if self.selection_taper:
                    buffer_ids = indices[:, -1].long()
                    buffer_relative = live_points[buffer_ids.clamp_min(0)] - live_points
                    support_sq = torch.where(
                        buffer_ids >= 0,
                        buffer_relative.square().sum(-1),
                        live_points.new_full((len(node_ids),), self.search_radius**2),
                    ).clamp_max(self.search_radius**2)[query_ids]
                    # The additive physical epsilon avoids a singular denominator
                    # for coincident candidate clusters. Equal-distance boundary
                    # ties all have zero weight, independent of backend tie IDs.
                    margin = (support_sq - relative.square().sum(-1)).clamp_min(0)
                    weights = (margin / (support_sq + self.normal_epsilon**2)).square()
                if self.activation_distance is not None:
                    # Clearance is a node/shell proxy, not a signed surface gap.
                    # Broad-phase proximity alone must not imply full activation.
                    weights = weights * smooth_contact_cutoff(
                        clearance, self.activation_distance
                    )
            if self.prune_zero_weight:
                # Exact zero only, never a heuristic small-weight threshold.
                # All compact-envelope boundaries have zero first derivative;
                # removing these messages preserves the response and gradients.
                active = weights.detach() > 0
                source, destination = source[active], destination[active]
                relative, clearance = relative[active], clearance[active]
                weights = weights[active]
                if relative_velocity is not None:
                    relative_velocity = relative_velocity[active]
            graphs.append(
                ContactGraph(
                    edge_index=torch.stack((source, destination)),
                    edge_features=contact_edge_features(
                        relative,
                        clearance,
                        self.search_radius,
                        relative_velocity=relative_velocity,
                        velocity_scale=self.velocity_scale,
                        normal_epsilon=self.normal_epsilon,
                    ),
                    obstacle_mask=torch.zeros(
                        source.numel(), dtype=torch.bool, device=physical.device
                    ),
                    edge_weights=weights,
                )
            )
        if not graphs:
            empty = ContactGraph.empty(
                physical.device, physical.dtype, self.feature_dim
            )
            return ContactGraph(
                empty.edge_index,
                empty.edge_features,
                empty.obstacle_mask,
                physical.new_empty(0) if self.smooth_cutoff else None,
            )
        return merge_contact_graphs(*graphs)
