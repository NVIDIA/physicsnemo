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

r"""Contact graph representation, feature encoding, and gated latent messages.

Candidate search is separate from the learned message block
(:class:`SparseContactBlock`). DeFormer's reference recipe populates the graph
with predictive node-to-face contact and live barycentric interpolation through
``SurfaceContactGraphBuilder``, the reference backend.
:class:`SparseContactGraphBuilder` is the simple node-to-node radius backend
(selected by ``contact_graph_backend: legacy`` in the crash example); the
``obstacle_mask`` field serves analytic obstacle contacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.experimental.nn.shell_contact import (
    CONTACT_FEATURE_DIM,
    KINEMATIC_CONTACT_FEATURE_DIM,
    contact_edge_features,
    smooth_contact_cutoff,
)

_torch_cluster = OptionalImport("torch_cluster")

_EPS = 1.0e-8

__all__ = [
    "CONTACT_FEATURE_DIM",
    "KINEMATIC_CONTACT_FEATURE_DIM",
    "ContactGraph",
    "SparseContactBlock",
    "SparseContactGraphBuilder",
    "contact_edge_features",
    "merge_contact_graphs",
    "smooth_contact_cutoff",
]


@dataclass(frozen=True)
class ContactGraph:
    r"""Directed sparse contact graph consumed by :class:`SparseContactBlock`.

    ``edge_index`` follows the PyG source-to-destination convention. Analytic
    obstacle contacts use a valid placeholder source node and set the corresponding
    ``obstacle_mask`` entry; the contact block then substitutes a learned obstacle
    embedding for that source latent.

    Surface contacts supply ``source_nodes [E,K]`` and live barycentric
    ``source_weights [E,K]``. Their source latent is a weighted interpolation,
    and ``edge_index[0]`` is only a valid placeholder, NOT the closest surface
    point. Use the interpolation or encoded displacement for geometry/plots.
    """

    edge_index: torch.Tensor
    edge_features: torch.Tensor
    obstacle_mask: torch.Tensor
    edge_weights: torch.Tensor | None = None
    source_nodes: torch.Tensor | None = None
    source_weights: torch.Tensor | None = None

    @classmethod
    def empty(
        cls,
        device: torch.device,
        dtype: torch.dtype,
        feature_dim: int = CONTACT_FEATURE_DIM,
    ) -> "ContactGraph":
        return cls(
            edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
            edge_features=torch.empty((0, feature_dim), dtype=dtype, device=device),
            obstacle_mask=torch.empty((0,), dtype=torch.bool, device=device),
        )

    def validate(self, num_nodes: int, feature_dim: int) -> None:
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError(
                "contact edge_index must have shape [2, E]; "
                f"got {tuple(self.edge_index.shape)}"
            )
        num_edges = self.edge_index.shape[1]
        if self.edge_index.dtype != torch.long:
            raise ValueError("contact edge_index must use torch.long indices")
        if self.edge_features.shape != (num_edges, feature_dim):
            raise ValueError(
                f"contact edge_features must have shape [E, {feature_dim}]; "
                f"got {tuple(self.edge_features.shape)}"
            )
        if self.obstacle_mask.shape != (num_edges,):
            raise ValueError(
                "contact obstacle_mask must have shape [E]; "
                f"got {tuple(self.obstacle_mask.shape)}"
            )
        if self.obstacle_mask.dtype != torch.bool:
            raise ValueError("contact obstacle_mask must use torch.bool")
        if self.edge_weights is not None:
            if self.edge_weights.shape != (num_edges,):
                raise ValueError("contact edge_weights must have shape [E]")
            # Taper weights lie in [0, 1]; area quadrature scales them by the
            # face area over a reference area, so only nonnegativity is required.
            if not torch.isfinite(self.edge_weights).all() or torch.any(
                self.edge_weights < 0
            ):
                raise ValueError("contact edge_weights must be finite and nonnegative")
        if num_edges:
            if torch.any(self.edge_index < 0) or torch.any(
                self.edge_index >= num_nodes
            ):
                raise ValueError("contact edge indices are outside the node range")
        if (self.source_nodes is None) != (self.source_weights is None):
            raise ValueError(
                "source_nodes and source_weights must be supplied together"
            )
        if self.source_nodes is not None:
            if (
                self.source_nodes.ndim != 2
                or self.source_nodes.shape[0] != num_edges
                or self.source_nodes.shape[1] < 1
                or self.source_nodes.dtype != torch.long
            ):
                raise ValueError("source_nodes must be int64 [E,K]")
            if self.source_weights.shape != self.source_nodes.shape:
                raise ValueError("source_weights must match source_nodes")
            if (
                self.source_nodes.device != self.edge_index.device
                or self.source_weights.device != self.edge_features.device
            ):
                raise ValueError("surface interpolation tensors must be device matched")
            if torch.any(self.source_nodes < 0) or torch.any(
                self.source_nodes >= num_nodes
            ):
                raise ValueError("surface source index outside node range")
            if (
                not torch.isfinite(self.source_weights).all()
                or torch.any(self.source_weights < 0)
                or not torch.allclose(
                    self.source_weights.sum(-1),
                    torch.ones_like(self.source_weights[:, 0]),
                    atol=1e-6,
                    rtol=1e-6,
                )
            ):
                raise ValueError(
                    "source_weights must be nonnegative barycentric weights summing to one"
                )

    def to(self, device: torch.device) -> "ContactGraph":
        return ContactGraph(
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            obstacle_mask=self.obstacle_mask.to(device),
            edge_weights=(
                None if self.edge_weights is None else self.edge_weights.to(device)
            ),
            source_nodes=None
            if self.source_nodes is None
            else self.source_nodes.to(device),
            source_weights=None
            if self.source_weights is None
            else self.source_weights.to(device),
        )


def merge_contact_graphs(*graphs: ContactGraph) -> ContactGraph:
    r"""Concatenate compatible contact graphs without changing edge order."""

    if not graphs:
        raise ValueError("At least one contact graph is required")
    feature_dim = graphs[0].edge_features.shape[-1]
    for graph in graphs:
        if graph.edge_features.shape[-1] != feature_dim:
            raise ValueError("All contact graphs must use the same feature dimension")
    width = max(
        (g.source_nodes.shape[1] for g in graphs if g.source_nodes is not None),
        default=0,
    )
    source_nodes, source_weights = [], []
    if width:
        for graph in graphs:
            nodes = graph.source_nodes
            weights = graph.source_weights
            if nodes is None:
                nodes = graph.edge_index[0, :, None]
                weights = graph.edge_features.new_ones((nodes.shape[0], 1))
            pad = width - nodes.shape[1]
            source_nodes.append(torch.cat((nodes, nodes[:, :1].expand(-1, pad)), -1))
            source_weights.append(torch.nn.functional.pad(weights, (0, pad)))
    return ContactGraph(
        source_nodes=torch.cat(source_nodes) if width else None,
        source_weights=torch.cat(source_weights) if width else None,
        edge_index=torch.cat([graph.edge_index for graph in graphs], dim=1),
        edge_features=torch.cat([graph.edge_features for graph in graphs], dim=0),
        obstacle_mask=torch.cat([graph.obstacle_mask for graph in graphs], dim=0),
        edge_weights=(
            torch.cat(
                [
                    graph.edge_weights
                    if graph.edge_weights is not None
                    else graph.edge_features.new_ones(graph.edge_index.shape[1])
                    for graph in graphs
                ]
            )
            if any(graph.edge_weights is not None for graph in graphs)
            else None
        ),
    )


class SparseContactGraphBuilder(nn.Module):
    r"""Build thickness-aware node-to-node contact candidates within a radius.

    The simple radius backend (``contact_graph_backend: legacy`` in the crash
    example): every pair of nodes of the same graph whose signed gap (centre
    distance minus the mean of the two shell thicknesses) lies within
    ``search_radius`` is a directed candidate edge, the ``max_neighbors``
    smallest gaps per destination node are kept, and the edge features are
    :func:`~physicsnemo.experimental.nn.shell_contact.contact_edge_features`.
    Small graphs use a deterministic ``torch.cdist`` implementation. Larger
    graphs use ``torch_cluster.radius_graph`` and therefore require
    PhysicsNeMo's ``gnns`` optional dependency group.

    Parameters
    ----------
    search_radius : float
        Positive signed-gap radius within which a node pair is a candidate.
    max_neighbors : int
        Candidates kept per destination node, smallest signed gap first.
    candidate_neighbors : int, optional
        Raw neighbors ``torch_cluster.radius_graph`` collects per node before
        filtering; ``None`` uses ``max(4 * max_neighbors, 64)``. Default is
        ``None``.
    exclude_structural_edges : bool, optional
        Drop candidates that are edges of ``structural_edge_index``. Default is
        ``True``.
    exclude_same_component : bool, optional
        Drop candidates within one component of ``component_ids`` (then
        required at forward). Default is ``False``.
    brute_force_threshold : int, optional
        Largest node count searched densely with ``torch.cdist``; above it
        ``torch_cluster`` is required. Default is 4096.

    Forward
    -------
    positions : torch.Tensor
        Node positions of shape ``[N, 3]``.
    structural_edge_index : torch.Tensor, optional
        Structural edges ``[2, E]`` to exclude.
    batch : torch.Tensor, optional
        Graph ID per node ``[N]``; candidates never cross graphs.
    shell_thickness : torch.Tensor, optional
        Nonnegative shell thickness per node, ``[N]`` or ``[N, 1]``.
    component_ids : torch.Tensor, optional
        Component ID per node ``[N]`` for ``exclude_same_component``.

    Outputs
    -------
    ContactGraph
        Candidate edges with their features and an all-false ``obstacle_mask``.
    """

    def __init__(
        self,
        search_radius: float,
        max_neighbors: int,
        candidate_neighbors: int | None = None,
        exclude_structural_edges: bool = True,
        exclude_same_component: bool = False,
        brute_force_threshold: int = 4096,
    ) -> None:
        super().__init__()
        if search_radius <= 0.0:
            raise ValueError("search_radius must be positive")
        if max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive")
        if candidate_neighbors is None:
            candidate_neighbors = max(4 * max_neighbors, 64)
        if candidate_neighbors < max_neighbors:
            raise ValueError("candidate_neighbors cannot be smaller than max_neighbors")
        if brute_force_threshold < 0:
            raise ValueError("brute_force_threshold cannot be negative")
        self.search_radius = float(search_radius)
        self.max_neighbors = int(max_neighbors)
        self.candidate_neighbors = int(candidate_neighbors)
        self.exclude_structural_edges = bool(exclude_structural_edges)
        self.exclude_same_component = bool(exclude_same_component)
        self.brute_force_threshold = int(brute_force_threshold)

    @staticmethod
    def _exclude_edges(
        edge_index: torch.Tensor,
        structural_edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0 or structural_edge_index.shape[1] == 0:
            return torch.ones(
                edge_index.shape[1], dtype=torch.bool, device=edge_index.device
            )
        candidate_codes = edge_index[0] * num_nodes + edge_index[1]
        structural_codes = (
            structural_edge_index[0] * num_nodes + structural_edge_index[1]
        )
        structural_codes = torch.unique(structural_codes).sort().values
        locations = torch.searchsorted(structural_codes, candidate_codes)
        bounded = locations < structural_codes.numel()
        locations = locations.clamp_max(structural_codes.numel() - 1)
        is_structural = bounded & (
            structural_codes.index_select(0, locations) == candidate_codes
        )
        return ~is_structural

    @staticmethod
    def _topk_per_destination(
        edge_index: torch.Tensor,
        score: torch.Tensor,
        max_neighbors: int,
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return torch.empty(0, dtype=torch.long, device=edge_index.device)
        # Stable lexicographic ordering: score first, then destination while
        # preserving score order within each destination group.
        by_score = torch.argsort(score, stable=True)
        destinations = edge_index[1].index_select(0, by_score)
        by_destination = torch.argsort(destinations, stable=True)
        order = by_score.index_select(0, by_destination)
        sorted_destinations = edge_index[1].index_select(0, order)

        group_start = torch.ones_like(sorted_destinations, dtype=torch.bool)
        group_start[1:] = sorted_destinations[1:] != sorted_destinations[:-1]
        group_ids = torch.cumsum(group_start.to(torch.long), dim=0) - 1
        starts = torch.nonzero(group_start, as_tuple=False).flatten()
        ranks = torch.arange(order.numel(), device=order.device) - starts.index_select(
            0, group_ids
        )
        return order[ranks < max_neighbors]

    def _candidate_edges(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        num_nodes = positions.shape[0]
        if num_nodes <= self.brute_force_threshold:
            distances = torch.cdist(positions.detach(), positions.detach())
            same_graph = batch[:, None] == batch[None, :]
            mask = same_graph & (distances <= radius) & (distances > _EPS)
            destination, source = torch.nonzero(mask, as_tuple=True)
            return torch.stack((source, destination), dim=0)
        if not _torch_cluster.available:
            raise ImportError(
                "torch_cluster is required for sparse contact search on graphs "
                f"larger than {self.brute_force_threshold} nodes"
            )
        return _torch_cluster.radius_graph(
            positions.detach(),
            r=radius,
            batch=batch,
            loop=False,
            max_num_neighbors=self.candidate_neighbors,
            flow="source_to_target",
        )

    def forward(
        self,
        positions: torch.Tensor,
        structural_edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        shell_thickness: torch.Tensor | None = None,
        component_ids: torch.Tensor | None = None,
    ) -> ContactGraph:
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape [N, 3]")
        num_nodes = positions.shape[0]
        device = positions.device
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        else:
            batch = batch.to(device=device, dtype=torch.long)
            if batch.shape != (num_nodes,):
                raise ValueError("batch must have shape [N]")
        if shell_thickness is None:
            shell_thickness = positions.new_zeros(num_nodes)
        else:
            shell_thickness = shell_thickness.to(device=device, dtype=positions.dtype)
            if shell_thickness.ndim == 2 and shell_thickness.shape[-1] == 1:
                shell_thickness = shell_thickness.squeeze(-1)
            if shell_thickness.shape != (num_nodes,):
                raise ValueError("shell_thickness must have shape [N] or [N, 1]")
            if torch.any(shell_thickness < 0):
                raise ValueError("shell_thickness cannot be negative")
        if component_ids is not None:
            component_ids = component_ids.to(device=device, dtype=torch.long)
            if component_ids.shape != (num_nodes,):
                raise ValueError("component_ids must have shape [N]")

        radius = self.search_radius
        if shell_thickness.numel():
            radius += float(shell_thickness.max().detach().item())
        edge_index = self._candidate_edges(positions, batch, radius)
        if edge_index.shape[1] == 0:
            return ContactGraph.empty(device, positions.dtype)

        source, destination = edge_index
        keep = torch.ones(source.shape[0], dtype=torch.bool, device=device)
        if self.exclude_structural_edges and structural_edge_index is not None:
            structural_edge_index = structural_edge_index.to(
                device=device, dtype=torch.long
            )
            keep &= self._exclude_edges(edge_index, structural_edge_index, num_nodes)
        if self.exclude_same_component:
            if component_ids is None:
                raise ValueError(
                    "component_ids are required when exclude_same_component=True"
                )
            keep &= component_ids[source] != component_ids[destination]

        relative = positions[source] - positions[destination]
        distance = torch.linalg.vector_norm(relative, dim=-1)
        signed_gap = distance - 0.5 * (
            shell_thickness[source] + shell_thickness[destination]
        )
        keep &= signed_gap <= self.search_radius
        edge_index = edge_index[:, keep]
        relative = relative[keep]
        signed_gap = signed_gap[keep]
        if edge_index.shape[1] == 0:
            return ContactGraph.empty(device, positions.dtype)

        selected = self._topk_per_destination(
            edge_index, signed_gap, self.max_neighbors
        )
        edge_index = edge_index.index_select(1, selected)
        relative = relative.index_select(0, selected)
        signed_gap = signed_gap.index_select(0, selected)
        return ContactGraph(
            edge_index=edge_index,
            edge_features=contact_edge_features(
                relative, signed_gap, self.search_radius
            ),
            obstacle_mask=torch.zeros(
                edge_index.shape[1], dtype=torch.bool, device=device
            ),
        )


class SparseContactBlock(nn.Module):
    r"""Convert sparse contact edges into a gated residual latent update.

    Each edge of the :class:`ContactGraph` forms a message from the source
    latent (the barycentric interpolation of ``source_nodes`` for surface
    contacts, the learned obstacle embedding where ``obstacle_mask`` is set),
    the destination latent and the edge features. Messages are scaled by the
    taper ``edge_weights``, aggregated per destination node and passed with the
    node latent through a node MLP with ``tanh`` output; the result is tapered
    so a node whose contacts vanish receives exactly zero, then scaled by a
    learnable gate. An empty graph returns the input with zero gradients for
    every parameter, so distributed ranks with and without contact agree.

    Parameters
    ----------
    hidden_dim : int
        Node latent width :math:`D`.
    contact_dim : int, optional
        Contact edge feature width. Default is ``CONTACT_FEATURE_DIM`` (8);
        ``KINEMATIC_CONTACT_FEATURE_DIM`` (12) with the velocity channels.
    gate_init : float, optional
        Initial value of the learnable output gate; 0.0 makes the block an
        identity at initialisation. Default is 0.0.
    aggregation : {"sum", "mean"}, optional
        Sum the weighted messages per destination node, or divide by the weight
        sum clamped at one. Default is ``"mean"``.

    Forward
    -------
    node_latent : torch.Tensor
        Node latent of shape ``[N, D]``.
    contact_graph : ContactGraph
        Validated against ``N`` and ``contact_dim``.
    return_correction : bool, optional
        Return the gated correction alone instead of ``node_latent`` plus it.
        Default is ``False``.

    Outputs
    -------
    torch.Tensor
        Updated latent, or the correction, of shape ``[N, D]``.
    """

    def __init__(
        self,
        hidden_dim: int,
        contact_dim: int = CONTACT_FEATURE_DIM,
        gate_init: float = 0.0,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or contact_dim <= 0:
            raise ValueError("hidden_dim and contact_dim must be positive")
        if aggregation not in ("sum", "mean"):
            raise ValueError("contact aggregation must be 'sum' or 'mean'")
        self.hidden_dim = hidden_dim
        self.contact_dim = contact_dim
        self.aggregation = aggregation
        self.obstacle_embedding = nn.Parameter(torch.zeros(hidden_dim))
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + contact_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        node_latent: torch.Tensor,
        contact_graph: ContactGraph,
        *,
        return_correction: bool = False,
    ) -> torch.Tensor:
        if node_latent.ndim != 2 or node_latent.shape[-1] != self.hidden_dim:
            raise ValueError(f"node_latent must have shape [N, {self.hidden_dim}]")
        contact_graph.validate(node_latent.shape[0], self.contact_dim)
        if contact_graph.edge_index.shape[1] == 0:
            # Different distributed ranks can have different contact counts.
            # Keep zero gradients for every contact parameter on empty graphs
            # so the recipe's dense all-reduce sees identical parameter layouts.
            zero = sum(
                parameter.reshape(-1)[0] * 0.0 for parameter in self.parameters()
            )
            correction = torch.zeros_like(node_latent) + zero.to(node_latent.dtype)
            return correction if return_correction else node_latent + correction

        source, destination = contact_graph.edge_index
        if contact_graph.source_nodes is not None:
            # Geometry weights remain live through both checkpoint levels.
            weights = contact_graph.source_weights.to(node_latent.dtype)
            # Accumulate one vertex at a time rather than materializing two
            # E x (3 or 4) x H intermediates. Geometry weights stay live.
            accumulation_dtype = (
                torch.float32
                if node_latent.dtype in (torch.float16, torch.bfloat16)
                else node_latent.dtype
            )
            source_latent = node_latent.index_select(
                0, contact_graph.source_nodes[:, 0]
            )
            source_latent = (source_latent * weights[:, :1]).to(accumulation_dtype)
            for vertex in range(1, contact_graph.source_nodes.shape[1]):
                source_latent = source_latent + (
                    node_latent.index_select(0, contact_graph.source_nodes[:, vertex])
                    * weights[:, vertex : vertex + 1]
                ).to(accumulation_dtype)
            source_latent = source_latent.to(node_latent.dtype)
        else:
            source_latent = node_latent.index_select(0, source)
        source_latent = torch.where(
            contact_graph.obstacle_mask.unsqueeze(-1),
            self.obstacle_embedding.to(node_latent.dtype).unsqueeze(0),
            source_latent,
        )
        destination_latent = node_latent.index_select(0, destination)
        messages = self.edge_mlp(
            torch.cat(
                (
                    source_latent,
                    destination_latent,
                    contact_graph.edge_features.to(node_latent.dtype),
                ),
                dim=-1,
            )
        )
        messages = messages.to(node_latent.dtype)
        weights = node_latent.new_ones((destination.shape[0], 1))
        if contact_graph.edge_weights is not None:
            weights = contact_graph.edge_weights.to(messages.dtype).unsqueeze(-1)
            messages = messages * weights
        aggregate = torch.zeros_like(node_latent)
        aggregate.index_add_(0, destination, messages)
        counts = node_latent.new_zeros((node_latent.shape[0], 1))
        counts.index_add_(
            0,
            destination,
            weights,
        )
        if self.aggregation == "mean":
            aggregate = aggregate / counts.clamp_min(1.0)
        correction = torch.tanh(
            self.node_mlp(torch.cat((node_latent, aggregate), dim=-1))
        )
        if contact_graph.edge_weights is None:
            activation = (counts > 0).to(correction.dtype)
        else:
            # Message weighting alone is insufficient: the node MLP has biases
            # and also sees h_i. Taper the entire correction so a disappearing
            # edge gives exactly zero influence, even with nonzero MLP biases.
            activation = -torch.expm1(-counts)
        correction = correction * activation
        correction = self.gate * correction
        return correction if return_correction else node_latent + correction
