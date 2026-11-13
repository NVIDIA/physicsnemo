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
r"""DeFormer's contact-aware corrector: message passing on a predicted shape.

The corrector is the second half of DeFormer. A frozen time-conditional network
(the teacher) predicts the deformed shape of a frame; the corrector reads that
shape and adds a correction by message passing on the structural mesh:

* every mesh edge carries its reference geometry, the same geometry measured
  on the predicted shape and their difference (the local stretch, bend and
  fold of the prediction), standardised with the reference edge statistics;
* contact pairs discovered on the predicted shape by the crossing-aware
  node-to-facet search of :mod:`physicsnemo.experimental.nn.shell_contact`
  become extra edges between the facet vertices and the node (in both
  directions), carrying the contact geometry, the barycentric weight of the
  vertex and the direction sign: the "world edges" of MeshGraphNet collision
  models;
* the node inputs are assembled by the caller; the DeFormer recipe uses the
  coordinates, the static features, a time code and the teacher's displacement.

There is no global block, so the receptive field is ``processor_size`` hops.
The decoder is zero-initialised, so the corrector starts exactly at the
teacher. One forward corrects one frame; the recipe's wrapper runs the teacher,
builds the node inputs and loops over frames.
"""

from __future__ import annotations

import math

import torch
from jaxtyping import Float, Int
from torch import nn

import physicsnemo
from physicsnemo.experimental.nn.shell_contact import (
    CONTACT_FEATURE_DIM,
    ShellContactSearch,
    shell_contact_edge_features,
    shell_contact_pairs,
)
from physicsnemo.models.meshgraphnet.meshgraphnet import MeshGraphNetProcessor
from physicsnemo.nn.module.gnn_layers import graph_types as _graph_types
from physicsnemo.nn.module.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from physicsnemo.nn.module.gnn_layers.utils import GraphType

EPS = 1.0e-8
MESH_EDGE_INPUT_DIM = 12
"""Reference geometry (4) + predicted geometry (4) + their difference (4)."""
CONTACT_EDGE_INPUT_DIM = CONTACT_FEATURE_DIM + 2
"""Contact geometry (8) + barycentric weight + direction sign."""


def mesh_edge_features(
    pos: Float[torch.Tensor, "num_nodes 3"],
    edge_index: Int[torch.Tensor, "2 num_edges"],
) -> Float[torch.Tensor, "num_edges 4"]:
    r"""Source-minus-destination displacement and distance per directed edge.

    The destination node receives the message. ``pos`` is normalized geometry
    (the reference, or a predicted configuration measured with the reference's
    statistics).

    Parameters
    ----------
    pos : torch.Tensor
        Node positions of shape :math:`(N, 3)`.
    edge_index : torch.Tensor
        Directed edges of shape :math:`(2, E)` as ``(source, destination)``.

    Returns
    -------
    torch.Tensor
        Displacement and its norm per edge, shape :math:`(E, 4)`.
    """
    pos = torch.as_tensor(pos)
    if not torch.compiler.is_compiling():
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(
                f"Expected pos of shape (N, 3) but got tensor of shape {tuple(pos.shape)}"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "Expected edge_index of shape (2, E) but got tensor of shape "
                f"{tuple(edge_index.shape)}"
            )
    displacement = pos[edge_index[0]] - pos[edge_index[1]]
    distance = torch.linalg.norm(displacement, dim=-1, keepdim=True)
    return torch.cat((displacement, distance), dim=-1)


def _edge_graph(edge_index: torch.Tensor, num_nodes: int):
    """A PyG graph carrying only the connectivity the processor needs."""
    if not getattr(_graph_types, "PYG_AVAILABLE", False):
        raise ImportError(
            "DeFormerCorrector's message passing needs torch_geometric (PyG); "
            "install PhysicsNeMo's graph extras"
        )
    return _graph_types.PyGData(edge_index=edge_index, num_nodes=num_nodes)


class DeFormerCorrector(physicsnemo.Module):
    r"""Local message-passing corrector of a teacher's prediction for one frame.

    Parameters
    ----------
    functional_dim : int, optional
        Width of the node inputs the caller assembles (the DeFormer recipe
        uses coords (3) + static features + time code + the teacher's
        displacement (3)). Default is 39.
    hidden_dim : int, optional
        MeshGraphNet width. Default is 128.
    processor_size : int, optional
        Number of message-passing layers. Default is 8.
    num_processor_checkpoint_segments : int, optional
        Activation-checkpoint segments across the processor layers (0: none);
        ``activation_checkpointing=True`` with 0 checkpoints every layer.
        Default is 0.
    aggregation : {"sum", "mean"}, optional
        Message aggregation. Default is ``"sum"``.
    use_contact : bool, optional
        Contact edges from the crossing-aware surface search on the predicted
        shape; needs ``graph.contact_faces``, ``graph.contact_surface_exclusions``
        and ``graph.shell_thickness`` plus the position statistics in the
        forward call. Default is ``True``.
    contact_search_implementation : {"warp", "torch"}, optional
        Backend of the contact search. Default is ``"warp"``.
    contact_surface_max_pairs : int, optional
        Active-pair budget of the search. Default is 8,000,000.
    contact_activation_distance : float, optional
        Band (in the position units) within which a node/facet pair is active.
        Default is 5.0.
    node_contact_radius : float, optional
        Scale of the contact edge features. Default is 10.0.
    contact_normal_epsilon : float, optional
        Regularisation of the facet normals. Default is 0.1.
    contact_surface_material_fan : bool, optional
        Material fan of the search. Default is ``True``.
    contact_side_from_reference : bool, optional
        Decide the crossing side on the reference configuration. Default is
        ``True``.
    contact_reference_clearance : float, optional
        Pairs closer than this in the reference are structure, not contact, and
        are dropped. Default is 10.0.
    contact_side_symmetric : bool, optional
        Symmetric side test. Default is ``False``.
    contact_crossing_radius : float, optional
        Crossing radius of the search. Default is ``None``.
    contact_clearance_approach_only : bool, optional
        Apply the clearance to approaching pairs only. Default is ``False``.
    contact_bidirectional : bool, optional
        Also send the mirrored message from the node to the facet vertices.
        Default is ``True``.
    activation_checkpointing : bool, optional
        Checkpoint every processor layer when
        ``num_processor_checkpoint_segments`` is 0. Default is ``False``.

    Forward
    -------
    node_features : torch.Tensor
        Node inputs of shape :math:`(N, D_{in})` with ``functional_dim``
        channels, assembled by the caller.
    coords : torch.Tensor
        Normalized reference positions of shape :math:`(N, 3)`.
    teacher_displacement : torch.Tensor
        The teacher's normalized displacement of shape :math:`(N, 3)`; the
        predicted shape is ``coords + teacher_displacement``.
    graph : GraphType
        Graph with the directed structural edges ``mesh_edge_index`` of shape
        :math:`(2, E)`, their standardized reference features
        ``mesh_edge_attr`` of shape :math:`(E, 4)` and the statistics
        ``mesh_edge_mean`` / ``mesh_edge_std`` of shape :math:`(4,)`; with
        contact, also ``contact_faces`` :math:`(F, 4)`,
        ``contact_surface_exclusions`` :math:`(2, X)` and ``shell_thickness``
        :math:`(N,)`, and optionally ``batch`` :math:`(N,)`.
    position_mean, position_std : torch.Tensor, optional
        Mean and standard deviation of shape :math:`(3,)` that map normalized
        positions to the physical units of the contact search; required with
        ``use_contact``. Default is ``None``.

    Outputs
    -------
    torch.Tensor
        Corrected normalized positions of shape :math:`(N, 3)`.
    """

    def __init__(
        self,
        *,
        functional_dim: int = 39,
        hidden_dim: int = 128,
        processor_size: int = 8,
        num_processor_checkpoint_segments: int = 0,
        aggregation: str = "sum",
        use_contact: bool = True,
        contact_search_implementation: str = "warp",
        contact_surface_max_pairs: int = 8_000_000,
        contact_activation_distance: float = 5.0,
        node_contact_radius: float = 10.0,
        contact_normal_epsilon: float = 0.1,
        contact_surface_material_fan: bool = True,
        contact_side_from_reference: bool = True,
        contact_reference_clearance: float | None = 10.0,
        contact_side_symmetric: bool = False,
        contact_crossing_radius: float | None = None,
        contact_clearance_approach_only: bool = False,
        contact_bidirectional: bool = True,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if functional_dim <= 0 or hidden_dim <= 0:
            raise ValueError("functional_dim and hidden_dim must be positive")
        if (
            isinstance(processor_size, bool)
            or int(processor_size) != processor_size
            or processor_size < 1
        ):
            raise ValueError("processor_size must be a positive integer")
        if aggregation not in ("sum", "mean"):
            raise ValueError("aggregation must be 'sum' or 'mean'")
        if not contact_activation_distance > 0 or not node_contact_radius > 0:
            raise ValueError(
                "contact_activation_distance and node_contact_radius must be positive"
            )
        if not math.isfinite(contact_normal_epsilon) or contact_normal_epsilon <= 0:
            raise ValueError("contact_normal_epsilon must be positive and finite")
        segments = int(num_processor_checkpoint_segments)
        if segments < 0:
            raise ValueError("num_processor_checkpoint_segments cannot be negative")
        if segments == 0 and activation_checkpointing:
            segments = int(processor_size)
        if segments and int(processor_size) % segments:
            raise ValueError(
                "processor_size must be divisible by num_processor_checkpoint_segments"
            )
        self.functional_dim = int(functional_dim)
        self.hidden_dim = int(hidden_dim)
        self.node_encoder = MeshGraphMLP(
            self.functional_dim, self.hidden_dim, self.hidden_dim, hidden_layers=1
        )
        self.mesh_edge_encoder = MeshGraphMLP(
            MESH_EDGE_INPUT_DIM, self.hidden_dim, self.hidden_dim, hidden_layers=1
        )
        self.processor = MeshGraphNetProcessor(
            processor_size=int(processor_size),
            input_dim_node=self.hidden_dim,
            input_dim_edge=self.hidden_dim,
            aggregation=aggregation,
            num_processor_checkpoint_segments=segments,
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, 3)
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.use_contact = bool(use_contact)
        self.contact_bidirectional = bool(contact_bidirectional)
        self.contact_feature_scale = float(node_contact_radius)
        self.contact_normal_epsilon = float(contact_normal_epsilon)
        if self.use_contact:
            self.contact_edge_encoder = MeshGraphMLP(
                CONTACT_EDGE_INPUT_DIM,
                self.hidden_dim,
                self.hidden_dim,
                hidden_layers=1,
            )
            self.contact_search = ShellContactSearch(
                activation_distance=float(contact_activation_distance),
                implementation=contact_search_implementation,
                max_pairs=int(contact_surface_max_pairs),
                material_fan=bool(contact_surface_material_fan),
                side_from_reference=bool(contact_side_from_reference),
                reference_clearance=contact_reference_clearance,
                side_symmetric=bool(contact_side_symmetric),
                crossing_radius=contact_crossing_radius,
                clearance_approach_only=bool(contact_clearance_approach_only),
            )
        else:
            self.contact_edge_encoder = None
            self.contact_search = None
        self.__name__ = "DeFormerCorrector"

    # ----- contact edges -----
    def contact_edges(
        self,
        predicted: Float[torch.Tensor, "num_nodes 3"],
        coords: Float[torch.Tensor, "num_nodes 3"],
        graph: GraphType,
        position_mean: Float[torch.Tensor, "3"],
        position_std: Float[torch.Tensor, "3"],
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        r"""Expanded contact edges on the predicted shape (normalized positions).

        Edges are emitted once per (pair, facet vertex) and are never merged, so a
        node inside the band of several adjacent facets receives parallel edges
        from their shared vertices and, with sum aggregation, its contact signal
        grows with the number of facets in range; this is intended, the
        multiplicity encodes how much of the opposing surface the node faces.

        Parameters
        ----------
        predicted : torch.Tensor
            Predicted normalized positions of shape :math:`(N, 3)`.
        coords : torch.Tensor
            Normalized reference positions of shape :math:`(N, 3)`.
        graph : GraphType
            Graph with ``contact_faces``, ``contact_surface_exclusions``,
            ``shell_thickness`` and optionally ``batch``.
        position_mean, position_std : torch.Tensor
            Normalization statistics of shape :math:`(3,)` of the positions.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor]
            ``(edge_index, edge_latent)`` with shapes :math:`(2, E_c)` and
            :math:`(E_c, D)`, or ``(None, placeholder)`` on a frame without
            contact, where the zero-valued placeholder keeps every contact
            parameter in the autograd graph (distributed gradient averaging
            needs the same layout on every rank).
        """
        faces = getattr(graph, "contact_faces", None)
        exclusions = getattr(graph, "contact_surface_exclusions", None)
        thickness = getattr(graph, "shell_thickness", None)
        if faces is None or exclusions is None or thickness is None:
            raise ValueError(
                "contact edges need graph.contact_faces, contact_surface_exclusions "
                "and shell_thickness"
            )
        with torch.autocast(device_type=predicted.device.type, enabled=False):
            scale = position_std.to(predicted.device).float()
            mean = position_mean.to(predicted.device).float()
            positions = predicted.detach().float() * scale + mean
            reference = coords.detach().float() * scale + mean
            pairs = shell_contact_pairs(
                positions,
                faces,
                thickness.to(device=predicted.device, dtype=torch.float32).reshape(-1),
                reference_positions=reference,
                excluded_pairs=exclusions,
                batch=getattr(graph, "batch", None),
                search=self.contact_search,
            )
            if pairs.num_pairs == 0:
                placeholder = sum(
                    p.reshape(-1)[0] * 0.0
                    for p in self.contact_edge_encoder.parameters()
                )
                return None, placeholder
            geometry = shell_contact_edge_features(
                pairs,
                positions,
                feature_scale=self.contact_feature_scale,
                normal_epsilon=self.contact_normal_epsilon,
            )
        # One directed edge per (facet vertex, node): the vertex sends the pair's
        # geometry weighted by its barycentric coordinate to the node.
        k = pairs.source_nodes.shape[1]
        source = pairs.source_nodes.reshape(-1)
        destination = pairs.query.repeat_interleave(k)
        weight = pairs.barycentric.float().reshape(-1)
        geometry = geometry.repeat_interleave(k, dim=0)
        # Drop zero-weight vertices: a triangle's padded fourth slot, or a node
        # projecting exactly onto the opposite edge.
        keep = weight > 0
        source, destination = source[keep], destination[keep]
        weight, geometry = weight[keep], geometry[keep]
        sign = torch.ones_like(weight)
        features = torch.cat((geometry, weight[:, None], sign[:, None]), dim=-1)
        index = torch.stack((source, destination))
        if self.contact_bidirectional:
            # The facet vertex receives the mirrored message: the displacement
            # (channels 0-2) and the normal (5-7) point the other way.
            mirrored = geometry.clone()
            mirrored[:, :3] = -mirrored[:, :3]
            mirrored[:, 5:8] = -mirrored[:, 5:8]
            features = torch.cat(
                (
                    features,
                    torch.cat((mirrored, weight[:, None], -sign[:, None]), dim=-1),
                )
            )
            index = torch.cat((index, torch.stack((destination, source))), dim=1)
        return index, self.contact_edge_encoder(features)

    # ----- one frame -----
    def forward(
        self,
        node_features: Float[torch.Tensor, "num_nodes input_dim"],
        coords: Float[torch.Tensor, "num_nodes 3"],
        teacher_displacement: Float[torch.Tensor, "num_nodes 3"],
        graph: GraphType,
        position_mean: Float[torch.Tensor, "3"] | None = None,
        position_std: Float[torch.Tensor, "3"] | None = None,
    ) -> Float[torch.Tensor, "num_nodes 3"]:
        """Correct the teacher's predicted shape of one frame."""
        n = coords.shape[0]
        if not torch.compiler.is_compiling():
            if node_features.ndim != 2 or node_features.shape != (
                n,
                self.functional_dim,
            ):
                raise ValueError(
                    f"Expected node_features of shape ({n}, {self.functional_dim}) but "
                    f"got tensor of shape {tuple(node_features.shape)}"
                )
            if coords.ndim != 2 or coords.shape[1] != 3:
                raise ValueError(
                    f"Expected coords of shape (N, 3) but got tensor of shape "
                    f"{tuple(coords.shape)}"
                )
            if teacher_displacement.shape != coords.shape:
                raise ValueError(
                    f"Expected teacher_displacement of shape {tuple(coords.shape)} but "
                    f"got tensor of shape {tuple(teacher_displacement.shape)}"
                )
            if getattr(graph, "mesh_edge_index", None) is None or any(
                getattr(graph, name, None) is None
                for name in ("mesh_edge_attr", "mesh_edge_mean", "mesh_edge_std")
            ):
                raise ValueError(
                    "DeFormerCorrector needs graph.mesh_edge_index, mesh_edge_attr, "
                    "mesh_edge_mean and mesh_edge_std"
                )
            if self.use_contact and (position_mean is None or position_std is None):
                raise ValueError(
                    "position_mean and position_std are required for the contact search"
                )
        edge_index = graph.mesh_edge_index
        prior = teacher_displacement.to(coords.dtype)
        predicted = coords + prior
        with torch.autocast(device_type=coords.device.type, enabled=False):
            mean = graph.mesh_edge_mean.to(device=coords.device, dtype=torch.float32)
            std = graph.mesh_edge_std.to(device=coords.device, dtype=torch.float32)
            predicted_attr = (
                mesh_edge_features(predicted.float(), edge_index) - mean
            ) / (std + EPS)
            reference_attr = graph.mesh_edge_attr.float()
            edge_in = torch.cat(
                (reference_attr, predicted_attr, predicted_attr - reference_attr),
                dim=-1,
            )
        node_latent = self.node_encoder(node_features)
        edge_latent = self.mesh_edge_encoder(edge_in.to(node_features.dtype))
        if self.use_contact:
            contact_index, contact_latent = self.contact_edges(
                predicted, coords, graph, position_mean, position_std
            )
            if contact_index is not None:
                edge_index = torch.cat((edge_index, contact_index), dim=1)
                edge_latent = torch.cat(
                    (edge_latent, contact_latent.to(edge_latent.dtype)), dim=0
                )
            else:
                node_latent = node_latent + contact_latent.to(node_latent.dtype)
        latent = self.processor(node_latent, edge_latent, _edge_graph(edge_index, n))
        return predicted + self.decoder(latent).to(coords.dtype)
