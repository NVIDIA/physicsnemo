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
"""MeshGraphNet corrector: local message passing on a teacher's predicted shape.

The corrector is the second half of DeFormer. A frozen time-conditional
network (the teacher) predicts the deformed shape of a frame; the corrector
reads that shape and adds a correction by message passing on the structural
mesh:

* every mesh edge carries its reference geometry, the same geometry measured
  on the predicted shape and their difference (the local stretch, bend and
  fold of the prediction), standardised with the reference edge statistics;
* contact pairs discovered on the predicted shape by the crossing-aware
  node-to-facet search of :mod:`physicsnemo.experimental.nn.shell_contact`
  become extra edges between the facet vertices and the node (in both
  directions), carrying the contact geometry, the barycentric weight of the
  vertex and the direction sign: the "world edges" of MeshGraphNet collision
  models;
* the node inputs are the coordinates, the static features, the time code
  and the teacher's displacement (and, with ``teacher_window``, the teacher's
  motion around the queried frame as backward and forward differences).

There is no global block, so the receptive field is ``processor_size`` hops.
The decoder is zero-initialised, so the corrector starts exactly at the
teacher. One forward corrects one frame; :class:`residual_model.ResidualTimeConditional`
runs the teacher, serves its fields and loops over frames.
"""

from __future__ import annotations

import math

import torch
from local_neighborhoods import mesh_edge_features
from rollout import TimeEmbedding
from time_conditional import TimeConditionalOutput, drop_retired_options
from torch import nn

import physicsnemo
from physicsnemo.models.meshgraphnet.meshgraphnet import MeshGraphNetProcessor
from physicsnemo.experimental.nn.shell_contact import (
    CONTACT_FEATURE_DIM,
    ShellContactSearch,
    shell_contact_edge_features,
    shell_contact_pairs,
)
from physicsnemo.nn.module.gnn_layers.graph_types import PyGData
from physicsnemo.nn.module.gnn_layers.mesh_graph_mlp import MeshGraphMLP

EPS = 1.0e-8
MESH_EDGE_INPUT_DIM = 12
"""Reference geometry (4) + predicted geometry (4) + their difference (4)."""
CONTACT_EDGE_INPUT_DIM = CONTACT_FEATURE_DIM + 2
"""Contact geometry (8) + barycentric weight + direction sign."""

_RETIRED_OPTIONS = {"contact_oracle": False}


class MeshGraphNetCorrector(physicsnemo.Module):
    """Local message-passing corrector of a teacher's prediction for one frame.

    Parameters
    ----------
    num_time_steps : int
        Frames per trajectory in the datapipe (``num_time_steps - 1`` future
        frames, labelled ``k / (num_time_steps - 1)`` as in the teacher).
    functional_dim : int
        Width of the node inputs: coords (3) + static features + time code
        (``time_dim``) + the teacher's displacement (3), + 6 with
        ``teacher_window``.
    hidden_dim, processor_size, aggregation
        MeshGraphNet width, number of message-passing layers and message
        aggregation (``"sum"`` or ``"mean"``).
    num_processor_checkpoint_segments : int
        Activation-checkpoint segments across the processor layers (0: none);
        ``activation_checkpointing=True`` with 0 checkpoints every layer.
    time_embedding, time_fourier_bands, time_dim
        The time code (as in the teacher).
    dt : float, optional
        Frame interval of the datapipe in seconds (bookkeeping only).
    use_contact : bool
        Contact edges from the crossing-aware surface search on the predicted
        shape; needs ``graph.contact_faces``, ``contact_surface_exclusions``
        and ``shell_thickness`` (``datapipe.contact_surface=true``).
    teacher_window : bool
        Expect the teacher's displacement one frame earlier and one frame
        later (``teacher_window`` argument of :meth:`forward`) and add the
        backward and forward differences, scaled by the number of future
        frames, to the node inputs.
    contact_search_implementation, contact_surface_max_pairs,
    contact_activation_distance, node_contact_radius, contact_normal_epsilon,
    contact_surface_material_fan, contact_side_from_reference,
    contact_reference_clearance, contact_side_symmetric, contact_crossing_radius,
    contact_clearance_approach_only
        Settings of the contact search (:class:`~physicsnemo.experimental.nn.shell_contact.ShellContactSearch`;
        ``node_contact_radius`` is the feature scale in mm, the activation
        distance the band, the reference clearance drops assembled pairs).
    contact_bidirectional : bool
        Also send the mirrored message from the node to the facet vertices.
    activation_checkpointing : bool
        Checkpoint every processor layer when ``num_processor_checkpoint_segments``
        is 0 (the full Yaris mesh has about six million directed shell edges plus
        the contact edges).
    local_knn_k, local_neighborhood
        Accepted and ignored: the datapipe reads them from the model config to
        build the teacher's neighbourhoods.
    """

    def __init__(
        self,
        *,
        num_time_steps: int,
        functional_dim: int = 39,
        hidden_dim: int = 128,
        processor_size: int = 8,
        num_processor_checkpoint_segments: int = 0,
        aggregation: str = "sum",
        time_embedding: str = "fourier",
        time_fourier_bands: int = 8,
        time_dim: int = 32,
        dt: float | None = None,
        use_contact: bool = True,
        teacher_window: bool = False,
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
        local_knn_k: int | None = None,
        local_neighborhood: str | None = None,
        **retired,
    ) -> None:
        super().__init__()
        unknown = drop_retired_options(retired, _RETIRED_OPTIONS, {})
        if unknown:
            raise TypeError(f"unexpected arguments: {sorted(unknown)}")
        if num_time_steps < 2:
            raise ValueError("num_time_steps must be at least 2")
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
        if dt is not None and not dt > 0:
            raise ValueError("dt must be positive when given")
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
        self.rollout_steps = int(num_time_steps) - 1
        self.dt = None if dt is None else float(dt)
        self.functional_dim = int(functional_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_embed = TimeEmbedding(time_embedding, time_fourier_bands, time_dim)
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
        self.teacher_window = bool(teacher_window)
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
        self.__name__ = "MeshGraphNetCorrector"

    # ----- contact edges -----
    def contact_edges(self, predicted, sample, data_stats):
        """Expanded contact edges on the predicted shape (normalized positions).

        Returns ``(edge_index [2, E], edge_latent [E, hidden])`` or ``(None,
        placeholder)`` on a frame without contact, where the zero-valued
        placeholder keeps every contact parameter in the autograd graph
        (distributed gradient averaging needs the same layout on every rank).
        """
        graph = sample.graph
        coords = sample.node_features["coords"]
        faces = getattr(graph, "contact_faces", None)
        exclusions = getattr(graph, "contact_surface_exclusions", None)
        thickness = getattr(graph, "shell_thickness", None)
        if faces is None or exclusions is None or thickness is None:
            raise ValueError(
                "contact edges need graph.contact_faces, contact_surface_exclusions "
                "and shell_thickness (datapipe.contact_surface=true)"
            )
        with torch.autocast(device_type=predicted.device.type, enabled=False):
            scale = data_stats["node"]["pos_std"].to(predicted.device).float()
            mean = data_stats["node"]["pos_mean"].to(predicted.device).float()
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
        sample,
        data_stats,
        teacher_displacement: torch.Tensor,
        teacher_window: torch.Tensor | None = None,
        return_auxiliary: bool = False,
    ):
        """Correct the teacher's prediction at the sample's time label.

        Parameters
        ----------
        sample : SimSample
            Node features (``coords``, ``features``, ``time``) and the graph with
            ``mesh_edge_index`` / ``mesh_edge_attr`` (``datapipe.mesh_edges=true``).
        data_stats : dict
            Normalization statistics of the datapipe.
        teacher_displacement : torch.Tensor
            ``[N, 3]`` the teacher's normalized displacement at this time.
        teacher_window : torch.Tensor, optional
            ``[N, 2, 3]`` the teacher's displacement one frame earlier and one
            frame later (required with ``teacher_window=True``).
        return_auxiliary : bool
            Return a :class:`TimeConditionalOutput` (whose provisional position
            is the final one) instead of the position tensor.

        Returns
        -------
        torch.Tensor or TimeConditionalOutput
            Corrected normalized positions ``[N, 3]``.
        """
        features = sample.node_features
        graph = sample.graph
        coords = features["coords"]
        n = coords.shape[0]
        time = float(features["time"])
        if not 0.0 <= time <= (self.rollout_steps - 1) / self.rollout_steps + 1e-6:
            raise ValueError(
                "time is outside the datapipe's normalized future-frame range"
            )
        prior = teacher_displacement.to(coords.dtype)
        static = features.get("features", coords.new_zeros((n, 0)))
        code = self.time_embed(features["time"]).to(coords.dtype).expand(n, -1)
        inputs = [coords, static, code, prior]
        if self.teacher_window:
            if teacher_window is None or tuple(teacher_window.shape) != (n, 2, 3):
                raise ValueError(
                    "teacher_window=True needs the teacher's displacement one frame "
                    "earlier and one frame later as a [N, 2, 3] tensor"
                )
            window = teacher_window.to(coords.dtype)
            # Backward and forward differences of the teacher's displacement,
            # scaled by the number of future frames so that a steady per-frame
            # change is of the prior's magnitude: the teacher's velocity before
            # and after the queried time, in normalized positions per unit label.
            scale = float(self.rollout_steps)
            inputs.append((prior - window[:, 0]) * scale)
            inputs.append((window[:, 1] - prior) * scale)
        node_in = torch.cat(inputs, dim=-1)
        if node_in.shape[1] != self.functional_dim:
            raise ValueError(
                f"node inputs are {node_in.shape[1]} wide; functional_dim="
                f"{self.functional_dim} must count coords, static features, the "
                "time code, the teacher displacement and the window differences if any"
            )
        edge_index = getattr(graph, "mesh_edge_index", None)
        if edge_index is None:
            raise ValueError(
                "MeshGraphNetCorrector needs graph.mesh_edge_index and mesh_edge_attr "
                "(datapipe.mesh_edges=true)"
            )
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
        node_latent = self.node_encoder(node_in)
        edge_latent = self.mesh_edge_encoder(edge_in.to(node_in.dtype))
        if self.use_contact:
            contact_index, contact_latent = self.contact_edges(
                predicted, sample, data_stats
            )
            if contact_index is not None:
                edge_index = torch.cat((edge_index, contact_index), dim=1)
                edge_latent = torch.cat(
                    (edge_latent, contact_latent.to(edge_latent.dtype)), dim=0
                )
            else:
                node_latent = node_latent + contact_latent.to(node_latent.dtype)
        latent = self.processor(
            node_latent, edge_latent, PyGData(edge_index=edge_index, num_nodes=n)
        )
        prediction = predicted + self.decoder(latent).to(coords.dtype)
        output = TimeConditionalOutput(prediction, prediction)
        return output if return_auxiliary else output.prediction
