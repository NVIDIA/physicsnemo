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
"""Independent-time DeFormer prediction with latent-preserving refinement.

For a requested future frame the model maps the initial mesh, its static
features and a time code to a deformed shape:

    Enc(X0, tau, q) -> pre-processor -> FLARE -> post stage A -> coarse head
                                                     |-> post stage B -> final

The coarse head is a zero-initialized side output of the full-width stage-A
latent; stage B is an independent copy of the post-processor whose correction is
added to the coarse position through the post residual gate. Both stages use
whatever local operator the base model is configured with (mesh MPNN, static
kNN attention, or MPNN messages on kNN slots), FiLM-conditioned on the mesh
context. There is no recurrent state and no BPTT: training samples independent
case-time pairs, evaluation queries every future frame from the initial mesh.

With ``use_contact=True`` the provisional (coarse) shape drives surface contact:
node-to-facet pairs are discovered on the stage-A position in physical units,
their geometry messages enter the stage-A latent through a gated contact block,
and stage B refines with that information. Contact never touches the coarse
head, so the provisional shape stays contact-free, and the contact features
are differentiable in the coarse position.
"""

from __future__ import annotations

import copy
import math
from typing import NamedTuple

import torch
from rigid_wall_contact import RigidWallContactEncoder
from rollout import TimeEmbedding, _global_tokens
from torch import nn

from physicsnemo.experimental.models.meshtransolver.conditional_norm import (
    convert_layer_norms,
    set_norm_condition,
)
from physicsnemo.experimental.models.meshtransolver.contact import (
    CONTACT_FEATURE_DIM,
    ContactGraph,
    SparseContactBlock,
    merge_contact_graphs,
)
from physicsnemo.experimental.models.meshtransolver.meshtransolver import MeshGeoFLARE
from physicsnemo.experimental.models.meshtransolver.surface_contact import (
    SurfaceContactGraphBuilder,
)


class TimeConditionalOutput(NamedTuple):
    prediction: torch.Tensor
    coarse_prediction: torch.Tensor


class MeshGeoFLARETimeConditional(MeshGeoFLARE):
    """Initial geometry, static features and a time code to a deformed shape.

    Parameters
    ----------
    num_time_steps : int
        Frames per trajectory in the datapipe; ``num_time_steps - 1`` future
        frames are queried at evaluation, labelled ``k / (num_time_steps - 1)``
        for ``k = 0, ...`` (the datapipe's ``one_time_step`` convention: code 0
        is the first future frame).
    time_embedding : {"scalar", "fourier"}
        Time code entering the node inputs. ``functional_dim`` must count its
        width (1 for scalar, ``time_dim`` for fourier).
    time_fourier_bands, time_dim : int
        Fourier harmonics and output width of the learned time code.
    time_as_global : bool
        Also feed the time code as a global token; ``global_dim`` must then
        equal ``time_dim`` plus the number of dataset global features.
    dt : float, optional
        Frame interval used by the datapipe; accepted for recipe compatibility
        (Yaris derives ``datapipe.dt`` from ``model.dt``) and stored only.
    norm_conditioning : {"none", "time"}
        ``"time"`` replaces every LayerNorm of the network (FLARE blocks, local
        processors, FiLM adapters and heads) by a conditional LayerNorm whose
        scale and offset are predicted from the time code through a
        zero-initialised linear layer (GenCast norm conditioning, adaLN-zero),
        so every layer is modulated by time rather than only the node inputs
        and the global token. Function-preserving at initialisation.
    use_contact : bool
        Surface contact on the provisional shape between the two post stages
        (see the module docstring). Requires ``graph.shell_thickness``,
        ``graph.contact_faces`` and ``graph.contact_surface_exclusions`` from
        the datapipe (``datapipe.contact_surface=true``).
    contact_dim, contact_gate_init, contact_aggregation
        Contact block width, residual gate and aggregation, as in MeshGeoFLARE.
        Geometry-only features are ``CONTACT_FEATURE_DIM`` wide.
    contact_search_implementation, contact_surface_max_pairs,
    contact_activation_distance, node_contact_radius, contact_normal_epsilon,
    contact_surface_material_fan
        Surface-contact search settings with the autoregressive recipe's names
        (``node_contact_radius`` is the feature scale).
    contact_side_from_reference : bool
        Crossing-aware signed gaps: a node on the other side of a facet than
        in the reference configuration has passed through the shell, so its
        gap turns negative and its direction flips (see
        ``SurfaceContactGraphBuilder``). Off by default.
    contact_reference_clearance : float or None
        Drop pairs whose reference activation gap is below this length
        (assembly neighbours already inside the structural neighbourhood), so
        the contact graph holds new contact only. ``None`` keeps every pair.
    contact_iterations : int
        Weight-tied correction passes. Pass 1 discovers contact on the
        provisional shape; every further pass rebuilds the contact graph on the
        previous pass's final shape and reruns the same stage B from the same
        stage-A latent, so late passes see a better shape than the provisional
        one. The loss applies to the last pass; everything stays differentiable.
    global_motion : bool
        Adds a global rigid-motion head: the mean stage-A latent and the time
        code give one translation (``global_motion_scale_mm`` per unit output)
        and one small rotation about the reference centroid (``scale / lever``
        radians per unit output) per graph, applied to every node and added to
        the provisional position. Zero-initialised, so it is function-preserving
        at the start. Motivated by the Yaris error structure: about half of the
        held-out squared error is a per-frame translation of whole regions.
    contact_include_velocity, contact_surface_predictive : bool
        Must stay ``False``: one query per time has no velocity estimate, so
        contact is discovered on the current provisional geometry only.
    two_stage : bool
        ``False`` drops the provisional head and the independent stage B: one
        post stack (``num_post_processor_layers`` deep) feeds the single gated
        head, the auxiliary output mirrors the prediction, and
        ``training.coarse_loss_weight`` should be 0. The ablation for the
        stage split itself; incompatible with provisional-shape contact.
    num_refinement_layers : int, optional
        Depth of stage B. ``None`` (default) keeps the original design: an
        independent, re-initialised copy of stage A. An integer builds stage B
        fresh with that many layers of the configured local operator, so the
        contact correction can be deeper than the provisional stage.
    refinement_input_displacement : bool
        Feed the provisional displacement (coarse minus initial position,
        normalized units) into stage B through a zero-initialised linear
        adapter, so the correction stage sees the prediction it corrects and
        not only the latent and the contact messages. Function-preserving at
        initialisation. Requires ``two_stage=True``.
    """

    def __init__(
        self,
        *,
        num_time_steps: int,
        time_embedding: str = "fourier",
        time_fourier_bands: int = 8,
        time_dim: int = 32,
        time_as_global: bool = True,
        dt: float | None = None,
        norm_conditioning: str = "none",
        use_contact: bool = False,
        contact_dim: int = CONTACT_FEATURE_DIM,
        contact_gate_init: float = 1.0e-3,
        contact_aggregation: str = "mean",
        contact_search_implementation: str = "warp",
        contact_surface_max_pairs: int = 8_000_000,
        contact_activation_distance: float = 5.0,
        node_contact_radius: float = 10.0,
        contact_normal_epsilon: float = 0.1,
        contact_surface_material_fan: bool = True,
        contact_include_velocity: bool = False,
        contact_surface_predictive: bool = False,
        contact_side_from_reference: bool = False,
        contact_reference_clearance: float | None = None,
        contact_iterations: int = 1,
        global_motion: bool = False,
        global_motion_scale_mm: float = 10.0,
        global_motion_lever_mm: float = 2000.0,
        part_tokens: bool = False,
        part_token_dim: int = 5,
        design_dim: int = 32,
        contact_source: str = "surface",
        wall_axis: int = 0,
        wall_sign: float = 1.0,
        clamp_to_wall: bool = False,
        two_stage: bool = True,
        num_refinement_layers: int | None = None,
        refinement_input_displacement: bool = False,
        **kwargs,
    ) -> None:
        if num_time_steps < 2:
            raise ValueError("num_time_steps must be at least 2")
        if use_contact and not two_stage:
            raise ValueError(
                "Provisional-shape contact needs the provisional head: two_stage=True"
            )
        if not two_stage and (
            num_refinement_layers is not None or refinement_input_displacement
        ):
            raise ValueError(
                "num_refinement_layers and refinement_input_displacement describe "
                "stage B and need two_stage=True"
            )
        if num_refinement_layers is not None and num_refinement_layers < 1:
            raise ValueError("num_refinement_layers must be at least 1 when given")
        if norm_conditioning not in ("none", "time"):
            raise ValueError("norm_conditioning must be 'none' or 'time'")
        if dt is not None and not dt > 0:
            raise ValueError("dt must be positive when given")
        if kwargs.get("out_dim", kwargs.get("output_dim", 3)) != 3:
            raise ValueError("This recipe predicts positions only (out_dim=3)")
        if contact_include_velocity or contact_surface_predictive:
            raise ValueError(
                "Provisional-shape contact is geometry-only: set "
                "contact_include_velocity=False and contact_surface_predictive=False"
            )
        if contact_source not in ("surface", "wall", "both"):
            raise ValueError("contact_source must be 'surface', 'wall' or 'both'")
        if (
            isinstance(contact_iterations, bool)
            or int(contact_iterations) != contact_iterations
            or contact_iterations < 1
        ):
            raise ValueError("contact_iterations must be a positive integer")
        if contact_iterations > 1 and not use_contact:
            raise ValueError("contact_iterations > 1 needs use_contact=True")
        self.contact_iterations = int(contact_iterations)
        if clamp_to_wall and contact_source == "surface":
            raise ValueError(
                "clamp_to_wall needs the rigid wall: contact_source wall or both"
            )
        if kwargs.get("use_te", False):
            raise ValueError("Latent refinement requires use_te=False")
        # The backbone never applies contact; it happens between the post stages.
        kwargs["use_contact"] = False
        super().__init__(**kwargs)
        self.rollout_steps = int(num_time_steps) - 1
        # Frame interval of the datapipe (bookkeeping only: recipes that derive
        # the datapipe dt from model.dt pass it; prediction never integrates).
        self.dt = None if dt is None else float(dt)
        self.time_embed = TimeEmbedding(time_embedding, time_fourier_bands, time_dim)
        self.time_as_global = bool(time_as_global)
        # Full-width FLARE output (hidden plus optional local features) to the
        # mesh width, never through the three-channel output.
        effective_dim = self.ln_mlp_out[0][0].normalized_shape[0]
        self.post_latent_input = (
            nn.Identity()
            if effective_dim == self.mesh_hidden_dim
            else nn.Linear(effective_dim, self.mesh_hidden_dim)
        )
        self.two_stage = bool(two_stage)
        if self.two_stage:
            self.coarse_decoder = nn.Sequential(
                nn.LayerNorm(self.mesh_hidden_dim),
                nn.Linear(self.mesh_hidden_dim, self.output_dim),
            )
            # Start from the undeformed mesh: coarse and final heads are zero.
            nn.init.zeros_(self.coarse_decoder[-1].weight)
            nn.init.zeros_(self.coarse_decoder[-1].bias)
            if num_refinement_layers is None:
                # Original design: an independent, re-initialised copy of stage A.
                self.refinement_processor = copy.deepcopy(self.post_processor)
                for module in self.refinement_processor.modules():
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()
            else:
                self.refinement_processor = self.make_local_processor(
                    int(num_refinement_layers),
                    int(kwargs.get("num_post_processor_checkpoint_segments", 0)),
                )
            self.refinement_context_film = copy.deepcopy(self.post_context_film)
            if refinement_input_displacement:
                self.coarse_feedback = nn.Linear(self.output_dim, self.mesh_hidden_dim)
                nn.init.zeros_(self.coarse_feedback.weight)
                nn.init.zeros_(self.coarse_feedback.bias)
            else:
                self.coarse_feedback = None
        else:
            self.coarse_decoder = None
            self.refinement_processor = None
            self.refinement_context_film = None
            self.coarse_feedback = None
        self.num_refinement_layers = num_refinement_layers
        del self.post_node_encoder
        last_linear = [
            m for m in self.post_node_decoder.modules() if isinstance(m, nn.Linear)
        ][-1]
        nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)
        # Contact modules come after every shared module, so a contact model
        # and its contact-free control share parameter initialisation exactly.
        self.provisional_contact = bool(use_contact)
        self.provisional_contact_dim = int(contact_dim)
        if self.provisional_contact:
            self.refinement_contact_block = SparseContactBlock(
                hidden_dim=self.mesh_hidden_dim,
                contact_dim=contact_dim,
                gate_init=contact_gate_init,
                aggregation=contact_aggregation,
            )
            self.node_contact_builder = (
                SurfaceContactGraphBuilder(
                    activation_distance=contact_activation_distance,
                    feature_scale=node_contact_radius,
                    normal_epsilon=contact_normal_epsilon,
                    include_velocity=False,
                    implementation=contact_search_implementation,
                    max_pairs=contact_surface_max_pairs,
                    prediction_horizon=0.0,
                    material_fan=contact_surface_material_fan,
                    side_from_reference=contact_side_from_reference,
                    reference_clearance=contact_reference_clearance,
                )
                if contact_source in ("surface", "both")
                else None
            )
            self.wall_contact = (
                RigidWallContactEncoder(
                    axis=wall_axis,
                    sign=wall_sign,
                    activation_distance=contact_activation_distance,
                    feature_scale=node_contact_radius,
                    normal_epsilon=contact_normal_epsilon,
                )
                if contact_source in ("wall", "both")
                else None
            )
        else:
            self.refinement_contact_block = None
            self.node_contact_builder = None
            self.wall_contact = None
        self.contact_source = contact_source
        self.clamp_to_wall = bool(clamp_to_wall)
        # Global rigid-motion head: the mean stage-A latent and the time code
        # predict one translation and one small rotation (about the reference
        # centroid) per graph, added to every node. Zero-initialised output.
        # Part-pooled design encoder: per-token MLP, attention pooling with a
        # learned query, zero-initialised projection to design_dim. The design
        # vector joins the global token (global_dim must include design_dim)
        # and, with global_motion, the rigid-motion head's input.
        self.part_tokens = bool(part_tokens)
        self.design_dim = int(design_dim) if self.part_tokens else 0
        if self.part_tokens:
            if part_token_dim <= 0 or design_dim <= 0:
                raise ValueError("part_token_dim and design_dim must be positive")
            width = max(64, self.design_dim)
            self.design_token_mlp = nn.Sequential(
                nn.Linear(part_token_dim, width), nn.GELU(), nn.Linear(width, width)
            )
            self.design_pool_query = nn.Parameter(torch.zeros(width))
            self.design_out = nn.Linear(width, self.design_dim)
            nn.init.zeros_(self.design_out.weight)
            nn.init.zeros_(self.design_out.bias)
        else:
            self.design_token_mlp = None
            self.design_pool_query = None
            self.design_out = None
        self.global_motion = bool(global_motion)
        if global_motion_scale_mm <= 0 or global_motion_lever_mm <= 0:
            raise ValueError(
                "global_motion_scale_mm and global_motion_lever_mm must be positive"
            )
        self.global_motion_scale_mm = float(global_motion_scale_mm)
        self.global_motion_lever_mm = float(global_motion_lever_mm)
        if self.global_motion:
            self.global_motion_head = nn.Sequential(
                nn.Linear(
                    self.mesh_hidden_dim + self.time_embed.dim + self.design_dim,
                    self.mesh_hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(self.mesh_hidden_dim, 6),
            )
            nn.init.zeros_(self.global_motion_head[-1].weight)
            nn.init.zeros_(self.global_motion_head[-1].bias)
        else:
            self.global_motion_head = None
        # Last, so that both post stages, the contact block and every head are covered.
        self.norm_conditioning = norm_conditioning
        self.num_conditional_norms = (
            convert_layer_norms(self, self.time_embed.dim)
            if norm_conditioning == "time"
            else 0
        )
        self.__name__ = "MeshGeoFLARETimeConditional"

    # ----- building blocks -----
    def _design_vector(self, sample):
        """``[1, design_dim]`` from the per-part tokens, permutation invariant."""
        if not self.part_tokens:
            return None
        parts = (
            None
            if sample.global_features is None
            else sample.global_features.get("parts")
        )
        if parts is None:
            raise ValueError("part_tokens=True needs sample.global_features['parts']")
        if parts.ndim == 3:
            parts = parts.reshape(-1, parts.shape[-1])
        tokens = self.design_token_mlp(parts.to(self.design_out.weight.dtype))
        weights = torch.softmax(tokens @ self.design_pool_query, dim=0)
        pooled = (weights[:, None] * tokens).sum(dim=0, keepdim=True)
        return self.design_out(pooled)

    def _time_inputs(self, sample, num_nodes: int):
        coords = sample.node_features["coords"]
        code = self.time_embed(sample.node_features["time"])
        if self.norm_conditioning == "time":
            # One condition per forward pass, read by every conditional norm
            # (also during activation-checkpoint recomputation in backward).
            set_norm_condition(self, code)
        code = code.to(coords.dtype)
        global_embedding = _global_tokens(sample)
        if self.time_as_global:
            token = code.view(1, 1, -1)
            global_embedding = (
                token
                if global_embedding is None
                else torch.cat((global_embedding, token), dim=-1)
            )
        design = self._design_vector(sample)
        if design is not None:
            design = design.to(coords.dtype).view(1, 1, -1)
            global_embedding = (
                design
                if global_embedding is None
                else torch.cat((global_embedding, design), dim=-1)
            )
        return code.expand(num_nodes, -1), global_embedding

    def coarse(self, sample):
        """Provisional position and the full-width stage-A latent at the sample's time."""
        coords = sample.node_features["coords"]
        features = sample.node_features.get(
            "features", coords.new_zeros((coords.shape[0], 0))
        )
        time_code, global_embedding = self._time_inputs(sample, coords.shape[0])
        nodes = torch.cat((coords, features, time_code), dim=-1)
        state = MeshGeoFLARE.forward(
            self,
            node_features=nodes,
            edge_features=getattr(sample.graph, "edge_attr", None),
            graph=sample.graph,
            geometry=coords,
            global_embedding=global_embedding,
            local_positions=coords,
            return_backbone=True,
        )
        latent = self.post_latent_input(state.point_features)
        if self.post_context_film is not None:
            latent = self.post_context_film(latent, state.mesh_context)
        latent = self._run_local_processor(
            self.post_processor,
            latent,
            state.edge_latent,
            state.graph,
            state.neighborhood,
        )
        if not self.two_stage:
            return None, latent, state
        coarse = coords + self.coarse_decoder(latent).to(coords.dtype)
        baseline = sample.node_features.get("baseline")
        if baseline is not None:
            # Output baseline (e.g. the training-mean displacement at this
            # time): the heads carry the residual.
            coarse = coarse + baseline.to(coarse.dtype)
        return coarse, latent, state

    def _wall_position(self, sample):
        wall = getattr(sample.graph, "rigid_wall", None)
        if wall is None:
            raise ValueError(
                "Rigid-wall contact needs graph.rigid_wall (datapipe.rigid_wall_position)"
            )
        return float(wall.reshape(-1)[0])

    def _physical_geometry(self, normalized, sample, data_stats):
        """Positions, reference positions and nodal thickness in float32 millimetres."""
        thickness = getattr(sample.graph, "shell_thickness", None)
        if thickness is None:
            raise ValueError("Provisional-shape contact needs graph.shell_thickness")
        scale = data_stats["node"]["pos_std"].to(normalized.device).float()
        mean = data_stats["node"]["pos_mean"].to(normalized.device).float()
        positions = normalized.float() * scale + mean
        reference = sample.node_features["coords"].detach().float() * scale + mean
        thickness = thickness.to(device=normalized.device, dtype=torch.float32)
        return positions, reference, thickness.reshape(-1)

    def _surface_contact_graph(self, positions, reference, thickness, sample):
        """TYPE7-like node-to-facet search on physical positions."""
        graph = sample.graph
        faces = getattr(graph, "contact_faces", None)
        exclusions = getattr(graph, "contact_surface_exclusions", None)
        if faces is None or exclusions is None:
            raise ValueError(
                "Surface contact needs graph.contact_faces and "
                "graph.contact_surface_exclusions (datapipe.contact_surface=true)"
            )
        return self.node_contact_builder(
            positions,
            structural_edge_index=graph.edge_index,
            batch=getattr(graph, "batch", None),
            shell_thickness=thickness,
            faces=faces,
            excluded_pairs=exclusions,
            reference_positions=reference,
        )

    def _contact_graph(self, coarse, sample, data_stats) -> ContactGraph:
        """Contact discovered on the provisional shape, in physical units.

        Surface contact (node-to-facet, TYPE7-like) and/or the analytic rigid
        wall, merged into one contact graph for the stage-B contact block.
        """
        # Geometry arithmetic and the search run in float32 physical units even
        # when the neural processors run under autocast.
        with torch.autocast(device_type=coarse.device.type, enabled=False):
            positions, reference, thickness = self._physical_geometry(
                coarse, sample, data_stats
            )
            graphs = []
            if self.node_contact_builder is not None:
                graphs.append(
                    self._surface_contact_graph(positions, reference, thickness, sample)
                )
            if self.wall_contact is not None:
                graphs.append(
                    self.wall_contact(positions, thickness, self._wall_position(sample))
                )
            return graphs[0] if len(graphs) == 1 else merge_contact_graphs(*graphs)

    def penetration_penalty(
        self,
        prediction,
        sample,
        data_stats,
        *,
        tolerance_mm: float = 0.0,
        length_scale_mm: float = 1.0,
    ):
        """Non-penetration penalty of one predicted frame, with diagnostics.

        The surface-contact search runs on the prediction in physical units.
        Every active pair's signed gap (negative inside the thickness envelope
        and, with ``contact_side_from_reference``, below minus the half
        thickness once a node has crossed a shell) becomes a depth
        ``relu(-gap - tolerance_mm)``; each node keeps its deepest pair and the
        loss is the mean over all nodes of ``(depth / length_scale_mm) ** 2``,
        the units of a physical position loss with the same scale. Zero on a
        prediction without penetration; the geometry gradient reaches the
        prediction through the live closest-point distances. Pairs dropped by
        ``contact_reference_clearance`` are not penalised: the penalty is about
        new contact, like the contact graph.
        """
        if self.node_contact_builder is None:
            raise ValueError(
                "penetration_penalty needs surface contact: use_contact=True with "
                "contact_source 'surface' or 'both'"
            )
        if not math.isfinite(tolerance_mm) or tolerance_mm < 0:
            raise ValueError("tolerance_mm must be finite and nonnegative")
        if not math.isfinite(length_scale_mm) or length_scale_mm <= 0:
            raise ValueError("length_scale_mm must be finite and positive")
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            positions, reference, thickness = self._physical_geometry(
                prediction, sample, data_stats
            )
            graph = self._surface_contact_graph(positions, reference, thickness, sample)
            # Channel 4 of the geometry features is the signed gap over the
            # feature scale (see contact_edge_features).
            gap = graph.edge_features[:, 4] * self.node_contact_builder.feature_scale
            depth = (-gap - tolerance_mm).clamp_min(0.0)
            node_depth = positions.new_zeros(positions.shape[0]).scatter_reduce(
                0, graph.edge_index[1], depth, reduce="amax", include_self=True
            )
            loss = (node_depth / length_scale_mm).square().mean()
            stats = self.node_contact_builder.last_discovery_stats or {}
            diagnostics = {
                "penetration_active_nodes": (node_depth > 0).sum().to(loss.dtype),
                "penetration_max_excess_mm": node_depth.detach().max(),
                "penetration_crossed_pairs": loss.new_tensor(
                    float(stats.get("crossed_pairs", 0))
                ),
            }
        return loss, diagnostics

    def _clamp(self, prediction, sample, data_stats):
        """Project the prediction onto the admissible side of the rigid wall."""
        if not self.clamp_to_wall:
            return prediction
        axis = self.wall_contact.axis
        std = data_stats["node"]["pos_std"].reshape(-1)[axis].to(prediction)
        mean = data_stats["node"]["pos_mean"].reshape(-1)[axis].to(prediction)
        limit = (self._wall_position(sample) - mean) / (std + 1e-12)
        column = prediction[:, axis]
        column = (
            torch.minimum(column, limit)
            if self.wall_contact.sign > 0
            else torch.maximum(column, limit)
        )
        return torch.cat(
            (prediction[:, :axis], column.unsqueeze(-1), prediction[:, axis + 1 :]),
            dim=-1,
        )

    def refine(
        self,
        latent,
        state,
        contact_graph: ContactGraph | None = None,
        coarse_displacement: torch.Tensor | None = None,
    ):
        """Final correction from an independent second post stage."""
        if self.refinement_contact_block is not None:
            if contact_graph is None:
                raise ValueError("contact_graph is required when use_contact=True")
            latent = latent + self.refinement_contact_block(
                latent, contact_graph, return_correction=True
            ).to(latent.dtype)
        elif contact_graph is not None:
            raise ValueError("contact_graph was provided but use_contact=False")
        if self.coarse_feedback is not None:
            if coarse_displacement is None:
                raise ValueError("coarse_displacement is required by the adapter")
            latent = latent + self.coarse_feedback(
                coarse_displacement.to(self.coarse_feedback.weight.dtype)
            ).to(latent.dtype)
        if self.refinement_context_film is not None:
            latent = self.refinement_context_film(latent, state.mesh_context)
        latent = self._run_local_processor(
            self.refinement_processor,
            latent,
            state.edge_latent,
            state.graph,
            state.neighborhood,
        )
        return self.mesh_post_residual_gate * self.post_node_decoder(latent)

    def _global_motion(self, latent, sample, data_stats):
        """Normalized displacement [N, 3] of one rigid motion per graph."""
        coords = sample.node_features["coords"]
        n = coords.shape[0]
        batch = getattr(sample.graph, "batch", None)
        if batch is None:
            batch = torch.zeros(n, dtype=torch.long, device=coords.device)
        num_graphs = int(batch.max().item()) + 1 if n else 1
        counts = torch.zeros(num_graphs, device=coords.device, dtype=torch.float32)
        counts.index_add_(0, batch, torch.ones(n, device=coords.device))
        counts = counts.clamp_min(1.0)
        pooled = (
            torch.zeros(
                (num_graphs, latent.shape[1]), device=coords.device, dtype=torch.float32
            ).index_add_(0, batch, latent.float())
            / counts[:, None]
        )
        code = self.time_embed(sample.node_features["time"]).float().reshape(1, -1)
        head_inputs = [pooled, code.expand(num_graphs, -1)]
        design = self._design_vector(sample)
        if design is not None:
            head_inputs.append(design.float().expand(num_graphs, -1))
        params = self.global_motion_head(
            torch.cat(head_inputs, dim=-1).to(self.global_motion_head[0].weight.dtype)
        ).float()
        with torch.autocast(device_type=coords.device.type, enabled=False):
            std = data_stats["node"]["pos_std"].to(coords.device).float().reshape(1, 3)
            mean = (
                data_stats["node"]["pos_mean"].to(coords.device).float().reshape(1, 3)
            )
            physical = coords.float() * std + mean
            centroid = (
                torch.zeros(
                    (num_graphs, 3), device=coords.device, dtype=torch.float32
                ).index_add_(0, batch, physical)
                / counts[:, None]
            )
            relative = physical - centroid[batch]
            translation = params[batch, :3] * self.global_motion_scale_mm
            rotation = params[batch, 3:] * (
                self.global_motion_scale_mm / self.global_motion_lever_mm
            )
            displacement = translation + torch.cross(rotation, relative, dim=-1)
            return (displacement / std).to(coords.dtype)

    def _at_time(self, sample, data_stats) -> TimeConditionalOutput:
        coarse, latent, state = self.coarse(sample)
        rigid = (
            self._global_motion(latent, sample, data_stats)
            if self.global_motion
            else None
        )
        if not self.two_stage:
            coords = sample.node_features["coords"]
            prediction = coords + (
                self.mesh_post_residual_gate * self.post_node_decoder(latent)
            ).to(coords.dtype)
            baseline = sample.node_features.get("baseline")
            if baseline is not None:
                prediction = prediction + baseline.to(prediction.dtype)
            if rigid is not None:
                prediction = prediction + rigid
            return TimeConditionalOutput(prediction, prediction)
        if rigid is not None:
            coarse = coarse + rigid
        coarse_displacement = coarse - sample.node_features["coords"]
        shape = coarse
        passes = self.contact_iterations if self.provisional_contact else 1
        for _ in range(passes):
            contact_graph = (
                self._contact_graph(shape, sample, data_stats)
                if self.provisional_contact
                else None
            )
            correction = self.refine(
                latent,
                state,
                contact_graph,
                coarse_displacement=coarse_displacement,
            ).to(coarse.dtype)
            shape = coarse + correction
        return TimeConditionalOutput(self._clamp(shape, sample, data_stats), coarse)

    # ----- recipe entry point -----
    def forward(self, sample, data_stats, return_auxiliary: bool = False):
        if self.training:
            time = sample.node_features["time"]
            if (
                not 0.0
                <= float(time)
                <= (self.rollout_steps - 1) / self.rollout_steps + 1e-6
            ):
                raise ValueError(
                    "time is outside the datapipe's normalized future-frame range"
                )
            output = self._at_time(sample, data_stats)
            return output if return_auxiliary else output.prediction
        outputs = []
        original = sample.node_features.get("time")
        baseline_all = sample.node_features.get("baseline_all")
        original_baseline = sample.node_features.get("baseline")
        # Cases may be shorter than the configured bound (Yaris: 19-31 frames);
        # query exactly the target's horizon while keeping the time label's
        # normalization fixed at rollout_steps so it means the same physical time.
        target = getattr(sample, "node_target", None)
        steps = (
            int(target.shape[1])
            if target is not None and target.ndim == 3
            else self.rollout_steps
        )
        if steps > self.rollout_steps:
            raise ValueError(
                f"Target has {steps} steps but the model was built for {self.rollout_steps}"
            )
        try:
            for k in range(steps):
                sample.node_features["time"] = torch.tensor(
                    k / self.rollout_steps, device=sample.node_features["coords"].device
                )
                if baseline_all is not None:
                    sample.node_features["baseline"] = baseline_all[:, k]
                outputs.append(self._at_time(sample, data_stats).prediction)
        finally:
            if original is None:
                sample.node_features.pop("time", None)
            else:
                sample.node_features["time"] = original
            if baseline_all is not None:
                if original_baseline is None:
                    sample.node_features.pop("baseline", None)
                else:
                    sample.node_features["baseline"] = original_baseline
        return torch.stack(outputs, dim=1)  # [N, T-1, 3]
