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
"""Time-conditional DeFormer: initial mesh and a time code to the deformed shape.

For a requested future frame the model maps the initial mesh, its static
features and a time code to the deformed positions:

    inputs (X0, features, tau) -> pre-processor -> FLARE backbone
        -> post stage A -> coarse head  (provisional shape)
        -> post stage B -> final head   (correction added to the provisional shape)

* The **time code** is a learned Fourier embedding of the normalized query time
  that enters every node's input and, with ``time_as_global``, the global token
  of the backbone.
* The **local stages** are the base model's local operator (static kNN
  neighbourhood attention in the DeFormer recipe), FiLM-conditioned on the
  mesh context: a pre-processor before the backbone and two post stages after
  it. Stage A feeds a zero-initialised coarse head (the provisional shape,
  supervised with ``training.auxiliary_loss_weight``); stage B is an independent,
  re-initialised copy of stage A whose gated correction gives the final shape.
* There is no recurrent state and no BPTT: training samples independent
  (case, time) pairs, evaluation queries every future frame from the initial
  mesh and returns the trajectory ``[N, T, 3]``.

The evaluation-time rigid-wall clamp (:mod:`deformer.shape_corrections`) is available
and off by default. The checkpoint format is the base model's
(``physicsnemo.Module.from_checkpoint``).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from rollout import TimeEmbedding, global_tokens
from torch import nn

from physicsnemo.experimental.models.deformer import DeFormer, MeshContextFiLM

from .shape_corrections import RigidWallClamp, correct_shape


class TimeConditionalOutput(NamedTuple):
    """Training output of a time-conditional model: the final position and the
    auxiliary (provisional) position of stage A, both normalized ``[N, 3]``; the
    trainer supervises the auxiliary one with ``training.auxiliary_loss_weight``."""

    prediction: torch.Tensor
    auxiliary_prediction: torch.Tensor


class DeFormerTimeConditional(DeFormer):
    """Time-conditional DeFormer (the "network" of the DeFormer paper).

    Parameters
    ----------
    num_time_steps : int
        Frames per trajectory in the datapipe; ``num_time_steps - 1`` future
        frames are queried at evaluation, labelled ``k / (num_time_steps - 1)``
        for ``k = 0, ...`` (the datapipe's ``one_time_step`` convention: label
        0 is the first future frame).
    time_embedding : {"scalar", "fourier"}
        Time code entering the node inputs. ``functional_dim`` must count its
        width (1 for scalar, ``time_dim`` for fourier).
    time_fourier_bands, time_dim : int
        Fourier harmonics and output width of the learned time code.
    time_as_global : bool
        Also feed the time code as the global token; ``global_dim`` must then
        equal ``time_dim`` plus the number of dataset global features.
    dt : float, optional
        Frame interval of the datapipe in seconds. Bookkeeping only: the Yaris
        recipes derive ``datapipe.dt`` from ``model.dt`` and the prediction
        never integrates in time.
    clamp_to_wall, wall_axis, wall_sign
        Evaluation-time rigid-wall clamp of the prediction (see
        :class:`deformer.shape_corrections.RigidWallClamp`); needs ``graph.rigid_wall``
        from ``datapipe.rigid_wall_position``.
    **kwargs
        :class:`~physicsnemo.experimental.models.deformer.DeFormer`
        settings (backbone, local operator, context fusion, activation
        checkpointing). ``out_dim`` must be 3 and ``use_te`` false. The base
        model is built without its output head (``output_head=False``):
        ``num_post_processor_layers``, ``num_post_processor_checkpoint_segments``
        and ``mesh_post_residual_gate_init`` size the two stages here.
    """

    input_kind = "graph"

    def __init__(
        self,
        *,
        num_time_steps: int,
        time_embedding: str = "fourier",
        time_fourier_bands: int = 8,
        time_dim: int = 32,
        time_as_global: bool = True,
        dt: float | None = None,
        clamp_to_wall: bool = False,
        wall_axis: int = 0,
        wall_sign: float = 1.0,
        **kwargs,
    ) -> None:
        if num_time_steps < 2:
            raise ValueError("num_time_steps must be at least 2")
        if dt is not None and not dt > 0:
            raise ValueError("dt must be positive when given")
        if kwargs.get("out_dim", 3) != 3:
            raise ValueError("This recipe predicts positions only (out_dim=3)")
        if kwargs.get("use_te", False):
            raise ValueError("The time-conditional DeFormer requires use_te=False")
        if "output_head" in kwargs:
            raise ValueError(
                "DeFormerTimeConditional builds its own decoder stages; "
                "output_head is not a setting of this model"
            )
        # Validated settings of the evaluation-time wall clamp.
        self.wall_clamp = RigidWallClamp(
            enabled=clamp_to_wall, axis=wall_axis, sign=wall_sign
        )
        # The base model supplies the backbone state and the pre-processor only
        # (no output head); the two decoder stages below are this model's.
        super().__init__(output_head=False, **kwargs)
        self.rollout_steps = int(num_time_steps) - 1
        self.dt = None if dt is None else float(dt)
        self.time_embed = TimeEmbedding(time_embedding, time_fourier_bands, time_dim)
        self.time_as_global = bool(time_as_global)
        # Full-width backbone output (hidden plus optional local features) to the
        # mesh width, never through the three-channel output.
        effective_dim = self.backbone_width
        self.post_latent_input = (
            nn.Identity()
            if effective_dim == self.mesh_hidden_dim
            else nn.Linear(effective_dim, self.mesh_hidden_dim)
        )
        # Stage A side head: the provisional position. Zero-initialised so the
        # model starts at the undeformed mesh.
        self.coarse_decoder = nn.Sequential(
            nn.LayerNorm(self.mesh_hidden_dim),
            nn.Linear(self.mesh_hidden_dim, self.output_dim),
        )
        self._zero_head(self.coarse_decoder[-1])
        # Stage A: a local processor of the configured post-stage depth on the
        # backbone latent, FiLM-conditioned on the mesh context when fusion is
        # on. Stage B: an independent processor and FiLM of the same shape
        # whose gated correction gives the final shape.
        fusion = self.mesh_context_fusion != "none"
        self.coarse_processor = self.make_local_processor(
            self.num_post_processor_layers, self.num_post_processor_checkpoint_segments
        )
        self.coarse_context_film = (
            MeshContextFiLM(self.mesh_hidden_dim) if fusion else None
        )
        self.refinement_processor = self.make_local_processor(
            self.num_post_processor_layers, self.num_post_processor_checkpoint_segments
        )
        self.refinement_context_film = (
            MeshContextFiLM(self.mesh_hidden_dim) if fusion else None
        )
        self.final_decoder = self.make_node_decoder()
        self.final_gate = nn.Parameter(torch.tensor(self.mesh_post_residual_gate_init))
        last_linear = [
            m for m in self.final_decoder.modules() if isinstance(m, nn.Linear)
        ][-1]
        self._zero_head(last_linear)
        self.__name__ = "DeFormerTimeConditional"

    @staticmethod
    def _zero_head(linear: nn.Linear) -> None:
        nn.init.zeros_(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)

    # ----- building blocks -----
    def _time_inputs(self, sample, num_nodes: int):
        """Per-node time code ``[N, time_dim]`` and the global token(s)."""
        coords = sample.node_features["coords"]
        code = self.time_embed(sample.node_features["time"]).to(coords.dtype)
        global_embedding = global_tokens(sample)
        if self.time_as_global:
            token = code.view(1, 1, -1)
            global_embedding = (
                token
                if global_embedding is None
                else torch.cat((global_embedding, token), dim=-1)
            )
        return code.expand(num_nodes, -1), global_embedding

    def coarse(self, sample):
        """Provisional position ``[N, 3]``, the stage-A latent and the backbone state."""
        coords = sample.node_features["coords"]
        features = sample.node_features.get(
            "features", coords.new_zeros((coords.shape[0], 0))
        )
        time_code, global_embedding = self._time_inputs(sample, coords.shape[0])
        nodes = torch.cat((coords, features, time_code), dim=-1)
        state = DeFormer.forward(
            self,
            node_features=nodes,
            graph=sample.graph,
            geometry=coords,
            global_embedding=global_embedding,
            local_positions=coords,
            return_backbone=True,
        )
        latent = self.post_latent_input(state.point_features)
        if self.coarse_context_film is not None:
            latent = self.coarse_context_film(latent, state.mesh_context)
        latent = self._run_local_processor(
            self.coarse_processor, latent, state.neighborhood
        )
        coarse = coords + self.coarse_decoder(latent).to(coords.dtype)
        return coarse, latent, state

    def refine(self, latent, state):
        """Gated correction ``[N, 3]`` of stage B from the stage-A latent."""
        if self.refinement_context_film is not None:
            latent = self.refinement_context_film(latent, state.mesh_context)
        latent = self._run_local_processor(
            self.refinement_processor, latent, state.neighborhood
        )
        return self.final_gate * self.final_decoder(latent)

    def predict_frame(self, sample, data_stats) -> TimeConditionalOutput:
        """Final and provisional position at the sample's time label.

        In evaluation mode the final position passes through the rigid-wall
        clamp when it is enabled.
        """
        coarse, latent, state = self.coarse(sample)
        shape = coarse + self.refine(latent, state).to(coarse.dtype)
        if not self.training:
            shape = correct_shape(shape, sample, data_stats, clamp=self.wall_clamp)
        return TimeConditionalOutput(shape, coarse)

    # ----- recipe entry point -----
    def forward(self, sample, data_stats, return_auxiliary: bool = False):
        """One frame in training (``[N, 3]``, or the :class:`TimeConditionalOutput`
        with ``return_auxiliary``), the trajectory ``[N, T, 3]`` in evaluation.

        Evaluation queries exactly the target's horizon (cases may be shorter
        than the configured bound) while the time label keeps its normalization
        by ``rollout_steps``, so a label means the same physical time everywhere.
        """
        if self.training:
            time = float(sample.node_features["time"])
            if not 0.0 <= time <= (self.rollout_steps - 1) / self.rollout_steps + 1e-6:
                raise ValueError(
                    "time is outside the datapipe's normalized future-frame range"
                )
            output = self.predict_frame(sample, data_stats)
            return output if return_auxiliary else output.prediction
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
        features = sample.node_features
        original = features.get("time")
        outputs = []
        try:
            for k in range(steps):
                features["time"] = torch.tensor(
                    k / self.rollout_steps, device=features["coords"].device
                )
                outputs.append(self.predict_frame(sample, data_stats).prediction)
        finally:
            if original is None:
                features.pop("time", None)
            else:
                features["time"] = original
        return torch.stack(outputs, dim=1)  # [N, T, 3]
